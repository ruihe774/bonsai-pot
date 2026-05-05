enable f16;

// Pass-2 of the multi-WG top-K reduction. Reads `num_partials * K_MAX` already-
// sorted-descending (f32 value, u32 index) pairs that pass-1 (`topk_partial`)
// wrote into a scratch region of `result`, and produces the global top-K.
//
// Algorithm: each thread directly loads one partial slot into its own
// K_MAX-entry shmem region (or `-inf` filler if `tid >= num_partials`), then
// runs the same parallel pairwise bitonic merge tree as pass-1's phase 2b.
// Skipping the per-thread heap-build/heap-sort phases is sound because each
// partial is *already* descending — pass-1 emitted it that way.
//
// dispatch: (1, 1, 1)

struct Params {
  partials_off: u32,    // u32 word offset into `result` where pass-1 wrote
  num_partials: u32,    // # partial slots; <= WG
  out_offset: u32,      // u32 word offset for the final top-K output in `result`
  k: u32,
};

var<immediate> p: Params;
@group(0) @binding(0) var<storage, read_write> result: array<u32>;

// WG must be a power of two >= num_partials so the bitonic merge tree
// reduces all partials. We always launch a full wave64 (WG=64); when
// num_partials < WG, the upper half loads `-inf` and the first merge
// stage prunes them out for free.
const WG: u32 = 64u;
const K_MAX: u32 = 32u;

// Logits are originally f16 — pass-1 widened them to f32 only at the inter-pass
// boundary. Demote back to f16 in shmem (lossless; halves LDS: 8 KiB → 4 KiB).
var<workgroup> sh_val: array<f16, 2048u>;   // WG * K_MAX
var<workgroup> sh_idx: array<u32, 2048u>;

const NEG_INF_H: f16 = -65504.0h;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let base = tid * K_MAX;

  // ---- Load this thread's partial slot directly. Threads with
  //       `tid >= num_partials` fill -inf so the merge tree treats them as
  //       empty. Each partial is already sorted descending, so we skip the
  //       heap-build and heap-sort phases entirely.
  if (tid < p.num_partials) {
    let slot_base = p.partials_off + tid * (2u * K_MAX);
    for (var i: u32 = 0u; i < K_MAX; i = i + 1u) {
      // Inter-pass values are f32; demote to f16 (the original logit precision).
      sh_val[base + i] = f16(bitcast<f32>(result[slot_base + i]));
      sh_idx[base + i] = result[slot_base + K_MAX + i];
    }
  } else {
    for (var i: u32 = 0u; i < K_MAX; i = i + 1u) {
      sh_val[base + i] = NEG_INF_H;
      sh_idx[base + i] = 0u;
    }
  }
  workgroupBarrier();

  // ---- Phase 2b: parallel pairwise bitonic merge across all WG threads. ----
  var s: u32 = WG / 2u;
  loop {
    if (s == 0u) { break; }
    let n_stage0 = s * K_MAX;
    for (var k: u32 = tid; k < n_stage0; k = k + WG) {
      let pair_id = k / K_MAX;
      let i = k % K_MAX;
      let a_base = pair_id * K_MAX;
      let b_base = (pair_id + s) * K_MAX;
      let bi = K_MAX - 1u - i;
      let av = sh_val[a_base + i];
      let bv = sh_val[b_base + bi];
      if (bv > av) {
        sh_val[a_base + i] = bv;
        sh_idx[a_base + i] = sh_idx[b_base + bi];
      }
    }
    workgroupBarrier();

    let n_perstage = s * (K_MAX / 2u);
    var stride: u32 = K_MAX / 2u;
    loop {
      if (stride == 0u) { break; }
      for (var k: u32 = tid; k < n_perstage; k = k + WG) {
        let pair_id = k / (K_MAX / 2u);
        let local = k % (K_MAX / 2u);
        let lo = ((local & ~(stride - 1u)) << 1u) | (local & (stride - 1u));
        let hi = lo | stride;
        let a_base = pair_id * K_MAX;
        let vlo = sh_val[a_base + lo];
        let vhi = sh_val[a_base + hi];
        if (vhi > vlo) {
          let ilo = sh_idx[a_base + lo];
          sh_val[a_base + lo] = vhi;
          sh_val[a_base + hi] = vlo;
          sh_idx[a_base + lo] = sh_idx[a_base + hi];
          sh_idx[a_base + hi] = ilo;
        }
      }
      workgroupBarrier();
      stride = stride / 2u;
    }

    s = s / 2u;
  }

  // ---- Write top-K to result, descending by value.
  if (tid == 0u) {
    let kk = min(p.k, K_MAX);
    for (var i: u32 = 0u; i < kk; i = i + 1u) {
      // CPU sampler reads f32; widen at the boundary.
      result[p.out_offset + i]      = bitcast<u32>(f32(sh_val[i]));
      result[p.out_offset + kk + i] = sh_idx[i];
    }
  }
}
