enable f16;

// Multi-WG pass-1 of top-K reduction. Each workgroup processes a
// `[wg.x * n_per_wg, min((wg.x+1)*n_per_wg, n))` slice of the f16 logits and
// emits its own top-K_MAX (descending) into a scratch slot in `result`. The
// follow-up `topk_merge` dispatch consumes those `num_partials * K_MAX`
// candidates and produces the final top-K.
//
// Algorithm matches the original single-WG `topk_reduce`:
//   Phase 1:  per-thread top-K_MAX MIN-HEAP over the WG's slice.
//   Phase 2a: per-thread heap-sort to descending in place.
//   Phase 2b: parallel pairwise bitonic merge across all WG threads → final
//             top-K_MAX of this WG, descending, in `sh[0..K_MAX]`.
//   Phase 3:  write top-K_MAX values+indices to scratch slot
//             `result[partials_off + wg.x * 2*K_MAX ..]`.
//
// dispatch: (NUM_PARTIAL_WG, 1, 1)

struct Params {
  n: u32,                  // total number of f16 logits
  in_offset: u32,          // f16 element offset into `logits`; must be even
  partials_off: u32,       // u32 word offset into `result` for pass-1 scratch
  n_per_wg: u32,           // # f16 logits assigned to each WG (last WG may overflow; clamped to n)
};

var<immediate> p: Params;
@group(0) @binding(0) var<storage, read> logits: array<u32>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;

const WG: u32 = 128u;
const K_MAX: u32 = 32u;

// Logits are originally f16 (LM-head matvec writes `f16(acc)`), so storing the
// per-thread heap values in f16 is lossless and halves LDS use (16 KiB → 8 KiB).
var<workgroup> sh_val: array<f16, 4096u>;   // WG * K_MAX
var<workgroup> sh_idx: array<u32, 4096u>;

// Most-negative finite f16 — sentinel for "empty heap slot". `-1e30` (used as
// the f32 sentinel) is out of f16 range; -65504 still compares less than every
// real logit in practice (per-token logit magnitudes are O(10) post-softmax-norm).
const NEG_INF_H: f16 = -65504.0h;

fn sift_down(base: u32) {
  var i: u32 = 0u;
  loop {
    let l = 2u * i + 1u;
    let r = l + 1u;
    var smallest = i;
    if (l < K_MAX && sh_val[base + l] < sh_val[base + smallest]) { smallest = l; }
    if (r < K_MAX && sh_val[base + r] < sh_val[base + smallest]) { smallest = r; }
    if (smallest == i) { break; }
    let tv = sh_val[base + i]; sh_val[base + i] = sh_val[base + smallest]; sh_val[base + smallest] = tv;
    let ti = sh_idx[base + i]; sh_idx[base + i] = sh_idx[base + smallest]; sh_idx[base + smallest] = ti;
    i = smallest;
  }
}

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.x;
  let base = tid * K_MAX;

  // Slice for this WG, clamped to n. `in_offset` is f16-elements and even, so
  // we can iterate in u32 words (= 2 f16) starting from
  // `(in_offset + i_lo) >> 1`.
  let i_lo = wg.x * p.n_per_wg;
  let i_hi = min(i_lo + p.n_per_wg, p.n);
  let in_word_off = p.in_offset >> 1u;
  let w_lo = i_lo >> 1u;             // i_lo is a multiple of 2 because n_per_wg is chosen even
  let w_hi = (i_hi + 1u) >> 1u;       // round up so we don't drop a trailing odd element

  // ---- Phase 1: build per-thread top-K min-heap.
  for (var i: u32 = 0u; i < K_MAX; i = i + 1u) {
    sh_val[base + i] = NEG_INF_H;
    sh_idx[base + i] = 0u;
  }
  for (var w: u32 = w_lo + tid; w < w_hi; w = w + WG) {
    // unpack2x16float widens the f16 pair to vec2<f32>; cast back to f16 for
    // the shmem heap (lossless — f16→f32 is exact).
    let pair_f32 = unpack2x16float(logits[in_word_off + w]);
    let i0 = w << 1u;
    let i1 = i0 + 1u;
    let v0 = f16(pair_f32.x);
    if (i0 < i_hi && v0 > sh_val[base]) {
      sh_val[base] = v0;
      sh_idx[base] = i0;
      sift_down(base);
    }
    if (i1 < i_hi) {
      let v1 = f16(pair_f32.y);
      if (v1 > sh_val[base]) {
        sh_val[base] = v1;
        sh_idx[base] = i1;
        sift_down(base);
      }
    }
  }

  // ---- Phase 2a: heap-sort to descending. ----
  for (var active_minus1: u32 = K_MAX - 1u; active_minus1 > 0u; active_minus1 = active_minus1 - 1u) {
    let last = base + active_minus1;
    let tv = sh_val[base]; sh_val[base] = sh_val[last]; sh_val[last] = tv;
    let ti = sh_idx[base]; sh_idx[base] = sh_idx[last]; sh_idx[last] = ti;
    var i: u32 = 0u;
    loop {
      let l = 2u * i + 1u;
      let r = l + 1u;
      var smallest = i;
      if (l < active_minus1 && sh_val[base + l] < sh_val[base + smallest]) { smallest = l; }
      if (r < active_minus1 && sh_val[base + r] < sh_val[base + smallest]) { smallest = r; }
      if (smallest == i) { break; }
      let tv = sh_val[base + i]; sh_val[base + i] = sh_val[base + smallest]; sh_val[base + smallest] = tv;
      let ti = sh_idx[base + i]; sh_idx[base + i] = sh_idx[base + smallest]; sh_idx[base + smallest] = ti;
      i = smallest;
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

  // ---- Phase 3: write top-K_MAX of this WG to scratch ----
  // Slot layout: [K_MAX f32 values | K_MAX u32 indices], starting at
  // result[partials_off + wg.x * 2*K_MAX].
  let slot_base = p.partials_off + wg.x * (2u * K_MAX);
  if (tid < K_MAX) {
    // Inter-pass storage stays f32 (untouched layout); widen the f16 LDS value
    // for free at the boundary.
    result[slot_base + tid]         = bitcast<u32>(f32(sh_val[tid]));
    result[slot_base + K_MAX + tid] = sh_idx[tid];
  }
}
