enable f16;

// Single-workgroup top-K reduction over `n` halves.
// Phase 1:  each thread maintains a per-thread top-K_MAX as a MIN-HEAP in
//           shared memory (insertion = O(log K) sift-down). Logits are read
//           as u32 (= 2 packed f16) and processed in pairs to halve issued
//           load instructions.
// Phase 2a: each thread heap-sorts its K_MAX entries into DESCENDING order
//           (extract-min K times, in-place; min ends up at the back, so the
//           front-to-back order is descending).
// Phase 2b: pairwise tree merge across threads, parallelized across all WG
//           threads via bitonic merge. Each merge takes two K_MAX-sorted-
//           descending halves and produces one K_MAX-sorted-descending run
//           (top-K of the union) using one stride-K_MAX max-pick stage plus
//           log2(K_MAX) bitonic-sort stages.
// Phase 3:  thread 0 writes sh[base..+kk] directly to `result` (descending).
// dispatch: (1, 1, 1)
//
// Output (at u32 offset `out_offset` into `result`):
//   result[0..k]      = top-K logit values (f32 bitcast to u32), DESCENDING
//   result[k..2*k]    = top-K vocab indices (u32), aligned with the values

struct Params {
  n: u32,            // number of f16 logits
  in_offset: u32,    // f16 element offset; must be even (ActLayout guarantees this)
  out_offset: u32,
  k: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
// Bound to the same act buffer as the shaders that produce logits (which use
// `array<f16>`), but viewed here as u32 so each load returns 2 packed f16.
@group(0) @binding(1) var<storage, read> logits: array<u32>;
@group(0) @binding(2) var<storage, read_write> result: array<u32>;

const WG: u32 = 256u;
const K_MAX: u32 = 32u;

var<workgroup> sh_val: array<f32, 8192u>;   // WG * K_MAX
var<workgroup> sh_idx: array<u32, 8192u>;

// Sift-down for a min-heap rooted at `base`, restoring heap property after
// (potentially) replacing the root.
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
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let base = tid * K_MAX;

  // ---- Phase 1: build per-thread top-K min-heap.
  // Initialize all to -inf so insertion always succeeds for the first K_MAX
  // elements (heap property trivially holds for an all-equal heap).
  for (var i: u32 = 0u; i < K_MAX; i = i + 1u) {
    sh_val[base + i] = -1e30;
    sh_idx[base + i] = 0u;
  }
  // Iterate in u32 words (= 2 f16 packed). `in_offset` is f16 element offset
  // and is guaranteed even by ActLayout. The tail of an odd `n` is handled
  // by clamping the second f16's index against n (its value is don't-care).
  let in_word_off = p.in_offset >> 1u;
  let n_words = (p.n + 1u) >> 1u;
  for (var w: u32 = tid; w < n_words; w = w + WG) {
    let pair = unpack2x16float(logits[in_word_off + w]);
    let i0 = w << 1u;
    let i1 = i0 + 1u;
    let v0 = pair.x;
    if (v0 > sh_val[base]) {
      sh_val[base] = v0;
      sh_idx[base] = i0;
      sift_down(base);
    }
    if (i1 < p.n) {
      let v1 = pair.y;
      if (v1 > sh_val[base]) {
        sh_val[base] = v1;
        sh_idx[base] = i1;
        sift_down(base);
      }
    }
  }

  // ---- Phase 2a: heap-sort to descending. At each step, swap heap[0] (min)
  //                with heap[active-1], then re-sift the [0..active-1) sub-heap.
  //                The minimum ends up at the back, the second-minimum just
  //                before it, etc. — so sh[base..base+K_MAX] is DESCENDING
  //                front-to-back when this loop finishes.
  for (var active_minus1: u32 = K_MAX - 1u; active_minus1 > 0u; active_minus1 = active_minus1 - 1u) {
    let last = base + active_minus1;
    let tv = sh_val[base]; sh_val[base] = sh_val[last]; sh_val[last] = tv;
    let ti = sh_idx[base]; sh_idx[base] = sh_idx[last]; sh_idx[last] = ti;
    // Re-sift the shrunken heap [0..active_minus1).
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

  // ---- Phase 2b: parallel pairwise bitonic merge. At each tree level, pairs
  // (a, b) of K_MAX-sorted-descending arrays are merged into one (top-K of
  // their union). The merge is implemented as a bitonic merge of the virtual
  // sequence c = a ++ reverse(b) of length 2*K_MAX:
  //   Stage 0: compare-swap c[i] vs c[i + K_MAX]; we keep the MAX in the lower
  //            half (a's slot) and discard the upper half (b's slot becomes
  //            dead). Reading c[i + K_MAX] = b[K_MAX - 1 - i] inlines the
  //            "reverse(b)" without touching memory.
  //   Stages 1..LOG2_K_MAX: in-place bitonic-sort of a's slot, descending,
  //            using strides K_MAX/2, K_MAX/4, ..., 1.
  // Every thread participates in every stage; work units are distributed by
  // tid stride so a single wave covers the work without idle gaps until the
  // final levels. On wave64 hardware (e.g. RDNA), workgroupBarrier across a
  // 64-thread workgroup is essentially free, so the extra per-stage barriers
  // don't dominate.
  var s: u32 = WG / 2u;
  loop {
    if (s == 0u) { break; }

    // Stage 0: a[i] = max(a[i], reverse(b)[i]) for all (pair, i). Discards b.
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

    // Stages 1..LOG2_K_MAX: bitonic sort each pair's a-slot descending.
    let n_perstage = s * (K_MAX / 2u);
    var stride: u32 = K_MAX / 2u;
    loop {
      if (stride == 0u) { break; }
      for (var k: u32 = tid; k < n_perstage; k = k + WG) {
        let pair_id = k / (K_MAX / 2u);
        let local = k % (K_MAX / 2u);
        // For stride-block of 2*stride elements, enumerate the "lower-index"
        // of each compare-swap pair: lo = ((local & ~(stride-1)) << 1) | (local & (stride-1)).
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

  // ---- Write top-K to result, DESCENDING by value.
  if (tid == 0u) {
    let kk = min(p.k, K_MAX);
    for (var i: u32 = 0u; i < kk; i = i + 1u) {
      let src = i;
      result[p.out_offset + i]      = bitcast<u32>(sh_val[src]);
      result[p.out_offset + kk + i] = sh_idx[src];
    }
  }
}
