enable f16;

// Single-workgroup top-K reduction over `n` halves.
// Phase 1:  each thread maintains a per-thread top-K_MAX as a MIN-HEAP in
//           shared memory (insertion = O(log K) sift-down). Logits are read
//           as u32 (= 2 packed f16) and processed in pairs to halve issued
//           load instructions.
// Phase 2a: each thread heap-sorts its K_MAX entries into DESCENDING order
//           (extract-min K times, in-place; min ends up at the back, so the
//           front-to-back order is descending).
// Phase 2b: pairwise tree merge across threads, in-place in shared memory,
//           via two-pointer-from-the-front, writing the merged run descending.
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

const WG: u32 = 64u;
const K_MAX: u32 = 32u;

var<workgroup> sh_val: array<f32, 2048u>;   // WG * K_MAX
var<workgroup> sh_idx: array<u32, 2048u>;

var<private> staged_val: array<f32, 32u>;
var<private> staged_idx: array<u32, 32u>;

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

  // ---- Phase 2b: tree reduction. Each level halves the active thread count.
  // Two-pointer merge from the front: both halves are descending, so the
  // larger of the two heads is the next-largest overall.
  var s: u32 = WG / 2u;
  loop {
    if (s == 0u) { break; }
    if (tid < s) {
      let a = base;
      let b = (tid + s) * K_MAX;
      var ia: u32 = 0u;
      var ib: u32 = 0u;
      for (var i: u32 = 0u; i < K_MAX; i = i + 1u) {
        let va = sh_val[a + ia];
        let vb = sh_val[b + ib];
        if (va > vb) {
          staged_val[i] = va;
          staged_idx[i] = sh_idx[a + ia];
          ia = ia + 1u;
        } else {
          staged_val[i] = vb;
          staged_idx[i] = sh_idx[b + ib];
          ib = ib + 1u;
        }
      }
      for (var i: u32 = 0u; i < K_MAX; i = i + 1u) {
        sh_val[a + i] = staged_val[i];
        sh_idx[a + i] = staged_idx[i];
      }
    }
    workgroupBarrier();
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
