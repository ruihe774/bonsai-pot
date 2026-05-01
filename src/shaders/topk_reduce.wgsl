enable f16;

// Single-workgroup top-K reduction over `n` halves.
// Phase 1: each thread maintains a per-thread top-K_MAX as a MIN-HEAP in
//          shared memory (insertion = O(log K) sift-down).
// Phase 2a: each thread heap-sorts its K_MAX entries into ASCENDING order
//          (extract-min K times, in-place).
// Phase 2b: pairwise tree merge across threads, in-place in shared memory,
//          via two-pointer-from-the-top.
// dispatch: (1, 1, 1)
//
// Output (at u32 offset `out_offset` into `result`):
//   result[0..k]      = top-K logit values (f32 bitcast to u32), DESCENDING
//   result[k..2*k]    = top-K vocab indices (u32), aligned with the values

struct Params {
  n: u32,
  in_offset: u32,
  out_offset: u32,
  k: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f16>;
@group(0) @binding(2) var<storage, read_write> result: array<u32>;

const WG: u32 = 64u;
const K_MAX: u32 = 64u;

var<workgroup> sh_val: array<f32, 4096u>;   // WG * K_MAX
var<workgroup> sh_idx: array<u32, 4096u>;

var<private> staged_val: array<f32, 64u>;
var<private> staged_idx: array<u32, 64u>;

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
  for (var i: u32 = tid; i < p.n; i = i + WG) {
    let v = f32(logits[p.in_offset + i]);
    if (v > sh_val[base]) {
      sh_val[base] = v;
      sh_idx[base] = i;
      sift_down(base);
    }
  }

  // ---- Phase 2a: heap-sort to ascending. Repeatedly swap root (min) with
  //                last element of the active heap, then sift down on the
  //                shrunken heap. Result: sh[base..base+K_MAX] sorted DESC.
  // Then reverse to get ASC. (We could also extract-min into the staged array
  // and copy back, but in-place is simpler.)
  // -- Sort descending: at each step, swap heap[0] (min) with heap[active-1],
  //                     then re-sift the [0..active-1] sub-heap.
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
  // Now sh[base..base+K_MAX] is sorted DESCENDING. Reverse to ASCENDING for the
  // two-pointer merge convention used in phase 2b.
  for (var i: u32 = 0u; i < K_MAX / 2u; i = i + 1u) {
    let j = K_MAX - 1u - i;
    let tv = sh_val[base + i]; sh_val[base + i] = sh_val[base + j]; sh_val[base + j] = tv;
    let ti = sh_idx[base + i]; sh_idx[base + i] = sh_idx[base + j]; sh_idx[base + j] = ti;
  }
  workgroupBarrier();

  // ---- Phase 2b: tree reduction. Each level halves the active thread count.
  var s: u32 = WG / 2u;
  loop {
    if (s == 0u) { break; }
    if (tid < s) {
      let a = base;
      let b = (tid + s) * K_MAX;
      var ia: i32 = i32(K_MAX) - 1;
      var ib: i32 = i32(K_MAX) - 1;
      for (var i: u32 = 0u; i < K_MAX; i = i + 1u) {
        let va = sh_val[a + u32(ia)];
        let vb = sh_val[b + u32(ib)];
        let dst = K_MAX - 1u - i;
        if (va > vb) {
          staged_val[dst] = va;
          staged_idx[dst] = sh_idx[a + u32(ia)];
          ia = ia - 1;
        } else {
          staged_val[dst] = vb;
          staged_idx[dst] = sh_idx[b + u32(ib)];
          ib = ib - 1;
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
      let src = K_MAX - 1u - i;
      result[p.out_offset + i]      = bitcast<u32>(sh_val[src]);
      result[p.out_offset + kk + i] = sh_idx[src];
    }
  }
}
