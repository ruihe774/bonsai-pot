enable f16;

// Combines per-chunk (m, l, o) partials produced by attention_split.wgsl into
// the final attention output via a flash-attention-style log-sum-exp merge.
//
// dispatch: (n_head, 1, 1). One workgroup per Q head. Three phases:
//   A) Parallel chunk-max reduction across threads → m_global (subgroupMax).
//   B) Parallel weight precompute into shmem + parallel sum of l_c*weight
//      across threads → l_global (subgroupAdd).
//   C) Per-thread o accumulation using the precomputed weights, then write.
//
// Cheap relative to the chunk pass: ~n_chunks_active * (head_dim + 2) f32
// reads per workgroup. The previous version had every thread re-scan all
// chunks twice (for m_global and weight); precomputing weights once and
// using subgroup reductions for the scalar reductions cuts redundant work.

struct Params {
  head_dim: u32,
  n_head: u32,
  out_offset: u32,
  n_chunks_active: u32,
  _p0: u32, _p1: u32, _p2: u32, _p3: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> act: array<f16>;
@group(0) @binding(2) var<storage, read> partials: array<f32>;

const SG_SIZE: u32 = {{SG_SIZE}}u;
const WG: u32 = 64u;
const N_SG: u32 = WG / SG_SIZE;
const ELEMS_PER_THREAD: u32 = 2u;
const PARTIAL_STRIDE: u32 = 130u;  // head_dim + 2
// Cap on chunk count we keep weights for in shmem. n_chunks = ceil(pos/32);
// 256 covers max_seq up to 8192, which exceeds the runtime's default 1024.
const MAX_CHUNKS: u32 = 256u;

var<workgroup> sg_partial: array<f32, N_SG>;
var<workgroup> weights_sh: array<f32, MAX_CHUNKS>;

fn wg_max(local: f32, tid: u32, sg_inv_id: u32) -> f32 {
  let sg_m = subgroupMax(local);
  if (N_SG == 1u) { return sg_m; }
  let sg_id = tid / SG_SIZE;
  if (sg_inv_id == 0u) { sg_partial[sg_id] = sg_m; }
  workgroupBarrier();
  if (sg_id == 0u) {
    var combined: f32 = -1e30;
    if (sg_inv_id < N_SG) { combined = sg_partial[sg_inv_id]; }
    let final_m = subgroupMax(combined);
    if (sg_inv_id == 0u) { sg_partial[0] = final_m; }
  }
  workgroupBarrier();
  return sg_partial[0];
}

fn wg_sum(local: f32, tid: u32, sg_inv_id: u32) -> f32 {
  let sg_s = subgroupAdd(local);
  if (N_SG == 1u) { return sg_s; }
  let sg_id = tid / SG_SIZE;
  if (sg_inv_id == 0u) { sg_partial[sg_id] = sg_s; }
  workgroupBarrier();
  if (sg_id == 0u) {
    var combined: f32 = 0.0;
    if (sg_inv_id < N_SG) { combined = sg_partial[sg_inv_id]; }
    let final_s = subgroupAdd(combined);
    if (sg_inv_id == 0u) { sg_partial[0] = final_s; }
  }
  workgroupBarrier();
  return sg_partial[0];
}

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_inv_id: u32,
) {
  let h = wg.x;
  if (h >= p.n_head) { return; }
  let tid = lid.x;
  let hd = p.head_dim;
  let n = p.n_chunks_active;
  let head_base = h * n * PARTIAL_STRIDE;

  // Phase A: parallel max over chunk-local m's.
  var m_local: f32 = -1e30;
  for (var c: u32 = tid; c < n; c += WG) {
    m_local = max(m_local, partials[head_base + c * PARTIAL_STRIDE + hd]);
  }
  let m_global = wg_max(m_local, tid, sg_inv_id);

  // Phase B: parallel weight precompute + parallel l_global reduction.
  var l_local: f32 = 0.0;
  for (var c: u32 = tid; c < n; c += WG) {
    let pb = head_base + c * PARTIAL_STRIDE;
    let m_c = partials[pb + hd];
    let l_c = partials[pb + hd + 1u];
    let w = exp(m_c - m_global);
    weights_sh[c] = w;
    l_local = l_local + l_c * w;
  }
  let l_global = wg_sum(l_local, tid, sg_inv_id);

  // Phase C: per-thread o accumulation using precomputed weights. The per-d
  // loops below assume head_dim == ELEMS_PER_THREAD * WG (true for Bonsai-4B);
  // models with a different head_dim would need ELEMS_PER_THREAD retemplated.
  var o: array<f32, ELEMS_PER_THREAD>;
  for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) { o[i] = 0.0; }
  for (var c: u32 = 0u; c < n; c++) {
    let pb = head_base + c * PARTIAL_STRIDE;
    let weight = weights_sh[c];
    for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
      let d = tid + i * WG;
      o[i] = o[i] + partials[pb + d] * weight;
    }
  }

  let inv_l = 1.0 / l_global;
  let out_base = p.out_offset + h * hd;
  for (var i: u32 = 0u; i < ELEMS_PER_THREAD; i++) {
    let d = tid + i * WG;
    act[out_base + d] = f16(o[i] * inv_l);
  }
}
