// Q8_0 per-32-lane quantize helpers.
//
// Contract: caller declares
//   layout(constant_id = 0) const uint SUBGROUP_SIZE = 32;
//   const uint NUM_SUBGROUPS = (WG + SUBGROUP_SIZE - 1u) / SUBGROUP_SIZE;
//   shared float sg_partial_amax[NUM_SUBGROUPS];
//
// block_max_abs(v) → float
//   Returns the max(abs(v)) across all 32 lanes of the current Q8_0 block.
//   Masks 1/2/4 stay in-subgroup (SUBGROUP_SIZE >= 8 enforced at load time).
//   Masks 8/16 are guarded by SUBGROUP_SIZE. When SUBGROUP_SIZE < 32 the
//   cross-cluster stitch path fires, writing per-subgroup partials to
//   sg_partial_amax[].
//
// pack4(qv) → uint
//   Gathers the 4 byte-sized quant values from the 4-lane cluster containing
//   the current lane and packs them into a single u32. The cluster always
//   sits inside one subgroup (SUBGROUP_SIZE >= 8), so no shmem is needed.

float block_max_abs(float v) {
    float m = abs(v);
    m = max(m, subgroupShuffleXor(m, 1u));
    m = max(m, subgroupShuffleXor(m, 2u));
    m = max(m, subgroupShuffleXor(m, 4u));
    if (SUBGROUP_SIZE >= 16u) {
        m = max(m, subgroupShuffleXor(m, 8u));
    }
    if (SUBGROUP_SIZE >= 32u) {
        m = max(m, subgroupShuffleXor(m, 16u));
    }
    if (SUBGROUP_SIZE < 32u) {
        // Stitch in-subgroup partials across the clusters that span this
        // 32-element Q8_0 block. Each cluster of SUBGROUP_SIZE lanes writes
        // one slot; 32/SUBGROUP_SIZE slots per block.
        uint tid = gl_LocalInvocationID.x;
        uint sg_lane = tid % SUBGROUP_SIZE;
        uint cluster_id = tid / SUBGROUP_SIZE;
        if (sg_lane == 0u) {
            sg_partial_amax[cluster_id] = m;
        }
        barrier();
        uint group_id = tid >> 5u;
        uint per_group = 32u / SUBGROUP_SIZE;
        uint g_base = group_id * per_group;
        float mm = 0.0;
        [[unroll]]
        for (uint i = 0u; i < per_group; i += 1u) {
            mm = max(mm, sg_partial_amax[g_base + i]);
        }
        barrier();
        m = mm;
    }
    return m;
}

uint pack4(uint qv) {
    // Gather bytes from the 4-lane cluster and pack into one u32.
    qv <<= (gl_SubgroupInvocationID & 3u) << 3u;
    qv |= subgroupShuffleXor(qv, 1u);
    qv |= subgroupShuffleXor(qv, 2u);
    return qv;
}
