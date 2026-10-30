// Workgroup-wide scalar reductions via subgroup ops + shmem merge.
//
// Contract: caller declares
//   layout(constant_id = 0) const uint SUBGROUP_SIZE = 32;
//   const uint NUM_SUBGROUPS = (WG + SUBGROUP_SIZE - 1u) / SUBGROUP_SIZE;
//   shared float sg_partial[NUM_SUBGROUPS];
//
// The NUM_SUBGROUPS > SUBGROUP_SIZE fan-in loop handles the lavapipe/Mali
// case where a single subgroup cannot hold all partial results; spirv-opt
// constant-folds the loop body away for the common NUM_SUBGROUPS <= SUBGROUP_SIZE
// case (the loop runs 0 or 1 iterations and the conditional is dead).

float wg_sum(float local_v) {
    float sg_sum = subgroupAdd(local_v);
    if (NUM_SUBGROUPS == 1u)
        return sg_sum;
    if (gl_SubgroupInvocationID == 0u)
        sg_partial[gl_SubgroupID] = sg_sum;
    barrier();
    if (gl_SubgroupID == 0u) {
        float c = 0.0;
        for (uint i = gl_SubgroupInvocationID; i < NUM_SUBGROUPS; i += SUBGROUP_SIZE)
            c += sg_partial[i];
        float f = subgroupAdd(c);
        if (gl_SubgroupInvocationID == 0u)
            sg_partial[0] = f;
    }
    barrier();
    return sg_partial[0];
}

float wg_max(float local_v) {
    float sg_m = subgroupMax(local_v);
    if (NUM_SUBGROUPS == 1u)
        return sg_m;
    if (gl_SubgroupInvocationID == 0u)
        sg_partial[gl_SubgroupID] = sg_m;
    barrier();
    if (gl_SubgroupID == 0u) {
        float combined = -1e30;
        for (uint i = gl_SubgroupInvocationID; i < NUM_SUBGROUPS; i += SUBGROUP_SIZE)
            combined = max(combined, sg_partial[i]);
        float final_m = subgroupMax(combined);
        if (gl_SubgroupInvocationID == 0u)
            sg_partial[0] = final_m;
    }
    barrier();
    return sg_partial[0];
}
