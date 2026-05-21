// Workgroup-wide vec4 addition via subgroup ops + shmem merge.
//
// Kept separate from subgroup_reduce_scalar.glsl so shaders that only need
// the scalar variants don't have to allocate sg_partial4[].
//
// Contract: caller declares
//   layout(constant_id = 0) const uint SUBGROUP_SIZE = 32;
//   const uint NUM_SUBGROUPS = (WG + SUBGROUP_SIZE - 1u) / SUBGROUP_SIZE;
//   shared vec4 sg_partial4[NUM_SUBGROUPS];
//
// On WGs that fit in a single subgroup (the common case for the attention
// kernels where WG <= 64 and SUBGROUP_SIZE >= 32) NUM_SUBGROUPS == 1 and the
// function early-returns after a single subgroupAdd — no shmem traffic, no
// barrier.

vec4 wg_sum_v4(vec4 local_v) {
    vec4 sg_sum = subgroupAdd(local_v);
    if (NUM_SUBGROUPS == 1u)
        return sg_sum;
    if (gl_SubgroupInvocationID == 0u)
        sg_partial4[gl_SubgroupID] = sg_sum;
    barrier();
    if (gl_SubgroupID == 0u) {
        vec4 combined = vec4(0.0);
        if (gl_SubgroupInvocationID < NUM_SUBGROUPS)
            combined = sg_partial4[gl_SubgroupInvocationID];
        vec4 final_sum = subgroupAdd(combined);
        if (gl_SubgroupInvocationID == 0u)
            sg_partial4[0] = final_sum;
    }
    barrier();
    return sg_partial4[0];
}
