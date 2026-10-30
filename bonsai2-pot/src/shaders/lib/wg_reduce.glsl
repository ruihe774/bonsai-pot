// Workgroup-wide reductions built from subgroup ops plus a shared-memory merge.
// Uses the runtime gl_NumSubgroups / gl_SubgroupID, so no subgroup-size
// specialization is needed. Supports up to 64 subgroups per workgroup.
//
// Every invocation of the workgroup must call these from uniform control flow.

shared float wg_red_sh[64];

float wg_sum(float v) {
    float s = subgroupAdd(v);
    if (gl_NumSubgroups == 1u) {
        return s;
    }
    if (subgroupElect()) {
        wg_red_sh[gl_SubgroupID] = s;
    }
    barrier();
    float t = 0.0;
    for (uint i = 0u; i < gl_NumSubgroups; i += 1u) {
        t += wg_red_sh[i];
    }
    barrier();
    return t;
}

float wg_max(float v) {
    float s = subgroupMax(v);
    if (gl_NumSubgroups == 1u) {
        return s;
    }
    if (subgroupElect()) {
        wg_red_sh[gl_SubgroupID] = s;
    }
    barrier();
    float t = wg_red_sh[0];
    for (uint i = 1u; i < gl_NumSubgroups; i += 1u) {
        t = max(t, wg_red_sh[i]);
    }
    barrier();
    return t;
}
