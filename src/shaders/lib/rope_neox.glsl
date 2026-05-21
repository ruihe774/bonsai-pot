// NEOX RoPE half-pair swap.
//
// Contract: caller declares
//   const uint HEAD_DIM = 128u;
//   const uint HALF_DIM = 64u;
//   shared float16_t rope_in_sh[HEAD_DIM];   // normed+weighted head, filled before call
//   RopeBuf { float16_t rope_cs[]; } at an appropriate binding.
//
// rope_neox(tid, cs_base) → float16_t
//   tid     : local thread id within the head (0..HEAD_DIM-1)
//   cs_base : element offset into rope_cs[] for the current token's cos/sin table
//
// The result is the RoPE-rotated value for lane `tid`. The caller writes it
// back to the activation buffer after the call.

float16_t rope_neox(uint tid, uint cs_base) {
    if (tid < HALF_DIM) {
        float16_t c = rope_cs[cs_base + tid * 2u];
        float16_t s = rope_cs[cs_base + tid * 2u + 1u];
        return rope_in_sh[tid] * c - rope_in_sh[tid + HALF_DIM] * s;
    } else {
        uint j = tid - HALF_DIM;
        float16_t c = rope_cs[cs_base + j * 2u];
        float16_t s = rope_cs[cs_base + j * 2u + 1u];
        return rope_in_sh[j] * s + rope_in_sh[tid] * c;
    }
}
