#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

// Q1_0 weights × Q8_0 activations -> f16 output. 64×64 tile, 256 threads.
//
// Apple variant: uses `simdgroup_matrix<half, 8, 8>` instructions for the inner
// matmul. M2 Pro and newer Apple GPUs lower simdgroup_matrix to the same f16
// FMA pipe, but the instruction lets the compiler schedule the work better via
// register tiling and broadcast reuse. Q1_0 weights are materialized to fp16
// just-in-time (per Q1_0 block, cooperatively, into threadgroup memory) so
// that the B tile is a contiguous fp16 block. Q8_0 activations are
// dequantized to fp16 once during the cooperative load into shmem.
//
// Tile layout:
//   - WG = 256 threads = 8 simdgroups (simd_size = 32 on Apple).
//   - Simdgroup grid: 2 (M) × 4 (N) within the WG.
//   - Per simdgroup: TM_SG = 4, TN_SG = 2 -> 32 (M) × 16 (N) output tile.
//   - WG output tile: 64 (M) × 64 (N).
//
// Accumulators are fp32 (`simdgroup_matrix<float, 8, 8>`) — fp16 C would
// overflow for K up to ~5760.
//
// Threadgroup memory: 16 KB a_sh (Q8_0 dequantized) + 16 KB w_sh (full Q1_0
// block, two contiguous 8 KB halves) = 32 KB, exactly the per-WG limit on
// Apple GPUs. Materializing the full block in one cooperative pass lets all
// 256 threads participate (the previous half-at-a-time scheme idled half the
// WG) and removes the per-half threadgroup barriers: 2 barriers per block
// iteration instead of 5. Weight scales `dw` are loaded inline per
// materialization thread rather than staged through shmem; the 4 threads
// sharing an n_local hit the same address so the load is cache-resident.
// Sign-word loads are coalesced by mapping (n_local = tid/4, s = tid%4) —
// 4 adjacent lanes hit one row's 16 contiguous bytes.

struct Params {
    uint k;
    uint n;
    uint m;
    uint w_d_offset;
    uint w_qs_offset;
    uint a_d_offset;
    uint a_qs_offset;
    uint out_offset;
    uint accumulate;
};

constant uint WG = 256u;
constant uint SG_GRID_M = 2u;
constant uint SG_GRID_N = 4u;
constant uint TM_SG = 4u;
constant uint TN_SG = 2u;
constant uint TILE_M = 64u; // SG_GRID_M * TM_SG * 8
constant uint TILE_N = 64u; // SG_GRID_N * TN_SG * 8
constant uint W_HALF = 64u * TILE_N;

kernel void cs_main(
    constant Params &p [[buffer(0)]],
    device const uint *weights [[buffer(1)]],
    device const uint *acts [[buffer(2)]],
    device half *y [[buffer(3)]],
    uint3 wg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]]
) {
    // Full-Q1_0-block fp16 weight tile, row-major as B[k_in_block, n_local],
    // laid out as two contiguous halves of 64 K-rows × 64 N-cols each (8 KB
    // per half, 16 KB total). The MMA inner loop walks the halves
    // sequentially; materialization fills both at once so all 256 threads
    // are active.
    threadgroup half w_sh[2u * W_HALF];
    // Activations as half4, dequantized from Q8_0 once per Q1_0 block.
    // Layout: a_sh[m_local * 32 + v4_local], v4_local = sub*8 + i covers
    // [b*128 + sub*32 + i*4 .. i*4 + 3] in element space. 64 tokens × 32 half4
    // = 2048 half4 = 16 KB.
    threadgroup half4 a_sh[TILE_M * 32u];

    const uint n_base = wg_id.x * TILE_N;
    const uint m_base = wg_id.y * TILE_M;
    const uint nb_q1 = p.k / 128u;
    const uint nb_q8 = p.k / 32u;

    // Simdgroup position within the WG (4 in M, 2 in N).
    const uint sg_m = sg_id / SG_GRID_N;      // 0..3
    const uint sg_n = sg_id % SG_GRID_N;      // 0..1
    const uint sg_m_base = sg_m * TM_SG * 8u; // 0, 16, 32, 48
    const uint sg_n_base = sg_n * TN_SG * 8u; // 0, 32

    // C accumulators in registers, fp32.
    simdgroup_matrix<float, 8, 8> C[TM_SG][TN_SG];
    for (uint im = 0u; im < TM_SG; ++im) {
        for (uint in_ = 0u; in_ < TN_SG; ++in_) {
            C[im][in_] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    threadgroup const half *a_base = (threadgroup const half *)a_sh + sg_m_base * 128u; // (sg_m_base) M-rows offset

    for (uint b = 0u; b < nb_q1; ++b) {
        // ---- Cooperative load: activations (dequant Q8_0 -> half4) ----
        // 2048 half4 across 256 threads -> 8 half4 per thread.
        for (uint li = 0u; li < 8u; ++li) {
            uint idx = li * WG + tid;       // 0..2047
            uint m_local = idx / 32u;       // 0..63 (token within tile)
            uint v4_local = idx % 32u;      // 0..31 (half4 index within 128 elems)
            uint sub = v4_local / 8u;       // 0..3 (Q8_0 sub-block)
            uint v4_in_sub = v4_local % 8u; // 0..7 (half4 within sub-block)

            uint m_idx = m_base + m_local;
            uint a_block = b * 4u + sub;

            uint d_off = p.a_d_offset + (m_idx * nb_q8 + a_block) * 4u;
            float a_d = as_type<float>(acts[d_off >> 2]);

            uint qs_off = p.a_qs_offset + m_idx * p.k + b * 128u + sub * 32u + v4_in_sub * 4u;
            uint q_packed = acts[qs_off >> 2];
            char4 q_chars = as_type<char4>(q_packed);

            float4 q_f = float4(q_chars) * a_d;
            a_sh[idx] = half4(q_f);
        }

        // ---- Cooperative materialize: full Q1_0 block (both halves) ----
        // Each of 256 threads owns one (n_local, s) pair where
        //   n_local = tid / 4   (0..63, the N-col)
        //   s       = tid % 4   (0..3, the sign-byte chunk in the 16-byte block)
        // s = 2*half_idx + kc_in_half — s=0,1 → half 0; s=2,3 → half 1.
        // Adjacent lanes (varying s) read 4 consecutive uints from the same
        // row, coalescing the 16-byte row into one transaction; the 4 lanes
        // sharing an n_local hit the same `dw` address so the inline scale
        // load is cache-resident.
        {
            uint n_local = tid / 4u;
            uint s = tid % 4u;
            uint half_idx = s >> 1u;
            uint kc = s & 1u;
            uint n_idx = n_base + n_local;
            uint sign_off = p.w_qs_offset + n_idx * (nb_q1 * 16u) + b * 16u + s * 4u;
            uint sign_word = weights[sign_off >> 2];
            uint b_offset = p.w_d_offset + (n_idx * nb_q1 + b) * 2u;
            uint word = weights[b_offset >> 2];
            uint half_bits = (word >> ((b_offset & 2u) * 8u));
            half dw = as_type<half2>(half_bits).x;
            half neg_dw = -dw;
            threadgroup half *w_dst = w_sh + half_idx * W_HALF;
            uint k0 = kc * 32u;
            for (uint i = 0u; i < 32u; ++i) {
                w_dst[(k0 + i) * TILE_N + n_local] = select(neg_dw, dw, ((sign_word >> i) & 1u) != 0u);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- MMA both halves; no inter-half barrier needed ----
        for (uint half_idx = 0u; half_idx < 2u; ++half_idx) {
            threadgroup const half *b_base_sg = w_sh + half_idx * W_HALF + sg_n_base;
            uint k_base_in_block = half_idx * 64u;
            for (uint ks = 0u; ks < 8u; ++ks) {
                uint k_off_in_block = k_base_in_block + ks * 8u;
                uint k_off_in_half = ks * 8u;

                simdgroup_matrix<half, 8, 8> Atiles[TM_SG];
                for (uint im = 0u; im < TM_SG; ++im) {
                    threadgroup const half *a_ptr = a_base + (im * 8u) * 128u + k_off_in_block;
                    simdgroup_load(Atiles[im], a_ptr, 128u);
                }

                simdgroup_matrix<half, 8, 8> Btiles[TN_SG];
                for (uint in_ = 0u; in_ < TN_SG; ++in_) {
                    threadgroup const half *b_ptr = b_base_sg + k_off_in_half * TILE_N + in_ * 8u;
                    simdgroup_load(Btiles[in_], b_ptr, TILE_N);
                }

                for (uint im = 0u; im < TM_SG; ++im) {
                    for (uint in_ = 0u; in_ < TN_SG; ++in_) {
                        simdgroup_multiply_accumulate(C[im][in_], Atiles[im], Btiles[in_], C[im][in_]);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Store accumulators to global y ----
    // Each simdgroup writes its 16×32 sub-tile. Store via simdgroup_store into
    // a small per-WG fp32 staging area would cost shmem; instead, extract via
    // thread_elements() and write directly to global. Simpler: store C tile by
    // tile to a small shmem fp32 buffer, then cooperatively write to y with
    // the accumulate / bounds logic. Reuses a_sh's space (sized 16 KB > 16 KB
    // needed for 64×64 fp32 = 16 KB).
    threadgroup float *c_sh = (threadgroup float *)a_sh;
    for (uint im = 0u; im < TM_SG; ++im) {
        for (uint in_ = 0u; in_ < TN_SG; ++in_) {
            uint m_off = sg_m_base + im * 8u;
            uint n_off = sg_n_base + in_ * 8u;
            simdgroup_store(C[im][in_], c_sh + m_off * TILE_N + n_off, TILE_N);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cooperatively write 64*64 = 4096 outputs across 256 threads = 16 each.
    for (uint li = 0u; li < 16u; ++li) {
        uint idx = li * WG + tid;    // 0..4095
        uint m_local = idx / TILE_N; // 0..63
        uint n_local = idx % TILE_N; // 0..63
        uint m_idx = m_base + m_local;
        if (m_idx >= p.m)
            continue;
        uint n_idx = n_base + n_local;
        if (n_idx >= p.n)
            continue;
        uint yi = p.out_offset + m_idx * p.n + n_idx;
        float val = c_sh[m_local * TILE_N + n_local];
        y[yi] = half(select(0.0f, float(y[yi]), p.accumulate != 0u) + val);
    }
}
