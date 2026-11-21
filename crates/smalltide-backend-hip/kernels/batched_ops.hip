// SmallTide RDNA 4 (gfx1201) Parallel Batched Kernels for Concurrent Requests,
// Chunked Prefill, and `dspark` Speculative Verification:
// 1. `smalltide_gemm_fp8_e4m3_batched_f32`: reuses each FP8 weight load across up to 8 tokens!
// 2. `smalltide_gemm_nvfp4_batched_f32`   : unpacks each NVFP4 block ONCE across up to 8 tokens!
// 3. `smalltide_paged_flash_decode_fp8_split`: batched PagedAttention across `num_seqs` concurrent users!

// ---------------------------------------------------------------------------
// 1. Batched FP8 E4M3FN GEMM: Y[b, n] = sum_k X[b, k] * (W_fp8[n, k] * row_scale[n])
//    Processes up to BM=8 tokens per block-tile, reading each weight uint4 ONCE!
//    Grid : (ceil(N / 4), ceil(B / 8), 1)
//    Block: (128, 1, 1) -> 4 Wave32 warps, 1 warp per output row
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_gemm_fp8_e4m3_batched_f32(
    float* __restrict__ y,             // [B, N]
    const float* __restrict__ x,       // [B, K]
    const uint8_t* __restrict__ w_fp8, // [N, K]
    const float* __restrict__ row_scale,// [N] (nullable)
    uint32_t batch_size,
    uint32_t n_rows,
    uint32_t k_cols
) {
    uint32_t warp_in_block = threadIdx.x >> 5;
    uint32_t lane = threadIdx.x & 31u;
    uint32_t row = blockIdx.x * 4u + warp_in_block;
    uint32_t b0 = blockIdx.y * 8u;
    if (row >= n_rows || b0 >= batch_size) return;

    uint32_t bm = min(8u, batch_size - b0);
    const uint4* w_vec = reinterpret_cast<const uint4*>(w_fp8 + static_cast<size_t>(row) * k_cols);
    uint32_t k_chunks16 = k_cols >> 4;

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint32_t c = lane; c < k_chunks16; c += 32u) {
        // Read & decode 16 FP8 weights ONCE into registers
        uint4 w16 = w_vec[c];
        float wf[16];
        wf[0]  = decode_fp8_e4m3fn(w16.x & 0xFFu);
        wf[1]  = decode_fp8_e4m3fn((w16.x >> 8) & 0xFFu);
        wf[2]  = decode_fp8_e4m3fn((w16.x >> 16) & 0xFFu);
        wf[3]  = decode_fp8_e4m3fn((w16.x >> 24) & 0xFFu);
        wf[4]  = decode_fp8_e4m3fn(w16.y & 0xFFu);
        wf[5]  = decode_fp8_e4m3fn((w16.y >> 8) & 0xFFu);
        wf[6]  = decode_fp8_e4m3fn((w16.y >> 16) & 0xFFu);
        wf[7]  = decode_fp8_e4m3fn((w16.y >> 24) & 0xFFu);
        wf[8]  = decode_fp8_e4m3fn(w16.z & 0xFFu);
        wf[9]  = decode_fp8_e4m3fn((w16.z >> 8) & 0xFFu);
        wf[10] = decode_fp8_e4m3fn((w16.z >> 16) & 0xFFu);
        wf[11] = decode_fp8_e4m3fn((w16.z >> 24) & 0xFFu);
        wf[12] = decode_fp8_e4m3fn(w16.w & 0xFFu);
        wf[13] = decode_fp8_e4m3fn((w16.w >> 8) & 0xFFu);
        wf[14] = decode_fp8_e4m3fn((w16.w >> 16) & 0xFFu);
        wf[15] = decode_fp8_e4m3fn((w16.w >> 24) & 0xFFu);

        uint32_t base4 = c << 2;
        #pragma unroll
        for (uint32_t m = 0u; m < 8u; ++m) {
            if (m < bm) {
                const float4* xv = reinterpret_cast<const float4*>(x + static_cast<size_t>(b0 + m) * k_cols);
                float4 x0 = xv[base4 + 0];
                float4 x1 = xv[base4 + 1];
                float4 x2 = xv[base4 + 2];
                float4 x3 = xv[base4 + 3];
                float a = acc[m];
                a = fmaf(wf[0], x0.x, a);  a = fmaf(wf[1], x0.y, a);  a = fmaf(wf[2], x0.z, a);  a = fmaf(wf[3], x0.w, a);
                a = fmaf(wf[4], x1.x, a);  a = fmaf(wf[5], x1.y, a);  a = fmaf(wf[6], x1.z, a);  a = fmaf(wf[7], x1.w, a);
                a = fmaf(wf[8], x2.x, a);  a = fmaf(wf[9], x2.y, a);  a = fmaf(wf[10], x2.z, a); a = fmaf(wf[11], x2.w, a);
                a = fmaf(wf[12], x3.x, a); a = fmaf(wf[13], x3.y, a); a = fmaf(wf[14], x3.z, a); a = fmaf(wf[15], x3.w, a);
                acc[m] = a;
            }
        }
    }

    float sc = (row_scale != nullptr) ? row_scale[row] : 1.0f;
    #pragma unroll
    for (uint32_t m = 0u; m < 8u; ++m) {
        if (m < bm) {
            float sum = wave32_sum(acc[m]);
            if (lane == 0u) {
                y[static_cast<size_t>(b0 + m) * n_rows + row] = sum * sc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Batched NVFP4 GEMM: Y[b, n] = sum_k X[b, k] * NVFP4(W[n, k]) * outer_scale
//    Unpacks each 16-element NVFP4 block ONCE and multiplies against up to BM=8 tokens!
//    Grid : (ceil(N / 4), ceil(B / 8), 1)
//    Block: (128, 1, 1)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_gemm_nvfp4_batched_f32(
    float* __restrict__ y,                // [B, N]
    const float* __restrict__ x,          // [B, K]
    const uint8_t* __restrict__ w_packed, // [N, K/2]
    const uint8_t* __restrict__ w_scales, // [N, K/16] FP8 E4M3
    float outer_scale,
    uint32_t batch_size,
    uint32_t n_rows,
    uint32_t k_cols
) {
    uint32_t warp_in_block = threadIdx.x >> 5;
    uint32_t lane = threadIdx.x & 31u;
    uint32_t row = blockIdx.x * 4u + warp_in_block;
    uint32_t b0 = blockIdx.y * 8u;
    if (row >= n_rows || b0 >= batch_size) return;

    uint32_t bm = min(8u, batch_size - b0);
    uint32_t n_blocks16 = k_cols >> 4;
    const uint2* w_row = reinterpret_cast<const uint2*>(w_packed + static_cast<size_t>(row) * (k_cols >> 1));
    const uint8_t* s_row = w_scales + static_cast<size_t>(row) * n_blocks16;

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint32_t b = lane; b < n_blocks16; b += 32u) {
        uint2 w8 = w_row[b];
        float blk_scale = decode_fp8_e4m3fn(s_row[b]);
        float wf[16];
        uint32_t lo = w8.x;
        wf[0]  = decode_fp4_e2m1(lo & 0xFu);
        wf[1]  = decode_fp4_e2m1((lo >> 4) & 0xFu);
        wf[2]  = decode_fp4_e2m1((lo >> 8) & 0xFu);
        wf[3]  = decode_fp4_e2m1((lo >> 12) & 0xFu);
        wf[4]  = decode_fp4_e2m1((lo >> 16) & 0xFu);
        wf[5]  = decode_fp4_e2m1((lo >> 20) & 0xFu);
        wf[6]  = decode_fp4_e2m1((lo >> 24) & 0xFu);
        wf[7]  = decode_fp4_e2m1((lo >> 28) & 0xFu);
        uint32_t hi = w8.y;
        wf[8]  = decode_fp4_e2m1(hi & 0xFu);
        wf[9]  = decode_fp4_e2m1((hi >> 4) & 0xFu);
        wf[10] = decode_fp4_e2m1((hi >> 8) & 0xFu);
        wf[11] = decode_fp4_e2m1((hi >> 12) & 0xFu);
        wf[12] = decode_fp4_e2m1((hi >> 16) & 0xFu);
        wf[13] = decode_fp4_e2m1((hi >> 20) & 0xFu);
        wf[14] = decode_fp4_e2m1((hi >> 24) & 0xFu);
        wf[15] = decode_fp4_e2m1((hi >> 28) & 0xFu);

        uint32_t base4 = b << 2;
        #pragma unroll
        for (uint32_t m = 0u; m < 8u; ++m) {
            if (m < bm) {
                const float4* xv = reinterpret_cast<const float4*>(x + static_cast<size_t>(b0 + m) * k_cols);
                float4 x0 = xv[base4 + 0];
                float4 x1 = xv[base4 + 1];
                float4 x2 = xv[base4 + 2];
                float4 x3 = xv[base4 + 3];
                float sub = 0.0f;
                sub = fmaf(wf[0], x0.x, sub);  sub = fmaf(wf[1], x0.y, sub);  sub = fmaf(wf[2], x0.z, sub);  sub = fmaf(wf[3], x0.w, sub);
                sub = fmaf(wf[4], x1.x, sub);  sub = fmaf(wf[5], x1.y, sub);  sub = fmaf(wf[6], x1.z, sub);  sub = fmaf(wf[7], x1.w, sub);
                sub = fmaf(wf[8], x2.x, sub);  sub = fmaf(wf[9], x2.y, sub);  sub = fmaf(wf[10], x2.z, sub); sub = fmaf(wf[11], x2.w, sub);
                sub = fmaf(wf[12], x3.x, sub); sub = fmaf(wf[13], x3.y, sub); sub = fmaf(wf[14], x3.z, sub); sub = fmaf(wf[15], x3.w, sub);
                acc[m] = fmaf(sub, blk_scale, acc[m]);
            }
        }
    }

    #pragma unroll
    for (uint32_t m = 0u; m < 8u; ++m) {
        if (m < bm) {
            float sum = wave32_sum(acc[m]);
            if (lane == 0u) {
                y[static_cast<size_t>(b0 + m) * n_rows + row] = sum * outer_scale;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Batched PagedAttention FP8 Split-K FlashDecoding for Concurrent Requests!
//    Page layout (PAGE_TOKENS = 16):
//      k_pages, v_pages: [num_phys_pages, n_kv_heads, 16, 128] uint8 FP8
//      k_scales, v_scales: [num_phys_pages, n_kv_heads, 16] float
//      block_tables: [num_seqs, max_pages_per_seq] uint32
//      seq_lens: [num_seqs] uint32
//    Grid : (n_kv_heads, num_splits, num_seqs) -> processes ALL concurrent users in 1 launch!
//    Block: (128, 1, 1)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_paged_flash_decode_fp8_split(
    float* __restrict__ partial_out,       // [num_seqs, n_q_heads, num_splits, 128]
    float* __restrict__ partial_m,         // [num_seqs, n_q_heads, num_splits]
    float* __restrict__ partial_l,         // [num_seqs, n_q_heads, num_splits]
    const float* __restrict__ q,           // [num_seqs, n_q_heads, 128]
    const uint8_t* __restrict__ k_pages,   // [num_phys_pages, n_kv_heads, 16, 128]
    const float* __restrict__ k_scales,    // [num_phys_pages, n_kv_heads, 16]
    const uint8_t* __restrict__ v_pages,   // [num_phys_pages, n_kv_heads, 16, 128]
    const float* __restrict__ v_scales,    // [num_phys_pages, n_kv_heads, 16]
    const uint32_t* __restrict__ block_tables,
    const uint32_t* __restrict__ seq_lens,
    uint32_t num_seqs,
    uint32_t max_pages_per_seq,
    uint32_t n_q_heads,
    uint32_t n_kv_heads,
    uint32_t num_splits,
    float inv_sqrt_d
) {
    uint32_t kvh = blockIdx.x;
    uint32_t split = blockIdx.y;
    uint32_t seq = blockIdx.z;
    if (kvh >= n_kv_heads || split >= num_splits || seq >= num_seqs) return;

    uint32_t seq_len = seq_lens[seq];
    uint32_t gqa_ratio = n_q_heads / n_kv_heads;
    uint32_t qh0 = kvh * gqa_ratio;

    uint32_t tokens_per_split = (seq_len + num_splits - 1u) / num_splits;
    uint32_t t_start = split * tokens_per_split;
    uint32_t t_end = min(t_start + tokens_per_split, seq_len);

    uint32_t warp_id = threadIdx.x >> 5;
    uint32_t lane = threadIdx.x & 31u;

    const float* q_seq = q + static_cast<size_t>(seq) * n_q_heads * 128u;
    float4 qv0 = reinterpret_cast<const float4*>(q_seq + (qh0 + 0u) * 128u)[lane];
    float4 qv1 = (gqa_ratio > 1u) ? reinterpret_cast<const float4*>(q_seq + (qh0 + 1u) * 128u)[lane] : make_float4(0,0,0,0);
    float4 qv2 = (gqa_ratio > 2u) ? reinterpret_cast<const float4*>(q_seq + (qh0 + 2u) * 128u)[lane] : make_float4(0,0,0,0);

    float m0 = -1e30f, l0 = 0.0f; float4 o0 = make_float4(0,0,0,0);
    float m1 = -1e30f, l1 = 0.0f; float4 o1 = make_float4(0,0,0,0);
    float m2 = -1e30f, l2 = 0.0f; float4 o2 = make_float4(0,0,0,0);

    const uint32_t* bt = block_tables + static_cast<size_t>(seq) * max_pages_per_seq;
    const float inv_sqrt_d_log2 = inv_sqrt_d * 1.4426950408889634f;
    size_t page_kv_stride = static_cast<size_t>(n_kv_heads) * 16u * 128u;
    size_t page_sc_stride = static_cast<size_t>(n_kv_heads) * 16u;

    for (uint32_t t = t_start + warp_id; t < t_end; t += 4u) {
        uint32_t logical_page = t >> 4;
        uint32_t in_page = t & 15u;
        uint32_t phys_page = bt[logical_page];

        size_t kv_off = static_cast<size_t>(phys_page) * page_kv_stride
                      + (static_cast<size_t>(kvh) * 16u + in_page) * 128u;
        size_t sc_off = static_cast<size_t>(phys_page) * page_sc_stride
                      + static_cast<size_t>(kvh) * 16u + in_page;

        uint32_t k4 = reinterpret_cast<const uint32_t*>(k_pages + kv_off)[lane];
        float kx = decode_fp8_e4m3fn(k4 & 0xFFu);
        float ky = decode_fp8_e4m3fn((k4 >> 8) & 0xFFu);
        float kz = decode_fp8_e4m3fn((k4 >> 16) & 0xFFu);
        float kw = decode_fp8_e4m3fn((k4 >> 24) & 0xFFu);
        float k_scale = k_scales[sc_off] * inv_sqrt_d_log2;

        uint32_t v4 = reinterpret_cast<const uint32_t*>(v_pages + kv_off)[lane];
        float v_scale = v_scales[sc_off];
        float vx = decode_fp8_e4m3fn(v4 & 0xFFu) * v_scale;
        float vy = decode_fp8_e4m3fn((v4 >> 8) & 0xFFu) * v_scale;
        float vz = decode_fp8_e4m3fn((v4 >> 16) & 0xFFu) * v_scale;
        float vw = decode_fp8_e4m3fn((v4 >> 24) & 0xFFu) * v_scale;

        {
            float d = fmaf(kx, qv0.x, fmaf(ky, qv0.y, fmaf(kz, qv0.z, kw * qv0.w)));
            float sc = wave32_sum(d) * k_scale;
            float mn = fmaxf(m0, sc);
            float al = exp2f(m0 - mn);
            float p  = exp2f(sc - mn);
            l0 = l0 * al + p; m0 = mn;
            o0.x = fmaf(vx, p, o0.x * al); o0.y = fmaf(vy, p, o0.y * al);
            o0.z = fmaf(vz, p, o0.z * al); o0.w = fmaf(vw, p, o0.w * al);
        }
        if (gqa_ratio > 1u) {
            float d = fmaf(kx, qv1.x, fmaf(ky, qv1.y, fmaf(kz, qv1.z, kw * qv1.w)));
            float sc = wave32_sum(d) * k_scale;
            float mn = fmaxf(m1, sc);
            float al = exp2f(m1 - mn);
            float p  = exp2f(sc - mn);
            l1 = l1 * al + p; m1 = mn;
            o1.x = fmaf(vx, p, o1.x * al); o1.y = fmaf(vy, p, o1.y * al);
            o1.z = fmaf(vz, p, o1.z * al); o1.w = fmaf(vw, p, o1.w * al);
        }
        if (gqa_ratio > 2u) {
            float d = fmaf(kx, qv2.x, fmaf(ky, qv2.y, fmaf(kz, qv2.z, kw * qv2.w)));
            float sc = wave32_sum(d) * k_scale;
            float mn = fmaxf(m2, sc);
            float al = exp2f(m2 - mn);
            float p  = exp2f(sc - mn);
            l2 = l2 * al + p; m2 = mn;
            o2.x = fmaf(vx, p, o2.x * al); o2.y = fmaf(vy, p, o2.y * al);
            o2.z = fmaf(vz, p, o2.z * al); o2.w = fmaf(vw, p, o2.w * al);
        }
    }

    const float LN2 = 0.6931471805599453f;
    m0 *= LN2; m1 *= LN2; m2 *= LN2;

    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float4 s_o[4][32];

    auto flush_q = [&](uint32_t qh, float mi, float li, float4 oi) {
        if (lane == 0u) { s_m[warp_id] = mi; s_l[warp_id] = li; }
        s_o[warp_id][lane] = oi;
        __syncthreads();
        if (warp_id == 0u) {
            float m_blk = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3]));
            float l_blk = 0.0f;
            float4 o_blk = make_float4(0,0,0,0);
            for (uint32_t w = 0u; w < 4u; ++w) {
                if (s_l[w] > 0.0f) {
                    float f = expf(s_m[w] - m_blk);
                    l_blk += s_l[w] * f;
                    float4 ow = s_o[w][lane];
                    o_blk.x = fmaf(ow.x, f, o_blk.x); o_blk.y = fmaf(ow.y, f, o_blk.y);
                    o_blk.z = fmaf(ow.z, f, o_blk.z); o_blk.w = fmaf(ow.w, f, o_blk.w);
                }
            }
            size_t meta_idx = (static_cast<size_t>(seq) * n_q_heads + qh) * num_splits + split;
            if (lane == 0u) {
                partial_m[meta_idx] = m_blk;
                partial_l[meta_idx] = l_blk;
            }
            reinterpret_cast<float4*>(partial_out + meta_idx * 128u)[lane] = o_blk;
        }
        __syncthreads();
    };

    flush_q(qh0 + 0u, m0, l0, o0);
    if (gqa_ratio > 1u) flush_q(qh0 + 1u, m1, l1, o1);
    if (gqa_ratio > 2u) flush_q(qh0 + 2u, m2, l2, o2);
}
