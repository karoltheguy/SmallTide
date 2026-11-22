// SmallTide RDNA 4 (gfx1201) rows-batched kernels for chunked prefill (M tokens per launch).
// Layouts are row-major with the token row outermost unless noted, so M = 1 reproduces the
// single-token decode layouts exactly.

// ---------------------------------------------------------------------------
// Batched F32 GEMM: Y[m, n] = sum_k X[m, k] * W[n, k]  (router, sinks, dyn conv taps)
//    Grid: (ceil(N / 4), ceil(M / 8)), Block: 128 (1 warp per output column n)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_gemm_f32_rows(
    float* __restrict__ y, const float* __restrict__ x, const float* __restrict__ w,
    uint32_t m_rows, uint32_t n_rows, uint32_t k_cols
) {
    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t n = blockIdx.x * 4u + wid;
    uint32_t b0 = blockIdx.y * 8u;
    if (n >= n_rows || b0 >= m_rows) return;
    uint32_t bm = min(8u, m_rows - b0);
    const float4* wv = reinterpret_cast<const float4*>(w + static_cast<size_t>(n) * k_cols);
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t i = lane; i < (k_cols >> 2); i += 32u) {
        float4 a = wv[i];
        #pragma unroll
        for (uint32_t r = 0; r < 8u; ++r) {
            if (r < bm) {
                float4 b = reinterpret_cast<const float4*>(x + static_cast<size_t>(b0 + r) * k_cols)[i];
                acc[r] = fmaf(a.x, b.x, fmaf(a.y, b.y, fmaf(a.z, b.z, fmaf(a.w, b.w, acc[r]))));
            }
        }
    }
    #pragma unroll
    for (uint32_t r = 0; r < 8u; ++r) {
        if (r < bm) {
            float s = wave32_sum(acc[r]);
            if (lane == 0u) y[static_cast<size_t>(b0 + r) * n_rows + n] = s;
        }
    }
}

// ---------------------------------------------------------------------------
// Attention-residual mix for M rows: out[q][m] from states [blocks[s][m]..., cur[m]].
//    blocks: [nb][M][H], cur: [M][H], queries: [nq][H], out: [nq][M][H]
//    Grid: (nq, M), Block: 256
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_attn_res_mix_rows_f32(
    float* __restrict__ out, const float* __restrict__ blocks, const float* __restrict__ cur,
    const float* __restrict__ queries, uint32_t num_blocks, uint32_t num_queries, uint32_t hidden,
    float eps, uint32_t normalize_values, uint32_t m_rows
) {
    uint32_t q_idx = blockIdx.x, row = blockIdx.y;
    if (q_idx >= num_queries || row >= m_rows) return;
    const float* query = queries + static_cast<size_t>(q_idx) * hidden;
    float* out_row = out + (static_cast<size_t>(q_idx) * m_rows + row) * hidden;
    const float* cur_row = cur + static_cast<size_t>(row) * hidden;
    uint32_t total = num_blocks + 1u;
    __shared__ float s_scores[16];
    __shared__ float s_inv[16];
    __shared__ float s_wsq[32];
    __shared__ float s_wdot[32];
    uint32_t lane = threadIdx.x & 31u, wid = threadIdx.x >> 5;
    float inv_sqrt_h = rsqrtf(static_cast<float>(hidden));
    for (uint32_t s = 0u; s < total; ++s) {
        const float* st = (s < num_blocks) ? (blocks + (static_cast<size_t>(s) * m_rows + row) * hidden) : cur_row;
        float sq = 0.0f, dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < hidden; d += blockDim.x) {
            float v = st[d];
            sq = fmaf(v, v, sq);
            dot = fmaf(v, query[d], dot);
        }
        float a = wave32_sum(sq), b = wave32_sum(dot);
        if (lane == 0u) { s_wsq[wid] = a; s_wdot[wid] = b; }
        __syncthreads();
        if (threadIdx.x == 0u) {
            float tq = 0.0f, td = 0.0f;
            for (uint32_t w = 0u; w < (blockDim.x >> 5); ++w) { tq += s_wsq[w]; td += s_wdot[w]; }
            float ir = rsqrtf(tq / static_cast<float>(hidden) + eps);
            s_inv[s] = ir;
            s_scores[s] = td * ir * inv_sqrt_h;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        float m = s_scores[0];
        for (uint32_t s = 1u; s < total; ++s) m = fmaxf(m, s_scores[s]);
        float sum = 0.0f;
        for (uint32_t s = 0u; s < total; ++s) { s_scores[s] = expf(s_scores[s] - m); sum += s_scores[s]; }
        float inv = 1.0f / fmaxf(sum, 1e-12f);
        for (uint32_t s = 0u; s < total; ++s) s_scores[s] *= inv;
    }
    __syncthreads();
    for (uint32_t d = threadIdx.x; d < hidden; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t s = 0u; s < total; ++s) {
            const float* st = (s < num_blocks) ? (blocks + (static_cast<size_t>(s) * m_rows + row) * hidden) : cur_row;
            float v = st[d];
            if (normalize_values != 0u) v *= s_inv[s];
            acc = fmaf(s_scores[s], v, acc);
        }
        out_row[d] = acc;
    }
}

// ---------------------------------------------------------------------------
// Per-head L2 clip for M rows (heads of all rows flattened), optional per-channel scale
// (period = heads_per_row) and optional log-scaled tau of the row's position (global layers).
//    Grid: (total_heads), Block: 128
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_head_l2_clip_rows_f32(
    float* __restrict__ x, const float* __restrict__ scale, uint32_t total_heads, uint32_t head_dim,
    uint32_t heads_per_row, float threshold, uint32_t pos0, float tau_alpha, float tau_floor
) {
    uint32_t h = blockIdx.x;
    if (h >= total_heads) return;
    float* rowp = x + static_cast<size_t>(h) * head_dim;
    const float* srow = (scale != nullptr) ? (scale + static_cast<size_t>(h % heads_per_row) * head_dim) : nullptr;
    float sq = 0.0f;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float v = rowp[d];
        if (srow != nullptr) { v *= srow[d]; rowp[d] = v; }
        sq = fmaf(v, v, sq);
    }
    __shared__ float wsum[4];
    __shared__ float factor;
    float s = wave32_sum(sq);
    if ((threadIdx.x & 31u) == 0u) wsum[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0u) {
        float tot = 0.0f;
        for (uint32_t w = 0u; w < (blockDim.x >> 5); ++w) tot += wsum[w];
        float clip = threshold / (sqrtf(tot) + 1e-6f);
        float f = (clip < 1.0f) ? clip : 1.0f;
        if (tau_alpha != 0.0f) {
            float p = static_cast<float>(pos0 + h / heads_per_row) + 1.0f;
            f *= 1.0f + tau_alpha * logf(fmaxf(p / tau_floor, 1.0f));
        }
        factor = f;
    }
    __syncthreads();
    float f = factor;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) rowp[d] *= f;
}

// ---------------------------------------------------------------------------
// Causal short conv (k=4) over M consecutive tokens, continuing from / updating `state`.
//    out, x: [M][dim]; state: [3][dim] = (t-3, t-2, t-1); weight: [dim][4];
//    dyn: nullable [M][dyn_stride] with 4 lag taps per head (lag0 = current token).
//    Grid: ceil(dim / 256), Block: 256 (one thread per channel, sequential over rows)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_sconv_k4_rows_f32(
    float* __restrict__ out, float* __restrict__ state, const float* __restrict__ x,
    const float* __restrict__ weight, const float* __restrict__ dyn, uint32_t dim, uint32_t head_dim,
    uint32_t m_rows, uint32_t dyn_stride, float* __restrict__ hist
) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= dim) return;
    float s0 = state[c], s1 = state[dim + c], s2 = state[2u * dim + c];
    if (hist != nullptr) {
        // [3 + m_rows][dim]: the old state, then every input (speculative-verify rewind).
        hist[c] = s0; hist[dim + c] = s1; hist[2u * dim + c] = s2;
        for (uint32_t t = 0u; t < m_rows; ++t) hist[static_cast<size_t>(3u + t) * dim + c] = x[static_cast<size_t>(t) * dim + c];
    }
    const float4 w = reinterpret_cast<const float4*>(weight)[c];
    uint32_t h = (head_dim > 0u) ? (c / head_dim) : 0u;
    for (uint32_t t = 0u; t < m_rows; ++t) {
        float cur = x[static_cast<size_t>(t) * dim + c];
        float y = fmaf(s0, w.x, fmaf(s1, w.y, fmaf(s2, w.z, cur * w.w)));
        if (dyn != nullptr) {
            const float4 d = reinterpret_cast<const float4*>(dyn + static_cast<size_t>(t) * dyn_stride)[h];
            y = fmaf(cur, d.x, fmaf(s2, d.y, fmaf(s1, d.z, fmaf(s0, d.w, y))));
            y = (cur + y) * (1.0f / 3.0f);
        } else {
            y = (cur + y) * 0.4472135954999579f;
        }
        out[static_cast<size_t>(t) * dim + c] = y;
        s0 = s1; s1 = s2; s2 = cur;
    }
    state[c] = s0; state[dim + c] = s1; state[2u * dim + c] = s2;
}

// ---------------------------------------------------------------------------
// FP8 KV store for M rows at positions pos0..pos0+M-1 (ring slot = pos % max_seq if local).
//    k_in, v_in: [M][n_kv_heads][head_dim]; Grid: (n_kv_heads, M), Block: head_dim (128)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_kv_store_fp8_rows(
    uint8_t* __restrict__ k_cache, float* __restrict__ k_scales, uint8_t* __restrict__ v_cache,
    float* __restrict__ v_scales, const float* __restrict__ k_in, const float* __restrict__ v_in,
    uint32_t n_kv_heads, uint32_t max_seq, uint32_t head_dim, uint32_t pos0, uint32_t is_local
) {
    uint32_t h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    if (h >= n_kv_heads || d >= head_dim) return;
    size_t src = (static_cast<size_t>(t) * n_kv_heads + h) * head_dim + d;
    float kv = k_in[src], vv = v_in[src];
    __shared__ float sk[4], sv[4];
    float km = fabsf(kv), vm = fabsf(vv);
    for (int o = 16; o > 0; o >>= 1) {
        km = fmaxf(km, __shfl_xor(km, o, 32));
        vm = fmaxf(vm, __shfl_xor(vm, o, 32));
    }
    if ((d & 31u) == 0u) { sk[d >> 5] = km; sv[d >> 5] = vm; }
    __syncthreads();
    float kmax = fmaxf(fmaxf(sk[0], sk[1]), fmaxf(sk[2], sk[3]));
    float vmax = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3]));
    float ks = (kmax > 0.0f) ? (kmax / 448.0f) : 1.0f;
    float vs = (vmax > 0.0f) ? (vmax / 448.0f) : 1.0f;
    uint32_t pos = pos0 + t;
    uint32_t slot = (is_local != 0u) ? (pos % max_seq) : pos;
    size_t si = static_cast<size_t>(h) * max_seq + slot;
    if (d == 0u) { k_scales[si] = ks; v_scales[si] = vs; }
    k_cache[si * head_dim + d] = encode_fp8_e4m3fn(kv / ks);
    v_cache[si * head_dim + d] = encode_fp8_e4m3fn(vv / vs);
}

// ---------------------------------------------------------------------------
// Causal prefill attention over the FP8 KV cache with relative bias and sink logits.
//    q, out: [M][n_q_heads][128]; rel_table: nullable [M][n_q_heads][rel_extent]; sinks: [M][n_q_heads]
//    Row t is position p = pos0 + t; keys [max(0, p+1-window), p] (window = 0 -> all of [0, p]).
//    Grid: (n_kv_heads, M), Block: 128 (4 warps stride over keys, 3 GQA q heads per block)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_attn_prefill_fp8(
    float* __restrict__ out, const float* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const float* __restrict__ k_scales, const uint8_t* __restrict__ v_cache, const float* __restrict__ v_scales,
    const float* __restrict__ rel_table, const float* __restrict__ sinks, uint32_t n_q_heads, uint32_t n_kv_heads,
    uint32_t max_seq, uint32_t window, uint32_t pos0, uint32_t rel_extent, float inv_sqrt_d
) {
    uint32_t kvh = blockIdx.x, t = blockIdx.y;
    if (kvh >= n_kv_heads) return;
    uint32_t g = n_q_heads / n_kv_heads; // 3
    uint32_t qh0 = kvh * g;
    uint32_t p = pos0 + t;
    uint32_t k_start = (window != 0u && p + 1u > window) ? (p + 1u - window) : 0u;
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const float* qrow = q + static_cast<size_t>(t) * n_q_heads * 128u;
    float4 qv[3];
    for (uint32_t i = 0u; i < 3u; ++i) {
        qv[i] = (i < g) ? reinterpret_cast<const float4*>(qrow + (qh0 + i) * 128u)[lane] : make_float4(0, 0, 0, 0);
    }
    const float* rel[3];
    for (uint32_t i = 0u; i < 3u; ++i) {
        rel[i] = (rel_table != nullptr && i < g)
            ? rel_table + (static_cast<size_t>(t) * n_q_heads + qh0 + i) * rel_extent : nullptr;
    }
    const float LOG2E = 1.4426950408889634f;
    float sc_mul = inv_sqrt_d * LOG2E;
    float m[3] = {-1e30f, -1e30f, -1e30f}, l[3] = {0, 0, 0};
    float4 o[3] = {make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0)};
    const uint8_t* kb = k_cache + static_cast<size_t>(kvh) * max_seq * 128u;
    const uint8_t* vb = v_cache + static_cast<size_t>(kvh) * max_seq * 128u;
    const float* ksb = k_scales + static_cast<size_t>(kvh) * max_seq;
    const float* vsb = v_scales + static_cast<size_t>(kvh) * max_seq;
    for (uint32_t kp = k_start + warp; kp <= p; kp += 4u) {
        uint32_t slot = (window != 0u) ? (kp % max_seq) : kp;
        uint32_t dist = p - kp;
        uint32_t k4 = reinterpret_cast<const uint32_t*>(kb + static_cast<size_t>(slot) * 128u)[lane];
        float kx = decode_fp8_e4m3fn(k4 & 0xFFu), ky = decode_fp8_e4m3fn((k4 >> 8) & 0xFFu);
        float kz = decode_fp8_e4m3fn((k4 >> 16) & 0xFFu), kw = decode_fp8_e4m3fn((k4 >> 24) & 0xFFu);
        float kscale = ksb[slot] * sc_mul;
        uint32_t v4 = reinterpret_cast<const uint32_t*>(vb + static_cast<size_t>(slot) * 128u)[lane];
        float vscale = vsb[slot];
        float vx = decode_fp8_e4m3fn(v4 & 0xFFu) * vscale, vy = decode_fp8_e4m3fn((v4 >> 8) & 0xFFu) * vscale;
        float vz = decode_fp8_e4m3fn((v4 >> 16) & 0xFFu) * vscale, vw = decode_fp8_e4m3fn((v4 >> 24) & 0xFFu) * vscale;
        #pragma unroll
        for (uint32_t i = 0u; i < 3u; ++i) {
            if (i >= g) break;
            float d = fmaf(kx, qv[i].x, fmaf(ky, qv[i].y, fmaf(kz, qv[i].z, kw * qv[i].w)));
            float rb = (rel[i] != nullptr && dist < rel_extent) ? rel[i][dist] * LOG2E : 0.0f;
            float sc = wave32_sum(d) * kscale + rb;
            float mn = fmaxf(m[i], sc);
            float al = exp2f(m[i] - mn), pr = exp2f(sc - mn);
            l[i] = l[i] * al + pr; m[i] = mn;
            o[i].x = fmaf(vx, pr, o[i].x * al); o[i].y = fmaf(vy, pr, o[i].y * al);
            o[i].z = fmaf(vz, pr, o[i].z * al); o[i].w = fmaf(vw, pr, o[i].w * al);
        }
    }
    // Merge the 4 warps (log2 domain) and the sink logit, normalize, write.
    __shared__ float s_m[4], s_l[4];
    __shared__ float4 s_o[4][32];
    for (uint32_t i = 0u; i < g && i < 3u; ++i) {
        if (lane == 0u) { s_m[warp] = m[i]; s_l[warp] = l[i]; }
        s_o[warp][lane] = o[i];
        __syncthreads();
        if (warp == 0u) {
            uint32_t qh = qh0 + i;
            float sink = (sinks != nullptr) ? sinks[static_cast<size_t>(t) * n_q_heads + qh] * LOG2E : -1e30f;
            float mb = sink;
            for (uint32_t w = 0u; w < 4u; ++w) if (s_l[w] > 0.0f) mb = fmaxf(mb, s_m[w]);
            float lb = (sinks != nullptr) ? exp2f(sink - mb) : 0.0f;
            float4 ob = make_float4(0, 0, 0, 0);
            for (uint32_t w = 0u; w < 4u; ++w) {
                if (s_l[w] > 0.0f) {
                    float f = exp2f(s_m[w] - mb);
                    lb += s_l[w] * f;
                    float4 ow = s_o[w][lane];
                    ob.x = fmaf(ow.x, f, ob.x); ob.y = fmaf(ow.y, f, ob.y);
                    ob.z = fmaf(ow.z, f, ob.z); ob.w = fmaf(ow.w, f, ob.w);
                }
            }
            float inv = (lb > 1e-30f) ? 1.0f / lb : 0.0f;
            ob.x *= inv; ob.y *= inv; ob.z *= inv; ob.w *= inv;
            reinterpret_cast<float4*>(out + (static_cast<size_t>(t) * n_q_heads + qh) * 128u)[lane] = ob;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Router top-8 for M rows from precomputed logits: sigmoid scores, select by score + bias,
// weights = score / sum * route_scale. route: [M][16] u32 = 8 ids then 8 f32 weights.
//    Grid: M, Block: 128 (one thread per expert)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_router_topk8_rows(
    uint32_t* __restrict__ route, const float* __restrict__ logits, const float* __restrict__ bias,
    float route_scale, uint32_t row_stride_words
) {
    uint32_t row = blockIdx.x, e = threadIdx.x;
    __shared__ float s_sel[128];
    __shared__ float s_score[128];
    __shared__ float s_bv[4];
    __shared__ uint32_t s_bi[4];
    __shared__ uint32_t s_ids[8];
    float lg = logits[static_cast<size_t>(row) * 128u + e];
    float sc = 1.0f / (1.0f + expf(-lg));
    s_score[e] = sc;
    s_sel[e] = sc + ((bias != nullptr) ? bias[e] : 0.0f);
    __syncthreads();
    for (uint32_t k = 0u; k < 8u; ++k) {
        float v = s_sel[e];
        uint32_t i = e;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor(v, o, 32);
            uint32_t oi = __shfl_xor(i, o, 32);
            if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
        }
        if ((e & 31u) == 0u) { s_bv[e >> 5] = v; s_bi[e >> 5] = i; }
        __syncthreads();
        if (e == 0u) {
            float bv = s_bv[0]; uint32_t bi = s_bi[0];
            for (uint32_t w = 1u; w < 4u; ++w) {
                if (s_bv[w] > bv || (s_bv[w] == bv && s_bi[w] < bi)) { bv = s_bv[w]; bi = s_bi[w]; }
            }
            s_ids[k] = bi;
            s_sel[bi] = -1e30f;
        }
        __syncthreads();
    }
    if (e == 0u) {
        float sum = 0.0f;
        for (uint32_t k = 0u; k < 8u; ++k) sum += s_score[s_ids[k]];
        float f = route_scale / (sum + 1e-20f);
        uint32_t* r = route + static_cast<size_t>(row) * row_stride_words;
        for (uint32_t k = 0u; k < 8u; ++k) {
            r[k] = s_ids[k];
            r[8u + k] = __float_as_uint(s_score[s_ids[k]] * f);
        }
    }
}

// ---------------------------------------------------------------------------
// Loop controller prep for M rows: out[m] = [unit_rms(cur[m]), unit_rms(msg[m])]
//    out: [M][2H]; Grid: (2, M), Block: 256
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_loop_control_prep_rows_f32(
    float* __restrict__ out, const float* __restrict__ cur, const float* __restrict__ msg,
    uint32_t hidden, float eps
) {
    uint32_t row = blockIdx.y;
    const float* src = ((blockIdx.x == 0u) ? cur : msg) + static_cast<size_t>(row) * hidden;
    float* dst = out + (static_cast<size_t>(row) * 2u + blockIdx.x) * hidden;
    float sq = 0.0f;
    for (uint32_t d = threadIdx.x; d < hidden; d += blockDim.x) { float v = src[d]; sq = fmaf(v, v, sq); }
    __shared__ float s_w[32];
    __shared__ float inv;
    float ws = wave32_sum(sq);
    if ((threadIdx.x & 31u) == 0u) s_w[threadIdx.x >> 5] = ws;
    __syncthreads();
    if (threadIdx.x == 0u) {
        float t = 0.0f;
        for (uint32_t w = 0u; w < (blockDim.x >> 5); ++w) t += s_w[w];
        inv = rsqrtf(t / static_cast<float>(hidden) + eps);
    }
    __syncthreads();
    for (uint32_t d = threadIdx.x; d < hidden; d += blockDim.x) dst[d] = src[d] * inv;
}

// ---------------------------------------------------------------------------
// Expert-grouped MoE for prefill.
//   work: [n_work][4] u32 = (slab slot, pair_start, n_pairs, unused), pairs: [P] u32 = row*8 + k
//   route: [M][16] (ids, weights); x: [M][H]; mid: [M*8][I]; tmp: [M*8][H]; out: [M][H]
//   Each work item is one expert with <= 16 (row, k) pairs; weights are read once per item.
// ---------------------------------------------------------------------------
#define ST_MOE_PAIRS 16u
#define ST_MOE_ROWS_PER_WARP 8u

__device__ __forceinline__ void moe_dot_nvfp4_multi(
    float* acc, const float* __restrict__ x, const uint32_t* __restrict__ prow, uint32_t np,
    const uint8_t* __restrict__ w_row, const uint8_t* __restrict__ s_row, uint32_t k_cols, uint32_t x_stride,
    uint32_t lane
) {
    const uint2* wv = reinterpret_cast<const uint2*>(w_row);
    for (uint32_t b = lane; b < (k_cols >> 4); b += 32u) {
        uint2 w8 = wv[b];
        float bs = decode_fp8_e4m3fn(s_row[b]);
        float wf[16];
        #pragma unroll
        for (uint32_t j = 0u; j < 8u; ++j) {
            wf[j] = decode_fp4_e2m1((w8.x >> (4u * j)) & 0xFu) * bs;
            wf[8u + j] = decode_fp4_e2m1((w8.y >> (4u * j)) & 0xFu) * bs;
        }
        for (uint32_t pi = 0u; pi < np; ++pi) {
            const float4* xv = reinterpret_cast<const float4*>(x + static_cast<size_t>(prow[pi]) * x_stride) + (b << 2);
            float4 a = xv[0], c = xv[1], d = xv[2], e = xv[3];
            float s = acc[pi];
            s = fmaf(wf[0], a.x, s); s = fmaf(wf[1], a.y, s); s = fmaf(wf[2], a.z, s); s = fmaf(wf[3], a.w, s);
            s = fmaf(wf[4], c.x, s); s = fmaf(wf[5], c.y, s); s = fmaf(wf[6], c.z, s); s = fmaf(wf[7], c.w, s);
            s = fmaf(wf[8], d.x, s); s = fmaf(wf[9], d.y, s); s = fmaf(wf[10], d.z, s); s = fmaf(wf[11], d.w, s);
            s = fmaf(wf[12], e.x, s); s = fmaf(wf[13], e.y, s); s = fmaf(wf[14], e.z, s); s = fmaf(wf[15], e.w, s);
            acc[pi] = s;
        }
    }
}

__device__ __forceinline__ void moe_dot_mxfp8_multi(
    float* acc, const float* __restrict__ x, const uint32_t* __restrict__ prow, uint32_t np,
    const uint8_t* __restrict__ w_row, const uint8_t* __restrict__ s_row, uint32_t k_cols, uint32_t x_stride,
    uint32_t lane
) {
    const uint4* wv = reinterpret_cast<const uint4*>(w_row);
    for (uint32_t b = lane; b < (k_cols >> 5); b += 32u) {
        float bs = decode_ue8m0(s_row[b]);
        uint4 wa = wv[b * 2u], wb = wv[b * 2u + 1u];
        uint32_t words[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
        float wf[32];
        #pragma unroll
        for (uint32_t j = 0u; j < 8u; ++j) {
            #pragma unroll
            for (uint32_t q = 0u; q < 4u; ++q) wf[j * 4u + q] = decode_fp8_e4m3fn((words[j] >> (8u * q)) & 0xFFu) * bs;
        }
        for (uint32_t pi = 0u; pi < np; ++pi) {
            const float4* xv = reinterpret_cast<const float4*>(x + static_cast<size_t>(prow[pi]) * x_stride) + (b << 3);
            float s = acc[pi];
            #pragma unroll
            for (uint32_t j = 0u; j < 8u; ++j) {
                float4 a = xv[j];
                s = fmaf(wf[j * 4u], a.x, s); s = fmaf(wf[j * 4u + 1u], a.y, s);
                s = fmaf(wf[j * 4u + 2u], a.z, s); s = fmaf(wf[j * 4u + 3u], a.w, s);
            }
            acc[pi] = s;
        }
    }
}

// Pass 1: mid[pair][r] = SiTU(gate_r . x[row], up_r . x[row]); w13 rows interleaved (2r gate, 2r+1 up).
//   Grid: (n_work, ceil(I / (8 warps * 8 rows))), Block: 256
extern "C" __global__ void smalltide_moe_w13_grouped(
    float* __restrict__ mid, const float* __restrict__ x, const uint8_t* __restrict__ pool,
    const uint32_t* __restrict__ work, const uint32_t* __restrict__ pairs, uint64_t slab_stride,
    uint32_t hidden, uint32_t inter, float gate_cap, float up_cap
) {
    uint32_t wi = blockIdx.x;
    uint32_t slot = work[wi * 4u], p0 = work[wi * 4u + 1u], np = work[wi * 4u + 2u];
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    __shared__ uint32_t s_rows[ST_MOE_PAIRS];
    __shared__ uint32_t s_pairs[ST_MOE_PAIRS];
    if (threadIdx.x < np) { uint32_t pr = pairs[p0 + threadIdx.x]; s_pairs[threadIdx.x] = pr; s_rows[threadIdx.x] = pr >> 3; }
    __syncthreads();
    const uint8_t* slab = pool + static_cast<size_t>(slot) * slab_stride;
    const ExpertSlabHeaderGpu* hdr = reinterpret_cast<const ExpertSlabHeaderGpu*>(slab);
    const uint8_t* w13 = slab + hdr->w13_weight_offset;
    const uint8_t* s13 = slab + hdr->w13_scale_offset;
    uint32_t r0 = (blockIdx.y * (blockDim.x >> 5) + warp) * ST_MOE_ROWS_PER_WARP;
    for (uint32_t rr = 0u; rr < ST_MOE_ROWS_PER_WARP; ++rr) {
        uint32_t r = r0 + rr;
        if (r >= inter) break;
        float ga[ST_MOE_PAIRS], ua[ST_MOE_PAIRS];
        for (uint32_t i = 0u; i < ST_MOE_PAIRS; ++i) { ga[i] = 0.0f; ua[i] = 0.0f; }
        if (hdr->dtype_tag == 0u) {
            size_t ws = hidden >> 1, ss = hidden >> 4;
            moe_dot_nvfp4_multi(ga, x, s_rows, np, w13 + (2u * r) * ws, s13 + (2u * r) * ss, hidden, hidden, lane);
            moe_dot_nvfp4_multi(ua, x, s_rows, np, w13 + (2u * r + 1u) * ws, s13 + (2u * r + 1u) * ss, hidden, hidden, lane);
        } else {
            size_t ws = hidden, ss = hidden >> 5;
            moe_dot_mxfp8_multi(ga, x, s_rows, np, w13 + (2u * r) * ws, s13 + (2u * r) * ss, hidden, hidden, lane);
            moe_dot_mxfp8_multi(ua, x, s_rows, np, w13 + (2u * r + 1u) * ws, s13 + (2u * r + 1u) * ss, hidden, hidden, lane);
        }
        float osc = (hdr->dtype_tag == 0u) ? hdr->w13_outer_scale : 1.0f;
        for (uint32_t pi = 0u; pi < np; ++pi) {
            float g = wave32_sum(ga[pi]) * osc;
            float u = wave32_sum(ua[pi]) * osc;
            if (lane == 0u) mid[static_cast<size_t>(s_pairs[pi]) * inter + r] = situ_f32(g, u, gate_cap, up_cap);
        }
    }
}

// Pass 2: tmp[pair][h] = weight(pair) * (w2_h . mid[pair]).
//   Grid: (n_work, ceil(H / (8 warps * 8 rows))), Block: 256
extern "C" __global__ void smalltide_moe_w2_grouped(
    float* __restrict__ tmp, const float* __restrict__ mid, const uint8_t* __restrict__ pool,
    const uint32_t* __restrict__ work, const uint32_t* __restrict__ pairs, const uint32_t* __restrict__ route,
    uint64_t slab_stride, uint32_t hidden, uint32_t inter
) {
    uint32_t wi = blockIdx.x;
    uint32_t slot = work[wi * 4u], p0 = work[wi * 4u + 1u], np = work[wi * 4u + 2u];
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    __shared__ uint32_t s_pairs[ST_MOE_PAIRS];
    __shared__ float s_w[ST_MOE_PAIRS];
    if (threadIdx.x < np) {
        uint32_t pr = pairs[p0 + threadIdx.x];
        s_pairs[threadIdx.x] = pr;
        s_w[threadIdx.x] = __uint_as_float(route[(pr >> 3) * 16u + 8u + (pr & 7u)]);
    }
    __syncthreads();
    const uint8_t* slab = pool + static_cast<size_t>(slot) * slab_stride;
    const ExpertSlabHeaderGpu* hdr = reinterpret_cast<const ExpertSlabHeaderGpu*>(slab);
    const uint8_t* w2 = slab + hdr->w2_weight_offset;
    const uint8_t* s2 = slab + hdr->w2_scale_offset;
    uint32_t h0 = (blockIdx.y * (blockDim.x >> 5) + warp) * ST_MOE_ROWS_PER_WARP;
    for (uint32_t hh = 0u; hh < ST_MOE_ROWS_PER_WARP; ++hh) {
        uint32_t h = h0 + hh;
        if (h >= hidden) break;
        float acc[ST_MOE_PAIRS];
        for (uint32_t i = 0u; i < ST_MOE_PAIRS; ++i) acc[i] = 0.0f;
        if (hdr->dtype_tag == 0u) {
            moe_dot_nvfp4_multi(acc, mid, s_pairs, np, w2 + static_cast<size_t>(h) * (inter >> 1),
                                s2 + static_cast<size_t>(h) * (inter >> 4), inter, inter, lane);
        } else {
            moe_dot_mxfp8_multi(acc, mid, s_pairs, np, w2 + static_cast<size_t>(h) * inter,
                                s2 + static_cast<size_t>(h) * (inter >> 5), inter, inter, lane);
        }
        float osc = (hdr->dtype_tag == 0u) ? hdr->w2_outer_scale : 1.0f;
        for (uint32_t pi = 0u; pi < np; ++pi) {
            float v = wave32_sum(acc[pi]) * osc;
            if (lane == 0u) tmp[static_cast<size_t>(s_pairs[pi]) * hidden + h] = v * s_w[pi];
        }
    }
}

// Pass 3: out[m][h] = sum_k tmp[m*8+k][h] (fixed order: deterministic).
//   Grid: (ceil(H / 256), M), Block: 256
extern "C" __global__ void smalltide_moe_reduce_rows(
    float* __restrict__ out, const float* __restrict__ tmp, uint32_t hidden
) {
    uint32_t h = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;
    if (h >= hidden) return;
    const float* t = tmp + static_cast<size_t>(m) * 8u * hidden + h;
    float s = 0.0f;
    #pragma unroll
    for (uint32_t k = 0u; k < 8u; ++k) s += t[static_cast<size_t>(k) * hidden];
    out[static_cast<size_t>(m) * hidden + h] = s;
}

// ---------------------------------------------------------------------------
// WMMA (matrix core) expert-grouped MoE for prefill, gfx12 v_wmma_f32_16x16x16_f16.
// Same work / pairs layout as the scalar grouped kernels: one work item = one expert with up to
// 16 (row, k) pairs = the M = 16 rows of a WMMA tile. Each warp computes one 16-column tile of
// output weight rows over the full K. Weights are dequantized to f16 per fragment (NVFP4 values
// times their FP8 block scale are exact in f16); activations are rounded to f16; f32 accumulate.
// Fragment layout (verified): A lane l = row l%16, k (l/16)*8..+8; B lane l = col l%16, same k;
// C lane l = col l%16, rows (l/16)*8..+8.
// ---------------------------------------------------------------------------
typedef _Float16 st_half8 __attribute__((ext_vector_type(8)));
typedef float st_float8 __attribute__((ext_vector_type(8)));

// 8 consecutive f32 activations -> f16 fragment (zeros for padding rows).
__device__ __forceinline__ st_half8 wmma_load_x8(const float* __restrict__ p, bool valid) {
    st_half8 a;
    if (valid) {
        float4 u = reinterpret_cast<const float4*>(p)[0], v = reinterpret_cast<const float4*>(p)[1];
        a[0] = (_Float16)u.x; a[1] = (_Float16)u.y; a[2] = (_Float16)u.z; a[3] = (_Float16)u.w;
        a[4] = (_Float16)v.x; a[5] = (_Float16)v.y; a[6] = (_Float16)v.z; a[7] = (_Float16)v.w;
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) a[i] = (_Float16)0.0f;
    }
    return a;
}

// 8 consecutive weights (k0..k0+7, k0 % 8 == 0) of one row -> f16 fragment.
__device__ __forceinline__ st_half8 wmma_load_w8(uint32_t tag, const uint8_t* __restrict__ w_row,
                                                  const uint8_t* __restrict__ s_row, uint32_t k0) {
    st_half8 b;
    if (tag == 0u) {
        uint32_t w = *reinterpret_cast<const uint32_t*>(w_row + (k0 >> 1));
        float s = decode_fp8_e4m3fn(s_row[k0 >> 4]);
        float4 ev, od;
        e2m1x8(w, ev, od);
        b[0] = (_Float16)(ev.x * s); b[1] = (_Float16)(od.x * s); b[2] = (_Float16)(ev.y * s); b[3] = (_Float16)(od.y * s);
        b[4] = (_Float16)(ev.z * s); b[5] = (_Float16)(od.z * s); b[6] = (_Float16)(ev.w * s); b[7] = (_Float16)(od.w * s);
    } else {
        uint2 w = *reinterpret_cast<const uint2*>(w_row + k0);
        float s = decode_ue8m0(s_row[k0 >> 5]);
        float4 a = decode_fp8x4(w.x), c = decode_fp8x4(w.y);
        b[0] = (_Float16)(a.x * s); b[1] = (_Float16)(a.y * s); b[2] = (_Float16)(a.z * s); b[3] = (_Float16)(a.w * s);
        b[4] = (_Float16)(c.x * s); b[5] = (_Float16)(c.y * s); b[6] = (_Float16)(c.z * s); b[7] = (_Float16)(c.w * s);
    }
    return b;
}

// Pass 1: mid[pair][r] = SiTU(gate_r . x[row], up_r . x[row]); w13 rows interleaved (2r gate, 2r+1 up).
//   Grid: (n_work, ceil(2I / 128)), Block: 256 (8 warps x 16 w13 rows)
extern "C" __global__ void smalltide_moe_w13_grouped_wmma(
    float* __restrict__ mid, const float* __restrict__ x, const uint8_t* __restrict__ pool,
    const uint32_t* __restrict__ work, const uint32_t* __restrict__ pairs, uint64_t slab_stride,
    uint32_t hidden, uint32_t inter, float gate_cap, float up_cap
) {
    uint32_t wi = blockIdx.x;
    uint32_t slot = work[wi * 4u], p0 = work[wi * 4u + 1u], np = work[wi * 4u + 2u];
    uint32_t warp = threadIdx.x >> 5, l = threadIdx.x & 31u;
    __shared__ uint32_t s_pairs[ST_MOE_PAIRS];
    if (threadIdx.x < np) s_pairs[threadIdx.x] = pairs[p0 + threadIdx.x];
    __syncthreads();
    uint32_t n0 = (blockIdx.y * 8u + warp) * 16u; // first w13 row of this warp's tile
    if (n0 >= 2u * inter) return;
    const uint8_t* slab = pool + static_cast<size_t>(slot) * slab_stride;
    const ExpertSlabHeaderGpu* hdr = reinterpret_cast<const ExpertSlabHeaderGpu*>(slab);
    uint32_t tag = hdr->dtype_tag;
    size_t wrow = (tag == 0u) ? (hidden >> 1) : hidden;
    size_t srow = (tag == 0u) ? (hidden >> 4) : (hidden >> 5);
    const uint8_t* w = slab + hdr->w13_weight_offset + (n0 + (l & 15u)) * wrow;
    const uint8_t* sc = slab + hdr->w13_scale_offset + (n0 + (l & 15u)) * srow;
    uint32_t m = l & 15u;
    bool valid = m < np;
    const float* xr = x + static_cast<size_t>(valid ? (s_pairs[m] >> 3) : 0u) * hidden;
    uint32_t kh = (l >> 4) * 8u;
    st_float8 c = {};
    for (uint32_t k = 0u; k < hidden; k += 16u) {
        st_half8 a = wmma_load_x8(xr + k + kh, valid);
        st_half8 b = wmma_load_w8(tag, w, sc, k + kh);
        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
    }
    float osc = (tag == 0u) ? hdr->w13_outer_scale : 1.0f;
    uint32_t r = (n0 + (l & 15u)) >> 1;
    #pragma unroll
    for (uint32_t i = 0u; i < 8u; ++i) {
        float g = c[i] * osc;
        float u = __shfl_xor(c[i], 1, 32) * osc; // odd column = up of the same r
        uint32_t mm = (l >> 4) * 8u + i;
        if ((l & 1u) == 0u && mm < np) {
            mid[static_cast<size_t>(s_pairs[mm]) * inter + r] = situ_f32(g, u, gate_cap, up_cap);
        }
    }
}

// Pass 2: tmp[pair][h] = weight(pair) * (w2_h . mid[pair]).
//   Grid: (n_work, ceil(H / 128)), Block: 256 (8 warps x 16 hidden rows)
extern "C" __global__ void smalltide_moe_w2_grouped_wmma(
    float* __restrict__ tmp, const float* __restrict__ mid, const uint8_t* __restrict__ pool,
    const uint32_t* __restrict__ work, const uint32_t* __restrict__ pairs, const uint32_t* __restrict__ route,
    uint64_t slab_stride, uint32_t hidden, uint32_t inter
) {
    uint32_t wi = blockIdx.x;
    uint32_t slot = work[wi * 4u], p0 = work[wi * 4u + 1u], np = work[wi * 4u + 2u];
    uint32_t warp = threadIdx.x >> 5, l = threadIdx.x & 31u;
    __shared__ uint32_t s_pairs[ST_MOE_PAIRS];
    __shared__ float s_w[ST_MOE_PAIRS];
    if (threadIdx.x < np) {
        uint32_t pr = pairs[p0 + threadIdx.x];
        s_pairs[threadIdx.x] = pr;
        s_w[threadIdx.x] = __uint_as_float(route[(pr >> 3) * 16u + 8u + (pr & 7u)]);
    }
    __syncthreads();
    uint32_t n0 = (blockIdx.y * 8u + warp) * 16u;
    if (n0 >= hidden) return;
    const uint8_t* slab = pool + static_cast<size_t>(slot) * slab_stride;
    const ExpertSlabHeaderGpu* hdr = reinterpret_cast<const ExpertSlabHeaderGpu*>(slab);
    uint32_t tag = hdr->dtype_tag;
    size_t wrow = (tag == 0u) ? (inter >> 1) : inter;
    size_t srow = (tag == 0u) ? (inter >> 4) : (inter >> 5);
    const uint8_t* w = slab + hdr->w2_weight_offset + (n0 + (l & 15u)) * wrow;
    const uint8_t* sc = slab + hdr->w2_scale_offset + (n0 + (l & 15u)) * srow;
    uint32_t m = l & 15u;
    bool valid = m < np;
    const float* ar = mid + static_cast<size_t>(valid ? s_pairs[m] : 0u) * inter;
    uint32_t kh = (l >> 4) * 8u;
    st_float8 c = {};
    for (uint32_t k = 0u; k < inter; k += 16u) {
        st_half8 a = wmma_load_x8(ar + k + kh, valid);
        st_half8 b = wmma_load_w8(tag, w, sc, k + kh);
        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
    }
    float osc = (tag == 0u) ? hdr->w2_outer_scale : 1.0f;
    uint32_t hh = n0 + (l & 15u);
    #pragma unroll
    for (uint32_t i = 0u; i < 8u; ++i) {
        uint32_t mm = (l >> 4) * 8u + i;
        if (mm < np) tmp[static_cast<size_t>(s_pairs[mm]) * hidden + hh] = c[i] * osc * s_w[mm];
    }
}

// ---------------------------------------------------------------------------
// WMMA grouped MoE v2: f16 activations (converted once per layer), f16 `mid`, K loop unrolled by
// 4 with every load of the 4 steps issued before the 4 WMMAs (latency hiding), dtype branch
// hoisted out of the loop.
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_f32_to_f16(_Float16* __restrict__ out, const float* __restrict__ in, uint32_t n) {
    uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (i + 3u < n) {
        float4 v = reinterpret_cast<const float4*>(in)[i >> 2];
        out[i] = (_Float16)v.x; out[i + 1u] = (_Float16)v.y; out[i + 2u] = (_Float16)v.z; out[i + 3u] = (_Float16)v.w;
    } else {
        for (; i < n; ++i) out[i] = (_Float16)in[i];
    }
}

__device__ __forceinline__ st_half8 wmma_nvfp4_frag(uint32_t w, float s) {
    float4 ev, od;
    e2m1x8(w, ev, od);
    st_half8 b;
    b[0] = (_Float16)(ev.x * s); b[1] = (_Float16)(od.x * s); b[2] = (_Float16)(ev.y * s); b[3] = (_Float16)(od.y * s);
    b[4] = (_Float16)(ev.z * s); b[5] = (_Float16)(od.z * s); b[6] = (_Float16)(ev.w * s); b[7] = (_Float16)(od.w * s);
    return b;
}

__device__ __forceinline__ st_half8 wmma_mxfp8_frag(uint2 w, float s) {
    float4 a = decode_fp8x4(w.x), c = decode_fp8x4(w.y);
    st_half8 b;
    b[0] = (_Float16)(a.x * s); b[1] = (_Float16)(a.y * s); b[2] = (_Float16)(a.z * s); b[3] = (_Float16)(a.w * s);
    b[4] = (_Float16)(c.x * s); b[5] = (_Float16)(c.y * s); b[6] = (_Float16)(c.z * s); b[7] = (_Float16)(c.w * s);
    return b;
}

__device__ __forceinline__ st_half8 as_half8(uint4 v) {
    return __builtin_bit_cast(st_half8, v);
}

// c += A[16 x K] (f16 rows `ar`, this lane's row) x B[K x 16] (weight rows, this lane's column).
// K % 64 == 0. k is permuted consistently for A and B (a dot product is order-free): in each
// 64-wide block, half-wave h owns the 32 consecutive k at h*32, step s uses 8 of them - so a lane
// fetches its 4 steps' weights in one 16-byte (NVFP4) / two 16-byte (MXFP8) loads and its
// activations in four 16-byte loads. `kh` = half-wave index (0 or 1).
__device__ __forceinline__ st_float8 wmma_dot_k(uint32_t tag, const _Float16* __restrict__ ar, bool valid,
                                                const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc,
                                                uint32_t K, uint32_t kh, st_float8 c) {
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    uint32_t hoff = (kh != 0u) ? 32u : 0u;
    if (tag == 0u) {
        uint32_t k = 0u;
        // Two 64-k blocks per iteration: both blocks' loads are in flight before the 8 WMMAs.
        for (; k + 128u <= K; k += 128u) {
            uint32_t b0 = k + hoff, b1 = k + 64u + hoff;
            const uint4* ap0 = reinterpret_cast<const uint4*>(ar + b0);
            const uint4* ap1 = reinterpret_cast<const uint4*>(ar + b1);
            uint4 wv0 = *reinterpret_cast<const uint4*>(w + (b0 >> 1));
            uint4 wv1 = *reinterpret_cast<const uint4*>(w + (b1 >> 1));
            uint8_t q0 = sc[b0 >> 4], q1 = sc[(b0 >> 4) + 1u], q2 = sc[b1 >> 4], q3 = sc[(b1 >> 4) + 1u];
            uint4 a00 = valid ? ap0[0] : zero, a01 = valid ? ap0[1] : zero, a02 = valid ? ap0[2] : zero, a03 = valid ? ap0[3] : zero;
            uint4 a10 = valid ? ap1[0] : zero, a11 = valid ? ap1[1] : zero, a12 = valid ? ap1[2] : zero, a13 = valid ? ap1[3] : zero;
            float s0 = decode_fp8_e4m3fn(q0), s1 = decode_fp8_e4m3fn(q1), s2 = decode_fp8_e4m3fn(q2), s3 = decode_fp8_e4m3fn(q3);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a00), wmma_nvfp4_frag(wv0.x, s0), c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a01), wmma_nvfp4_frag(wv0.y, s0), c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a02), wmma_nvfp4_frag(wv0.z, s1), c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a03), wmma_nvfp4_frag(wv0.w, s1), c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a10), wmma_nvfp4_frag(wv1.x, s2), c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a11), wmma_nvfp4_frag(wv1.y, s2), c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a12), wmma_nvfp4_frag(wv1.z, s3), c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a13), wmma_nvfp4_frag(wv1.w, s3), c);
        }
        for (; k < K; k += 64u) {
            uint32_t base = k + hoff;
            const uint4* ap = reinterpret_cast<const uint4*>(ar + base);
            uint4 av0 = valid ? ap[0] : zero, av1 = valid ? ap[1] : zero, av2 = valid ? ap[2] : zero, av3 = valid ? ap[3] : zero;
            uint4 wv = *reinterpret_cast<const uint4*>(w + (base >> 1));
            float s0 = decode_fp8_e4m3fn(sc[base >> 4]), s1 = decode_fp8_e4m3fn(sc[(base >> 4) + 1u]);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(av0), wmma_nvfp4_frag(wv.x, s0), c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(av1), wmma_nvfp4_frag(wv.y, s0), c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(av2), wmma_nvfp4_frag(wv.z, s1), c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(av3), wmma_nvfp4_frag(wv.w, s1), c);
        }
    } else {
        uint32_t k = 0u;
        for (; k + 128u <= K; k += 128u) {
            uint32_t b0 = k + hoff, b1 = k + 64u + hoff;
            const uint4* ap0 = reinterpret_cast<const uint4*>(ar + b0);
            const uint4* ap1 = reinterpret_cast<const uint4*>(ar + b1);
            const uint4* wp0 = reinterpret_cast<const uint4*>(w + b0);
            const uint4* wp1 = reinterpret_cast<const uint4*>(w + b1);
            uint4 wa0 = wp0[0], wb0 = wp0[1], wa1 = wp1[0], wb1 = wp1[1];
            uint8_t q0 = sc[b0 >> 5], q1 = sc[b1 >> 5];
            uint4 a00 = valid ? ap0[0] : zero, a01 = valid ? ap0[1] : zero, a02 = valid ? ap0[2] : zero, a03 = valid ? ap0[3] : zero;
            uint4 a10 = valid ? ap1[0] : zero, a11 = valid ? ap1[1] : zero, a12 = valid ? ap1[2] : zero, a13 = valid ? ap1[3] : zero;
            float s0 = decode_ue8m0(q0), s1 = decode_ue8m0(q1);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a00), wmma_mxfp8_frag(make_uint2(wa0.x, wa0.y), s0), c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a01), wmma_mxfp8_frag(make_uint2(wa0.z, wa0.w), s0), c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a02), wmma_mxfp8_frag(make_uint2(wb0.x, wb0.y), s0), c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a03), wmma_mxfp8_frag(make_uint2(wb0.z, wb0.w), s0), c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a10), wmma_mxfp8_frag(make_uint2(wa1.x, wa1.y), s1), c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a11), wmma_mxfp8_frag(make_uint2(wa1.z, wa1.w), s1), c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a12), wmma_mxfp8_frag(make_uint2(wb1.x, wb1.y), s1), c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a13), wmma_mxfp8_frag(make_uint2(wb1.z, wb1.w), s1), c);
        }
        for (; k < K; k += 64u) {
            uint32_t base = k + hoff;
            const uint4* ap = reinterpret_cast<const uint4*>(ar + base);
            uint4 av0 = valid ? ap[0] : zero, av1 = valid ? ap[1] : zero, av2 = valid ? ap[2] : zero, av3 = valid ? ap[3] : zero;
            const uint4* wp = reinterpret_cast<const uint4*>(w + base);
            uint4 wa = wp[0], wb = wp[1];
            float s0 = decode_ue8m0(sc[base >> 5]);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(av0), wmma_mxfp8_frag(make_uint2(wa.x, wa.y), s0), c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(av1), wmma_mxfp8_frag(make_uint2(wa.z, wa.w), s0), c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(av2), wmma_mxfp8_frag(make_uint2(wb.x, wb.y), s0), c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(av3), wmma_mxfp8_frag(make_uint2(wb.z, wb.w), s0), c);
        }
    }
    return c;
}

// Pass 1: mid16[pair][r] = SiTU(gate_r . x16[row], up_r . x16[row]).  Grid: (n_work, ceil(2I / 128)), Block: 256
extern "C" __global__ void smalltide_moe_w13_grouped_wmma2(
    _Float16* __restrict__ mid16, const _Float16* __restrict__ x16, const uint8_t* __restrict__ pool,
    const uint32_t* __restrict__ work, const uint32_t* __restrict__ pairs, uint64_t slab_stride,
    uint32_t hidden, uint32_t inter, float gate_cap, float up_cap
) {
    uint32_t wi = blockIdx.x;
    uint32_t slot = work[wi * 4u], p0 = work[wi * 4u + 1u], np = work[wi * 4u + 2u];
    if (np == 0u) return; // unused work item (GPU-built plans have a fixed upper bound)
    uint32_t warp = threadIdx.x >> 5, l = threadIdx.x & 31u;
    __shared__ uint32_t s_pairs[ST_MOE_PAIRS];
    if (threadIdx.x < np) s_pairs[threadIdx.x] = pairs[p0 + threadIdx.x];
    __syncthreads();
    uint32_t n0 = (blockIdx.y * 8u + warp) * 16u;
    if (n0 >= 2u * inter) return;
    const uint8_t* slab = pool + static_cast<size_t>(slot) * slab_stride;
    const ExpertSlabHeaderGpu* hdr = reinterpret_cast<const ExpertSlabHeaderGpu*>(slab);
    uint32_t tag = hdr->dtype_tag;
    size_t wrow = (tag == 0u) ? (hidden >> 1) : hidden;
    size_t srow = (tag == 0u) ? (hidden >> 4) : (hidden >> 5);
    const uint8_t* w = slab + hdr->w13_weight_offset + (n0 + (l & 15u)) * wrow;
    const uint8_t* sc = slab + hdr->w13_scale_offset + (n0 + (l & 15u)) * srow;
    uint32_t m = l & 15u;
    bool valid = m < np;
    const _Float16* xr = x16 + static_cast<size_t>(valid ? (s_pairs[m] >> 3) : 0u) * hidden;
    st_float8 c = {};
    c = wmma_dot_k(tag, xr, valid, w, sc, hidden, l >> 4, c);
    float osc = (tag == 0u) ? hdr->w13_outer_scale : 1.0f;
    uint32_t r = (n0 + (l & 15u)) >> 1;
    #pragma unroll
    for (uint32_t i = 0u; i < 8u; ++i) {
        float g = c[i] * osc;
        float u = __shfl_xor(c[i], 1, 32) * osc;
        uint32_t mm = (l >> 4) * 8u + i;
        if ((l & 1u) == 0u && mm < np) {
            mid16[static_cast<size_t>(s_pairs[mm]) * inter + r] = (_Float16)situ_f32(g, u, gate_cap, up_cap);
        }
    }
}

// Pass 2: tmp[pair][h] = weight(pair) * (w2_h . mid16[pair]).  Grid: (n_work, ceil(H / 128)), Block: 256
extern "C" __global__ void smalltide_moe_w2_grouped_wmma2(
    float* __restrict__ tmp, const _Float16* __restrict__ mid16, const uint8_t* __restrict__ pool,
    const uint32_t* __restrict__ work, const uint32_t* __restrict__ pairs, const uint32_t* __restrict__ route,
    uint64_t slab_stride, uint32_t hidden, uint32_t inter
) {
    uint32_t wi = blockIdx.x;
    uint32_t slot = work[wi * 4u], p0 = work[wi * 4u + 1u], np = work[wi * 4u + 2u];
    if (np == 0u) return;
    uint32_t warp = threadIdx.x >> 5, l = threadIdx.x & 31u;
    __shared__ uint32_t s_pairs[ST_MOE_PAIRS];
    __shared__ float s_w[ST_MOE_PAIRS];
    if (threadIdx.x < np) {
        uint32_t pr = pairs[p0 + threadIdx.x];
        s_pairs[threadIdx.x] = pr;
        // route == nullptr: unweighted (the caller's post kernel applies weights + renorm).
        s_w[threadIdx.x] = (route != nullptr) ? __uint_as_float(route[(pr >> 3) * 16u + 8u + (pr & 7u)]) : 1.0f;
    }
    __syncthreads();
    uint32_t n0 = (blockIdx.y * 8u + warp) * 16u;
    if (n0 >= hidden) return;
    const uint8_t* slab = pool + static_cast<size_t>(slot) * slab_stride;
    const ExpertSlabHeaderGpu* hdr = reinterpret_cast<const ExpertSlabHeaderGpu*>(slab);
    uint32_t tag = hdr->dtype_tag;
    size_t wrow = (tag == 0u) ? (inter >> 1) : inter;
    size_t srow = (tag == 0u) ? (inter >> 4) : (inter >> 5);
    const uint8_t* w = slab + hdr->w2_weight_offset + (n0 + (l & 15u)) * wrow;
    const uint8_t* sc = slab + hdr->w2_scale_offset + (n0 + (l & 15u)) * srow;
    uint32_t m = l & 15u;
    bool valid = m < np;
    const _Float16* ar = mid16 + static_cast<size_t>(valid ? s_pairs[m] : 0u) * inter;
    st_float8 c = {};
    c = wmma_dot_k(tag, ar, valid, w, sc, inter, l >> 4, c);
    float osc = (tag == 0u) ? hdr->w2_outer_scale : 1.0f;
    uint32_t hh = n0 + (l & 15u);
    #pragma unroll
    for (uint32_t i = 0u; i < 8u; ++i) {
        uint32_t mm = (l >> 4) * 8u + i;
        if (mm < np) tmp[static_cast<size_t>(s_pairs[mm]) * hidden + hh] = c[i] * osc * s_w[mm];
    }
}

// ---------------------------------------------------------------------------
// WMMA flash attention for prefill (same semantics as smalltide_attn_prefill_fp8).
// Block = 4 warps on one KV head and 16 query vectors (query vector qi = row t * g + head i of
// the GQA group). Warps split the key range in 16-key tiles (tile j -> warp j % 4) and merge.
// Per tile: S = Q K^T (8 WMMA over head_dim 128; FP8 K -> f16 exact, K scale applied to S),
// masks + relative bias + online softmax, O += P V (8 WMMA; V scale folded into P, V staged
// in LDS as f16). Sinks add to the normalizer only.
//   Grid: (n_kv_heads, ceil(rows * g / 16)), Block: 128
// ---------------------------------------------------------------------------
#define AW_WARPS 4u

extern "C" __global__ void smalltide_attn_prefill_wmma(
    float* __restrict__ out, const float* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const float* __restrict__ k_scales, const uint8_t* __restrict__ v_cache, const float* __restrict__ v_scales,
    const float* __restrict__ rel_table, const float* __restrict__ sinks, uint32_t n_q_heads, uint32_t n_kv_heads,
    uint32_t max_seq, uint32_t window, uint32_t pos0, uint32_t rel_extent, float inv_sqrt_d, uint32_t rows
) {
    const float LOG2E = 1.4426950408889634f;
    uint32_t kvh = blockIdx.x;
    uint32_t g = n_q_heads / n_kv_heads;
    uint32_t qh0 = kvh * g;
    uint32_t nq = rows * g;
    uint32_t q_base = blockIdx.y * 16u;
    uint32_t warp = threadIdx.x >> 5, l = threadIdx.x & 31u;
    uint32_t nl = l & 15u, half_ = l >> 4;

    __shared__ _Float16 s_p[AW_WARPS][16][16];
    __shared__ _Float16 s_v[AW_WARPS][16][128];
    __shared__ float s_m[AW_WARPS][16], s_l[AW_WARPS][16];
    __shared__ float s_o[AW_WARPS][16][128 + 4];

    // Query vectors of this block: positions, rows and their key ranges.
    auto qinfo = [&](uint32_t m, uint32_t& t, uint32_t& qh, uint32_t& p, uint32_t& lo) {
        uint32_t qi = min(q_base + m, nq - 1u);
        t = qi / g;
        qh = qh0 + qi % g;
        p = pos0 + t;
        lo = (window != 0u && p + 1u > window) ? (p + 1u - window) : 0u;
    };
    uint32_t t_first, qh_x, p_first, lo_first, t_last, p_last, lo_last;
    qinfo(0u, t_first, qh_x, p_first, lo_first);
    qinfo(15u, t_last, qh_x, p_last, lo_last);
    // Key tiles are aligned to absolute multiples of 16 and owned by warp (tile index % 4), so a
    // query's result does not depend on which other queries share its block (prefill segmentation
    // invariance: prefix-cache restores must reproduce a cold prefill bit for bit).
    uint32_t key_lo = lo_first & ~15u;     // smallest window start (positions grow with m)
    uint32_t key_hi = p_last;              // inclusive
    uint32_t n_tiles = (key_hi - key_lo) / 16u + 1u;
    uint32_t tile0 = key_lo / 16u;

    // A fragment of Q (row m = l % 16, dims half*8 + 16c .. +8), pre-scaled to the log2 domain.
    uint32_t tA, qhA, pA, loA;
    qinfo(nl, tA, qhA, pA, loA);
    bool validA = q_base + nl < nq;
    const float* qrow = q + (static_cast<size_t>(tA) * n_q_heads + qhA) * 128u;
    float qs = inv_sqrt_d * LOG2E;
    st_half8 qa[8];
    #pragma unroll
    for (uint32_t c = 0u; c < 8u; ++c) {
        const float4* src = reinterpret_cast<const float4*>(qrow + c * 16u + half_ * 8u);
        float4 u = validA ? src[0] : make_float4(0, 0, 0, 0), v = validA ? src[1] : make_float4(0, 0, 0, 0);
        qa[c][0] = (_Float16)(u.x * qs); qa[c][1] = (_Float16)(u.y * qs); qa[c][2] = (_Float16)(u.z * qs); qa[c][3] = (_Float16)(u.w * qs);
        qa[c][4] = (_Float16)(v.x * qs); qa[c][5] = (_Float16)(v.y * qs); qa[c][6] = (_Float16)(v.z * qs); qa[c][7] = (_Float16)(v.w * qs);
    }
    // Rows this lane holds in the C layout: m = half * 8 + i.
    uint32_t tC[8], qhC[8], pC[8], loC[8];
    #pragma unroll
    for (uint32_t i = 0u; i < 8u; ++i) qinfo(half_ * 8u + i, tC[i], qhC[i], pC[i], loC[i]);

    const uint8_t* kb = k_cache + static_cast<size_t>(kvh) * max_seq * 128u;
    const uint8_t* vb = v_cache + static_cast<size_t>(kvh) * max_seq * 128u;
    const float* ksb = k_scales + static_cast<size_t>(kvh) * max_seq;
    const float* vsb = v_scales + static_cast<size_t>(kvh) * max_seq;

    float m_run[8], l_run[8];
    st_float8 o[8];
    #pragma unroll
    for (uint32_t i = 0u; i < 8u; ++i) { m_run[i] = -1e30f; l_run[i] = 0.0f; }
    #pragma unroll
    for (uint32_t c = 0u; c < 8u; ++c) o[c] = (st_float8){};

    uint32_t first = (warp + AW_WARPS - (tile0 % AW_WARPS)) % AW_WARPS;
    for (uint32_t tile = first; tile < n_tiles; tile += AW_WARPS) {
        uint32_t k0 = key_lo + tile * 16u;
        // This lane's key (column n = l % 16) for S and its cache slot.
        uint32_t kp = k0 + nl;
        bool kin = kp <= key_hi;
        uint32_t slot = (window != 0u) ? (kp % max_seq) : kp;
        // S = Q K^T over 8 dim chunks.
        st_float8 sacc = {};
        #pragma unroll
        for (uint32_t c = 0u; c < 8u; ++c) {
            uint2 kw = kin ? *reinterpret_cast<const uint2*>(kb + static_cast<size_t>(slot) * 128u + c * 16u + half_ * 8u) : make_uint2(0u, 0u);
            float4 a = decode_fp8x4(kw.x), b = decode_fp8x4(kw.y);
            st_half8 kf;
            kf[0] = (_Float16)a.x; kf[1] = (_Float16)a.y; kf[2] = (_Float16)a.z; kf[3] = (_Float16)a.w;
            kf[4] = (_Float16)b.x; kf[5] = (_Float16)b.y; kf[6] = (_Float16)b.z; kf[7] = (_Float16)b.w;
            sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(qa[c], kf, sacc);
        }
        float kscale = kin ? ksb[slot] : 0.0f;
        float vscale = kin ? vsb[slot] : 0.0f;
        // Logits, masks, relative bias; online softmax (row max over the 16 key lanes).
        float pr[8];
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; ++i) {
            bool ok = kin && kp <= pC[i] && kp >= loC[i] && (q_base + half_ * 8u + i) < nq;
            float sc = -1e30f;
            if (ok) {
                uint32_t dist = pC[i] - kp;
                float rb = (rel_table != nullptr && dist < rel_extent)
                    ? rel_table[(static_cast<size_t>(tC[i]) * n_q_heads + qhC[i]) * rel_extent + dist] * LOG2E : 0.0f;
                sc = sacc[i] * kscale + rb;
            }
            float mx = sc;
            mx = fmaxf(mx, __shfl_xor(mx, 8, 32));
            mx = fmaxf(mx, __shfl_xor(mx, 4, 32));
            mx = fmaxf(mx, __shfl_xor(mx, 2, 32));
            mx = fmaxf(mx, __shfl_xor(mx, 1, 32));
            float mn = fmaxf(m_run[i], mx);
            float al = exp2f(m_run[i] - mn);
            float pv = ok ? exp2f(sc - mn) : 0.0f;
            l_run[i] = l_run[i] * al + pv; // per-lane partial (reduced at the end)
            m_run[i] = mn;
            #pragma unroll
            for (uint32_t c = 0u; c < 8u; ++c) o[c][i] *= al;
            pr[i] = pv * vscale;
        }
        // P (C layout: lane = key column) -> LDS -> A layout (lane = query row).
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; ++i) s_p[warp][half_ * 8u + i][nl] = (_Float16)pr[i];
        // V tile [16 keys][128] FP8 -> f16 (raw; the V scale is in P).
        #pragma unroll
        for (uint32_t r = 0u; r < 16u; ++r) {
            uint32_t kr = k0 + r;
            uint32_t vslot = (window != 0u) ? (kr % max_seq) : kr;
            uint32_t vw = (kr <= key_hi) ? reinterpret_cast<const uint32_t*>(vb + static_cast<size_t>(vslot) * 128u)[l] : 0u;
            float4 vv = decode_fp8x4(vw);
            s_v[warp][r][l * 4u + 0u] = (_Float16)vv.x; s_v[warp][r][l * 4u + 1u] = (_Float16)vv.y;
            s_v[warp][r][l * 4u + 2u] = (_Float16)vv.z; s_v[warp][r][l * 4u + 3u] = (_Float16)vv.w;
        }
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        st_half8 pa;
        #pragma unroll
        for (uint32_t j = 0u; j < 8u; ++j) pa[j] = s_p[warp][nl][half_ * 8u + j];
        #pragma unroll
        for (uint32_t c = 0u; c < 8u; ++c) {
            st_half8 vf;
            #pragma unroll
            for (uint32_t j = 0u; j < 8u; ++j) vf[j] = s_v[warp][half_ * 8u + j][c * 16u + nl];
            o[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(pa, vf, o[c]);
        }
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
    }

    // Row sums over the 16 key lanes; publish (m, l, O) per warp.
    #pragma unroll
    for (uint32_t i = 0u; i < 8u; ++i) {
        float s = l_run[i];
        s += __shfl_xor(s, 8, 32); s += __shfl_xor(s, 4, 32); s += __shfl_xor(s, 2, 32); s += __shfl_xor(s, 1, 32);
        if (nl == 0u) { s_m[warp][half_ * 8u + i] = m_run[i]; s_l[warp][half_ * 8u + i] = s; }
        #pragma unroll
        for (uint32_t c = 0u; c < 8u; ++c) s_o[warp][half_ * 8u + i][c * 16u + nl] = o[c][i];
    }
    __syncthreads();
    // Merge the warps + the sink; 128 threads = 16 rows x 8 threads x 16 dims.
    uint32_t mrow = threadIdx.x >> 3, dpart = (threadIdx.x & 7u) * 16u;
    if (q_base + mrow < nq) {
        uint32_t t, qh, p, lo;
        qinfo(mrow, t, qh, p, lo);
        float sink = (sinks != nullptr) ? sinks[static_cast<size_t>(t) * n_q_heads + qh] * LOG2E : -1e30f;
        float mb = sink;
        for (uint32_t w = 0u; w < AW_WARPS; ++w) if (s_l[w][mrow] > 0.0f) mb = fmaxf(mb, s_m[w][mrow]);
        float lb = (sinks != nullptr) ? exp2f(sink - mb) : 0.0f;
        float f[AW_WARPS];
        for (uint32_t w = 0u; w < AW_WARPS; ++w) {
            f[w] = (s_l[w][mrow] > 0.0f) ? exp2f(s_m[w][mrow] - mb) : 0.0f;
            lb += s_l[w][mrow] * f[w];
        }
        float inv = (lb > 1e-30f) ? 1.0f / lb : 0.0f;
        float* dst = out + (static_cast<size_t>(t) * n_q_heads + qh) * 128u + dpart;
        for (uint32_t d = 0u; d < 16u; ++d) {
            float acc = 0.0f;
            for (uint32_t w = 0u; w < AW_WARPS; ++w) acc = fmaf(s_o[w][mrow][dpart + d], f[w], acc);
            dst[d] = acc * inv;
        }
    }
}

// ---------------------------------------------------------------------------
// WMMA FP8 GEMM for prefill rows: Y[m][n] = row_scale[n] * sum_k X[m][k] * W_fp8[n][k].
// Block 256 = 8 warps; block tile 128 rows x 64 cols; warp w = rows w*16..+16 x 4 N tiles (one A
// fragment feeds 4 WMMAs). FP8 -> f16 is exact; X rounded to f16; f32 accumulate; 2-step
// prefetch. K % 32 == 0.   Grid: (ceil(N / 64), ceil(M / 128)), Block: 256
// ---------------------------------------------------------------------------
__device__ __forceinline__ st_half8 fp8x8_to_half8(uint2 w) {
    float4 a = decode_fp8x4(w.x), b = decode_fp8x4(w.y);
    st_half8 h;
    h[0] = (_Float16)a.x; h[1] = (_Float16)a.y; h[2] = (_Float16)a.z; h[3] = (_Float16)a.w;
    h[4] = (_Float16)b.x; h[5] = (_Float16)b.y; h[6] = (_Float16)b.z; h[7] = (_Float16)b.w;
    return h;
}

extern "C" __global__ void smalltide_gemm_fp8_rows_wmma(
    float* __restrict__ y, const float* __restrict__ x, const uint8_t* __restrict__ w_fp8,
    const float* __restrict__ row_scale, uint32_t m_rows, uint32_t n_rows, uint32_t k_cols
) {
    uint32_t warp = threadIdx.x >> 5, l = threadIdx.x & 31u;
    uint32_t m0 = blockIdx.y * 128u + warp * 16u;
    if (m0 >= m_rows) return;
    uint32_t n0 = blockIdx.x * 64u;
    uint32_t am = m0 + (l & 15u);
    bool a_ok = am < m_rows;
    const float* xr = x + static_cast<size_t>(a_ok ? am : 0u) * k_cols;
    const uint8_t* wr[4];
    bool b_ok[4];
    #pragma unroll
    for (uint32_t j = 0u; j < 4u; ++j) {
        uint32_t n = n0 + 16u * j + (l & 15u);
        b_ok[j] = n < n_rows;
        wr[j] = w_fp8 + static_cast<size_t>(b_ok[j] ? n : 0u) * k_cols;
    }
    st_float8 c[4] = {};
    // k permuted consistently for A and B: in each 32-wide block half-wave h owns the 16
    // consecutive k at h*16 (step s uses 8 of them): one 16-byte weight load per N tile.
    uint32_t hoff = (l >> 4) * 16u;
    for (uint32_t k = 0u; k < k_cols; k += 32u) {
        uint32_t base = k + hoff;
        const float4* xp = reinterpret_cast<const float4*>(xr + base);
        float4 x0 = a_ok ? xp[0] : make_float4(0, 0, 0, 0), x1 = a_ok ? xp[1] : make_float4(0, 0, 0, 0);
        float4 x2 = a_ok ? xp[2] : make_float4(0, 0, 0, 0), x3 = a_ok ? xp[3] : make_float4(0, 0, 0, 0);
        uint4 wv[4];
        #pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) wv[j] = b_ok[j] ? *reinterpret_cast<const uint4*>(wr[j] + base) : make_uint4(0u, 0u, 0u, 0u);
        st_half8 a0, a1;
        a0[0] = (_Float16)x0.x; a0[1] = (_Float16)x0.y; a0[2] = (_Float16)x0.z; a0[3] = (_Float16)x0.w;
        a0[4] = (_Float16)x1.x; a0[5] = (_Float16)x1.y; a0[6] = (_Float16)x1.z; a0[7] = (_Float16)x1.w;
        a1[0] = (_Float16)x2.x; a1[1] = (_Float16)x2.y; a1[2] = (_Float16)x2.z; a1[3] = (_Float16)x2.w;
        a1[4] = (_Float16)x3.x; a1[5] = (_Float16)x3.y; a1[6] = (_Float16)x3.z; a1[7] = (_Float16)x3.w;
        #pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            c[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, fp8x8_to_half8(make_uint2(wv[j].x, wv[j].y)), c[j]);
            c[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, fp8x8_to_half8(make_uint2(wv[j].z, wv[j].w)), c[j]);
        }
    }
    #pragma unroll
    for (uint32_t j = 0u; j < 4u; ++j) {
        uint32_t n = n0 + 16u * j + (l & 15u);
        if (n >= n_rows) continue;
        float sc = (row_scale != nullptr) ? row_scale[n] : 1.0f;
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; ++i) {
            uint32_t mm = m0 + (l >> 4) * 8u + i;
            if (mm < m_rows) y[static_cast<size_t>(mm) * n_rows + n] = c[j][i] * sc;
        }
    }
}

// Same arithmetic per output element as smalltide_gemm_fp8_rows_wmma (k order and WMMA steps
// identical), for m_rows <= 16: one warp per 16-column tile.  Grid: (ceil(N / 128)), Block: 256
extern "C" __global__ void smalltide_gemm_fp8_rows_wmma_small(
    float* __restrict__ y, const float* __restrict__ x, const uint8_t* __restrict__ w_fp8,
    const float* __restrict__ row_scale, uint32_t m_rows, uint32_t n_rows, uint32_t k_cols
) {
    uint32_t warp = threadIdx.x >> 5, l = threadIdx.x & 31u;
    uint32_t n0 = (blockIdx.x * 8u + warp) * 16u;
    if (n0 >= n_rows) return;
    uint32_t am = l & 15u;
    bool a_ok = am < m_rows;
    const float* xr = x + static_cast<size_t>(a_ok ? am : 0u) * k_cols;
    uint32_t n = n0 + (l & 15u);
    bool b_ok = n < n_rows;
    const uint8_t* wr = w_fp8 + static_cast<size_t>(b_ok ? n : 0u) * k_cols;
    st_float8 c = {};
    uint32_t hoff = (l >> 4) * 16u;
    for (uint32_t k = 0u; k < k_cols; k += 32u) {
        uint32_t base = k + hoff;
        const float4* xp = reinterpret_cast<const float4*>(xr + base);
        float4 x0 = a_ok ? xp[0] : make_float4(0, 0, 0, 0), x1 = a_ok ? xp[1] : make_float4(0, 0, 0, 0);
        float4 x2 = a_ok ? xp[2] : make_float4(0, 0, 0, 0), x3 = a_ok ? xp[3] : make_float4(0, 0, 0, 0);
        uint4 wv = b_ok ? *reinterpret_cast<const uint4*>(wr + base) : make_uint4(0u, 0u, 0u, 0u);
        st_half8 a0, a1;
        a0[0] = (_Float16)x0.x; a0[1] = (_Float16)x0.y; a0[2] = (_Float16)x0.z; a0[3] = (_Float16)x0.w;
        a0[4] = (_Float16)x1.x; a0[5] = (_Float16)x1.y; a0[6] = (_Float16)x1.z; a0[7] = (_Float16)x1.w;
        a1[0] = (_Float16)x2.x; a1[1] = (_Float16)x2.y; a1[2] = (_Float16)x2.z; a1[3] = (_Float16)x2.w;
        a1[4] = (_Float16)x3.x; a1[5] = (_Float16)x3.y; a1[6] = (_Float16)x3.z; a1[7] = (_Float16)x3.w;
        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, fp8x8_to_half8(make_uint2(wv.x, wv.y)), c);
        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, fp8x8_to_half8(make_uint2(wv.z, wv.w)), c);
    }
    if (!b_ok) return;
    float sc = (row_scale != nullptr) ? row_scale[n] : 1.0f;
    #pragma unroll
    for (uint32_t i = 0u; i < 8u; ++i) {
        uint32_t mm = (l >> 4) * 8u + i;
        if (mm < m_rows) y[static_cast<size_t>(mm) * n_rows + n] = c[i] * sc;
    }
}

// ---------------------------------------------------------------------------
// Batched-decode FP8 GEMV for up to 8 rows: one warp per output row, two 16-byte weight loads in
// flight, packed FP8 decode, each decoded weight applied to every row (bandwidth-bound like the
// single-row GEMV).   Grid: (ceil(N / 4)), Block: 128
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_gemv_fp8_rows(
    float* __restrict__ y, const float* __restrict__ x, const uint8_t* __restrict__ w_fp8,
    const float* __restrict__ row_scale, uint32_t m_rows, uint32_t n_rows, uint32_t k_cols
) {
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t row = blockIdx.x * 4u + warp;
    if (row >= n_rows) return;
    const uint4* wv = reinterpret_cast<const uint4*>(w_fp8 + static_cast<size_t>(row) * k_cols);
    uint32_t k16 = k_cols >> 4;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    auto dot16 = [&](uint4 w16, uint32_t c) {
        float4 d0 = decode_fp8x4(w16.x), d1 = decode_fp8x4(w16.y), d2 = decode_fp8x4(w16.z), d3 = decode_fp8x4(w16.w);
        #pragma unroll
        for (uint32_t m = 0u; m < 8u; ++m) {
            if (m < m_rows) {
                const float4* xv = reinterpret_cast<const float4*>(x + static_cast<size_t>(m) * k_cols) + (c << 2);
                float4 x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
                float a = acc[m];
                a = fmaf(d0.x, x0.x, a); a = fmaf(d0.y, x0.y, a); a = fmaf(d0.z, x0.z, a); a = fmaf(d0.w, x0.w, a);
                a = fmaf(d1.x, x1.x, a); a = fmaf(d1.y, x1.y, a); a = fmaf(d1.z, x1.z, a); a = fmaf(d1.w, x1.w, a);
                a = fmaf(d2.x, x2.x, a); a = fmaf(d2.y, x2.y, a); a = fmaf(d2.z, x2.z, a); a = fmaf(d2.w, x2.w, a);
                a = fmaf(d3.x, x3.x, a); a = fmaf(d3.y, x3.y, a); a = fmaf(d3.z, x3.z, a); a = fmaf(d3.w, x3.w, a);
                acc[m] = a;
            }
        }
    };
    uint32_t c = lane;
    for (; c + 32u < k16; c += 64u) {
        uint4 wa = wv[c];
        uint4 wb = wv[c + 32u];
        dot16(wa, c);
        dot16(wb, c + 32u);
    }
    if (c < k16) dot16(wv[c], c);
    float sc = (row_scale != nullptr) ? row_scale[row] : 1.0f;
    #pragma unroll
    for (uint32_t m = 0u; m < 8u; ++m) {
        if (m < m_rows) {
            float sum = wave32_sum(acc[m]);
            if (lane == 0u) y[static_cast<size_t>(m) * n_rows + row] = sum * sc;
        }
    }
}

// ---------------------------------------------------------------------------
// Speculative-verify rewind: every short-conv state takes the 3 history rows that follow the
// first `keep` inputs (state after `keep` of the verified rows).
//   entries: [n][4] u64 = (hist ptr, state ptr, dim, unused).  Grid: (n, ceil(maxdim / 256)), Block: 256
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_sconv_rewind(const uint64_t* __restrict__ entries, uint32_t keep) {
    const uint64_t* e = entries + static_cast<size_t>(blockIdx.x) * 4u;
    const float* hist = reinterpret_cast<const float*>(e[0]);
    float* state = reinterpret_cast<float*>(e[1]);
    uint32_t dim = static_cast<uint32_t>(e[2]);
    uint32_t c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= dim) return;
    #pragma unroll
    for (uint32_t i = 0u; i < 3u; ++i) state[static_cast<size_t>(i) * dim + c] = hist[static_cast<size_t>(keep + i) * dim + c];
}

// Greedy argmax of `rows` rows of `n` f32 each: out[r] = index (lowest on ties).  Grid: (rows), Block: 1024
extern "C" __global__ void smalltide_argmax_rows(uint32_t* __restrict__ out, const float* __restrict__ x, uint32_t n) {
    const float* row = x + static_cast<size_t>(blockIdx.x) * n;
    float bv = -INFINITY;
    uint32_t bi = 0xFFFFFFFFu;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = row[i];
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    __shared__ float sv[32];
    __shared__ uint32_t si[32];
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor(bv, o, 32);
        uint32_t oi = __shfl_xor(bi, o, 32);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    uint32_t w = threadIdx.x >> 5, l = threadIdx.x & 31u;
    if (l == 0u) { sv[w] = bv; si[w] = bi; }
    __syncthreads();
    if (w == 0u) {
        bv = (l < (blockDim.x >> 5)) ? sv[l] : -INFINITY;
        bi = (l < (blockDim.x >> 5)) ? si[l] : 0xFFFFFFFFu;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor(bv, o, 32);
            uint32_t oi = __shfl_xor(bi, o, 32);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (l == 0u) out[blockIdx.x] = bi;
    }
}

// ---------------------------------------------------------------------------
// GPU-built expert grouping for <= 8 decode / verify rows (no host sync): one work item per
// distinct published expert of the rows' top-8, its (row, k) pairs listed; 64 work items always
// (unused: n_pairs = 0, skipped by the WMMA kernels).  Grid: 1, Block: 32
//   work: [64][4] u32 (slot, pair_start, n_pairs, 0), pairs: [64] u32
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_moe_group_rows(
    uint32_t* __restrict__ work, uint32_t* __restrict__ pairs, const uint32_t* __restrict__ route,
    const uint32_t* __restrict__ slot_table, uint32_t rows
) {
    if (threadIdx.x != 0u) return;
    uint32_t np_total = rows * 8u;
    uint32_t used = 0u, nw = 0u;
    uint64_t done = 0ull;
    for (uint32_t p = 0u; p < np_total; ++p) {
        if (done & (1ull << p)) continue;
        uint32_t e = route[(p >> 3) * 16u + (p & 7u)];
        uint32_t slot = slot_table[e];
        uint32_t start = used;
        for (uint32_t q = p; q < np_total; ++q) {
            if (!(done & (1ull << q)) && route[(q >> 3) * 16u + (q & 7u)] == e) {
                done |= (1ull << q);
                if (slot != 0xFFFFFFFFu) pairs[used++] = q;
            }
        }
        if (slot != 0xFFFFFFFFu) {
            work[nw * 4u] = slot; work[nw * 4u + 1u] = start; work[nw * 4u + 2u] = used - start; work[nw * 4u + 3u] = 0u;
            ++nw;
        }
    }
    for (; nw < 64u; ++nw) { work[nw * 4u] = 0u; work[nw * 4u + 1u] = 0u; work[nw * 4u + 2u] = 0u; work[nw * 4u + 3u] = 0u; }
}

// out[r][h] = sum_k w_rk * renorm_r * tmp[r*8+k][h] over published experts (no conv / residual).
//   Grid: (ceil(H / 256), rows), Block: 256
extern "C" __global__ void smalltide_moe_post_rows(
    float* __restrict__ out, const float* __restrict__ tmp, const uint32_t* __restrict__ slot_table,
    const uint32_t* __restrict__ route, uint32_t hidden
) {
    uint32_t h = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (h >= hidden) return;
    const uint32_t* ids = route + r * 16u;
    const float* wts = reinterpret_cast<const float*>(route + r * 16u + 8u);
    float tot = 0.0f, ex = 0.0f;
    for (uint32_t k = 0u; k < 8u; ++k) {
        tot += wts[k];
        if (slot_table[ids[k]] != 0xFFFFFFFFu) ex += wts[k];
    }
    float renorm = (ex > 1e-8f) ? (tot / ex) : 0.0f;
    float acc = 0.0f;
    for (uint32_t k = 0u; k < 8u; ++k) {
        if (slot_table[ids[k]] != 0xFFFFFFFFu) acc = fmaf(wts[k] * renorm, tmp[(static_cast<size_t>(r) * 8u + k) * hidden + h], acc);
    }
    out[static_cast<size_t>(r) * hidden + h] = acc;
}

// ---------------------------------------------------------------------------
// Multi-row FP8 GEMV v2 (<= 8 rows): the activations of a 512-wide K tile are staged in LDS
// once per block; 16 warps (one output row each) stream 16-byte weight chunks against it.
//   Grid: (ceil(N / 16)), Block: 512
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_gemv_fp8_rows2(
    float* __restrict__ y, const float* __restrict__ x, const uint8_t* __restrict__ w_fp8,
    const float* __restrict__ row_scale, uint32_t m_rows, uint32_t n_rows, uint32_t k_cols
) {
    __shared__ float4 xs[8][128]; // 8 rows x 512 floats
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t row = blockIdx.x * 16u + warp;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const uint8_t* wr = w_fp8 + static_cast<size_t>(row < n_rows ? row : 0u) * k_cols;
    for (uint32_t k0 = 0u; k0 < k_cols; k0 += 512u) {
        uint32_t tw = min(512u, k_cols - k0) >> 2; // float4s per row in this tile
        for (uint32_t i = threadIdx.x; i < m_rows * 128u; i += blockDim.x) {
            uint32_t r = i >> 7, c = i & 127u;
            xs[r][c] = (c < tw) ? reinterpret_cast<const float4*>(x + static_cast<size_t>(r) * k_cols + k0)[c] : make_float4(0, 0, 0, 0);
        }
        __syncthreads();
        uint32_t kk = k0 + lane * 16u;
        if (row < n_rows && kk < k_cols) {
            uint4 w16 = *reinterpret_cast<const uint4*>(wr + kk);
            float4 d0 = decode_fp8x4(w16.x), d1 = decode_fp8x4(w16.y), d2 = decode_fp8x4(w16.z), d3 = decode_fp8x4(w16.w);
            #pragma unroll
            for (uint32_t m = 0u; m < 8u; ++m) {
                if (m < m_rows) {
                    float4 x0 = xs[m][lane * 4u], x1 = xs[m][lane * 4u + 1u], x2 = xs[m][lane * 4u + 2u], x3 = xs[m][lane * 4u + 3u];
                    float a = acc[m];
                    a = fmaf(d0.x, x0.x, a); a = fmaf(d0.y, x0.y, a); a = fmaf(d0.z, x0.z, a); a = fmaf(d0.w, x0.w, a);
                    a = fmaf(d1.x, x1.x, a); a = fmaf(d1.y, x1.y, a); a = fmaf(d1.z, x1.z, a); a = fmaf(d1.w, x1.w, a);
                    a = fmaf(d2.x, x2.x, a); a = fmaf(d2.y, x2.y, a); a = fmaf(d2.z, x2.z, a); a = fmaf(d2.w, x2.w, a);
                    a = fmaf(d3.x, x3.x, a); a = fmaf(d3.y, x3.y, a); a = fmaf(d3.z, x3.z, a); a = fmaf(d3.w, x3.w, a);
                    acc[m] = a;
                }
            }
        }
        __syncthreads();
    }
    if (row >= n_rows) return;
    float sc = (row_scale != nullptr) ? row_scale[row] : 1.0f;
    #pragma unroll
    for (uint32_t m = 0u; m < 8u; ++m) {
        if (m < m_rows) {
            float sum = wave32_sum(acc[m]);
            if (lane == 0u) y[static_cast<size_t>(m) * n_rows + row] = sum * sc;
        }
    }
}

// Parallel expert grouping (replaces the single-thread plan): Grid 1, Block 64.
extern "C" __global__ void smalltide_moe_group_rows2(
    uint32_t* __restrict__ work, uint32_t* __restrict__ pairs, const uint32_t* __restrict__ route,
    const uint32_t* __restrict__ slot_table, uint32_t rows
) {
    __shared__ uint32_t s_e[64], s_cnt[64], s_wi[64], s_off[64];
    __shared__ uint32_t s_nw;
    uint32_t p = threadIdx.x, np_total = rows * 8u;
    bool valid = p < np_total;
    uint32_t e = valid ? route[(p >> 3) * 16u + (p & 7u)] : 0xFFFFFFFFu;
    s_e[p] = e;
    __syncthreads();
    uint32_t slot = valid ? slot_table[e] : 0xFFFFFFFFu;
    bool first = valid && slot != 0xFFFFFFFFu;
    for (uint32_t q = 0u; q < p && first; ++q) if (s_e[q] == e) first = false;
    uint32_t cnt = 0u;
    if (first) for (uint32_t q = p; q < np_total; ++q) cnt += (s_e[q] == e) ? 1u : 0u;
    s_cnt[p] = first ? cnt : 0u;
    s_wi[p] = first ? 1u : 0u;
    __syncthreads();
    if (p == 0u) {
        uint32_t wsum = 0u, osum = 0u;
        for (uint32_t q = 0u; q < 64u; ++q) {
            uint32_t f = s_wi[q], c = s_cnt[q];
            s_wi[q] = wsum; s_off[q] = osum;
            wsum += f; osum += c;
        }
        s_nw = wsum;
    }
    __syncthreads();
    if (first) {
        uint32_t wi = s_wi[p], off = s_off[p];
        work[wi * 4u] = slot; work[wi * 4u + 1u] = off; work[wi * 4u + 2u] = cnt; work[wi * 4u + 3u] = 0u;
        uint32_t j = 0u;
        for (uint32_t q = p; q < np_total; ++q) if (s_e[q] == e) pairs[off + j++] = q;
    }
    if (p >= s_nw) { work[p * 4u] = 0u; work[p * 4u + 1u] = 0u; work[p * 4u + 2u] = 0u; work[p * 4u + 3u] = 0u; }
}

// ===========================================================================
// DSpark drafter (Inkling-2 mtp.safetensors) kernels.
// ===========================================================================

// out[r][d] = RMSNorm(sum_t a_t * taps[t][r][d]) * w[d]   (taps: [4][tap_stride][D])
//   Grid: rows, Block: 1024 (D <= 3072)
extern "C" __global__ void smalltide_dspark_mix_rms(
    float* __restrict__ out, const float* __restrict__ taps, uint32_t tap_stride, float a0, float a1, float a2, float a3,
    const float* __restrict__ w, uint32_t D, float eps
) {
    __shared__ float red[32];
    uint32_t r = blockIdx.x, t = threadIdx.x;
    float v[3];
    float sq = 0.0f;
    #pragma unroll
    for (uint32_t e = 0u; e < 3u; ++e) {
        uint32_t d = t + e * blockDim.x;
        float m = 0.0f;
        if (d < D) {
            size_t base = static_cast<size_t>(r) * D + d, ts = static_cast<size_t>(tap_stride) * D;
            m = taps[base] * a0 + taps[ts + base] * a1 + taps[2u * ts + base] * a2 + taps[3u * ts + base] * a3;
        }
        v[e] = m;
        sq = fmaf(m, m, sq);
    }
    sq = wave32_sum(sq);
    if ((t & 31u) == 0u) red[t >> 5] = sq;
    __syncthreads();
    if (t < 32u) {
        float s = (t < (blockDim.x >> 5)) ? red[t] : 0.0f;
        s = wave32_sum(s);
        if (t == 0u) red[0] = s;
    }
    __syncthreads();
    float inv = rsqrtf(red[0] / static_cast<float>(D) + eps);
    #pragma unroll
    for (uint32_t e = 0u; e < 3u; ++e) {
        uint32_t d = t + e * blockDim.x;
        if (d < D) out[static_cast<size_t>(r) * D + d] = v[e] * inv * w[d];
    }
}

// Per-head RMSNorm (weight w[128]) + RoPE (neox halves, theta), in place: x[rows][nh][128] at
// positions pos0 + row.   Grid: (nh, rows), Block: 64
extern "C" __global__ void smalltide_dspark_head_norm_rope(
    float* __restrict__ x, const float* __restrict__ w, uint32_t nh, uint32_t pos0, float log2_theta, float eps
) {
    __shared__ float red[2];
    uint32_t h = blockIdx.x, r = blockIdx.y, i = threadIdx.x;
    float* p = x + (static_cast<size_t>(r) * nh + h) * 128u;
    float a = p[i], b = p[i + 64u];
    float sq = wave32_sum(a * a + b * b);
    if ((i & 31u) == 0u) red[i >> 5] = sq;
    __syncthreads();
    float inv = rsqrtf((red[0] + red[1]) / 128.0f + eps);
    a = a * inv * w[i];
    b = b * inv * w[i + 64u];
    float inv_freq = exp2f(-log2_theta * (2.0f * static_cast<float>(i)) / 128.0f);
    float ang = static_cast<float>(pos0 + r) * inv_freq;
    float sn, cs;
    sincosf(ang, &sn, &cs);
    p[i] = a * cs - b * sn;
    p[i + 64u] = b * cs + a * sn;
}

// Context K/V store (f16): rows of k / v [rows][dim] at positions pos0.. (ring: slot = pos % ring).
//   Grid: rows, Block: 256
extern "C" __global__ void smalltide_dspark_ctx_store(
    _Float16* __restrict__ kc, _Float16* __restrict__ vc, const float* __restrict__ k, const float* __restrict__ v,
    uint32_t pos0, uint32_t ring, uint32_t dim
) {
    uint32_t r = blockIdx.x;
    uint32_t pos = pos0 + r;
    uint32_t slot = (ring != 0u) ? (pos % ring) : pos;
    for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
        kc[static_cast<size_t>(slot) * dim + d] = (_Float16)k[static_cast<size_t>(r) * dim + d];
        vc[static_cast<size_t>(slot) * dim + d] = (_Float16)v[static_cast<size_t>(r) * dim + d];
    }
}

// Drafter attention: query row r, head h over [context positions lo..P) | the nblk block rows
// (non-causal). q [rows][nh*128], kb / vb [nblk][nkv*128], kc / vc f16 [slots][nkv*128].
//   Grid: (nh, rows), Block: 256 (8 warps split the keys)
extern "C" __global__ void smalltide_dspark_attn(
    float* __restrict__ out, const float* __restrict__ q, const float* __restrict__ kb, const float* __restrict__ vb,
    const _Float16* __restrict__ kc, const _Float16* __restrict__ vc, uint32_t P, uint32_t lo, uint32_t ring,
    uint32_t nh, uint32_t nkv, uint32_t nblk, float scale
) {
    uint32_t h = blockIdx.x, r = blockIdx.y;
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t kvh = h / (nh / nkv);
    uint32_t kvd = nkv * 128u;
    float4 qv = reinterpret_cast<const float4*>(q + (static_cast<size_t>(r) * nh + h) * 128u)[lane];
    uint32_t nctx = (P > lo) ? (P - lo) : 0u;
    uint32_t total = nctx + nblk;
    float m = -1e30f, l = 0.0f;
    float4 o = make_float4(0, 0, 0, 0);
    for (uint32_t j = warp; j < total; j += 8u) {
        float4 kv4, vv4;
        if (j < nctx) {
            uint32_t pos = lo + j;
            uint32_t slot = (ring != 0u) ? (pos % ring) : pos;
            const _Float16* kp = kc + static_cast<size_t>(slot) * kvd + kvh * 128u + lane * 4u;
            const _Float16* vp = vc + static_cast<size_t>(slot) * kvd + kvh * 128u + lane * 4u;
            kv4 = make_float4((float)kp[0], (float)kp[1], (float)kp[2], (float)kp[3]);
            vv4 = make_float4((float)vp[0], (float)vp[1], (float)vp[2], (float)vp[3]);
        } else {
            uint32_t br = j - nctx;
            kv4 = reinterpret_cast<const float4*>(kb + static_cast<size_t>(br) * kvd + kvh * 128u)[lane];
            vv4 = reinterpret_cast<const float4*>(vb + static_cast<size_t>(br) * kvd + kvh * 128u)[lane];
        }
        float sc = wave32_sum(qv.x * kv4.x + qv.y * kv4.y + qv.z * kv4.z + qv.w * kv4.w) * scale;
        float mn = fmaxf(m, sc);
        float al = __expf(m - mn), pr = __expf(sc - mn);
        l = l * al + pr;
        m = mn;
        o.x = fmaf(vv4.x, pr, o.x * al); o.y = fmaf(vv4.y, pr, o.y * al);
        o.z = fmaf(vv4.z, pr, o.z * al); o.w = fmaf(vv4.w, pr, o.w * al);
    }
    __shared__ float s_m[8], s_l[8];
    __shared__ float4 s_o[8][32];
    if (lane == 0u) { s_m[warp] = m; s_l[warp] = l; }
    s_o[warp][lane] = o;
    __syncthreads();
    if (warp == 0u) {
        float mb = -1e30f;
        for (uint32_t w = 0u; w < 8u; ++w) if (s_l[w] > 0.0f) mb = fmaxf(mb, s_m[w]);
        float lb = 0.0f;
        float4 ob = make_float4(0, 0, 0, 0);
        for (uint32_t w = 0u; w < 8u; ++w) {
            if (s_l[w] > 0.0f) {
                float f = __expf(s_m[w] - mb);
                lb += s_l[w] * f;
                float4 ow = s_o[w][lane];
                ob.x = fmaf(ow.x, f, ob.x); ob.y = fmaf(ow.y, f, ob.y); ob.z = fmaf(ow.z, f, ob.z); ob.w = fmaf(ow.w, f, ob.w);
            }
        }
        float inv = (lb > 0.0f) ? 1.0f / lb : 0.0f;
        reinterpret_cast<float4*>(out + (static_cast<size_t>(r) * nh + h) * 128u)[lane] = make_float4(ob.x * inv, ob.y * inv, ob.z * inv, ob.w * inv);
    }
}

// out[i] = silu(g[i]) * u[i].   Grid: ceil(n / 256), Block: 256
extern "C" __global__ void smalltide_silu_mul(float* __restrict__ out, const float* __restrict__ g, const float* __restrict__ u, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = g[i];
    out[i] = x / (1.0f + __expf(-x)) * u[i];
}

// Markov head input: z = [state(R) | w1[prev](R, bf16) | h(D) | f_rnn(D)]; prev = anchor at step 0,
// else drafts[step - 1].   Grid: ceil((2R + 2D) / 256), Block: 256
extern "C" __global__ void smalltide_dspark_z(
    float* __restrict__ z, const float* __restrict__ state, const uint16_t* __restrict__ w1, const uint32_t* __restrict__ drafts,
    uint32_t step, uint32_t anchor, const float* __restrict__ h, const float* __restrict__ frnn, uint32_t R, uint32_t D
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = 2u * R + 2u * D;
    if (i >= n) return;
    float v;
    if (i < R) {
        v = state[i];
    } else if (i < 2u * R) {
        uint32_t prev = (step == 0u) ? anchor : drafts[step - 1u];
        v = __uint_as_float(static_cast<uint32_t>(w1[static_cast<size_t>(prev) * R + (i - R)]) << 16);
    } else if (i < 2u * R + D) {
        v = h[i - 2u * R];
    } else {
        v = frnn[i - 2u * R - D];
    }
    z[i] = v;
}

// Markov cell: raw = rnn_proj(z) (no bias yet): gate / cand / out chunks of R.
//   Grid: ceil(R / 256), Block: 256
extern "C" __global__ void smalltide_dspark_cell(
    float* __restrict__ state, float* __restrict__ latent, const float* __restrict__ raw, const float* __restrict__ bias, uint32_t R
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R) return;
    float g = raw[i] + bias[i], c = raw[R + i] + bias[R + i], o = raw[2u * R + i] + bias[2u * R + i];
    float gate = 1.0f / (1.0f + __expf(-g));
    latent[i] = tanhf(o);
    state[i] = gate * state[i] + (1.0f - gate) * tanhf(c);
}

// Step logits y[v] = cap * tanh((base[v] + fold_b[v] + sum_r latent[r] fold[r][v]) / cap) for v < V,
// block-wise argmax partials.   Grid: ceil(V / 256), Block: 256
extern "C" __global__ void smalltide_dspark_logits_argmax(
    float* __restrict__ pmax, uint32_t* __restrict__ pidx, const float* __restrict__ base, const float* __restrict__ fold_b,
    const _Float16* __restrict__ fold, const float* __restrict__ latent, uint32_t V, uint32_t R, float cap
) {
    __shared__ float s_lat[256];
    __shared__ float s_v[8];
    __shared__ uint32_t s_i[8];
    for (uint32_t i = threadIdx.x; i < R; i += blockDim.x) s_lat[i] = latent[i];
    __syncthreads();
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    float y = -INFINITY;
    if (v < V) {
        float acc = base[v] + fold_b[v];
        for (uint32_t r = 0u; r < R; ++r) acc = fmaf(s_lat[r], (float)fold[static_cast<size_t>(r) * V + v], acc);
        y = cap * tanhf(acc / cap);
    }
    uint32_t bi = (v < V) ? v : 0xFFFFFFFFu;
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor(y, o, 32);
        uint32_t oi = __shfl_xor(bi, o, 32);
        if (ov > y || (ov == y && oi < bi)) { y = ov; bi = oi; }
    }
    uint32_t w = threadIdx.x >> 5;
    if ((threadIdx.x & 31u) == 0u) { s_v[w] = y; s_i[w] = bi; }
    __syncthreads();
    if (threadIdx.x == 0u) {
        float bv = s_v[0];
        uint32_t bb = s_i[0];
        for (uint32_t k = 1u; k < (blockDim.x >> 5); ++k) if (s_v[k] > bv || (s_v[k] == bv && s_i[k] < bb)) { bv = s_v[k]; bb = s_i[k]; }
        pmax[blockIdx.x] = bv;
        pidx[blockIdx.x] = bb;
    }
}

// drafts[step] = argmax over the partials.   Grid: 1, Block: 1024
extern "C" __global__ void smalltide_dspark_argmax_final(
    uint32_t* __restrict__ drafts, uint32_t step, const float* __restrict__ pmax, const uint32_t* __restrict__ pidx, uint32_t n
) {
    float bv = -INFINITY;
    uint32_t bi = 0xFFFFFFFFu;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = pmax[i];
        uint32_t ix = pidx[i];
        if (v > bv || (v == bv && ix < bi)) { bv = v; bi = ix; }
    }
    __shared__ float sv[32];
    __shared__ uint32_t si[32];
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor(bv, o, 32);
        uint32_t oi = __shfl_xor(bi, o, 32);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    uint32_t w = threadIdx.x >> 5, l = threadIdx.x & 31u;
    if (l == 0u) { sv[w] = bv; si[w] = bi; }
    __syncthreads();
    if (w == 0u) {
        bv = (l < (blockDim.x >> 5)) ? sv[l] : -INFINITY;
        bi = (l < (blockDim.x >> 5)) ? si[l] : 0xFFFFFFFFu;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor(bv, o, 32);
            uint32_t oi = __shfl_xor(bi, o, 32);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (l == 0u) drafts[step] = bi;
    }
}

// ---------------------------------------------------------------------------
// Verify MoE epilogue for `rows` consecutive positions: per row the renormalized expert sum
// (tmp [rows*8][H], skipped experts excluded), the MLP short conv run row by row (+ history
// [3 + rows][H]) and the residual: cur[r] = ffn_inp[r] + sconv(sum_r).
//   Grid: ceil(H / 256), Block: 256
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_moe_post_sconv_rows(
    float* __restrict__ cur, const float* __restrict__ ffn_inp, const float* __restrict__ tmp,
    const uint32_t* __restrict__ slot_table, const uint32_t* __restrict__ route, float* __restrict__ state,
    const float* __restrict__ weight, float* __restrict__ hist, uint32_t hidden, uint32_t rows
) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hidden) return;
    float s0 = state[c], s1 = state[hidden + c], s2 = state[2u * hidden + c];
    if (hist != nullptr) { hist[c] = s0; hist[hidden + c] = s1; hist[2u * hidden + c] = s2; }
    const float4 w = reinterpret_cast<const float4*>(weight)[c];
    for (uint32_t r = 0u; r < rows; ++r) {
        const uint32_t* ids = route + r * 16u;
        const float* wts = reinterpret_cast<const float*>(route + r * 16u + 8u);
        float tot = 0.0f, ex = 0.0f;
        for (uint32_t k = 0u; k < 8u; ++k) {
            tot += wts[k];
            if (slot_table[ids[k]] != 0xFFFFFFFFu) ex += wts[k];
        }
        float renorm = (ex > 1e-8f) ? (tot / ex) : 0.0f;
        float acc = 0.0f;
        for (uint32_t k = 0u; k < 8u; ++k) {
            if (slot_table[ids[k]] != 0xFFFFFFFFu) acc = fmaf(wts[k] * renorm, tmp[(static_cast<size_t>(r) * 8u + k) * hidden + c], acc);
        }
        if (hist != nullptr) hist[static_cast<size_t>(3u + r) * hidden + c] = acc;
        float y = fmaf(s0, w.x, fmaf(s1, w.y, fmaf(s2, w.z, acc * w.w)));
        y = (acc + y) * 0.4472135954999579f;
        cur[static_cast<size_t>(r) * hidden + c] = ffn_inp[static_cast<size_t>(r) * hidden + c] + y;
        s0 = s1; s1 = s2; s2 = acc;
    }
    state[c] = s0; state[hidden + c] = s1; state[2u * hidden + c] = s2;
}
