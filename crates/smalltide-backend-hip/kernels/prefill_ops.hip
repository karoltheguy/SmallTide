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
    uint32_t m_rows, uint32_t dyn_stride
) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= dim) return;
    float s0 = state[c], s1 = state[dim + c], s2 = state[2u * dim + c];
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
