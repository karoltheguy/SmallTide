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
//    An output row depends on its own and the three previous INPUT rows only, so the rows are
//    independent: one block per (256 channels, ST_SCONV_ROWS rows), each thread one channel.
//    `out` must not alias `x`.
//    Grid: (ceil(dim / 256), ceil(M / ST_SCONV_ROWS)), Block: 256
// ---------------------------------------------------------------------------
#define ST_SCONV_ROWS 16u
extern "C" __global__ void smalltide_sconv_k4_rows_f32(
    float* __restrict__ out, float* __restrict__ state, const float* __restrict__ x,
    const float* __restrict__ weight, const float* __restrict__ dyn, uint32_t dim, uint32_t head_dim,
    uint32_t m_rows, uint32_t dyn_stride, float* __restrict__ hist
) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t t0 = blockIdx.y * ST_SCONV_ROWS;
    if (c >= dim || t0 >= m_rows) return;
    uint32_t t1 = min(m_rows, t0 + ST_SCONV_ROWS);
    // The three inputs before row t0: the saved state for the first block, x itself after it.
    float s0, s1, s2;
    if (t0 == 0u) {
        s0 = state[c]; s1 = state[dim + c]; s2 = state[2u * dim + c];
    } else {
        s0 = x[static_cast<size_t>(t0 - 3u) * dim + c];
        s1 = x[static_cast<size_t>(t0 - 2u) * dim + c];
        s2 = x[static_cast<size_t>(t0 - 1u) * dim + c];
    }
    const float o0 = s0, o1 = s1, o2 = s2;
    if (hist != nullptr) {
        // [3 + m_rows][dim]: the old state, then every input (speculative-verify rewind).
        if (t0 == 0u) { hist[c] = s0; hist[dim + c] = s1; hist[2u * dim + c] = s2; }
        for (uint32_t t = t0; t < t1; ++t) hist[static_cast<size_t>(3u + t) * dim + c] = x[static_cast<size_t>(t) * dim + c];
    }
    const float4 w = reinterpret_cast<const float4*>(weight)[c];
    uint32_t h = (head_dim > 0u) ? (c / head_dim) : 0u;
    for (uint32_t t = t0; t < t1; ++t) {
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
    if (t0 == 0u) {
        // The new state is the last three inputs of the whole call (the first block is the only
        // one that reads or writes `state`).
        float n0, n1, n2;
        if (m_rows >= 3u) {
            n0 = x[static_cast<size_t>(m_rows - 3u) * dim + c];
            n1 = x[static_cast<size_t>(m_rows - 2u) * dim + c];
            n2 = x[static_cast<size_t>(m_rows - 1u) * dim + c];
        } else if (m_rows == 2u) {
            n0 = o2; n1 = x[c]; n2 = x[dim + c];
        } else {
            n0 = o1; n1 = o2; n2 = x[c];
        }
        state[c] = n0; state[dim + c] = n1; state[2u * dim + c] = n2;
    }
}

// ---------------------------------------------------------------------------
// FP8 KV store for M rows at positions pos0..pos0+M-1 (ring slot = pos % max_seq if local).
//    k_in, v_in: [M][n_kv_heads][head_dim]; Grid: (n_kv_heads, M), Block: head_dim (128)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_kv_store_fp8_rows(
    uint8_t* __restrict__ k_cache, float* __restrict__ k_scales, uint8_t* __restrict__ v_cache,
    float* __restrict__ v_scales, const float* __restrict__ k_in, const float* __restrict__ v_in,
    uint32_t n_kv_heads, uint32_t max_seq, uint32_t head_dim, uint32_t pos0, uint32_t is_local,
    const uint32_t* __restrict__ page_table, uint32_t v_q4
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
    float ks = (kmax > 0.0f) ? (kmax / 448.0f) : 1.0f;
    uint32_t pos = pos0 + t;
    uint32_t slot = (is_local != 0u) ? (pos % max_seq) : pos;
    size_t si = static_cast<size_t>(kv_phys(page_table, slot)) * n_kv_heads + h;
    if (d == 0u) k_scales[si] = ks;
    k_cache[si * head_dim + d] = encode_fp8_e4m3fn(kv / ks);
    if (v_q4 != 0u) {
        kv_store_v_q4(v_cache, v_scales, si, d, vv);
    } else {
        float vmax = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3]));
        float vs = (vmax > 0.0f) ? (vmax / 448.0f) : 1.0f;
        if (d == 0u) v_scales[si] = vs;
        v_cache[si * head_dim + d] = encode_fp8_e4m3fn(vv / vs);
    }
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
    uint32_t max_seq, uint32_t window, uint32_t pos0, uint32_t rel_extent, float inv_sqrt_d,
    const uint32_t* __restrict__ page_table, uint32_t v_q4
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
    // Position-major cache ([slot][n_kv_heads][128], scales [slot][n_kv_heads]).
    const uint8_t* kb = k_cache + static_cast<size_t>(kvh) * 128u;
    const uint8_t* vb = v_cache + static_cast<size_t>(kvh) * 128u;
    const float* ksb = k_scales + kvh;
    const float* vsb = v_scales + kvh;
    const size_t kv_stride = static_cast<size_t>(n_kv_heads) * 128u;
    for (uint32_t kp = k_start + warp; kp <= p; kp += 4u) {
        uint32_t slot = kv_phys(page_table, (window != 0u) ? (kp % max_seq) : kp);
        uint32_t dist = p - kp;
        uint32_t k4 = reinterpret_cast<const uint32_t*>(kb + static_cast<size_t>(slot) * kv_stride)[lane];
        float kx = decode_fp8_e4m3fn(k4 & 0xFFu), ky = decode_fp8_e4m3fn((k4 >> 8) & 0xFFu);
        float kz = decode_fp8_e4m3fn((k4 >> 16) & 0xFFu), kw = decode_fp8_e4m3fn((k4 >> 24) & 0xFFu);
        float kscale = ksb[static_cast<size_t>(slot) * n_kv_heads] * sc_mul;
        float vx, vy, vz, vw;
        if (v_q4 != 0u) {
            float4 vv = kv_load_v_q4(v_cache, v_scales, static_cast<size_t>(slot) * n_kv_heads + kvh, lane);
            vx = vv.x; vy = vv.y; vz = vv.z; vw = vv.w;
        } else {
            uint32_t v4 = reinterpret_cast<const uint32_t*>(vb + static_cast<size_t>(slot) * kv_stride)[lane];
            float vscale = vsb[static_cast<size_t>(slot) * n_kv_heads];
            vx = decode_fp8_e4m3fn(v4 & 0xFFu) * vscale; vy = decode_fp8_e4m3fn((v4 >> 8) & 0xFFu) * vscale;
            vz = decode_fp8_e4m3fn((v4 >> 16) & 0xFFu) * vscale; vw = decode_fp8_e4m3fn((v4 >> 24) & 0xFFu) * vscale;
        }
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
// Multi-M-tile WMMA grouped MoE for prefill: up to 64 (row, k) pairs per work item (1..=4 M-tiles
// of 16), decoding every NVFP4/MXFP8 weight fragment ONCE across all active M-tiles.
// Exact same K-order and WMMA arithmetic as `smalltide_moe_*_grouped_wmma2`.
// ---------------------------------------------------------------------------
#define ST_MOE_PAIRS_M64 64u

template <uint32_t MG>
__device__ __forceinline__ void wmma_dot_k_mg(
    uint32_t tag,
    const _Float16* const ar[MG],
    const bool valid[MG],
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ sc,
    uint32_t K,
    uint32_t kh,
    st_float8 c[MG]
) {
    if constexpr (MG == 1u) {
        c[0] = wmma_dot_k(tag, ar[0], valid[0], w, sc, K, kh, c[0]);
        return;
    }
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    uint32_t hoff = (kh != 0u) ? 32u : 0u;
    if (tag == 0u) {
        for (uint32_t k = 0u; k < K; k += 64u) {
            uint32_t base = k + hoff;
            uint4 wv = *reinterpret_cast<const uint4*>(w + (base >> 1));
            uint8_t q0 = sc[base >> 4], q1 = sc[(base >> 4) + 1u];
            uint4 a0[MG], a1[MG];
            #pragma unroll
            for (uint32_t g = 0u; g < MG; ++g) {
                const uint4* ap = reinterpret_cast<const uint4*>(ar[g] + base);
                a0[g] = valid[g] ? ap[0] : zero;
                a1[g] = valid[g] ? ap[1] : zero;
            }
            float s0 = decode_fp8_e4m3fn(q0), s1 = decode_fp8_e4m3fn(q1);
            st_half8 b0 = wmma_nvfp4_frag(wv.x, s0);
            st_half8 b1 = wmma_nvfp4_frag(wv.y, s0);
            #pragma unroll
            for (uint32_t g = 0u; g < MG; ++g) {
                c[g] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a0[g]), b0, c[g]);
                c[g] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a1[g]), b1, c[g]);
            }
            #pragma unroll
            for (uint32_t g = 0u; g < MG; ++g) {
                const uint4* ap = reinterpret_cast<const uint4*>(ar[g] + base);
                a0[g] = valid[g] ? ap[2] : zero;
                a1[g] = valid[g] ? ap[3] : zero;
            }
            st_half8 b2 = wmma_nvfp4_frag(wv.z, s1);
            st_half8 b3 = wmma_nvfp4_frag(wv.w, s1);
            #pragma unroll
            for (uint32_t g = 0u; g < MG; ++g) {
                c[g] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a0[g]), b2, c[g]);
                c[g] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a1[g]), b3, c[g]);
            }
        }
    } else {
        for (uint32_t k = 0u; k < K; k += 64u) {
            uint32_t base = k + hoff;
            const uint4* wp = reinterpret_cast<const uint4*>(w + base);
            uint4 wa = wp[0], wb = wp[1];
            uint8_t q0 = sc[base >> 5];
            uint4 a0[MG], a1[MG];
            #pragma unroll
            for (uint32_t g = 0u; g < MG; ++g) {
                const uint4* ap = reinterpret_cast<const uint4*>(ar[g] + base);
                a0[g] = valid[g] ? ap[0] : zero;
                a1[g] = valid[g] ? ap[1] : zero;
            }
            float s0 = decode_ue8m0(q0);
            st_half8 b0 = wmma_mxfp8_frag(make_uint2(wa.x, wa.y), s0);
            st_half8 b1 = wmma_mxfp8_frag(make_uint2(wa.z, wa.w), s0);
            #pragma unroll
            for (uint32_t g = 0u; g < MG; ++g) {
                c[g] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a0[g]), b0, c[g]);
                c[g] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a1[g]), b1, c[g]);
            }
            #pragma unroll
            for (uint32_t g = 0u; g < MG; ++g) {
                const uint4* ap = reinterpret_cast<const uint4*>(ar[g] + base);
                a0[g] = valid[g] ? ap[2] : zero;
                a1[g] = valid[g] ? ap[3] : zero;
            }
            st_half8 b2 = wmma_mxfp8_frag(make_uint2(wb.x, wb.y), s0);
            st_half8 b3 = wmma_mxfp8_frag(make_uint2(wb.z, wb.w), s0);
            #pragma unroll
            for (uint32_t g = 0u; g < MG; ++g) {
                c[g] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a0[g]), b2, c[g]);
                c[g] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(as_half8(a1[g]), b3, c[g]);
            }
        }
    }
}

template <uint32_t MG>
__device__ __forceinline__ void moe_w13_m64_body(
    _Float16* __restrict__ mid16, const _Float16* __restrict__ x16,
    const uint32_t* s_pairs, uint32_t np, uint32_t tag,
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc,
    float osc, uint32_t n0, uint32_t l, uint32_t hidden, uint32_t inter,
    float gate_cap, float up_cap
) {
    const _Float16* xr[MG];
    bool valid[MG];
    st_float8 c[MG];
    #pragma unroll
    for (uint32_t g = 0u; g < MG; ++g) {
        uint32_t m = 16u * g + (l & 15u);
        valid[g] = m < np;
        xr[g] = x16 + static_cast<size_t>(valid[g] ? (s_pairs[m] >> 3) : 0u) * hidden;
        c[g] = {};
    }
    wmma_dot_k_mg<MG>(tag, xr, valid, w, sc, hidden, l >> 4, c);
    uint32_t r = (n0 + (l & 15u)) >> 1;
    #pragma unroll
    for (uint32_t g = 0u; g < MG; ++g) {
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; ++i) {
            float gval = c[g][i] * osc;
            float uval = __shfl_xor(c[g][i], 1, 32) * osc;
            uint32_t mm = 16u * g + (l >> 4) * 8u + i;
            if ((l & 1u) == 0u && mm < np) {
                mid16[static_cast<size_t>(s_pairs[mm]) * inter + r] = (_Float16)situ_f32(gval, uval, gate_cap, up_cap);
            }
        }
    }
}

// Pass 1 (m64): mid16[pair][r] = SiTU(gate_r . x16[row], up_r . x16[row]), up to 64 pairs/work-item.
//   Grid: (n_work, ceil(2I / 128)), Block: 256
extern "C" __global__ void smalltide_moe_w13_grouped_wmma_m64(
    _Float16* __restrict__ mid16, const _Float16* __restrict__ x16, const uint8_t* __restrict__ pool,
    const uint32_t* __restrict__ work, const uint32_t* __restrict__ pairs, uint64_t slab_stride,
    uint32_t hidden, uint32_t inter, float gate_cap, float up_cap
) {
    uint32_t wi = blockIdx.x;
    uint32_t slot = work[wi * 4u], p0 = work[wi * 4u + 1u], np = work[wi * 4u + 2u];
    if (np == 0u) return;
    uint32_t warp = threadIdx.x >> 5, l = threadIdx.x & 31u;
    __shared__ uint32_t s_pairs[ST_MOE_PAIRS_M64];
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
    float osc = (tag == 0u) ? hdr->w13_outer_scale : 1.0f;
    if (np <= 16u) {
        moe_w13_m64_body<1u>(mid16, x16, s_pairs, np, tag, w, sc, osc, n0, l, hidden, inter, gate_cap, up_cap);
    } else if (np <= 32u) {
        moe_w13_m64_body<2u>(mid16, x16, s_pairs, np, tag, w, sc, osc, n0, l, hidden, inter, gate_cap, up_cap);
    } else if (np <= 48u) {
        moe_w13_m64_body<3u>(mid16, x16, s_pairs, np, tag, w, sc, osc, n0, l, hidden, inter, gate_cap, up_cap);
    } else {
        moe_w13_m64_body<4u>(mid16, x16, s_pairs, np, tag, w, sc, osc, n0, l, hidden, inter, gate_cap, up_cap);
    }
}

template <uint32_t MG>
__device__ __forceinline__ void moe_w2_m64_body(
    float* __restrict__ tmp, const _Float16* __restrict__ mid16,
    const uint32_t* s_pairs, const float* s_w, uint32_t np, uint32_t tag,
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc,
    float osc, uint32_t n0, uint32_t l, uint32_t hidden, uint32_t inter
) {
    const _Float16* ar[MG];
    bool valid[MG];
    st_float8 c[MG];
    #pragma unroll
    for (uint32_t g = 0u; g < MG; ++g) {
        uint32_t m = 16u * g + (l & 15u);
        valid[g] = m < np;
        ar[g] = mid16 + static_cast<size_t>(valid[g] ? s_pairs[m] : 0u) * inter;
        c[g] = {};
    }
    wmma_dot_k_mg<MG>(tag, ar, valid, w, sc, inter, l >> 4, c);
    uint32_t hh = n0 + (l & 15u);
    #pragma unroll
    for (uint32_t g = 0u; g < MG; ++g) {
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; ++i) {
            uint32_t mm = 16u * g + (l >> 4) * 8u + i;
            if (mm < np) tmp[static_cast<size_t>(s_pairs[mm]) * hidden + hh] = c[g][i] * osc * s_w[mm];
        }
    }
}

// Pass 2 (m64): tmp[pair][h] = weight(pair) * (w2_h . mid16[pair]), up to 64 pairs/work-item.
//   Grid: (n_work, ceil(H / 128)), Block: 256
extern "C" __global__ void smalltide_moe_w2_grouped_wmma_m64(
    float* __restrict__ tmp, const _Float16* __restrict__ mid16, const uint8_t* __restrict__ pool,
    const uint32_t* __restrict__ work, const uint32_t* __restrict__ pairs, const uint32_t* __restrict__ route,
    uint64_t slab_stride, uint32_t hidden, uint32_t inter
) {
    uint32_t wi = blockIdx.x;
    uint32_t slot = work[wi * 4u], p0 = work[wi * 4u + 1u], np = work[wi * 4u + 2u];
    if (np == 0u) return;
    uint32_t warp = threadIdx.x >> 5, l = threadIdx.x & 31u;
    __shared__ uint32_t s_pairs[ST_MOE_PAIRS_M64];
    __shared__ float s_w[ST_MOE_PAIRS_M64];
    if (threadIdx.x < np) {
        uint32_t pr = pairs[p0 + threadIdx.x];
        s_pairs[threadIdx.x] = pr;
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
    float osc = (tag == 0u) ? hdr->w2_outer_scale : 1.0f;
    if (np <= 16u) {
        moe_w2_m64_body<1u>(tmp, mid16, s_pairs, s_w, np, tag, w, sc, osc, n0, l, hidden, inter);
    } else if (np <= 32u) {
        moe_w2_m64_body<2u>(tmp, mid16, s_pairs, s_w, np, tag, w, sc, osc, n0, l, hidden, inter);
    } else if (np <= 48u) {
        moe_w2_m64_body<3u>(tmp, mid16, s_pairs, s_w, np, tag, w, sc, osc, n0, l, hidden, inter);
    } else {
        moe_w2_m64_body<4u>(tmp, mid16, s_pairs, s_w, np, tag, w, sc, osc, n0, l, hidden, inter);
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
    uint32_t max_seq, uint32_t window, uint32_t pos0, uint32_t rel_extent, float inv_sqrt_d, uint32_t rows,
    const uint32_t* __restrict__ page_table, uint32_t v_q4
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

    // Position-major cache ([slot][n_kv_heads][128], scales [slot][n_kv_heads]).
    const uint8_t* kb = k_cache + static_cast<size_t>(kvh) * 128u;
    const uint8_t* vb = v_cache + static_cast<size_t>(kvh) * 128u;
    const float* ksb = k_scales + kvh;
    const float* vsb = v_scales + kvh;
    const size_t kv_stride = static_cast<size_t>(n_kv_heads) * 128u;

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
        // The tile's first key in the ring, once: a 32-bit modulo is a long instruction sequence
        // here, and the 16 keys of a tile need only an add and a wrap.
        uint32_t ring0 = (window != 0u) ? (k0 % max_seq) : k0;
        uint32_t ks = ring0 + nl;
        if (window != 0u && ks >= max_seq) ks -= max_seq;
        uint32_t slot = kv_phys(page_table, ks);
        // S = Q K^T over 8 dim chunks.
        st_float8 sacc = {};
        #pragma unroll
        for (uint32_t c = 0u; c < 8u; ++c) {
            uint2 kw = kin ? *reinterpret_cast<const uint2*>(kb + static_cast<size_t>(slot) * kv_stride + c * 16u + half_ * 8u) : make_uint2(0u, 0u);
            float4 a = decode_fp8x4(kw.x), b = decode_fp8x4(kw.y);
            st_half8 kf;
            kf[0] = (_Float16)a.x; kf[1] = (_Float16)a.y; kf[2] = (_Float16)a.z; kf[3] = (_Float16)a.w;
            kf[4] = (_Float16)b.x; kf[5] = (_Float16)b.y; kf[6] = (_Float16)b.z; kf[7] = (_Float16)b.w;
            sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(qa[c], kf, sacc);
        }
        float kscale = kin ? ksb[static_cast<size_t>(slot) * n_kv_heads] : 0.0f;
        float vscale = (v_q4 != 0u) ? 1.0f : (kin ? vsb[static_cast<size_t>(slot) * n_kv_heads] : 0.0f);
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
            uint32_t vs = ring0 + r;
            if (window != 0u && vs >= max_seq) vs -= max_seq;
            uint32_t vslot = kv_phys(page_table, vs);
            float4 vv;
            if (kr <= key_hi) {
                vv = (v_q4 != 0u)
                    ? kv_load_v_q4(v_cache, v_scales, static_cast<size_t>(vslot) * n_kv_heads + kvh, l)
                    : decode_fp8x4(reinterpret_cast<const uint32_t*>(vb + static_cast<size_t>(vslot) * kv_stride)[l]);
            } else {
                vv = make_float4(0, 0, 0, 0);
            }
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
// V tiles for smalltide_attn_prefill_wmma_vt: the values of keys key_lo..=key_hi (key_lo a
// multiple of 16) decoded to f16 once, dim-major per 16-key tile: vt16[tile][head][dim][16].
// A prefill pass has ~190 query blocks per KV head and each of them needs every tile of its
// window: decoding the tile inside the attention kernel repeats the same work that many times
// (about three quarters of that kernel's time). The f16 values are exactly the ones the
// attention kernels compute themselves; keys past key_hi are 0 (they are masked, but a stale
// NaN would still poison P * V).
//   Tile index: (ring slot of its first key) / 16 for a local ring (max_seq % 16 == 0), key / 16
//   for a global cache.   Grid: (n_kv_heads, n_tiles), Block: 128 (one thread a dim)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_kv_v_tiles_f16(
    uint4* __restrict__ vt16, const uint8_t* __restrict__ v_cache, const float* __restrict__ v_scales,
    uint32_t n_kv_heads, uint32_t max_seq, uint32_t window, uint32_t key_lo, uint32_t key_hi,
    const uint32_t* __restrict__ page_table, uint32_t v_q4
) {
    uint32_t kvh = blockIdx.x, d = threadIdx.x;
    uint32_t k0 = key_lo + blockIdx.y * 16u;
    uint32_t ring0 = (window != 0u) ? (k0 % max_seq) : k0;
    const uint8_t* vb = v_cache + static_cast<size_t>(kvh) * 128u;
    const size_t kv_stride = static_cast<size_t>(n_kv_heads) * 128u;
    st_half8 h[2];
    #pragma unroll
    for (uint32_t r = 0u; r < 16u; ++r) {
        uint32_t kr = k0 + r;
        uint32_t vs = ring0 + r;
        if (window != 0u && vs >= max_seq) vs -= max_seq;
        uint32_t vslot = kv_phys(page_table, vs);
        float v = 0.0f;
        if (kr <= key_hi) {
            // The component d & 3 of what the attention kernels load for lane d >> 2.
            float4 vv = (v_q4 != 0u)
                ? kv_load_v_q4(v_cache, v_scales, static_cast<size_t>(vslot) * n_kv_heads + kvh, d >> 2)
                : decode_fp8x4(reinterpret_cast<const uint32_t*>(vb + static_cast<size_t>(vslot) * kv_stride)[d >> 2]);
            uint32_t i = d & 3u;
            v = (i == 0u) ? vv.x : (i == 1u) ? vv.y : (i == 2u) ? vv.z : vv.w;
        }
        h[r >> 3][r & 7u] = (_Float16)v;
    }
    uint4* dst = vt16 + ((static_cast<size_t>(ring0 >> 4) * n_kv_heads + kvh) * 128u + d) * 2u;
    dst[0] = __builtin_bit_cast(uint4, h[0]);
    dst[1] = __builtin_bit_cast(uint4, h[1]);
}

// The same as smalltide_attn_prefill_wmma, value for value, with the V tile read pre-decoded.
extern "C" __global__ void smalltide_attn_prefill_wmma_vt(
    float* __restrict__ out, const float* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const float* __restrict__ k_scales, const uint8_t* __restrict__ v_cache, const float* __restrict__ v_scales,
    const float* __restrict__ rel_table, const float* __restrict__ sinks, uint32_t n_q_heads, uint32_t n_kv_heads,
    uint32_t max_seq, uint32_t window, uint32_t pos0, uint32_t rel_extent, float inv_sqrt_d, uint32_t rows,
    const uint32_t* __restrict__ page_table, uint32_t v_q4, const uint4* __restrict__ vt16
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

    // Position-major cache ([slot][n_kv_heads][128], scales [slot][n_kv_heads]).
    const uint8_t* kb = k_cache + static_cast<size_t>(kvh) * 128u;
    const uint8_t* vb = v_cache + static_cast<size_t>(kvh) * 128u;
    const float* ksb = k_scales + kvh;
    const float* vsb = v_scales + kvh;
    const size_t kv_stride = static_cast<size_t>(n_kv_heads) * 128u;

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
        // The tile's first key in the ring, once: a 32-bit modulo is a long instruction sequence
        // here, and the 16 keys of a tile need only an add and a wrap.
        uint32_t ring0 = (window != 0u) ? (k0 % max_seq) : k0;
        uint32_t ks = ring0 + nl;
        if (window != 0u && ks >= max_seq) ks -= max_seq;
        uint32_t slot = kv_phys(page_table, ks);
        // S = Q K^T over 8 dim chunks.
        st_float8 sacc = {};
        #pragma unroll
        for (uint32_t c = 0u; c < 8u; ++c) {
            uint2 kw = kin ? *reinterpret_cast<const uint2*>(kb + static_cast<size_t>(slot) * kv_stride + c * 16u + half_ * 8u) : make_uint2(0u, 0u);
            float4 a = decode_fp8x4(kw.x), b = decode_fp8x4(kw.y);
            st_half8 kf;
            kf[0] = (_Float16)a.x; kf[1] = (_Float16)a.y; kf[2] = (_Float16)a.z; kf[3] = (_Float16)a.w;
            kf[4] = (_Float16)b.x; kf[5] = (_Float16)b.y; kf[6] = (_Float16)b.z; kf[7] = (_Float16)b.w;
            sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(qa[c], kf, sacc);
        }
        float kscale = kin ? ksb[static_cast<size_t>(slot) * n_kv_heads] : 0.0f;
        float vscale = (v_q4 != 0u) ? 1.0f : (kin ? vsb[static_cast<size_t>(slot) * n_kv_heads] : 0.0f);
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
        // The V tile comes pre-decoded (smalltide_kv_v_tiles_f16): [tile][head][dim][16 keys] f16,
        // so this lane's 8 keys of one dim are a single 16-byte load - no decode, no LDS staging.
        const uint4* vtile = vt16 + (static_cast<size_t>(ring0 >> 4) * n_kv_heads + kvh) * 256u;
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        st_half8 pa;
        #pragma unroll
        for (uint32_t j = 0u; j < 8u; ++j) pa[j] = s_p[warp][nl][half_ * 8u + j];
        #pragma unroll
        for (uint32_t c = 0u; c < 8u; ++c) {
            st_half8 vf = as_half8(vtile[(c * 16u + nl) * 2u + half_]);
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

// Split-K WMMA flash attention for low-row speculative verify (rows <= 8): splits key tiles across
// `num_splits` blocks in grid.z so all 64 CUs stay busy even when `ceil(rows * g / 16) == 1`.
//   Grid: (n_kv_heads, ceil(rows * g / 16), num_splits), Block: 128
extern "C" __global__ void smalltide_attn_prefill_wmma_split(
    float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
    const float* __restrict__ q, const uint8_t* __restrict__ k_cache,
    const float* __restrict__ k_scales, const uint8_t* __restrict__ v_cache, const float* __restrict__ v_scales,
    const float* __restrict__ rel_table, uint32_t n_q_heads, uint32_t n_kv_heads,
    uint32_t max_seq, uint32_t window, uint32_t pos0, uint32_t rel_extent, float inv_sqrt_d, uint32_t rows,
    uint32_t num_splits, const uint32_t* __restrict__ page_table, uint32_t v_q4
) {
    const float LOG2E = 1.4426950408889634f;
    uint32_t kvh = blockIdx.x;
    uint32_t g = n_q_heads / n_kv_heads;
    uint32_t qh0 = kvh * g;
    uint32_t nq = rows * g;
    uint32_t q_base = blockIdx.y * 16u;
    uint32_t sp = blockIdx.z;
    uint32_t warp = threadIdx.x >> 5, l = threadIdx.x & 31u;
    uint32_t nl = l & 15u, half_ = l >> 4;

    __shared__ _Float16 s_p[AW_WARPS][16][16];
    __shared__ _Float16 s_v[AW_WARPS][16][128];
    __shared__ float s_m[AW_WARPS][16], s_l[AW_WARPS][16];
    __shared__ float s_o[AW_WARPS][16][128 + 4];

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
    uint32_t key_lo = lo_first & ~15u;
    uint32_t key_hi = p_last;
    uint32_t n_tiles = (key_hi - key_lo) / 16u + 1u;
    uint32_t tiles_per_sp = (n_tiles + num_splits - 1u) / num_splits;
    uint32_t t_begin = sp * tiles_per_sp;
    uint32_t t_end = min(n_tiles, t_begin + tiles_per_sp);

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
    uint32_t tC[8], qhC[8], pC[8], loC[8];
    #pragma unroll
    for (uint32_t i = 0u; i < 8u; ++i) qinfo(half_ * 8u + i, tC[i], qhC[i], pC[i], loC[i]);

    const uint8_t* kb = k_cache + static_cast<size_t>(kvh) * 128u;
    const uint8_t* vb = v_cache + static_cast<size_t>(kvh) * 128u;
    const float* ksb = k_scales + kvh;
    const float* vsb = v_scales + kvh;
    const size_t kv_stride = static_cast<size_t>(n_kv_heads) * 128u;

    float m_run[8], l_run[8];
    st_float8 o[8];
    #pragma unroll
    for (uint32_t i = 0u; i < 8u; ++i) { m_run[i] = -1e30f; l_run[i] = 0.0f; }
    #pragma unroll
    for (uint32_t c = 0u; c < 8u; ++c) o[c] = (st_float8){};

    for (uint32_t tile = t_begin + warp; tile < t_end; tile += AW_WARPS) {
        uint32_t k0 = key_lo + tile * 16u;
        uint32_t kp = k0 + nl;
        bool kin = kp <= key_hi;
        // The tile's first key in the ring, once: a 32-bit modulo is a long instruction sequence
        // here, and the 16 keys of a tile need only an add and a wrap.
        uint32_t ring0 = (window != 0u) ? (k0 % max_seq) : k0;
        uint32_t ks = ring0 + nl;
        if (window != 0u && ks >= max_seq) ks -= max_seq;
        uint32_t slot = kv_phys(page_table, ks);
        st_float8 sacc = {};
        #pragma unroll
        for (uint32_t c = 0u; c < 8u; ++c) {
            uint2 kw = kin ? *reinterpret_cast<const uint2*>(kb + static_cast<size_t>(slot) * kv_stride + c * 16u + half_ * 8u) : make_uint2(0u, 0u);
            float4 a = decode_fp8x4(kw.x), b = decode_fp8x4(kw.y);
            st_half8 kf;
            kf[0] = (_Float16)a.x; kf[1] = (_Float16)a.y; kf[2] = (_Float16)a.z; kf[3] = (_Float16)a.w;
            kf[4] = (_Float16)b.x; kf[5] = (_Float16)b.y; kf[6] = (_Float16)b.z; kf[7] = (_Float16)b.w;
            sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(qa[c], kf, sacc);
        }
        float kscale = kin ? ksb[static_cast<size_t>(slot) * n_kv_heads] : 0.0f;
        float vscale = (v_q4 != 0u) ? 1.0f : (kin ? vsb[static_cast<size_t>(slot) * n_kv_heads] : 0.0f);
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
            l_run[i] = l_run[i] * al + pv;
            m_run[i] = mn;
            #pragma unroll
            for (uint32_t c = 0u; c < 8u; ++c) o[c][i] *= al;
            pr[i] = pv * vscale;
        }
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; ++i) s_p[warp][half_ * 8u + i][nl] = (_Float16)pr[i];
        #pragma unroll
        for (uint32_t r = 0u; r < 16u; ++r) {
            uint32_t kr = k0 + r;
            uint32_t vs = ring0 + r;
            if (window != 0u && vs >= max_seq) vs -= max_seq;
            uint32_t vslot = kv_phys(page_table, vs);
            float4 vv;
            if (kr <= key_hi) {
                vv = (v_q4 != 0u)
                    ? kv_load_v_q4(v_cache, v_scales, static_cast<size_t>(vslot) * n_kv_heads + kvh, l)
                    : decode_fp8x4(reinterpret_cast<const uint32_t*>(vb + static_cast<size_t>(vslot) * kv_stride)[l]);
            } else {
                vv = make_float4(0, 0, 0, 0);
            }
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

    #pragma unroll
    for (uint32_t i = 0u; i < 8u; ++i) {
        float s = l_run[i];
        s += __shfl_xor(s, 8, 32); s += __shfl_xor(s, 4, 32); s += __shfl_xor(s, 2, 32); s += __shfl_xor(s, 1, 32);
        if (nl == 0u) { s_m[warp][half_ * 8u + i] = m_run[i]; s_l[warp][half_ * 8u + i] = s; }
        #pragma unroll
        for (uint32_t c = 0u; c < 8u; ++c) s_o[warp][half_ * 8u + i][c * 16u + nl] = o[c][i];
    }
    __syncthreads();
    uint32_t mrow = threadIdx.x >> 3, dpart = (threadIdx.x & 7u) * 16u;
    if (q_base + mrow < nq) {
        uint32_t t, qh, p, lo;
        qinfo(mrow, t, qh, p, lo);
        float mb = -1e30f;
        for (uint32_t w = 0u; w < AW_WARPS; ++w) if (s_l[w][mrow] > 0.0f) mb = fmaxf(mb, s_m[w][mrow]);
        float lb = 0.0f;
        float f[AW_WARPS];
        for (uint32_t w = 0u; w < AW_WARPS; ++w) {
            f[w] = (s_l[w][mrow] > 0.0f) ? exp2f(s_m[w][mrow] - mb) : 0.0f;
            lb += s_l[w][mrow] * f[w];
        }
        size_t pidx = (static_cast<size_t>(t) * n_q_heads + qh) * num_splits + sp;
        if (dpart == 0u) { part_m[pidx] = mb; part_l[pidx] = lb; }
        float* dst = part_o + pidx * 128u + dpart;
        for (uint32_t d = 0u; d < 16u; ++d) {
            float acc = 0.0f;
            for (uint32_t w = 0u; w < AW_WARPS; ++w) acc = fmaf(s_o[w][mrow][dpart + d], f[w], acc);
            dst[d] = acc;
        }
    }
}

// Reduces Split-K partials of smalltide_attn_prefill_wmma_split + sinks.
//   Grid: (n_q_heads, rows, 1), Block: (32, 1, 1)
extern "C" __global__ void smalltide_attn_prefill_wmma_reduce(
    float* __restrict__ out, const float* __restrict__ part_o, const float* __restrict__ part_m,
    const float* __restrict__ part_l, const float* __restrict__ sinks,
    uint32_t n_q_heads, uint32_t num_splits
) {
    const float LOG2E = 1.4426950408889634f;
    uint32_t qh = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    size_t base = (static_cast<size_t>(t) * n_q_heads + qh) * num_splits;
    float sink = (sinks != nullptr) ? sinks[static_cast<size_t>(t) * n_q_heads + qh] * LOG2E : -1e30f;
    float mb = sink;
    for (uint32_t s = 0u; s < num_splits; ++s) {
        if (part_l[base + s] > 0.0f) mb = fmaxf(mb, part_m[base + s]);
    }
    float lb = (sinks != nullptr) ? exp2f(sink - mb) : 0.0f;
    float4 ob = make_float4(0, 0, 0, 0);
    for (uint32_t s = 0u; s < num_splits; ++s) {
        float ls = part_l[base + s];
        if (ls > 0.0f) {
            float f = exp2f(part_m[base + s] - mb);
            lb += ls * f;
            float4 ow = reinterpret_cast<const float4*>(part_o + (base + s) * 128u)[lane];
            ob.x = fmaf(ow.x, f, ob.x); ob.y = fmaf(ow.y, f, ob.y);
            ob.z = fmaf(ow.z, f, ob.z); ob.w = fmaf(ow.w, f, ob.w);
        }
    }
    float inv = (lb > 1e-30f) ? (1.0f / lb) : 0.0f;
    reinterpret_cast<float4*>(out + (static_cast<size_t>(t) * n_q_heads + qh) * 128u)[lane] =
        make_float4(ob.x * inv, ob.y * inv, ob.z * inv, ob.w * inv);
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

// smalltide_gemm_fp8_rows_wmma_x16: high-occupancy FP8 WMMA GEMM (`x` pre-converted to f16).
// 2 warps/block (64 threads) along M, each warp computes 2 M-tiles (32 rows) x 4 N-tiles (64 cols)
// -> 64 M-rows x 64 N-cols per block. Keeps 4x activation reuse across N (fast at M = 1024) AND
// launches 4x more M-blocks than 256-row blocks so M = 256 chunks saturate all 64 CUs.
//   Grid: (ceil(N / 64), ceil(M / 64)), Block: 64
#define ST_GEMM_RG 2u
extern "C" __global__ void smalltide_gemm_fp8_rows_wmma_x16(
    float* __restrict__ y, const _Float16* __restrict__ x, const uint8_t* __restrict__ w_fp8,
    const float* __restrict__ row_scale, uint32_t m_rows, uint32_t n_rows, uint32_t k_cols
) {
    uint32_t warp = threadIdx.x >> 5, l = threadIdx.x & 31u;
    uint32_t m0 = (blockIdx.y * 2u + warp) * (16u * ST_GEMM_RG);
    if (m0 >= m_rows) return;
    uint32_t n0 = blockIdx.x * 64u;
    if (n0 >= n_rows) return;
    const _Float16* xr[ST_GEMM_RG];
    bool a_ok[ST_GEMM_RG];
    #pragma unroll
    for (uint32_t g = 0u; g < ST_GEMM_RG; ++g) {
        uint32_t am = m0 + 16u * g + (l & 15u);
        a_ok[g] = am < m_rows;
        xr[g] = x + static_cast<size_t>(a_ok[g] ? am : 0u) * k_cols;
    }
    const uint8_t* wr[4];
    bool b_ok[4];
    #pragma unroll
    for (uint32_t j = 0u; j < 4u; ++j) {
        uint32_t n = n0 + 16u * j + (l & 15u);
        b_ok[j] = n < n_rows;
        wr[j] = w_fp8 + static_cast<size_t>(b_ok[j] ? n : 0u) * k_cols;
    }
    st_float8 c[ST_GEMM_RG][4] = {};
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    uint32_t hoff = (l >> 4) * 16u;
    for (uint32_t k = 0u; k < k_cols; k += 32u) {
        uint32_t base = k + hoff;
        st_half8 b0[4], b1[4];
        #pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            uint4 wv = b_ok[j] ? *reinterpret_cast<const uint4*>(wr[j] + base) : zero;
            b0[j] = fp8x8_to_half8(make_uint2(wv.x, wv.y));
            b1[j] = fp8x8_to_half8(make_uint2(wv.z, wv.w));
        }
        #pragma unroll
        for (uint32_t g = 0u; g < ST_GEMM_RG; ++g) {
            const uint4* xp = reinterpret_cast<const uint4*>(xr[g] + base);
            st_half8 a0 = as_half8(a_ok[g] ? xp[0] : zero), a1 = as_half8(a_ok[g] ? xp[1] : zero);
            #pragma unroll
            for (uint32_t j = 0u; j < 4u; ++j) {
                c[g][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, b0[j], c[g][j]);
                c[g][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, b1[j], c[g][j]);
            }
        }
    }
    #pragma unroll
    for (uint32_t j = 0u; j < 4u; ++j) {
        uint32_t n = n0 + 16u * j + (l & 15u);
        if (n >= n_rows) continue;
        float sc = (row_scale != nullptr) ? row_scale[n] : 1.0f;
        #pragma unroll
        for (uint32_t g = 0u; g < ST_GEMM_RG; ++g) {
            #pragma unroll
            for (uint32_t i = 0u; i < 8u; ++i) {
                uint32_t mm = m0 + 16u * g + (l >> 4) * 8u + i;
                if (mm < m_rows) y[static_cast<size_t>(mm) * n_rows + n] = c[g][j][i] * sc;
            }
        }
    }
}

// Full-occupancy FP8 WMMA GEMM for small M or narrow N (`x` pre-converted to f16):
// 2 warps/block (32 N-columns/block -> 100% active warps across all CUs, zero dead M-warps)
// and each warp decodes its 16-column FP8 weight tile ONCE for all RG*16 rows.
//   Grid: (ceil(N / 32), ceil(M / (16 * RG)), 1), Block: 64
template <uint32_t RG>
__device__ __forceinline__ void gemm_fp8_rows_wmma_x16_mid(
    float* __restrict__ y, const _Float16* __restrict__ x, const uint8_t* __restrict__ w_fp8,
    const float* __restrict__ row_scale, uint32_t m_rows, uint32_t n_rows, uint32_t k_cols
) {
    uint32_t warp = threadIdx.x >> 5, l = threadIdx.x & 31u;
    uint32_t m0 = blockIdx.y * (16u * RG);
    if (m0 >= m_rows) return;
    uint32_t n0 = (blockIdx.x * 2u + warp) * 16u;
    if (n0 >= n_rows) return;
    uint32_t n = n0 + (l & 15u);
    bool b_ok = n < n_rows;
    const uint8_t* wr = w_fp8 + static_cast<size_t>(b_ok ? n : 0u) * k_cols;
    const _Float16* xr[RG];
    bool a_ok[RG];
    #pragma unroll
    for (uint32_t g = 0u; g < RG; ++g) {
        uint32_t am = m0 + 16u * g + (l & 15u);
        a_ok[g] = am < m_rows;
        xr[g] = x + static_cast<size_t>(a_ok[g] ? am : 0u) * k_cols;
    }
    st_float8 c[RG] = {};
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    uint32_t hoff = (l >> 4) * 16u;
    for (uint32_t k = 0u; k < k_cols; k += 32u) {
        uint32_t base = k + hoff;
        uint4 wv = b_ok ? *reinterpret_cast<const uint4*>(wr + base) : zero;
        st_half8 b0 = fp8x8_to_half8(make_uint2(wv.x, wv.y));
        st_half8 b1 = fp8x8_to_half8(make_uint2(wv.z, wv.w));
        #pragma unroll
        for (uint32_t g = 0u; g < RG; ++g) {
            const uint4* xp = reinterpret_cast<const uint4*>(xr[g] + base);
            st_half8 a0 = as_half8(a_ok[g] ? xp[0] : zero);
            st_half8 a1 = as_half8(a_ok[g] ? xp[1] : zero);
            c[g] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, b0, c[g]);
            c[g] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, b1, c[g]);
        }
    }
    if (!b_ok) return;
    float sc = (row_scale != nullptr) ? row_scale[n] : 1.0f;
    #pragma unroll
    for (uint32_t g = 0u; g < RG; ++g) {
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; ++i) {
            uint32_t mm = m0 + 16u * g + (l >> 4) * 8u + i;
            if (mm < m_rows) y[static_cast<size_t>(mm) * n_rows + n] = c[g][i] * sc;
        }
    }
}

extern "C" __global__ void smalltide_gemm_fp8_rows_wmma_x16_m32(
    float* __restrict__ y, const _Float16* __restrict__ x, const uint8_t* __restrict__ w_fp8,
    const float* __restrict__ row_scale, uint32_t m_rows, uint32_t n_rows, uint32_t k_cols
) { gemm_fp8_rows_wmma_x16_mid<2u>(y, x, w_fp8, row_scale, m_rows, n_rows, k_cols); }

extern "C" __global__ void smalltide_gemm_fp8_rows_wmma_x16_m64(
    float* __restrict__ y, const _Float16* __restrict__ x, const uint8_t* __restrict__ w_fp8,
    const float* __restrict__ row_scale, uint32_t m_rows, uint32_t n_rows, uint32_t k_cols
) { gemm_fp8_rows_wmma_x16_mid<4u>(y, x, w_fp8, row_scale, m_rows, n_rows, k_cols); }

// Same arithmetic per output element as smalltide_gemm_fp8_rows_wmma (k order and WMMA steps
// identical), for m_rows <= 16: one warp per 32-column tile (2 N-tiles reuse A fragment, 2x K-unroll).
//   Grid: (ceil(N / 256)), Block: 256
extern "C" __global__ void smalltide_gemm_fp8_rows_wmma_small(
    float* __restrict__ y, const float* __restrict__ x, const uint8_t* __restrict__ w_fp8,
    const float* __restrict__ row_scale, uint32_t m_rows, uint32_t n_rows, uint32_t k_cols
) {
    uint32_t warp = threadIdx.x >> 5, l = threadIdx.x & 31u;
    uint32_t n0 = (blockIdx.x * 8u + warp) * 32u;
    if (n0 >= n_rows) return;
    uint32_t am = l & 15u;
    bool a_ok = am < m_rows;
    const float* xr = x + static_cast<size_t>(a_ok ? am : 0u) * k_cols;
    uint32_t n_a = n0 + (l & 15u);
    uint32_t n_b = n0 + 16u + (l & 15u);
    bool b_ok0 = n_a < n_rows;
    bool b_ok1 = n_b < n_rows;
    const uint8_t* wr0 = w_fp8 + static_cast<size_t>(b_ok0 ? n_a : 0u) * k_cols;
    const uint8_t* wr1 = w_fp8 + static_cast<size_t>(b_ok1 ? n_b : 0u) * k_cols;
    st_float8 c0 = {}, c1 = {};
    uint32_t hoff = (l >> 4) * 16u;
    auto step32 = [&](uint32_t base, uint4 wv0, uint4 wv1) {
        const float4* xp = reinterpret_cast<const float4*>(xr + base);
        float4 x0 = a_ok ? xp[0] : make_float4(0, 0, 0, 0), x1 = a_ok ? xp[1] : make_float4(0, 0, 0, 0);
        float4 x2 = a_ok ? xp[2] : make_float4(0, 0, 0, 0), x3 = a_ok ? xp[3] : make_float4(0, 0, 0, 0);
        st_half8 a0, a1;
        a0[0] = (_Float16)x0.x; a0[1] = (_Float16)x0.y; a0[2] = (_Float16)x0.z; a0[3] = (_Float16)x0.w;
        a0[4] = (_Float16)x1.x; a0[5] = (_Float16)x1.y; a0[6] = (_Float16)x1.z; a0[7] = (_Float16)x1.w;
        a1[0] = (_Float16)x2.x; a1[1] = (_Float16)x2.y; a1[2] = (_Float16)x2.z; a1[3] = (_Float16)x2.w;
        a1[4] = (_Float16)x3.x; a1[5] = (_Float16)x3.y; a1[6] = (_Float16)x3.z; a1[7] = (_Float16)x3.w;
        c0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, fp8x8_to_half8(make_uint2(wv0.x, wv0.y)), c0);
        c1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, fp8x8_to_half8(make_uint2(wv1.x, wv1.y)), c1);
        c0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, fp8x8_to_half8(make_uint2(wv0.z, wv0.w)), c0);
        c1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, fp8x8_to_half8(make_uint2(wv1.z, wv1.w)), c1);
    };
    uint32_t k = 0u;
    for (; k + 64u <= k_cols; k += 64u) {
        uint32_t b0 = k + hoff, b1 = k + 32u + hoff;
        uint4 w00 = b_ok0 ? *reinterpret_cast<const uint4*>(wr0 + b0) : make_uint4(0u, 0u, 0u, 0u);
        uint4 w10 = b_ok1 ? *reinterpret_cast<const uint4*>(wr1 + b0) : make_uint4(0u, 0u, 0u, 0u);
        uint4 w01 = b_ok0 ? *reinterpret_cast<const uint4*>(wr0 + b1) : make_uint4(0u, 0u, 0u, 0u);
        uint4 w11 = b_ok1 ? *reinterpret_cast<const uint4*>(wr1 + b1) : make_uint4(0u, 0u, 0u, 0u);
        step32(b0, w00, w10);
        step32(b1, w01, w11);
    }
    if (k < k_cols) {
        uint32_t b0 = k + hoff;
        uint4 w00 = b_ok0 ? *reinterpret_cast<const uint4*>(wr0 + b0) : make_uint4(0u, 0u, 0u, 0u);
        uint4 w10 = b_ok1 ? *reinterpret_cast<const uint4*>(wr1 + b0) : make_uint4(0u, 0u, 0u, 0u);
        step32(b0, w00, w10);
    }
    if (b_ok0) {
        float sc0 = (row_scale != nullptr) ? row_scale[n_a] : 1.0f;
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; ++i) {
            uint32_t mm = (l >> 4) * 8u + i;
            if (mm < m_rows) y[static_cast<size_t>(mm) * n_rows + n_a] = c0[i] * sc0;
        }
    }
    if (b_ok1) {
        float sc1 = (row_scale != nullptr) ? row_scale[n_b] : 1.0f;
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; ++i) {
            uint32_t mm = (l >> 4) * 8u + i;
            if (mm < m_rows) y[static_cast<size_t>(mm) * n_rows + n_b] = c1[i] * sc1;
        }
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
// Native BF16 multi-row kernels:
//   - smalltide_gemv_bf16_rows             : 1..=8 rows bandwidth-bound BF16 GEMV
//   - smalltide_gemm_bf16_rows_wmma_small  : <= 16 rows, 2 N-tiles/warp native BF16 WMMA
//   - smalltide_gemm_bf16_rows_wmma_m32/64 : 17..=64 rows, 2 warps/block full-occupancy BF16 WMMA
//   (> 64 rows uses smalltide_vit_gemm_bf16 in vision_ops.hip)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_gemv_bf16_rows(
    float* __restrict__ y, const float* __restrict__ x, const uint16_t* __restrict__ w_bf16,
    uint32_t m_rows, uint32_t n_rows, uint32_t k_cols
) {
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t row = blockIdx.x * 4u + warp;
    if (row >= n_rows) return;
    const uint4* wv = reinterpret_cast<const uint4*>(w_bf16 + static_cast<size_t>(row) * k_cols);
    uint32_t k8 = k_cols >> 3;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    auto dot8 = [&](uint4 w8, uint32_t c) {
        float4 d0 = decode_bf16x4(make_uint2(w8.x, w8.y));
        float4 d1 = decode_bf16x4(make_uint2(w8.z, w8.w));
        #pragma unroll
        for (uint32_t m = 0u; m < 8u; ++m) {
            if (m < m_rows) {
                const float4* xv = reinterpret_cast<const float4*>(x + static_cast<size_t>(m) * k_cols) + (c << 1);
                float4 x0 = xv[0], x1 = xv[1];
                float a = acc[m];
                a = fmaf(d0.x, x0.x, a); a = fmaf(d0.y, x0.y, a); a = fmaf(d0.z, x0.z, a); a = fmaf(d0.w, x0.w, a);
                a = fmaf(d1.x, x1.x, a); a = fmaf(d1.y, x1.y, a); a = fmaf(d1.z, x1.z, a); a = fmaf(d1.w, x1.w, a);
                acc[m] = a;
            }
        }
    };
    uint32_t c = lane;
    for (; c + 32u < k8; c += 64u) {
        uint4 wa = wv[c];
        uint4 wb = wv[c + 32u];
        dot8(wa, c);
        dot8(wb, c + 32u);
    }
    if (c < k8) dot8(wv[c], c);
    #pragma unroll
    for (uint32_t m = 0u; m < 8u; ++m) {
        if (m < m_rows) {
            float sum = wave32_sum(acc[m]);
            if (lane == 0u) y[static_cast<size_t>(m) * n_rows + row] = sum;
        }
    }
}

extern "C" __global__ void smalltide_gemm_bf16_rows_wmma_small(
    float* __restrict__ y, const float* __restrict__ x, const uint16_t* __restrict__ w_bf16,
    uint32_t m_rows, uint32_t n_rows, uint32_t k_cols
) {
    uint32_t warp = threadIdx.x >> 5, l = threadIdx.x & 31u;
    uint32_t n0 = (blockIdx.x * 8u + warp) * 32u;
    if (n0 >= n_rows) return;
    uint32_t am = l & 15u;
    bool a_ok = am < m_rows;
    const float* xr = x + static_cast<size_t>(a_ok ? am : 0u) * k_cols;
    uint32_t n_a = n0 + (l & 15u);
    uint32_t n_b = n0 + 16u + (l & 15u);
    bool b_ok0 = n_a < n_rows;
    bool b_ok1 = n_b < n_rows;
    const uint16_t* wr0 = w_bf16 + static_cast<size_t>(b_ok0 ? n_a : 0u) * k_cols;
    const uint16_t* wr1 = w_bf16 + static_cast<size_t>(b_ok1 ? n_b : 0u) * k_cols;
    st_float8 c0 = {}, c1 = {};
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    uint32_t hoff = (l >> 4) * 16u;
    for (uint32_t k = 0u; k < k_cols; k += 32u) {
        uint32_t base = k + hoff;
        const float4* xp = reinterpret_cast<const float4*>(xr + base);
        float4 x0 = a_ok ? xp[0] : make_float4(0, 0, 0, 0), x1 = a_ok ? xp[1] : make_float4(0, 0, 0, 0);
        float4 x2 = a_ok ? xp[2] : make_float4(0, 0, 0, 0), x3 = a_ok ? xp[3] : make_float4(0, 0, 0, 0);
        st_bf16_8 a0, a1;
        a0[0] = f32_to_bf16_rne(x0.x); a0[1] = f32_to_bf16_rne(x0.y); a0[2] = f32_to_bf16_rne(x0.z); a0[3] = f32_to_bf16_rne(x0.w);
        a0[4] = f32_to_bf16_rne(x1.x); a0[5] = f32_to_bf16_rne(x1.y); a0[6] = f32_to_bf16_rne(x1.z); a0[7] = f32_to_bf16_rne(x1.w);
        a1[0] = f32_to_bf16_rne(x2.x); a1[1] = f32_to_bf16_rne(x2.y); a1[2] = f32_to_bf16_rne(x2.z); a1[3] = f32_to_bf16_rne(x2.w);
        a1[4] = f32_to_bf16_rne(x3.x); a1[5] = f32_to_bf16_rne(x3.y); a1[6] = f32_to_bf16_rne(x3.z); a1[7] = f32_to_bf16_rne(x3.w);
        const uint4* wp0 = reinterpret_cast<const uint4*>(wr0 + base);
        const uint4* wp1 = reinterpret_cast<const uint4*>(wr1 + base);
        c0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0, as_bf16_8(b_ok0 ? wp0[0] : zero), c0);
        c1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0, as_bf16_8(b_ok1 ? wp1[0] : zero), c1);
        c0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1, as_bf16_8(b_ok0 ? wp0[1] : zero), c0);
        c1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1, as_bf16_8(b_ok1 ? wp1[1] : zero), c1);
    }
    if (b_ok0) {
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; ++i) {
            uint32_t mm = (l >> 4) * 8u + i;
            if (mm < m_rows) y[static_cast<size_t>(mm) * n_rows + n_a] = c0[i];
        }
    }
    if (b_ok1) {
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; ++i) {
            uint32_t mm = (l >> 4) * 8u + i;
            if (mm < m_rows) y[static_cast<size_t>(mm) * n_rows + n_b] = c1[i];
        }
    }
}

template <uint32_t RG>
__device__ __forceinline__ void gemm_bf16_rows_wmma_mid(
    float* __restrict__ y, const float* __restrict__ x, const uint16_t* __restrict__ w_bf16,
    uint32_t m_rows, uint32_t n_rows, uint32_t k_cols
) {
    uint32_t warp = threadIdx.x >> 5, l = threadIdx.x & 31u;
    uint32_t n0 = (blockIdx.x * 2u + warp) * 16u;
    if (n0 >= n_rows) return;
    uint32_t n = n0 + (l & 15u);
    bool b_ok = n < n_rows;
    const uint16_t* wr = w_bf16 + static_cast<size_t>(b_ok ? n : 0u) * k_cols;
    const float* xr[RG];
    bool a_ok[RG];
    #pragma unroll
    for (uint32_t g = 0u; g < RG; ++g) {
        uint32_t am = 16u * g + (l & 15u);
        a_ok[g] = am < m_rows;
        xr[g] = x + static_cast<size_t>(a_ok[g] ? am : 0u) * k_cols;
    }
    st_float8 c[RG] = {};
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    uint32_t hoff = (l >> 4) * 16u;
    for (uint32_t k = 0u; k < k_cols; k += 32u) {
        uint32_t base = k + hoff;
        const uint4* wp = reinterpret_cast<const uint4*>(wr + base);
        st_bf16_8 b0 = as_bf16_8(b_ok ? wp[0] : zero);
        st_bf16_8 b1 = as_bf16_8(b_ok ? wp[1] : zero);
        #pragma unroll
        for (uint32_t g = 0u; g < RG; ++g) {
            const float4* xp = reinterpret_cast<const float4*>(xr[g] + base);
            float4 x0 = a_ok[g] ? xp[0] : make_float4(0, 0, 0, 0), x1 = a_ok[g] ? xp[1] : make_float4(0, 0, 0, 0);
            float4 x2 = a_ok[g] ? xp[2] : make_float4(0, 0, 0, 0), x3 = a_ok[g] ? xp[3] : make_float4(0, 0, 0, 0);
            st_bf16_8 a0, a1;
            a0[0] = f32_to_bf16_rne(x0.x); a0[1] = f32_to_bf16_rne(x0.y); a0[2] = f32_to_bf16_rne(x0.z); a0[3] = f32_to_bf16_rne(x0.w);
            a0[4] = f32_to_bf16_rne(x1.x); a0[5] = f32_to_bf16_rne(x1.y); a0[6] = f32_to_bf16_rne(x1.z); a0[7] = f32_to_bf16_rne(x1.w);
            a1[0] = f32_to_bf16_rne(x2.x); a1[1] = f32_to_bf16_rne(x2.y); a1[2] = f32_to_bf16_rne(x2.z); a1[3] = f32_to_bf16_rne(x2.w);
            a1[4] = f32_to_bf16_rne(x3.x); a1[5] = f32_to_bf16_rne(x3.y); a1[6] = f32_to_bf16_rne(x3.z); a1[7] = f32_to_bf16_rne(x3.w);
            c[g] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0, b0, c[g]);
            c[g] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1, b1, c[g]);
        }
    }
    if (!b_ok) return;
    #pragma unroll
    for (uint32_t g = 0u; g < RG; ++g) {
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; ++i) {
            uint32_t mm = 16u * g + (l >> 4) * 8u + i;
            if (mm < m_rows) y[static_cast<size_t>(mm) * n_rows + n] = c[g][i];
        }
    }
}

extern "C" __global__ void smalltide_gemm_bf16_rows_wmma_m32(
    float* __restrict__ y, const float* __restrict__ x, const uint16_t* __restrict__ w_bf16,
    uint32_t m_rows, uint32_t n_rows, uint32_t k_cols
) { gemm_bf16_rows_wmma_mid<2u>(y, x, w_bf16, m_rows, n_rows, k_cols); }

extern "C" __global__ void smalltide_gemm_bf16_rows_wmma_m64(
    float* __restrict__ y, const float* __restrict__ x, const uint16_t* __restrict__ w_bf16,
    uint32_t m_rows, uint32_t n_rows, uint32_t k_cols
) { gemm_bf16_rows_wmma_mid<4u>(y, x, w_bf16, m_rows, n_rows, k_cols); }

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

// Parallel greedy argmax of `rows` rows of `n` f32:
// Pass 1: Grid (16, rows, 1), Block 256 -> 16 (val, idx) partials per row.
extern "C" __global__ void smalltide_argmax_rows_part(
    float* __restrict__ pval, uint32_t* __restrict__ pidx, const float* __restrict__ x, uint32_t n
) {
    uint32_t r = blockIdx.y;
    const float* row = x + static_cast<size_t>(r) * n;
    uint32_t stride = gridDim.x * blockDim.x;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float bv = -1e30f;
    uint32_t bi = 0xFFFFFFFFu;
    for (uint32_t i = tid; i < n; i += stride) {
        float v = row[i];
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor(bv, o, 32);
        uint32_t oi = __shfl_xor(bi, o, 32);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sv[8];
    __shared__ uint32_t si[8];
    uint32_t w = threadIdx.x >> 5, l = threadIdx.x & 31u;
    if (l == 0u) { sv[w] = bv; si[w] = bi; }
    __syncthreads();
    if (w == 0u && l == 0u) {
        bv = sv[0]; bi = si[0];
        for (uint32_t k = 1u; k < (blockDim.x >> 5); ++k) {
            if (sv[k] > bv || (sv[k] == bv && si[k] < bi)) { bv = sv[k]; bi = si[k]; }
        }
        size_t o = static_cast<size_t>(r) * gridDim.x + blockIdx.x;
        pval[o] = bv;
        pidx[o] = bi;
    }
}

// Pass 2: Grid (rows, 1, 1), Block 32 -> reduces `nparts` (<= 32) partials per row into out[r].
extern "C" __global__ void smalltide_argmax_rows_reduce(
    uint32_t* __restrict__ out, const float* __restrict__ pval, const uint32_t* __restrict__ pidx, uint32_t nparts
) {
    uint32_t r = blockIdx.x, l = threadIdx.x;
    size_t base = static_cast<size_t>(r) * nparts;
    float bv = (l < nparts) ? pval[base + l] : -1e30f;
    uint32_t bi = (l < nparts) ? pidx[base + l] : 0xFFFFFFFFu;
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor(bv, o, 32);
        uint32_t oi = __shfl_xor(bi, o, 32);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (l == 0u) out[r] = bi;
}

// Pass 2b: reduces `nparts` partials into `row_max[r]` and zeroes `counts[r]` for logits filtering.
// Grid: (rows, 1, 1), Block: 32
extern "C" __global__ void smalltide_argmax_rows_reduce_max(
    float* __restrict__ row_max, uint32_t* __restrict__ counts, const float* __restrict__ pval, uint32_t nparts
) {
    uint32_t r = blockIdx.x, l = threadIdx.x;
    size_t base = static_cast<size_t>(r) * nparts;
    float bv = (l < nparts) ? pval[base + l] : -1e30f;
    for (int o = 16; o > 0; o >>= 1) bv = fmaxf(bv, __shfl_xor(bv, o, 32));
    if (l == 0u) {
        row_max[r] = bv;
        counts[r] = 0u;
    }
}

// Extracts all `(idx, val)` entries per row with `val >= row_max[r] - margin` into compact `[rows][cap]`
// buffers (typically ~20-80 entries/row at T=0.6, shrinking verify D2H from 6.4 MB to 16 KB).
// Grid: (ceil(n / 1024), rows, 1), Block: 256
extern "C" __global__ void smalltide_logits_filter_rows(
    uint32_t* __restrict__ out_idx, float* __restrict__ out_val, uint32_t* __restrict__ counts,
    const float* __restrict__ row_max, const float* __restrict__ x, uint32_t n, float margin, uint32_t cap
) {
    uint32_t r = blockIdx.y;
    float cutoff = row_max[r] - margin;
    const float* row = x + static_cast<size_t>(r) * n;
    uint32_t base = blockIdx.x * 1024u + threadIdx.x;
    #pragma unroll
    for (uint32_t j = 0u; j < 4u; ++j) {
        uint32_t i = base + j * 256u;
        if (i < n) {
            float v = row[i];
            if (v >= cutoff) {
                uint32_t pos = atomicAdd(&counts[r], 1u);
                if (pos < cap) {
                    size_t o = static_cast<size_t>(r) * cap + pos;
                    out_idx[o] = i;
                    out_val[o] = v;
                }
            }
        }
    }
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

// ===========================================================================
// DSpark optimized kernels (FP8 fold, split+fused RNN, grouped Split-K FP8 attn, fused norms).
// ===========================================================================

// Quantizes fold[R][V] f32 -> fold_fp8[R][V_pad] E4M3FN + fold_scales[R] f32 (load time).
//   Grid: (R, 1, 1), Block: (1024, 1, 1)
extern "C" __global__ void smalltide_dspark_quant_fold_fp8(
    uint8_t* __restrict__ dst, float* __restrict__ scales, const float* __restrict__ src,
    uint32_t V, uint32_t V_pad
) {
    __shared__ float s_mx[32];
    __shared__ float s_inv;
    uint32_t r = blockIdx.x, t = threadIdx.x;
    const float* row = src + static_cast<size_t>(r) * V;
    float mx = 0.0f;
    for (uint32_t v = t; v < V; v += blockDim.x) mx = fmaxf(mx, fabsf(row[v]));
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
    if ((t & 31u) == 0u) s_mx[t >> 5] = mx;
    __syncthreads();
    if (t < 32u) {
        float m = s_mx[t];
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
        if (t == 0u) {
            float sc = (m > 1e-12f) ? (m / 448.0f) : 1.0f;
            scales[r] = sc;
            s_inv = (m > 1e-12f) ? (448.0f / m) : 0.0f;
        }
    }
    __syncthreads();
    float inv = s_inv;
    uint8_t* out = dst + static_cast<size_t>(r) * V_pad;
    for (uint32_t v = t; v < V_pad; v += blockDim.x) {
        out[v] = (v < V) ? encode_fp8_e4m3fn(row[v] * inv) : 0u;
    }
}

__device__ __forceinline__ void merge_top2(
    float& v1, uint32_t& i1, float& v2,
    float ov1, uint32_t oi1, float ov2
) {
    if (oi1 == 0xFFFFFFFFu || oi1 == i1) return;
    if (ov1 > v1 || (ov1 == v1 && oi1 < i1)) {
        v2 = fmaxf(v1, ov2);
        v1 = ov1;
        i1 = oi1;
    } else if (ov1 > v2) {
        v2 = ov1;
    }
}

// Coalesced 4-vocab-per-thread FP8 fold matvec + block top-2 argmax (drops monotonic tanhf!).
// Writes top-1 (pmax, pidx) and runner-up (pmax2) per block so final reduction outputs logit margin.
//   Grid: (ceil(V / 1024), 1, 1), Block: (256, 1, 1)
extern "C" __global__ void smalltide_dspark_logits_argmax_fp8(
    float* __restrict__ pmax, uint32_t* __restrict__ pidx, float* __restrict__ pmax2,
    const float* __restrict__ base,
    const float* __restrict__ fold_b, const uint8_t* __restrict__ fold_fp8,
    const float* __restrict__ fold_scales, const float* __restrict__ latent,
    uint32_t V, uint32_t V_pad, uint32_t R
) {
    __shared__ float s_lat[256];
    __shared__ float s_v1[8], s_v2[8];
    __shared__ uint32_t s_i1[8];
    for (uint32_t i = threadIdx.x; i < R; i += blockDim.x) s_lat[i] = latent[i] * fold_scales[i];
    __syncthreads();

    uint32_t v0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    float y1 = -1e30f, y2 = -1e30f;
    uint32_t bi = 0xFFFFFFFFu;
    if (v0 < V) {
        float a0 = (v0 + 0u < V) ? (base[v0 + 0u] + fold_b[v0 + 0u]) : -1e30f;
        float a1 = (v0 + 1u < V) ? (base[v0 + 1u] + fold_b[v0 + 1u]) : -1e30f;
        float a2 = (v0 + 2u < V) ? (base[v0 + 2u] + fold_b[v0 + 2u]) : -1e30f;
        float a3 = (v0 + 3u < V) ? (base[v0 + 3u] + fold_b[v0 + 3u]) : -1e30f;

        size_t stride4 = static_cast<size_t>(V_pad >> 2);
        const uint32_t* fp = reinterpret_cast<const uint32_t*>(fold_fp8) + (v0 >> 2);
        for (uint32_t r = 0u; r < R; r += 4u) {
            uint32_t w0 = fp[0];
            uint32_t w1 = fp[stride4];
            uint32_t w2 = fp[2u * stride4];
            uint32_t w3 = fp[3u * stride4];
            fp += 4u * stride4;
            float l0 = s_lat[r + 0u], l1 = s_lat[r + 1u], l2 = s_lat[r + 2u], l3 = s_lat[r + 3u];
            float4 d0 = decode_fp8x4(w0);
            a0 = fmaf(l0, d0.x, a0); a1 = fmaf(l0, d0.y, a1); a2 = fmaf(l0, d0.z, a2); a3 = fmaf(l0, d0.w, a3);
            float4 d1 = decode_fp8x4(w1);
            a0 = fmaf(l1, d1.x, a0); a1 = fmaf(l1, d1.y, a1); a2 = fmaf(l1, d1.z, a2); a3 = fmaf(l1, d1.w, a3);
            float4 d2 = decode_fp8x4(w2);
            a0 = fmaf(l2, d2.x, a0); a1 = fmaf(l2, d2.y, a1); a2 = fmaf(l2, d2.z, a2); a3 = fmaf(l2, d2.w, a3);
            float4 d3 = decode_fp8x4(w3);
            a0 = fmaf(l3, d3.x, a0); a1 = fmaf(l3, d3.y, a1); a2 = fmaf(l3, d3.z, a2); a3 = fmaf(l3, d3.w, a3);
        }
        y1 = a0; bi = v0;
        if (v0 + 1u < V) merge_top2(y1, bi, y2, a1, v0 + 1u, -1e30f);
        if (v0 + 2u < V) merge_top2(y1, bi, y2, a2, v0 + 2u, -1e30f);
        if (v0 + 3u < V) merge_top2(y1, bi, y2, a3, v0 + 3u, -1e30f);
    }
    for (int o = 16; o > 0; o >>= 1) {
        float ov1 = __shfl_xor(y1, o, 32);
        uint32_t oi1 = __shfl_xor(bi, o, 32);
        float ov2 = __shfl_xor(y2, o, 32);
        merge_top2(y1, bi, y2, ov1, oi1, ov2);
    }
    uint32_t w = threadIdx.x >> 5;
    if ((threadIdx.x & 31u) == 0u) { s_v1[w] = y1; s_i1[w] = bi; s_v2[w] = y2; }
    __syncthreads();
    if (threadIdx.x == 0u) {
        float bv1 = s_v1[0], bv2 = s_v2[0];
        uint32_t bb = s_i1[0];
        for (uint32_t k = 1u; k < (blockDim.x >> 5); ++k) {
            merge_top2(bv1, bb, bv2, s_v1[k], s_i1[k], s_v2[k]);
        }
        pmax[blockIdx.x] = bv1;
        pidx[blockIdx.x] = bb;
        pmax2[blockIdx.x] = bv2;
    }
}

// Final top-2 reduction: writes drafts[step] = top1_idx and margins[step] = top1_val - top2_val.
//   Grid: (1, 1, 1), Block: (256, 1, 1)
extern "C" __global__ void smalltide_dspark_argmax_final_margin(
    uint32_t* __restrict__ drafts, float* __restrict__ margins, uint32_t step,
    const float* __restrict__ pmax, const uint32_t* __restrict__ pidx, const float* __restrict__ pmax2, uint32_t n
) {
    float bv1 = -1e30f, bv2 = -1e30f;
    uint32_t bi = 0xFFFFFFFFu;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        merge_top2(bv1, bi, bv2, pmax[i], pidx[i], pmax2[i]);
    }
    for (int o = 16; o > 0; o >>= 1) {
        float ov1 = __shfl_xor(bv1, o, 32);
        uint32_t oi1 = __shfl_xor(bi, o, 32);
        float ov2 = __shfl_xor(bv2, o, 32);
        merge_top2(bv1, bi, bv2, ov1, oi1, ov2);
    }
    __shared__ float sv1[8], sv2[8];
    __shared__ uint32_t si1[8];
    uint32_t w = threadIdx.x >> 5, l = threadIdx.x & 31u;
    if (l == 0u) { sv1[w] = bv1; si1[w] = bi; sv2[w] = bv2; }
    __syncthreads();
    if (w == 0u && l == 0u) {
        bv1 = sv1[0]; bi = si1[0]; bv2 = sv2[0];
        for (uint32_t k = 1u; k < (blockDim.x >> 5); ++k) {
            merge_top2(bv1, bi, bv2, sv1[k], si1[k], sv2[k]);
        }
        drafts[step] = bi;
        margins[step] = bv1 - bv2;
    }
}

// Fused Markov RNN step: gathers [state | bf16(w1[prev])] into LDS, multiplies by f16 W_dyn
// [3R x 2R] (786 KB in L2), adds precomputed (frnn_proj + rnn_b) and hb_proj[step], updates state/latent.
//   Grid: (ceil(R / 8), 1, 1), Block: (256, 1, 1) (8 warps, 1 warp per output i in 0..R)
extern "C" __global__ void smalltide_dspark_rnn_step(
    float* __restrict__ state, float* __restrict__ latent, const uint16_t* __restrict__ w1,
    const uint32_t* __restrict__ drafts, uint32_t step, uint32_t anchor,
    const _Float16* __restrict__ w_dyn, const float* __restrict__ frnn_proj,
    const float* __restrict__ hb_proj, const float* __restrict__ rnn_b, uint32_t R
) {
    __shared__ float s_dyn[512];
    uint32_t prev = (step == 0u) ? anchor : drafts[step - 1u];
    for (uint32_t t = threadIdx.x; t < R; t += blockDim.x) {
        s_dyn[t] = state[t];
        s_dyn[R + t] = __uint_as_float(static_cast<uint32_t>(w1[static_cast<size_t>(prev) * R + t]) << 16);
    }
    __syncthreads();
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t i = blockIdx.x * 8u + warp;
    if (i >= R) return;
    uint32_t cols = 2u * R;
    const _Float16* wg = w_dyn + static_cast<size_t>(i) * cols;
    const _Float16* wc = w_dyn + static_cast<size_t>(R + i) * cols;
    const _Float16* wo = w_dyn + static_cast<size_t>(2u * R + i) * cols;
    float ag = 0.0f, ac = 0.0f, ao = 0.0f;
    for (uint32_t c = lane * 8u; c < cols; c += 256u) {
        float4 x0 = *reinterpret_cast<const float4*>(s_dyn + c);
        float4 x1 = *reinterpret_cast<const float4*>(s_dyn + c + 4u);
        st_half8 hg = *reinterpret_cast<const st_half8*>(wg + c);
        st_half8 hc = *reinterpret_cast<const st_half8*>(wc + c);
        st_half8 ho = *reinterpret_cast<const st_half8*>(wo + c);
        ag = fmaf((float)hg[0], x0.x, fmaf((float)hg[1], x0.y, fmaf((float)hg[2], x0.z, fmaf((float)hg[3], x0.w, ag))));
        ag = fmaf((float)hg[4], x1.x, fmaf((float)hg[5], x1.y, fmaf((float)hg[6], x1.z, fmaf((float)hg[7], x1.w, ag))));
        ac = fmaf((float)hc[0], x0.x, fmaf((float)hc[1], x0.y, fmaf((float)hc[2], x0.z, fmaf((float)hc[3], x0.w, ac))));
        ac = fmaf((float)hc[4], x1.x, fmaf((float)hc[5], x1.y, fmaf((float)hc[6], x1.z, fmaf((float)hc[7], x1.w, ac))));
        ao = fmaf((float)ho[0], x0.x, fmaf((float)ho[1], x0.y, fmaf((float)ho[2], x0.z, fmaf((float)ho[3], x0.w, ao))));
        ao = fmaf((float)ho[4], x1.x, fmaf((float)ho[5], x1.y, fmaf((float)ho[6], x1.z, fmaf((float)ho[7], x1.w, ao))));
    }
    ag = wave32_sum(ag);
    ac = wave32_sum(ac);
    ao = wave32_sum(ao);
    if (lane == 0u) {
        float g = ag + frnn_proj[i] + hb_proj[i] + rnn_b[i];
        float c = ac + frnn_proj[R + i] + hb_proj[R + i] + rnn_b[R + i];
        float o = ao + frnn_proj[2u * R + i] + hb_proj[2u * R + i] + rnn_b[2u * R + i];
        float gate = 1.0f / (1.0f + __expf(-g));
        latent[i] = tanhf(o);
        state[i] = gate * s_dyn[i] + (1.0f - gate) * tanhf(c);
    }
}

// Context K/V store (FP8 E4M3FN + per-head-pos f32 scales): rows of k / v [rows][nkv*128].
// `page_table` (nullable): the sequence's pages in the shared KV pool (see kv_phys).
//   Grid: (rows, nkv, 1), Block: (128, 1, 1)
extern "C" __global__ void smalltide_dspark_ctx_store_fp8(
    uint8_t* __restrict__ kc, float* __restrict__ ks, uint8_t* __restrict__ vc, float* __restrict__ vs,
    const float* __restrict__ k, const float* __restrict__ v, uint32_t pos0, uint32_t ring, uint32_t nkv,
    uint32_t v_q4, const uint32_t* __restrict__ page_table
) {
    __shared__ float sk[4], sv[4];
    uint32_t r = blockIdx.x, kvh = blockIdx.y, d = threadIdx.x;
    uint32_t pos = pos0 + r;
    uint32_t slot = kv_phys(page_table, (ring != 0u) ? (pos % ring) : pos);
    size_t src_off = (static_cast<size_t>(r) * nkv + kvh) * 128u + d;
    float kv = k[src_off], vv = v[src_off];
    float mk = fabsf(kv), mv = fabsf(vv);
    for (int o = 16; o > 0; o >>= 1) {
        mk = fmaxf(mk, __shfl_xor(mk, o, 32));
        mv = fmaxf(mv, __shfl_xor(mv, o, 32));
    }
    if ((d & 31u) == 0u) { sk[d >> 5] = mk; sv[d >> 5] = mv; }
    __syncthreads();
    float kmax = fmaxf(fmaxf(sk[0], sk[1]), fmaxf(sk[2], sk[3]));
    float ksc = (kmax > 1e-12f) ? (kmax / 448.0f) : 1.0f;
    size_t dst_head = static_cast<size_t>(slot) * nkv + kvh;
    if (d == 0u) ks[dst_head] = ksc;
    kc[dst_head * 128u + d] = encode_fp8_e4m3fn(kv / ksc);
    if (v_q4 != 0u) {
        kv_store_v_q4(vc, vs, dst_head, d, vv);
    } else {
        float vmax = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3]));
        float vsc = (vmax > 1e-12f) ? (vmax / 448.0f) : 1.0f;
        if (d == 0u) vs[dst_head] = vsc;
        vc[dst_head * 128u + d] = encode_fp8_e4m3fn(vv / vsc);
    }
}

// Grouped Split-K FP8 drafter attention: 1 block per (kv_head, split), 8 warps = 8 block rows,
// each warp computes all 3 GQA query heads against LDS-staged 16-token FP8 K/V tiles (48x less KV
// traffic than per-(head,row) f16 attention!). When num_splits == 1 writes directly to `out`.
//   Grid: (nkv, num_splits, 1), Block: (256, 1, 1)   (nh = 3 * nkv, nblk = 8)
extern "C" __global__ void smalltide_dspark_attn_fp8_split(
    float* __restrict__ out, float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
    const float* __restrict__ q, const float* __restrict__ kb, const float* __restrict__ vb,
    const uint8_t* __restrict__ kc, const float* __restrict__ ks,
    const uint8_t* __restrict__ vc, const float* __restrict__ vs,
    uint32_t P, uint32_t lo, uint32_t ring, uint32_t nh, uint32_t nkv, uint32_t nblk,
    uint32_t num_splits, float scale, uint32_t v_q4, const uint32_t* __restrict__ page_table
) {
    uint32_t kvh = blockIdx.x, sp = blockIdx.y;
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t r = warp; // 8 warps = 8 block rows
    uint32_t gqa = nh / nkv; // 3
    uint32_t qh0 = kvh * gqa;
    uint32_t kvd = nkv * 128u;

    // Hold all 3 GQA queries for block row `r` in registers (zero LDS traffic for Q).
    float4 qv[3];
    float m[3] = {-1e30f, -1e30f, -1e30f};
    float l[3] = {0.0f, 0.0f, 0.0f};
    float4 o[3] = {make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0)};
    #pragma unroll
    for (uint32_t g = 0u; g < 3u; ++g) {
        qv[g] = reinterpret_cast<const float4*>(q + (static_cast<size_t>(r) * nh + qh0 + g) * 128u)[lane];
    }

    uint32_t nctx = (P > lo) ? (P - lo) : 0u;
    uint32_t per = (nctx + num_splits - 1u) / num_splits;
    uint32_t c_start = sp * per;
    uint32_t c_end = min(nctx, c_start + per);

    __shared__ uint32_t s_k8[16][32];
    __shared__ uint32_t s_v8[16][32];
    __shared__ float s_ks[16], s_vs[16];

    uint32_t lt = threadIdx.x >> 4;   // 0..15 token in tile
    uint32_t lw = threadIdx.x & 15u;  // 0..15 -> 2 uint32_t (8 bytes) per thread
    for (uint32_t j0 = c_start; j0 < c_end; j0 += 16u) {
        uint32_t tlen = min(16u, c_end - j0);
        if (lt < tlen) {
            uint32_t pos = lo + j0 + lt;
            uint32_t slot = kv_phys(page_table, (ring != 0u) ? (pos % ring) : pos);
            size_t head_idx = static_cast<size_t>(slot) * nkv + kvh;
            uint2 kw = reinterpret_cast<const uint2*>(kc + head_idx * 128u)[lw];
            s_k8[lt][lw * 2u + 0u] = kw.x; s_k8[lt][lw * 2u + 1u] = kw.y;
            if (v_q4 != 0u) {
                s_v8[lt][lw] = reinterpret_cast<const uint32_t*>(vc + head_idx * 64u)[lw];
                if (lw < 4u) {
                    s_v8[lt][16u + lw] = reinterpret_cast<const uint32_t*>(vs)[head_idx * 4u + lw];
                }
                if (lw == 0u) {
                    s_ks[lt] = ks[head_idx] * scale;
                    s_vs[lt] = 1.0f;
                }
            } else {
                uint2 vw = reinterpret_cast<const uint2*>(vc + head_idx * 128u)[lw];
                s_v8[lt][lw * 2u + 0u] = vw.x; s_v8[lt][lw * 2u + 1u] = vw.y;
                if (lw == 0u) {
                    s_ks[lt] = ks[head_idx] * scale;
                    s_vs[lt] = vs[head_idx];
                }
            }
        }
        __syncthreads();
        for (uint32_t t = 0u; t < tlen; ++t) {
            float4 kv4 = decode_fp8x4(s_k8[t][lane]);
            float4 vv4;
            if (v_q4 != 0u) {
                uint32_t w = s_v8[t][lane >> 1];
                uint32_t v2 = (lane & 1u) ? (w >> 16) : (w & 0xFFFFu);
                uint32_t pku = s_v8[t][16u + (lane >> 3)];
                KvV4Scale pk;
                static_assert(sizeof(KvV4Scale) == 4, "KvV4Scale must be 4 bytes");
                __builtin_memcpy(&pk, &pku, 4);
                float vlo = static_cast<float>(pk.lo);
                float vsc_g = static_cast<float>(pk.sc);
                vv4 = make_float4(
                    fmaf(static_cast<float>(v2 & 0xFu), vsc_g, vlo),
                    fmaf(static_cast<float>((v2 >> 4) & 0xFu), vsc_g, vlo),
                    fmaf(static_cast<float>((v2 >> 8) & 0xFu), vsc_g, vlo),
                    fmaf(static_cast<float>((v2 >> 12) & 0xFu), vsc_g, vlo)
                );
            } else {
                vv4 = decode_fp8x4(s_v8[t][lane]);
            }
            float ksc = s_ks[t], vsc = s_vs[t];
            #pragma unroll
            for (uint32_t g = 0u; g < 3u; ++g) {
                float sc = wave32_sum(qv[g].x * kv4.x + qv[g].y * kv4.y + qv[g].z * kv4.z + qv[g].w * kv4.w) * ksc;
                float mn = fmaxf(m[g], sc);
                float al = __expf(m[g] - mn), pr = __expf(sc - mn);
                l[g] = l[g] * al + pr;
                m[g] = mn;
                float pv = pr * vsc;
                o[g].x = fmaf(vv4.x, pv, o[g].x * al); o[g].y = fmaf(vv4.y, pv, o[g].y * al);
                o[g].z = fmaf(vv4.z, pv, o[g].z * al); o[g].w = fmaf(vv4.w, pv, o[g].w * al);
            }
        }
        __syncthreads();
    }

    // Last split attends to the `nblk` F32 block tokens (non-causal).
    if (sp == num_splits - 1u) {
        for (uint32_t br = 0u; br < nblk; ++br) {
            float4 kv4 = reinterpret_cast<const float4*>(kb + static_cast<size_t>(br) * kvd + kvh * 128u)[lane];
            float4 vv4 = reinterpret_cast<const float4*>(vb + static_cast<size_t>(br) * kvd + kvh * 128u)[lane];
            #pragma unroll
            for (uint32_t g = 0u; g < 3u; ++g) {
                float sc = wave32_sum(qv[g].x * kv4.x + qv[g].y * kv4.y + qv[g].z * kv4.z + qv[g].w * kv4.w) * scale;
                float mn = fmaxf(m[g], sc);
                float al = __expf(m[g] - mn), pr = __expf(sc - mn);
                l[g] = l[g] * al + pr;
                m[g] = mn;
                o[g].x = fmaf(vv4.x, pr, o[g].x * al); o[g].y = fmaf(vv4.y, pr, o[g].y * al);
                o[g].z = fmaf(vv4.z, pr, o[g].z * al); o[g].w = fmaf(vv4.w, pr, o[g].w * al);
            }
        }
    }

    if (num_splits == 1u) {
        #pragma unroll
        for (uint32_t g = 0u; g < 3u; ++g) {
            float inv = (l[g] > 0.0f) ? (1.0f / l[g]) : 0.0f;
            reinterpret_cast<float4*>(out + (static_cast<size_t>(r) * nh + qh0 + g) * 128u)[lane] =
                make_float4(o[g].x * inv, o[g].y * inv, o[g].z * inv, o[g].w * inv);
        }
    } else {
        #pragma unroll
        for (uint32_t g = 0u; g < 3u; ++g) {
            size_t pidx = (static_cast<size_t>(r) * nh + qh0 + g) * num_splits + sp;
            if (lane == 0u) { part_m[pidx] = m[g]; part_l[pidx] = l[g]; }
            reinterpret_cast<float4*>(part_o + pidx * 128u)[lane] = o[g];
        }
    }
}

// Reduces Split-K partials of smalltide_dspark_attn_fp8_split.  Grid: (nh, rows, 1), Block: (32, 1, 1)
extern "C" __global__ void smalltide_dspark_attn_reduce(
    float* __restrict__ out, const float* __restrict__ part_o, const float* __restrict__ part_m,
    const float* __restrict__ part_l, uint32_t nh, uint32_t num_splits
) {
    uint32_t qh = blockIdx.x, r = blockIdx.y, lane = threadIdx.x;
    size_t base = (static_cast<size_t>(r) * nh + qh) * num_splits;
    float mb = -1e30f;
    for (uint32_t s = 0u; s < num_splits; ++s) {
        if (part_l[base + s] > 0.0f) mb = fmaxf(mb, part_m[base + s]);
    }
    float lb = 0.0f;
    float4 ob = make_float4(0, 0, 0, 0);
    for (uint32_t s = 0u; s < num_splits; ++s) {
        float ls = part_l[base + s];
        if (ls > 0.0f) {
            float f = __expf(part_m[base + s] - mb);
            lb += ls * f;
            float4 ow = reinterpret_cast<const float4*>(part_o + (base + s) * 128u)[lane];
            ob.x = fmaf(ow.x, f, ob.x); ob.y = fmaf(ow.y, f, ob.y);
            ob.z = fmaf(ow.z, f, ob.z); ob.w = fmaf(ow.w, f, ob.w);
        }
    }
    float inv = (lb > 0.0f) ? (1.0f / lb) : 0.0f;
    reinterpret_cast<float4*>(out + (static_cast<size_t>(r) * nh + qh) * 128u)[lane] =
        make_float4(ob.x * inv, ob.y * inv, ob.z * inv, ob.w * inv);
}

// Fused drafter block input prep: row 0 = RMSNorm(anchor_in, embed_norm), rows 1..7 = mask_embed;
// writes `res[r]` and `xn[r] = RMSNorm(res[r], in_ln)`.   Grid: (8, 1, 1), Block: (1024, 1, 1)
extern "C" __global__ void smalltide_dspark_embed_prep(
    float* __restrict__ res, float* __restrict__ xn, const float* __restrict__ anchor_in,
    const float* __restrict__ embed_norm, const float* __restrict__ mask_embed,
    const float* __restrict__ in_ln, uint32_t D, float eps
) {
    __shared__ float red[32];
    uint32_t r = blockIdx.x, t = threadIdx.x;
    float v[3];
    if (r == 0u) {
        float sq = 0.0f;
        #pragma unroll
        for (uint32_t e = 0u; e < 3u; ++e) {
            uint32_t d = t + e * blockDim.x;
            float x = (d < D) ? anchor_in[d] : 0.0f;
            v[e] = x;
            sq = fmaf(x, x, sq);
        }
        sq = wave32_sum(sq);
        if ((t & 31u) == 0u) red[t >> 5] = sq;
        __syncthreads();
        if (t < 32u) {
            float s = wave32_sum(red[t]);
            if (t == 0u) red[0] = s;
        }
        __syncthreads();
        float inv0 = rsqrtf(red[0] / static_cast<float>(D) + eps);
        #pragma unroll
        for (uint32_t e = 0u; e < 3u; ++e) {
            uint32_t d = t + e * blockDim.x;
            if (d < D) v[e] = v[e] * inv0 * embed_norm[d];
        }
    } else {
        #pragma unroll
        for (uint32_t e = 0u; e < 3u; ++e) {
            uint32_t d = t + e * blockDim.x;
            v[e] = (d < D) ? mask_embed[d] : 0.0f;
        }
    }
    float sq = 0.0f;
    #pragma unroll
    for (uint32_t e = 0u; e < 3u; ++e) {
        uint32_t d = t + e * blockDim.x;
        if (d < D) {
            res[static_cast<size_t>(r) * D + d] = v[e];
            sq = fmaf(v[e], v[e], sq);
        }
    }
    sq = wave32_sum(sq);
    if ((t & 31u) == 0u) red[t >> 5] = sq;
    __syncthreads();
    if (t < 32u) {
        float s = wave32_sum(red[t]);
        if (t == 0u) red[0] = s;
    }
    __syncthreads();
    float inv = rsqrtf(red[0] / static_cast<float>(D) + eps);
    #pragma unroll
    for (uint32_t e = 0u; e < 3u; ++e) {
        uint32_t d = t + e * blockDim.x;
        if (d < D) xn[static_cast<size_t>(r) * D + d] = v[e] * inv * in_ln[d];
    }
}

// Fused residual add + RMSNorm across rows: res += tmp; xn = RMSNorm(res, w).
//   Grid: (rows, 1, 1), Block: (1024, 1, 1) (D <= 3072)
extern "C" __global__ void smalltide_add_rmsnorm_rows(
    float* __restrict__ res, float* __restrict__ xn, const float* __restrict__ tmp,
    const float* __restrict__ w, uint32_t D, float eps
) {
    __shared__ float red[32];
    uint32_t r = blockIdx.x, t = threadIdx.x;
    float v[3];
    float sq = 0.0f;
    #pragma unroll
    for (uint32_t e = 0u; e < 3u; ++e) {
        uint32_t d = t + e * blockDim.x;
        if (d < D) {
            size_t idx = static_cast<size_t>(r) * D + d;
            float s = res[idx] + tmp[idx];
            res[idx] = s;
            v[e] = s;
            sq = fmaf(s, s, sq);
        } else {
            v[e] = 0.0f;
        }
    }
    sq = wave32_sum(sq);
    if ((t & 31u) == 0u) red[t >> 5] = sq;
    __syncthreads();
    if (t < 32u) {
        float s = wave32_sum(red[t]);
        if (t == 0u) red[0] = s;
    }
    __syncthreads();
    float inv = rsqrtf(red[0] / static_cast<float>(D) + eps);
    #pragma unroll
    for (uint32_t e = 0u; e < 3u; ++e) {
        uint32_t d = t + e * blockDim.x;
        if (d < D) xn[static_cast<size_t>(r) * D + d] = v[e] * inv * w[d];
    }
}

// Fused Q + K per-head RMSNorm + RoPE in one launch.  Grid: (hq + hk, rows, 1), Block: (64, 1, 1)
extern "C" __global__ void smalltide_dspark_qk_norm_rope(
    float* __restrict__ q, float* __restrict__ k, const float* __restrict__ qw, const float* __restrict__ kw,
    uint32_t hq, uint32_t hk, uint32_t pos0, float log2_theta, float eps
) {
    __shared__ float red[2];
    uint32_t bx = blockIdx.x, r = blockIdx.y, i = threadIdx.x;
    bool is_q = bx < hq;
    uint32_t h = is_q ? bx : (bx - hq);
    uint32_t nh = is_q ? hq : hk;
    float* p = (is_q ? q : k) + (static_cast<size_t>(r) * nh + h) * 128u;
    const float* w = is_q ? qw : kw;
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

// ===========================================================================
// DFlash 2 drafter kernels (5-layer sliding-window GQA-4 + 2-tap grouped conv + exact Top-16)
// ===========================================================================

// Grouped Split-K FP8 sliding-window attention for DFlash 2 (GQA = 4, nh = 32, nkv = 8, nblk = 8).
// Exact per-row sliding window visibility: row `r` at position `P + r` attends to context `pos < P`
// iff `(P + r) < pos + window` (i.e. `query_pos - key_pos < window`), plus all `nblk` block rows.
//   Grid: (nkv, num_splits, 1), Block: (256, 1, 1)
extern "C" __global__ void smalltide_dflash2_attn_fp8_split(
    float* __restrict__ out, float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
    const float* __restrict__ q, const float* __restrict__ kb, const float* __restrict__ vb,
    const uint8_t* __restrict__ kc, const float* __restrict__ ks,
    const uint8_t* __restrict__ vc, const float* __restrict__ vs,
    uint32_t P, uint32_t lo, uint32_t window, uint32_t ring, uint32_t nh, uint32_t nkv, uint32_t nblk,
    uint32_t num_splits, float scale, uint32_t v_q4, const uint32_t* __restrict__ page_table
) {
    uint32_t kvh = blockIdx.x, sp = blockIdx.y;
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t r = warp; // 8 warps = 8 block rows
    uint32_t qh0 = kvh * 4u;
    uint32_t kvd = nkv * 128u;
    uint32_t q_pos = P + r;

    float4 qv[4];
    float m[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    float l[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 o[4] = {
        make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0),
        make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0)
    };
    #pragma unroll
    for (uint32_t g = 0u; g < 4u; ++g) {
        qv[g] = reinterpret_cast<const float4*>(q + (static_cast<size_t>(r) * nh + qh0 + g) * 128u)[lane];
    }

    uint32_t nctx = (P > lo) ? (P - lo) : 0u;
    uint32_t per = (nctx + num_splits - 1u) / num_splits;
    uint32_t c_start = sp * per;
    uint32_t c_end = min(nctx, c_start + per);

    __shared__ uint32_t s_k8[16][32];
    __shared__ uint32_t s_v8[16][32];
    __shared__ float s_ks[16], s_vs[16];

    uint32_t lt = threadIdx.x >> 4;
    uint32_t lw = threadIdx.x & 15u;
    for (uint32_t j0 = c_start; j0 < c_end; j0 += 16u) {
        uint32_t tlen = min(16u, c_end - j0);
        if (lt < tlen) {
            uint32_t pos = lo + j0 + lt;
            uint32_t slot = kv_phys(page_table, (ring != 0u) ? (pos % ring) : pos);
            size_t head_idx = static_cast<size_t>(slot) * nkv + kvh;
            uint2 kw = reinterpret_cast<const uint2*>(kc + head_idx * 128u)[lw];
            s_k8[lt][lw * 2u + 0u] = kw.x; s_k8[lt][lw * 2u + 1u] = kw.y;
            if (v_q4 != 0u) {
                s_v8[lt][lw] = reinterpret_cast<const uint32_t*>(vc + head_idx * 64u)[lw];
                if (lw < 4u) {
                    s_v8[lt][16u + lw] = reinterpret_cast<const uint32_t*>(vs)[head_idx * 4u + lw];
                }
                if (lw == 0u) {
                    s_ks[lt] = ks[head_idx] * scale;
                    s_vs[lt] = 1.0f;
                }
            } else {
                uint2 vw = reinterpret_cast<const uint2*>(vc + head_idx * 128u)[lw];
                s_v8[lt][lw * 2u + 0u] = vw.x; s_v8[lt][lw * 2u + 1u] = vw.y;
                if (lw == 0u) {
                    s_ks[lt] = ks[head_idx] * scale;
                    s_vs[lt] = vs[head_idx];
                }
            }
        }
        __syncthreads();
        for (uint32_t t = 0u; t < tlen; ++t) {
            uint32_t pos = lo + j0 + t;
            if (pos + window <= q_pos) continue;
            float4 kv4 = decode_fp8x4(s_k8[t][lane]);
            float4 vv4;
            if (v_q4 != 0u) {
                uint32_t w = s_v8[t][lane >> 1];
                uint32_t v2 = (lane & 1u) ? (w >> 16) : (w & 0xFFFFu);
                uint32_t pku = s_v8[t][16u + (lane >> 3)];
                KvV4Scale pk;
                static_assert(sizeof(KvV4Scale) == 4, "KvV4Scale must be 4 bytes");
                __builtin_memcpy(&pk, &pku, 4);
                float vlo = static_cast<float>(pk.lo);
                float vsc_g = static_cast<float>(pk.sc);
                vv4 = make_float4(
                    fmaf(static_cast<float>(v2 & 0xFu), vsc_g, vlo),
                    fmaf(static_cast<float>((v2 >> 4) & 0xFu), vsc_g, vlo),
                    fmaf(static_cast<float>((v2 >> 8) & 0xFu), vsc_g, vlo),
                    fmaf(static_cast<float>((v2 >> 12) & 0xFu), vsc_g, vlo)
                );
            } else {
                vv4 = decode_fp8x4(s_v8[t][lane]);
            }
            float ksc = s_ks[t], vsc = s_vs[t];
            #pragma unroll
            for (uint32_t g = 0u; g < 4u; ++g) {
                float sc = wave32_sum(qv[g].x * kv4.x + qv[g].y * kv4.y + qv[g].z * kv4.z + qv[g].w * kv4.w) * ksc;
                float mn = fmaxf(m[g], sc);
                float al = __expf(m[g] - mn), pr = __expf(sc - mn);
                l[g] = l[g] * al + pr;
                m[g] = mn;
                float pv = pr * vsc;
                o[g].x = fmaf(vv4.x, pv, o[g].x * al); o[g].y = fmaf(vv4.y, pv, o[g].y * al);
                o[g].z = fmaf(vv4.z, pv, o[g].z * al); o[g].w = fmaf(vv4.w, pv, o[g].w * al);
            }
        }
        __syncthreads();
    }

    if (sp == num_splits - 1u) {
        for (uint32_t br = 0u; br < nblk; ++br) {
            float4 kv4 = reinterpret_cast<const float4*>(kb + static_cast<size_t>(br) * kvd + kvh * 128u)[lane];
            float4 vv4 = reinterpret_cast<const float4*>(vb + static_cast<size_t>(br) * kvd + kvh * 128u)[lane];
            #pragma unroll
            for (uint32_t g = 0u; g < 4u; ++g) {
                float sc = wave32_sum(qv[g].x * kv4.x + qv[g].y * kv4.y + qv[g].z * kv4.z + qv[g].w * kv4.w) * scale;
                float mn = fmaxf(m[g], sc);
                float al = __expf(m[g] - mn), pr = __expf(sc - mn);
                l[g] = l[g] * al + pr;
                m[g] = mn;
                o[g].x = fmaf(vv4.x, pr, o[g].x * al); o[g].y = fmaf(vv4.y, pr, o[g].y * al);
                o[g].z = fmaf(vv4.z, pr, o[g].z * al); o[g].w = fmaf(vv4.w, pr, o[g].w * al);
            }
        }
    }

    if (num_splits == 1u) {
        #pragma unroll
        for (uint32_t g = 0u; g < 4u; ++g) {
            float inv = (l[g] > 0.0f) ? (1.0f / l[g]) : 0.0f;
            reinterpret_cast<float4*>(out + (static_cast<size_t>(r) * nh + qh0 + g) * 128u)[lane] =
                make_float4(o[g].x * inv, o[g].y * inv, o[g].z * inv, o[g].w * inv);
        }
    } else {
        #pragma unroll
        for (uint32_t g = 0u; g < 4u; ++g) {
            size_t pidx = (static_cast<size_t>(r) * nh + qh0 + g) * num_splits + sp;
            if (lane == 0u) { part_m[pidx] = m[g]; part_l[pidx] = l[g]; }
            reinterpret_cast<float4*>(part_o + pidx * 128u)[lane] = o[g];
        }
    }
}

// DFlash 2 grouped 2-tap dynamic causal convolution (prepare / side 0).
//   coeff: [rows, 2, 2, G], base: [2, 2, D] (side 0 = base[0..2*D]).
//   Grid: (rows, 1, 1), Block: (256, 1, 1)
extern "C" __global__ void smalltide_dflash2_conv_prepare(
    float* __restrict__ out, const float* __restrict__ x, const float* __restrict__ coeff,
    const float* __restrict__ base, uint32_t D, uint32_t group_size
) {
    uint32_t r = blockIdx.x;
    uint32_t G = D / group_size;
    const float* c_row = coeff + static_cast<size_t>(r) * 4u * G;
    const float* x_cur = x + static_cast<size_t>(r) * D;
    const float* x_prev = (r > 0u) ? (x + static_cast<size_t>(r - 1u) * D) : nullptr;
    float* o_row = out + static_cast<size_t>(r) * D;
    for (uint32_t d = threadIdx.x; d < D; d += blockDim.x) {
        uint32_t g = d / group_size;
        float k0 = base[d] + c_row[g];
        float v = k0 * x_cur[d];
        if (r > 0u) {
            float k1 = base[D + d] + c_row[G + g];
            v = fmaf(k1, x_prev[d], v);
        }
        o_row[d] = v;
    }
}

// DFlash 2 fused grouped 2-tap dynamic conv (finish / side 1) + residual add + RMSNorm.
//   res += conv_side1(tmp, coeff, base); xn = RMSNorm(res, w, eps).
//   Grid: (rows, 1, 1), Block: (1024, 1, 1) (D <= 3072)
extern "C" __global__ void smalltide_dflash2_conv_finish_add_rmsnorm(
    float* __restrict__ res, float* __restrict__ xn, const float* __restrict__ tmp,
    const float* __restrict__ coeff, const float* __restrict__ base, const float* __restrict__ w,
    uint32_t D, uint32_t group_size, float eps
) {
    __shared__ float red[32];
    uint32_t r = blockIdx.x, t = threadIdx.x;
    uint32_t G = D / group_size;
    const float* c_row = coeff + static_cast<size_t>(r) * 4u * G;
    const float* t_cur = tmp + static_cast<size_t>(r) * D;
    const float* t_prev = (r > 0u) ? (tmp + static_cast<size_t>(r - 1u) * D) : nullptr;
    float v[3];
    float sq = 0.0f;
    #pragma unroll
    for (uint32_t e = 0u; e < 3u; ++e) {
        uint32_t d = t + e * blockDim.x;
        if (d < D) {
            uint32_t g = d / group_size;
            float k0 = base[2u * D + d] + c_row[2u * G + g];
            float cv = k0 * t_cur[d];
            if (r > 0u) {
                float k1 = base[3u * D + d] + c_row[3u * G + g];
                cv = fmaf(k1, t_prev[d], cv);
            }
            size_t idx = static_cast<size_t>(r) * D + d;
            float s = res[idx] + cv;
            res[idx] = s;
            v[e] = s;
            sq = fmaf(s, s, sq);
        } else {
            v[e] = 0.0f;
        }
    }
    sq = wave32_sum(sq);
    if ((t & 31u) == 0u) red[t >> 5] = sq;
    __syncthreads();
    if (t < 32u) {
        float s = wave32_sum(red[t]);
        if (t == 0u) red[0] = s;
    }
    __syncthreads();
    float inv = rsqrtf(red[0] / static_cast<float>(D) + eps);
    #pragma unroll
    for (uint32_t e = 0u; e < 3u; ++e) {
        uint32_t d = t + e * blockDim.x;
        if (d < D) xn[static_cast<size_t>(r) * D + d] = v[e] * inv * w[d];
    }
}

__device__ __forceinline__ bool topk_better(float v1, uint32_t i1, float v2, uint32_t i2) {
    return (v1 > v2) || (v1 == v2 && i1 < i2);
}

// Exact GPU Top-16 per row, Pass 1: 1 block per 4096 columns (16 coalesced floats per thread).
//   Grid: (num_blocks, rows, 1), Block: (256, 1, 1) where num_blocks = ceil(n / 4096).
extern "C" __global__ void smalltide_topk16_rows_part(
    float* __restrict__ part_val, uint32_t* __restrict__ part_idx,
    const float* __restrict__ x, uint32_t n, uint32_t num_blocks
) {
    __shared__ float sv[8];
    __shared__ uint32_t si[8];
    uint32_t bx = blockIdx.x, row = blockIdx.y, t = threadIdx.x;
    uint32_t lane = t & 31u, wid = t >> 5;
    const float* xr = x + static_cast<size_t>(row) * n;
    uint32_t base_col = bx * 4096u;

    float v[16];
    uint32_t id[16];
    #pragma unroll
    for (uint32_t e = 0u; e < 16u; ++e) {
        uint32_t col = base_col + e * 256u + t;
        if (col < n) {
            v[e] = xr[col];
            id[e] = col;
        } else {
            v[e] = -1e38f;
            id[e] = 0xFFFFFFFFu;
        }
    }

    size_t out_base = (static_cast<size_t>(row) * num_blocks + bx) * 16u;
    for (uint32_t k = 0u; k < 16u; ++k) {
        float bv = -1e38f;
        uint32_t bi = 0xFFFFFFFFu;
        #pragma unroll
        for (uint32_t e = 0u; e < 16u; ++e) {
            if (topk_better(v[e], id[e], bv, bi)) { bv = v[e]; bi = id[e]; }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor(bv, off, 32);
            uint32_t oi = __shfl_xor(bi, off, 32);
            if (topk_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0u) { sv[wid] = bv; si[wid] = bi; }
        __syncthreads();
        if (wid == 0u) {
            bv = (lane < 8u) ? sv[lane] : -1e38f;
            bi = (lane < 8u) ? si[lane] : 0xFFFFFFFFu;
            #pragma unroll
            for (int off = 4; off > 0; off >>= 1) {
                float ov = __shfl_xor(bv, off, 32);
                uint32_t oi = __shfl_xor(bi, off, 32);
                if (topk_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0u) {
                sv[0] = bv; si[0] = bi;
                part_val[out_base + k] = bv;
                part_idx[out_base + k] = bi;
            }
        }
        __syncthreads();
        uint32_t win = si[0];
        #pragma unroll
        for (uint32_t e = 0u; e < 16u; ++e) {
            if (id[e] == win) { v[e] = -1e38f; id[e] = 0xFFFFFFFFu; }
        }
        __syncthreads();
    }
}

// Exact GPU Top-16 per row, Pass 2: reduces `num_blocks * 16` (<= 2048) candidates to sorted top-16.
//   Grid: (rows, 1, 1), Block: (256, 1, 1)
extern "C" __global__ void smalltide_topk16_rows_reduce(
    float* __restrict__ out_val, uint32_t* __restrict__ out_idx,
    const float* __restrict__ part_val, const uint32_t* __restrict__ part_idx,
    uint32_t num_blocks
) {
    __shared__ float sv[8];
    __shared__ uint32_t si[8];
    uint32_t row = blockIdx.x, t = threadIdx.x;
    uint32_t lane = t & 31u, wid = t >> 5;
    uint32_t total = num_blocks * 16u;
    size_t in_base = static_cast<size_t>(row) * total;

    float v[8];
    uint32_t id[8];
    #pragma unroll
    for (uint32_t e = 0u; e < 8u; ++e) {
        uint32_t j = e * 256u + t;
        if (j < total) {
            v[e] = part_val[in_base + j];
            id[e] = part_idx[in_base + j];
        } else {
            v[e] = -1e38f;
            id[e] = 0xFFFFFFFFu;
        }
    }

    size_t out_base = static_cast<size_t>(row) * 16u;
    for (uint32_t k = 0u; k < 16u; ++k) {
        float bv = -1e38f;
        uint32_t bi = 0xFFFFFFFFu;
        #pragma unroll
        for (uint32_t e = 0u; e < 8u; ++e) {
            if (topk_better(v[e], id[e], bv, bi)) { bv = v[e]; bi = id[e]; }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor(bv, off, 32);
            uint32_t oi = __shfl_xor(bi, off, 32);
            if (topk_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0u) { sv[wid] = bv; si[wid] = bi; }
        __syncthreads();
        if (wid == 0u) {
            bv = (lane < 8u) ? sv[lane] : -1e38f;
            bi = (lane < 8u) ? si[lane] : 0xFFFFFFFFu;
            #pragma unroll
            for (int off = 4; off > 0; off >>= 1) {
                float ov = __shfl_xor(bv, off, 32);
                uint32_t oi = __shfl_xor(bi, off, 32);
                if (topk_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0u) {
                sv[0] = bv; si[0] = bi;
                out_val[out_base + k] = bv;
                out_idx[out_base + k] = bi;
            }
        }
        __syncthreads();
        uint32_t win = si[0];
        #pragma unroll
        for (uint32_t e = 0u; e < 8u; ++e) {
            if (id[e] == win) { v[e] = -1e38f; id[e] = 0xFFFFFFFFu; }
        }
        __syncthreads();
    }
}
