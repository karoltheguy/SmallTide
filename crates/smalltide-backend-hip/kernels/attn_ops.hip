// SmallTide RDNA 4 (gfx1201) Router Top-8, sconv (k=4), QK L2-Clip/VNorm,
// and FP8 KV Cache + 256K FlashDecoding GQA Attention Kernels.

// ---------------------------------------------------------------------------
// 1. Router Top-8 (128 experts):
//    Grid: (1, 1, 1), Block: (128, 1, 1) -> 1 thread per expert (4 Wave32 warps)
//    Computes logit[e] = dot(x, gate_w[e]) + gate_b[e], selects Top-8,
//    applies softmax over selected top-8 (or sigmoid/norm) * route_scale.
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_router_topk8_f32(
    uint32_t* __restrict__ topk_ids,      // [8]
    float* __restrict__ topk_weights,     // [8]
    const float* __restrict__ x,          // [hidden_size]
    const float* __restrict__ gate_w,     // [128, hidden_size]
    const float* __restrict__ gate_b,     // [128] (nullable)
    uint32_t hidden_size,
    float route_scale
) {
    uint32_t e = threadIdx.x; // 0..127
    __shared__ float s_logits[128];
    __shared__ uint8_t s_used[128];

    const float4* x_vec = reinterpret_cast<const float4*>(x);
    const float4* w_vec = reinterpret_cast<const float4*>(gate_w + static_cast<size_t>(e) * hidden_size);
    uint32_t k4 = hidden_size >> 2;
    float acc = 0.0f;
    for (uint32_t i = 0; i < k4; ++i) {
        float4 xv = x_vec[i];
        float4 wv = w_vec[i];
        acc = fmaf(xv.x, wv.x, acc);
        acc = fmaf(xv.y, wv.y, acc);
        acc = fmaf(xv.z, wv.z, acc);
        acc = fmaf(xv.w, wv.w, acc);
    }
    if (gate_b != nullptr) {
        acc += gate_b[e];
    }
    s_logits[e] = acc;
    s_used[e] = 0u;
    __syncthreads();

    if (e == 0u) {
        // Inkling-2-Mini exact router (`LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`):
        // scores = sigmoid(dot(x, gate_w)), select Top-8 using (scores + gate_b),
        // weight = scores[top8] / (sum(scores[top8]) + 1e-20) * route_scale.
        __shared__ float s_scores[128];
        for (uint32_t i = 0u; i < 128u; ++i) {
            float raw = s_logits[i] - ((gate_b != nullptr) ? gate_b[i] : 0.0f);
            float sig = 1.0f / (1.0f + expf(-raw));
            s_scores[i] = sig;
            s_logits[i] = sig + ((gate_b != nullptr) ? gate_b[i] : 0.0f);
        }

        float top_sum = 0.0f;
        for (uint32_t k = 0u; k < 8u; ++k) {
            uint32_t best_i = 0u;
            float best_v = -1e30f;
            for (uint32_t i = 0u; i < 128u; ++i) {
                if (!s_used[i] && s_logits[i] > best_v) {
                    best_v = s_logits[i];
                    best_i = i;
                }
            }
            s_used[best_i] = 1u;
            topk_ids[k] = best_i;
            topk_weights[k] = s_scores[best_i];
            top_sum += s_scores[best_i];
        }
        float scale = route_scale / (top_sum + 1e-20f);
        for (uint32_t k = 0u; k < 8u; ++k) {
            topk_weights[k] *= scale;
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Causal Depthwise Short Conv (sconv_kernel_size = 4) for single decode step:
//    Exact Inkling-2-Mini formula (`inkling2.cpp:196-238`):
//      out = (cur + sum_{k=0..3} w[k]*x[t-3+k] [+ sum_{lag=0..3} dyn[head,lag]*x[t-lag]])
//            / sqrt((dyn != null ? 9.0 : 5.0))
//    state: [3, dim] rolling history (t-3, t-2, t-1) updated in-place
//    weight: [dim, 4] ([w(t-3), w(t-2), w(t-1), w(t)]) per channel
//    dyn_taps: optional [n_kv_heads * 4] ([lag=0 (t), lag=1 (t-1), lag=2 (t-2), lag=3 (t-3)])
//    Grid: (ceil(dim / 256), 1, 1), Block: (256, 1, 1)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_sconv_k4_decode_f32(
    float* __restrict__ out,            // [dim]
    float* __restrict__ state,          // [3 * dim]: [t-3, t-2, t-1]
    const float* __restrict__ x,        // [dim] current t
    const float* __restrict__ weight,   // [dim * 4]: [w(t-3), w(t-2), w(t-1), w(t)]
    const float* __restrict__ dyn_taps, // nullable [n_heads * 4]: [lag0(t), lag1(t-1), lag2(t-2), lag3(t-3)]
    uint32_t dim,
    uint32_t head_dim                   // e.g. 128 when dyn_taps != null
) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= dim) return;

    float s0 = state[c];
    float s1 = state[dim + c];
    float s2 = state[2u * dim + c];
    float cur = x[c];

    const float4 w = reinterpret_cast<const float4*>(weight)[c];
    float y = fmaf(s0, w.x, fmaf(s1, w.y, fmaf(s2, w.z, cur * w.w)));

    if (dyn_taps != nullptr) {
        uint32_t h = c / head_dim;
        const float4 d = reinterpret_cast<const float4*>(dyn_taps)[h];
        y = fmaf(cur, d.x, fmaf(s2, d.y, fmaf(s1, d.z, fmaf(s0, d.w, y))));
        y = (cur + y) * (1.0f / 3.0f); // 1 / sqrt(2*4 + 1) = 1/3
    } else {
        y = (cur + y) * 0.4472135954999579f; // 1 / sqrt(4 + 1) = 1/sqrt(5)
    }

    state[c]            = s1;
    state[dim + c]      = s2;
    state[2u * dim + c] = cur;
    out[c] = y;
}

// ---------------------------------------------------------------------------
// 3. Encode FP32 -> OCP FP8 E4M3FN on GPU (for writing current token into FP8 KV Cache)
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t encode_fp8_e4m3fn(float x) {
    uint32_t sign = (x < 0.0f) ? 0x80u : 0x00u;
    float ax = fabsf(x);
    if (ax == 0.0f) return static_cast<uint8_t>(sign);
    if (ax >= 448.0f) return static_cast<uint8_t>(sign | 0x7Eu);
    if (ax < (1.0f / 64.0f)) {
        uint32_t m = static_cast<uint32_t>(rintf(ax * 512.0f));
        return static_cast<uint8_t>(sign | ((m >= 8u) ? 0x08u : m));
    }
    uint32_t bits = __float_as_uint(ax);
    int f32_exp = static_cast<int>((bits >> 23) & 0xFFu) - 127;
    uint32_t f32_mant = bits & 0x007FFFFFu;
    uint32_t exp_field = static_cast<uint32_t>(f32_exp + 7);
    uint32_t top3 = f32_mant >> 20;
    uint32_t rem = f32_mant & 0x000FFFFFu;
    uint32_t round_up = (rem > 0x00080000u || (rem == 0x00080000u && (top3 & 1u))) ? 1u : 0u;
    uint32_t mant_field = top3 + round_up;
    if (mant_field == 8u) {
        mant_field = 0u;
        exp_field += 1u;
    }
    if (exp_field >= 15u && mant_field >= 7u) return static_cast<uint8_t>(sign | 0x7Eu);
    return static_cast<uint8_t>(sign | ((exp_field << 3) | mant_field));
}

// ---------------------------------------------------------------------------
// 4. Write 1 Decode Token's K & V into FP8 KV Cache:
//    k_in, v_in: [n_kv_heads, head_dim] (8 x 128) FP32
//    k_cache, v_cache: [max_seq, n_kv_heads, head_dim] uint8 FP8 (position-major: a sequence's
//    cache grows as one contiguous range, so it can be paged)
//    k_scales, v_scales: [max_seq, n_kv_heads] float (per-head-token absmax / 448.0)
//    Grid: (n_kv_heads, 1, 1), Block: (128, 1, 1) -> 1 thread per head_dim element
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_kv_store_fp8_decode(
    uint8_t* __restrict__ k_cache,
    float* __restrict__ k_scales,
    uint8_t* __restrict__ v_cache,
    float* __restrict__ v_scales,
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    uint32_t n_kv_heads,
    uint32_t max_seq,
    uint32_t head_dim,
    uint32_t slot_pos // pos % window_size for local layers, pos for global layers
) {
    uint32_t h = blockIdx.x;
    uint32_t d = threadIdx.x;
    if (h >= n_kv_heads || d >= head_dim) return;

    float kv = k_in[h * head_dim + d];
    float vv = v_in[h * head_dim + d];

    // Warp + block max_abs reduction across 128 threads (4 Wave32 warps)
    __shared__ float sk_max[4];
    __shared__ float sv_max[4];

    auto warp_max = [](float val) {
        val = fmaxf(val, __shfl_xor(val, 16, 32));
        val = fmaxf(val, __shfl_xor(val, 8, 32));
        val = fmaxf(val, __shfl_xor(val, 4, 32));
        val = fmaxf(val, __shfl_xor(val, 2, 32));
        val = fmaxf(val, __shfl_xor(val, 1, 32));
        return val;
    };

    uint32_t lane = d & 31u;
    uint32_t wid = d >> 5;
    float km = warp_max(fabsf(kv));
    float vm = warp_max(fabsf(vv));
    if (lane == 0u) {
        sk_max[wid] = km;
        sv_max[wid] = vm;
    }
    __syncthreads();

    __shared__ float k_scale_s;
    __shared__ float v_scale_s;
    if (d == 0u) {
        float kmax = fmaxf(fmaxf(sk_max[0], sk_max[1]), fmaxf(sk_max[2], sk_max[3]));
        float vmax = fmaxf(fmaxf(sv_max[0], sv_max[1]), fmaxf(sv_max[2], sv_max[3]));
        k_scale_s = (kmax > 0.0f) ? (kmax / 448.0f) : 1.0f;
        v_scale_s = (vmax > 0.0f) ? (vmax / 448.0f) : 1.0f;
        size_t s_idx = static_cast<size_t>(slot_pos) * n_kv_heads + h;
        k_scales[s_idx] = k_scale_s;
        v_scales[s_idx] = v_scale_s;
    }
    __syncthreads();

    size_t dst_idx = (static_cast<size_t>(slot_pos) * n_kv_heads + h) * head_dim + d;
    k_cache[dst_idx] = encode_fp8_e4m3fn(kv / k_scale_s);
    v_cache[dst_idx] = encode_fp8_e4m3fn(vv / v_scale_s);
}

// ---------------------------------------------------------------------------
// 5. GQA-Fused Split-K FlashDecoding FP8 KV Attention with Relative Bias + Local Ring!
//    Each block owns 1 KV head and processes ALL `gqa_ratio` (= 3 for 24:8) Q heads
//    simultaneously so K[t] and V[t] are read ONCE from GDDR6 instead of 3x!
//    Supports both Local Sliding-Window Ring (is_local=1, max_seq=1024) and Global 256K (is_local=0).
//    Grid : (n_kv_heads, num_splits, 1)
//    Block: (128, 1, 1) -> 4 Wave32 warps
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_flash_decode_fp8_split(
    float* __restrict__ partial_out,     // [n_q_heads, num_splits, 128]
    float* __restrict__ partial_m,       // [n_q_heads, num_splits]
    float* __restrict__ partial_l,       // [n_q_heads, num_splits]
    const float* __restrict__ q,         // [n_q_heads, 128]
    const uint8_t* __restrict__ k_cache, // [n_kv_heads, max_seq, 128]
    const float* __restrict__ k_scales,  // [n_kv_heads, max_seq]
    const uint8_t* __restrict__ v_cache, // [n_kv_heads, max_seq, 128]
    const float* __restrict__ v_scales,  // [n_kv_heads, max_seq]
    const float* __restrict__ rel_table, // nullable [n_q_heads, rel_extent]
    uint32_t n_q_heads,
    uint32_t n_kv_heads,
    uint32_t max_seq,
    uint32_t q_pos,                      // 0-based position of current token
    uint32_t rel_extent,                 // 1024
    uint32_t is_local,                   // 1 = sliding window ring of size max_seq, 0 = global [0..=q_pos]
    uint32_t num_splits,
    float inv_sqrt_d,
    uint32_t window,                     // local layers: attention window (<= max_seq ring size)
    const uint32_t* __restrict__ page_table // nullable: paged KV pool (see kv_phys)
) {
    uint32_t kvh = blockIdx.x;
    uint32_t split = blockIdx.y;
    if (kvh >= n_kv_heads || split >= num_splits) return;

    uint32_t gqa_ratio = n_q_heads / n_kv_heads; // 3 for Inkling-2-Mini (24:8)
    uint32_t qh0 = kvh * gqa_ratio;

    uint32_t win = (window != 0u) ? window : max_seq;
    uint32_t t_win_start = (is_local != 0u && (q_pos + 1u) > win) ? (q_pos + 1u - win) : 0u;
    uint32_t n_valid = q_pos + 1u - t_win_start;

    uint32_t tokens_per_split = (n_valid + num_splits - 1u) / num_splits;
    uint32_t i_start = split * tokens_per_split;
    uint32_t i_end = min(i_start + tokens_per_split, n_valid);

    uint32_t warp_id = threadIdx.x >> 5; // 0..3
    uint32_t lane = threadIdx.x & 31u;   // 0..31 -> 4 dims each

    // Preload up to 3 Q heads for this KV head into registers (Inkling-2-Mini GQA = 3:1)
    float4 qv0 = reinterpret_cast<const float4*>(q + (qh0 + 0u) * 128u)[lane];
    float4 qv1 = (gqa_ratio > 1u) ? reinterpret_cast<const float4*>(q + (qh0 + 1u) * 128u)[lane] : make_float4(0,0,0,0);
    float4 qv2 = (gqa_ratio > 2u) ? reinterpret_cast<const float4*>(q + (qh0 + 2u) * 128u)[lane] : make_float4(0,0,0,0);

    float m0 = -1e30f, l0 = 0.0f; float4 o0 = make_float4(0,0,0,0);
    float m1 = -1e30f, l1 = 0.0f; float4 o1 = make_float4(0,0,0,0);
    float m2 = -1e30f, l2 = 0.0f; float4 o2 = make_float4(0,0,0,0);

    // Position-major cache ([slot][n_kv_heads][128], scales [slot][n_kv_heads]).
    const uint8_t* k_base = k_cache + static_cast<size_t>(kvh) * 128u;
    const uint8_t* v_base = v_cache + static_cast<size_t>(kvh) * 128u;
    const float* ks_base = k_scales + kvh;
    const float* vs_base = v_scales + kvh;
    const size_t kv_stride = static_cast<size_t>(n_kv_heads) * 128u;

    const float* rel0 = (rel_table != nullptr) ? (rel_table + static_cast<size_t>(qh0 + 0u) * rel_extent) : nullptr;
    const float* rel1 = (rel_table != nullptr && gqa_ratio > 1u) ? (rel_table + static_cast<size_t>(qh0 + 1u) * rel_extent) : nullptr;
    const float* rel2 = (rel_table != nullptr && gqa_ratio > 2u) ? (rel_table + static_cast<size_t>(qh0 + 2u) * rel_extent) : nullptr;

    // Pre-multiply inv_sqrt_d by log2(e) so exp2f() executes as single-cycle hardware v_exp_f32!
    const float LOG2E = 1.4426950408889634f;
    float inv_sqrt_d_log2 = inv_sqrt_d * LOG2E;

    for (uint32_t idx = i_start + warp_id; idx < i_end; idx += 4u) {
        uint32_t t_pos = t_win_start + idx;
        uint32_t slot = kv_phys(page_table, (is_local != 0u) ? (t_pos % max_seq) : t_pos);
        uint32_t dist = q_pos - t_pos;

        // Read K[slot] ONCE from GDDR6 and decode 4 FP8 elements in registers
        uint32_t k4 = reinterpret_cast<const uint32_t*>(k_base + static_cast<size_t>(slot) * kv_stride)[lane];
        float kx = decode_fp8_e4m3fn(k4 & 0xFFu);
        float ky = decode_fp8_e4m3fn((k4 >> 8) & 0xFFu);
        float kz = decode_fp8_e4m3fn((k4 >> 16) & 0xFFu);
        float kw = decode_fp8_e4m3fn((k4 >> 24) & 0xFFu);
        float k_scale = ks_base[static_cast<size_t>(slot) * n_kv_heads] * inv_sqrt_d_log2;

        // Read V[slot] ONCE from GDDR6 and decode 4 FP8 elements in registers
        uint32_t v4 = reinterpret_cast<const uint32_t*>(v_base + static_cast<size_t>(slot) * kv_stride)[lane];
        float v_scale = vs_base[static_cast<size_t>(slot) * n_kv_heads];
        float vx = decode_fp8_e4m3fn(v4 & 0xFFu) * v_scale;
        float vy = decode_fp8_e4m3fn((v4 >> 8) & 0xFFu) * v_scale;
        float vz = decode_fp8_e4m3fn((v4 >> 16) & 0xFFu) * v_scale;
        float vw = decode_fp8_e4m3fn((v4 >> 24) & 0xFFu) * v_scale;

        // Update Q head 0 (all scores in log2 domain -> single-cycle v_exp_f32!)
        {
            float d = fmaf(kx, qv0.x, fmaf(ky, qv0.y, fmaf(kz, qv0.z, kw * qv0.w)));
            float rb = (rel0 != nullptr && dist < rel_extent) ? (rel0[dist] * LOG2E) : 0.0f;
            float sc = wave32_sum(d) * k_scale + rb;
            float mn = fmaxf(m0, sc);
            float al = exp2f(m0 - mn);
            float p  = exp2f(sc - mn);
            l0 = l0 * al + p; m0 = mn;
            o0.x = fmaf(vx, p, o0.x * al); o0.y = fmaf(vy, p, o0.y * al);
            o0.z = fmaf(vz, p, o0.z * al); o0.w = fmaf(vw, p, o0.w * al);
        }
        // Update Q head 1
        if (gqa_ratio > 1u) {
            float d = fmaf(kx, qv1.x, fmaf(ky, qv1.y, fmaf(kz, qv1.z, kw * qv1.w)));
            float rb = (rel1 != nullptr && dist < rel_extent) ? (rel1[dist] * LOG2E) : 0.0f;
            float sc = wave32_sum(d) * k_scale + rb;
            float mn = fmaxf(m1, sc);
            float al = exp2f(m1 - mn);
            float p  = exp2f(sc - mn);
            l1 = l1 * al + p; m1 = mn;
            o1.x = fmaf(vx, p, o1.x * al); o1.y = fmaf(vy, p, o1.y * al);
            o1.z = fmaf(vz, p, o1.z * al); o1.w = fmaf(vw, p, o1.w * al);
        }
        // Update Q head 2
        if (gqa_ratio > 2u) {
            float d = fmaf(kx, qv2.x, fmaf(ky, qv2.y, fmaf(kz, qv2.z, kw * qv2.w)));
            float rb = (rel2 != nullptr && dist < rel_extent) ? (rel2[dist] * LOG2E) : 0.0f;
            float sc = wave32_sum(d) * k_scale + rb;
            float mn = fmaxf(m2, sc);
            float al = exp2f(m2 - mn);
            float p  = exp2f(sc - mn);
            l2 = l2 * al + p; m2 = mn;
            o2.x = fmaf(vx, p, o2.x * al); o2.y = fmaf(vy, p, o2.y * al);
            o2.z = fmaf(vz, p, o2.z * al); o2.w = fmaf(vw, p, o2.w * al);
        }
    }

    // Convert log2 max back to natural log domain for reduce pass
    const float LN2 = 0.6931471805599453f;
    m0 *= LN2; m1 *= LN2; m2 *= LN2;

    // Reduce the 4 warps in LDS for each of the 3 Q heads sequentially (reusing 2 KB LDS)
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
            size_t meta_idx = static_cast<size_t>(qh) * num_splits + split;
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

// ---------------------------------------------------------------------------
// 6. FlashDecoding Reduce Pass (merges `num_splits` + optional attention sink logit)
//    Grid: (n_q_heads, 1, 1), Block: (32, 1, 1) -> 1 Wave32 warp per Q head!
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_flash_decode_fp8_reduce(
    float* __restrict__ out,              // [n_q_heads, 128]
    const float* __restrict__ partial_out,// [n_q_heads, num_splits, 128]
    const float* __restrict__ partial_m,  // [n_q_heads, num_splits]
    const float* __restrict__ partial_l,  // [n_q_heads, num_splits]
    const float* __restrict__ sink_logit, // [n_q_heads] (nullable)
    uint32_t n_q_heads,
    uint32_t num_splits
) {
    uint32_t qh = blockIdx.x;
    uint32_t lane = threadIdx.x; // 0..31
    if (qh >= n_q_heads) return;

    size_t base = static_cast<size_t>(qh) * num_splits;
    float m_max = (sink_logit != nullptr) ? sink_logit[qh] : -1e30f;
    for (uint32_t s = 0u; s < num_splits; ++s) {
        if (partial_l[base + s] > 0.0f) {
            m_max = fmaxf(m_max, partial_m[base + s]);
        }
    }

    float l_sum = (sink_logit != nullptr) ? expf(sink_logit[qh] - m_max) : 0.0f;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint32_t s = 0u; s < num_splits; ++s) {
        float ls = partial_l[base + s];
        if (ls > 0.0f) {
            float factor = expf(partial_m[base + s] - m_max);
            l_sum += ls * factor;
            float4 po = reinterpret_cast<const float4*>(partial_out + (base + s) * 128u)[lane];
            acc.x = fmaf(po.x, factor, acc.x);
            acc.y = fmaf(po.y, factor, acc.y);
            acc.z = fmaf(po.z, factor, acc.z);
            acc.w = fmaf(po.w, factor, acc.w);
        }
    }

    float inv_l = (l_sum > 1e-12f) ? (1.0f / l_sum) : 0.0f;
    acc.x *= inv_l;
    acc.y *= inv_l;
    acc.z *= inv_l;
    acc.w *= inv_l;
    reinterpret_cast<float4*>(out + qh * 128u)[lane] = acc;
}
