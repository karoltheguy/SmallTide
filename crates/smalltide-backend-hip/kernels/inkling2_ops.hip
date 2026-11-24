// SmallTide RDNA 4 (gfx1201) Inkling-2-Mini Specialized Layer Kernels:
// - QK / Rel L2-norm clipping (`qk_l2_clip_threshold = 22.627417`, `rel_l2_clip_threshold = 8.0`)
// - Parameter-free Value RMSNorm (`use_value_norm_no_gamma = true`)
// - Attention output gating (`gate_attn`)
// - Multi-query Attention Residuals mixing (`attn_res` & Loop cross-pass messages)
// - Relative position bias table projection (`rel_logits_proj`)
// - Loop Core rank-64 controller prep & gated update
// - F32 GEMV, elementwise add/scale/copy, final logit softcapping, and parallel argmax

// ---------------------------------------------------------------------------
// 1. Per-Head L2 Norm Clipping + Optional Per-Channel Scale
//    Exact `inkling2.cpp:165-176`:
//      if (scale) cur[h, d] *= scale[h, d];
//      norm = sqrt(sum_d cur[h, d]^2) + 1e-6;
//      cur[h, d] *= min(threshold / norm, 1.0);
//    x: [n_heads, head_dim] (in-place), scale: [n_heads, head_dim] (nullable)
//    Grid: (n_heads, 1, 1), Block: (128, 1, 1)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_head_l2_clip_f32(
    float* __restrict__ x,
    const float* __restrict__ scale,
    uint32_t n_heads,
    uint32_t head_dim,
    float threshold
) {
    uint32_t h = blockIdx.x;
    if (h >= n_heads) return;
    float* row = x + static_cast<size_t>(h) * head_dim;
    const float* srow = (scale != nullptr) ? (scale + static_cast<size_t>(h) * head_dim) : nullptr;

    float local_sq = 0.0f;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float v = row[d];
        if (srow != nullptr) {
            v *= srow[d];
            row[d] = v;
        }
        local_sq = fmaf(v, v, local_sq);
    }

    __shared__ float wsum[4];
    float s = wave32_sum(local_sq);
    uint32_t lane = threadIdx.x & 31u;
    uint32_t wid = threadIdx.x >> 5;
    if (lane == 0u) wsum[wid] = s;
    __syncthreads();

    __shared__ float factor;
    if (threadIdx.x == 0u) {
        float total = wsum[0] + wsum[1] + wsum[2] + wsum[3];
        float norm = sqrtf(total) + 1e-6f;
        float clip = threshold / norm;
        factor = (clip < 1.0f) ? clip : 1.0f;
    }
    __syncthreads();

    float f = factor;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        row[d] *= f;
    }
}

// ---------------------------------------------------------------------------
// 2. Parameter-Free Value RMSNorm (`use_value_norm_no_gamma = true`)
//    v: [n_kv_heads, head_dim] (in-place)
//    Grid: (n_kv_heads, 1, 1), Block: (128, 1, 1)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_value_norm_no_gamma_f32(
    float* __restrict__ v,
    uint32_t n_kv_heads,
    uint32_t head_dim,
    float eps
) {
    uint32_t h = blockIdx.x;
    if (h >= n_kv_heads) return;
    float* row = v + static_cast<size_t>(h) * head_dim;

    float local_sq = 0.0f;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = row[d];
        local_sq = fmaf(val, val, local_sq);
    }

    __shared__ float wsum[4];
    float s = wave32_sum(local_sq);
    uint32_t lane = threadIdx.x & 31u;
    uint32_t wid = threadIdx.x >> 5;
    if (lane == 0u) wsum[wid] = s;
    __syncthreads();

    __shared__ float inv_rms;
    if (threadIdx.x == 0u) {
        float total = wsum[0] + wsum[1] + wsum[2] + wsum[3];
        inv_rms = rsqrtf(total / static_cast<float>(head_dim) + eps);
    }
    __syncthreads();

    float ir = inv_rms;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        row[d] *= ir;
    }
}

// ---------------------------------------------------------------------------
// 3. Attention Output Gate Multiply: out[i] = attn[i] * sigmoid(gate[i])
//    Grid: (ceil(n / 256), 1, 1), Block: (256, 1, 1)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_gate_attn_sigmoid_mul_f32(
    float* __restrict__ out,
    const float* __restrict__ attn,
    const float* __restrict__ gate,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = 1.0f / (1.0f + expf(-gate[i]));
    out[i] = attn[i] * g;
}

// ---------------------------------------------------------------------------
// 4. Multi-Query Attention Residuals Mixing (`inkling2.cpp:179-192`):
//    States = [blocks[0], ..., blocks[num_blocks - 1], cur] (total = num_blocks + 1 <= 16).
//    For each query q in 0..num_queries:
//      norm_s = rms_norm(state[s], eps)
//      score[s] = dot(norm_s, queries[q]) / sqrt(hidden)
//      weights = softmax(score)
//      out[q] = sum_s weights[s] * (normalize_values ? norm_s : state[s])
//    Grid: (num_queries, 1, 1), Block: (256, 1, 1)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_attn_res_mix_f32(
    float* __restrict__ out,         // [num_queries, hidden]
    const float* __restrict__ blocks,// [num_blocks, hidden] (nullable if num_blocks == 0)
    const float* __restrict__ cur,   // [hidden]
    const float* __restrict__ queries,// [num_queries, hidden]
    uint32_t num_blocks,
    uint32_t num_queries,
    uint32_t hidden,
    float eps,
    uint32_t normalize_values
) {
    uint32_t q_idx = blockIdx.x;
    if (q_idx >= num_queries) return;

    const float* query = queries + static_cast<size_t>(q_idx) * hidden;
    float* out_row = out + static_cast<size_t>(q_idx) * hidden;

    uint32_t total_states = num_blocks + 1u;
    __shared__ float s_scores[16];
    __shared__ float s_inv_rms[16];
    __shared__ float s_warp[8];

    uint32_t lane = threadIdx.x & 31u;
    uint32_t wid = threadIdx.x >> 5;
    float inv_sqrt_h = rsqrtf(static_cast<float>(hidden));

    for (uint32_t s = 0u; s < total_states; ++s) {
        const float* st = (s < num_blocks) ? (blocks + static_cast<size_t>(s) * hidden) : cur;
        float sq = 0.0f;
        for (uint32_t d = threadIdx.x; d < hidden; d += blockDim.x) {
            float v = st[d];
            sq = fmaf(v, v, sq);
        }
        float ws = wave32_sum(sq);
        if (lane == 0u) s_warp[wid] = ws;
        __syncthreads();
        if (threadIdx.x == 0u) {
            float tot = 0.0f;
            for (uint32_t w = 0u; w < 8u; ++w) tot += s_warp[w];
            s_inv_rms[s] = rsqrtf(tot / static_cast<float>(hidden) + eps);
        }
        __syncthreads();

        float ir = s_inv_rms[s];
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < hidden; d += blockDim.x) {
            dot = fmaf(st[d] * ir, query[d], dot);
        }
        float wd = wave32_sum(dot);
        if (lane == 0u) s_warp[wid] = wd;
        __syncthreads();
        if (threadIdx.x == 0u) {
            float tot = 0.0f;
            for (uint32_t w = 0u; w < 8u; ++w) tot += s_warp[w];
            s_scores[s] = tot * inv_sqrt_h;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0u) {
        float m = s_scores[0];
        for (uint32_t s = 1u; s < total_states; ++s) m = fmaxf(m, s_scores[s]);
        float sum = 0.0f;
        for (uint32_t s = 0u; s < total_states; ++s) {
            s_scores[s] = expf(s_scores[s] - m);
            sum += s_scores[s];
        }
        float inv = 1.0f / fmaxf(sum, 1e-12f);
        for (uint32_t s = 0u; s < total_states; ++s) s_scores[s] *= inv;
    }
    __syncthreads();

    for (uint32_t d = threadIdx.x; d < hidden; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t s = 0u; s < total_states; ++s) {
            const float* st = (s < num_blocks) ? (blocks + static_cast<size_t>(s) * hidden) : cur;
            float v = st[d];
            if (normalize_values != 0u) v *= s_inv_rms[s];
            acc = fmaf(s_scores[s], v, acc);
        }
        out_row[d] = acc;
    }
}

// ---------------------------------------------------------------------------
// 5. F32 GEMV for small high-precision projections (`attn_sink`, `k/v_sconv_dyn_proj`)
//    y[n] = sum_k W[n, k] * x[k]  (k_cols multiple of 4)
//    Grid: (ceil(n_rows / 4), 1, 1), Block: (128, 1, 1) -> 1 Wave32 warp per row
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_gemv_f32(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ w,
    uint32_t n_rows,
    uint32_t k_cols
) {
    uint32_t wid = threadIdx.x >> 5;
    uint32_t lane = threadIdx.x & 31u;
    uint32_t row = blockIdx.x * 4u + wid;
    if (row >= n_rows) return;

    const float4* w_vec = reinterpret_cast<const float4*>(w + static_cast<size_t>(row) * k_cols);
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    uint32_t k4 = k_cols >> 2;
    float acc = 0.0f;
    for (uint32_t i = lane; i < k4; i += 32u) {
        float4 wv = w_vec[i];
        float4 xv = x_vec[i];
        acc = fmaf(wv.x, xv.x, acc);
        acc = fmaf(wv.y, xv.y, acc);
        acc = fmaf(wv.z, xv.z, acc);
        acc = fmaf(wv.w, xv.w, acc);
    }
    float sum = wave32_sum(acc);
    if (lane == 0u) {
        y[row] = sum;
    }
}

// ---------------------------------------------------------------------------
// 6. Relative Position Bias Table Projection (`inkling2.cpp:358`):
//    r: [n_heads, 16], rel_proj: [rel_extent, 16] (transposed on load!)
//    table: [n_heads, rel_extent], table[qh, dist] = dot(r[qh], rel_proj[dist])
//    Grid: (ceil(rel_extent / 128), n_heads, 1), Block: (128, 1, 1)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_rel_table_f32(
    float* __restrict__ table,
    const float* __restrict__ r,
    const float* __restrict__ rel_proj,
    uint32_t n_heads,
    uint32_t rel_extent
) {
    uint32_t qh = blockIdx.y;
    uint32_t dist = blockIdx.x * blockDim.x + threadIdx.x;
    if (qh >= n_heads) return;

    __shared__ float4 sr[4];
    if (threadIdx.x < 4u) {
        sr[threadIdx.x] = reinterpret_cast<const float4*>(r + static_cast<size_t>(qh) * 16u)[threadIdx.x];
    }
    __syncthreads();
    if (dist >= rel_extent) return;

    const float4* p = reinterpret_cast<const float4*>(rel_proj + static_cast<size_t>(dist) * 16u);
    float acc = 0.0f;
    #pragma unroll
    for (uint32_t i = 0u; i < 4u; ++i) {
        float4 pv = p[i];
        float4 rv = sr[i];
        acc = fmaf(pv.x, rv.x, acc);
        acc = fmaf(pv.y, rv.y, acc);
        acc = fmaf(pv.z, rv.z, acc);
        acc = fmaf(pv.w, rv.w, acc);
    }
    table[static_cast<size_t>(qh) * rel_extent + dist] = acc;
}

// ---------------------------------------------------------------------------
// 7. Loop Core Controller Prep: out[0..H] = rmsnorm(cur), out[H..2H] = rmsnorm(msg)
//    Grid: (2, 1, 1), Block: (256, 1, 1)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_loop_control_prep_f32(
    float* __restrict__ out,
    const float* __restrict__ cur,
    const float* __restrict__ msg,
    uint32_t hidden,
    float eps
) {
    const float* src = (blockIdx.x == 0u) ? cur : msg;
    float* dst = out + static_cast<size_t>(blockIdx.x) * hidden;

    float sq = 0.0f;
    for (uint32_t d = threadIdx.x; d < hidden; d += blockDim.x) {
        float v = src[d];
        sq = fmaf(v, v, sq);
    }
    __shared__ float s_warp[8];
    float ws = wave32_sum(sq);
    uint32_t lane = threadIdx.x & 31u;
    uint32_t wid = threadIdx.x >> 5;
    if (lane == 0u) s_warp[wid] = ws;
    __syncthreads();

    __shared__ float inv_rms;
    if (threadIdx.x == 0u) {
        float tot = 0.0f;
        for (uint32_t w = 0u; w < 8u; ++w) tot += s_warp[w];
        inv_rms = rsqrtf(tot / static_cast<float>(hidden) + eps);
    }
    __syncthreads();

    float ir = inv_rms;
    for (uint32_t d = threadIdx.x; d < hidden; d += blockDim.x) {
        dst[d] = src[d] * ir;
    }
}

// ---------------------------------------------------------------------------
// 8. Loop Core Controller Apply (`inkling2.cpp:245-248`):
//    out[i] = (1.0 - tanh(forget[i])) * cur[i] + tanh(gain[i]) * msg[i]
//    Grid: (ceil(n / 256), 1, 1), Block: (256, 1, 1)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_loop_control_apply_f32(
    float* __restrict__ out,
    const float* __restrict__ cur,
    const float* __restrict__ msg,
    const float* __restrict__ gain_raw,
    const float* __restrict__ forget_raw,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = tanhf(gain_raw[i]);
    float f = tanhf(forget_raw[i]);
    out[i] = (1.0f - f) * cur[i] + g * msg[i];
}

// ---------------------------------------------------------------------------
// 9. Simple Elementwise Ops (Residual Add, Tau Scale, Device-to-Device Copy)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_vec_add_f32(
    float* __restrict__ out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

extern "C" __global__ void smalltide_vec_scale_f32(
    float* __restrict__ x,
    float alpha,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= alpha;
}

extern "C" __global__ void smalltide_vec_copy_f32(
    float* __restrict__ dst,
    const float* __restrict__ src,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

extern "C" __global__ void smalltide_zero_f32(
    float* __restrict__ dst,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = 0.0f;
}

// ---------------------------------------------------------------------------
// 10. Final Logit Softcapping (`final_logit_softcapping = 30.0`)
//     Grid: (ceil(vocab_size / 256), 1, 1), Block: (256, 1, 1)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_final_logit_softcap_f32(
    float* __restrict__ logits,
    uint32_t vocab_size,
    float cap
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vocab_size) return;
    logits[i] = cap * tanhf(logits[i] / cap);
}

// ---------------------------------------------------------------------------
// 11. Parallel Argmax (64 blocks x 256 threads -> 64 partial (val, idx) pairs)
//     Grid: (64, 1, 1), Block: (256, 1, 1)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_argmax_partial_f32(
    float* __restrict__ out_vals,   // [64]
    uint32_t* __restrict__ out_idx, // [64]
    const float* __restrict__ x,
    uint32_t n
) {
    uint32_t stride = gridDim.x * blockDim.x;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float best_v = -1e38f;
    uint32_t best_i = 0u;
    for (uint32_t i = tid; i < n; i += stride) {
        float v = x[i];
        if (v > best_v) {
            best_v = v;
            best_i = i;
        }
    }
    __shared__ float s_val[256];
    __shared__ uint32_t s_idx[256];
    s_val[threadIdx.x] = best_v;
    s_idx[threadIdx.x] = best_i;
    __syncthreads();
    for (uint32_t off = 128u; off > 0u; off >>= 1) {
        if (threadIdx.x < off) {
            if (s_val[threadIdx.x + off] > s_val[threadIdx.x]) {
                s_val[threadIdx.x] = s_val[threadIdx.x + off];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + off];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        out_vals[blockIdx.x] = s_val[0];
        out_idx[blockIdx.x] = s_idx[0];
    }
}

// ---------------------------------------------------------------------------
// 12. GPU-side rescue check handoff kernels (coherent host-mapped control buffers)
// ---------------------------------------------------------------------------

// Copies `n` u32 routing words (16 for single-token decode, up to 128 for 8-row verify) from
// `src` to coherent host-mapped `dst[0..n]`, flushes system-scope stores, then publishes `ticket`
// to `dst[sentinel_idx]` (disjoint from all routing rows so shrinking `m` never collides with old expert IDs).
// Grid: (1, 1, 1), Block: (32, 1, 1) (single wave32)
extern "C" __global__ void smalltide_rescue_write_route(
    uint32_t* __restrict__ dst,
    const uint32_t* __restrict__ src,
    uint32_t n,
    uint32_t sentinel_idx,
    uint32_t ticket
) {
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        __scoped_atomic_store_n(dst + i, src[i], __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    }
    __builtin_amdgcn_wave_barrier();
    __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    if (threadIdx.x == 0u) {
        __scoped_atomic_store_n(dst + sentinel_idx, ticket, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    }
}

// Runs on compute_stream right after hipStreamWaitValue32 before MoE kernels:
//   patch_hdr[0] == 0          : no rescue on this layer (immediate exit, no GL2 flush)
//   patch_hdr[0] == 0xFFFFFFFE : invalidate GL2 only (0 table patches)
//   patch_hdr[0] == 0xFFFFFFFF : invalidate GL2 + copy all `total_entries` from `full_table`
//   otherwise (`n`)            : invalidate GL2 + apply `n` (idx, slot) pairs from `patch_hdr[1..]`
// Grid: (1, 1, 1), Block: (256, 1, 1)
extern "C" __global__ void smalltide_rescue_apply_patch(
    uint32_t* __restrict__ slot_table,
    const uint32_t* __restrict__ patch_hdr,
    const uint32_t* __restrict__ full_table,
    uint32_t total_entries,
    uint32_t* __restrict__ probe,
    uint32_t layer_id,
    uint32_t ticket
) {
    if (threadIdx.x == 0u && probe != nullptr) {
        __scoped_atomic_store_n(probe + 0, (ticket << 12) | ((layer_id & 0xFFu) << 4) | 3u, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    }
    uint32_t n = __scoped_atomic_load_n(patch_hdr, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    if (n == 0u) {
        if (threadIdx.x == 0u && probe != nullptr) {
            __scoped_atomic_store_n(probe + 0, (ticket << 12) | ((layer_id & 0xFFu) << 4) | 4u, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
        }
        return;
    }
    // System-scope acquire invalidates GPU GL0/GL1/GL2 so freshly SDMA'd slabs in vram_pool
    // and host-written patch entries are visible to this kernel and subsequent MoE kernels.
    __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
    if (n != 0xFFFFFFFEu) {
        if (n == 0xFFFFFFFFu) {
            for (uint32_t i = threadIdx.x; i < total_entries; i += blockDim.x) {
                slot_table[i] = __scoped_atomic_load_n(full_table + i, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
            }
        } else {
            for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
                uint32_t idx = __scoped_atomic_load_n(patch_hdr + 1u + 2u * i, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
                uint32_t val = __scoped_atomic_load_n(patch_hdr + 2u + 2u * i, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
                if (idx < total_entries) slot_table[idx] = val;
            }
        }
    }
    if (threadIdx.x == 0u && probe != nullptr) {
        __scoped_atomic_store_n(probe + 1, n, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
        __scoped_atomic_store_n(probe + 0, (ticket << 12) | ((layer_id & 0xFFu) << 4) | 5u, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    }
}

// Fused GPU-side spin-wait + post-wait slot-table patch & system-scope GL2 invalidate.
// Avoids beta hipStreamWaitValue32 CP packets entirely while keeping 1 launch per layer.
// Grid: (1, 1, 1), Block: (256, 1, 1)
extern "C" __global__ void smalltide_rescue_wait_and_patch(
    uint32_t* __restrict__ slot_table,
    const uint32_t* __restrict__ flag,
    uint32_t ticket,
    const uint32_t* __restrict__ patch_hdr,
    const uint32_t* __restrict__ full_table,
    uint32_t total_entries,
    uint32_t* __restrict__ probe,
    uint32_t layer_id
) {
    __shared__ uint32_t s_n;
    if (threadIdx.x == 0u) {
        if (probe != nullptr) {
            __scoped_atomic_store_n(probe + 0, (ticket << 12) | ((layer_id & 0xFFu) << 4) | 2u, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
        }
        while (__scoped_atomic_load_n(flag, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM) < ticket) {
            __builtin_amdgcn_s_sleep(64);
        }
        // Broadcast patch header count from thread 0 after system-acquire so all 8 wavefronts
        // observe the exact same `n` without inter-wave cache divergence.
        __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
        s_n = __scoped_atomic_load_n(patch_hdr, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
        if (probe != nullptr) {
            __scoped_atomic_store_n(probe + 0, (ticket << 12) | ((layer_id & 0xFFu) << 4) | 3u, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
        }
    }
    __syncthreads();
    uint32_t n = s_n;
    if (n == 0u) {
        if (threadIdx.x == 0u && probe != nullptr) {
            __scoped_atomic_store_n(probe + 0, (ticket << 12) | ((layer_id & 0xFFu) << 4) | 4u, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
        }
        return;
    }
    __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
    if (n != 0xFFFFFFFEu) {
        if (n == 0xFFFFFFFFu) {
            for (uint32_t i = threadIdx.x; i < total_entries; i += blockDim.x) {
                slot_table[i] = __scoped_atomic_load_n(full_table + i, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
            }
        } else {
            for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
                uint32_t idx = __scoped_atomic_load_n(patch_hdr + 1u + 2u * i, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
                uint32_t val = __scoped_atomic_load_n(patch_hdr + 2u + 2u * i, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
                if (idx < total_entries) slot_table[idx] = val;
            }
        }
    }
    if (threadIdx.x == 0u && probe != nullptr) {
        __scoped_atomic_store_n(probe + 1, n, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
        __scoped_atomic_store_n(probe + 0, (ticket << 12) | ((layer_id & 0xFFu) << 4) | 5u, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    }
}
