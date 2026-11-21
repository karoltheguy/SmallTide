// SmallTide RDNA 4 (gfx1201) Indirect MoE SwiGLU & Router Kernels
// Matches `smalltide_io::stide_pack::ExpertSlabHeader` byte-for-byte.

struct ExpertSlabHeaderGpu {
    float w13_outer_scale;
    float w2_outer_scale;
    uint32_t w13_weight_offset;
    uint32_t w13_scale_offset;
    uint32_t w2_weight_offset;
    uint32_t w2_scale_offset;
    uint32_t dtype_tag; // 0 = NVFP4, 1 = MXFP8
    uint32_t _reserved[9];
};

__device__ __forceinline__ float softcap_tanh(float x, float cap) {
    return (cap > 0.0f) ? (cap * tanhf(x / cap)) : x;
}

__device__ __forceinline__ float sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Exact Inkling-2-Mini SiTU activation (beta=4.0, linear_beta=25.0):
// beta * tanh(gate / beta) * sigmoid(gate) * linear_beta * tanh(up / linear_beta)
__device__ __forceinline__ float situ_f32(float gate, float up, float gate_cap, float up_cap) {
    float g = softcap_tanh(gate, gate_cap) * sigmoid_f32(gate);
    float u = softcap_tanh(up, up_cap);
    return g * u;
}

// Helper: NVFP4 dot product over one row of length k_cols (multiple of 16)
__device__ __forceinline__ float dot_nvfp4_row_wave32(
    const float4* __restrict__ x_vec,
    const uint8_t* __restrict__ w_packed_row,
    const uint8_t* __restrict__ w_scales_row,
    uint32_t k_cols,
    uint32_t lane
) {
    uint32_t n_blocks16 = k_cols >> 4;
    const uint2* w_row = reinterpret_cast<const uint2*>(w_packed_row);
    float acc = 0.0f;
    for (uint32_t b = lane; b < n_blocks16; b += 32u) {
        uint2 w8 = w_row[b];
        float blk_scale = decode_fp8_e4m3fn(w_scales_row[b]);
        uint32_t base4 = b << 2;
        float4 x0 = x_vec[base4 + 0];
        float4 x1 = x_vec[base4 + 1];
        float4 x2 = x_vec[base4 + 2];
        float4 x3 = x_vec[base4 + 3];

        float sub = 0.0f;
        uint32_t lo = w8.x;
        sub = fmaf(decode_fp4_e2m1(lo & 0xFu),         x0.x, sub);
        sub = fmaf(decode_fp4_e2m1((lo >> 4) & 0xFu),  x0.y, sub);
        sub = fmaf(decode_fp4_e2m1((lo >> 8) & 0xFu),  x0.z, sub);
        sub = fmaf(decode_fp4_e2m1((lo >> 12) & 0xFu), x0.w, sub);
        sub = fmaf(decode_fp4_e2m1((lo >> 16) & 0xFu), x1.x, sub);
        sub = fmaf(decode_fp4_e2m1((lo >> 20) & 0xFu), x1.y, sub);
        sub = fmaf(decode_fp4_e2m1((lo >> 24) & 0xFu), x1.z, sub);
        sub = fmaf(decode_fp4_e2m1((lo >> 28) & 0xFu), x1.w, sub);

        uint32_t hi = w8.y;
        sub = fmaf(decode_fp4_e2m1(hi & 0xFu),         x2.x, sub);
        sub = fmaf(decode_fp4_e2m1((hi >> 4) & 0xFu),  x2.y, sub);
        sub = fmaf(decode_fp4_e2m1((hi >> 8) & 0xFu),  x2.z, sub);
        sub = fmaf(decode_fp4_e2m1((hi >> 12) & 0xFu), x2.w, sub);
        sub = fmaf(decode_fp4_e2m1((hi >> 16) & 0xFu), x3.x, sub);
        sub = fmaf(decode_fp4_e2m1((hi >> 20) & 0xFu), x3.y, sub);
        sub = fmaf(decode_fp4_e2m1((hi >> 24) & 0xFu), x3.z, sub);
        sub = fmaf(decode_fp4_e2m1((hi >> 28) & 0xFu), x3.w, sub);

        acc = fmaf(sub, blk_scale, acc);
    }
    return wave32_sum(acc);
}

// Helper: MXFP8 dot product over one row of length k_cols (multiple of 32)
__device__ __forceinline__ float dot_mxfp8_row_wave32(
    const float4* __restrict__ x_vec,
    const uint8_t* __restrict__ w_fp8_row,
    const uint8_t* __restrict__ w_scales_row,
    uint32_t k_cols,
    uint32_t lane
) {
    uint32_t n_blocks32 = k_cols >> 5;
    const uint4* w_vec = reinterpret_cast<const uint4*>(w_fp8_row);
    float acc = 0.0f;
    for (uint32_t b = lane; b < n_blocks32; b += 32u) {
        float blk_scale = decode_ue8m0(w_scales_row[b]);
        uint4 wa = w_vec[b * 2u + 0u];
        uint4 wb = w_vec[b * 2u + 1u];
        uint32_t base4 = b << 3;

        float sub = 0.0f;
        #define DOT4_MX(U32, FV) \
            sub = fmaf(decode_fp8_e4m3fn((U32) & 0xFFu),         (FV).x, sub); \
            sub = fmaf(decode_fp8_e4m3fn(((U32) >> 8) & 0xFFu),  (FV).y, sub); \
            sub = fmaf(decode_fp8_e4m3fn(((U32) >> 16) & 0xFFu), (FV).z, sub); \
            sub = fmaf(decode_fp8_e4m3fn(((U32) >> 24) & 0xFFu), (FV).w, sub);

        DOT4_MX(wa.x, x_vec[base4 + 0]);
        DOT4_MX(wa.y, x_vec[base4 + 1]);
        DOT4_MX(wa.z, x_vec[base4 + 2]);
        DOT4_MX(wa.w, x_vec[base4 + 3]);
        DOT4_MX(wb.x, x_vec[base4 + 4]);
        DOT4_MX(wb.y, x_vec[base4 + 5]);
        DOT4_MX(wb.z, x_vec[base4 + 6]);
        DOT4_MX(wb.w, x_vec[base4 + 7]);
        #undef DOT4_MX

        acc = fmaf(sub, blk_scale, acc);
    }
    return wave32_sum(acc);
}

// ---------------------------------------------------------------------------
// Indirect MoE Pass 1 (w13 + softcapped SwiGLU) for NVFP4 & MXFP8 slabs!
//   Grid : (ceil(intermediate_size / 4), top_k, 1)
//   Block: (128, 1, 1) -> 4 Wave32 warps, each computes 1 (gate, up) pair
//   `slot_table` maps `expert_id (0..127) -> slab slot index` (0xFFFFFFFF = skip!)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_moe_w13_swiglu_indirect(
    float* __restrict__ mid,                // [top_k, intermediate_size]
    const float* __restrict__ x,            // [hidden_size]
    const uint8_t* __restrict__ expert_pool,// contiguous slab pool
    const uint32_t* __restrict__ slot_table,// [n_routed_experts] for this physical layer
    const uint32_t* __restrict__ topk_ids,  // [top_k]
    uint64_t slab_stride_bytes,
    uint32_t hidden_size,
    uint32_t intermediate_size,
    float gate_softcap,
    float up_softcap
) {
    uint32_t k_idx = blockIdx.y;
    uint32_t exp_id = topk_ids[k_idx];
    uint32_t slot = slot_table[exp_id];
    if (slot == 0xFFFFFFFFu) {
        return; // Skipped miss (--no-rescue / threshold mode): zero DRAM waste!
    }

    uint32_t warp_in_block = threadIdx.x >> 5;
    uint32_t lane = threadIdx.x & 31u;
    uint32_t row = blockIdx.x * 4u + warp_in_block;
    if (row >= intermediate_size) return;

    const uint8_t* slab = expert_pool + static_cast<size_t>(slot) * static_cast<size_t>(slab_stride_bytes);
    const ExpertSlabHeaderGpu* hdr = reinterpret_cast<const ExpertSlabHeaderGpu*>(slab);
    const uint8_t* w13_w = slab + hdr->w13_weight_offset;
    const uint8_t* w13_s = slab + hdr->w13_scale_offset;
    const float4* x_vec = reinterpret_cast<const float4*>(x);

    float gate_val = 0.0f;
    float up_val = 0.0f;
    // Inkling-2-Mini stores `w13_weight` interleaved: row 2*r = gate[r], row 2*r+1 = up[r]!
    uint32_t gate_row = row * 2u;
    uint32_t up_row = gate_row + 1u;
    if (hdr->dtype_tag == 0u) {
        // NVFP4: w13 is [2 * intermediate_size, hidden_size]
        size_t w_row_stride = hidden_size >> 1;
        size_t s_row_stride = hidden_size >> 4;
        gate_val = dot_nvfp4_row_wave32(
            x_vec,
            w13_w + static_cast<size_t>(gate_row) * w_row_stride,
            w13_s + static_cast<size_t>(gate_row) * s_row_stride,
            hidden_size,
            lane
        ) * hdr->w13_outer_scale;
        up_val = dot_nvfp4_row_wave32(
            x_vec,
            w13_w + static_cast<size_t>(up_row) * w_row_stride,
            w13_s + static_cast<size_t>(up_row) * s_row_stride,
            hidden_size,
            lane
        ) * hdr->w13_outer_scale;
    } else {
        // MXFP8: on-disk safetensors stores [N, K/32] UE8M0 scales
        size_t w_row_stride = hidden_size;
        size_t s_row_stride = hidden_size >> 5;
        gate_val = dot_mxfp8_row_wave32(
            x_vec,
            w13_w + static_cast<size_t>(gate_row) * w_row_stride,
            w13_s + static_cast<size_t>(gate_row) * s_row_stride,
            hidden_size,
            lane
        );
        up_val = dot_mxfp8_row_wave32(
            x_vec,
            w13_w + static_cast<size_t>(up_row) * w_row_stride,
            w13_s + static_cast<size_t>(up_row) * s_row_stride,
            hidden_size,
            lane
        );
    }

    if (lane == 0u) {
        mid[static_cast<size_t>(k_idx) * intermediate_size + row] =
            situ_f32(gate_val, up_val, gate_softcap, up_softcap);
    }
}

// ---------------------------------------------------------------------------
// Indirect MoE Pass 2 (w2 down-projection + weighted sum across all resident Top-K experts)
//   Grid : (ceil(hidden_size / 4), 1, 1)
//   Block: (128, 1, 1) -> 4 warps, 1 warp per output hidden dim, zero atomics!
//   Automatically computes exact renormalization if any Top-K expert has slot == 0xFFFFFFFF!
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_moe_w2_accum_indirect(
    float* __restrict__ out,                  // [hidden_size]
    const float* __restrict__ mid,            // [top_k, intermediate_size]
    const uint8_t* __restrict__ expert_pool,  // contiguous slab pool
    const uint32_t* __restrict__ slot_table,  // [n_routed_experts] for this physical layer
    const uint32_t* __restrict__ topk_ids,    // [top_k]
    const float* __restrict__ topk_weights,   // [top_k]
    uint64_t slab_stride_bytes,
    uint32_t hidden_size,
    uint32_t intermediate_size,
    uint32_t top_k
) {
    uint32_t warp_in_block = threadIdx.x >> 5;
    uint32_t lane = threadIdx.x & 31u;
    uint32_t row = blockIdx.x * 4u + warp_in_block;
    if (row >= hidden_size) return;

    // Compute exact renormalization factor across resident vs skipped experts in registers.
    float total_w = 0.0f;
    float exec_w = 0.0f;
    for (uint32_t k = 0u; k < top_k; ++k) {
        float w = topk_weights[k];
        total_w += w;
        if (slot_table[topk_ids[k]] != 0xFFFFFFFFu) {
            exec_w += w;
        }
    }
    float renorm = (exec_w > 1e-8f) ? (total_w / exec_w) : 0.0f;

    float total_out = 0.0f;
    for (uint32_t k = 0u; k < top_k; ++k) {
        uint32_t slot = slot_table[topk_ids[k]];
        if (slot == 0xFFFFFFFFu) continue;

        const uint8_t* slab = expert_pool + static_cast<size_t>(slot) * static_cast<size_t>(slab_stride_bytes);
        const ExpertSlabHeaderGpu* hdr = reinterpret_cast<const ExpertSlabHeaderGpu*>(slab);
        const uint8_t* w2_w = slab + hdr->w2_weight_offset;
        const uint8_t* w2_s = slab + hdr->w2_scale_offset;
        const float4* m_vec = reinterpret_cast<const float4*>(mid + static_cast<size_t>(k) * intermediate_size);

        float val = 0.0f;
        if (hdr->dtype_tag == 0u) {
            size_t w_row_stride = intermediate_size >> 1;
            size_t s_row_stride = intermediate_size >> 4;
            val = dot_nvfp4_row_wave32(
                m_vec,
                w2_w + static_cast<size_t>(row) * w_row_stride,
                w2_s + static_cast<size_t>(row) * s_row_stride,
                intermediate_size,
                lane
            ) * hdr->w2_outer_scale;
        } else {
            size_t w_row_stride = intermediate_size;
            size_t s_row_stride = intermediate_size >> 5;
            val = dot_mxfp8_row_wave32(
                m_vec,
                w2_w + static_cast<size_t>(row) * w_row_stride,
                w2_s + static_cast<size_t>(row) * s_row_stride,
                intermediate_size,
                lane
            );
        }
        total_out = fmaf(val, topk_weights[k] * renorm, total_out);
    }

    if (lane == 0u) {
        out[row] = total_out;
    }
}

// ===========================================================================
// Decode MoE v2 (bandwidth-oriented): one warp per weight row, 16-byte loads per lane,
// all 8 experts in parallel (grid.y), unweighted per-expert outputs + fused reduce/post.
// ===========================================================================

// 8 E2M1 nibbles of one word -> floats via byte-permute LUT to FP8 E4M3 codes + hardware cvt.
// Every E2M1 value {0,.5,1,1.5,2,3,4,6} is exact in E4M3 (0x00,30,38,3C,40,44,48,4C).
// Returns even nibbles (values 0,2,4,6) in `ev` and odd nibbles (1,3,5,7) in `od`.
__device__ __forceinline__ void e2m1x8(uint32_t w, float4& ev, float4& od) {
    const uint32_t LUT_LO = 0x3C383000u, LUT_HI = 0x4C484440u;
    uint32_t lo = w & 0x0F0F0F0Fu;
    uint32_t hi = (w >> 4) & 0x0F0F0F0Fu;
    uint32_t c_lo = __builtin_amdgcn_perm(LUT_HI, LUT_LO, lo & 0x07070707u) | ((lo & 0x08080808u) << 4);
    uint32_t c_hi = __builtin_amdgcn_perm(LUT_HI, LUT_LO, hi & 0x07070707u) | ((hi & 0x08080808u) << 4);
    ev = decode_fp8x4(c_lo);
    od = decode_fp8x4(c_hi);
}

// dot of a 16-byte NVFP4 chunk (32 values, 2 scale blocks) with 32 consecutive activations.
__device__ __forceinline__ float nvfp4_chunk_dot(uint4 w, float s0, float s1, const float4* __restrict__ x8) {
    uint32_t wd[4] = {w.x, w.y, w.z, w.w};
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (uint32_t j = 0u; j < 4u; ++j) {
        float4 xa = x8[2u * j], xb = x8[2u * j + 1u];
        float4 ev, od;
        e2m1x8(wd[j], ev, od);
        // word j holds values 8j..8j+7: even nibbles -> xa.x, xa.z, xb.x, xb.z; odd -> xa.y, xa.w, xb.y, xb.w
        float a = ev.x * xa.x;
        a = fmaf(od.x, xa.y, a); a = fmaf(ev.y, xa.z, a); a = fmaf(od.y, xa.w, a);
        a = fmaf(ev.z, xb.x, a); a = fmaf(od.z, xb.y, a); a = fmaf(ev.w, xb.z, a); a = fmaf(od.w, xb.w, a);
        if (j < 2u) acc0 += a; else acc1 += a;
    }
    return acc0 * s0 + acc1 * s1;
}

// dot of a 16-byte MXFP8 chunk (16 values; one UE8M0 scale per 32) with 16 activations.
__device__ __forceinline__ float mxfp8_chunk_dot(uint4 w, const float4* __restrict__ x4) {
    float4 a = decode_fp8x4(w.x), b = decode_fp8x4(w.y), c = decode_fp8x4(w.z), d = decode_fp8x4(w.w);
    float4 xa = x4[0], xb = x4[1], xc = x4[2], xd = x4[3];
    float s = a.x * xa.x;
    s = fmaf(a.y, xa.y, s); s = fmaf(a.z, xa.z, s); s = fmaf(a.w, xa.w, s);
    s = fmaf(b.x, xb.x, s); s = fmaf(b.y, xb.y, s); s = fmaf(b.z, xb.z, s); s = fmaf(b.w, xb.w, s);
    s = fmaf(c.x, xc.x, s); s = fmaf(c.y, xc.y, s); s = fmaf(c.z, xc.z, s); s = fmaf(c.w, xc.w, s);
    s = fmaf(d.x, xd.x, s); s = fmaf(d.y, xd.y, s); s = fmaf(d.z, xd.z, s); s = fmaf(d.w, xd.w, s);
    return s;
}

// Row dot over k_cols activations (one warp), NVFP4 or MXFP8 per the slab header tag.
__device__ __forceinline__ float moe_row_dot(uint32_t dtype_tag, const uint8_t* __restrict__ w_row,
    const uint8_t* __restrict__ s_row, const float* __restrict__ x, uint32_t k_cols, uint32_t lane) {
    float acc = 0.0f;
    const uint4* wv = reinterpret_cast<const uint4*>(w_row);
    const float4* xv = reinterpret_cast<const float4*>(x);
    if (dtype_tag == 0u) {
        uint32_t n16 = k_cols >> 5; // 32 values per 16-byte chunk
        for (uint32_t c = lane; c < n16; c += 32u) {
            float s0 = decode_fp8_e4m3fn(s_row[2u * c]);
            float s1 = decode_fp8_e4m3fn(s_row[2u * c + 1u]);
            acc += nvfp4_chunk_dot(wv[c], s0, s1, xv + 8u * c);
        }
    } else {
        uint32_t n16 = k_cols >> 4; // 16 values per chunk, scale per 32
        for (uint32_t c = lane; c < n16; c += 32u) {
            float sc = decode_ue8m0(s_row[c >> 1]);
            acc = fmaf(mxfp8_chunk_dot(wv[c], xv + 4u * c), sc, acc);
        }
    }
    return wave32_sum(acc);
}

//   mid[k][r] = SiTU(gate_r . x, up_r . x) for routed expert k (rows interleaved 2r / 2r+1).
//   Grid: (ceil(I / 8), top_k), Block: 256 (8 warps, one intermediate row each)
extern "C" __global__ void smalltide_moe_w13_decode(
    float* __restrict__ mid, const float* __restrict__ x, const uint8_t* __restrict__ pool,
    const uint32_t* __restrict__ slot_table, const uint32_t* __restrict__ topk_ids,
    uint64_t slab_stride, uint32_t hidden, uint32_t inter, float gate_cap, float up_cap
) {
    uint32_t k = blockIdx.y;
    uint32_t slot = slot_table[topk_ids[k]];
    if (slot == 0xFFFFFFFFu) return;
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t r = blockIdx.x * (blockDim.x >> 5) + warp;
    if (r >= inter) return;
    const uint8_t* slab = pool + static_cast<size_t>(slot) * slab_stride;
    const ExpertSlabHeaderGpu* hdr = reinterpret_cast<const ExpertSlabHeaderGpu*>(slab);
    const uint8_t* w = slab + hdr->w13_weight_offset;
    const uint8_t* sc = slab + hdr->w13_scale_offset;
    size_t wrow = (hdr->dtype_tag == 0u) ? (hidden >> 1) : hidden;
    size_t srow = (hdr->dtype_tag == 0u) ? (hidden >> 4) : (hidden >> 5);
    float g = moe_row_dot(hdr->dtype_tag, w + (2u * r) * wrow, sc + (2u * r) * srow, x, hidden, lane);
    float u = moe_row_dot(hdr->dtype_tag, w + (2u * r + 1u) * wrow, sc + (2u * r + 1u) * srow, x, hidden, lane);
    if (hdr->dtype_tag == 0u) { g *= hdr->w13_outer_scale; u *= hdr->w13_outer_scale; }
    if (lane == 0u) mid[static_cast<size_t>(k) * inter + r] = situ_f32(g, u, gate_cap, up_cap);
}

//   tmp[k][h] = w2_h . mid[k] (unweighted; 0 for a skipped expert).
//   Grid: (ceil(H / 8), top_k), Block: 256
extern "C" __global__ void smalltide_moe_w2_decode(
    float* __restrict__ tmp, const float* __restrict__ mid, const uint8_t* __restrict__ pool,
    const uint32_t* __restrict__ slot_table, const uint32_t* __restrict__ topk_ids,
    uint64_t slab_stride, uint32_t hidden, uint32_t inter
) {
    uint32_t k = blockIdx.y;
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t h = blockIdx.x * (blockDim.x >> 5) + warp;
    if (h >= hidden) return;
    uint32_t slot = slot_table[topk_ids[k]];
    if (slot == 0xFFFFFFFFu) {
        if (lane == 0u) tmp[static_cast<size_t>(k) * hidden + h] = 0.0f;
        return;
    }
    const uint8_t* slab = pool + static_cast<size_t>(slot) * slab_stride;
    const ExpertSlabHeaderGpu* hdr = reinterpret_cast<const ExpertSlabHeaderGpu*>(slab);
    const uint8_t* w = slab + hdr->w2_weight_offset;
    const uint8_t* sc = slab + hdr->w2_scale_offset;
    size_t wrow = (hdr->dtype_tag == 0u) ? (inter >> 1) : inter;
    size_t srow = (hdr->dtype_tag == 0u) ? (inter >> 4) : (inter >> 5);
    float v = moe_row_dot(hdr->dtype_tag, w + h * wrow, sc + h * srow, mid + static_cast<size_t>(k) * inter, inter, lane);
    if (hdr->dtype_tag == 0u) v *= hdr->w2_outer_scale;
    if (lane == 0u) tmp[static_cast<size_t>(k) * hidden + h] = v;
}

//   moe = sum_k w_k * renorm * tmp[k] (renorm over present experts); optionally fused with the
//   post-MLP short conv and residual: out = residual + sconv(moe) (sconv_state non-null).
//   Grid: ceil(H / 256), Block: 256
extern "C" __global__ void smalltide_moe_post_decode(
    float* __restrict__ out, const float* __restrict__ tmp, const uint32_t* __restrict__ slot_table,
    const uint32_t* __restrict__ topk_ids, const float* __restrict__ topk_w, uint32_t top_k,
    float* __restrict__ sconv_state, const float* __restrict__ sconv_w, const float* __restrict__ residual,
    uint32_t hidden
) {
    uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    float tot = 0.0f, ex = 0.0f;
    for (uint32_t k = 0u; k < top_k; ++k) {
        float w = topk_w[k];
        tot += w;
        if (slot_table[topk_ids[k]] != 0xFFFFFFFFu) ex += w;
    }
    float renorm = (ex > 1e-8f) ? (tot / ex) : 0.0f;
    float acc = 0.0f;
    for (uint32_t k = 0u; k < top_k; ++k) {
        if (slot_table[topk_ids[k]] != 0xFFFFFFFFu) acc = fmaf(topk_w[k] * renorm, tmp[static_cast<size_t>(k) * hidden + h], acc);
    }
    if (sconv_state != nullptr) {
        float s0 = sconv_state[h], s1 = sconv_state[hidden + h], s2 = sconv_state[2u * hidden + h];
        const float4 w = reinterpret_cast<const float4*>(sconv_w)[h];
        float y = fmaf(s0, w.x, fmaf(s1, w.y, fmaf(s2, w.z, acc * w.w)));
        y = (acc + y) * 0.4472135954999579f;
        sconv_state[h] = s1; sconv_state[hidden + h] = s2; sconv_state[2u * hidden + h] = acc;
        out[h] = residual[h] + y;
    } else {
        out[h] = acc;
    }
}

// ===========================================================================
// Decode MoE v3: one block per (expert, 32-row tile); slot + slab header resolved once per
// block into LDS (instead of 3-4 dependent global loads per warp), each warp owns 4 rows and
// keeps two rows' loads in flight.
// ===========================================================================
#define ST_MOE3_ROWS_PER_BLOCK 32u
#define ST_MOE3_MAX_K 2304u // hidden (w13) / intermediate (w2) staged in LDS

struct MoeBlockCtx {
    const uint8_t* w;
    const uint8_t* sc;
    float outer;
    uint32_t tag;
    uint32_t valid;
};

__device__ __forceinline__ void moe3_resolve(MoeBlockCtx* ctx, const uint8_t* __restrict__ pool,
    const uint32_t* __restrict__ slot_table, const uint32_t* __restrict__ topk_ids, uint32_t k,
    uint64_t slab_stride, bool w13) {
    if (threadIdx.x == 0u) {
        uint32_t slot = slot_table[topk_ids[k]];
        ctx->valid = (slot != 0xFFFFFFFFu) ? 1u : 0u;
        if (slot != 0xFFFFFFFFu) {
            const uint8_t* slab = pool + static_cast<size_t>(slot) * slab_stride;
            const ExpertSlabHeaderGpu* hdr = reinterpret_cast<const ExpertSlabHeaderGpu*>(slab);
            ctx->w = slab + (w13 ? hdr->w13_weight_offset : hdr->w2_weight_offset);
            ctx->sc = slab + (w13 ? hdr->w13_scale_offset : hdr->w2_scale_offset);
            ctx->tag = hdr->dtype_tag;
            ctx->outer = (hdr->dtype_tag == 0u) ? (w13 ? hdr->w13_outer_scale : hdr->w2_outer_scale) : 1.0f;
        }
    }
    __syncthreads();
}

// Two independent row dots per warp (more loads in flight).
__device__ __forceinline__ void moe3_row_dot2(uint32_t tag, const uint8_t* __restrict__ wa,
    const uint8_t* __restrict__ sa, const uint8_t* __restrict__ wb, const uint8_t* __restrict__ sb,
    const float* __restrict__ x, uint32_t k_cols, uint32_t lane, float* oa, float* ob) {
    float acc_a = 0.0f, acc_b = 0.0f;
    const uint4* va = reinterpret_cast<const uint4*>(wa);
    const uint4* vb = reinterpret_cast<const uint4*>(wb);
    const float4* xv = reinterpret_cast<const float4*>(x);
    if (tag == 0u) {
        uint32_t n16 = k_cols >> 5;
        for (uint32_t c = lane; c < n16; c += 32u) {
            uint4 a = va[c], b = vb[c];
            float a0 = decode_fp8_e4m3fn(sa[2u * c]), a1 = decode_fp8_e4m3fn(sa[2u * c + 1u]);
            float b0 = decode_fp8_e4m3fn(sb[2u * c]), b1 = decode_fp8_e4m3fn(sb[2u * c + 1u]);
            acc_a += nvfp4_chunk_dot(a, a0, a1, xv + 8u * c);
            acc_b += nvfp4_chunk_dot(b, b0, b1, xv + 8u * c);
        }
    } else {
        uint32_t n16 = k_cols >> 4;
        for (uint32_t c = lane; c < n16; c += 32u) {
            uint4 a = va[c], b = vb[c];
            float sa_ = decode_ue8m0(sa[c >> 1]), sb_ = decode_ue8m0(sb[c >> 1]);
            acc_a = fmaf(mxfp8_chunk_dot(a, xv + 4u * c), sa_, acc_a);
            acc_b = fmaf(mxfp8_chunk_dot(b, xv + 4u * c), sb_, acc_b);
        }
    }
    *oa = wave32_sum(acc_a);
    *ob = wave32_sum(acc_b);
}

//   Grid: (ceil(I / 32), top_k), Block: 256 (8 warps x 4 intermediate rows)
extern "C" __global__ void smalltide_moe_w13_decode3(
    float* __restrict__ mid, const float* __restrict__ x, const uint8_t* __restrict__ pool,
    const uint32_t* __restrict__ slot_table, const uint32_t* __restrict__ topk_ids,
    uint64_t slab_stride, uint32_t hidden, uint32_t inter, float gate_cap, float up_cap
) {
    __shared__ MoeBlockCtx ctx;
    __shared__ float4 sx4[ST_MOE3_MAX_K / 4];
    uint32_t k = blockIdx.y;
    // Stage the activation vector in LDS once per block (it is re-read for every weight chunk).
    for (uint32_t i = threadIdx.x; i < (hidden >> 2); i += blockDim.x) sx4[i] = reinterpret_cast<const float4*>(x)[i];
    moe3_resolve(&ctx, pool, slot_table, topk_ids, k, slab_stride, true);
    if (ctx.valid == 0u) return;
    const float* xs = reinterpret_cast<const float*>(sx4);
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    size_t wrow = (ctx.tag == 0u) ? (hidden >> 1) : hidden;
    size_t srow = (ctx.tag == 0u) ? (hidden >> 4) : (hidden >> 5);
    uint32_t r0 = blockIdx.x * ST_MOE3_ROWS_PER_BLOCK + warp * 4u;
    for (uint32_t rr = 0u; rr < 4u; ++rr) {
        uint32_t r = r0 + rr;
        if (r >= inter) break;
        float g, u;
        moe3_row_dot2(ctx.tag, ctx.w + (2u * r) * wrow, ctx.sc + (2u * r) * srow,
                      ctx.w + (2u * r + 1u) * wrow, ctx.sc + (2u * r + 1u) * srow, xs, hidden, lane, &g, &u);
        if (lane == 0u) mid[static_cast<size_t>(k) * inter + r] = situ_f32(g * ctx.outer, u * ctx.outer, gate_cap, up_cap);
    }
}

//   Grid: (ceil(H / 32), top_k), Block: 256 (8 warps x 4 output rows, 2 at a time)
extern "C" __global__ void smalltide_moe_w2_decode3(
    float* __restrict__ tmp, const float* __restrict__ mid, const uint8_t* __restrict__ pool,
    const uint32_t* __restrict__ slot_table, const uint32_t* __restrict__ topk_ids,
    uint64_t slab_stride, uint32_t hidden, uint32_t inter
) {
    __shared__ MoeBlockCtx ctx;
    __shared__ float4 sm4[ST_MOE3_MAX_K / 4];
    uint32_t k = blockIdx.y;
    {
        const float4* src = reinterpret_cast<const float4*>(mid + static_cast<size_t>(k) * inter);
        for (uint32_t i = threadIdx.x; i < (inter >> 2); i += blockDim.x) sm4[i] = src[i];
    }
    moe3_resolve(&ctx, pool, slot_table, topk_ids, k, slab_stride, false);
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t h0 = blockIdx.x * ST_MOE3_ROWS_PER_BLOCK + warp * 4u;
    if (ctx.valid == 0u) {
        for (uint32_t i = lane; i < 4u; i += 32u) {
            if (h0 + i < hidden) tmp[static_cast<size_t>(k) * hidden + h0 + i] = 0.0f;
        }
        return;
    }
    size_t wrow = (ctx.tag == 0u) ? (inter >> 1) : inter;
    size_t srow = (ctx.tag == 0u) ? (inter >> 4) : (inter >> 5);
    const float* m = reinterpret_cast<const float*>(sm4);
    for (uint32_t hh = 0u; hh < 4u; hh += 2u) {
        uint32_t h = h0 + hh;
        if (h >= hidden) break;
        uint32_t h2 = (h + 1u < hidden) ? h + 1u : h;
        float a, b2;
        moe3_row_dot2(ctx.tag, ctx.w + h * wrow, ctx.sc + h * srow, ctx.w + h2 * wrow, ctx.sc + h2 * srow,
                      m, inter, lane, &a, &b2);
        if (lane == 0u) {
            tmp[static_cast<size_t>(k) * hidden + h] = a * ctx.outer;
            if (h + 1u < hidden) tmp[static_cast<size_t>(k) * hidden + h + 1u] = b2 * ctx.outer;
        }
    }
}
