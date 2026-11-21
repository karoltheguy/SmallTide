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
