// SmallTide RDNA 4 (gfx1201) HIP Kernels
// Compiled offline via: hipcc --genco --offload-arch=gfx1201 -O3
// Loaded dynamically at runtime through amdhip64_7.dll (hipModuleLoadData).

#include <hip/hip_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Fast Exact Low-Precision Decoders (FP4 E2M1, FP8 E4M3FN, UE8M0)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float decode_fp4_e2m1(uint32_t nibble) {
    // 1 sign bit, 2 exp bits (bias 1), 1 mantissa bit: {0, .5, 1, 1.5, 2, 3, 4, 6}, from a
    // packed-register LUT (an indexed local array compiles to a per-lane memory lookup).
    float v = static_cast<float>((0xC8643210u >> ((nibble & 7u) << 2)) & 0xFu) * 0.5f;
    return (nibble & 8u) ? -v : v;
}

__device__ __forceinline__ float decode_fp8_e4m3fn(uint32_t b) {
    // gfx12 converts OCP FP8 E4M3FN in hardware (v_cvt_f32_fp8). Producers never emit the NaN
    // codes 0x7F/0xFF (the encoders saturate at 0x7E), so no NaN guard is needed.
    return __builtin_amdgcn_cvt_f32_fp8(static_cast<int>(b & 0xFFu), 0);
}

/// Software reference decoder (kept for parity tests).
__device__ __forceinline__ float decode_fp8_e4m3fn_sw(uint32_t b) {
    uint32_t sign = (b & 0x80u) << 24;
    uint32_t exp  = (b >> 3) & 0x0Fu;
    uint32_t mant = b & 0x07u;
    if (exp == 0u) {
        float val = static_cast<float>(mant) * (1.0f / 512.0f);
        return sign ? -val : val;
    }
    if (exp == 15u && mant == 7u) {
        return 0.0f;
    }
    uint32_t f32_bits = sign | ((exp + 120u) << 23) | (mant << 20);
    return __uint_as_float(f32_bits);
}

/// 4 FP8 E4M3FN bytes of a word -> 4 floats (2x v_cvt_pk_f32_fp8).
__device__ __forceinline__ float4 decode_fp8x4(uint32_t w) {
    auto lo = __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(w), false);
    auto hi = __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(w), true);
    return make_float4(lo[0], lo[1], hi[0], hi[1]);
}

/// FP4 E2M1 magnitude x2 from a packed register LUT (no memory lookup): {0,1,2,3,4,6,8,12}.
__device__ __forceinline__ float fp4x2(uint32_t n) {
    float v = static_cast<float>((0xC8643210u >> ((n & 7u) << 2)) & 0xFu);
    return (n & 8u) ? -v : v;
}

__device__ __forceinline__ float decode_ue8m0(uint32_t e) {
    if (e == 0u) return 0.0f;
    if (e == 255u) return 0.0f;
    return __uint_as_float(e << 23);
}

// Wave32 warp reduction (RDNA 4 native wavefront size = 32)
__device__ __forceinline__ float wave32_sum(float val) {
    val += __shfl_xor(val, 16, 32);
    val += __shfl_xor(val, 8, 32);
    val += __shfl_xor(val, 4, 32);
    val += __shfl_xor(val, 2, 32);
    val += __shfl_xor(val, 1, 32);
    return val;
}

// ---------------------------------------------------------------------------
// 1. No-op Kernel (Phase 0/1 WDDM Launch Latency vs hipGraph Benchmark)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_noop(uint32_t* counter) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && counter != nullptr) {
        counter[0] += 1u;
    }
}

// ---------------------------------------------------------------------------
// 2. RMSNorm (FP32 in/out, FP32 gamma) - 1 block (256 threads) per row
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_rmsnorm_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    uint32_t hidden,
    float eps
) {
    uint32_t row = blockIdx.x;
    const float* x_row = x + static_cast<size_t>(row) * hidden;
    float* o_row = out + static_cast<size_t>(row) * hidden;

    float local_sq = 0.0f;
    for (uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = x_row[i];
        local_sq += v * v;
    }

    __shared__ float warp_sums[32];
    float wsum = wave32_sum(local_sq);
    uint32_t lane = threadIdx.x & 31u;
    uint32_t wid  = threadIdx.x >> 5;
    if (lane == 0u) warp_sums[wid] = wsum;
    __syncthreads();

    float total_sq = 0.0f;
    if (wid == 0u) {
        total_sq = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        total_sq = wave32_sum(total_sq);
    }
    __shared__ float inv_rms;
    if (threadIdx.x == 0u) {
        inv_rms = rsqrtf(total_sq / static_cast<float>(hidden) + eps);
    }
    __syncthreads();

    float s = inv_rms;
    for (uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        o_row[i] = x_row[i] * s * weight[i];
    }
}

// ---------------------------------------------------------------------------
// 3. FP8 E4M3FN GEMV (Batch = 1 decode): y[n] = sum_k x[k] * (W_fp8[n,k] * row_scale[n])
//    Grid: (ceil(N / 4), 1, 1), Block: (128, 1, 1) -> 4 warps (Wave32), 1 warp per output row!
//    K must be a multiple of 16 (Inkling-2-Mini: 2304, 3072, 1152 -> all multiples of 128!).
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_gemv_fp8_e4m3_f32(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ w_fp8,
    const float* __restrict__ row_scale,
    uint32_t n_rows,
    uint32_t k_cols
) {
    uint32_t warp_in_block = threadIdx.x >> 5;
    uint32_t lane = threadIdx.x & 31u;
    uint32_t row = blockIdx.x * 4u + warp_in_block;
    if (row >= n_rows) return;

    const uint4* w_vec = reinterpret_cast<const uint4*>(w_fp8 + static_cast<size_t>(row) * k_cols);
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    uint32_t k_chunks16 = k_cols >> 4; // 16 bytes = 16 FP8 elements per uint4

    // Two 16-byte weight loads in flight per lane per iteration, 4 independent accumulators,
    // packed hardware FP8 conversion.
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    auto dot16 = [&](uint4 w16, uint32_t c) {
        uint32_t base4 = c << 2;
        float4 x0 = x_vec[base4 + 0], x1 = x_vec[base4 + 1], x2 = x_vec[base4 + 2], x3 = x_vec[base4 + 3];
        float4 d0 = decode_fp8x4(w16.x), d1 = decode_fp8x4(w16.y), d2 = decode_fp8x4(w16.z), d3 = decode_fp8x4(w16.w);
        a0 = fmaf(d0.x, x0.x, a0); a0 = fmaf(d0.y, x0.y, a0); a0 = fmaf(d0.z, x0.z, a0); a0 = fmaf(d0.w, x0.w, a0);
        a1 = fmaf(d1.x, x1.x, a1); a1 = fmaf(d1.y, x1.y, a1); a1 = fmaf(d1.z, x1.z, a1); a1 = fmaf(d1.w, x1.w, a1);
        a2 = fmaf(d2.x, x2.x, a2); a2 = fmaf(d2.y, x2.y, a2); a2 = fmaf(d2.z, x2.z, a2); a2 = fmaf(d2.w, x2.w, a2);
        a3 = fmaf(d3.x, x3.x, a3); a3 = fmaf(d3.y, x3.y, a3); a3 = fmaf(d3.z, x3.z, a3); a3 = fmaf(d3.w, x3.w, a3);
    };
    uint32_t c = lane;
    for (; c + 32u < k_chunks16; c += 64u) {
        uint4 wa = w_vec[c];
        uint4 wb = w_vec[c + 32u];
        dot16(wa, c);
        dot16(wb, c + 32u);
    }
    if (c < k_chunks16) dot16(w_vec[c], c);
    float acc = (a0 + a1) + (a2 + a3);

    float sum = wave32_sum(acc);
    if (lane == 0u) {
        float sc = (row_scale != nullptr) ? row_scale[row] : 1.0f;
        y[row] = sum * sc;
    }
}

// ---------------------------------------------------------------------------
// 4. NVFP4 (ModelOpt) GEMV (Batch = 1 decode):
//    Block size = 16 elements = 8 packed bytes (`uint2`) + 1 FP8 E4M3 scale byte.
//    Global `outer_scale` multiplies the row sum.
//    Grid: (ceil(N / 4), 1, 1), Block: (128, 1, 1) -> 4 warps, 1 warp per output row.
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_gemv_nvfp4_f32(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ w_packed, // [N, K/2]
    const uint8_t* __restrict__ w_scales, // [N, K/16] FP8 E4M3
    float outer_scale,
    uint32_t n_rows,
    uint32_t k_cols
) {
    uint32_t warp_in_block = threadIdx.x >> 5;
    uint32_t lane = threadIdx.x & 31u;
    uint32_t row = blockIdx.x * 4u + warp_in_block;
    if (row >= n_rows) return;

    uint32_t n_blocks16 = k_cols >> 4;
    const uint2* w_row = reinterpret_cast<const uint2*>(w_packed + static_cast<size_t>(row) * (k_cols >> 1));
    const uint8_t* s_row = w_scales + static_cast<size_t>(row) * n_blocks16;
    const float4* x_vec = reinterpret_cast<const float4*>(x);

    float acc = 0.0f;
    for (uint32_t b = lane; b < n_blocks16; b += 32u) {
        uint2 w8 = w_row[b]; // 8 bytes = 16 FP4 nibbles
        float blk_scale = decode_fp8_e4m3fn(s_row[b]);
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

    float sum = wave32_sum(acc);
    if (lane == 0u) {
        y[row] = sum * outer_scale;
    }
}

// ---------------------------------------------------------------------------
// 5. MXFP8 GEMV (Batch = 1 decode, Inkling-2-Mini Layers 20-23):
//    Verified on-disk safetensors shape: `w_scales` is [N, K/32] UE8M0 (`[128, 2304, 72]`).
//    Grid: (ceil(N / 4), 1, 1), Block: (128, 1, 1) -> 4 warps, 1 warp per output row.
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_gemv_mxfp8_f32(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ w_fp8,    // [N, K]
    const uint8_t* __restrict__ w_scales, // [N, K/32] UE8M0
    uint32_t n_rows,
    uint32_t k_cols
) {
    uint32_t warp_in_block = threadIdx.x >> 5;
    uint32_t lane = threadIdx.x & 31u;
    uint32_t row = blockIdx.x * 4u + warp_in_block;
    if (row >= n_rows) return;

    uint32_t n_blocks32 = k_cols >> 5;
    const uint4* w_vec = reinterpret_cast<const uint4*>(w_fp8 + static_cast<size_t>(row) * k_cols);
    const uint8_t* s_row = w_scales + static_cast<size_t>(row) * n_blocks32;
    const float4* x_vec = reinterpret_cast<const float4*>(x);

    float acc = 0.0f;
    for (uint32_t b = lane; b < n_blocks32; b += 32u) {
        float blk_scale = decode_ue8m0(s_row[b]);
        uint4 wa = w_vec[b * 2u + 0u];
        uint4 wb = w_vec[b * 2u + 1u];
        uint32_t base4 = b << 3;

        float sub = 0.0f;
        #define DOT4_FP8(U32, FV) \
            sub = fmaf(decode_fp8_e4m3fn((U32) & 0xFFu),         (FV).x, sub); \
            sub = fmaf(decode_fp8_e4m3fn(((U32) >> 8) & 0xFFu),  (FV).y, sub); \
            sub = fmaf(decode_fp8_e4m3fn(((U32) >> 16) & 0xFFu), (FV).z, sub); \
            sub = fmaf(decode_fp8_e4m3fn(((U32) >> 24) & 0xFFu), (FV).w, sub);

        DOT4_FP8(wa.x, x_vec[base4 + 0]);
        DOT4_FP8(wa.y, x_vec[base4 + 1]);
        DOT4_FP8(wa.z, x_vec[base4 + 2]);
        DOT4_FP8(wa.w, x_vec[base4 + 3]);
        DOT4_FP8(wb.x, x_vec[base4 + 4]);
        DOT4_FP8(wb.y, x_vec[base4 + 5]);
        DOT4_FP8(wb.z, x_vec[base4 + 6]);
        DOT4_FP8(wb.w, x_vec[base4 + 7]);
        #undef DOT4_FP8

        acc = fmaf(sub, blk_scale, acc);
    }

    float sum = wave32_sum(acc);
    if (lane == 0u) {
        y[row] = sum;
    }
}

// Paged KV: a sequence position's row in a KV pool shared by all sequences. `page_table` maps
// the sequence's pages (KV_PAGE_POS positions each) to pool pages; null: the position itself
// (an unpaged cache, e.g. a sliding-window ring).
#define KV_PAGE_SHIFT 11u
__device__ __forceinline__ uint32_t kv_phys(const uint32_t* __restrict__ page_table, uint32_t slot) {
    return (page_table != nullptr)
        ? ((page_table[slot >> KV_PAGE_SHIFT] << KV_PAGE_SHIFT) | (slot & ((1u << KV_PAGE_SHIFT) - 1u)))
        : slot;
}

#include "moe_ops.hip"
#include "attn_ops.hip"
#include "inkling2_ops.hip"
#include "batched_ops.hip"
#include "prefill_ops.hip"

// ---------------------------------------------------------------------------
// Multi-segment GEMV: up to 8 independent y_s = W_s . x_s in one launch (FP8 E4M3 row-scaled or
// F32 weights per segment). One warp per output row; the warp finds its segment by row prefix.
//    Grid: ceil(total_rows / 4), Block: 128
// ---------------------------------------------------------------------------
struct GemvSeg {
    const uint8_t* w;
    const float* scale; // FP8 row scales (unused for F32)
    const float* x;
    float* y;
    uint32_t n_rows;
    uint32_t k_cols;
    uint32_t is_f32;
    uint32_t row_start;
};
struct GemvMultiArgs {
    GemvSeg seg[8];
    uint32_t n_seg;
    uint32_t total_rows;
};

extern "C" __global__ void smalltide_gemv_multi(GemvMultiArgs a) {
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t grow = blockIdx.x * 4u + warp;
    if (grow >= a.total_rows) return;
    uint32_t si = 0u;
    #pragma unroll
    for (uint32_t i = 1u; i < 8u; ++i) {
        if (i < a.n_seg && grow >= a.seg[i].row_start) si = i;
    }
    const GemvSeg sg = a.seg[si];
    uint32_t row = grow - sg.row_start;
    const float4* xv = reinterpret_cast<const float4*>(sg.x);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (sg.is_f32 != 0u) {
        const float4* wv = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(sg.w) + static_cast<size_t>(row) * sg.k_cols);
        for (uint32_t i = lane; i < (sg.k_cols >> 2); i += 32u) {
            float4 w4 = wv[i], x4 = xv[i];
            a0 = fmaf(w4.x, x4.x, a0); a1 = fmaf(w4.y, x4.y, a1); a2 = fmaf(w4.z, x4.z, a2); a3 = fmaf(w4.w, x4.w, a3);
        }
        float s = wave32_sum((a0 + a1) + (a2 + a3));
        if (lane == 0u) sg.y[row] = s;
        return;
    }
    const uint4* w_vec = reinterpret_cast<const uint4*>(sg.w + static_cast<size_t>(row) * sg.k_cols);
    uint32_t k16 = sg.k_cols >> 4;
    auto dot16 = [&](uint4 w16, uint32_t c) {
        uint32_t b4 = c << 2;
        float4 x0 = xv[b4 + 0], x1 = xv[b4 + 1], x2 = xv[b4 + 2], x3 = xv[b4 + 3];
        float4 d0 = decode_fp8x4(w16.x), d1 = decode_fp8x4(w16.y), d2 = decode_fp8x4(w16.z), d3 = decode_fp8x4(w16.w);
        a0 = fmaf(d0.x, x0.x, a0); a0 = fmaf(d0.y, x0.y, a0); a0 = fmaf(d0.z, x0.z, a0); a0 = fmaf(d0.w, x0.w, a0);
        a1 = fmaf(d1.x, x1.x, a1); a1 = fmaf(d1.y, x1.y, a1); a1 = fmaf(d1.z, x1.z, a1); a1 = fmaf(d1.w, x1.w, a1);
        a2 = fmaf(d2.x, x2.x, a2); a2 = fmaf(d2.y, x2.y, a2); a2 = fmaf(d2.z, x2.z, a2); a2 = fmaf(d2.w, x2.w, a2);
        a3 = fmaf(d3.x, x3.x, a3); a3 = fmaf(d3.y, x3.y, a3); a3 = fmaf(d3.z, x3.z, a3); a3 = fmaf(d3.w, x3.w, a3);
    };
    uint32_t c = lane;
    for (; c + 32u < k16; c += 64u) {
        uint4 wa = w_vec[c];
        uint4 wb = w_vec[c + 32u];
        dot16(wa, c);
        dot16(wb, c + 32u);
    }
    if (c < k16) dot16(w_vec[c], c);
    float s = wave32_sum((a0 + a1) + (a2 + a3));
    if (lane == 0u) sg.y[row] = s * sg.scale[row];
}

// ---------------------------------------------------------------------------
// Fused decode attention prep (one launch instead of ~10):
//   blocks [0, n_kv)           : K and V of one KV head: causal sconv (static + optional dynamic
//                                taps, state update), K L2 clip, V unit RMS norm, FP8 KV store
//   blocks [n_kv, n_kv+n_q)    : Q head: scaled L2 clip, then tau (global layers)
//   blocks [n_kv+n_q, +n_q)    : R head (d_rel=16): scaled L2 clip, tau, relative bias table row
//   Block: 128 threads (= head_dim)
// ---------------------------------------------------------------------------
struct AttnPrepArgs {
    float* q;               // [n_q, 128] in place
    float* r;               // [n_q, 16] in place
    const float* k_raw;     // [n_kv*128]
    const float* v_raw;
    const float* k_dyn;     // nullable [n_kv*4]
    const float* v_dyn;
    float* sconv_k;         // [3][n_kv*128]
    float* sconv_v;
    const float* k_w;       // [n_kv*128][4]
    const float* v_w;
    const float* q_scale;   // [n_q][128]
    const float* r_scale;   // [n_q][16]
    const float* rel_proj;  // [rel_extent][16]
    float* rel_table;       // [n_q][rel_extent]
    uint8_t* k_cache;
    float* k_scales;
    uint8_t* v_cache;
    float* v_scales;
    uint32_t n_q;
    uint32_t n_kv;
    uint32_t max_seq;
    uint32_t slot;
    uint32_t rel_extent;
    float qk_thr;
    float rel_thr;
    float tau;
    float eps;
    const uint32_t* page_table; // nullable (see kv_phys)
};

__device__ __forceinline__ float block128_sum(float v, float* sh) {
    v = wave32_sum(v);
    __syncthreads();
    if ((threadIdx.x & 31u) == 0u) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    return sh[0] + sh[1] + sh[2] + sh[3];
}

__device__ __forceinline__ float block128_max(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
    __syncthreads();
    if ((threadIdx.x & 31u) == 0u) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    return fmaxf(fmaxf(sh[0], sh[1]), fmaxf(sh[2], sh[3]));
}

__device__ __forceinline__ float sconv_step(float* st, const float* w4, const float* dyn, uint32_t c, uint32_t dim,
                                            float cur, uint32_t head) {
    float s0 = st[c], s1 = st[dim + c], s2 = st[2u * dim + c];
    const float4 w = reinterpret_cast<const float4*>(w4)[c];
    float y = fmaf(s0, w.x, fmaf(s1, w.y, fmaf(s2, w.z, cur * w.w)));
    if (dyn != nullptr) {
        const float4 d = reinterpret_cast<const float4*>(dyn)[head];
        y = fmaf(cur, d.x, fmaf(s2, d.y, fmaf(s1, d.z, fmaf(s0, d.w, y))));
        y = (cur + y) * (1.0f / 3.0f);
    } else {
        y = (cur + y) * 0.4472135954999579f;
    }
    st[c] = s1; st[dim + c] = s2; st[2u * dim + c] = cur;
    return y;
}

extern "C" __global__ void smalltide_attn_prep_decode(AttnPrepArgs a) {
    __shared__ float sh[4];
    uint32_t b = blockIdx.x, t = threadIdx.x;
    if (b < a.n_kv) {
        uint32_t h = b, dim = a.n_kv * 128u, c = h * 128u + t;
        float k = sconv_step(a.sconv_k, a.k_w, a.k_dyn, c, dim, a.k_raw[c], h);
        float v = sconv_step(a.sconv_v, a.v_w, a.v_dyn, c, dim, a.v_raw[c], h);
        // K: L2 clip
        float kn = sqrtf(block128_sum(k * k, sh)) + 1e-6f;
        k *= fminf(a.qk_thr / kn, 1.0f);
        // V: unit RMS norm (no gamma)
        float vs = block128_sum(v * v, sh);
        v *= rsqrtf(vs / 128.0f + a.eps);
        // FP8 store with per-(head, token) absmax scales
        float km = block128_max(fabsf(k), sh);
        float vm = block128_max(fabsf(v), sh);
        float ks = (km > 0.0f) ? km / 448.0f : 1.0f;
        float vsc = (vm > 0.0f) ? vm / 448.0f : 1.0f;
        size_t si = static_cast<size_t>(kv_phys(a.page_table, a.slot)) * a.n_kv + h;
        if (t == 0u) { a.k_scales[si] = ks; a.v_scales[si] = vsc; }
        a.k_cache[si * 128u + t] = encode_fp8_e4m3fn(k / ks);
        a.v_cache[si * 128u + t] = encode_fp8_e4m3fn(v / vsc);
        return;
    }
    b -= a.n_kv;
    if (b < a.n_q) {
        float* row = a.q + b * 128u;
        float x = row[t] * a.q_scale[b * 128u + t];
        float n = sqrtf(block128_sum(x * x, sh)) + 1e-6f;
        row[t] = x * fminf(a.qk_thr / n, 1.0f) * a.tau;
        return;
    }
    b -= a.n_q;
    if (b < a.n_q) {
        __shared__ float rr[16];
        float x = (t < 16u) ? a.r[b * 16u + t] * a.r_scale[b * 16u + t] : 0.0f;
        float n = sqrtf(block128_sum(x * x, sh)) + 1e-6f;
        float y = x * fminf(a.rel_thr / n, 1.0f) * a.tau;
        if (t < 16u) { a.r[b * 16u + t] = y; rr[t] = y; }
        __syncthreads();
        for (uint32_t d = t; d < a.rel_extent; d += 128u) {
            const float4* pr = reinterpret_cast<const float4*>(a.rel_proj + static_cast<size_t>(d) * 16u);
            float acc = 0.0f;
            #pragma unroll
            for (uint32_t i = 0u; i < 4u; ++i) {
                float4 pv = pr[i];
                acc = fmaf(pv.x, rr[4u * i], fmaf(pv.y, rr[4u * i + 1u], fmaf(pv.z, rr[4u * i + 2u], fmaf(pv.w, rr[4u * i + 3u], acc))));
            }
            a.rel_table[static_cast<size_t>(b) * a.rel_extent + d] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Fused single-workgroup decode mix (Block: 1024 threads, hidden <= 3072, <= 15 blocks):
//   A (o_raw != null): x = sconv(o_raw); boundary ? (blocks[nb] = residual, cur = x)
//                                                 : cur = residual + x          -> writes cur
//   B (copy_out != null): copy_out = cur
//   C: m_q = attn-residual mix of [blocks[0..nb'], cur] with query q (identity if queries null)
//   D: mix_out[q] = m_q (optional); norm_out[q] = rmsnorm(m_q) * norm_w[q] (if norm_w0 != null)
// ---------------------------------------------------------------------------
struct FusedMixArgs {
    float* cur;
    const float* o_raw;
    float* sconv_state;
    const float* sconv_w;
    const float* residual;
    float* blocks;
    float* copy_out;
    const float* queries;
    float* mix_out;
    const float* norm_w0;
    const float* norm_w1;
    const float* norm_w2;
    float* norm_out;
    uint32_t nb;
    uint32_t boundary;
    uint32_t nq;
    uint32_t hidden;
    float eps;
};

__device__ __forceinline__ void fm_reduce4(float v0, float v1, float v2, float v3, float (*red)[32], float* out) {
    uint32_t lane = threadIdx.x & 31u, w = threadIdx.x >> 5;
    v0 = wave32_sum(v0); v1 = wave32_sum(v1); v2 = wave32_sum(v2); v3 = wave32_sum(v3);
    if (lane == 0u) { red[0][w] = v0; red[1][w] = v1; red[2][w] = v2; red[3][w] = v3; }
    __syncthreads();
    if (threadIdx.x < 4u) {
        float s = 0.0f;
        for (uint32_t k = 0u; k < (blockDim.x >> 5); ++k) s += red[threadIdx.x][k];
        out[threadIdx.x] = s;
    }
    __syncthreads();
}

extern "C" __global__ void smalltide_fused_mix_decode(FusedMixArgs a) {
    __shared__ float red[4][32];
    __shared__ float tot[4];
    __shared__ float s_inv[16];
    __shared__ float s_w[16][3];
    const uint32_t H = a.hidden, nt = blockDim.x, t = threadIdx.x;
    float c[3];
    #pragma unroll
    for (uint32_t e = 0u; e < 3u; ++e) {
        uint32_t h = t + e * nt;
        c[e] = (h < H) ? a.cur[h] : 0.0f;
    }
    uint32_t nbm = a.nb;
    if (a.o_raw != nullptr) {
        #pragma unroll
        for (uint32_t e = 0u; e < 3u; ++e) {
            uint32_t h = t + e * nt;
            if (h < H) {
                float s0 = a.sconv_state[h], s1 = a.sconv_state[H + h], s2 = a.sconv_state[2u * H + h];
                float xin = a.o_raw[h];
                const float4 w = reinterpret_cast<const float4*>(a.sconv_w)[h];
                float y = fmaf(s0, w.x, fmaf(s1, w.y, fmaf(s2, w.z, xin * w.w)));
                y = (xin + y) * 0.4472135954999579f;
                a.sconv_state[h] = s1; a.sconv_state[H + h] = s2; a.sconv_state[2u * H + h] = xin;
                float res = a.residual[h];
                if (a.boundary != 0u) {
                    a.blocks[static_cast<size_t>(a.nb) * H + h] = res;
                    c[e] = y;
                } else {
                    c[e] = res + y;
                }
                a.cur[h] = c[e];
            }
        }
        if (a.boundary != 0u) nbm = a.nb + 1u;
    }
    if (a.copy_out != nullptr) {
        #pragma unroll
        for (uint32_t e = 0u; e < 3u; ++e) {
            uint32_t h = t + e * nt;
            if (h < H) a.copy_out[h] = c[e];
        }
    }
    const uint32_t nq = a.nq;
    float m[3][3];
    if (a.queries == nullptr) {
        #pragma unroll
        for (uint32_t q = 0u; q < 3u; ++q) {
            #pragma unroll
            for (uint32_t e = 0u; e < 3u; ++e) m[q][e] = c[e];
        }
    } else {
        float inv_sqrt_h = rsqrtf(static_cast<float>(H));
        float qv[3][3];
        #pragma unroll
        for (uint32_t q = 0u; q < 3u; ++q) {
            #pragma unroll
            for (uint32_t e = 0u; e < 3u; ++e) {
                uint32_t h = t + e * nt;
                qv[q][e] = (q < nq && h < H) ? a.queries[static_cast<size_t>(q) * H + h] : 0.0f;
            }
        }
        for (uint32_t s = 0u; s <= nbm; ++s) {
            float sq = 0.0f, d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
            #pragma unroll
            for (uint32_t e = 0u; e < 3u; ++e) {
                uint32_t h = t + e * nt;
                float x = (h < H) ? ((s < nbm) ? a.blocks[static_cast<size_t>(s) * H + h] : c[e]) : 0.0f;
                sq = fmaf(x, x, sq);
                d0 = fmaf(x, qv[0][e], d0); d1 = fmaf(x, qv[1][e], d1); d2 = fmaf(x, qv[2][e], d2);
            }
            fm_reduce4(sq, d0, d1, d2, red, tot);
            if (t == 0u) {
                float ir = rsqrtf(tot[0] / static_cast<float>(H) + a.eps);
                s_inv[s] = ir;
                s_w[s][0] = tot[1] * ir * inv_sqrt_h;
                s_w[s][1] = tot[2] * ir * inv_sqrt_h;
                s_w[s][2] = tot[3] * ir * inv_sqrt_h;
            }
            __syncthreads();
        }
        if (t < nq) {
            float mx = s_w[0][t];
            for (uint32_t s = 1u; s <= nbm; ++s) mx = fmaxf(mx, s_w[s][t]);
            float sum = 0.0f;
            for (uint32_t s = 0u; s <= nbm; ++s) { float e_ = expf(s_w[s][t] - mx); s_w[s][t] = e_; sum += e_; }
            float inv = 1.0f / fmaxf(sum, 1e-12f);
            for (uint32_t s = 0u; s <= nbm; ++s) s_w[s][t] *= inv;
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t q = 0u; q < 3u; ++q) {
            #pragma unroll
            for (uint32_t e = 0u; e < 3u; ++e) m[q][e] = 0.0f;
        }
        for (uint32_t s = 0u; s <= nbm; ++s) {
            float ir = s_inv[s];
            float w0 = s_w[s][0], w1 = s_w[s][1], w2 = s_w[s][2];
            #pragma unroll
            for (uint32_t e = 0u; e < 3u; ++e) {
                uint32_t h = t + e * nt;
                float x = (h < H) ? ((s < nbm) ? a.blocks[static_cast<size_t>(s) * H + h] : c[e]) * ir : 0.0f;
                m[0][e] = fmaf(w0, x, m[0][e]); m[1][e] = fmaf(w1, x, m[1][e]); m[2][e] = fmaf(w2, x, m[2][e]);
            }
        }
    }
    #pragma unroll
    for (uint32_t q = 0u; q < 3u; ++q) {
        if (q >= nq) break;
        if (a.mix_out != nullptr) {
            #pragma unroll
            for (uint32_t e = 0u; e < 3u; ++e) {
                uint32_t h = t + e * nt;
                if (h < H) a.mix_out[static_cast<size_t>(q) * H + h] = m[q][e];
            }
        }
        const float* nw = (q == 0u) ? a.norm_w0 : ((q == 1u) ? a.norm_w1 : a.norm_w2);
        if (nw != nullptr) {
            float sq = 0.0f;
            #pragma unroll
            for (uint32_t e = 0u; e < 3u; ++e) sq = fmaf(m[q][e], m[q][e], sq);
            fm_reduce4(sq, 0.0f, 0.0f, 0.0f, red, tot);
            float ir = rsqrtf(tot[0] / static_cast<float>(H) + a.eps);
            #pragma unroll
            for (uint32_t e = 0u; e < 3u; ++e) {
                uint32_t h = t + e * nt;
                if (h < H) a.norm_out[static_cast<size_t>(q) * H + h] = m[q][e] * ir * nw[h];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-segment GEMV over up to 8 rows (batched decode / speculative verify): one launch for
// several projections of the same rows. Segment s: y_s[t] = W_s . x_s[t] (FP8 row-scaled or
// F32 weights); rows are contiguous (x stride k_cols, y stride n_rows).
//   Grid: ceil(total_rows / 4), Block: 128
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_gemv_multi_rows(GemvMultiArgs a, uint32_t m_rows) {
    uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t grow = blockIdx.x * 4u + warp;
    if (grow >= a.total_rows) return;
    uint32_t si = 0u;
    #pragma unroll
    for (uint32_t i = 1u; i < 8u; ++i) {
        if (i < a.n_seg && grow >= a.seg[i].row_start) si = i;
    }
    const GemvSeg sg = a.seg[si];
    uint32_t row = grow - sg.row_start;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (sg.is_f32 != 0u) {
        const float4* wv = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(sg.w) + static_cast<size_t>(row) * sg.k_cols);
        for (uint32_t i = lane; i < (sg.k_cols >> 2); i += 32u) {
            float4 w4 = wv[i];
            #pragma unroll
            for (uint32_t t = 0u; t < 8u; ++t) {
                if (t < m_rows) {
                    float4 x4 = reinterpret_cast<const float4*>(sg.x + static_cast<size_t>(t) * sg.k_cols)[i];
                    acc[t] = fmaf(w4.x, x4.x, fmaf(w4.y, x4.y, fmaf(w4.z, x4.z, fmaf(w4.w, x4.w, acc[t]))));
                }
            }
        }
        #pragma unroll
        for (uint32_t t = 0u; t < 8u; ++t) {
            if (t < m_rows) {
                float s = wave32_sum(acc[t]);
                if (lane == 0u) sg.y[static_cast<size_t>(t) * sg.n_rows + row] = s;
            }
        }
        return;
    }
    const uint4* w_vec = reinterpret_cast<const uint4*>(sg.w + static_cast<size_t>(row) * sg.k_cols);
    uint32_t k16 = sg.k_cols >> 4;
    auto dot16 = [&](uint4 w16, uint32_t c) {
        float4 d0 = decode_fp8x4(w16.x), d1 = decode_fp8x4(w16.y), d2 = decode_fp8x4(w16.z), d3 = decode_fp8x4(w16.w);
        #pragma unroll
        for (uint32_t t = 0u; t < 8u; ++t) {
            if (t < m_rows) {
                const float4* xv = reinterpret_cast<const float4*>(sg.x + static_cast<size_t>(t) * sg.k_cols) + (c << 2);
                float4 x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
                float s = acc[t];
                s = fmaf(d0.x, x0.x, s); s = fmaf(d0.y, x0.y, s); s = fmaf(d0.z, x0.z, s); s = fmaf(d0.w, x0.w, s);
                s = fmaf(d1.x, x1.x, s); s = fmaf(d1.y, x1.y, s); s = fmaf(d1.z, x1.z, s); s = fmaf(d1.w, x1.w, s);
                s = fmaf(d2.x, x2.x, s); s = fmaf(d2.y, x2.y, s); s = fmaf(d2.z, x2.z, s); s = fmaf(d2.w, x2.w, s);
                s = fmaf(d3.x, x3.x, s); s = fmaf(d3.y, x3.y, s); s = fmaf(d3.z, x3.z, s); s = fmaf(d3.w, x3.w, s);
                acc[t] = s;
            }
        }
    };
    uint32_t c = lane;
    for (; c + 32u < k16; c += 64u) {
        uint4 wa = w_vec[c];
        uint4 wb = w_vec[c + 32u];
        dot16(wa, c);
        dot16(wb, c + 32u);
    }
    if (c < k16) dot16(w_vec[c], c);
    float sc = sg.scale[row];
    #pragma unroll
    for (uint32_t t = 0u; t < 8u; ++t) {
        if (t < m_rows) {
            float s = wave32_sum(acc[t]);
            if (lane == 0u) sg.y[static_cast<size_t>(t) * sg.n_rows + row] = s * sc;
        }
    }
}

// Multi-segment WMMA FP8 GEMM for <= 16 rows (all segments FP8, n_rows % 16 == 0, k_cols % 32 == 0).
// 2 warps/block (32 rows/block -> high CU occupancy on 2304-row down/o projections!) + 2x K-unroll.
//   Grid: (ceil(total_rows / 32)), Block: 64
extern "C" __global__ void smalltide_gemv_multi_wmma(GemvMultiArgs a, uint32_t m_rows) {
    uint32_t warp = threadIdx.x >> 5, l = threadIdx.x & 31u;
    uint32_t n0 = (blockIdx.x * 2u + warp) * 16u;
    if (n0 >= a.total_rows) return;
    uint32_t si = 0u;
    #pragma unroll
    for (uint32_t i = 1u; i < 8u; ++i) {
        if (i < a.n_seg && n0 >= a.seg[i].row_start) si = i;
    }
    const GemvSeg sg = a.seg[si];
    uint32_t row = (n0 - sg.row_start) + (l & 15u);
    bool b_ok = row < sg.n_rows;
    uint32_t am = l & 15u;
    bool a_ok = am < m_rows;
    const float* xr = sg.x + static_cast<size_t>(a_ok ? am : 0u) * sg.k_cols;
    const uint8_t* wr = sg.w + static_cast<size_t>(b_ok ? row : 0u) * sg.k_cols;
    st_float8 c = {};
    uint32_t hoff = (l >> 4) * 16u;
    auto step32 = [&](uint32_t base, uint4 wv) {
        const float4* xp = reinterpret_cast<const float4*>(xr + base);
        float4 x0 = a_ok ? xp[0] : make_float4(0, 0, 0, 0), x1 = a_ok ? xp[1] : make_float4(0, 0, 0, 0);
        float4 x2 = a_ok ? xp[2] : make_float4(0, 0, 0, 0), x3 = a_ok ? xp[3] : make_float4(0, 0, 0, 0);
        st_half8 a0, a1;
        a0[0] = (_Float16)x0.x; a0[1] = (_Float16)x0.y; a0[2] = (_Float16)x0.z; a0[3] = (_Float16)x0.w;
        a0[4] = (_Float16)x1.x; a0[5] = (_Float16)x1.y; a0[6] = (_Float16)x1.z; a0[7] = (_Float16)x1.w;
        a1[0] = (_Float16)x2.x; a1[1] = (_Float16)x2.y; a1[2] = (_Float16)x2.z; a1[3] = (_Float16)x2.w;
        a1[4] = (_Float16)x3.x; a1[5] = (_Float16)x3.y; a1[6] = (_Float16)x3.z; a1[7] = (_Float16)x3.w;
        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, fp8x8_to_half8(make_uint2(wv.x, wv.y)), c);
        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, fp8x8_to_half8(make_uint2(wv.z, wv.w)), c);
    };
    uint32_t k = 0u;
    for (; k + 64u <= sg.k_cols; k += 64u) {
        uint32_t b0 = k + hoff, b1 = k + 32u + hoff;
        uint4 w0 = b_ok ? *reinterpret_cast<const uint4*>(wr + b0) : make_uint4(0u, 0u, 0u, 0u);
        uint4 w1 = b_ok ? *reinterpret_cast<const uint4*>(wr + b1) : make_uint4(0u, 0u, 0u, 0u);
        step32(b0, w0);
        step32(b1, w1);
    }
    if (k < sg.k_cols) {
        uint32_t b0 = k + hoff;
        uint4 w0 = b_ok ? *reinterpret_cast<const uint4*>(wr + b0) : make_uint4(0u, 0u, 0u, 0u);
        step32(b0, w0);
    }
    if (!b_ok) return;
    float sc = (sg.scale != nullptr) ? sg.scale[row] : 1.0f;
    #pragma unroll
    for (uint32_t i = 0u; i < 8u; ++i) {
        uint32_t mm = (l >> 4) * 8u + i;
        if (mm < m_rows) sg.y[static_cast<size_t>(mm) * sg.n_rows + row] = c[i] * sc;
    }
}

// ---------------------------------------------------------------------------
// Attention prep for `rows` consecutive positions of one sequence (speculative verify): the
// decode fusion with the K/V short convs run row by row (+ verify histories), FP8 KV store at
// positions pos0.., Q/R clips with each row's own tau, relative-bias tables.
//   Grid: n_kv + 2 * n_q * rows, Block: 128
// ---------------------------------------------------------------------------
struct AttnPrepRowsExt {
    float* hist_k; // nullable [3 + rows][n_kv*128]
    float* hist_v;
    uint32_t pos0;
    uint32_t is_local;
    uint32_t rows;
    float tau_alpha;
    float tau_floor;
};

__device__ __forceinline__ float prep_tau(const AttnPrepRowsExt& e, uint32_t row) {
    if (e.tau_alpha == 0.0f) return 1.0f;
    float p = static_cast<float>(e.pos0 + row) + 1.0f;
    return 1.0f + e.tau_alpha * logf(fmaxf(p / e.tau_floor, 1.0f));
}

extern "C" __global__ void smalltide_attn_prep_rows(AttnPrepArgs a, AttnPrepRowsExt e) {
    __shared__ float sh[4];
    uint32_t b = blockIdx.x, t = threadIdx.x;
    if (b < a.n_kv) {
        uint32_t h = b, dim = a.n_kv * 128u, c = h * 128u + t;
        if (e.hist_k != nullptr) {
            #pragma unroll
            for (uint32_t j = 0u; j < 3u; ++j) {
                e.hist_k[j * dim + c] = a.sconv_k[j * dim + c];
                e.hist_v[j * dim + c] = a.sconv_v[j * dim + c];
            }
        }
        for (uint32_t r = 0u; r < e.rows; ++r) {
            float kr = a.k_raw[static_cast<size_t>(r) * dim + c], vr = a.v_raw[static_cast<size_t>(r) * dim + c];
            if (e.hist_k != nullptr) {
                e.hist_k[static_cast<size_t>(3u + r) * dim + c] = kr;
                e.hist_v[static_cast<size_t>(3u + r) * dim + c] = vr;
            }
            const float* kd = (a.k_dyn != nullptr) ? a.k_dyn + static_cast<size_t>(r) * a.n_kv * 4u : nullptr;
            const float* vd = (a.v_dyn != nullptr) ? a.v_dyn + static_cast<size_t>(r) * a.n_kv * 4u : nullptr;
            float k = sconv_step(a.sconv_k, a.k_w, kd, c, dim, kr, h);
            float v = sconv_step(a.sconv_v, a.v_w, vd, c, dim, vr, h);
            float kn = sqrtf(block128_sum(k * k, sh)) + 1e-6f;
            k *= fminf(a.qk_thr / kn, 1.0f);
            float vs = block128_sum(v * v, sh);
            v *= rsqrtf(vs / 128.0f + a.eps);
            float km = block128_max(fabsf(k), sh);
            float vm = block128_max(fabsf(v), sh);
            float ks = (km > 0.0f) ? km / 448.0f : 1.0f;
            float vsc = (vm > 0.0f) ? vm / 448.0f : 1.0f;
            uint32_t pos = e.pos0 + r;
            uint32_t slot = (e.is_local != 0u) ? (pos % a.max_seq) : pos;
            size_t si = static_cast<size_t>(kv_phys(a.page_table, slot)) * a.n_kv + h;
            if (t == 0u) { a.k_scales[si] = ks; a.v_scales[si] = vsc; }
            a.k_cache[si * 128u + t] = encode_fp8_e4m3fn(k / ks);
            a.v_cache[si * 128u + t] = encode_fp8_e4m3fn(v / vsc);
        }
        return;
    }
    b -= a.n_kv;
    uint32_t nqr = a.n_q * e.rows;
    if (b < nqr) {
        uint32_t row = b / a.n_q, head = b % a.n_q;
        float* p = a.q + static_cast<size_t>(b) * 128u;
        float x = p[t] * a.q_scale[head * 128u + t];
        float n = sqrtf(block128_sum(x * x, sh)) + 1e-6f;
        p[t] = x * (fminf(a.qk_thr / n, 1.0f) * prep_tau(e, row));
        return;
    }
    b -= nqr;
    if (b < nqr) {
        uint32_t row = b / a.n_q, head = b % a.n_q;
        __shared__ float rr[16];
        float x = (t < 16u) ? a.r[static_cast<size_t>(b) * 16u + t] * a.r_scale[head * 16u + t] : 0.0f;
        float n = sqrtf(block128_sum(x * x, sh)) + 1e-6f;
        float y = x * (fminf(a.rel_thr / n, 1.0f) * prep_tau(e, row));
        if (t < 16u) { a.r[static_cast<size_t>(b) * 16u + t] = y; rr[t] = y; }
        __syncthreads();
        for (uint32_t d = t; d < a.rel_extent; d += 128u) {
            const float4* pr = reinterpret_cast<const float4*>(a.rel_proj + static_cast<size_t>(d) * 16u);
            float acc = 0.0f;
            #pragma unroll
            for (uint32_t i = 0u; i < 4u; ++i) {
                float4 pv = pr[i];
                acc = fmaf(pv.x, rr[4u * i], fmaf(pv.y, rr[4u * i + 1u], fmaf(pv.z, rr[4u * i + 2u], fmaf(pv.w, rr[4u * i + 3u], acc))));
            }
            a.rel_table[static_cast<size_t>(b) * a.rel_extent + d] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Rows version of the fused decode mix (one block per row, hidden <= 3072, <= 15 blocks):
//   A (add_src != null): boundary ? (blocks[nb][r] = residual[r], cur[r] = add_src[r])
//                                 : cur[r] = residual[r] + add_src[r]
//   B (copy_out != null): copy_out[r] = cur[r]
//   C: m_q = attention-residual mix of [blocks[0..nb'][r], cur[r]] with query q (identity if
//      queries == null); D: mix_out[q][r] = m_q (optional), norm_out[q][r] = rmsnorm(m_q) * w_q.
//   blocks: [nb][M][H]; mix_out / norm_out: [nq][M][H].   Grid: M, Block: 1024
// ---------------------------------------------------------------------------
struct MixRowsArgs {
    float* cur;
    const float* add_src;
    const float* residual;
    float* blocks;
    float* copy_out;
    const float* queries;
    float* mix_out;
    const float* norm_w0;
    const float* norm_w1;
    const float* norm_w2;
    float* norm_out;
    uint32_t nb;
    uint32_t boundary;
    uint32_t nq;
    uint32_t hidden;
    uint32_t rows;
    float eps;
};

extern "C" __global__ void smalltide_mix_rows(MixRowsArgs a) {
    __shared__ float red[4][32];
    __shared__ float tot[4];
    __shared__ float s_inv[16];
    __shared__ float s_w[16][3];
    const uint32_t H = a.hidden, nt = blockDim.x, t = threadIdx.x, r = blockIdx.x, M = a.rows;
    const size_t ro = static_cast<size_t>(r) * H;
    float c[3];
    uint32_t nbm = a.nb;
    if (a.add_src != nullptr) {
        #pragma unroll
        for (uint32_t e = 0u; e < 3u; ++e) {
            uint32_t h = t + e * nt;
            c[e] = 0.0f;
            if (h < H) {
                float res = a.residual[ro + h], y = a.add_src[ro + h];
                if (a.boundary != 0u) {
                    a.blocks[(static_cast<size_t>(a.nb) * M + r) * H + h] = res;
                    c[e] = y;
                } else {
                    c[e] = res + y;
                }
                a.cur[ro + h] = c[e];
            }
        }
        if (a.boundary != 0u) nbm = a.nb + 1u;
    } else {
        #pragma unroll
        for (uint32_t e = 0u; e < 3u; ++e) {
            uint32_t h = t + e * nt;
            c[e] = (h < H) ? a.cur[ro + h] : 0.0f;
        }
    }
    if (a.copy_out != nullptr) {
        #pragma unroll
        for (uint32_t e = 0u; e < 3u; ++e) {
            uint32_t h = t + e * nt;
            if (h < H) a.copy_out[ro + h] = c[e];
        }
    }
    const uint32_t nq = a.nq;
    float m[3][3];
    if (a.queries == nullptr) {
        #pragma unroll
        for (uint32_t q = 0u; q < 3u; ++q) {
            #pragma unroll
            for (uint32_t e = 0u; e < 3u; ++e) m[q][e] = c[e];
        }
    } else {
        float inv_sqrt_h = rsqrtf(static_cast<float>(H));
        float qv[3][3];
        #pragma unroll
        for (uint32_t q = 0u; q < 3u; ++q) {
            #pragma unroll
            for (uint32_t e = 0u; e < 3u; ++e) {
                uint32_t h = t + e * nt;
                qv[q][e] = (q < nq && h < H) ? a.queries[static_cast<size_t>(q) * H + h] : 0.0f;
            }
        }
        for (uint32_t s = 0u; s <= nbm; ++s) {
            float sq = 0.0f, d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
            #pragma unroll
            for (uint32_t e = 0u; e < 3u; ++e) {
                uint32_t h = t + e * nt;
                float x = (h < H) ? ((s < nbm) ? a.blocks[(static_cast<size_t>(s) * M + r) * H + h] : c[e]) : 0.0f;
                sq = fmaf(x, x, sq);
                d0 = fmaf(x, qv[0][e], d0); d1 = fmaf(x, qv[1][e], d1); d2 = fmaf(x, qv[2][e], d2);
            }
            fm_reduce4(sq, d0, d1, d2, red, tot);
            if (t == 0u) {
                float ir = rsqrtf(tot[0] / static_cast<float>(H) + a.eps);
                s_inv[s] = ir;
                s_w[s][0] = tot[1] * ir * inv_sqrt_h;
                s_w[s][1] = tot[2] * ir * inv_sqrt_h;
                s_w[s][2] = tot[3] * ir * inv_sqrt_h;
            }
            __syncthreads();
        }
        if (t < nq) {
            float mx = s_w[0][t];
            for (uint32_t s = 1u; s <= nbm; ++s) mx = fmaxf(mx, s_w[s][t]);
            float sum = 0.0f;
            for (uint32_t s = 0u; s <= nbm; ++s) { float e_ = expf(s_w[s][t] - mx); s_w[s][t] = e_; sum += e_; }
            float inv = 1.0f / fmaxf(sum, 1e-12f);
            for (uint32_t s = 0u; s <= nbm; ++s) s_w[s][t] *= inv;
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t q = 0u; q < 3u; ++q) {
            #pragma unroll
            for (uint32_t e = 0u; e < 3u; ++e) m[q][e] = 0.0f;
        }
        for (uint32_t s = 0u; s <= nbm; ++s) {
            float ir = s_inv[s];
            float w0 = s_w[s][0], w1 = s_w[s][1], w2 = s_w[s][2];
            #pragma unroll
            for (uint32_t e = 0u; e < 3u; ++e) {
                uint32_t h = t + e * nt;
                float x = (h < H) ? ((s < nbm) ? a.blocks[(static_cast<size_t>(s) * M + r) * H + h] : c[e]) * ir : 0.0f;
                m[0][e] = fmaf(w0, x, m[0][e]); m[1][e] = fmaf(w1, x, m[1][e]); m[2][e] = fmaf(w2, x, m[2][e]);
            }
        }
    }
    #pragma unroll
    for (uint32_t q = 0u; q < 3u; ++q) {
        if (q >= nq) break;
        const size_t qo = (static_cast<size_t>(q) * M + r) * H;
        if (a.mix_out != nullptr) {
            #pragma unroll
            for (uint32_t e = 0u; e < 3u; ++e) {
                uint32_t h = t + e * nt;
                if (h < H) a.mix_out[qo + h] = m[q][e];
            }
        }
        const float* nw = (q == 0u) ? a.norm_w0 : ((q == 1u) ? a.norm_w1 : a.norm_w2);
        if (nw != nullptr) {
            float sq = 0.0f;
            #pragma unroll
            for (uint32_t e = 0u; e < 3u; ++e) sq = fmaf(m[q][e], m[q][e], sq);
            fm_reduce4(sq, 0.0f, 0.0f, 0.0f, red, tot);
            float ir = rsqrtf(tot[0] / static_cast<float>(H) + a.eps);
            #pragma unroll
            for (uint32_t e = 0u; e < 3u; ++e) {
                uint32_t h = t + e * nt;
                if (h < H) a.norm_out[qo + h] = m[q][e] * ir * nw[h];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Loop controllers for up to 4 items over the same rows (speculative verify / batched decode):
//   prep:  out_i[r] = [rms(cur_i[r]), rms(msg_i[r])]           Grid: (2, rows, items), Block 256
//   apply: c = (1 - tanh(f)) * cur + tanh(g) * msg -> out_i; norm_out_i = rmsnorm(c) * w_i
//                                                             Grid: (rows, items), Block 1024
// ---------------------------------------------------------------------------
struct CtrlMultiArgs {
    float* out[4];
    const float* cur[4];
    const float* msg[4];
    const float* gain[4];
    const float* forget[4];
    const float* norm_w[4];
    float* norm_out[4];
    uint32_t hidden;
    float eps;
};

extern "C" __global__ void smalltide_ctrl_prep_multi(CtrlMultiArgs a) {
    uint32_t side = blockIdx.x, row = blockIdx.y, it = blockIdx.z, H = a.hidden;
    const float* src = ((side == 0u) ? a.cur[it] : a.msg[it]) + static_cast<size_t>(row) * H;
    float* dst = a.out[it] + (static_cast<size_t>(row) * 2u + side) * H;
    float sq = 0.0f;
    for (uint32_t d = threadIdx.x; d < H; d += blockDim.x) { float v = src[d]; sq = fmaf(v, v, sq); }
    __shared__ float s_w[32];
    __shared__ float inv;
    float ws = wave32_sum(sq);
    if ((threadIdx.x & 31u) == 0u) s_w[threadIdx.x >> 5] = ws;
    __syncthreads();
    if (threadIdx.x == 0u) {
        float t = 0.0f;
        for (uint32_t w = 0u; w < (blockDim.x >> 5); ++w) t += s_w[w];
        inv = rsqrtf(t / static_cast<float>(H) + a.eps);
    }
    __syncthreads();
    for (uint32_t d = threadIdx.x; d < H; d += blockDim.x) dst[d] = src[d] * inv;
}

extern "C" __global__ void smalltide_ctrl_apply_norm_multi(CtrlMultiArgs a) {
    __shared__ float red[4][32];
    __shared__ float tot[4];
    uint32_t row = blockIdx.x, it = blockIdx.y, H = a.hidden, nt = blockDim.x, t = threadIdx.x;
    size_t ro = static_cast<size_t>(row) * H;
    float c[3];
    float sq = 0.0f;
    #pragma unroll
    for (uint32_t e = 0u; e < 3u; ++e) {
        uint32_t h = t + e * nt;
        c[e] = 0.0f;
        if (h < H) {
            float g = tanhf(a.gain[it][ro + h]);
            float f = tanhf(a.forget[it][ro + h]);
            c[e] = (1.0f - f) * a.cur[it][ro + h] + g * a.msg[it][ro + h];
            a.out[it][ro + h] = c[e];
            sq = fmaf(c[e], c[e], sq);
        }
    }
    const float* nw = a.norm_w[it];
    if (nw == nullptr) return;
    fm_reduce4(sq, 0.0f, 0.0f, 0.0f, red, tot);
    float ir = rsqrtf(tot[0] / static_cast<float>(H) + a.eps);
    #pragma unroll
    for (uint32_t e = 0u; e < 3u; ++e) {
        uint32_t h = t + e * nt;
        if (h < H) a.norm_out[it][ro + h] = c[e] * ir * nw[h];
    }
}

// Expert grouping (as smalltide_moe_group_rows2) plus the f32 -> f16 copy of the MoE input in the
// same launch.   Grid: 1, Block: 256 (threads < 64 plan; all convert)
extern "C" __global__ void smalltide_moe_group_rows3(
    uint32_t* __restrict__ work, uint32_t* __restrict__ pairs, const uint32_t* __restrict__ route,
    const uint32_t* __restrict__ slot_table, uint32_t rows, _Float16* __restrict__ x16,
    const float* __restrict__ x, uint32_t n_x
) {
    for (uint32_t i = threadIdx.x * 4u; i < n_x; i += blockDim.x * 4u) {
        if (i + 3u < n_x) {
            float4 v = *reinterpret_cast<const float4*>(x + i);
            x16[i] = (_Float16)v.x; x16[i + 1u] = (_Float16)v.y; x16[i + 2u] = (_Float16)v.z; x16[i + 3u] = (_Float16)v.w;
        } else {
            for (uint32_t j = i; j < n_x; ++j) x16[j] = (_Float16)x[j];
        }
    }
    __shared__ uint32_t s_e[64], s_cnt[64], s_wi[64], s_off[64];
    __shared__ uint32_t s_nw;
    uint32_t p = threadIdx.x, np_total = rows * 8u;
    bool lead = p < 64u;
    bool valid = p < np_total;
    uint32_t e = valid ? route[(p >> 3) * 16u + (p & 7u)] : 0xFFFFFFFFu;
    if (lead) s_e[p] = e;
    __syncthreads();
    uint32_t slot = valid ? slot_table[e] : 0xFFFFFFFFu;
    bool first = valid && slot != 0xFFFFFFFFu;
    for (uint32_t q = 0u; q < p && first; ++q) if (s_e[q] == e) first = false;
    uint32_t cnt = 0u;
    if (first) for (uint32_t q = p; q < np_total; ++q) cnt += (s_e[q] == e) ? 1u : 0u;
    if (lead) { s_cnt[p] = first ? cnt : 0u; s_wi[p] = first ? 1u : 0u; }
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
    if (lead && p >= s_nw) { work[p * 4u] = 0u; work[p * 4u + 1u] = 0u; work[p * 4u + 2u] = 0u; work[p * 4u + 3u] = 0u; }
}
