// SmallTide RDNA 4 (gfx1201) kernels for Inkling 2's vision tower (transformers'
// Inkling2VisionModel: a 23-layer ViT with a class token, LayerNorm, LayerScale, axial 2D RoPE and
// full attention over one image, then the 2x2 merger). Activations are f32 on the device; the
// weights are the checkpoint's BF16, streamed a layer at a time (see vision.rs).

typedef short st_short8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ float vit_bf16(uint16_t h) {
    return __uint_as_float(static_cast<uint32_t>(h) << 16);
}

// f32 -> bf16 bits, round to nearest even
__device__ __forceinline__ short vit_to_bf16(float f) {
    uint32_t b = __float_as_uint(f);
    b += 0x7fffu + ((b >> 16) & 1u);
    return static_cast<short>(b >> 16);
}

// ---------------------------------------------------------------------------
// Y[m][n] = sum_k bf16(X[m][k]) * W[n][k]: X f32 (rounded to bf16 as the reference's activations
// are), W bf16, Y f32. k_cols a multiple of 32.
//    Grid: (ceil(N / 64), ceil(M / 128)), Block: 256 (a warp 16 rows, 4 N tiles)
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_vit_gemm_bf16(
    float* __restrict__ y, const float* __restrict__ x, const uint16_t* __restrict__ w,
    uint32_t m_rows, uint32_t n_rows, uint32_t k_cols
) {
    uint32_t warp = threadIdx.x >> 5, l = threadIdx.x & 31u;
    uint32_t m0 = blockIdx.y * 128u + warp * 16u;
    if (m0 >= m_rows) return;
    uint32_t n0 = blockIdx.x * 64u;
    uint32_t am = m0 + (l & 15u);
    bool a_ok = am < m_rows;
    const float* xr = x + static_cast<size_t>(a_ok ? am : 0u) * k_cols;
    const uint16_t* wr[4];
    bool b_ok[4];
    #pragma unroll
    for (uint32_t j = 0u; j < 4u; ++j) {
        uint32_t n = n0 + 16u * j + (l & 15u);
        b_ok[j] = n < n_rows;
        wr[j] = w + static_cast<size_t>(b_ok[j] ? n : 0u) * k_cols;
    }
    st_float8 c[4] = {};
    // in each 32-wide block of k, half-wave h owns the 16 consecutive k at h * 16
    uint32_t hoff = (l >> 4) * 16u;
    for (uint32_t k = 0u; k < k_cols; k += 32u) {
        uint32_t base = k + hoff;
        st_short8 a0, a1;
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; ++i) {
            a0[i] = a_ok ? vit_to_bf16(xr[base + i]) : (short)0;
            a1[i] = a_ok ? vit_to_bf16(xr[base + 8u + i]) : (short)0;
        }
        #pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            st_short8 b0, b1;
            #pragma unroll
            for (uint32_t i = 0u; i < 8u; ++i) {
                b0[i] = b_ok[j] ? static_cast<short>(wr[j][base + i]) : (short)0;
                b1[i] = b_ok[j] ? static_cast<short>(wr[j][base + 8u + i]) : (short)0;
            }
            c[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0, b0, c[j]);
            c[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1, b1, c[j]);
        }
    }
    #pragma unroll
    for (uint32_t j = 0u; j < 4u; ++j) {
        uint32_t n = n0 + 16u * j + (l & 15u);
        if (n >= n_rows) continue;
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; ++i) {
            uint32_t mm = m0 + (l >> 4) * 8u + i;
            if (mm < m_rows) y[static_cast<size_t>(mm) * n_rows + n] = c[j][i];
        }
    }
}

// ---------------------------------------------------------------------------
// A Linear's epilogue over [m][n]: t = y + bias (bias bf16 [n] or null); gelu (erf) when `gelu`;
// t *= gamma (LayerScale, bf16 [n] or null); out = residual + t when residual is set. out may be y
// or residual.
//    Grid: (ceil(N / 256), M), Block: 256
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_vit_epilogue(
    float* out, const float* y, const uint16_t* __restrict__ bias, const uint16_t* __restrict__ gamma,
    const float* residual, uint32_t m_rows, uint32_t n_cols, uint32_t gelu
) {
    uint32_t n = blockIdx.x * 256u + threadIdx.x, r = blockIdx.y;
    if (n >= n_cols || r >= m_rows) return;
    size_t i = static_cast<size_t>(r) * n_cols + n;
    float t = y[i];
    if (bias != nullptr) t += vit_bf16(bias[n]);
    if (gelu != 0u) t = 0.5f * t * (1.0f + erff(t * 0.70710678118654752f));
    if (gamma != nullptr) t *= vit_bf16(gamma[n]);
    if (residual != nullptr) t += residual[i];
    out[i] = t;
}

// ---------------------------------------------------------------------------
// nn.LayerNorm over rows of n (biased variance): out = (x - mean) * rsqrt(var + eps) * w + b.
// out may be x.
//    Grid: (rows), Block: 256
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_vit_ln(
    float* out, const float* x, const uint16_t* __restrict__ w, const uint16_t* __restrict__ b,
    uint32_t rows, uint32_t n, float eps
) {
    uint32_t r = blockIdx.x;
    if (r >= rows) return;
    const float* xr = x + static_cast<size_t>(r) * n;
    float* orow = out + static_cast<size_t>(r) * n;
    __shared__ float s_part[8];
    __shared__ float s_stat[2];
    uint32_t lane = threadIdx.x & 31u, wid = threadIdx.x >> 5;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += 256u) sum += xr[i];
    sum = wave32_sum(sum);
    if (lane == 0u) s_part[wid] = sum;
    __syncthreads();
    if (threadIdx.x == 0u) {
        float t = 0.0f;
        for (uint32_t i = 0u; i < 8u; ++i) t += s_part[i];
        s_stat[0] = t / static_cast<float>(n);
    }
    __syncthreads();
    float mean = s_stat[0];
    float sq = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += 256u) {
        float d = xr[i] - mean;
        sq += d * d;
    }
    sq = wave32_sum(sq);
    __syncthreads();
    if (lane == 0u) s_part[wid] = sq;
    __syncthreads();
    if (threadIdx.x == 0u) {
        float t = 0.0f;
        for (uint32_t i = 0u; i < 8u; ++i) t += s_part[i];
        s_stat[1] = rsqrtf(t / static_cast<float>(n) + eps);
    }
    __syncthreads();
    float rstd = s_stat[1];
    for (uint32_t i = threadIdx.x; i < n; i += 256u) {
        orow[i] = (xr[i] - mean) * rstd * vit_bf16(w[i]) + vit_bf16(b[i]);
    }
}

// ---------------------------------------------------------------------------
// Axial 2D RoPE on q and k of qkv [N][3][H][64] in place: a head's interleaved pair (2j, 2j + 1)
// turns by angle j of the row's table (cos / sin [N][32]: 16 column angles, then 16 row angles).
//    Grid: (N), Block: 256
// ---------------------------------------------------------------------------
extern "C" __global__ void smalltide_vit_rope(
    float* qkv, uint32_t n_rows, uint32_t heads, const float* __restrict__ cs, const float* __restrict__ sn
) {
    uint32_t r = blockIdx.x;
    if (r >= n_rows) return;
    uint32_t width = heads * 64u;
    float* row = qkv + static_cast<size_t>(r) * 3u * width;
    uint32_t pairs = heads * 32u * 2u;
    for (uint32_t pi = threadIdx.x; pi < pairs; pi += 256u) {
        uint32_t which = pi / (heads * 32u);
        uint32_t in = pi % (heads * 32u);
        uint32_t j = in & 31u;
        float* p = row + which * width + (in >> 5) * 64u + 2u * j;
        float c = cs[static_cast<size_t>(r) * 32u + j], s = sn[static_cast<size_t>(r) * 32u + j];
        float x1 = p[0], x2 = p[1];
        p[0] = x1 * c - x2 * s;
        p[1] = x2 * c + x1 * s;
    }
}

// ---------------------------------------------------------------------------
// Full attention over one image (scaled_dot_product_attention, no mask, scale 1/8):
// qkv [N][3][H][64] -> out [N][H][64]. A thread a query row of one head (its q and running output
// in registers); K and V are staged 32 keys at a time in shared memory.
//    Grid: (ceil(N / 64), H), Block: 64
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(64) smalltide_vit_attn(
    float* __restrict__ out, const float* __restrict__ qkv, uint32_t n_rows, uint32_t heads
) {
    __shared__ float s_k[32 * 64];
    __shared__ float s_v[32 * 64];
    uint32_t t = threadIdx.x, hd = blockIdx.y;
    uint32_t qi = blockIdx.x * 64u + t;
    bool ok = qi < n_rows;
    size_t stride = static_cast<size_t>(heads) * 64u * 3u;
    size_t k_off = static_cast<size_t>(heads) * 64u + hd * 64u;
    size_t v_off = static_cast<size_t>(heads) * 128u + hd * 64u;
    float q[64], acc[64];
    const float* qr = qkv + static_cast<size_t>(ok ? qi : 0u) * stride + hd * 64u;
    #pragma unroll
    for (uint32_t d = 0u; d < 64u; ++d) {
        q[d] = qr[d] * 0.125f;
        acc[d] = 0.0f;
    }
    float m = -1.0e30f, l = 0.0f;
    for (uint32_t k0 = 0u; k0 < n_rows; k0 += 32u) {
        uint32_t tn = min(32u, n_rows - k0);
        for (uint32_t r = 0u; r < tn; ++r) {
            const float* kr = qkv + static_cast<size_t>(k0 + r) * stride;
            s_k[r * 64u + t] = kr[k_off + t];
            s_v[r * 64u + t] = kr[v_off + t];
        }
        __syncthreads();
        for (uint32_t r = 0u; r < tn; ++r) {
            const float* kk = s_k + r * 64u;
            const float* vv = s_v + r * 64u;
            float s = 0.0f;
            #pragma unroll
            for (uint32_t d = 0u; d < 64u; ++d) s = fmaf(q[d], kk[d], s);
            if (s > m) {
                float sc = __expf(m - s);
                l *= sc;
                #pragma unroll
                for (uint32_t d = 0u; d < 64u; ++d) acc[d] *= sc;
                m = s;
            }
            float p = __expf(s - m);
            l += p;
            #pragma unroll
            for (uint32_t d = 0u; d < 64u; ++d) acc[d] = fmaf(p, vv[d], acc[d]);
        }
        __syncthreads();
    }
    if (ok) {
        float inv = 1.0f / l;
        float* o = out + static_cast<size_t>(qi) * heads * 64u + hd * 64u;
        #pragma unroll
        for (uint32_t d = 0u; d < 64u; ++d) o[d] = acc[d] * inv;
    }
}
