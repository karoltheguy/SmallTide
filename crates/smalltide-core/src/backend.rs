//! Swappable GPU/CPU compute backend trait (HAL).
//!
//! Phase 1 implements this trait in `smalltide-backend-hip` (dynamic `amdhip64_7.dll` + `.hsaco`).
//! Future backends (`smalltide-backend-cuda`, `smalltide-backend-vulkan`) drop in behind this trait
//! without touching model, Tide, or server crates.

use crate::error::Result;
use crate::types::DType;

/// Opaque handle to a device-local buffer (VRAM), supporting zero-cost byte sub-slices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceBuf {
    pub id: u64,
    pub offset_bytes: usize,
    pub size_bytes: usize,
}

impl DeviceBuf {
    /// Returns a zero-cost sub-slice view into this device buffer.
    #[inline]
    pub fn slice(self, rel_offset_bytes: usize, size_bytes: usize) -> Self {
        debug_assert!(
            rel_offset_bytes + size_bytes <= self.size_bytes,
            "DeviceBuf::slice out of bounds ({} + {} > {})",
            rel_offset_bytes,
            size_bytes,
            self.size_bytes
        );
        Self {
            id: self.id,
            offset_bytes: self.offset_bytes + rel_offset_bytes,
            size_bytes,
        }
    }
}

/// Opaque handle to a page-locked host staging buffer (pinned RAM).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PinnedBuf {
    pub id: u64,
    pub size_bytes: usize,
}

/// Opaque handle to an execution or transfer stream/queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Stream(pub u64);

/// Opaque handle to a GPU synchronization event / timeline checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Event(pub u64);

/// Opaque handle to a captured GPU execution graph (e.g., `hipGraphExec_t` / CUDA Graph).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Graph(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueKind {
    Compute,
    AsyncCopyH2D,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub backend_name: &'static str,
    pub device_name: String,
    pub arch_name: String,
    pub total_vram_bytes: usize,
    pub free_vram_bytes: usize,
    pub supports_fp8_wmma: bool,
    pub supports_graphs: bool,
}

/// Parameters for indirect MoE SwiGLU/SiTU execution using the GPU slot indirection table.
#[derive(Debug, Clone, Copy)]
pub struct MoeDispatchParams {
    pub batch_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub top_k: usize,
    pub weight_dtype: DType,
    pub expert_stride_bytes: usize,
}

/// Parameters for Split-K FlashDecoding FP8 KV attention (local sliding-window or global 256K).
#[derive(Debug, Clone, Copy)]
pub struct FlashDecodeParams {
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub max_seq: usize,
    pub q_pos: usize,
    pub rel_extent: usize,
    pub is_local: bool,
    pub num_splits: usize,
    pub inv_sqrt_d: f32,
}

/// Minimal backend-agnostic compute & async-transfer abstraction.
pub trait ComputeBackend: Send + Sync {
    fn device_info(&self) -> Result<DeviceInfo>;

    // --- Memory management ---
    fn alloc_device(&self, size_bytes: usize) -> Result<DeviceBuf>;
    fn free_device(&self, buf: DeviceBuf) -> Result<()>;

    fn alloc_pinned(&self, size_bytes: usize) -> Result<PinnedBuf>;
    fn free_pinned(&self, buf: PinnedBuf) -> Result<()>;

    /// Obtain a mutable host slice into a pinned staging buffer.
    ///
    /// # Safety
    /// Caller must ensure no in-flight async DMA overlaps the accessed byte range.
    unsafe fn pinned_as_mut_slice(&self, buf: PinnedBuf) -> Result<&mut [u8]>;

    // --- Streams, Async DMA & Synchronization ---
    fn create_stream(&self, kind: QueueKind) -> Result<Stream>;
    fn destroy_stream(&self, stream: Stream) -> Result<()>;

    fn create_event(&self) -> Result<Event>;
    fn destroy_event(&self, event: Event) -> Result<()>;

    fn record_event(&self, event: Event, stream: Stream) -> Result<()>;
    fn stream_wait_event(&self, stream: Stream, event: Event) -> Result<()>;
    fn sync_event(&self, event: Event) -> Result<()>;
    fn sync_stream(&self, stream: Stream) -> Result<()>;

    /// Non-blocking H2D DMA from pinned staging slice into device buffer at byte offsets.
    fn memcpy_h2d_async(
        &self,
        dst: DeviceBuf,
        dst_offset: usize,
        src: PinnedBuf,
        src_offset: usize,
        len: usize,
        stream: Stream,
    ) -> Result<()>;

    /// Synchronous D2H readback for small metadata (e.g., router Top-K indices/weights, sampled token).
    fn memcpy_d2h_sync(
        &self,
        dst: &mut [u8],
        src: DeviceBuf,
        src_offset: usize,
        stream: Stream,
    ) -> Result<()>;

    // --- Graph Capture (optional fast path; returns Unsupported if unavailable) ---
    fn begin_graph_capture(&self, _stream: Stream) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("graph capture".into()))
    }
    fn end_graph_capture(&self, _stream: Stream) -> Result<Graph> {
        Err(crate::SmallTideError::Unsupported("graph capture".into()))
    }
    fn launch_graph(&self, _graph: Graph, _stream: Stream) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("graph capture".into()))
    }
    fn destroy_graph(&self, _graph: Graph) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("graph capture".into()))
    }

    // --- Core Neural Ops ---
    fn rms_norm(
        &self,
        out: DeviceBuf,
        x: DeviceBuf,
        weight: DeviceBuf,
        rows: usize,
        hidden: usize,
        eps: f32,
        stream: Stream,
    ) -> Result<()>;

    /// Linear projection `out = x @ weight^T` supporting BF16, FP8E4M3, MXFP8, NVFP4 weights.
    #[allow(clippy::too_many_arguments)]
    fn matmul(
        &self,
        out: DeviceBuf,
        x: DeviceBuf,
        weight: DeviceBuf,
        weight_dtype: DType,
        m: usize,
        n: usize,
        k: usize,
        stream: Stream,
    ) -> Result<()>;

    /// FP8 E4M3FN row-scaled GEMV: `out[n] = (W_fp8[n, :] · x) * row_scales[n]`.
    #[allow(clippy::too_many_arguments)]
    fn gemv_fp8(
        &self,
        out: DeviceBuf,
        x: DeviceBuf,
        w_fp8: DeviceBuf,
        row_scales: DeviceBuf,
        n_rows: usize,
        k_cols: usize,
        stream: Stream,
    ) -> Result<()>;

    /// F32 GEMV for small high-precision projections (`attn_sink`, `k/v_sconv_dyn_proj`).
    #[allow(clippy::too_many_arguments)]
    fn gemv_f32(
        &self,
        out: DeviceBuf,
        x: DeviceBuf,
        w_f32: DeviceBuf,
        n_rows: usize,
        k_cols: usize,
        stream: Stream,
    ) -> Result<()>;

    /// Execute active MoE experts residing in `expert_pool` via GPU `slot_table` indirection.
    #[allow(clippy::too_many_arguments)]
    fn moe_swiglu_indirect(
        &self,
        out: DeviceBuf,
        x: DeviceBuf,
        expert_pool: DeviceBuf,
        slot_table: DeviceBuf,
        topk_ids: DeviceBuf,
        topk_weights: DeviceBuf,
        params: MoeDispatchParams,
        stream: Stream,
    ) -> Result<()>;

    /// Zero-allocation indirect MoE using caller-owned `mid_scratch` (`[top_k * intermediate_size * 4]`).
    #[allow(clippy::too_many_arguments)]
    fn moe_swiglu_indirect_with_scratch(
        &self,
        out: DeviceBuf,
        mid_scratch: DeviceBuf,
        x: DeviceBuf,
        expert_pool: DeviceBuf,
        slot_table: DeviceBuf,
        topk_ids: DeviceBuf,
        topk_weights: DeviceBuf,
        params: MoeDispatchParams,
        stream: Stream,
    ) -> Result<()>;

    /// Sigmoid + bias-aided Top-8 MoE router.
    #[allow(clippy::too_many_arguments)]
    fn router_topk8(
        &self,
        topk_ids: DeviceBuf,
        topk_weights: DeviceBuf,
        x: DeviceBuf,
        gate_w: DeviceBuf,
        gate_b: Option<DeviceBuf>,
        hidden_size: usize,
        route_scale: f32,
        stream: Stream,
    ) -> Result<()>;

    /// Inkling-2-Mini causal short convolution (`k=4`) with residual + optional dynamic taps.
    #[allow(clippy::too_many_arguments)]
    fn sconv_k4_decode(
        &self,
        out: DeviceBuf,
        state: DeviceBuf,
        x: DeviceBuf,
        weight: DeviceBuf,
        dyn_taps: Option<DeviceBuf>,
        dim: usize,
        head_dim: usize,
        stream: Stream,
    ) -> Result<()>;

    /// Per-head L2 clipping after optional per-channel scale `[n_heads, head_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn head_l2_clip(
        &self,
        x: DeviceBuf,
        scale: Option<DeviceBuf>,
        n_heads: usize,
        head_dim: usize,
        threshold: f32,
        stream: Stream,
    ) -> Result<()>;

    /// Parameter-free per-head Value RMSNorm (`use_value_norm_no_gamma = true`).
    fn value_norm_no_gamma(
        &self,
        v: DeviceBuf,
        n_kv_heads: usize,
        head_dim: usize,
        eps: f32,
        stream: Stream,
    ) -> Result<()>;

    /// Relative position bias table: `table[qh, dist] = dot(r[qh], rel_proj[dist])`.
    #[allow(clippy::too_many_arguments)]
    fn rel_table(
        &self,
        table: DeviceBuf,
        r: DeviceBuf,
        rel_proj: DeviceBuf,
        n_heads: usize,
        rel_extent: usize,
        stream: Stream,
    ) -> Result<()>;

    /// Write one token's K/V into FP8 KV cache at `slot_pos`.
    #[allow(clippy::too_many_arguments)]
    fn kv_store_fp8_decode(
        &self,
        k_cache: DeviceBuf,
        k_scales: DeviceBuf,
        v_cache: DeviceBuf,
        v_scales: DeviceBuf,
        k_in: DeviceBuf,
        v_in: DeviceBuf,
        n_kv_heads: usize,
        max_seq: usize,
        head_dim: usize,
        slot_pos: usize,
        stream: Stream,
    ) -> Result<()>;

    /// Split-K FlashDecoding FP8 KV attention + reduce with relative bias table & sink logits.
    #[allow(clippy::too_many_arguments)]
    fn flash_decode_fp8(
        &self,
        out: DeviceBuf,
        partial_out: DeviceBuf,
        partial_m: DeviceBuf,
        partial_l: DeviceBuf,
        q: DeviceBuf,
        k_cache: DeviceBuf,
        k_scales: DeviceBuf,
        v_cache: DeviceBuf,
        v_scales: DeviceBuf,
        rel_table: Option<DeviceBuf>,
        sink_logit: Option<DeviceBuf>,
        params: FlashDecodeParams,
        stream: Stream,
    ) -> Result<()>;

    /// `out[i] = attn[i] * sigmoid(gate[i])`.
    fn gate_attn_sigmoid_mul(
        &self,
        out: DeviceBuf,
        attn: DeviceBuf,
        gate: DeviceBuf,
        n: usize,
        stream: Stream,
    ) -> Result<()>;

    /// Multi-query Attention Residuals mix over `[blocks[0..num_blocks], cur]`.
    #[allow(clippy::too_many_arguments)]
    fn attn_res_mix(
        &self,
        out: DeviceBuf,
        blocks: Option<DeviceBuf>,
        cur: DeviceBuf,
        queries: DeviceBuf,
        num_blocks: usize,
        num_queries: usize,
        hidden: usize,
        eps: f32,
        normalize_values: bool,
        stream: Stream,
    ) -> Result<()>;

    /// Loop controller prep: `out[0..H] = rmsnorm(cur)`, `out[H..2H] = rmsnorm(msg)`.
    #[allow(clippy::too_many_arguments)]
    fn loop_control_prep(
        &self,
        out: DeviceBuf,
        cur: DeviceBuf,
        msg: DeviceBuf,
        hidden: usize,
        eps: f32,
        stream: Stream,
    ) -> Result<()>;

    /// Loop controller apply: `out = (1 - tanh(forget)) * cur + tanh(gain) * msg`.
    #[allow(clippy::too_many_arguments)]
    fn loop_control_apply(
        &self,
        out: DeviceBuf,
        cur: DeviceBuf,
        msg: DeviceBuf,
        gain_raw: DeviceBuf,
        forget_raw: DeviceBuf,
        n: usize,
        stream: Stream,
    ) -> Result<()>;

    fn vec_add(
        &self,
        out: DeviceBuf,
        a: DeviceBuf,
        b: DeviceBuf,
        n: usize,
        stream: Stream,
    ) -> Result<()>;

    fn vec_scale(&self, x: DeviceBuf, alpha: f32, n: usize, stream: Stream) -> Result<()>;

    fn vec_copy(&self, dst: DeviceBuf, src: DeviceBuf, n: usize, stream: Stream) -> Result<()>;

    fn zero_f32(&self, dst: DeviceBuf, n: usize, stream: Stream) -> Result<()>;

    fn final_logit_softcap(
        &self,
        logits: DeviceBuf,
        vocab_size: usize,
        cap: f32,
        stream: Stream,
    ) -> Result<()>;

    /// Parallel partial argmax (64 blocks -> 64 `(val, idx)` pairs in `out_vals`, `out_idx`).
    fn argmax_partial_f32(
        &self,
        out_vals: DeviceBuf,
        out_idx: DeviceBuf,
        x: DeviceBuf,
        n: usize,
        stream: Stream,
    ) -> Result<()>;
}
