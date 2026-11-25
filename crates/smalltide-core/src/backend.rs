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
    /// Paged KV pool: the sequence's page table (`None`: the cache is the sequence's own).
    pub page_table: Option<DeviceBuf>,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub max_seq: usize,
    pub q_pos: usize,
    pub rel_extent: usize,
    pub is_local: bool,
    pub num_splits: usize,
    pub inv_sqrt_d: f32,
    /// Local layers: attention window when the ring (`max_seq`) is larger; 0 = `max_seq`.
    pub window: usize,
    /// True when V cache is packed Q4_asym32 (64B nibbles + 16B scales per head) instead of FP8.
    pub v_q4: bool,
}

/// Parameters for the multi-row causal prefill attention.
#[derive(Debug, Clone, Copy)]
pub struct PrefillAttnParams {
    /// Paged KV pool: the sequence's page table (`None`: the cache is the sequence's own).
    pub page_table: Option<DeviceBuf>,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    /// KV buffer length (ring size for local layers).
    pub max_seq: usize,
    /// Sliding window (local layers); 0 = global causal attention over `[0, pos]`.
    pub window: usize,
    pub pos0: usize,
    pub rows: usize,
    pub rel_extent: usize,
    pub inv_sqrt_d: f32,
    /// True when V cache is packed Q4_asym32 instead of FP8.
    pub v_q4: bool,
    /// Scratch for the pre-decoded f16 V tiles of a many-row (prefill) pass: 256 bytes per key
    /// and KV head, indexed by ring slot / 16 (local) or position / 16 (global). `None`: the
    /// attention kernel decodes V itself (the same values).
    pub v_tiles: Option<DeviceBuf>,
}

/// Expert-grouped MoE launch for prefill (see `moe_grouped`).
#[derive(Debug, Clone, Copy)]
pub struct GroupedMoeParams {
    pub rows: usize,
    pub n_work: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub expert_stride_bytes: usize,
    /// Byte offset of the pair list inside the descriptor buffer.
    pub pairs_offset_bytes: usize,
}

/// One output segment of `ComputeBackend::gemv_multi`: `y = W . x` with FP8 E4M3 row-scaled
/// weights (`row_scales = Some`, `is_bf16 = false`), native BF16 weights (`is_bf16 = true`),
/// or F32 weights (`row_scales = None`, `is_bf16 = false`).
#[derive(Debug, Clone, Copy)]
pub struct GemvSegment {
    pub w: DeviceBuf,
    pub row_scales: Option<DeviceBuf>,
    pub x: DeviceBuf,
    pub y: DeviceBuf,
    pub n_rows: usize,
    pub k_cols: usize,
    pub is_bf16: bool,
}

/// Inputs/outputs of the fused decode attention prep (see `ComputeBackend::attn_prep_decode`).
#[derive(Debug, Clone, Copy)]
pub struct AttnPrepParams {
    /// Paged KV pool: the sequence's page table (`None`: the cache is the sequence's own).
    pub page_table: Option<DeviceBuf>,
    pub q: DeviceBuf,
    pub r: DeviceBuf,
    pub k_raw: DeviceBuf,
    pub v_raw: DeviceBuf,
    pub k_dyn: Option<DeviceBuf>,
    pub v_dyn: Option<DeviceBuf>,
    pub sconv_k: DeviceBuf,
    pub sconv_v: DeviceBuf,
    pub k_w: DeviceBuf,
    pub v_w: DeviceBuf,
    pub q_scale: DeviceBuf,
    pub r_scale: DeviceBuf,
    pub rel_proj: DeviceBuf,
    pub rel_table: DeviceBuf,
    pub k_cache: DeviceBuf,
    pub k_scales: DeviceBuf,
    pub v_cache: DeviceBuf,
    pub v_scales: DeviceBuf,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub max_seq: usize,
    pub slot: usize,
    pub rel_extent: usize,
    pub qk_threshold: f32,
    pub rel_threshold: f32,
    pub tau: f32,
    pub eps: f32,
    /// True to store V in Q4_asym32 (64B nibbles + 16B scales per head) instead of FP8.
    pub v_q4: bool,
}

/// Rows residual / attention-residual mix / RMSNorm (see `ComputeBackend::mix_rows`).
#[derive(Debug, Clone, Copy)]
pub struct MixRowsParams {
    pub cur: DeviceBuf,
    pub add_src: Option<DeviceBuf>,
    pub residual: Option<DeviceBuf>,
    pub blocks: Option<DeviceBuf>,
    pub copy_out: Option<DeviceBuf>,
    pub queries: Option<DeviceBuf>,
    pub mix_out: Option<DeviceBuf>,
    pub norm_w: [Option<DeviceBuf>; 3],
    pub norm_out: Option<DeviceBuf>,
    pub nb: usize,
    pub boundary: bool,
    pub nq: usize,
    pub hidden: usize,
    pub rows: usize,
    pub eps: f32,
}

/// Fused decode residual / attention-residual mix / RMSNorm (see `ComputeBackend::fused_mix_decode`).
#[derive(Debug, Clone, Copy)]
pub struct FusedMixParams {
    /// `[H]` current residual stream (read; written when `sconv` is set).
    pub cur: DeviceBuf,
    /// `(x, state [3*H], weight [H*4])`: `cur = (boundary ? 0 : residual) + sconv(x)`.
    pub sconv: Option<(DeviceBuf, DeviceBuf, DeviceBuf)>,
    pub residual: DeviceBuf,
    /// `[max_blocks][H]` block states; on a boundary `blocks[nb] = residual`.
    pub blocks: DeviceBuf,
    pub nb: usize,
    pub boundary: bool,
    /// Optional copy of the (updated) `cur`.
    pub copy_out: Option<DeviceBuf>,
    /// `[nq][H]` mix queries (None = identity mix).
    pub queries: Option<DeviceBuf>,
    pub nq: usize,
    /// `[nq][H]` mix outputs.
    pub mix_out: Option<DeviceBuf>,
    /// RMSNorm weights per query (norm skipped when `norm_w[0]` is None).
    pub norm_w: [Option<DeviceBuf>; 3],
    /// `[nq][H]` normed outputs.
    pub norm_out: DeviceBuf,
    pub hidden: usize,
    pub eps: f32,
}

/// Minimal backend-agnostic compute & async-transfer abstraction.
pub trait ComputeBackend: Send + Sync {
    fn device_info(&self) -> Result<DeviceInfo>;

    // --- Memory management ---
    fn alloc_device(&self, size_bytes: usize) -> Result<DeviceBuf>;
    fn free_device(&self, buf: DeviceBuf) -> Result<()>;

    fn alloc_pinned(&self, size_bytes: usize) -> Result<PinnedBuf>;

    /// Pinned host memory that kernels can read/write directly over PCIe (zero-copy): returns a
    /// `DeviceBuf` view for kernels and the owning `PinnedBuf` for host access. Free the
    /// `PinnedBuf` (the `DeviceBuf` view is released with `free_device`).
    // ---- Virtual-memory paging (paged KV) ----
    /// Mapping granularity in bytes, or `None` when the backend cannot page memory.
    fn vmm_granularity(&self) -> Option<usize> {
        None
    }
    /// Reserves `size` bytes of device address space (nothing behind it until mapped).
    fn vmm_reserve(&self, _size: usize) -> Result<DeviceBuf> {
        Err(crate::SmallTideError::Unsupported("vmm".into()))
    }
    /// Creates a physical allocation of `size` bytes (a multiple of the granularity).
    fn vmm_create(&self, _size: usize) -> Result<u64> {
        Err(crate::SmallTideError::Unsupported("vmm".into()))
    }
    /// Maps physical allocation `handle` (all of it) at `va + offset`, read / write.
    fn vmm_map(&self, _va: DeviceBuf, _offset: usize, _handle: u64) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("vmm".into()))
    }
    /// Unmaps `size` bytes at `va + offset`.
    fn vmm_unmap(&self, _va: DeviceBuf, _offset: usize, _size: usize) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("vmm".into()))
    }

    /// GPU-side wait: `stream` stalls until the u32 at `flag` (device view of mapped host memory)
    /// is >= `value`. Unsupported backends return an error.
    fn stream_wait_value_ge(&self, _stream: Stream, _flag: DeviceBuf, _value: u32) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("stream wait value".into()))
    }

    /// Allocates coherent host-mapped memory suitable for CPU<->GPU signals and wait flags.
    fn alloc_host_signal(&self, size_bytes: usize) -> Result<(DeviceBuf, PinnedBuf)> {
        self.alloc_host_mapped(size_bytes)
    }

    /// Copies `n_words` u32 values from `src` to coherent host-mapped `dst[0..n_words]`, flushes
    /// system-scope stores, and publishes `ticket` to `dst[sentinel_idx]`.
    fn rescue_write_route(
        &self,
        _dst: DeviceBuf,
        _src: DeviceBuf,
        _n_words: usize,
        _sentinel_idx: usize,
        _ticket: u32,
        _stream: Stream,
    ) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("rescue_write_route".into()))
    }

    /// Post-wait compute-stream rescue barrier: if `patch_hdr[0] != 0`, invalidates GPU L2
    /// (system-scope acquire) and applies slot-table updates on `stream`.
    #[allow(clippy::too_many_arguments)]
    fn rescue_apply_patch(
        &self,
        _slot_table: DeviceBuf,
        _patch_hdr: DeviceBuf,
        _full_table: DeviceBuf,
        _total_entries: usize,
        _probe: Option<DeviceBuf>,
        _layer_id: usize,
        _ticket: u32,
        _stream: Stream,
    ) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("rescue_apply_patch".into()))
    }

    /// Fused GPU-side spin-wait (`*flag >= ticket`) + compute-stream slot-table patch + GL2
    /// invalidate. Replaces `stream_wait_value_ge` + `rescue_apply_patch` in one kernel launch.
    #[allow(clippy::too_many_arguments)]
    fn rescue_wait_and_patch(
        &self,
        _slot_table: DeviceBuf,
        _flag: DeviceBuf,
        _ticket: u32,
        _patch_hdr: DeviceBuf,
        _full_table: DeviceBuf,
        _total_entries: usize,
        _probe: Option<DeviceBuf>,
        _layer_id: usize,
        _stream: Stream,
    ) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("rescue_wait_and_patch".into()))
    }

    fn alloc_host_mapped(&self, _size_bytes: usize) -> Result<(DeviceBuf, PinnedBuf)> {
        Err(crate::SmallTideError::Unsupported("host-mapped memory".into()))
    }
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
    /// Non-blocking: true once all work captured by the last `record_event` has completed.
    fn query_event(&self, event: Event) -> Result<bool>;
    fn sync_stream(&self, stream: Stream) -> Result<()>;

    /// Dev profiling: marks the start of a GPU timeline segment named `label` on `stream`
    /// (the segment ends at the next mark). No-op unless the backend enabled segment timing.
    fn seg_mark(&self, _label: &'static str, _stream: Stream) {}

    /// Non-blocking D2H DMA from device buffer into a pinned host buffer (ordered on `stream`).
    fn memcpy_d2h_async(
        &self,
        dst: PinnedBuf,
        dst_offset: usize,
        src: DeviceBuf,
        src_offset: usize,
        len: usize,
        stream: Stream,
    ) -> Result<()>;

    /// Non-blocking device-to-device copy (ordered on `stream`).
    fn memcpy_d2d_async(
        &self,
        dst: DeviceBuf,
        dst_offset: usize,
        src: DeviceBuf,
        src_offset: usize,
        len: usize,
        stream: Stream,
    ) -> Result<()>;

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

    // --- Rows-batched ops (chunked prefill; M = `rows`) ---

    /// `y[m, n] = (W_fp8[n, :] . x[m, :]) * row_scales[n]` for `rows` rows.
    #[allow(clippy::too_many_arguments)]
    /// `gemm_fp8_rows` for batched decode (<= 8 rows): a bandwidth-bound GEMV over all rows.
    #[allow(clippy::too_many_arguments)]
    fn gemm_fp8_rows_decode(&self, y: DeviceBuf, x: DeviceBuf, w_fp8: DeviceBuf, row_scales: DeviceBuf,
        rows: usize, n_rows: usize, k_cols: usize, stream: Stream) -> Result<()>;

    fn gemm_fp8_rows(&self, y: DeviceBuf, x: DeviceBuf, w_fp8: DeviceBuf, row_scales: DeviceBuf,
        rows: usize, n_rows: usize, k_cols: usize, stream: Stream) -> Result<()>;

    /// `out[n] = W_bf16[n, :] . x` (single-row decode / unembed with native BF16 weights).
    fn gemv_bf16(&self, _out: DeviceBuf, _x: DeviceBuf, _w_bf16: DeviceBuf,
        _n_rows: usize, _k_cols: usize, _stream: Stream) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("gemv_bf16".into()))
    }

    /// `y[m, n] = W_bf16[n, :] . x[m, :]` (decode, verify, or prefill with native BF16 weights).
    fn gemm_bf16_rows(&self, _y: DeviceBuf, _x: DeviceBuf, _w_bf16: DeviceBuf,
        _rows: usize, _n_rows: usize, _k_cols: usize, _stream: Stream) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("gemm_bf16_rows".into()))
    }

    /// `y[m, n] = W_f32[n, :] . x[m, :]`.
    #[allow(clippy::too_many_arguments)]
    fn gemm_f32_rows(&self, y: DeviceBuf, x: DeviceBuf, w: DeviceBuf, rows: usize, n_rows: usize,
        k_cols: usize, stream: Stream) -> Result<()>;

    /// Attention-residual mix for `rows` rows: blocks `[nb][M][H]`, cur `[M][H]`, out `[nq][M][H]`.
    #[allow(clippy::too_many_arguments)]
    fn attn_res_mix_rows(&self, out: DeviceBuf, blocks: Option<DeviceBuf>, cur: DeviceBuf,
        queries: DeviceBuf, num_blocks: usize, num_queries: usize, hidden: usize, eps: f32,
        rows: usize, stream: Stream) -> Result<()>;

    /// Per-head L2 clip over `total_heads` (rows x heads_per_row) with optional scale and tau.
    #[allow(clippy::too_many_arguments)]
    fn head_l2_clip_rows(&self, x: DeviceBuf, scale: Option<DeviceBuf>, total_heads: usize,
        head_dim: usize, heads_per_row: usize, threshold: f32, tau: Option<(usize, f32, f32)>,
        stream: Stream) -> Result<()>;

    /// `sconv_k4_rows` that also writes `hist` = `[3 + rows][dim]` (old state, then the inputs).
    #[allow(clippy::too_many_arguments)]
    fn sconv_k4_rows_hist(&self, out: DeviceBuf, state: DeviceBuf, x: DeviceBuf, weight: DeviceBuf,
        dyn_taps: Option<DeviceBuf>, dim: usize, head_dim: usize, rows: usize, dyn_stride: usize,
        hist: Option<DeviceBuf>, stream: Stream) -> Result<()>;

    /// Rewinds short-conv states from their histories: `entries` = device `[n][4]` u64 (hist, state,
    /// dim, 0); every state becomes the history rows `keep..keep+3`.
    fn sconv_rewind(&self, entries: DeviceBuf, n: usize, max_dim: usize, keep: usize, stream: Stream) -> Result<()>;

    /// Grouped (WMMA) MoE for <= 8 decode / verify rows with the expert grouping built on the GPU:
    /// `desc` receives the plan (work at 0, pairs at 1024 bytes); `tmp` gets unweighted outputs
    /// (`moe_post_rows` / `moe_decode_post` apply weights + renorm). `xscratch` (>= rows * H f32) is
    /// clobbered (f16 activations).
    #[allow(clippy::too_many_arguments)]
    fn moe_rows_grouped(&self, mid: DeviceBuf, tmp: DeviceBuf, x: DeviceBuf, xscratch: DeviceBuf, route: DeviceBuf,
        slot_table: DeviceBuf, pool: DeviceBuf, desc: DeviceBuf, rows: usize, params: MoeDispatchParams,
        stream: Stream) -> Result<()>;

    /// `out[r] = sum_k w_rk * renorm_r * tmp[r*8+k]` for `rows` rows (no conv / residual).
    fn moe_post_rows(&self, out: DeviceBuf, tmp: DeviceBuf, slot_table: DeviceBuf, route: DeviceBuf,
        rows: usize, hidden: usize, stream: Stream) -> Result<()>;

    /// Verify MoE epilogue: per row renormalized expert sum, MLP short conv row by row (+ history),
    /// residual `cur = ffn_inp + conv(sum)`.
    #[allow(clippy::too_many_arguments)]
    fn moe_post_sconv_rows(&self, cur: DeviceBuf, ffn_inp: DeviceBuf, tmp: DeviceBuf, slot_table: DeviceBuf, route: DeviceBuf,
        state: DeviceBuf, weight: DeviceBuf, hist: Option<DeviceBuf>, hidden: usize, rows: usize, stream: Stream) -> Result<()>;

    /// Rows (<= 8, one sequence or lanes) version of `fused_mix_decode` without the short conv:
    /// residual add / block boundary, copy, attention-residual mix, per-query RMSNorm.
    fn mix_rows(&self, p: MixRowsParams, stream: Stream) -> Result<()>;

    /// Loop-controller prep for up to 4 items `(out, cur, msg)` over the same `rows`.
    fn ctrl_prep_multi(&self, items: &[(DeviceBuf, DeviceBuf, DeviceBuf)], hidden: usize, eps: f32, rows: usize, stream: Stream) -> Result<()>;
    /// Loop-controller apply for up to 4 items `(out, cur, msg, gain, forget, norm)` (`norm`:
    /// optional `(weight, norm_out)` RMSNorm of the result) over the same `rows`.
    #[allow(clippy::type_complexity)]
    fn ctrl_apply_norm_multi(&self, items: &[(DeviceBuf, DeviceBuf, DeviceBuf, DeviceBuf, DeviceBuf, Option<(DeviceBuf, DeviceBuf)>)],
        hidden: usize, eps: f32, rows: usize, stream: Stream) -> Result<()>;

    /// `out` (f16) = `x` (f32) over `n` elements.
    fn f32_to_f16(&self, out: DeviceBuf, x: DeviceBuf, n: usize, stream: Stream) -> Result<()>;

    // ---- Vision tower (f32 activations, BF16 weights) ----
    /// `y[m][n]` (f32) = `x[m][k]` (f32, rounded to bf16) . `w[n][k]`^T (bf16); `k` a multiple of 32.
    #[allow(clippy::too_many_arguments)]
    fn vit_gemm_bf16(&self, _y: DeviceBuf, _x: DeviceBuf, _w: DeviceBuf, _m: usize, _n: usize, _k: usize,
        _stream: Stream) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("vision".into()))
    }
    /// A Linear's epilogue over `[m][n]`: `+ bias`, GELU when `gelu`, `* gamma`, `+ residual`
    /// (each optional; bias / gamma bf16 `[n]`). `out` may be `y` or `residual`.
    #[allow(clippy::too_many_arguments)]
    fn vit_epilogue(&self, _out: DeviceBuf, _y: DeviceBuf, _bias: Option<DeviceBuf>, _gamma: Option<DeviceBuf>,
        _residual: Option<DeviceBuf>, _m: usize, _n: usize, _gelu: bool, _stream: Stream) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("vision".into()))
    }
    /// LayerNorm over `rows` rows of `n` (weight / bias bf16 `[n]`). `out` may be `x`.
    #[allow(clippy::too_many_arguments)]
    fn vit_ln(&self, _out: DeviceBuf, _x: DeviceBuf, _w: DeviceBuf, _b: DeviceBuf, _rows: usize, _n: usize, _eps: f32,
        _stream: Stream) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("vision".into()))
    }
    /// Axial 2D RoPE on q and k of `qkv [rows][3][heads][64]` in place; `cs` / `sn` f32 `[rows][32]`.
    fn vit_rope(&self, _qkv: DeviceBuf, _rows: usize, _heads: usize, _cs: DeviceBuf, _sn: DeviceBuf,
        _stream: Stream) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("vision".into()))
    }
    /// Full attention over `rows`: `qkv [rows][3][heads][64]` -> `out [rows][heads][64]`.
    fn vit_attn(&self, _out: DeviceBuf, _qkv: DeviceBuf, _rows: usize, _heads: usize, _stream: Stream) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("vision".into()))
    }

    // ---- DSpark drafter ----
    /// `out[r] = RMSNorm(sum_t alpha[t] * taps[t][r]) * w` over `rows`; taps `[4][tap_stride][d]`.
    #[allow(clippy::too_many_arguments)]
    fn dspark_mix_rms(&self, out: DeviceBuf, taps: DeviceBuf, tap_stride: usize, alpha: [f32; 4], w: DeviceBuf,
        rows: usize, d: usize, eps: f32, stream: Stream) -> Result<()>;
    /// Per-head RMSNorm (w `[128]`) + neox RoPE at positions `pos0 + row`, in place on `[rows][nh][128]`.
    #[allow(clippy::too_many_arguments)]
    fn dspark_head_norm_rope(&self, x: DeviceBuf, w: DeviceBuf, nh: usize, rows: usize, pos0: usize, theta: f32,
        eps: f32, stream: Stream) -> Result<()>;
    /// f16 context K/V store of `rows` rows at positions `pos0..` (`ring` = 0: linear).
    #[allow(clippy::too_many_arguments)]
    fn dspark_ctx_store(&self, kc: DeviceBuf, vc: DeviceBuf, k: DeviceBuf, v: DeviceBuf, pos0: usize, rows: usize,
        ring: usize, dim: usize, stream: Stream) -> Result<()>;
    /// Drafter attention of `rows` query rows over context `[lo, p)` plus `nblk` block keys.
    #[allow(clippy::too_many_arguments)]
    fn dspark_attn(&self, out: DeviceBuf, q: DeviceBuf, kb: DeviceBuf, vb: DeviceBuf, kc: DeviceBuf, vc: DeviceBuf,
        rows: usize, p: usize, lo: usize, ring: usize, nh: usize, nkv: usize, nblk: usize, scale: f32,
        stream: Stream) -> Result<()>;
    /// `out = silu(g) * u` over `n` elements.
    fn silu_mul(&self, out: DeviceBuf, g: DeviceBuf, u: DeviceBuf, n: usize, stream: Stream) -> Result<()>;
    /// One Markov-head step: z assembly, rnn GEMV, cell, folded logits + argmax into `drafts[step]`.
    #[allow(clippy::too_many_arguments)]
    fn dspark_markov_step(&self, z: DeviceBuf, raw: DeviceBuf, state: DeviceBuf, latent: DeviceBuf, w1: DeviceBuf,
        drafts: DeviceBuf, step: usize, anchor: u32, h: DeviceBuf, frnn: DeviceBuf, rnn_w: DeviceBuf, rnn_b: DeviceBuf,
        base: DeviceBuf, fold_b: DeviceBuf, fold16: DeviceBuf, pmax: DeviceBuf, pidx: DeviceBuf, r: usize, d: usize,
        v: usize, cap: f32, stream: Stream) -> Result<()>;

    /// Quantizes `[r][v]` f32 fold matrix -> `[r][v_pad]` FP8 E4M3FN + `[r]` f32 row scales.
    #[allow(clippy::too_many_arguments)]
    fn dspark_quant_fold_fp8(&self, _dst: DeviceBuf, _scales: DeviceBuf, _src: DeviceBuf,
        _r: usize, _v: usize, _v_pad: usize, _stream: Stream) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("dspark_quant_fold_fp8".into()))
    }

    /// Optimized Markov step: fused RNN step (f16 W_dyn + precomputed frnn_proj & hb_proj) +
    /// vectorized FP8 fold matvec + top-2 argmax into `drafts[step]` and `margins[step]`.
    #[allow(clippy::too_many_arguments)]
    fn dspark_markov_step_fast(&self, _state: DeviceBuf, _latent: DeviceBuf, _w1: DeviceBuf,
        _drafts: DeviceBuf, _margins: DeviceBuf, _step: usize, _anchor: u32, _w_dyn: DeviceBuf,
        _frnn_proj: DeviceBuf, _hb_proj: DeviceBuf, _rnn_b: DeviceBuf, _base: DeviceBuf,
        _fold_b: DeviceBuf, _fold_fp8: DeviceBuf, _fold_scales: DeviceBuf,
        _pmax: DeviceBuf, _pidx: DeviceBuf, _pmax2: DeviceBuf,
        _r: usize, _v: usize, _v_pad: usize, _stream: Stream) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("dspark_markov_step_fast".into()))
    }

    /// Parallel 2-pass greedy argmax of `rows` rows of `n` f32: `out[r]` u32 (`scratch` holds
    /// `rows * 16` f32 values + `rows * 16` u32 indices).
    fn argmax_rows_fast(&self, out: DeviceBuf, scratch: DeviceBuf, x: DeviceBuf, rows: usize, n: usize, stream: Stream) -> Result<()> {
        let _ = scratch;
        self.argmax_rows(out, x, rows, n, stream)
    }

    /// Split-K WMMA causal attention for low-row speculative verify (`rows <= 8`). Falls back to
    /// `attn_prefill_fp8` when key count is short or backend lacks Split-K.
    #[allow(clippy::too_many_arguments)]
    fn attn_prefill_fp8_split(&self, out: DeviceBuf, _part_o: DeviceBuf, _part_m: DeviceBuf, _part_l: DeviceBuf,
        q: DeviceBuf, k_cache: DeviceBuf, k_scales: DeviceBuf, v_cache: DeviceBuf, v_scales: DeviceBuf,
        rel_table: Option<DeviceBuf>, sinks: Option<DeviceBuf>, params: PrefillAttnParams, stream: Stream) -> Result<()> {
        self.attn_prefill_fp8(out, q, k_cache, k_scales, v_cache, v_scales, rel_table, sinks, params, stream)
    }

    /// FP8/K8V4 context K/V store of `rows` rows at positions `pos0..` (`ring` = 0: linear;
    /// `page_table`: the sequence's pages when the cache is the shared paged KV pool).
    #[allow(clippy::too_many_arguments)]
    fn dspark_ctx_store_fp8(&self, _kc: DeviceBuf, _ks: DeviceBuf, _vc: DeviceBuf, _vs: DeviceBuf,
        _k: DeviceBuf, _v: DeviceBuf, _pos0: usize, _rows: usize, _ring: usize, _nkv: usize,
        _v_q4: bool, _page_table: Option<DeviceBuf>, _stream: Stream) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("dspark_ctx_store_fp8".into()))
    }

    /// Grouped Split-K FP8/K8V4 drafter attention (`rows == nblk == 8`, `nh == 3 * nkv`).
    #[allow(clippy::too_many_arguments)]
    fn dspark_attn_fp8(&self, _out: DeviceBuf, _part_o: DeviceBuf, _part_m: DeviceBuf, _part_l: DeviceBuf,
        _q: DeviceBuf, _kb: DeviceBuf, _vb: DeviceBuf, _kc: DeviceBuf, _ks: DeviceBuf,
        _vc: DeviceBuf, _vs: DeviceBuf, _rows: usize, _p: usize, _lo: usize, _ring: usize,
        _nh: usize, _nkv: usize, _nblk: usize, _scale: f32, _v_q4: bool, _page_table: Option<DeviceBuf>, _stream: Stream) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("dspark_attn_fp8".into()))
    }

    /// Fused drafter block input prep (`res[0]=rmsnorm(anchor_in, embed_norm)`, `res[1..8]=mask_embed`, `xn=rmsnorm(res, in_ln)`).
    #[allow(clippy::too_many_arguments)]
    fn dspark_embed_prep(&self, _res: DeviceBuf, _xn: DeviceBuf, _anchor_in: DeviceBuf,
        _embed_norm: DeviceBuf, _mask_embed: DeviceBuf, _in_ln: DeviceBuf, _d: usize, _eps: f32,
        _stream: Stream) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("dspark_embed_prep".into()))
    }

    /// Fused `res += tmp; xn = RMSNorm(res, w)` over `rows` rows.
    #[allow(clippy::too_many_arguments)]
    fn add_rmsnorm_rows(&self, _res: DeviceBuf, _xn: DeviceBuf, _tmp: DeviceBuf, _w: DeviceBuf,
        _rows: usize, _d: usize, _eps: f32, _stream: Stream) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("add_rmsnorm_rows".into()))
    }

    /// Fused Q + K per-head RMSNorm + neox RoPE in one launch.
    #[allow(clippy::too_many_arguments)]
    fn dspark_qk_norm_rope(&self, _q: DeviceBuf, _k: DeviceBuf, _qw: DeviceBuf, _kw: DeviceBuf,
        _hq: usize, _hk: usize, _rows: usize, _pos0: usize, _theta: f32, _eps: f32,
        _stream: Stream) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("dspark_qk_norm_rope".into()))
    }

    /// DFlash 2 grouped Split-K FP8/K8V4 sliding-window attention (`rows == nblk == 8`, `nh == 4 * nkv`).
    #[allow(clippy::too_many_arguments)]
    fn dflash2_attn_fp8(&self, _out: DeviceBuf, _part_o: DeviceBuf, _part_m: DeviceBuf, _part_l: DeviceBuf,
        _q: DeviceBuf, _kb: DeviceBuf, _vb: DeviceBuf, _kc: DeviceBuf, _ks: DeviceBuf,
        _vc: DeviceBuf, _vs: DeviceBuf, _rows: usize, _p: usize, _window: usize, _ring: usize,
        _nh: usize, _nkv: usize, _nblk: usize, _scale: f32, _v_q4: bool, _page_table: Option<DeviceBuf>,
        _stream: Stream) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("dflash2_attn_fp8".into()))
    }

    /// DFlash 2 grouped 2-tap dynamic causal convolution (prepare / side 0).
    #[allow(clippy::too_many_arguments)]
    fn dflash2_conv_prepare(&self, _out: DeviceBuf, _x: DeviceBuf, _coeff: DeviceBuf, _base: DeviceBuf,
        _rows: usize, _d: usize, _group_size: usize, _stream: Stream) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("dflash2_conv_prepare".into()))
    }

    /// DFlash 2 fused grouped 2-tap dynamic conv (finish / side 1) + residual add + RMSNorm.
    #[allow(clippy::too_many_arguments)]
    fn dflash2_conv_finish_add_rmsnorm(&self, _res: DeviceBuf, _xn: DeviceBuf, _tmp: DeviceBuf,
        _coeff: DeviceBuf, _base: DeviceBuf, _w: DeviceBuf, _rows: usize, _d: usize,
        _group_size: usize, _eps: f32, _stream: Stream) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("dflash2_conv_finish_add_rmsnorm".into()))
    }

    /// Exact top-16 `(value, index)` per row over `x [rows, n]`.
    #[allow(clippy::too_many_arguments)]
    fn topk16_rows(&self, _out_val: DeviceBuf, _out_idx: DeviceBuf,
        _part_val: DeviceBuf, _part_idx: DeviceBuf, _x: DeviceBuf,
        _rows: usize, _n: usize, _stream: Stream) -> Result<()> {
        Err(crate::SmallTideError::Unsupported("topk16_rows".into()))
    }

    /// Device address of `buf` (for pointer tables consumed by kernels).
    fn device_address(&self, buf: DeviceBuf) -> Result<u64>;

    /// Greedy argmax of `rows` rows of `n` f32: `out[r]` u32.
    fn argmax_rows(&self, out: DeviceBuf, x: DeviceBuf, rows: usize, n: usize, stream: Stream) -> Result<()>;

    /// Causal k=4 short conv over `rows` consecutive tokens (continues and updates `state`).
    #[allow(clippy::too_many_arguments)]
    fn sconv_k4_rows(&self, out: DeviceBuf, state: DeviceBuf, x: DeviceBuf, weight: DeviceBuf,
        dyn_taps: Option<DeviceBuf>, dim: usize, head_dim: usize, rows: usize, dyn_stride: usize,
        stream: Stream) -> Result<()>;

    /// FP8/K8V4 KV store of `rows` tokens at positions `pos0..` (`is_local`: ring of `max_seq`).
    #[allow(clippy::too_many_arguments)]
    fn kv_store_fp8_rows(&self, k_cache: DeviceBuf, k_scales: DeviceBuf, v_cache: DeviceBuf,
        v_scales: DeviceBuf, k_in: DeviceBuf, v_in: DeviceBuf, n_kv_heads: usize, max_seq: usize,
        head_dim: usize, pos0: usize, rows: usize, is_local: bool, page_table: Option<DeviceBuf>,
        v_q4: bool, stream: Stream) -> Result<()>;

    /// Multi-row causal attention over the FP8 KV cache (relative bias + sinks).
    #[allow(clippy::too_many_arguments)]
    fn attn_prefill_fp8(&self, out: DeviceBuf, q: DeviceBuf, k_cache: DeviceBuf, k_scales: DeviceBuf,
        v_cache: DeviceBuf, v_scales: DeviceBuf, rel_table: Option<DeviceBuf>, sinks: Option<DeviceBuf>,
        params: PrefillAttnParams, stream: Stream) -> Result<()>;

    /// Router top-8 for `rows` rows from logits `[M][128]`; writes route rows of 16 u32 words.
    #[allow(clippy::too_many_arguments)]
    fn router_topk8_rows(&self, route: DeviceBuf, logits: DeviceBuf, bias: Option<DeviceBuf>,
        route_scale: f32, rows: usize, row_stride_words: usize, stream: Stream) -> Result<()>;

    /// Loop controller prep for `rows` rows: out `[M][2H]`.
    fn loop_control_prep_rows(&self, out: DeviceBuf, cur: DeviceBuf, msg: DeviceBuf, hidden: usize,
        eps: f32, rows: usize, stream: Stream) -> Result<()>;

    /// Expert-grouped MoE (w13+SiTU -> w2 -> fixed-order reduce) for prefill. `desc` holds
    /// `[n_work][4]` work items `(slot, pair_start, n_pairs, _)` then the pair list `row*8+k`.
    /// When `reduce` is false only the per-pair partials in `tmp` are written.
    #[allow(clippy::too_many_arguments)]
    fn moe_grouped(&self, out: DeviceBuf, mid: DeviceBuf, tmp: DeviceBuf, x: DeviceBuf, route: DeviceBuf,
        pool: DeviceBuf, desc: DeviceBuf, params: GroupedMoeParams, reduce: bool, stream: Stream) -> Result<()>;

    /// Decode MoE (v2): `mid[k][I]` then unweighted per-expert outputs `tmp[k][H]` for the
    /// routed experts (skipped experts write 0).
    #[allow(clippy::too_many_arguments)]
    fn moe_decode_experts(&self, mid: DeviceBuf, tmp: DeviceBuf, x: DeviceBuf, pool: DeviceBuf,
        slot_table: DeviceBuf, topk_ids: DeviceBuf, params: MoeDispatchParams, stream: Stream) -> Result<()>;

    /// Batched decode MoE over `rows` independent rows (<= 8): `route` is `[rows][16]` (8 ids,
    /// 8 weights), `x` `[rows][H]`, `mid` `[rows*8][I]`, `tmp` `[rows*8][H]` (unweighted, 0 for
    /// missing experts). Experts routed by several rows are dequantized once for all of them.
    #[allow(clippy::too_many_arguments)]
    fn moe_decode_rows(&self, mid: DeviceBuf, tmp: DeviceBuf, x: DeviceBuf, pool: DeviceBuf,
        slot_table: DeviceBuf, route: DeviceBuf, rows: usize, params: MoeDispatchParams, stream: Stream) -> Result<()>;

    /// `out = residual + sconv(sum_k w_k*renorm*tmp[k])` (or just the weighted sum when
    /// `sconv` is None), renormalized over the experts present in `slot_table`.
    #[allow(clippy::too_many_arguments)]
    fn moe_decode_post(&self, out: DeviceBuf, tmp: DeviceBuf, slot_table: DeviceBuf, topk_ids: DeviceBuf,
        topk_weights: DeviceBuf, top_k: usize, sconv: Option<(DeviceBuf, DeviceBuf, DeviceBuf)>,
        hidden: usize, stream: Stream) -> Result<()>;

    /// One decode token's attention prep: K/V short convs (+dynamic taps), K L2 clip, V unit
    /// RMS norm, FP8 KV store at `slot`, Q/R scaled L2 clip (+tau) and the relative-bias table.
    fn attn_prep_decode(&self, p: AttnPrepParams, stream: Stream) -> Result<()>;

    /// `attn_prep_decode` over `rows` consecutive positions `pos0..` of one sequence (short convs
    /// row by row; `hist` = (k, v) verify histories `[3 + rows][dim]`; `tau` = (alpha, floor) for
    /// global layers; `p.slot` / `p.tau` unused).
    #[allow(clippy::too_many_arguments)]
    fn attn_prep_rows(&self, p: AttnPrepParams, hist: Option<(DeviceBuf, DeviceBuf)>, pos0: usize, is_local: bool,
        rows: usize, tau: Option<(f32, f32)>, stream: Stream) -> Result<()>;

    /// One-workgroup decode fusion of: attention-output short conv + residual / block boundary,
    /// copy of the residual, attention-residual mix with up to 3 queries, per-query RMSNorm.
    fn fused_mix_decode(&self, p: FusedMixParams, stream: Stream) -> Result<()>;

    /// `gemv_multi` over `rows` (<= 8) contiguous rows of every segment's x / y.
    fn gemv_multi_rows(&self, segs: &[GemvSegment], rows: usize, stream: Stream) -> Result<()>;

    /// Several independent GEMVs in as few launches as possible (8 segments per launch).
    fn gemv_multi(&self, segs: &[GemvSegment], stream: Stream) -> Result<()>;

    /// Final fixed-order reduce of `tmp[M*8][H]` into `out[M][H]`.
    fn moe_reduce_rows(&self, out: DeviceBuf, tmp: DeviceBuf, hidden: usize, rows: usize, stream: Stream) -> Result<()>;

    /// Extracts `(idx, val)` entries per row with `val >= row_max[r] - margin` into compact
    /// `[rows][cap]` buffers (for fast stochastic verify sampling: ~16 KB D2H instead of 6.4 MB).
    #[allow(clippy::too_many_arguments)]
    fn logits_filter_rows(
        &self,
        out_idx: DeviceBuf,
        out_val: DeviceBuf,
        counts: DeviceBuf,
        row_max: DeviceBuf,
        scratch: DeviceBuf,
        x: DeviceBuf,
        rows: usize,
        n: usize,
        margin: f32,
        cap: usize,
        stream: Stream,
    ) -> Result<()>;
}
