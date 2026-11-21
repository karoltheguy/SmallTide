//! Live 48-Logical-Layer Inkling-2-Mini Inference Engine (`Inkling2Engine<B: ComputeBackend>`).
//!
//! 100% backend-agnostic over `smalltide_core::ComputeBackend`:
//! - Executes exact Inkling-2-Mini Loop architecture (`6` Prelude + `3x12` Loop Core + `6` Coda)
//! - Hybrid attention: 40 local sliding-window (`1024`) + 8 global (`max_ctx` up to `256K`) FP8 KV
//! - Relative position bias (`wr_du` + `rel_logits_proj`), per-head L2 clip, parameter-free V norm,
//!   per-query attention sinks, log-scaled `tau` on global Q/R, residual short-convs (`k=4`, dynamic on global K/V)
//! - Attention Residuals (`attn_res`) + rank-64 cross-pass Loop Core controllers
//! - Integrated 2-stage TideLM adaptive expert executor (`TideExecutor<B>`) with zero-sync fast path
//!   for 100%-VRAM-resident layers (e.g., all 36 Loop Core layers on RX 9070 XT).

use crate::weights::{DenseLayerGpu, Fp8MatrixGpu, Inkling2DenseGpu, LoopControllerGpu};
use smalltide_core::backend::{AttnPrepParams, FlashDecodeParams, GemvSegment, MoeDispatchParams};
use smalltide_core::{
    ComputeBackend, DType, DeviceBuf, Inkling2Config, PhysicalLayerId, PinnedBuf, Result, Stream,
    TideTelemetry,
};
use smalltide_io::quant::bf16_to_f32;
use smalltide_tide::runtime::TideExecutor;
use std::sync::Arc;
use std::time::Instant;

const MAX_FLASH_SPLITS: usize = 64;
/// Maximum tokens per batched prefill chunk. Local KV rings hold `window + PREFILL_CHUNK` slots so
/// a chunk's own keys never overwrite keys an earlier row of the same chunk still attends to.
pub const PREFILL_CHUNK: usize = 128;
const MAX_ATTN_RES_BLOCKS: usize = 8;

/// Optional per-component wall-clock breakdown (accumulated across steps when profiling is enabled).
#[derive(Debug, Default, Clone)]
pub struct StepProfile {
    pub steps: u64,
    pub embed_ms: f64,
    pub res_and_ctrl_ms: f64,
    pub attn_proj_ms: f64,
    pub sconv_clip_rel_ms: f64,
    pub flash_attn_ms: f64,
    pub attn_out_ms: f64,
    pub router_d2h_ms: f64,
    pub tide_rescue_ms: f64,
    pub moe_ms: f64,
    pub lm_head_ms: f64,
    pub tide_post_ms: f64,
}

/// Sampling configuration for text generation.
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0, // Greedy by default for deterministic verification
            top_p: 0.95,
            top_k: 40,
            repetition_penalty: 1.0,
            seed: 42,
        }
    }
}

/// Rewindable sequence state (local KV rings + short-conv histories) after `pos` tokens,
/// held in pinned host RAM. Created and restored by `Inkling2Engine::{snapshot,restore}_state`.
pub struct SeqSnapshot {
    pub pos: usize,
    host: PinnedBuf,
}

/// Per-logical-layer FP8 KV cache + 4-channel causal short-convolution state (`k = 4`, history `3`).
pub(crate) struct LogicalLayerState {
    pub(crate) _is_local: bool,
    /// KV buffer length (ring size for local layers).
    pub(crate) max_seq: usize,
    /// Attention window (local layers; == max_seq for global layers).
    pub(crate) window: usize,
    pub(crate) k_cache: DeviceBuf,  // [8, max_seq, 128] u8 FP8
    pub(crate) k_scales: DeviceBuf, // [8, max_seq] f32
    pub(crate) v_cache: DeviceBuf,  // [8, max_seq, 128] u8 FP8
    pub(crate) v_scales: DeviceBuf, // [8, max_seq] f32
    pub(crate) sconv_k: DeviceBuf,  // [3, 1024] f32
    pub(crate) sconv_v: DeviceBuf,  // [3, 1024] f32
    pub(crate) sconv_attn: DeviceBuf, // [3, 2304] f32
    pub(crate) sconv_mlp: DeviceBuf,  // [3, 2304] f32
}

/// Reusable activation scratch buffers in VRAM (~4 MiB total — zero per-token allocations!).
pub(crate) struct ActivationScratch {
    cur: DeviceBuf,         // [H]
    embed_tmp: DeviceBuf,   // [H]
    inp_sa: DeviceBuf,      // [H]
    ffn_inp: DeviceBuf,     // [H]
    blocks: DeviceBuf,      // [MAX_ATTN_RES_BLOCKS, H]
    prelude_cur: DeviceBuf, // [H]
    bank: DeviceBuf,        // [MAX_ATTN_RES_BLOCKS, H]
    bank_out: DeviceBuf,    // [H]
    messages: DeviceBuf,    // [4, H]

    in_qkv: DeviceBuf,      // [3, H]
    ctrl_qkv: DeviceBuf,    // [3, H]
    norm_qkv: DeviceBuf,    // [3, H]

    ctrl_prep: DeviceBuf,   // [2 * H]
    ctrl_gate: DeviceBuf,   // [rank = 64]
    ctrl_gain: DeviceBuf,   // [H]
    ctrl_forget: DeviceBuf, // [H]

    q: DeviceBuf,           // [24 * 128 = 3072]
    k_raw: DeviceBuf,       // [1024]
    k: DeviceBuf,           // [1024]
    v_raw: DeviceBuf,       // [1024]
    v: DeviceBuf,           // [1024]
    k_dyn: DeviceBuf,       // [32]
    v_dyn: DeviceBuf,       // [32]
    r: DeviceBuf,           // [24 * 16 = 384]
    sinks: DeviceBuf,       // [24]
    rel_table: DeviceBuf,   // [24 * 1024]

    partial_out: DeviceBuf, // [24 * MAX_FLASH_SPLITS * 128]
    partial_m: DeviceBuf,   // [24 * MAX_FLASH_SPLITS]
    partial_l: DeviceBuf,   // [24 * MAX_FLASH_SPLITS]
    attn_out: DeviceBuf,    // [3072]
    gate_out: DeviceBuf,    // [3072]
    o_raw: DeviceBuf,       // [H]
    sconv_tmp: DeviceBuf,   // [H]

    moe_in: DeviceBuf,      // [H]
    moe_ctrl: DeviceBuf,    // [H]
    moe_norm: DeviceBuf,    // [H]
    /// Per-logical-layer router output log: `[48][16]` u32 words = 8 ids then 8 f32 weights.
    /// One combined D2H per partial layer, one async D2H per token for heat bookkeeping.
    route_log: DeviceBuf,
    router_logits: DeviceBuf, // [128] f32
    moe_mid: DeviceBuf,     // [8 * 1152] f32
    moe_tmp: DeviceBuf,     // [8 * H] per-expert outputs
    moe_out: DeviceBuf,     // [H]

    pub(crate) logits: DeviceBuf, // [unpadded_vocab_size] f32
    argmax_buf: DeviceBuf,  // [64] f32 vals + [64] u32 idx (single D2H)
}

const ROUTE_WORDS: usize = 16;

/// Optional per-token hidden-state dump for reference verification (`SMALLTIDE_DUMP_DIR`).
pub(crate) struct DumpState {
    pub(crate) dir: std::path::PathBuf,
    pub(crate) hidden: Vec<f32>, // [num_logical * H]
    pub(crate) routes: Vec<u32>, // [num_logical * 8]
}

pub struct Inkling2Engine<B: ComputeBackend> {
    pub backend: Arc<B>,
    pub cfg: Inkling2Config,
    pub dense: Inkling2DenseGpu<B>,
    pub tide: TideExecutor<B>,
    pub global_max_seq: usize,
    pub(crate) layer_states: Vec<LogicalLayerState>,
    pub(crate) scratch: ActivationScratch,
    /// Rows-batched activation buffers for chunked prefill (allocated on first use).
    pub(crate) pf: Option<crate::prefill::PrefillScratch>,
    embed_pinned: PinnedBuf,
    pub(crate) route_log_inflight: bool,
    /// A post-decode Tide step deferred into the next forward (see `post_decode_tide_step`).
    pub(crate) tide_step_pending: bool,
    route_log_is_prefill: bool,
    /// Logical layers whose routing the scheduler already saw this token (partial layers).
    layer_planned: Vec<bool>,
    /// Partial layers dispatched without a host readback (rescue disabled): their Tide
    /// bookkeeping (hit/miss, heat, async promotion queue) runs from the routing log afterwards.
    layer_deferred: Vec<bool>,
    pub(crate) dump: Option<DumpState>,
    pub(crate) dump_token: (u32, usize),
    rng_state: u64,
    tokens_decoded: u64,
    pub profile_enabled: bool,
    pub profile: StepProfile,
}

impl<B: ComputeBackend> Inkling2Engine<B> {
    pub fn new(
        cfg: Inkling2Config,
        dense: Inkling2DenseGpu<B>,
        tide: TideExecutor<B>,
        global_max_seq: usize,
    ) -> Result<Self> {
        let backend = dense.backend.clone();
        let stream = tide.compute_stream;
        let h = cfg.hidden_size;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let num_logical = cfg.num_logical_layers();

        let mut layer_states = Vec::with_capacity(num_logical);
        for il in 0..num_logical {
            let (phys, _) = crate::map_logical_to_physical(&cfg, smalltide_core::LogicalLayerId(il as u16));
            let is_local = cfg.is_local_layer(phys.0 as usize);
            let (max_seq, window) = if is_local {
                (cfg.local_window_size + PREFILL_CHUNK, cfg.local_window_size)
            } else {
                let g = global_max_seq.max(cfg.local_window_size);
                (g, g)
            };
            let kv_bytes = cfg.num_key_value_heads * max_seq * cfg.head_dim;
            let sc_bytes = cfg.num_key_value_heads * max_seq * 4;
            let sconv_k = backend.alloc_device(3 * kv_dim * 4)?;
            let sconv_v = backend.alloc_device(3 * kv_dim * 4)?;
            let sconv_attn = backend.alloc_device(3 * h * 4)?;
            let sconv_mlp = backend.alloc_device(3 * h * 4)?;
            backend.zero_f32(sconv_k, 3 * kv_dim, stream)?;
            backend.zero_f32(sconv_v, 3 * kv_dim, stream)?;
            backend.zero_f32(sconv_attn, 3 * h, stream)?;
            backend.zero_f32(sconv_mlp, 3 * h, stream)?;
            layer_states.push(LogicalLayerState {
                _is_local: is_local,
                max_seq,
                window,
                k_cache: backend.alloc_device(kv_bytes)?,
                k_scales: backend.alloc_device(sc_bytes)?,
                v_cache: backend.alloc_device(kv_bytes)?,
                v_scales: backend.alloc_device(sc_bytes)?,
                sconv_k,
                sconv_v,
                sconv_attn,
                sconv_mlp,
            });
        }

        let alloc_f32 = |n: usize| backend.alloc_device(n * 4);
        let scratch = ActivationScratch {
            cur: alloc_f32(h)?,
            embed_tmp: alloc_f32(h)?,
            inp_sa: alloc_f32(h)?,
            ffn_inp: alloc_f32(h)?,
            blocks: alloc_f32(MAX_ATTN_RES_BLOCKS * h)?,
            prelude_cur: alloc_f32(h)?,
            bank: alloc_f32(MAX_ATTN_RES_BLOCKS * h)?,
            bank_out: alloc_f32(h)?,
            messages: alloc_f32(4 * h)?,
            in_qkv: alloc_f32(3 * h)?,
            ctrl_qkv: alloc_f32(3 * h)?,
            norm_qkv: alloc_f32(3 * h)?,
            ctrl_prep: alloc_f32(4 * 2 * h)?,
            ctrl_gate: alloc_f32(4 * cfg.loop_gate_rank)?,
            ctrl_gain: alloc_f32(4 * h)?,
            ctrl_forget: alloc_f32(4 * h)?,
            q: alloc_f32(q_dim)?,
            k_raw: alloc_f32(kv_dim)?,
            k: alloc_f32(kv_dim)?,
            v_raw: alloc_f32(kv_dim)?,
            v: alloc_f32(kv_dim)?,
            k_dyn: alloc_f32(cfg.num_key_value_heads * cfg.sconv_kernel_size)?,
            v_dyn: alloc_f32(cfg.num_key_value_heads * cfg.sconv_kernel_size)?,
            r: alloc_f32(cfg.num_attention_heads * cfg.d_rel)?,
            sinks: alloc_f32(cfg.num_attention_heads)?,
            rel_table: alloc_f32(cfg.num_attention_heads * cfg.rel_extent)?,
            partial_out: alloc_f32(cfg.num_attention_heads * MAX_FLASH_SPLITS * cfg.head_dim)?,
            partial_m: alloc_f32(cfg.num_attention_heads * MAX_FLASH_SPLITS)?,
            partial_l: alloc_f32(cfg.num_attention_heads * MAX_FLASH_SPLITS)?,
            attn_out: alloc_f32(q_dim)?,
            gate_out: alloc_f32(q_dim)?,
            o_raw: alloc_f32(h)?,
            sconv_tmp: alloc_f32(h)?,
            moe_in: alloc_f32(h)?,
            moe_ctrl: alloc_f32(h)?,
            moe_norm: alloc_f32(h)?,
            route_log: backend.alloc_device(num_logical * ROUTE_WORDS * 4)?,
            router_logits: alloc_f32(cfg.n_routed_experts)?,
            moe_mid: alloc_f32(cfg.num_experts_per_tok * cfg.intermediate_size)?,
            moe_tmp: alloc_f32(cfg.num_experts_per_tok * h)?,
            moe_out: alloc_f32(h)?,
            logits: alloc_f32(cfg.unpadded_vocab_size)?,
            argmax_buf: backend.alloc_device(128 * 4)?,
        };
        backend.zero_f32(scratch.route_log, num_logical * ROUTE_WORDS, stream)?;

        // Prefill scratch is allocated eagerly (before the expert pools) so it is resident VRAM.
        let pf = crate::prefill::PrefillScratch::new(&*backend, &cfg)?;
        let embed_pinned = backend.alloc_pinned(h * 4)?;
        backend.sync_stream(stream)?;
        let dump = std::env::var_os("SMALLTIDE_DUMP_DIR").map(|d| {
            let dir = std::path::PathBuf::from(d);
            let _ = std::fs::create_dir_all(&dir);
            DumpState {
                dir,
                hidden: vec![0.0; num_logical * h],
                routes: vec![0; num_logical * 8],
            }
        });

        Ok(Self {
            backend,
            cfg,
            dense,
            tide,
            global_max_seq,
            layer_states,
            scratch,
            pf: Some(pf),
            embed_pinned,
            route_log_inflight: false,
            tide_step_pending: false,
            route_log_is_prefill: false,
            layer_planned: vec![false; num_logical],
            layer_deferred: vec![false; num_logical],
            dump,
            dump_token: (0, 0),
            rng_state: 0x9E37_79B9_7F4A_7C15,
            tokens_decoded: 0,
            profile_enabled: false,
            profile: StepProfile::default(),
        })
    }

    /// Total KV cache + short-conv VRAM allocated across all 48 logical layers.
    pub fn kv_vram_bytes(&self) -> usize {
        self.layer_states
            .iter()
            .map(|s| {
                s.k_cache.size_bytes
                    + s.k_scales.size_bytes
                    + s.v_cache.size_bytes
                    + s.v_scales.size_bytes
                    + s.sconv_k.size_bytes
                    + s.sconv_v.size_bytes
                    + s.sconv_attn.size_bytes
                    + s.sconv_mlp.size_bytes
            })
            .sum()
    }

    /// Device regions that define the rewindable sequence state: every local (sliding) layer's
    /// FP8 KV ring + scales and every layer's short-conv histories. Global KV is append-only
    /// (positions >= pos are simply overwritten later), so it is not part of a snapshot.
    fn snapshot_regions(&self) -> Vec<DeviceBuf> {
        let mut v = Vec::new();
        for st in &self.layer_states {
            if st._is_local {
                v.extend([st.k_cache, st.k_scales, st.v_cache, st.v_scales]);
            }
            v.extend([st.sconv_k, st.sconv_v, st.sconv_attn, st.sconv_mlp]);
        }
        v
    }

    /// Bytes one `SeqSnapshot` occupies in pinned host RAM.
    pub fn snapshot_bytes(&self) -> usize {
        self.snapshot_regions().iter().map(|b| b.size_bytes).sum()
    }

    /// Captures the rewindable state after `pos` tokens into `snap` (allocating its pinned buffer
    /// on first use). Stream-ordered on the compute stream: no CPU stall.
    pub fn snapshot_state(&mut self, pos: usize, snap: &mut Option<SeqSnapshot>) -> Result<()> {
        let s = self.tide.compute_stream;
        let total = self.snapshot_bytes();
        if snap.as_ref().map_or(true, |x| x.host.size_bytes < total) {
            if let Some(old) = snap.take() {
                self.backend.sync_stream(s)?;
                self.backend.free_pinned(old.host)?;
            }
            *snap = Some(SeqSnapshot { pos, host: self.backend.alloc_pinned(total)? });
        }
        let sn = snap.as_mut().unwrap();
        sn.pos = pos;
        let mut off = 0usize;
        for b in self.snapshot_regions() {
            self.backend.memcpy_d2h_async(sn.host, off, b, 0, b.size_bytes, s)?;
            off += b.size_bytes;
        }
        Ok(())
    }

    /// Restores a snapshot taken by `snapshot_state`; the next token to process is `snap.pos`.
    /// Valid only if the global KV for positions `< snap.pos` still holds the same tokens.
    pub fn restore_state(&mut self, snap: &SeqSnapshot) -> Result<()> {
        let s = self.tide.compute_stream;
        let mut off = 0usize;
        for b in self.snapshot_regions() {
            self.backend.memcpy_h2d_async(b, 0, snap.host, off, b.size_bytes, s)?;
            off += b.size_bytes;
        }
        Ok(())
    }

    /// Frees a snapshot's pinned buffer.
    pub fn free_snapshot(&mut self, snap: SeqSnapshot) -> Result<()> {
        self.backend.sync_stream(self.tide.compute_stream)?;
        self.backend.free_pinned(snap.host)
    }

    /// Resets all 48 logical layers' short-conv history for a fresh sequence (pos = 0).
    pub fn reset_sequence(&mut self) -> Result<()> {
        let s = self.tide.compute_stream;
        let h = self.cfg.hidden_size;
        let kv_dim = self.cfg.num_key_value_heads * self.cfg.head_dim;
        for st in &self.layer_states {
            self.backend.zero_f32(st.sconv_k, 3 * kv_dim, s)?;
            self.backend.zero_f32(st.sconv_v, 3 * kv_dim, s)?;
            self.backend.zero_f32(st.sconv_attn, 3 * h, s)?;
            self.backend.zero_f32(st.sconv_mlp, 3 * h, s)?;
        }
        self.backend.sync_stream(s)?;
        Ok(())
    }

    /// Dev benchmark: wall time of the decode MoE kernels (w13+w2) reading 8 experts at the given
    /// pool `slots` (NVFP4 pool unless `mxfp8`), optionally after evicting the GPU caches.
    pub fn debug_bench_moe(&mut self, slots: &[u32], mxfp8: bool, iters: usize, flush: bool) -> Result<f64> {
        let s = self.tide.compute_stream;
        let b = self.backend.clone();
        let h = self.cfg.hidden_size;
        let n = self.cfg.n_routed_experts;
        let table = b.alloc_device(n * 4)?;
        let ids = b.alloc_device(8 * 4)?;
        let host = b.alloc_pinned(n * 4 + 64)?;
        unsafe {
            let hb = b.pinned_as_mut_slice(host)?;
            let words = std::slice::from_raw_parts_mut(hb.as_mut_ptr() as *mut u32, n + 16);
            for w in words.iter_mut() { *w = u32::MAX; }
            for (i, &sl) in slots.iter().enumerate() { words[i] = sl; }
            for i in 0..8 { words[n + i] = i as u32; }
        }
        b.memcpy_h2d_async(table, 0, host, 0, n * 4, s)?;
        b.memcpy_h2d_async(ids, 0, host, n * 4, 32, s)?;
        let flush_buf = b.alloc_device(256 << 20)?;
        let (pool, stride) = if mxfp8 {
            (self.tide.vram_pool_mxfp8, self.tide.mxfp8_layout.aligned_slab_bytes)
        } else {
            (self.tide.vram_pool_nvfp4, self.tide.nvfp4_layout.aligned_slab_bytes)
        };
        let params = MoeDispatchParams {
            batch_size: 1,
            hidden_size: h,
            intermediate_size: self.cfg.intermediate_size,
            top_k: 8,
            weight_dtype: if mxfp8 { DType::MXFP8 } else { DType::NVFP4 },
            expert_stride_bytes: stride,
        };
        b.sync_stream(s)?;
        let mut total = 0.0;
        for _ in 0..iters {
            if flush {
                b.vec_copy(flush_buf.slice(128 << 20, 128 << 20), flush_buf.slice(0, 128 << 20), 32 << 20, s)?;
            }
            b.sync_stream(s)?;
            let t0 = Instant::now();
            // Back-to-back launches so the result is in-stream kernel time, not sync latency.
            let reps = if flush { 1 } else { 20 };
            for _ in 0..reps {
                b.moe_decode_experts(self.scratch.moe_mid, self.scratch.moe_tmp, self.scratch.moe_norm, pool, table, ids, params, s)?;
            }
            b.sync_stream(s)?;
            total += t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        }
        b.free_device(flush_buf)?;
        b.free_device(table)?;
        b.free_device(ids)?;
        b.free_pinned(host)?;
        Ok(total / iters as f64)
    }

    /// Mode 2: Prefill -> Decode Boundary Tide Rebalance.
    pub fn rebalance_after_prefill(&mut self, max_swaps: usize) -> Result<usize> {
        let ops = self.tide.scheduler.plan_prefill_decode_rebalance(max_swaps);
        let n = ops.len();
        if n > 0 {
            self.tide.execute_transfers(&ops, true, false)?;
        }
        Ok(n)
    }

    /// Mode 3 + Periodic Rebalance: Called once after each decoded token.
    ///
    /// Only the cheap part runs here (routing-log readback while the stream is idle after
    /// sampling); the CPU-heavy part (staging memcpys for async promotions, rebalance) is
    /// deferred into the next `forward_token`, after the prelude layers are enqueued, so it
    /// overlaps GPU work instead of leaving the GPU idle between tokens.
    pub fn post_decode_tide_step(&mut self) -> Result<()> {
        self.consume_route_log()?;
        self.tokens_decoded += 1;
        self.tide_step_pending = true;
        Ok(())
    }

    /// Runs a deferred `post_decode_tide_step` (no-op if none is pending).
    pub fn run_deferred_tide_step(&mut self) -> Result<()> {
        if !self.tide_step_pending {
            return Ok(());
        }
        self.tide_step_pending = false;
        let _ = self.tide.poll_stage2_completions()?;
        let async_ops = self.tide.scheduler.plan_async_step();
        if !async_ops.is_empty() {
            // Stage 2 Disk->RAM runs asynchronously on dedicated OS thread; Stage 1 RAM->VRAM on copy stream
            self.tide.execute_transfers(&async_ops, false, true)?;
        }
        let reb_ops = self
            .tide
            .scheduler
            .maybe_periodic_rebalance(self.tokens_decoded, 64, 16, false);
        if !reb_ops.is_empty() {
            self.tide.execute_transfers(&reb_ops, false, false)?;
        }
        Ok(())
    }

    /// Folds the last token's router log (fully-resident layers, which skip the per-layer
    /// readback) into the cache's heat / prefill scores, so eviction and priors see every layer.
    fn consume_route_log(&mut self) -> Result<()> {
        if !self.route_log_inflight {
            return Ok(());
        }
        self.route_log_inflight = false;
        let num_logical = self.layer_planned.len();
        // Synchronous readback: called right after sampling, when the compute stream is idle.
        // (An async D2H + event here completed tens of ms late on Windows.)
        let mut bytes = vec![0u8; num_logical * ROUTE_WORDS * 4];
        self.backend
            .memcpy_d2h_sync(&mut bytes, self.scratch.route_log, 0, self.tide.compute_stream)?;
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for il in 0..num_logical {
            if self.layer_planned[il] {
                continue;
            }
            let (phys, _) = crate::map_logical_to_physical(
                &self.cfg,
                smalltide_core::LogicalLayerId(il as u16),
            );
            let w = &words[il * ROUTE_WORDS..(il + 1) * ROUTE_WORDS];
            let mut ids = [0u16; 8];
            let mut ws = [0f32; 8];
            let mut tot = 0.0f32;
            for k in 0..8 {
                ids[k] = w[k] as u16;
                ws[k] = f32::from_bits(w[8 + k]);
                tot += ws[k];
            }
            if self.layer_deferred[il] {
                // Same bookkeeping as the synchronous path (rescue is disabled, so no transfers).
                let _ = self.tide.scheduler.plan_layer_dispatch(phys, &ids, &ws);
                continue;
            }
            if self.route_log_is_prefill {
                self.tide.scheduler.record_prefill_routing(phys, &ids, &ws);
            } else {
                let inv = if tot > 1e-8 { 1.0 / tot } else { 0.0 };
                for k in 0..8 {
                    self.tide.scheduler.cache.touch_expert(
                        smalltide_core::ExpertId { layer: phys, expert: ids[k] },
                        ws[k] * inv,
                    );
                }
            }
        }
        Ok(())
    }

    /// Dev: drop the pending routing-log readback (used before graph capture experiments).
    pub fn clear_route_log_flag(&mut self) {
        self.route_log_inflight = false;
    }

    pub fn telemetry(&self) -> &TideTelemetry {
        &self.tide.scheduler.telemetry
    }

    #[inline]
    pub(crate) fn fp8_mm(&self, out: DeviceBuf, x: DeviceBuf, m: &Fp8MatrixGpu, s: Stream) -> Result<()> {
        self.backend
            .gemv_fp8(out, x, m.data, m.row_scales, m.n_rows, m.k_cols, s)
    }

    /// Exact rank-64 cross-pass controller (`inkling2.cpp:241-249`).
    fn seg(&self, m: &Fp8MatrixGpu, x: DeviceBuf, y: DeviceBuf) -> GemvSegment {
        GemvSegment { w: m.data, row_scales: Some(m.row_scales), x, y, n_rows: m.n_rows, k_cols: m.k_cols }
    }

    /// Rank-64 cross-pass controllers (`inkling2.cpp:241-249`) for up to 4 inputs at once:
    /// `(out, cur, msg, controller)`. Down / gain+forget projections are batched launches.
    fn run_controls(&self, items: &[(DeviceBuf, DeviceBuf, DeviceBuf, &LoopControllerGpu)], s: Stream) -> Result<()> {
        let h = self.cfg.hidden_size;
        let r = self.cfg.loop_gate_rank;
        let sl = |buf: DeviceBuf, i: usize, n: usize| buf.slice(i * n * 4, n * 4);
        for (i, (_, cur, msg, _)) in items.iter().enumerate() {
            self.backend.loop_control_prep_rows(sl(self.scratch.ctrl_prep, i, 2 * h), *cur, *msg, h, self.cfg.rms_norm_eps, 1, s)?;
        }
        let downs: Vec<GemvSegment> = items
            .iter()
            .enumerate()
            .map(|(i, (_, _, _, c))| self.seg(&c.down, sl(self.scratch.ctrl_prep, i, 2 * h), sl(self.scratch.ctrl_gate, i, r)))
            .collect();
        self.backend.gemv_multi(&downs, s)?;
        let mut gf = Vec::with_capacity(2 * items.len());
        for (i, (_, _, _, c)) in items.iter().enumerate() {
            gf.push(self.seg(&c.gain, sl(self.scratch.ctrl_gate, i, r), sl(self.scratch.ctrl_gain, i, h)));
            gf.push(self.seg(&c.forget, sl(self.scratch.ctrl_gate, i, r), sl(self.scratch.ctrl_forget, i, h)));
        }
        self.backend.gemv_multi(&gf, s)?;
        for (i, (out, cur, msg, _)) in items.iter().enumerate() {
            self.backend.loop_control_apply(*out, *cur, *msg, sl(self.scratch.ctrl_gain, i, h), sl(self.scratch.ctrl_forget, i, h), h, s)?;
        }
        Ok(())
    }

    #[inline]
    fn prof_mark(&self, s: Stream) -> Option<Instant> {
        if self.profile_enabled {
            let _ = self.backend.sync_stream(s);
            Some(Instant::now())
        } else {
            None
        }
    }

    #[inline]
    fn prof_end(&self, t: Option<Instant>, s: Stream) -> f64 {
        if let Some(t0) = t {
            let _ = self.backend.sync_stream(s);
            t0.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        }
    }

    /// Executes 1 logical layer (`il in 0..48`, physical `ip in 0..24`).
    fn forward_layer(
        &mut self,
        il: usize,
        ip: usize,
        pos: usize,
        is_prefill: bool,
        num_blocks: &mut usize,
        messages: Option<DeviceBuf>,
    ) -> Result<()> {
        let s = self.tide.compute_stream;
        let h = self.cfg.hidden_size;
        let h_bytes = h * 4;
        let q_dim = self.cfg.num_attention_heads * self.cfg.head_dim;
        let eps = self.cfg.rms_norm_eps;

        self.backend.seg_mark("pre_attn(mix,ctrl,norm)", s);
        let t_rc = self.prof_mark(s);
        let layer: &DenseLayerGpu = &self.dense.layers[ip];
        let is_local = layer.is_local;
        let has_msg = messages.is_some();

        // Save inpSA = cur
        self.backend.vec_copy(self.scratch.inp_sa, self.scratch.cur, h, s)?;

        // 1. Attention Residual Mix -> inputs[0..3]
        let (in0, in1, in2) = if let Some(attn_q) = layer.attn_res_query {
            let blocks_view = Some(self.scratch.blocks.slice(0, (*num_blocks) * h_bytes));
            if is_local {
                let dst0 = self.scratch.in_qkv.slice(0, h_bytes);
                self.backend.attn_res_mix_rows(
                    dst0,
                    blocks_view,
                    self.scratch.cur,
                    attn_q,
                    *num_blocks,
                    1,
                    h,
                    eps,
                    1,
                    s,
                )?;
                (dst0, dst0, dst0)
            } else {
                self.backend.attn_res_mix_rows(
                    self.scratch.in_qkv,
                    blocks_view,
                    self.scratch.cur,
                    attn_q,
                    *num_blocks,
                    3,
                    h,
                    eps,
                    1,
                    s,
                )?;
                (
                    self.scratch.in_qkv.slice(0, h_bytes),
                    self.scratch.in_qkv.slice(h_bytes, h_bytes),
                    self.scratch.in_qkv.slice(2 * h_bytes, h_bytes),
                )
            }
        } else {
            (self.scratch.cur, self.scratch.cur, self.scratch.cur)
        };

        // 2. Cross-Pass Loop Controllers for Q, K, V (pass > 0)
        let (in0, in1, in2) = if let Some(msgs) = messages {
            let ctrls = layer
                .controllers
                .as_ref()
                .expect("loop core layer must have controllers");
            let c0 = self.scratch.ctrl_qkv.slice(0, h_bytes);
            let c1 = self.scratch.ctrl_qkv.slice(h_bytes, h_bytes);
            let c2 = self.scratch.ctrl_qkv.slice(2 * h_bytes, h_bytes);
            self.run_controls(
                &[
                    (c0, in0, msgs.slice(0, h_bytes), &ctrls[0]),
                    (c1, in1, msgs.slice(h_bytes, h_bytes), &ctrls[1]),
                    (c2, in2, msgs.slice(2 * h_bytes, h_bytes), &ctrls[2]),
                ],
                s,
            )?;
            (c0, c1, c2)
        } else {
            (in0, in1, in2)
        };

        // 3. Attention Input Norms (`inkling2.cpp:313-324`)
        let (nq, nk, nv) = if is_local {
            let norm_w = layer.attn_norm.expect("local layer attn_norm");
            if has_msg {
                let n0 = self.scratch.norm_qkv.slice(0, h_bytes);
                let n1 = self.scratch.norm_qkv.slice(h_bytes, h_bytes);
                let n2 = self.scratch.norm_qkv.slice(2 * h_bytes, h_bytes);
                self.backend.rms_norm(n0, in0, norm_w, 1, h, eps, s)?;
                self.backend.rms_norm(n1, in1, norm_w, 1, h, eps, s)?;
                self.backend.rms_norm(n2, in2, norm_w, 1, h, eps, s)?;
                (n0, n1, n2)
            } else {
                let n0 = self.scratch.norm_qkv.slice(0, h_bytes);
                self.backend.rms_norm(n0, in0, norm_w, 1, h, eps, s)?;
                (n0, n0, n0)
            }
        } else {
            let n0 = self.scratch.norm_qkv.slice(0, h_bytes);
            let n1 = self.scratch.norm_qkv.slice(h_bytes, h_bytes);
            let n2 = self.scratch.norm_qkv.slice(2 * h_bytes, h_bytes);
            self.backend
                .rms_norm(n0, in0, layer.q_input_norm.unwrap(), 1, h, eps, s)?;
            self.backend
                .rms_norm(n1, in1, layer.k_input_norm.unwrap(), 1, h, eps, s)?;
            self.backend
                .rms_norm(n2, in2, layer.v_input_norm.unwrap(), 1, h, eps, s)?;
            (n0, n1, n2)
        };
        let dt_rc = self.prof_end(t_rc, s);
        if self.profile_enabled { self.profile.res_and_ctrl_ms += dt_rc; }

        // 4. Q, K, V, R Projections + Causal Short Convolutions on K & V
        self.backend.seg_mark("qkvr_proj", s);
        let t_ap = self.prof_mark(s);
        let layer = &self.dense.layers[ip];
        // One launch for every projection of the normed inputs: Q, K, V, R, the output gate
        // (FP8) and the attention sinks / dynamic conv taps (F32).
        let nh = self.cfg.num_attention_heads;
        let taps = self.cfg.num_key_value_heads * self.cfg.sconv_kernel_size;
        let mut segs = vec![
            self.seg(&layer.wq, nq, self.scratch.q),
            self.seg(&layer.wk, nk, self.scratch.k_raw),
            self.seg(&layer.wv, nv, self.scratch.v_raw),
            self.seg(&layer.wr, nq, self.scratch.r),
            self.seg(&layer.gate_attn, nq, self.scratch.gate_out),
            GemvSegment { w: layer.attn_sink, row_scales: None, x: nq, y: self.scratch.sinks, n_rows: nh, k_cols: h },
        ];
        if let Some(w) = layer.k_sconv_dyn_proj {
            segs.push(GemvSegment { w, row_scales: None, x: nk, y: self.scratch.k_dyn, n_rows: taps, k_cols: h });
        }
        if let Some(w) = layer.v_sconv_dyn_proj {
            segs.push(GemvSegment { w, row_scales: None, x: nv, y: self.scratch.v_dyn, n_rows: taps, k_cols: h });
        }
        self.backend.gemv_multi(&segs, s)?;
        let dt_ap = self.prof_end(t_ap, s);
        if self.profile_enabled { self.profile.attn_proj_ms += dt_ap; }

        self.backend.seg_mark("sconv,clip,sink,rel,kv", s);
        let t_sc = self.prof_mark(s);
        let layer = &self.dense.layers[ip];
        let st = &self.layer_states[il];
        // 5./6. One fused launch: K/V short convs, K clip, V norm, FP8 KV store, Q/R clip + tau,
        // relative-bias table.
        let tau = if is_local {
            1.0
        } else {
            let ratio = ((pos as f32 + 1.0) / (self.cfg.log_scaling_n_floor as f32)).max(1.0);
            1.0 + self.cfg.log_scaling_alpha * ratio.ln()
        };
        let slot_pos = if is_local { pos % st.max_seq } else { pos };
        self.backend.attn_prep_decode(
            AttnPrepParams {
                q: self.scratch.q,
                r: self.scratch.r,
                k_raw: self.scratch.k_raw,
                v_raw: self.scratch.v_raw,
                k_dyn: layer.k_sconv_dyn_proj.map(|_| self.scratch.k_dyn),
                v_dyn: layer.v_sconv_dyn_proj.map(|_| self.scratch.v_dyn),
                sconv_k: st.sconv_k,
                sconv_v: st.sconv_v,
                k_w: layer.k_sconv_static,
                v_w: layer.v_sconv_static,
                q_scale: layer.q_l2_clip_scale,
                r_scale: layer.rel_l2_clip_scale,
                rel_proj: layer.rel_logits_proj,
                rel_table: self.scratch.rel_table,
                k_cache: st.k_cache,
                k_scales: st.k_scales,
                v_cache: st.v_cache,
                v_scales: st.v_scales,
                n_q_heads: self.cfg.num_attention_heads,
                n_kv_heads: self.cfg.num_key_value_heads,
                max_seq: st.max_seq,
                slot: slot_pos,
                rel_extent: self.cfg.rel_extent,
                qk_threshold: self.cfg.qk_l2_clip_threshold,
                rel_threshold: self.cfg.rel_l2_clip_threshold,
                tau,
                eps,
            },
            s,
        )?;
        let dt_sc = self.prof_end(t_sc, s);
        if self.profile_enabled { self.profile.sconv_clip_rel_ms += dt_sc; }

        self.backend.seg_mark("attention", s);
        let t_fa = self.prof_mark(s);
        let st = &self.layer_states[il];
        let n_valid = if is_local {
            (pos + 1).min(st.window)
        } else {
            pos + 1
        };
        let num_splits = n_valid.div_ceil(128).clamp(1, MAX_FLASH_SPLITS);
        self.backend.flash_decode_fp8(
            self.scratch.attn_out,
            self.scratch.partial_out,
            self.scratch.partial_m,
            self.scratch.partial_l,
            self.scratch.q,
            st.k_cache,
            st.k_scales,
            st.v_cache,
            st.v_scales,
            Some(self.scratch.rel_table),
            Some(self.scratch.sinks),
            FlashDecodeParams {
                n_q_heads: self.cfg.num_attention_heads,
                n_kv_heads: self.cfg.num_key_value_heads,
                max_seq: st.max_seq,
                q_pos: pos,
                rel_extent: self.cfg.rel_extent,
                is_local,
                num_splits,
                inv_sqrt_d: 1.0 / (self.cfg.head_dim as f32).sqrt(),
                window: if is_local { st.window } else { 0 },
            },
            s,
        )?;
        let dt_fa = self.prof_end(t_fa, s);
        if self.profile_enabled { self.profile.flash_attn_ms += dt_fa; }

        // 7. Output Gate + Wo + Post-Attention Short Conv
        self.backend.seg_mark("gate,wo,sconv,mlp_mix,norm", s);
        let t_ao = self.prof_mark(s);
        let layer = &self.dense.layers[ip];
        let st = &self.layer_states[il];
        self.backend.gate_attn_sigmoid_mul(
            self.scratch.attn_out,
            self.scratch.attn_out,
            self.scratch.gate_out,
            q_dim,
            s,
        )?;
        self.fp8_mm(self.scratch.o_raw, self.scratch.attn_out, &layer.wo, s)?;
        self.backend.sconv_k4_decode(
            self.scratch.sconv_tmp,
            st.sconv_attn,
            self.scratch.o_raw,
            layer.attn_sconv,
            None,
            h,
            0,
            s,
        )?;

        // 8. Block Residual Boundary (`inkling2.cpp:381-388`)
        if ip % self.cfg.n_layers_per_attn_res_block == 0 {
            let dst_blk = self.scratch.blocks.slice((*num_blocks) * h_bytes, h_bytes);
            self.backend.vec_copy(dst_blk, self.scratch.inp_sa, h, s)?;
            *num_blocks += 1;
            self.backend.vec_copy(self.scratch.cur, self.scratch.sconv_tmp, h, s)?;
        } else {
            self.backend
                .vec_add(self.scratch.cur, self.scratch.inp_sa, self.scratch.sconv_tmp, h, s)?;
        }
        self.backend.vec_copy(self.scratch.ffn_inp, self.scratch.cur, h, s)?;

        // 9. MoE Block: Mix -> Optional Loop Controller -> RMSNorm -> Router -> Tide -> Indirect MoE -> Short Conv
        let blocks_view = Some(self.scratch.blocks.slice(0, (*num_blocks) * h_bytes));
        self.backend.attn_res_mix_rows(
            self.scratch.moe_in,
            blocks_view,
            self.scratch.cur,
            layer.mlp_res_query,
            *num_blocks,
            1,
            h,
            eps,
            1,
            s,
        )?;
        let moe_in_final = if let Some(msgs) = messages {
            let ctrls = layer.controllers.as_ref().unwrap();
            self.run_controls(&[(self.scratch.moe_ctrl, self.scratch.moe_in, msgs.slice(3 * h_bytes, h_bytes), &ctrls[3])], s)?;
            self.scratch.moe_ctrl
        } else {
            self.scratch.moe_in
        };
        self.backend
            .rms_norm(self.scratch.moe_norm, moe_in_final, layer.mlp_norm, 1, h, eps, s)?;
        let dt_ao = self.prof_end(t_ao, s);
        if self.profile_enabled { self.profile.attn_out_ms += dt_ao; }

        self.backend.seg_mark("router", s);
        let t_rt = self.prof_mark(s);
        let layer = &self.dense.layers[ip];
        let topk_ids = self.scratch.route_log.slice(il * ROUTE_WORDS * 4, 32);
        let topk_weights = self.scratch.route_log.slice(il * ROUTE_WORDS * 4 + 32, 32);
        let route_row = self.scratch.route_log.slice(il * ROUTE_WORDS * 4, ROUTE_WORDS * 4);
        // Parallel router: multi-workgroup logits GEMV + block-parallel top-8 (the old single-
        // workgroup kernel cost ~430 us per layer, ~20 ms per token).
        self.backend.gemm_f32_rows(
            self.scratch.router_logits,
            self.scratch.moe_norm,
            layer.router_weight,
            1,
            self.cfg.n_routed_experts,
            h,
            s,
        )?;
        self.backend.router_topk8_rows(
            route_row,
            self.scratch.router_logits,
            Some(layer.router_bias),
            self.cfg.route_scale,
            1,
            ROUTE_WORDS,
            s,
        )?;

        // TideLM Adaptive Expert Dispatch:
        // Zero-sync fast path when all 128 experts of this physical layer are already VRAM-resident!
        let phys_id = PhysicalLayerId(ip as u16);
        let fully_resident = self.tide.scheduler.is_layer_fully_resident_vram(phys_id);
        let mut rescue_ms_local = 0.0f64;
        let need_dump_routes = self.dump.is_some();
        // Rescue disabled: nothing would be copied in, so a partial layer needs no host readback
        // either - the kernels skip unpublished experts and the bookkeeping is deferred.
        let defer = !fully_resident
            && !is_prefill
            && !need_dump_routes
            && matches!(self.tide.scheduler.tide_cfg.rescue_policy, smalltide_core::RescuePolicy::Disabled);
        if defer {
            self.layer_deferred[il] = true;
        } else if fully_resident && !need_dump_routes {
            self.tide
                .scheduler
                .record_full_layer_hit(phys_id, self.cfg.num_experts_per_tok as u64);
        } else if fully_resident {
            self.tide
                .scheduler
                .record_full_layer_hit(phys_id, self.cfg.num_experts_per_tok as u64);
            let mut rb = [0u8; 64];
            self.backend.memcpy_d2h_sync(&mut rb, route_row, 0, s)?;
            if let Some(d) = self.dump.as_mut() {
                for k in 0..8 {
                    d.routes[il * 8 + k] =
                        u32::from_le_bytes(rb[k * 4..(k + 1) * 4].try_into().unwrap());
                }
            }
        } else {
            // One combined 64 B readback (ids + weights are adjacent in the routing log).
            let mut rb = [0u8; 64];
            self.backend.memcpy_d2h_sync(&mut rb, route_row, 0, s)?;
            let mut topk_ids_u16 = [0u16; 8];
            let mut topk_w_f32 = [0.0f32; 8];
            for k in 0..8 {
                topk_ids_u16[k] =
                    u32::from_le_bytes(rb[k * 4..(k + 1) * 4].try_into().unwrap()) as u16;
                topk_w_f32[k] =
                    f32::from_le_bytes(rb[32 + k * 4..32 + (k + 1) * 4].try_into().unwrap());
            }
            if let Some(d) = self.dump.as_mut() {
                for k in 0..8 {
                    d.routes[il * 8 + k] = topk_ids_u16[k] as u32;
                }
            }
            self.layer_planned[il] = true;
            if is_prefill {
                self.tide
                    .scheduler
                    .record_prefill_routing(phys_id, &topk_ids_u16, &topk_w_f32);
            }
            let plan = self
                .tide
                .scheduler
                .plan_layer_dispatch(phys_id, &topk_ids_u16, &topk_w_f32);
            if !plan.sync_transfers.is_empty() {
                let tr0 = Instant::now();
                // Sync path: compute waits for the copies and publishes everything pending.
                self.tide
                    .execute_transfers(&plan.sync_transfers, true, false)?;
                if self.profile_enabled {
                    let _ = self.backend.sync_stream(self.tide.copy_stream);
                    rescue_ms_local = tr0.elapsed().as_secs_f64() * 1000.0;
                }
            } else if plan.wait_pending {
                // A top-k expert's async DMA is still in flight: wait on the GPU, then publish.
                self.tide.wait_and_publish_all_pending()?;
            }
            if let Some(d) = self.dump.as_mut() {
                // Bit 31 marks experts the MoE kernel will skip (unpublished in the GPU table).
                for k in 0..8 {
                    let id = smalltide_core::ExpertId { layer: phys_id, expert: topk_ids_u16[k] };
                    if !self.tide.scheduler.cache.is_vram_hit(id) {
                        d.routes[il * 8 + k] |= 0x8000_0000;
                    }
                }
            }
        }
        let dt_rt = self.prof_end(t_rt, s);
        if self.profile_enabled {
            self.profile.router_d2h_ms += (dt_rt - rescue_ms_local).max(0.0);
            self.profile.tide_rescue_ms += rescue_ms_local;
        }

        self.backend.seg_mark(if self.cfg.is_loop_core_layer(ip) { "moe(loop core)" } else { "moe(prelude/coda)" }, s);
        let t_moe = self.prof_mark(s);
        let is_mxfp8 = self.cfg.is_mxfp8_layer(ip);
        let (pool, stride, dtype) = if is_mxfp8 {
            (
                self.tide.vram_pool_mxfp8,
                self.tide.mxfp8_layout.aligned_slab_bytes,
                DType::MXFP8,
            )
        } else {
            (
                self.tide.vram_pool_nvfp4,
                self.tide.nvfp4_layout.aligned_slab_bytes,
                DType::NVFP4,
            )
        };
        let layer_slot_table = self.tide.gpu_slot_table.slice(
            ip * self.cfg.n_routed_experts * 4,
            self.cfg.n_routed_experts * 4,
        );
        let params = MoeDispatchParams {
            batch_size: 1,
            hidden_size: h,
            intermediate_size: self.cfg.intermediate_size,
            top_k: self.cfg.num_experts_per_tok,
            weight_dtype: dtype,
            expert_stride_bytes: stride,
        };
        self.backend.moe_decode_experts(
            self.scratch.moe_mid,
            self.scratch.moe_tmp,
            self.scratch.moe_norm,
            pool,
            layer_slot_table,
            topk_ids,
            params,
            s,
        )?;
        // Weighted/renormalized expert sum + post-MLP short conv + residual in one kernel.
        let st = &self.layer_states[il];
        self.backend.moe_decode_post(
            self.scratch.cur,
            self.scratch.moe_tmp,
            layer_slot_table,
            topk_ids,
            topk_weights,
            self.cfg.num_experts_per_tok,
            Some((st.sconv_mlp, self.dense.layers[ip].mlp_sconv, self.scratch.ffn_inp)),
            h,
            s,
        )?;
        let dt_moe = self.prof_end(t_moe, s);
        if self.profile_enabled { self.profile.moe_ms += dt_moe; }
        if self.dump.is_some() {
            let mut hb = vec![0u8; h * 4];
            self.backend.memcpy_d2h_sync(&mut hb, self.scratch.cur, 0, s)?;
            let d = self.dump.as_mut().unwrap();
            for (i, c) in hb.chunks_exact(4).enumerate() {
                d.hidden[il * h + i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        Ok(())
    }

    /// Writes `tok{pos}.bin`: magic, pos, token, num_logical, hidden, has_logits, then
    /// `[num_logical][hidden]` f32 layer outputs, `[num_logical][8]` u32 routed ids and
    /// optionally `[unpadded_vocab]` f32 logits.
    pub(crate) fn write_dump(&mut self, has_logits: bool) -> Result<()> {
        let s = self.tide.compute_stream;
        let vocab = self.cfg.unpadded_vocab_size;
        let mut logits = Vec::new();
        if has_logits {
            logits = vec![0u8; vocab * 4];
            self.backend.memcpy_d2h_sync(&mut logits, self.scratch.logits, 0, s)?;
        }
        let (tok, pos) = self.dump_token;
        let d = self.dump.as_ref().unwrap();
        let n_log = self.layer_planned.len();
        let mut out = Vec::with_capacity(24 + d.hidden.len() * 4 + d.routes.len() * 4 + logits.len());
        for v in [0x5354_4450u32, pos as u32, tok, n_log as u32, self.cfg.hidden_size as u32, has_logits as u32] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in &d.hidden {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in &d.routes {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&logits);
        std::fs::write(d.dir.join(format!("tok{pos}.bin")), out)
            .map_err(|e| smalltide_core::SmallTideError::Backend { backend: "dump", message: e.to_string() })?;
        Ok(())
    }

    /// Runs a complete 48-logical-layer forward step for `token_id` at position `pos`.
    /// When `need_logits` is false (e.g. non-final prefill tokens), skips the 200K `lm_head` GEMV.
    pub fn forward_token(
        &mut self,
        token_id: u32,
        pos: usize,
        is_prefill: bool,
        need_logits: bool,
    ) -> Result<()> {
        let s = self.tide.compute_stream;
        let h = self.cfg.hidden_size;
        let h_bytes = h * 4;
        let eps = self.cfg.rms_norm_eps;

        // Token boundary: fold the previous token's routing log into heat, then publish any async
        // promotions whose DMA has landed (non-blocking query; table upload is stream-ordered).
        self.consume_route_log()?;
        let published = self.tide.poll_pending()?;
        self.tide.scheduler.telemetry.stage1_async_published += published as u64;
        self.layer_planned.iter_mut().for_each(|p| *p = false);
        self.layer_deferred.iter_mut().for_each(|p| *p = false);
        self.dump_token = (token_id, pos);

        self.backend.seg_mark("token_start(embed,tables)", s);
        // 0. Host RAM BF16 Embedding Lookup -> Pinned -> GPU -> Embed RMSNorm
        let tok = (token_id as usize).min(self.cfg.vocab_size - 1);
        let row_off = tok * h * 2;
        let row_bytes = &self.dense.embed_bf16_host[row_off..row_off + h * 2];
        unsafe {
            let dst = self.backend.pinned_as_mut_slice(self.embed_pinned)?;
            let dst_f32 = std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, h);
            for i in 0..h {
                let bits = u16::from_le_bytes([row_bytes[2 * i], row_bytes[2 * i + 1]]);
                dst_f32[i] = bf16_to_f32(bits);
            }
        }
        self.backend
            .memcpy_h2d_async(self.scratch.embed_tmp, 0, self.embed_pinned, 0, h_bytes, s)?;
        self.backend.rms_norm(
            self.scratch.cur,
            self.scratch.embed_tmp,
            self.dense.embed_norm,
            1,
            h,
            eps,
            s,
        )?;

        let mut num_blocks = 0usize;

        // 1. Prelude: physical layers 0..6 (logical 0..6)
        let start = self.cfg.loop_start_layer;
        let end = self.cfg.loop_end_exclusive;
        let core_len = end - start;
        for ip in 0..start {
            self.forward_layer(ip, ip, pos, is_prefill, &mut num_blocks, None)?;
        }

        // The previous token's deferred Tide work runs here, overlapping the GPU's prelude.
        self.run_deferred_tide_step()?;

        // Save prelude output & prelude block count for all 3 loop passes (`inkling2-loop.cpp:88-89`)
        self.backend
            .vec_copy(self.scratch.prelude_cur, self.scratch.cur, h, s)?;
        let prelude_num_blocks = num_blocks;

        // 2. Loop Core: 3 passes over physical layers 6..18 (logical 6..42)
        for pass in 0..self.cfg.loop_passes {
            let mut bank_count = 0usize;
            if pass > 0 {
                // `bank = blocks[prelude_num_blocks + 1 .. num_blocks]`, `bank_out = cur`
                let bank_start = prelude_num_blocks + 1;
                bank_count = num_blocks.saturating_sub(bank_start);
                if bank_count > 0 {
                    self.backend.vec_copy(
                        self.scratch.bank.slice(0, bank_count * h_bytes),
                        self.scratch.blocks.slice(bank_start * h_bytes, bank_count * h_bytes),
                        bank_count * h,
                        s,
                    )?;
                }
                self.backend
                    .vec_copy(self.scratch.bank_out, self.scratch.cur, h, s)?;
                // Reset cur & blocks to prelude state (prelude_blocks are still in blocks[0..prelude_num_blocks])
                self.backend
                    .vec_copy(self.scratch.cur, self.scratch.prelude_cur, h, s)?;
                num_blocks = prelude_num_blocks;
            }

            for ip in start..end {
                let il = start + pass * core_len + (ip - start);
                if pass == 0 {
                    self.forward_layer(il, ip, pos, is_prefill, &mut num_blocks, None)?;
                } else {
                    let msg_q = self.dense.layers[ip]
                        .loop_msg_queries
                        .expect("loop core layer must have loop_msg_queries");
                    let bank_view = if bank_count > 0 {
                        Some(self.scratch.bank.slice(0, bank_count * h_bytes))
                    } else {
                        None
                    };
                    self.backend.attn_res_mix_rows(
                        self.scratch.messages,
                        bank_view,
                        self.scratch.bank_out,
                        msg_q,
                        bank_count,
                        4,
                        h,
                        eps,
                        1,
                        s,
                    )?;
                    self.forward_layer(
                        il,
                        ip,
                        pos,
                        is_prefill,
                        &mut num_blocks,
                        Some(self.scratch.messages),
                    )?;
                }
            }
        }

        // 3. Coda: physical layers 18..24 (logical 42..48)
        let coda_logical_base = start + self.cfg.loop_passes * core_len;
        for ip in end..self.cfg.num_physical_layers {
            let il = coda_logical_base + (ip - end);
            self.forward_layer(il, ip, pos, is_prefill, &mut num_blocks, None)?;
        }

        // Async routing-log readback (consumed at the next token boundary).
        // Routing log is read back at the next token boundary (stream idle by then).
        self.route_log_inflight = true;
        self.route_log_is_prefill = is_prefill;

        if !need_logits {
            if self.dump.is_some() {
                self.write_dump(false)?;
            }
            return Ok(());
        }

        // 4. Output Head (`inkling2.cpp:421-445`)
        self.backend.seg_mark("lm_head", s);
        let t_lm = self.prof_mark(s);
        let blocks_view = Some(self.scratch.blocks.slice(0, num_blocks * h_bytes));
        self.backend.attn_res_mix_rows(
            self.scratch.moe_in,
            blocks_view,
            self.scratch.cur,
            self.dense.output_res_query,
            num_blocks,
            1,
            h,
            eps,
            1,
            s,
        )?;
        self.backend.rms_norm(
            self.scratch.moe_norm,
            self.scratch.moe_in,
            self.dense.final_norm,
            1,
            h,
            eps,
            s,
        )?;
        // Compute only the unpadded 200,058 vocab rows (skipping 966 padded rows!)
        self.backend.gemv_fp8(
            self.scratch.logits,
            self.scratch.moe_norm,
            self.dense.unembed_fp8.data,
            self.dense.unembed_fp8.row_scales,
            self.cfg.unpadded_vocab_size,
            h,
            s,
        )?;
        if self.cfg.final_logit_softcapping > 0.0 {
            self.backend.final_logit_softcap(
                self.scratch.logits,
                self.cfg.unpadded_vocab_size,
                self.cfg.final_logit_softcapping,
                s,
            )?;
        }
        let dt_lm = self.prof_end(t_lm, s);
        if self.profile_enabled {
            self.profile.lm_head_ms += dt_lm;
            self.profile.steps += 1;
        }
        if self.dump.is_some() {
            self.write_dump(true)?;
        }
        Ok(())
    }

    /// Samples the next token ID from the logits computed by `forward_token(..., need_logits = true)`.
    pub fn sample_next_token(
        &mut self,
        params: &SamplingParams,
        recent_tokens: &[u32],
    ) -> Result<u32> {
        let s = self.tide.compute_stream;
        let vocab = self.cfg.unpadded_vocab_size;

        self.backend.seg_mark("sample", s);
        // Fast GPU argmax path when greedy and no repetition penalty (~5 µs, 512 B D2H!)
        if params.temperature <= 0.0 && (params.repetition_penalty - 1.0).abs() < 1e-6 {
            self.backend.argmax_partial_f32(
                self.scratch.argmax_buf.slice(0, 256),
                self.scratch.argmax_buf.slice(256, 256),
                self.scratch.logits,
                vocab,
                s,
            )?;
            let mut bytes = [0u8; 128 * 4];
            self.backend
                .memcpy_d2h_sync(&mut bytes, self.scratch.argmax_buf, 0, s)?;
            let mut best_v = f32::NEG_INFINITY;
            let mut best_i = 0u32;
            for b in 0..64 {
                let v = f32::from_le_bytes(bytes[b * 4..(b + 1) * 4].try_into().unwrap());
                let idx = u32::from_le_bytes(bytes[256 + b * 4..256 + (b + 1) * 4].try_into().unwrap());
                if v > best_v {
                    best_v = v;
                    best_i = idx;
                }
            }
            return Ok(best_i);
        }

        // General host sampling path (temperature + top-k + top-p + repetition penalty)
        let mut logits = vec![0.0f32; vocab];
        let logits_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(logits.as_mut_ptr() as *mut u8, vocab * 4) };
        self.backend
            .memcpy_d2h_sync(logits_bytes, self.scratch.logits, 0, s)?;

        if (params.repetition_penalty - 1.0).abs() > 1e-6 {
            for &t in recent_tokens {
                if (t as usize) < vocab {
                    let v = &mut logits[t as usize];
                    if *v > 0.0 {
                        *v /= params.repetition_penalty;
                    } else {
                        *v *= params.repetition_penalty;
                    }
                }
            }
        }

        if params.temperature <= 0.0 {
            let mut best_i = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best_i = i;
                }
            }
            return Ok(best_i as u32);
        }

        let inv_t = 1.0 / params.temperature;
        let mut indexed: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v * inv_t))
            .collect();
        let k = params.top_k.clamp(1, vocab);
        indexed.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        indexed.truncate(k);
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

        let max_l = indexed[0].1;
        let mut sum = 0.0f32;
        for (_, v) in &mut indexed {
            *v = (*v - max_l).exp();
            sum += *v;
        }
        let inv_sum = 1.0 / sum.max(1e-12);
        let mut cum = 0.0f32;
        let mut cut = indexed.len();
        for (i, (_, p)) in indexed.iter_mut().enumerate() {
            *p *= inv_sum;
            cum += *p;
            if cum >= params.top_p {
                cut = i + 1;
                break;
            }
        }
        indexed.truncate(cut);
        let renorm: f32 = indexed.iter().map(|(_, p)| *p).sum();

        // xorshift64*
        self.rng_state ^= self.rng_state >> 12;
        self.rng_state ^= self.rng_state << 25;
        self.rng_state ^= self.rng_state >> 27;
        let u = ((self.rng_state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32)
            / ((1u32 << 24) as f32);
        let target = u * renorm;
        let mut acc = 0.0f32;
        for &(id, p) in &indexed {
            acc += p;
            if acc >= target {
                return Ok(id);
            }
        }
        Ok(indexed[0].0)
    }
}

impl<B: ComputeBackend> Drop for Inkling2Engine<B> {
    fn drop(&mut self) {
        let _ = self.backend.sync_stream(self.tide.compute_stream);
        if let Some(pf) = self.pf.take() {
            pf.free(&*self.backend);
        }
        let _ = self.backend.free_pinned(self.embed_pinned);
        for st in &self.layer_states {
            let _ = self.backend.free_device(st.k_cache);
            let _ = self.backend.free_device(st.k_scales);
            let _ = self.backend.free_device(st.v_cache);
            let _ = self.backend.free_device(st.v_scales);
            let _ = self.backend.free_device(st.sconv_k);
            let _ = self.backend.free_device(st.sconv_v);
            let _ = self.backend.free_device(st.sconv_attn);
            let _ = self.backend.free_device(st.sconv_mlp);
        }
        let s = &self.scratch;
        for buf in [
            s.cur, s.embed_tmp, s.inp_sa, s.ffn_inp, s.blocks, s.prelude_cur, s.bank, s.bank_out,
            s.messages, s.in_qkv, s.ctrl_qkv, s.norm_qkv, s.ctrl_prep, s.ctrl_gate, s.ctrl_gain,
            s.ctrl_forget, s.q, s.k_raw, s.k, s.v_raw, s.v, s.k_dyn, s.v_dyn, s.r, s.sinks,
            s.rel_table, s.partial_out, s.partial_m, s.partial_l, s.attn_out, s.gate_out, s.o_raw,
            s.sconv_tmp, s.moe_in, s.moe_ctrl, s.moe_norm, s.route_log, s.router_logits, s.moe_mid, s.moe_tmp,
            s.moe_out, s.logits, s.argmax_buf,
        ] {
            let _ = self.backend.free_device(buf);
        }
    }
}
