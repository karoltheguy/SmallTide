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
use smalltide_core::backend::{AttnPrepParams, FlashDecodeParams, FusedMixParams, GemvSegment, MoeDispatchParams};
use smalltide_core::{
    ComputeBackend, DType, DeviceBuf, Inkling2Config, PhysicalLayerId, PinnedBuf, Result, Stream,
    TideTelemetry,
};
use smalltide_io::quant::bf16_to_f32;
use smalltide_tide::runtime::TideExecutor;
use std::sync::Arc;
use std::time::Instant;

pub(crate) const MAX_FLASH_SPLITS: usize = 64;
/// Maximum tokens per batched prefill chunk. Local KV rings hold `window + PREFILL_CHUNK` slots so
/// a chunk's own keys never overwrite keys an earlier row of the same chunk still attends to.
pub const PREFILL_CHUNK: usize = 256;
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
    /// Overlapped rescue checks: host time waiting for the routing row, planning, and count.
    pub rescue_wait_ms: f64,
    pub rescue_plan_ms: f64,
    pub rescue_checks: u64,
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

/// Rewindable sequence state (local KV rings + short-conv histories, plus optional DSpark drafter
/// KV state) after `pos` tokens, held in pinned host RAM. Created and restored by
/// `Inkling2Engine::{snapshot,restore}_state`.
pub struct SeqSnapshot {
    pub pos: usize,
    pub drafter_valid: bool,
    host: PinnedBuf,
}

impl SeqSnapshot {
    /// Pinned host bytes this snapshot holds.
    pub fn host_bytes(&self) -> usize {
        self.host.size_bytes
    }
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
    /// Paged global layers: the lane's page table into the shared KV pool (`k_cache` etc. are
    /// then the pool's buffers); `None`: the caches are this lane's own.
    pub(crate) page_table: Option<DeviceBuf>,
}

/// The shared global-KV pool: per global logical layer (K, K scales, V, V scales).
#[derive(Clone)]
pub(crate) struct KvPoolBufs {
    pub(crate) layers: Vec<(DeviceBuf, DeviceBuf, DeviceBuf, DeviceBuf)>,
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

    pub(crate) partial_out: DeviceBuf, // [24 * MAX_FLASH_SPLITS * 128]
    pub(crate) partial_m: DeviceBuf,   // [24 * MAX_FLASH_SPLITS]
    pub(crate) partial_l: DeviceBuf,   // [24 * MAX_FLASH_SPLITS]
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
/// Coherent host-mapped routing mirror stride: up to 8 verify rows x 16 words + 1 ticket sentinel,
/// padded to 576 B (cache-line aligned).
const ROUTE_MIRROR_STRIDE_WORDS: usize = 144;
/// Cache-line stride (64 B = 16 u32s) per layer for `hipStreamWaitValue32` flags.
const WAIT_FLAG_STRIDE_WORDS: usize = 16;
/// Per-layer rescue table patch buffer stride: 1 count header + up to 255 `(idx, slot)` pairs (2 KiB).
const PATCH_MAX_PAIRS: usize = 255;
const PATCH_STRIDE_WORDS: usize = 512;

/// Optional per-token hidden-state dump for reference verification (`SMALLTIDE_DUMP_DIR`).
pub(crate) struct DumpState {
    pub(crate) dir: std::path::PathBuf,
    pub(crate) hidden: Vec<f32>, // [num_logical * H]
    pub(crate) routes: Vec<u32>, // [num_logical * 8]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RescueWaitMode {
    Off,
    WaitValue,
    Spin,
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
    /// Overlapped rescue check: pinned `[logical][16]` mirror of the routing log rows and the
    /// event marking a row's copy (allocated on first use).
    route_pinned: Option<(DeviceBuf, PinnedBuf, smalltide_core::Event)>,
    /// GPU-side rescue checks: `None` until initialized, then `Off` / `WaitValue` / `Spin`.
    /// The token's whole pass is queued with a GPU-side wait before each checked layer's experts;
    /// the host services the checks afterwards and releases each wait by writing its flag.
    rescue_wait: Option<RescueWaitMode>,
    /// Mapped `[logical]` cache-line-spaced u32 release flags (device view, host) and this token's ticket.
    wait_flags: Option<(DeviceBuf, PinnedBuf)>,
    /// Mapped `[logical][PATCH_STRIDE_WORDS]` slot-table patch buffers applied on `compute_stream`
    /// right after each GPU-side rescue wait, plus a `[total_experts]` fallback full-table mirror.
    rescue_patches: Option<(DeviceBuf, PinnedBuf)>,
    rescue_full_table: Option<(DeviceBuf, PinnedBuf)>,
    /// Mapped `[8]` u32 GPU breadcrumb buffer (`[0] = (ticket<<12)|(layer<<4)|stage`, `[1] = n_patches`).
    rescue_probe: Option<(DeviceBuf, PinnedBuf)>,
    wait_ticket: u32,
    pub(crate) pending_checks: Vec<usize>,
    /// True during a `verify` forward when inline GPU-side TideLM rescue is active.
    pub(crate) verify_rescue_active: bool,
    /// Experts rescued inline during the current `verify` pass (credited in `post_batch_tide_step`).
    verify_rescued: Vec<smalltide_core::ExpertId>,
    /// Expert map: the decoded tokens' routing (host copies of what the engine already read back),
    /// the last single-token decode's (lane, pos, token) and the last batched round's rows.
    pub routes: smalltide_tide::expert_map::RouteLog,
    /// Paged global KV (one fixed pool shared by the lanes); `None`: every lane owns its caches.
    kv_pool: Option<(crate::kv_pager::KvPool, KvPoolBufs)>,
    map_last_tok: Option<(usize, usize, u32)>,
    map_batch_rows: Vec<(usize, usize, u32)>,
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
    /// Prefill under `RescuePolicy::Threshold`: stage a missing expert only if its summed routing
    /// share over the chunk is at least this many tokens' worth (else skip + renormalize).
    pub prefill_min_share: f32,
    pub profile: StepProfile,
    /// Concurrent sequences ("lanes"): every lane owns its KV caches and short-conv histories.
    /// The active lane's state lives in `layer_states`; the others are parked here (the active
    /// lane's entry is `None`). See `select_lane`.
    lane_store: Vec<Option<ParkedLane>>,
    lane_max_seq: Vec<usize>,
    active_lane: usize,
    /// Batched decode: `[MAX_BATCH_ROWS][unpadded_vocab]` logits and `[logical][MAX_BATCH_ROWS][16]`
    /// routing log (allocated on first use).
    pub(crate) batch_logits: Option<DeviceBuf>,
    pub(crate) batch_route_log: Option<DeviceBuf>,
    /// Rows of the batched round whose routing log awaits `post_batch_tide_step`.
    batch_pending_rows: usize,
    /// The rows forward in progress is a batched decode round (GEMV-style kernels for <= 8 rows).
    pub(crate) rows_decode: bool,
    /// Lanes of the rows of the last batched round (Tide demand bookkeeping per lane).
    batch_lanes: Vec<usize>,
    /// Speculative verify: short-conv histories `[logical][k, v, attn, mlp][3 + MAX_BATCH_ROWS][dim]`.
    pub(crate) verify_hist: Option<DeviceBuf>,
    /// Per lane: device rewind table (hist, state, dim, 0) x 4 per logical layer.
    rewind_tables: Vec<Option<DeviceBuf>>,
    /// Drafter taps of the last rows forward: `[4][PREFILL_CHUNK][H]` (layer 4 / 22 / 40 outputs,
    /// final normed hidden), filled when `capture_taps`.
    pub(crate) taps_buf: Option<DeviceBuf>,
    pub capture_taps: bool,
    /// `[MAX_BATCH_ROWS * 33]` u32/f32 argmax output + parallel reduction scratch.
    argmax_rows_buf: Option<DeviceBuf>,
    /// Starting position of the most recent `verify` whose drafter tap ingest is deferred to `verify_rewind`.
    verify_ingest_pos0: Option<usize>,
    /// Prompt-aware loading: max async swaps between prefill chunks / before a request.
    pub prefill_chunk_swaps: usize,
    pub anticipate_swaps: usize,
    /// Wait for the anticipated swaps before prefill (the prompt itself then hits them).
    pub anticipate_wait: bool,
    /// DSpark drafter (speculative decoding); its context follows the active lane's prefill /
    /// verify rows.
    pub drafter: Option<crate::drafter::Drafter<B>>,
    /// Rows of the last tap-capturing rows forward (the last one is the newest position).
    pub last_tap_rows: usize,
    /// Dev stats: rows run by the last batched / verify forward, and the summed number of
    /// distinct resident experts those rounds touched (all layers).
    batch_rows_run: usize,
    pub dev_rounds: u64,
    pub dev_unique_resident: u64,
}

/// Most rows (concurrent sequences) one batched decode round can carry.
pub const MAX_BATCH_ROWS: usize = 8;

/// A lane's sequence state while another lane is active.
pub(crate) struct ParkedLane {
    layer_states: Vec<LogicalLayerState>,
    rng_state: u64,
}


/// Allocates the shared global-KV pool of `pages` pages.
fn alloc_kv_pool<B: ComputeBackend>(backend: &B, cfg: &Inkling2Config, pages: usize) -> Result<KvPoolBufs> {
    let (n, hd) = (cfg.num_key_value_heads, cfg.head_dim);
    let pos = pages * crate::kv_pager::PAGE_POS;
    let mut layers = Vec::new();
    for il in 0..cfg.num_logical_layers() {
        let (phys, _) = crate::map_logical_to_physical(cfg, smalltide_core::LogicalLayerId(il as u16));
        if !cfg.is_local_layer(phys.0 as usize) {
            layers.push((
                backend.alloc_device(pos * n * hd)?,
                backend.alloc_device(pos * n * 4)?,
                backend.alloc_device(pos * n * hd)?,
                backend.alloc_device(pos * n * 4)?,
            ));
        }
    }
    Ok(KvPoolBufs { layers })
}

/// One sequence's per-logical-layer KV caches (local rings + global KV of `global_max_seq`
/// positions) and short-conv histories.
pub(crate) fn alloc_layer_states<B: ComputeBackend>(
    backend: &B,
    cfg: &Inkling2Config,
    global_max_seq: usize,
    stream: smalltide_core::Stream,
    paged: Option<(&KvPoolBufs, DeviceBuf)>,
) -> Result<Vec<LogicalLayerState>> {
    let mut global_i = 0usize;
    let h = cfg.hidden_size;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
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
        // Paged global layers point at the shared pool; local rings are the lane's own.
        let pooled = match paged {
            Some((pool, table)) if !is_local => {
                let l = pool.layers[global_i];
                global_i += 1;
                Some((l, table))
            }
            _ => None,
        };
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
            k_cache: match pooled { Some((l, _)) => l.0, None => backend.alloc_device(kv_bytes)? },
            k_scales: match pooled { Some((l, _)) => l.1, None => backend.alloc_device(sc_bytes)? },
            v_cache: match pooled { Some((l, _)) => l.2, None => backend.alloc_device(kv_bytes)? },
            v_scales: match pooled { Some((l, _)) => l.3, None => backend.alloc_device(sc_bytes)? },
            page_table: pooled.map(|(_, t)| t),
            sconv_k,
            sconv_v,
            sconv_attn,
            sconv_mlp,
        });
    }
    Ok(layer_states)
}

impl<B: ComputeBackend> Inkling2Engine<B> {
    /// `kv_pool_pages`: page the global KV out of a pool of that many pages (when the backend can
    /// page memory); `None` keeps one fixed cache per lane.
    pub fn new(
        cfg: Inkling2Config,
        dense: Inkling2DenseGpu<B>,
        tide: TideExecutor<B>,
        global_max_seq: usize,
        kv_pool_pages: Option<usize>,
    ) -> Result<Self> {
        let backend = dense.backend.clone();
        let stream = tide.compute_stream;
        let h = cfg.hidden_size;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let num_logical = cfg.num_logical_layers();

        let (layer_states, kv_pool) = match kv_pool_pages {
            Some(pages) => {
                let bufs = alloc_kv_pool(&*backend, &cfg, pages)?;
                let mut pool = crate::kv_pager::KvPool::new(pages, global_max_seq.max(cfg.local_window_size));
                let table = pool.add_lane(&*backend)?;
                let st = alloc_layer_states(&*backend, &cfg, global_max_seq, stream, Some((&bufs, table)))?;
                (st, Some((pool, bufs)))
            }
            None => (alloc_layer_states(&*backend, &cfg, global_max_seq, stream, None)?, None),
        };

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
            prefill_min_share: 1.0,
            lane_store: vec![None],
            lane_max_seq: vec![global_max_seq],
            active_lane: 0,
            batch_logits: None,
            batch_route_log: None,
            batch_pending_rows: 0,
            rows_decode: false,
            batch_lanes: Vec::new(),
            verify_hist: None,
            rewind_tables: Vec::new(),
            taps_buf: None,
            capture_taps: false,
            drafter: None,
            route_pinned: None,
            rescue_wait: None,
            wait_flags: None,
            rescue_patches: None,
            rescue_full_table: None,
            rescue_probe: None,
            wait_ticket: 0,
            pending_checks: Vec::new(),
            verify_rescue_active: false,
            verify_rescued: Vec::new(),
            routes: smalltide_tide::expert_map::RouteLog::default(),
            kv_pool,
            map_last_tok: None,
            map_batch_rows: Vec::new(),
            last_tap_rows: 0,
            batch_rows_run: 0,
            dev_rounds: 0,
            dev_unique_resident: 0,
            argmax_rows_buf: None,
            verify_ingest_pos0: None,
            prefill_chunk_swaps: 96,
            anticipate_swaps: 256,
            anticipate_wait: true,
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

    /// Per global layer: `(tensor, bytes per position over all KV heads)` - the caches are
    /// position-major, so a prefix of `pos` positions is the tensor's first `pos * bytes` bytes.
    fn global_kv_regions(&self) -> Vec<(DeviceBuf, usize)> {
        let hd = self.cfg.head_dim;
        let heads = self.cfg.num_key_value_heads;
        let mut v = Vec::new();
        for st in &self.layer_states {
            if !st._is_local {
                v.push((st.k_cache, heads * hd));
                v.push((st.k_scales, heads * 4));
                v.push((st.v_cache, heads * hd));
                v.push((st.v_scales, heads * 4));
            }
        }
        v
    }

    /// Bytes of the fixed part of a `SeqSnapshot` (local KV rings + short-conv histories).
    pub fn snapshot_bytes(&self) -> usize {
        self.snapshot_regions().iter().map(|b| b.size_bytes).sum()
    }

    /// Bytes of a `SeqSnapshot` of `pos` tokens (fixed part + the global KV of the prefix,
    /// plus the DSpark drafter's local + global KV when attached).
    pub fn snapshot_bytes_at(&self, pos: usize) -> usize {
        let mut total = self.snapshot_bytes()
            + self.global_kv_regions().iter().map(|r| pos * r.1).sum::<usize>();
        if let Some(d) = self.drafter.as_ref() {
            total += d.local_kv_regions().iter().map(|b| b.size_bytes).sum::<usize>();
            total += d.global_kv_regions().iter().map(|r| pos * r.1).sum::<usize>();
        }
        total
    }

    /// Captures the state after `pos` tokens into `snap` (allocating or growing its pinned buffer).
    pub fn snapshot_state(&mut self, pos: usize, snap: &mut Option<SeqSnapshot>) -> Result<()> {
        let save_drafter = self.drafter.is_some();
        self.snapshot_state_opt(pos, snap, save_drafter)
    }

    /// Like `snapshot_state`, allowing callers (e.g., multi-lane server) to specify whether the
    /// drafter's KV state is currently coherent for this sequence and should be captured too.
    pub fn snapshot_state_opt(
        &mut self,
        pos: usize,
        snap: &mut Option<SeqSnapshot>,
        save_drafter: bool,
    ) -> Result<()> {
        let s = self.tide.compute_stream;
        let save_drafter = save_drafter && self.drafter.is_some();
        let target_total = self.snapshot_bytes()
            + self.global_kv_regions().iter().map(|r| pos * r.1).sum::<usize>();
        let drafter_total = if save_drafter {
            let d = self.drafter.as_ref().unwrap();
            d.local_kv_regions().iter().map(|b| b.size_bytes).sum::<usize>()
                + d.global_kv_regions().iter().map(|r| pos * r.1).sum::<usize>()
        } else {
            0
        };
        let total = target_total + drafter_total;
        if snap.as_ref().map_or(true, |x| x.host.size_bytes < total) {
            if let Some(old) = snap.take() {
                self.backend.sync_stream(s)?;
                self.backend.free_pinned(old.host)?;
            }
            // A little headroom so a recycled buffer fits the next, slightly longer prefix.
            let alloc = total + total / 8;
            *snap = Some(SeqSnapshot { pos, drafter_valid: save_drafter, host: self.backend.alloc_pinned(alloc)? });
        }
        let sn = snap.as_mut().unwrap();
        sn.pos = pos;
        sn.drafter_valid = save_drafter;
        let mut off = 0usize;
        for b in self.snapshot_regions() {
            self.backend.memcpy_d2h_async(sn.host, off, b, 0, b.size_bytes, s)?;
            off += b.size_bytes;
        }
        let runs = self.kv_runs(pos);
        for (buf, per_pos) in self.global_kv_regions() {
            for &(first, n) in &runs {
                let len = n * per_pos;
                self.backend.memcpy_d2h_async(sn.host, off, buf.slice(first * per_pos, len), 0, len, s)?;
                off += len;
            }
        }
        if save_drafter {
            let d = self.drafter.as_ref().unwrap();
            for b in d.local_kv_regions() {
                self.backend.memcpy_d2h_async(sn.host, off, b, 0, b.size_bytes, s)?;
                off += b.size_bytes;
            }
            for (buf, per_pos) in d.global_kv_regions() {
                let len = pos * per_pos;
                if len > 0 {
                    self.backend.memcpy_d2h_async(sn.host, off, buf.slice(0, len), 0, len, s)?;
                    off += len;
                }
            }
        }
        Ok(())
    }

    /// Restores a snapshot taken by `snapshot_state`; the next token to process is `snap.pos`.
    /// With `global = false` the device global KV is assumed to still hold this prefix (the live
    /// sequence starts with it) and is not copied back. When `snap.drafter_valid` is true and a
    /// drafter is attached, its local rings and global KV are also restored (~4.2 MiB + 2.1 KB/pos).
    pub fn restore_state(&mut self, snap: &SeqSnapshot, global: bool) -> Result<()> {
        let s = self.tide.compute_stream;
        self.kv_release(self.active_lane, snap.pos)?;
        self.kv_ensure(self.active_lane, snap.pos)?;
        let mut off = 0usize;
        for b in self.snapshot_regions() {
            self.backend.memcpy_h2d_async(b, 0, snap.host, off, b.size_bytes, s)?;
            off += b.size_bytes;
        }
        if global {
            let runs = self.kv_runs(snap.pos);
            for (buf, per_pos) in self.global_kv_regions() {
                for &(first, n) in &runs {
                    let len = n * per_pos;
                    self.backend.memcpy_h2d_async(buf.slice(first * per_pos, len), 0, snap.host, off, len, s)?;
                    off += len;
                }
            }
        } else {
            off += self.global_kv_regions().iter().map(|r| snap.pos * r.1).sum::<usize>();
        }
        if snap.drafter_valid {
            if let Some(d) = self.drafter.as_ref() {
                for b in d.local_kv_regions() {
                    self.backend.memcpy_h2d_async(b, 0, snap.host, off, b.size_bytes, s)?;
                    off += b.size_bytes;
                }
                for (buf, per_pos) in d.global_kv_regions() {
                    let len = snap.pos * per_pos;
                    if len > 0 {
                        self.backend.memcpy_h2d_async(buf.slice(0, len), 0, snap.host, off, len, s)?;
                        off += len;
                    }
                }
            }
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
        self.kv_release(self.active_lane, 0)?;
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
        // Async: the copies overlap the first decode tokens and are published once they land
        // (the copy stream waits for every kernel that could still read the evicted slots).
        let ops = self.tide.scheduler.plan_prefill_decode_rebalance(max_swaps);
        let n = ops.len();
        if n > 0 {
            self.tide.execute_transfers(&ops, false, true)?;
        }
        Ok(n)
    }

    /// Prompt-aware loading before a request's prefill: async swaps toward `demand` (expected
    /// routing shares, e.g. the conversation's previous turn). Returns the number of transfers.
    pub fn anticipate(&mut self, demand: &[f32]) -> Result<usize> {
        if self.anticipate_swaps == 0 {
            return Ok(0);
        }
        let ops = self.tide.scheduler.plan_anticipated_swaps(demand, self.anticipate_swaps);
        let n = ops.len();
        if n > 0 {
            self.tide.execute_transfers(&ops, false, true)?;
        }
        if n > 0 && self.anticipate_wait {
            self.tide.wait_and_publish_all_pending()?;
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
        if let Some((lane, pos, token)) = self.map_last_tok.take().filter(|_| !self.route_log_is_prefill) {
            let picks = self.route_picks(&words, num_logical, 0, ROUTE_WORDS);
            self.routes.push(lane, pos, token, 8, picks);
        }
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

    /// Allocates `extra` more lanes (concurrent sequences) of `max_seq` positions each. Call before
    /// the expert pools are allocated (VRAM placement order).
    pub fn add_lanes(&mut self, extra: usize, max_seq: usize) -> Result<()> {
        let s = self.tide.compute_stream;
        // Paged: every lane may reach the full context (the pool, not the lane, bounds it).
        let max_seq = if self.kv_pool.is_some() { self.lane_max_seq[0] } else { max_seq };
        for _ in 0..extra {
            let states = match self.kv_pool.as_mut() {
                Some((pool, bufs)) => {
                    let table = pool.add_lane(&*self.backend)?;
                    let bufs = bufs.clone();
                    alloc_layer_states(&*self.backend, &self.cfg, max_seq, s, Some((&bufs, table)))?
                }
                None => alloc_layer_states(&*self.backend, &self.cfg, max_seq, s, None)?,
            };
            self.lane_store.push(Some(ParkedLane { layer_states: states, rng_state: 0x9E37_79B9_7F4A_7C15 ^ (self.lane_store.len() as u64) }));
            self.lane_max_seq.push(max_seq.max(self.cfg.local_window_size));
        }
        self.backend.sync_stream(s)?;
        Ok(())
    }

    pub fn num_lanes(&self) -> usize {
        self.lane_store.len()
    }

    pub fn active_lane(&self) -> usize {
        self.active_lane
    }

    /// Context length (positions) of lane `i`.
    pub fn lane_max_seq(&self, i: usize) -> usize {
        self.lane_max_seq[i]
    }

    /// Makes lane `i` the active sequence: every single-sequence operation (`prefill`,
    /// `forward_token`, sampling, snapshots) then works on its state. Host-side swap only; all
    /// GPU work stays ordered on the compute stream.
    pub fn select_lane(&mut self, i: usize) {
        if i == self.active_lane {
            return;
        }
        let parked = ParkedLane {
            layer_states: std::mem::take(&mut self.layer_states),
            rng_state: self.rng_state,
        };
        self.lane_store[self.active_lane] = Some(parked);
        let next = self.lane_store[i].take().expect("lane state parked");
        self.layer_states = next.layer_states;
        self.rng_state = next.rng_state;
        self.global_max_seq = self.lane_max_seq[i];
        self.active_lane = i;
        self.tide.scheduler.cache.set_current_lane(i);
    }

    /// The single-row logits buffer (`[unpadded_vocab]` f32).
    pub fn logits_buf(&self) -> DeviceBuf {
        self.scratch.logits
    }

    /// The batched-decode logits buffer (`[MAX_BATCH_ROWS][unpadded_vocab]` f32), once allocated.
    pub fn batch_logits_buf(&self) -> Option<DeviceBuf> {
        self.batch_logits
    }

    /// Layer `il`'s state of `lane` (active or parked).
    pub(crate) fn lane_layer_state(&self, lane: usize, il: usize) -> &LogicalLayerState {
        if lane == self.active_lane {
            &self.layer_states[il]
        } else {
            &self.lane_store[lane].as_ref().expect("parked lane").layer_states[il]
        }
    }

    /// One decode token for each of several lanes in a single batched pass: `rows[i] =
    /// (lane, token, pos)` with distinct lanes. Weight-heavy work (projections, router, experts,
    /// lm_head) runs once over all rows; KV caches and short-conv state are each row's lane's.
    /// Missing experts are skipped (no rescue in batched rounds). Then sample each row with
    /// `sample_batch_row` and call `post_batch_tide_step`.
    pub fn forward_batch(&mut self, rows: &[(usize, u32, usize)]) -> Result<()> {
        for &(lane, _, pos) in rows {
            self.kv_ensure(lane, pos + 1)?;
        }
        self.map_batch_rows = rows.iter().map(|&(lane, tok, pos)| (lane, pos, tok)).collect();
        let m = rows.len();
        if m == 0 || m > MAX_BATCH_ROWS {
            return Err(smalltide_core::SmallTideError::Backend {
                backend: "engine",
                message: format!("forward_batch: {m} rows (1..={MAX_BATCH_ROWS})"),
            });
        }
        if self.pf.is_none() {
            self.pf = Some(crate::prefill::PrefillScratch::new(&*self.backend, &self.cfg)?);
        }
        if self.batch_logits.is_none() {
            self.batch_logits = Some(self.backend.alloc_device(MAX_BATCH_ROWS * self.cfg.unpadded_vocab_size * 4)?);
            self.batch_route_log = Some(self.backend.alloc_device(self.layer_states.len() * MAX_BATCH_ROWS * 64)?);
        }
        // Token boundary bookkeeping, as in forward_token.
        self.consume_route_log()?;
        self.run_deferred_tide_step()?;
        let published = self.tide.poll_pending()?;
        self.tide.scheduler.telemetry.stage1_async_published += published as u64;
        let tokens: Vec<u32> = rows.iter().map(|r| r.1).collect();
        let lp: Vec<(usize, usize)> = rows.iter().map(|r| (r.0, r.2)).collect();
        self.rows_forward(&tokens, crate::prefill::RowsMode::Decode { rows: &lp }, true)?;
        self.batch_pending_rows = m;
        self.batch_lanes = rows.iter().map(|r| r.0).collect();
        Ok(())
    }

    /// History slice of logical layer `il`, short conv `kind` (0 k, 1 v, 2 attn-out, 3 mlp).
    pub(crate) fn verify_hist_slice(&self, il: usize, kind: usize) -> DeviceBuf {
        let kv = self.cfg.num_key_value_heads * self.cfg.head_dim;
        let h = self.cfg.hidden_size;
        let rows = 3 + MAX_BATCH_ROWS;
        let per_layer = rows * (2 * kv + 2 * h);
        let (off, dim) = match kind {
            0 => (0, kv),
            1 => (rows * kv, kv),
            2 => (rows * 2 * kv, h),
            _ => (rows * (2 * kv + h), h),
        };
        self.verify_hist.expect("verify hist").slice((il * per_layer + off) * 4, rows * dim * 4)
    }

    /// Allocates the rows-batched buffers (prefill chunks, batched decode, verify) now, e.g.
    /// before the expert pools so they are placed in VRAM first.
    pub fn preallocate_rows_buffers(&mut self) -> Result<()> {
        self.ensure_rows_buffers()
    }

    fn ensure_rows_buffers(&mut self) -> Result<()> {
        if self.pf.is_none() {
            self.pf = Some(crate::prefill::PrefillScratch::new(&*self.backend, &self.cfg)?);
        }
        if self.batch_logits.is_none() {
            self.batch_logits = Some(self.backend.alloc_device(MAX_BATCH_ROWS * self.cfg.unpadded_vocab_size * 4)?);
            self.batch_route_log = Some(self.backend.alloc_device(self.layer_states.len() * MAX_BATCH_ROWS * 64)?);
        }
        Ok(())
    }

    /// Allocates the speculative-decoding buffers (verify histories, drafter taps). Call before
    /// the expert pools are allocated (VRAM placement order).
    pub fn enable_speculative_buffers(&mut self) -> Result<()> {
        let kv = self.cfg.num_key_value_heads * self.cfg.head_dim;
        let h = self.cfg.hidden_size;
        let n_log = self.layer_states.len();
        if self.verify_hist.is_none() {
            self.verify_hist = Some(self.backend.alloc_device(n_log * (3 + MAX_BATCH_ROWS) * (2 * kv + 2 * h) * 4)?);
            self.taps_buf = Some(self.backend.alloc_device(4 * PREFILL_CHUNK * h * 4)?);
            self.argmax_rows_buf = Some(self.backend.alloc_device(ARGMAX_FILTER_BUF_BYTES)?);
        }
        self.ensure_rows_buffers()
    }

    /// Rescue wait mode (`SMALLTIDE_RESCUE_WAIT`: `0` = host waits, `spin` or `1` = GPU spin-wait
    /// kernel, `waitvalue` = `hipStreamWaitValue32`; forced `Off` while profiling).
    fn rescue_wait_mode(&mut self) -> Result<RescueWaitMode> {
        if self.profile_enabled {
            return Ok(RescueWaitMode::Off);
        }
        if let Some(m) = self.rescue_wait {
            return Ok(m);
        }
        let raw = std::env::var("SMALLTIDE_RESCUE_WAIT").unwrap_or_else(|_| "1".into());
        let mut mode = match raw.trim().to_ascii_lowercase().as_str() {
            "0" | "off" | "false" => RescueWaitMode::Off,
            "waitvalue" | "wait_value" | "wv" => RescueWaitMode::WaitValue,
            "spin" | "1" | "on" | "true" | "" => RescueWaitMode::Spin,
            _ => RescueWaitMode::Spin,
        };
        if mode != RescueWaitMode::Off {
            let n_log = self.layer_planned.len();
            let total_experts = self.cfg.num_physical_layers * self.cfg.n_routed_experts;
            match self.backend.alloc_host_signal(n_log * WAIT_FLAG_STRIDE_WORDS * 4) {
                Ok((view, host)) => {
                    unsafe {
                        let p = self.backend.pinned_as_mut_slice(host)?.as_mut_ptr() as *mut u32;
                        for i in 0..n_log * WAIT_FLAG_STRIDE_WORDS {
                            std::ptr::write_volatile(p.add(i), 0);
                        }
                    }
                    let mut usable = true;
                    if mode == RescueWaitMode::WaitValue {
                        let s = self.tide.compute_stream;
                        if self.backend.stream_wait_value_ge(s, view.slice(0, 4), 0).is_err() {
                            usable = false;
                        }
                    }
                    if !usable {
                        let _ = self.backend.free_device(view);
                        let _ = self.backend.free_pinned(host);
                        mode = RescueWaitMode::Off;
                    } else {
                        let (pview, phost) = self.backend.alloc_host_signal(n_log * PATCH_STRIDE_WORDS * 4)?;
                        let (ftview, fthost) = self.backend.alloc_host_signal(total_experts * 4)?;
                        let (prview, prhost) = self.backend.alloc_host_signal(8 * 4)?;
                        unsafe {
                            let pp = self.backend.pinned_as_mut_slice(phost)?.as_mut_ptr() as *mut u32;
                            for i in 0..n_log * PATCH_STRIDE_WORDS {
                                std::ptr::write_volatile(pp.add(i), 0);
                            }
                            let pr = self.backend.pinned_as_mut_slice(prhost)?.as_mut_ptr() as *mut u32;
                            for i in 0..8 {
                                std::ptr::write_volatile(pr.add(i), 0);
                            }
                        }
                        self.wait_flags = Some((view, host));
                        self.rescue_patches = Some((pview, phost));
                        self.rescue_full_table = Some((ftview, fthost));
                        self.rescue_probe = Some((prview, prhost));
                    }
                }
                Err(_) => mode = RescueWaitMode::Off,
            }
        }
        let desc = match mode {
            RescueWaitMode::Off => "host waits per layer",
            RescueWaitMode::WaitValue => "GPU-side waits (hipStreamWaitValue32)",
            RescueWaitMode::Spin => "GPU-side waits (spin-wait kernel)",
        };
        eprintln!("[rescue] checks: {desc}");
        self.rescue_wait = Some(mode);
        Ok(mode)
    }

    /// After the token's pass is queued: services the rescue checks in layer order. Each check
    /// spins on its coherent routing mirror's ticket sentinel (written with system-scope release
    /// right before the layer's GPU-side wait), plans, copies rescued slabs on `copy_stream`,
    /// writes any slot-table patches to the coherent patch buffer (applied on `compute_stream`
    /// with a system-scope GL2 invalidate right after the wait), then releases the wait.
    fn service_rescue_checks(&mut self, pos: usize) -> Result<()> {
        let s = self.tide.compute_stream;
        let (_, host, ev) = self.route_pinned.expect("route mirror");
        let (_, fhost) = self.wait_flags.expect("wait flags");
        let (_, phost) = self.rescue_patches.expect("rescue patches");
        let (_, fthost) = self.rescue_full_table.expect("rescue full table");
        let prhost = self.rescue_probe.map(|(_, h)| h);
        let ticket = self.wait_ticket;
        // Make the driver submit the queued pass; the host then only reads mapped memory.
        self.backend.record_event(ev, s)?;
        let _ = self.backend.query_event(ev)?;
        let checks = std::mem::take(&mut self.pending_checks);
        let trace = std::env::var_os("SMALLTIDE_RESCUE_TRACE").is_some();
        if trace {
            eprintln!("[rescue-trace] token ticket {ticket}: {} checks {:?}", checks.len(), checks);
        }
        let release_flag = |backend: &B, il: usize| {
            if let Ok(slice) = unsafe { backend.pinned_as_mut_slice(fhost) } {
                let f = slice.as_mut_ptr() as *mut u32;
                std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                unsafe { std::ptr::write_volatile(f.add(il * WAIT_FLAG_STRIDE_WORDS), ticket); }
            }
        };
        // Zero-overhead flight recorder: remembers recent layers' routing/rescue actions for diagnostics on fault.
        struct LayerRec {
            il: usize,
            phys: u16,
            ids: [u16; 8],
            ops_len: usize,
            wait_pending: bool,
            dirty_len: usize,
            first_dirty: Option<(usize, u32)>,
        }
        let mut history: Vec<LayerRec> = Vec::with_capacity(8);
        let mut res: Result<()> = Ok(());
        for (pos_idx, &il) in checks.iter().enumerate() {
            // Once the rescue budget is spent, release all remaining layers immediately without
            // stalling on them one by one; their bookkeeping runs from the routing log afterwards.
            if self.tide.scheduler.rescue_budget_spent() {
                for &rem_il in &checks[pos_idx..] {
                    self.layer_deferred[rem_il] = true;
                    release_flag(&*self.backend, rem_il);
                }
                break;
            }
            let tw = Instant::now();
            if trace {
                eprintln!("[rescue-trace]   layer {il}: waiting for routing row");
            }
            let row = match unsafe { self.backend.pinned_as_mut_slice(host) } {
                Ok(sl) => sl[il * ROUTE_MIRROR_STRIDE_WORDS * 4..].as_ptr() as *const u32,
                Err(e) => {
                    res = Err(e);
                    for &rem_il in &checks[pos_idx..] {
                        release_flag(&*self.backend, rem_il);
                    }
                    break;
                }
            };
            let mut spins = 0u64;
            let mut wait_err: Option<smalltide_core::SmallTideError> = None;
            loop {
                if unsafe { std::ptr::read_volatile(row.add(ROUTE_WORDS)) } == ticket {
                    std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
                    break;
                }
                spins += 1;
                if spins % 8192 == 0 {
                    // Keeps the driver submitting (cheap when nothing is left to submit).
                    if let Err(e) = self.backend.query_event(ev) {
                        wait_err = Some(e);
                        break;
                    }
                    if tw.elapsed().as_secs() > 10 {
                        // Dump rich GPU + host diagnostic probe state before failing.
                        let (pr0, pr1) = prhost
                            .and_then(|h| unsafe { self.backend.pinned_as_mut_slice(h).ok() })
                            .map(|sl| {
                                let p = sl.as_ptr() as *const u32;
                                unsafe { (std::ptr::read_volatile(p), std::ptr::read_volatile(p.add(1))) }
                            })
                            .unwrap_or((0, 0));
                        let pr_tkt = pr0 >> 12;
                        let pr_il = (pr0 >> 4) & 0xFF;
                        let pr_stage = pr0 & 0xF;
                        let stage_name = match pr_stage {
                            0 => "none",
                            2 => "entered_spin_wait",
                            3 => "post_wait_enter_patch",
                            4 => "patch_noop_done",
                            5 => "patch_applied_gl2_inv_done",
                            _ => "unknown",
                        };
                        let cur_sent = unsafe { std::ptr::read_volatile(row.add(ROUTE_WORDS)) };
                        let prev_il = if pos_idx > 0 { Some(checks[pos_idx - 1]) } else { None };
                        let (prev_flag, cur_flag) = unsafe {
                            self.backend.pinned_as_mut_slice(fhost).ok().map(|sl| {
                                let fp = sl.as_ptr() as *const u32;
                                let pf = prev_il.map(|p| std::ptr::read_volatile(fp.add(p * WAIT_FLAG_STRIDE_WORDS)));
                                let cf = std::ptr::read_volatile(fp.add(il * WAIT_FLAG_STRIDE_WORDS));
                                (pf, cf)
                            }).unwrap_or((None, 0))
                        };
                        let prev_patch_n = prev_il.and_then(|p| unsafe {
                            self.backend.pinned_as_mut_slice(phost).ok().map(|sl| {
                                let pp = sl.as_ptr() as *const u32;
                                std::ptr::read_volatile(pp.add(p * PATCH_STRIDE_WORDS))
                            })
                        });
                        let (n4, m8) = self.tide.pool_slots();
                        eprintln!(
                            "[rescue-probe] TIMEOUT waiting for layer {il} (pos_idx={pos_idx}/{}): ticket={ticket} pos={pos} lane={} tokens_decoded={} mode={:?} pools=(nvfp4:{n4},mxfp8:{m8})",
                            checks.len(), self.active_lane, self.tokens_decoded, self.rescue_wait
                        );
                        eprintln!(
                            "[rescue-probe]   GPU breadcrumb: raw=0x{pr0:08x} -> ticket={pr_tkt} layer={pr_il} stage={pr_stage} ({stage_name}) last_patch_n={pr1}"
                        );
                        eprintln!(
                            "[rescue-probe]   flags: prev_il={prev_il:?} prev_flag={prev_flag:?} prev_patch_n={prev_patch_n:?} cur_flag={cur_flag} cur_route_sentinel={cur_sent}"
                        );
                        for rec in history.iter().rev().take(6).rev() {
                            eprintln!(
                                "[rescue-probe]   recent il={} phys={} topk={:?} ops={} wait_pending={} dirty={} first_dirty={:?}",
                                rec.il, rec.phys, rec.ids, rec.ops_len, rec.wait_pending, rec.dirty_len, rec.first_dirty
                            );
                        }
                        let q_res = self.backend.query_event(ev);
                        eprintln!("[rescue-probe]   query_event(pass_end)={q_res:?}");
                        wait_err = Some(smalltide_core::SmallTideError::Backend {
                            backend: "engine",
                            message: format!(
                                "rescue check: routing row of layer {il} never arrived (gpu_probe=tkt:{pr_tkt}/il:{pr_il}/st:{pr_stage}:{stage_name}, prev_patch_n={prev_patch_n:?})"
                            ),
                        });
                        break;
                    }
                }
                std::hint::spin_loop();
            }
            if let Some(e) = wait_err {
                res = Err(e);
                for &rem_il in &checks[pos_idx..] {
                    release_flag(&*self.backend, rem_il);
                }
                break;
            }
            self.profile.rescue_wait_ms += tw.elapsed().as_secs_f64() * 1000.0;
            let tp = Instant::now();
            let mut ids = [0u16; 8];
            let mut ws = [0.0f32; 8];
            for k in 0..8 {
                ids[k] = unsafe { std::ptr::read_volatile(row.add(k)) } as u16;
                ws[k] = f32::from_bits(unsafe { std::ptr::read_volatile(row.add(8 + k)) });
            }
            let (phys, _) = crate::map_logical_to_physical(&self.cfg, smalltide_core::LogicalLayerId(il as u16));
            self.layer_planned[il] = true;
            let plan = self.tide.scheduler.plan_layer_dispatch(phys, &ids, &ws);
            let mut dirty_len = 0usize;
            let mut first_dirty = None;
            if !plan.sync_transfers.is_empty() || plan.wait_pending {
                if trace {
                    eprintln!("[rescue-trace]   layer {il}: rescue {} op(s), wait_pending {}", plan.sync_transfers.len(), plan.wait_pending);
                }
                match self.tide.execute_rescue_now(&plan.sync_transfers) {
                    Ok(dirty) => {
                        dirty_len = dirty.len();
                        first_dirty = dirty.first().copied();
                        let (n4, m8) = self.tide.pool_slots();
                        let scratch_n = smalltide_tide::runtime::DEFAULT_SCRATCH_RING_SLOTS;
                        let n_exp = self.cfg.n_routed_experts;
                        let mut bounds_err: Option<smalltide_core::SmallTideError> = None;
                        for &(idx, val) in &dirty {
                            if val != u32::MAX {
                                let pl = idx / n_exp;
                                let max_s = if self.cfg.is_mxfp8_layer(pl) { m8 + scratch_n } else { n4 + scratch_n };
                                if (val as usize) >= max_s {
                                    bounds_err = Some(smalltide_core::SmallTideError::Backend {
                                        backend: "engine",
                                        message: format!("rescue slot OOB: flat_idx={idx} phys={pl} slot={val} >= max={max_s}"),
                                    });
                                    break;
                                }
                            }
                        }
                        if let Some(e) = bounds_err {
                            res = Err(e);
                            for &rem_il in &checks[pos_idx..] {
                                release_flag(&*self.backend, rem_il);
                            }
                            break;
                        }
                        let patch_res: Result<()> = (|| {
                            unsafe {
                                let p = self.backend.pinned_as_mut_slice(phost)?.as_mut_ptr() as *mut u32;
                                let dst = p.add(il * PATCH_STRIDE_WORDS);
                                if dirty.is_empty() {
                                    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                                    std::ptr::write_volatile(dst, 0xFFFF_FFFEu32);
                                } else if dirty.len() <= PATCH_MAX_PAIRS {
                                    for (k, &(idx, val)) in dirty.iter().enumerate() {
                                        std::ptr::write_volatile(dst.add(1 + 2 * k), idx as u32);
                                        std::ptr::write_volatile(dst.add(2 + 2 * k), val);
                                    }
                                    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                                    std::ptr::write_volatile(dst, dirty.len() as u32);
                                } else {
                                    let tbl = self.tide.scheduler.cache.gpu_slot_table();
                                    let ft = self.backend.pinned_as_mut_slice(fthost)?.as_mut_ptr() as *mut u32;
                                    for (k, &v) in tbl.iter().enumerate() {
                                        std::ptr::write_volatile(ft.add(k), v);
                                    }
                                    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                                    std::ptr::write_volatile(dst, 0xFFFF_FFFFu32);
                                }
                            }
                            Ok(())
                        })();
                        if let Err(e) = patch_res {
                            res = Err(e);
                            for &rem_il in &checks[pos_idx..] {
                                release_flag(&*self.backend, rem_il);
                            }
                            break;
                        }
                        if trace {
                            eprintln!("[rescue-trace]   layer {il}: rescue done ({} patch(es))", dirty.len());
                        }
                    }
                    Err(e) => {
                        res = Err(e);
                        for &rem_il in &checks[pos_idx..] {
                            release_flag(&*self.backend, rem_il);
                        }
                        break;
                    }
                }
            }
            if history.len() == 8 {
                history.remove(0);
            }
            history.push(LayerRec {
                il,
                phys: phys.0,
                ids,
                ops_len: plan.sync_transfers.len(),
                wait_pending: plan.wait_pending,
                dirty_len,
                first_dirty,
            });
            self.profile.rescue_plan_ms += tp.elapsed().as_secs_f64() * 1000.0;
            self.profile.rescue_checks += 1;
            release_flag(&*self.backend, il);
        }
        res
    }

    /// Queues a GPU-side rescue check (route mirror + spin-wait/patch) for verify layer `il`
    /// (`m <= 8` rows) when inline verify rescue is active and physical layer `ip` is not 100%
    /// VRAM-resident.
    pub(crate) fn queue_verify_rescue_check(
        &mut self,
        il: usize,
        ip: usize,
        route: DeviceBuf,
        m: usize,
    ) -> Result<()> {
        if !self.verify_rescue_active {
            return Ok(());
        }
        if self.tide.scheduler.is_layer_fully_resident_vram(PhysicalLayerId(ip as u16)) {
            return Ok(());
        }
        let s = self.tide.compute_stream;
        if self.route_pinned.is_none() {
            let n_log = self.layer_planned.len();
            let (view, host) = self.backend.alloc_host_signal(n_log * ROUTE_MIRROR_STRIDE_WORDS * 4)?;
            unsafe {
                let p = self.backend.pinned_as_mut_slice(host)?.as_mut_ptr() as *mut u32;
                for i in 0..n_log * ROUTE_MIRROR_STRIDE_WORDS {
                    std::ptr::write_volatile(p.add(i), 0);
                }
            }
            self.route_pinned = Some((view, host, self.backend.create_event()?));
        }
        let (view, _, _) = self.route_pinned.unwrap();
        let row_off = il * ROUTE_MIRROR_STRIDE_WORDS * 4;
        self.backend.rescue_write_route(
            view.slice(row_off, ROUTE_MIRROR_STRIDE_WORDS * 4),
            route,
            m * ROUTE_WORDS,
            self.wait_ticket,
            s,
        )?;
        let (fview, _) = self.wait_flags.unwrap();
        let (pview, _) = self.rescue_patches.unwrap();
        let (ftview, _) = self.rescue_full_table.unwrap();
        let prview = self.rescue_probe.map(|(v, _)| v);
        let total_experts = self.cfg.num_physical_layers * self.cfg.n_routed_experts;
        let flag_slice = fview.slice(il * WAIT_FLAG_STRIDE_WORDS * 4, 4);
        let patch_slice = pview.slice(il * PATCH_STRIDE_WORDS * 4, PATCH_STRIDE_WORDS * 4);
        match self.rescue_wait_mode()? {
            RescueWaitMode::Spin => {
                self.backend.rescue_wait_and_patch(
                    self.tide.gpu_slot_table,
                    flag_slice,
                    self.wait_ticket,
                    patch_slice,
                    ftview,
                    total_experts,
                    prview,
                    il,
                    s,
                )?;
            }
            RescueWaitMode::WaitValue => {
                self.backend.stream_wait_value_ge(s, flag_slice, self.wait_ticket)?;
                self.backend.rescue_apply_patch(
                    self.tide.gpu_slot_table,
                    patch_slice,
                    ftview,
                    total_experts,
                    prview,
                    il,
                    self.wait_ticket,
                    s,
                )?;
            }
            RescueWaitMode::Off => return Ok(()),
        }
        self.pending_checks.push(il);
        Ok(())
    }

    /// Services the queued GPU-side inline rescue checks for a multi-row `verify` pass (`m <= 8`),
    /// copying high-weight missing experts once across all `m` rows before releasing each layer's wait.
    pub(crate) fn service_verify_rescue_checks(&mut self, m: usize, pos0: usize) -> Result<()> {
        let s = self.tide.compute_stream;
        let (_, host, ev) = self.route_pinned.expect("route mirror");
        let (_, fhost) = self.wait_flags.expect("wait flags");
        let (_, phost) = self.rescue_patches.expect("rescue patches");
        let (_, fthost) = self.rescue_full_table.expect("rescue full table");
        let prhost = self.rescue_probe.map(|(_, h)| h);
        let ticket = self.wait_ticket;
        let sentinel_off = m * ROUTE_WORDS;
        self.backend.record_event(ev, s)?;
        let _ = self.backend.query_event(ev)?;
        let checks = std::mem::take(&mut self.pending_checks);
        let release_flag = |backend: &B, il: usize| {
            if let Ok(slice) = unsafe { backend.pinned_as_mut_slice(fhost) } {
                let f = slice.as_mut_ptr() as *mut u32;
                std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                unsafe { std::ptr::write_volatile(f.add(il * WAIT_FLAG_STRIDE_WORDS), ticket); }
            }
        };
        let mut res: Result<()> = Ok(());
        for (pos_idx, &il) in checks.iter().enumerate() {
            if self.tide.scheduler.rescue_budget_spent() {
                for &rem_il in &checks[pos_idx..] {
                    release_flag(&*self.backend, rem_il);
                }
                break;
            }
            let tw = Instant::now();
            let row = match unsafe { self.backend.pinned_as_mut_slice(host) } {
                Ok(sl) => sl[il * ROUTE_MIRROR_STRIDE_WORDS * 4..].as_ptr() as *const u32,
                Err(e) => {
                    res = Err(e);
                    for &rem_il in &checks[pos_idx..] {
                        release_flag(&*self.backend, rem_il);
                    }
                    break;
                }
            };
            let mut spins = 0u64;
            let mut wait_err: Option<smalltide_core::SmallTideError> = None;
            loop {
                if unsafe { std::ptr::read_volatile(row.add(sentinel_off)) } == ticket {
                    std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
                    break;
                }
                spins += 1;
                if spins % 8192 == 0 {
                    if let Err(e) = self.backend.query_event(ev) {
                        wait_err = Some(e);
                        break;
                    }
                    if tw.elapsed().as_secs() > 10 {
                        let pr0 = prhost
                            .and_then(|h| unsafe { self.backend.pinned_as_mut_slice(h).ok() })
                            .map(|sl| unsafe { std::ptr::read_volatile(sl.as_ptr() as *const u32) })
                            .unwrap_or(0);
                        wait_err = Some(smalltide_core::SmallTideError::Backend {
                            backend: "engine",
                            message: format!(
                                "verify rescue check: layer {il} (pos0={pos0}, m={m}, ticket={ticket}) timed out (gpu_probe=0x{pr0:08x})"
                            ),
                        });
                        break;
                    }
                }
                std::hint::spin_loop();
            }
            if let Some(e) = wait_err {
                res = Err(e);
                for &rem_il in &checks[pos_idx..] {
                    release_flag(&*self.backend, rem_il);
                }
                break;
            }
            let mut rows_ids = [[0u16; 8]; MAX_BATCH_ROWS];
            let mut rows_ws = [[0.0f32; 8]; MAX_BATCH_ROWS];
            for r in 0..m {
                let base = r * ROUTE_WORDS;
                for k in 0..8 {
                    rows_ids[r][k] = unsafe { std::ptr::read_volatile(row.add(base + k)) } as u16;
                    rows_ws[r][k] = f32::from_bits(unsafe { std::ptr::read_volatile(row.add(base + 8 + k)) });
                }
            }
            let (phys, _) = crate::map_logical_to_physical(&self.cfg, smalltide_core::LogicalLayerId(il as u16));
            let (sync_transfers, wait_pending, rescued) =
                self.tide.scheduler.plan_verify_layer_dispatch(phys, &rows_ids[..m], &rows_ws[..m]);
            if !sync_transfers.is_empty() || wait_pending {
                match self.tide.execute_rescue_now(&sync_transfers) {
                    Ok(dirty) => {
                        self.verify_rescued.extend_from_slice(&rescued);
                        let (n4, m8) = self.tide.pool_slots();
                        let scratch_n = smalltide_tide::runtime::DEFAULT_SCRATCH_RING_SLOTS;
                        let n_exp = self.cfg.n_routed_experts;
                        let mut bounds_err: Option<smalltide_core::SmallTideError> = None;
                        for &(idx, val) in &dirty {
                            if val != u32::MAX {
                                let pl = idx / n_exp;
                                let max_s = if self.cfg.is_mxfp8_layer(pl) { m8 + scratch_n } else { n4 + scratch_n };
                                if (val as usize) >= max_s {
                                    bounds_err = Some(smalltide_core::SmallTideError::Backend {
                                        backend: "engine",
                                        message: format!("verify rescue slot OOB: flat_idx={idx} phys={pl} slot={val} >= max={max_s}"),
                                    });
                                    break;
                                }
                            }
                        }
                        if let Some(e) = bounds_err {
                            res = Err(e);
                            for &rem_il in &checks[pos_idx..] {
                                release_flag(&*self.backend, rem_il);
                            }
                            break;
                        }
                        let patch_res: Result<()> = (|| {
                            unsafe {
                                let p = self.backend.pinned_as_mut_slice(phost)?.as_mut_ptr() as *mut u32;
                                let dst = p.add(il * PATCH_STRIDE_WORDS);
                                if dirty.is_empty() {
                                    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                                    std::ptr::write_volatile(dst, 0xFFFF_FFFEu32);
                                } else if dirty.len() <= PATCH_MAX_PAIRS {
                                    for (k, &(idx, val)) in dirty.iter().enumerate() {
                                        std::ptr::write_volatile(dst.add(1 + 2 * k), idx as u32);
                                        std::ptr::write_volatile(dst.add(2 + 2 * k), val);
                                    }
                                    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                                    std::ptr::write_volatile(dst, dirty.len() as u32);
                                } else {
                                    let tbl = self.tide.scheduler.cache.gpu_slot_table();
                                    let ft = self.backend.pinned_as_mut_slice(fthost)?.as_mut_ptr() as *mut u32;
                                    for (k, &v) in tbl.iter().enumerate() {
                                        std::ptr::write_volatile(ft.add(k), v);
                                    }
                                    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                                    std::ptr::write_volatile(dst, 0xFFFF_FFFFu32);
                                }
                            }
                            Ok(())
                        })();
                        if let Err(e) = patch_res {
                            res = Err(e);
                            for &rem_il in &checks[pos_idx..] {
                                release_flag(&*self.backend, rem_il);
                            }
                            break;
                        }
                    }
                    Err(e) => {
                        res = Err(e);
                        for &rem_il in &checks[pos_idx..] {
                            release_flag(&*self.backend, rem_il);
                        }
                        break;
                    }
                }
            }
            release_flag(&*self.backend, il);
        }
        res
    }

    /// One token's picks over the logical layers from routing-log words (`off` its first word in
    /// layer 0, `stride` words a layer): (expert, share of the layer's routing weight, missed =
    /// not resident in VRAM now).
    fn route_picks(&self, words: &[u32], n_log: usize, off: usize, stride: usize) -> Vec<(u16, f32, bool)> {
        let mut picks = Vec::with_capacity(n_log * 8);
        for il in 0..n_log {
            let (phys, _) = crate::map_logical_to_physical(&self.cfg, smalltide_core::LogicalLayerId(il as u16));
            let w = &words[il * stride + off..il * stride + off + 16];
            let tot: f32 = (0..8).map(|k| f32::from_bits(w[8 + k])).sum();
            let inv = if tot > 1e-8 { 1.0 / tot } else { 0.0 };
            for k in 0..8 {
                let e = w[k] as u16;
                let missed = !self.tide.scheduler.cache.is_vram_hit(smalltide_core::ExpertId { layer: phys, expert: e });
                picks.push((e, f32::from_bits(w[8 + k]) * inv, missed));
            }
        }
        picks
    }

    /// The residency as the expert map shows it: one row per logical layer (the loop core's
    /// physical layers appear once per pass), host bookkeeping only. `since`: only routes newer
    /// than this sequence number are copied.
    pub fn expert_map(&self, busy: bool, since: u64) -> smalltide_tide::expert_map::ExpertMap {
        use smalltide_tide::expert_map::{ExpertMap, HOT, LOADING, RESIDENT};
        let cache = &self.tide.scheduler.cache;
        let e = self.cfg.n_routed_experts;
        let n_log = self.layer_planned.len();
        let layers: Vec<usize> = (0..n_log).collect();
        let mut state = Vec::with_capacity(n_log * e);
        let mut demand = Vec::with_capacity(n_log * e);
        for il in 0..n_log {
            let (phys, _) = crate::map_logical_to_physical(&self.cfg, smalltide_core::LogicalLayerId(il as u16));
            for x in 0..e {
                let id = smalltide_core::ExpertId { layer: phys, expert: x as u16 };
                let en = cache.entry(id);
                state.push(if cache.is_vram_pending(id) {
                    LOADING
                } else if !en.vram_slot.is_missing() && en.is_scratch_slot {
                    HOT
                } else if cache.is_vram_hit(id) {
                    RESIDENT
                } else {
                    0
                });
                demand.push(cache.value(id).max(0.0));
            }
        }
        let t = &self.tide.scheduler.telemetry;
        let counters = vec![
            ("experts moved", t.stage1_async_promotions as f64),
            ("inline rescues", t.stage1_sync_rescues as f64),
            ("skipped experts", t.stage1_skipped_tail_misses as f64),
            ("prompt-aware swaps", t.prompt_aware_swaps as f64),
            ("disk reads", t.stage2_disk_reads as f64),
        ];
        let (n4, m8) = self.tide.pool_slots();
        let routes = self.routes.recent(6.0, 400).into_iter().filter(|r| r.seq > since).collect();
        ExpertMap {
            layers,
            experts: e,
            state,
            demand: Some(demand),
            slots: n4 + m8,
            spare: smalltide_tide::runtime::DEFAULT_SCRATCH_RING_SLOTS * 2,
            world: 1,
            busy,
            counters,
            events: self.tide.scheduler.events.recent(6.0, 4000),
            routes,
        }
    }

    /// Backs `lane`'s global KV up to position `end` (exclusive); an error when the pool cannot.
    pub(crate) fn kv_ensure(&mut self, lane: usize, end: usize) -> Result<()> {
        if self.kv_reserve(lane, end)? {
            Ok(())
        } else {
            Err(smalltide_core::SmallTideError::Backend {
                backend: "engine",
                message: format!("KV pool full: lane {lane} needs {end} positions and no idle lane can give pages back"),
            })
        }
    }

    /// Like `kv_ensure`, but returns false instead of failing (the server stops that request).
    pub fn kv_reserve(&mut self, lane: usize, end: usize) -> Result<bool> {
        let (b, s) = (self.backend.clone(), self.tide.compute_stream);
        let Some((pool, _)) = self.kv_pool.as_mut() else { return Ok(true) };
        if end > self.lane_max_seq[lane] {
            return Ok(false);
        }
        pool.ensure(&*b, lane, end, s)
    }

    /// Returns `lane`'s global KV pages past position `keep` to the pool.
    pub fn kv_release(&mut self, lane: usize, keep: usize) -> Result<()> {
        if let Some((pool, _)) = self.kv_pool.as_mut() {
            pool.release(lane, keep);
        }
        Ok(())
    }

    /// Marks `lane` as serving a request: its KV pages are never reclaimed.
    pub fn set_lane_pinned(&mut self, lane: usize, pinned: bool) {
        if let Some((pool, _)) = self.kv_pool.as_mut() {
            pool.pinned[lane] = pinned;
        }
    }

    /// Lanes whose KV pages were reclaimed for another lane since the last call: their device
    /// sequence is gone (their conversations survive only in host snapshots).
    pub fn take_lost_lanes(&mut self) -> Vec<usize> {
        self.kv_pool.as_mut().map(|(p, _)| p.take_lost()).unwrap_or_default()
    }

    /// Paged KV: (pages in the pool, free pages, positions a page, bytes a page).
    pub fn kv_pool_info(&self) -> Option<(usize, usize, usize, usize)> {
        let per_pos: usize = self.global_kv_regions().iter().map(|r| r.1).sum();
        self.kv_pool.as_ref().map(|(p, _)| (p.pages_total(), p.pages_free(), crate::kv_pager::PAGE_POS, per_pos * crate::kv_pager::PAGE_POS))
    }

    /// The active lane's global KV of positions `[0, pos)` as (first pool position, positions)
    /// runs: one per page when paged, one run otherwise.
    fn kv_runs(&self, pos: usize) -> Vec<(usize, usize)> {
        match self.kv_pool.as_ref() {
            Some((pool, _)) => {
                let pp = crate::kv_pager::PAGE_POS;
                pool.lane_pages(self.active_lane)
                    .iter()
                    .enumerate()
                    .take_while(|(j, _)| j * pp < pos)
                    .map(|(j, &pg)| (pg as usize * pp, pp.min(pos - j * pp)))
                    .collect()
            }
            None if pos > 0 => vec![(0, pos)],
            None => vec![],
        }
    }

    /// Dev: the first global layer's cache at positions 0..2 of the active lane (K bytes [0..8]
    /// of head 0, K scale of head 0, V bytes [0..8]).
    pub fn debug_kv_peek(&self) -> Result<Vec<(Vec<u8>, f32, Vec<u8>)>> {
        let st = self.layer_states.iter().find(|st| !st._is_local).expect("a global layer");
        let (n, hd) = (self.cfg.num_key_value_heads, self.cfg.head_dim);
        let s = self.tide.compute_stream;
        let base = self.kv_runs(2).first().map(|r| r.0).unwrap_or(0);
        let mut out = Vec::new();
        for p in 0..2usize {
            let q = base + p;
            let mut k = vec![0u8; 8];
            self.backend.memcpy_d2h_sync(&mut k, st.k_cache.slice(q * n * hd, 8), 0, s)?;
            let mut v = vec![0u8; 8];
            self.backend.memcpy_d2h_sync(&mut v, st.v_cache.slice(q * n * hd, 8), 0, s)?;
            let mut sc = [0u8; 4];
            self.backend.memcpy_d2h_sync(&mut sc, st.k_scales.slice(q * n * 4, 4), 0, s)?;
            out.push((k, f32::from_le_bytes(sc), v));
        }
        Ok(out)
    }

    /// Dev: redirect the per-token dump (`SMALLTIDE_DUMP_DIR`) to `dir`.
    pub fn set_dump_dir(&mut self, dir: std::path::PathBuf) {
        if let Some(d) = self.dump.as_mut() {
            d.dir = dir;
        }
    }

    /// Attaches the DSpark drafter: taps are captured from now on (speculative buffers allocated).
    pub fn attach_drafter(&mut self, d: crate::drafter::Drafter<B>) -> Result<()> {
        self.drafter = Some(d);
        self.capture_taps = true;
        self.enable_speculative_buffers()
    }

    /// Context K/V of the drafter for the `m` rows just run at `pos0..` (no-op without drafter).
    pub(crate) fn drafter_ingest(&mut self, pos0: usize, m: usize) -> Result<()> {
        self.last_tap_rows = m;
        if let (Some(d), Some(taps)) = (self.drafter.as_ref(), self.taps_buf) {
            d.ingest(taps, PREFILL_CHUNK, pos0, m, self.tide.compute_stream)?;
        }
        Ok(())
    }

    /// Drafts `n` tokens after `anchor` (at position `p`); `tap_row`: row of position `p - 1` in
    /// the last tap-capturing forward (prefill chunk or verify).
    pub fn draft(&mut self, anchor: u32, p: usize, tap_row: usize, n: usize) -> Result<Vec<u32>> {
        let taps = self.taps_buf.expect("taps buffer");
        let s = self.tide.compute_stream;
        let d = self.drafter.as_mut().expect("drafter attached");
        d.draft(&self.dense, anchor, p, taps, PREFILL_CHUNK, tap_row, n, s)
    }

    /// Like `draft`, returning `(token_id, top1_minus_top2_logit_margin)` per drafted token.
    pub fn draft_with_margins(&mut self, anchor: u32, p: usize, tap_row: usize, n: usize) -> Result<Vec<(u32, f32)>> {
        let taps = self.taps_buf.expect("taps buffer");
        let s = self.tide.compute_stream;
        let d = self.drafter.as_mut().expect("drafter attached");
        d.draft_with_margins(&self.dense, anchor, p, taps, PREFILL_CHUNK, tap_row, n, s)
    }

    /// Speculative verify (greedy): skips monotonic final logit softcap; follow with `rows_argmax`
    /// and `verify_rewind`.
    pub fn verify(&mut self, tokens: &[u32], pos0: usize) -> Result<()> {
        self.verify_for_sampling(tokens, pos0, true)
    }

    /// Speculative verify: runs `tokens` (<= 8, the anchor then the drafts) of the active lane at
    /// positions `pos0..` in one pass; logits of every row land in the batch logits buffer and the
    /// drafter taps in `taps_buf` (if `capture_taps`). Set `greedy = false` when sampling with
    /// `temperature > 0` or `repetition_penalty != 1.0` so final logit softcapping is applied.
    pub fn verify_for_sampling(&mut self, tokens: &[u32], pos0: usize, greedy: bool) -> Result<()> {
        let m = tokens.len();
        self.kv_ensure(self.active_lane, pos0 + m)?;
        if m == 0 || m > MAX_BATCH_ROWS {
            return Err(smalltide_core::SmallTideError::Backend { backend: "engine", message: format!("verify: {m} rows") });
        }
        if self.verify_hist.is_none() {
            self.enable_speculative_buffers()?;
        }
        self.ensure_rows_buffers()?;
        self.consume_route_log()?;
        let use_verify_rescue = self.dump.is_none()
            && matches!(self.tide.scheduler.tide_cfg.rescue_policy, smalltide_core::RescuePolicy::Threshold(_))
            && std::env::var_os("SMALLTIDE_VERIFY_NO_RESCUE").is_none()
            && self.rescue_wait_mode()? != RescueWaitMode::Off;
        if use_verify_rescue {
            // Drain & publish any pending async promotions BEFORE queueing spin-waits, and defer
            // `run_deferred_tide_step` until after `service_verify_rescue_checks` (same invariant as `forward_token`).
            if self.tide.pending_batch_count() > 0 {
                let pub2 = self.tide.wait_and_publish_all_pending()?;
                self.tide.scheduler.telemetry.stage1_async_published += pub2 as u64;
            }
            self.tide.scheduler.begin_token();
            self.wait_ticket = self.wait_ticket.wrapping_add(1).max(1);
            if let Some((_, phost)) = self.rescue_patches {
                let n_log = self.layer_planned.len();
                unsafe {
                    let p = self.backend.pinned_as_mut_slice(phost)?.as_mut_ptr() as *mut u32;
                    for il in 0..n_log {
                        std::ptr::write_volatile(p.add(il * PATCH_STRIDE_WORDS), 0);
                    }
                }
            }
            self.pending_checks.clear();
            self.verify_rescued.clear();
        } else {
            self.run_deferred_tide_step()?;
            let published = self.tide.poll_pending()?;
            self.tide.scheduler.telemetry.stage1_async_published += published as u64;
            self.verify_rescued.clear();
        }
        self.verify_rescue_active = use_verify_rescue;
        self.map_batch_rows = tokens.iter().enumerate().map(|(i, &t)| (self.active_lane, pos0 + i, t)).collect();
        let fwd_res = self.rows_forward(tokens, crate::prefill::RowsMode::Verify { pos0, greedy }, true);
        self.verify_rescue_active = false;
        fwd_res?;
        // Defer drafter context KV ingest until `verify_rewind`, so `rows_argmax` / `verify_sample`
        // doesn't wait behind it and rejected rows are never ingested.
        self.verify_ingest_pos0 = self.capture_taps.then_some(pos0);
        self.batch_pending_rows = m;
        self.batch_rows_run = m;
        self.batch_lanes = vec![self.active_lane; m];
        Ok(())
    }

    /// Greedy next token of each of the last `m` verify / batch rows (one readback).
    pub fn rows_argmax(&mut self, m: usize) -> Result<Vec<u32>> {
        if self.argmax_rows_buf.is_none() {
            self.argmax_rows_buf = Some(self.backend.alloc_device(ARGMAX_FILTER_BUF_BYTES)?);
        }
        let buf = self.argmax_rows_buf.unwrap();
        let out = buf.slice(0, MAX_BATCH_ROWS * 4);
        let scratch = buf.slice(MAX_BATCH_ROWS * 8, MAX_BATCH_ROWS * 32 * 4);
        let s = self.tide.compute_stream;
        self.backend.argmax_rows_fast(out, scratch, self.batch_logits.expect("batch logits"), m, self.cfg.unpadded_vocab_size, s)?;
        let mut b = vec![0u8; m * 4];
        self.backend.memcpy_d2h_sync(&mut b, out.slice(0, m * 4), 0, s)?;
        Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    /// Exact lossless speculative verification for any `SamplingParams` (greedy or stochastic).
    ///
    /// - When `temperature <= 0` and `repetition_penalty == 1.0`, uses GPU `rows_argmax` (32 B D2H).
    /// - When `temperature > 0` and `repetition_penalty == 1.0`, runs GPU `logits_filter_rows`
    ///   (extracting only entries within `20 * T` of `row_max`, ~16 KB D2H instead of 6.4 MB!)
    ///   and performs exact rejection sampling (`q(d) = 1`) on the compact candidates, falling
    ///   back to full-row D2H only if a row overflows 256 candidates.
    /// - Otherwise, reads back the `block.len()` logit rows in one D2H copy and runs exact
    ///   rejection sampling for deterministic drafter proposals (`q(d) = 1`).
    ///
    /// Returns `(accepted_drafts, next_token)` where `block[1..=accepted_drafts]` are accepted
    /// and `next_token` is the replacement or bonus token (total `accepted_drafts + 1` tokens).
    pub fn verify_sample(
        &mut self,
        block: &[u32],
        params: &SamplingParams,
        recent_tokens: &[u32],
    ) -> Result<(usize, u32)> {
        let m = block.len();
        let n = m.saturating_sub(1);
        let no_rep = (params.repetition_penalty - 1.0).abs() < 1e-6;
        if params.temperature <= 0.0 && no_rep {
            let am = self.rows_argmax(m)?;
            let mut c = 0usize;
            while c < n && block[c + 1] == am[c] {
                c += 1;
            }
            return Ok((c, am[c]));
        }

        let vocab = self.cfg.unpadded_vocab_size;
        let s = self.tide.compute_stream;
        let bl = self.batch_logits.expect("batch logits");

        if params.temperature > 0.0 && no_rep {
            if self.argmax_rows_buf.is_none() {
                self.argmax_rows_buf = Some(self.backend.alloc_device(ARGMAX_FILTER_BUF_BYTES)?);
            }
            let buf = self.argmax_rows_buf.unwrap();
            let row_max = buf.slice(MAX_BATCH_ROWS * 4, MAX_BATCH_ROWS * 4);
            let scratch = buf.slice(MAX_BATCH_ROWS * 8, MAX_BATCH_ROWS * 32 * 4);
            let d2h_base = MAX_BATCH_ROWS * 40 * 4; // 1280
            let counts = buf.slice(d2h_base, MAX_BATCH_ROWS * 4);
            let out_idx = buf.slice(d2h_base + MAX_BATCH_ROWS * 4, MAX_BATCH_ROWS * FILTER_CAP * 4);
            let out_val = buf.slice(
                d2h_base + MAX_BATCH_ROWS * 4 + MAX_BATCH_ROWS * FILTER_CAP * 4,
                MAX_BATCH_ROWS * FILTER_CAP * 4,
            );
            let margin = 20.0 * params.temperature;
            self.backend.logits_filter_rows(
                out_idx, out_val, counts, row_max, scratch, bl, m, vocab, margin, FILTER_CAP, s,
            )?;
            let d2h_bytes = (MAX_BATCH_ROWS + 2 * MAX_BATCH_ROWS * FILTER_CAP) * 4;
            let mut raw = vec![0u8; d2h_bytes];
            self.backend.memcpy_d2h_sync(&mut raw, buf.slice(d2h_base, d2h_bytes), 0, s)?;

            let word = |w: usize| u32::from_le_bytes(raw[w * 4..(w + 1) * 4].try_into().unwrap());
            let idx_base = MAX_BATCH_ROWS;
            let val_base = MAX_BATCH_ROWS + MAX_BATCH_ROWS * FILTER_CAP;
            let inv_t = 1.0 / params.temperature;

            let row_probs = |r: usize| -> Result<Vec<(u32, f32)>> {
                let cnt = word(r) as usize;
                if cnt > 0 && cnt <= FILTER_CAP {
                    let mut cands = Vec::with_capacity(cnt);
                    for j in 0..cnt {
                        let id = word(idx_base + r * FILTER_CAP + j);
                        let v = f32::from_bits(word(val_base + r * FILTER_CAP + j));
                        cands.push((id, v * inv_t));
                    }
                    Ok(finalize_probs(cands, params.top_k.clamp(1, vocab), params.top_p))
                } else {
                    // Rare overflow (> 256 tokens within 20*T of max): read just this single row.
                    let mut row = vec![0.0f32; vocab];
                    let bytes: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut u8, vocab * 4)
                    };
                    self.backend.memcpy_d2h_sync(bytes, bl.slice(r * vocab * 4, vocab * 4), 0, s)?;
                    Ok(row_target_probs(&mut row, params, &[], &[]))
                }
            };

            for c in 0..n {
                let mut probs = row_probs(c)?;
                let draft = block[c + 1];
                let p_d = probs.iter().find(|&&(id, _)| id == draft).map(|&(_, p)| p).unwrap_or(0.0);
                if next_rand_f32(&mut self.rng_state) < p_d {
                    continue;
                }
                probs.retain(|&(id, _)| id != draft);
                if probs.is_empty() {
                    return Ok((c, draft));
                }
                return Ok((c, sample_from_probs(&probs, &mut self.rng_state)));
            }
            let probs = row_probs(n)?;
            return Ok((n, sample_from_probs(&probs, &mut self.rng_state)));
        }

        let mut all_logits = vec![0.0f32; m * vocab];
        let raw_bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(all_logits.as_mut_ptr() as *mut u8, m * vocab * 4)
        };
        self.backend.memcpy_d2h_sync(raw_bytes, bl.slice(0, m * vocab * 4), 0, s)?;

        Ok(verify_sample_rows(&mut all_logits, vocab, block, params, recent_tokens, &mut self.rng_state))
    }

    /// After a verify of `m` rows of which `keep` are committed: ingests ONLY the `keep`
    /// committed rows into the drafter's context KV, and rewinds every short conv of the active
    /// lane to its state after `keep` rows (the KV of later positions is overwritten later).
    pub fn verify_rewind(&mut self, m: usize, keep: usize) -> Result<()> {
        // Only committed rows count for Tide (heat / demand); call before post_batch_tide_step.
        self.batch_pending_rows = self.batch_pending_rows.min(keep);
        if let Some(pos0) = self.verify_ingest_pos0.take() {
            self.drafter_ingest(pos0, keep)?;
        }
        if keep >= m {
            return Ok(());
        }
        let lane = self.active_lane;
        while self.rewind_tables.len() <= lane {
            self.rewind_tables.push(None);
        }
        let n_log = self.layer_states.len();
        if self.rewind_tables[lane].is_none() {
            let kv = self.cfg.num_key_value_heads * self.cfg.head_dim;
            let h = self.cfg.hidden_size;
            let mut words: Vec<u64> = Vec::with_capacity(n_log * 16);
            for il in 0..n_log {
                let st = &self.layer_states[il];
                for (kind, state, dim) in [(0, st.sconv_k, kv), (1, st.sconv_v, kv), (2, st.sconv_attn, h), (3, st.sconv_mlp, h)] {
                    words.push(self.backend.device_address(self.verify_hist_slice(il, kind))?);
                    words.push(self.backend.device_address(state)?);
                    words.push(dim as u64);
                    words.push(0);
                }
            }
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let dev = self.backend.alloc_device(bytes.len())?;
            let host = self.backend.alloc_pinned(bytes.len())?;
            unsafe {
                self.backend.pinned_as_mut_slice(host)?[..bytes.len()].copy_from_slice(&bytes);
            }
            let s = self.tide.compute_stream;
            self.backend.memcpy_h2d_async(dev, 0, host, 0, bytes.len(), s)?;
            self.backend.sync_stream(s)?;
            self.backend.free_pinned(host)?;
            self.rewind_tables[lane] = Some(dev);
        }
        let h = self.cfg.hidden_size;
        self.backend.sconv_rewind(self.rewind_tables[lane].unwrap(), n_log * 4, h, keep, self.tide.compute_stream)
    }

    /// Samples row `i` of the last `forward_batch` with `lane`'s RNG.
    pub fn sample_batch_row(&mut self, i: usize, lane: usize, params: &SamplingParams, recent: &[u32]) -> Result<u32> {
        self.select_lane(lane);
        let vocab = self.cfg.unpadded_vocab_size;
        let logits = self.batch_logits.expect("batch logits").slice(i * vocab * 4, vocab * 4);
        self.sample_logits(logits, params, recent)
    }

    /// Tide bookkeeping of the last batched round (one readback of its routing log): hits,
    /// misses, heat, skipped weight, async promotion queue; the async step itself is deferred
    /// into the next forward like after a single-row token.
    pub fn post_batch_tide_step(&mut self) -> Result<()> {
        let m = std::mem::take(&mut self.batch_pending_rows);
        if m == 0 {
            return Ok(());
        }
        let n_log = self.layer_states.len();
        let log = self.batch_route_log.expect("batch route log");
        let mut bytes = vec![0u8; n_log * MAX_BATCH_ROWS * 64];
        self.backend.memcpy_d2h_sync(&mut bytes, log, 0, self.tide.compute_stream)?;
        let word = |i: usize| u32::from_le_bytes([bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]]);
        let run = std::mem::take(&mut self.batch_rows_run).max(m);
        self.dev_rounds += 1;
        {
            let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let rows = std::mem::take(&mut self.map_batch_rows);
            for (r, &(lane, pos, tok)) in rows.iter().enumerate().take(m) {
                let picks = self.route_picks(&words, n_log, r * 16, MAX_BATCH_ROWS * 16);
                self.routes.push(lane, pos, tok, 8, picks);
            }
        }
        for il in 0..n_log {
            let (phys, _) = crate::map_logical_to_physical(&self.cfg, smalltide_core::LogicalLayerId(il as u16));
            let mut seen: Vec<u16> = Vec::with_capacity(64);
            for r in 0..run {
                for k in 0..8 {
                    let e = word((il * MAX_BATCH_ROWS + r) * 16 + k) as u16;
                    if !seen.contains(&e) && self.tide.scheduler.cache.is_vram_hit(smalltide_core::ExpertId { layer: phys, expert: e }) {
                        seen.push(e);
                    }
                }
            }
            self.dev_unique_resident += seen.len() as u64;
        }
        let rescued = std::mem::take(&mut self.verify_rescued);
        for il in 0..n_log {
            let (phys, _) = crate::map_logical_to_physical(&self.cfg, smalltide_core::LogicalLayerId(il as u16));
            for r in 0..m {
                let lane = self.batch_lanes.get(r).copied().unwrap_or(self.active_lane);
                self.tide.scheduler.cache.set_current_lane(lane);
                let base = (il * MAX_BATCH_ROWS + r) * 16;
                let mut ids = [0u16; 8];
                let mut ws = [0f32; 8];
                for k in 0..8 {
                    ids[k] = word(base + k) as u16;
                    ws[k] = f32::from_bits(word(base + 8 + k));
                }
                self.tide.scheduler.record_batch_dispatch_with_rescued(phys, &ids, &ws, &rescued);
            }
        }
        self.tide.scheduler.cache.set_current_lane(self.active_lane);
        self.tokens_decoded += m as u64;
        self.tide_step_pending = true;
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

        // inpSA = cur, attention-residual mix -> inputs[0..3] and, without controllers, the input
        // norms: one fused launch.
        let nq_mix = if is_local { 1 } else { 3 };
        let norm_w = if has_msg {
            [None; 3]
        } else if is_local {
            [Some(layer.attn_norm.expect("local layer attn_norm")), None, None]
        } else {
            [layer.q_input_norm, layer.k_input_norm, layer.v_input_norm]
        };
        self.backend.fused_mix_decode(
            FusedMixParams {
                cur: self.scratch.cur,
                sconv: None,
                residual: self.scratch.cur,
                blocks: self.scratch.blocks,
                nb: *num_blocks,
                boundary: false,
                copy_out: Some(self.scratch.inp_sa),
                queries: layer.attn_res_query,
                nq: nq_mix,
                mix_out: if has_msg { Some(self.scratch.in_qkv) } else { None },
                norm_w,
                norm_out: self.scratch.norm_qkv,
                hidden: h,
                eps,
            },
            s,
        )?;
        let (in0, in1, in2) = if is_local {
            let d0 = self.scratch.in_qkv.slice(0, h_bytes);
            (d0, d0, d0)
        } else {
            (
                self.scratch.in_qkv.slice(0, h_bytes),
                self.scratch.in_qkv.slice(h_bytes, h_bytes),
                self.scratch.in_qkv.slice(2 * h_bytes, h_bytes),
            )
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

        // 3. Attention Input Norms (`inkling2.cpp:313-324`); already fused above without controllers.
        let (nq, nk, nv) = if !has_msg {
            let n0 = self.scratch.norm_qkv.slice(0, h_bytes);
            if is_local {
                (n0, n0, n0)
            } else {
                (n0, self.scratch.norm_qkv.slice(h_bytes, h_bytes), self.scratch.norm_qkv.slice(2 * h_bytes, h_bytes))
            }
        } else if is_local {
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
                page_table: st.page_table,
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
                page_table: st.page_table,
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
        // 8./9. Attention short conv + block residual boundary (`inkling2.cpp:381-388`) + ffn_inp
        // copy + MLP residual mix (+ MLP norm without a controller): one fused launch.
        let boundary = ip % self.cfg.n_layers_per_attn_res_block == 0;
        let has_ctrl = messages.is_some();
        self.backend.fused_mix_decode(
            FusedMixParams {
                cur: self.scratch.cur,
                sconv: Some((self.scratch.o_raw, st.sconv_attn, layer.attn_sconv)),
                residual: self.scratch.inp_sa,
                blocks: self.scratch.blocks,
                nb: *num_blocks,
                boundary,
                copy_out: Some(self.scratch.ffn_inp),
                queries: Some(layer.mlp_res_query),
                nq: 1,
                mix_out: if has_ctrl { Some(self.scratch.moe_in) } else { None },
                norm_w: if has_ctrl { [None; 3] } else { [Some(layer.mlp_norm), None, None] },
                norm_out: self.scratch.moe_norm,
                hidden: h,
                eps,
            },
            s,
        )?;
        if boundary {
            *num_blocks += 1;
        }
        if let Some(msgs) = messages {
            let ctrls = layer.controllers.as_ref().unwrap();
            self.run_controls(&[(self.scratch.moe_ctrl, self.scratch.moe_in, msgs.slice(3 * h_bytes, h_bytes), &ctrls[3])], s)?;
            self.backend
                .rms_norm(self.scratch.moe_norm, self.scratch.moe_ctrl, layer.mlp_norm, 1, h, eps, s)?;
        }
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
        let policy = self.tide.scheduler.tide_cfg.rescue_policy;
        let defer = !fully_resident
            && !is_prefill
            && !need_dump_routes
            && (matches!(policy, smalltide_core::RescuePolicy::Disabled) || self.tide.scheduler.rescue_budget_spent());
        // Threshold rescue in decode: the routing row goes to the host asynchronously and the
        // expert kernels are queued right away (unpublished experts skipped); the host decides
        // while they run and only a rescue re-runs this layer's experts. No GPU drain per layer.
        let overlap = !fully_resident
            && !defer
            && !is_prefill
            && !need_dump_routes
            && matches!(policy, smalltide_core::RescuePolicy::Threshold(_))
            && std::env::var_os("SMALLTIDE_RESCUE_SERIAL").is_none();
        let wait_mode = if overlap { self.rescue_wait_mode()? } else { RescueWaitMode::Off };
        let use_wait = wait_mode != RescueWaitMode::Off;
        if overlap {
            if self.route_pinned.is_none() {
                let n_log = self.layer_planned.len();
                let (view, host) = self.backend.alloc_host_signal(n_log * ROUTE_MIRROR_STRIDE_WORDS * 4)?;
                unsafe {
                    let p = self.backend.pinned_as_mut_slice(host)?.as_mut_ptr() as *mut u32;
                    for i in 0..n_log * ROUTE_MIRROR_STRIDE_WORDS {
                        std::ptr::write_volatile(p.add(i), 0);
                    }
                }
                self.route_pinned = Some((view, host, self.backend.create_event()?));
            }
            let (view, _, ev) = self.route_pinned.unwrap();
            let row_off = il * ROUTE_MIRROR_STRIDE_WORDS * 4;
            if use_wait {
                // Coherent route mirror + release-fenced ticket sentinel, then GPU-side wait,
                // then compute-stream table patch + system-scope GL2 invalidate before MoE.
                self.backend.rescue_write_route(
                    view.slice(row_off, ROUTE_MIRROR_STRIDE_WORDS * 4),
                    route_row,
                    ROUTE_WORDS,
                    self.wait_ticket,
                    s,
                )?;
                let (fview, _) = self.wait_flags.unwrap();
                let (pview, _) = self.rescue_patches.unwrap();
                let (ftview, _) = self.rescue_full_table.unwrap();
                let prview = self.rescue_probe.map(|(v, _)| v);
                let total_experts = self.cfg.num_physical_layers * self.cfg.n_routed_experts;
                let flag_slice = fview.slice(il * WAIT_FLAG_STRIDE_WORDS * 4, 4);
                let patch_slice = pview.slice(il * PATCH_STRIDE_WORDS * 4, PATCH_STRIDE_WORDS * 4);
                match wait_mode {
                    RescueWaitMode::Spin => {
                        self.backend.rescue_wait_and_patch(
                            self.tide.gpu_slot_table,
                            flag_slice,
                            self.wait_ticket,
                            patch_slice,
                            ftview,
                            total_experts,
                            prview,
                            il,
                            s,
                        )?;
                    }
                    RescueWaitMode::WaitValue => {
                        self.backend.stream_wait_value_ge(s, flag_slice, self.wait_ticket)?;
                        self.backend.rescue_apply_patch(
                            self.tide.gpu_slot_table,
                            patch_slice,
                            ftview,
                            total_experts,
                            prview,
                            il,
                            self.wait_ticket,
                            s,
                        )?;
                    }
                    RescueWaitMode::Off => unreachable!(),
                }
                self.pending_checks.push(il);
            } else {
                // A kernel store into mapped host memory (no copy-engine round trip in the stream).
                self.backend.vec_copy(view.slice(row_off, ROUTE_WORDS * 4), route_row, ROUTE_WORDS, s)?;
                self.backend.record_event(ev, s)?;
            }
        } else if defer {
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
        if overlap && !use_wait {
            let (_, host, ev) = self.route_pinned.unwrap();
            let tw = Instant::now();
            // Blocking event wait: on Windows every host round trip costs a GPU submission, so
            // polling (which forces extra submissions) is slower than one blocking wait.
            self.backend.sync_event(ev)?;
            self.profile.rescue_wait_ms += tw.elapsed().as_secs_f64() * 1000.0;
            let tp = Instant::now();
            let mut topk_ids_u16 = [0u16; 8];
            let mut topk_w_f32 = [0.0f32; 8];
            unsafe {
                let off = il * ROUTE_MIRROR_STRIDE_WORDS * 4;
                let rb = &self.backend.pinned_as_mut_slice(host)?[off..off + ROUTE_WORDS * 4];
                for k in 0..8 {
                    topk_ids_u16[k] = u32::from_le_bytes(rb[k * 4..(k + 1) * 4].try_into().unwrap()) as u16;
                    topk_w_f32[k] = f32::from_le_bytes(rb[32 + k * 4..32 + (k + 1) * 4].try_into().unwrap());
                }
            }
            self.layer_planned[il] = true;
            let plan = self.tide.scheduler.plan_layer_dispatch(phys_id, &topk_ids_u16, &topk_w_f32);
            self.profile.rescue_plan_ms += tp.elapsed().as_secs_f64() * 1000.0;
            self.profile.rescue_checks += 1;
            let rerun = if !plan.sync_transfers.is_empty() {
                // The rescue copy is ordered after the queued experts; compute waits for it.
                self.tide.execute_transfers(&plan.sync_transfers, true, false)?;
                true
            } else if plan.wait_pending {
                self.tide.wait_and_publish_all_pending()?;
                true
            } else {
                false
            };
            if rerun {
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
            }
        }
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

    /// Writes `tok{pos}.final.bin`: the final normed hidden (`[H]` f32, the lm_head input) of
    /// position `pos` - the drafter's last tap.
    pub(crate) fn write_dump_final(&self, pos: usize, row: DeviceBuf) -> Result<()> {
        let h = self.cfg.hidden_size;
        let mut hb = vec![0u8; h * 4];
        self.backend.memcpy_d2h_sync(&mut hb, row.slice(0, h * 4), 0, self.tide.compute_stream)?;
        let d = self.dump.as_ref().unwrap();
        std::fs::write(d.dir.join(format!("tok{pos}.final.bin")), hb)
            .map_err(|e| smalltide_core::SmallTideError::Backend { backend: "dump", message: e.to_string() })
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
        self.kv_ensure(self.active_lane, pos + 1)?;

        // Token boundary: fold the previous token's routing log into heat, then publish any async
        // promotions whose DMA has landed (non-blocking query; table upload is stream-ordered).
        self.consume_route_log()?;
        let published = self.tide.poll_pending()?;
        self.tide.scheduler.telemetry.stage1_async_published += published as u64;
        // When GPU-side rescue waits are active and an async/rebalance batch is still in flight on
        // copy_stream (e.g. Token 0 after rebalance_after_prefill or Token 65 after periodic
        // rebalance), drain & publish it via hipStreamWaitEvent on compute_stream BEFORE queueing
        // any WAIT_REG_MEM so compute_stream never stalls in WAIT_REG_MEM behind a large rebalance
        // batch and all newly loaded experts become immediate hits on this token.
        let use_wait_this_token = !is_prefill
            && self.dump.is_none()
            && matches!(self.tide.scheduler.tide_cfg.rescue_policy, smalltide_core::RescuePolicy::Threshold(_))
            && std::env::var_os("SMALLTIDE_RESCUE_SERIAL").is_none()
            && self.rescue_wait_mode()? != RescueWaitMode::Off;
        if use_wait_this_token && self.tide.pending_batch_count() > 0 {
            let pub2 = self.tide.wait_and_publish_all_pending()?;
            self.tide.scheduler.telemetry.stage1_async_published += pub2 as u64;
        }
        self.tide.scheduler.begin_token();
        self.wait_ticket = self.wait_ticket.wrapping_add(1).max(1);
        if let Some((_, phost)) = self.rescue_patches {
            let n_log = self.layer_planned.len();
            unsafe {
                let p = self.backend.pinned_as_mut_slice(phost)?.as_mut_ptr() as *mut u32;
                for il in 0..n_log {
                    std::ptr::write_volatile(p.add(il * PATCH_STRIDE_WORDS), 0);
                }
            }
        }
        self.pending_checks.clear();
        self.layer_planned.iter_mut().for_each(|p| *p = false);
        self.layer_deferred.iter_mut().for_each(|p| *p = false);
        self.dump_token = (token_id, pos);
        self.map_last_tok = (!is_prefill).then_some((self.active_lane, pos, token_id));

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

        // The previous token's deferred Tide work runs here, overlapping the GPU's prelude - unless
        // this pass has GPU-side rescue waits: its copies fence on the compute stream, which would
        // then sit behind an unreleased wait (deadlock with a rescue's copy-stream sync). Those
        // passes run it after their checks are serviced.
        if self.pending_checks.is_empty() {
            self.run_deferred_tide_step()?;
        }

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
        if self.dump.is_some() {
            self.write_dump_final(pos, self.scratch.moe_norm)?;
        }
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
        if !self.pending_checks.is_empty() {
            self.service_rescue_checks(pos)?;
            self.run_deferred_tide_step()?;
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
        let logits = self.scratch.logits;
        self.sample_logits(logits, params, recent_tokens)
    }

    /// Samples from `logits` (`[unpadded_vocab]` f32 on the device) with this lane's RNG.
    pub(crate) fn sample_logits(
        &mut self,
        logits_buf: DeviceBuf,
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
                logits_buf,
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
            .memcpy_d2h_sync(logits_bytes, logits_buf, 0, s)?;

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
        if let Some((v, h, ev)) = self.route_pinned.take() {
            let _ = self.backend.destroy_event(ev);
            let _ = self.backend.free_device(v);
            let _ = self.backend.free_pinned(h);
        }
        for pair in [self.wait_flags.take(), self.rescue_patches.take(), self.rescue_full_table.take(), self.rescue_probe.take()] {
            if let Some((v, h)) = pair {
                let _ = self.backend.free_device(v);
                let _ = self.backend.free_pinned(h);
            }
        }
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

const FILTER_CAP: usize = 256;
/// Layout: `[out: 8][row_max: 8][scratch: 8*32][counts: 8][idx: 8*256][val: 8*256]` (u32/f32 words).
const ARGMAX_FILTER_BUF_BYTES: usize =
    (MAX_BATCH_ROWS * 40 + MAX_BATCH_ROWS + 2 * MAX_BATCH_ROWS * FILTER_CAP) * 4;

fn finalize_probs(mut indexed: Vec<(u32, f32)>, k: usize, top_p: f32) -> Vec<(u32, f32)> {
    if indexed.len() > k {
        indexed.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        indexed.truncate(k);
    }
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
        if cum >= top_p {
            cut = i + 1;
            break;
        }
    }
    indexed.truncate(cut);
    let renorm: f32 = indexed.iter().map(|(_, p)| *p).sum();
    let inv_renorm = 1.0 / renorm.max(1e-12);
    for (_, p) in &mut indexed {
        *p *= inv_renorm;
    }
    indexed
}

#[inline]
fn next_rand_f32(rng: &mut u64) -> f32 {
    *rng ^= *rng >> 12;
    *rng ^= *rng << 25;
    *rng ^= *rng >> 27;
    (((*rng).wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / ((1u32 << 24) as f32)
}

/// Builds the exact filtered, normalized `(token_id, prob)` target distribution for one logit row,
/// matching `sample_logits` (`repetition_penalty` -> `/ temperature` -> `top_k` -> softmax -> `top_p`).
fn row_target_probs(
    logits: &mut [f32],
    params: &SamplingParams,
    recent_tokens: &[u32],
    extra_recent: &[u32],
) -> Vec<(u32, f32)> {
    let vocab = logits.len();
    if (params.repetition_penalty - 1.0).abs() > 1e-6 {
        let rep = params.repetition_penalty;
        for &t in recent_tokens.iter().chain(extra_recent.iter()) {
            if (t as usize) < vocab {
                let v = &mut logits[t as usize];
                if *v > 0.0 { *v /= rep; } else { *v *= rep; }
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
        return vec![(best_i as u32, 1.0)];
    }

    let inv_t = 1.0 / params.temperature;
    let k = params.top_k.clamp(1, vocab);
    let mut max_raw = f32::NEG_INFINITY;
    for &v in logits.iter() {
        if v > max_raw {
            max_raw = v;
        }
    }
    // Fast threshold prefilter: tokens > 25*T below max have exp(-25) ~ 1.4e-11 weight.
    let thresh = max_raw - 25.0 * params.temperature;
    let mut indexed: Vec<(u32, f32)> = Vec::with_capacity(k.min(256));
    for (i, &v) in logits.iter().enumerate() {
        if v >= thresh {
            indexed.push((i as u32, v * inv_t));
        }
    }
    if indexed.len() < k {
        indexed.clear();
        indexed.reserve(vocab);
        for (i, &v) in logits.iter().enumerate() {
            indexed.push((i as u32, v * inv_t));
        }
    }
    finalize_probs(indexed, k, params.top_p)
}

fn sample_from_probs(probs: &[(u32, f32)], rng: &mut u64) -> u32 {
    let total: f32 = probs.iter().map(|(_, p)| *p).sum();
    let target = next_rand_f32(rng) * total;
    let mut acc = 0.0f32;
    for &(id, p) in probs {
        acc += p;
        if acc >= target {
            return id;
        }
    }
    probs[0].0
}

/// Exact lossless speculative verification over `block.len()` flat logit rows (`[m][vocab]`).
/// Returns `(accepted_drafts, next_token)`.
pub(crate) fn verify_sample_rows(
    all_logits: &mut [f32],
    vocab: usize,
    block: &[u32],
    params: &SamplingParams,
    recent_tokens: &[u32],
    rng: &mut u64,
) -> (usize, u32) {
    let m = block.len();
    let n = m.saturating_sub(1);
    for c in 0..n {
        let row = &mut all_logits[c * vocab..(c + 1) * vocab];
        let mut probs = row_target_probs(row, params, recent_tokens, &block[1..=c]);
        let draft = block[c + 1];
        if params.temperature <= 0.0 {
            let best = probs[0].0;
            if best == draft {
                continue;
            }
            return (c, best);
        }
        let p_d = probs.iter().find(|&&(id, _)| id == draft).map(|&(_, p)| p).unwrap_or(0.0);
        let u = next_rand_f32(rng);
        if u < p_d {
            continue;
        }
        // Reject `draft`: sample from residual `max(0, p(y) - q(y))` = `p \ {draft}` (since `q(draft) = 1`).
        probs.retain(|&(id, _)| id != draft);
        if probs.is_empty() {
            return (c, draft);
        }
        return (c, sample_from_probs(&probs, rng));
    }
    // All `n` drafts accepted: sample the bonus token from row `n`.
    let bonus_row = &mut all_logits[n * vocab..(n + 1) * vocab];
    let probs = row_target_probs(bonus_row, params, recent_tokens, &block[1..=n]);
    let bonus = if params.temperature <= 0.0 {
        probs[0].0
    } else {
        sample_from_probs(&probs, rng)
    };
    (n, bonus)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_sample_exact_target_distribution() {
        // Verify that rejection sampling with deterministic draft `q(d) = 1` reproduces the
        // exact target probability distribution `p(y)` for every token (accepted OR rejected).
        let base_logits = [2.0f32, 1.5, 1.0, 0.2, -1.0, -3.0];
        let vocab = base_logits.len();
        let params = SamplingParams {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 4,
            repetition_penalty: 1.0,
            seed: 42,
        };
        let mut ref_logits = base_logits;
        let target_probs = row_target_probs(&mut ref_logits, &params, &[], &[]);

        for draft in [0u32, 1, 2, 5] {
            let mut rng = 0x1234_5678_9ABC_DEF1u64 ^ (draft as u64 + 1);
            let mut counts = vec![0usize; vocab];
            let trials = 60_000usize;
            let mut buf = vec![0.0f32; 2 * vocab];
            for _ in 0..trials {
                buf[..vocab].copy_from_slice(&base_logits);
                buf[vocab..].copy_from_slice(&base_logits);
                let (acc, next_tok) = verify_sample_rows(&mut buf, vocab, &[99, draft], &params, &[], &mut rng);
                let emitted = if acc == 1 { draft } else { next_tok };
                counts[emitted as usize] += 1;
            }
            for i in 0..vocab {
                let expected = target_probs.iter().find(|&&(id, _)| id == i as u32).map(|&(_, p)| p).unwrap_or(0.0);
                let empirical = counts[i] as f32 / trials as f32;
                assert!(
                    (empirical - expected).abs() < 0.01,
                    "draft={draft} tok={i}: empirical={empirical:.4} vs expected={expected:.4}"
                );
            }
        }
    }
}
