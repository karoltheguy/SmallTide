//! Backend-agnostic 2-stage runtime executor (`TideExecutor<B: ComputeBackend>`).
//!
//! Executes `TransferOp` plans emitted by `TideScheduler`:
//! - Stage 2 Async (`DiskToRam`): offloaded to dedicated `Stage2IoWorker` OS thread (`NO_BUFFERING`)
//!   so NVMe reads overlap with GPU decode compute without blocking the token loop.
//! - Stage 1 (`RamToVram`): copies RAM slab -> bounded pinned staging ring -> async H2D DMA -> VRAM
//! - Direct (`DiskToVramDirect`): blocking read via `Stage2IoWorker::read_sync` -> staging -> VRAM
//! - Transient (`ScratchRescue`): copies a cold 1-off sync miss into a tail Scratch Rescue Ring slot.
//!
//! ## Ordering invariants (the "async promotion race" fix)
//! 1. The GPU slot table is only ever written on the **compute stream** (stream-ordered with the
//!    MoE kernels that read it), from a small ring of dedicated pinned buffers.
//! 2. Before the copy stream overwrites any VRAM slot it waits on a compute-stream event recorded
//!    *after* the table upload that unmapped the slot's previous occupant, so no in-flight or
//!    later kernel can read a slot while it is being rewritten.
//! 3. A newly copied expert is published in the table only once its DMA is known complete:
//!    - sync path: the compute stream waits on the copy stream's event (GPU-side), then publishes;
//!    - async path: the CPU polls the batch's event (`poll_pending`) at the next token boundary.
//!    Until then the cache keeps it `vram_pending` and the scheduler never counts it as a hit.

use crate::cache::{ExpertClass, TransferOp};
use crate::io_worker::Stage2IoWorker;
use crate::scheduler::TideScheduler;
use smalltide_core::{
    ComputeBackend, DType, DeviceBuf, Event, ExpertId, Inkling2Config, PinnedBuf, QueueKind,
    Result, SlotIndex, SmallTideError, Stream, TideConfig,
};
use smalltide_io::stide_pack::{ExpertSlabLayout, StidePackReader};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

pub const DEFAULT_SCRATCH_RING_SLOTS: usize = 8;

/// Slots of the rescue ring of each pool, `(nvfp4, mxfp8)`: `SMALLTIDE_RESCUE_RING_NVFP4` and
/// `SMALLTIDE_RESCUE_RING_MXFP8`, 8 to 1024, default 8. An inline rescue copies its expert into
/// the next ring slot; the expert then counts as on the card until the ring has gone round, so a
/// larger ring saves the repeated rescues of the same experts at the cost of resident slots (a
/// ring slot is as large as a pool slot). Never below 8: a layer rescues up to 8 experts in a step.
pub fn rescue_ring_slots() -> (usize, usize) {
    static RING: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
    *RING.get_or_init(|| {
        let get = |k: &str| {
            std::env::var(k).ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(DEFAULT_SCRATCH_RING_SLOTS).clamp(DEFAULT_SCRATCH_RING_SLOTS, 1024)
        };
        (get("SMALLTIDE_RESCUE_RING_NVFP4"), get("SMALLTIDE_RESCUE_RING_MXFP8"))
    })
}
const TABLE_UPLOAD_RING: usize = 8;

struct TableUploadBuf {
    pinned: PinnedBuf,
    done: Event,
    used: bool,
}

const STAGE_KICK_DEFAULT: bool = true;

/// A RAM-tier expert copy in page-locked memory (`SMALLTIDE_RAM_PINNED_GB`): its upload to VRAM
/// goes straight from this buffer, without the copy into the staging ring that otherwise takes
/// about 0.2 of the 0.45 ms a rescued expert costs. One buffer per RAM slot, allocated on first
/// use while the page-locked budget lasts, and kept for the slot's later occupants.
struct PinnedSlab {
    /// The expert whose bytes the buffer holds now (None: nothing valid).
    owner: Option<ExpertId>,
    buf: PinnedBuf,
    len: usize,
    /// `uploads_issued` at the last upload queued from this buffer.
    upload_seq: u64,
}

pub struct TideExecutor<B: ComputeBackend> {
    pub backend: Arc<B>,
    pub scheduler: TideScheduler,
    pub nvfp4_layout: ExpertSlabLayout,
    pub mxfp8_layout: ExpertSlabLayout,

    /// Tier 0 VRAM pools: `[0..resident_slots]` resident + `[resident_slots..+scratch]` transient scratch ring.
    pub vram_pool_nvfp4: DeviceBuf,
    pub vram_pool_mxfp8: DeviceBuf,
    /// GPU-side indirection table (`[num_physical_layers * n_routed_experts]` u32).
    pub gpu_slot_table: DeviceBuf,

    /// Bounded page-locked DMA staging ring (read only by the copy stream).
    pinned_staging: PinnedBuf,
    staging_cursor: usize,

    /// Ring of dedicated pinned buffers for compute-stream slot-table uploads.
    table_ring: Vec<TableUploadBuf>,
    table_ring_cursor: usize,

    /// Tier 1 Host RAM warm expert slabs, keyed by `(class, ram_slot)` and tagged with the
    /// expert they hold so a re-used slot can never silently feed stale bytes to a DMA.
    ram_slabs: HashMap<(ExpertClass, SlotIndex), (ExpertId, Vec<u8>)>,
    /// Page-locked part of the RAM tier (see `PinnedSlab`): byte budget, bytes allocated, buffers.
    /// A RAM copy is in `pinned_slabs` or in `ram_slabs`, never both.
    ram_pinned_budget: usize,
    ram_pinned_bytes: usize,
    pinned_slabs: HashMap<(ExpertClass, SlotIndex), PinnedSlab>,
    /// Uploads queued straight from page-locked RAM copies, and how many of them the host has
    /// since seen land (a copy-stream sync): a buffer with a later upload is not overwritten.
    uploads_issued: u64,
    uploads_synced: u64,
    /// Experts uploaded straight from a page-locked RAM copy.
    pub pinned_uploads: u64,
    /// Dev: (rescues, host ms issuing (RAM -> staging copies), ms waiting for the copy stream).
    pub rescue_stats: (u64, f64, f64),
    /// The last `execute_rescue_now`: ms filling the staging buffer (RAM / disk reads), ms waiting for the copies to land.
    pub last_rescue_ms: (f64, f64),
    /// Seconds spent copying experts from the RAM tier into the staging buffer, and how many
    /// (rescues, promotions and prefill staging alike): the largest host-side cost of a rescue.
    pub stage_copy: (f64, u64),
    /// Rescue batches by size (1, 2, 3, 4 or more experts): (batches, staging ms, wait ms).
    pub rescue_hist: [(u64, f64, f64); 4],
    /// Dev: (seconds, experts, of which had no RAM slot at all) read from disk while staging.
    pub stage_disk: (f64, u64, u64),
    /// Start a staged expert's transfer while the next one is staged (`SMALLTIDE_STAGE_KICK=0`
    /// turns it off), see `kick_copy_stream`.
    stage_kick: bool,
    kick_event: Event,
    /// Dedicated background Stage 2 NVMe I/O worker thread.
    stage2_worker: Option<Stage2IoWorker>,

    /// Async promotion batches whose DMA may still be in flight (copy-stream order).
    pending_batches: VecDeque<(Event, Vec<(ExpertId, SlotIndex)>)>,
    free_events: Vec<Event>,
    /// Longest an inline rescue keeps the GPU inside its wait kernel for the copy to land
    /// (`SMALLTIDE_RESCUE_HOLD_MS`, default 250). Windows resets a GPU whose engine does not
    /// answer for about 2 s (hipError_t=719), and a copy to VRAM has been seen to take 2.6 s.
    rescue_hold_limit: std::time::Duration,
    /// A rescue's copy outlived the hold limit and has not landed yet (since when; the event is
    /// recorded behind it on the copy stream). Inline rescues pause until it lands.
    copy_stall: Option<(std::time::Instant, Event)>,
    /// Rescues whose copy outlived the hold limit, since start.
    pub rescue_hold_timeouts: u64,

    pub compute_stream: Stream,
    pub copy_stream: Stream,
    pub dma_event: Event,
    /// Resident slots (excluding the scratch ring) the full pools are allocated with.
    pool_slots: (usize, usize),
    pools_allocated: bool,
    /// When set, the MXFP8 pool lives in pinned, device-mapped host memory (zero-copy over
    /// PCIe) and slabs are written by CPU copies instead of DMA.
    mxfp8_host_mapped: bool,
    mxfp8_host: Option<PinnedBuf>,
    /// Recorded on the compute stream after a table upload that unmaps slots about to be reused.
    compute_fence: Event,
}

impl<B: ComputeBackend> TideExecutor<B> {
    pub fn new(
        backend: Arc<B>,
        model_cfg: Inkling2Config,
        tide_cfg: TideConfig,
        nvfp4_vram_slots: usize,
        mxfp8_vram_slots: usize,
        ram_budget_bytes: usize,
        disk_pack: Option<StidePackReader>,
    ) -> Result<Self> {
        let nvfp4_layout = ExpertSlabLayout::for_inkling2(&model_cfg, DType::NVFP4);
        let mxfp8_layout = ExpertSlabLayout::for_inkling2(&model_cfg, DType::MXFP8);

        let resident_nvfp4 = nvfp4_vram_slots.max(1);
        let resident_mxfp8 = mxfp8_vram_slots.max(1);
        let (ring4, ring8) = rescue_ring_slots();

        // Resident budget excludes the tail Scratch Rescue Ring slots
        let unified_resident_vram_bytes = resident_nvfp4 * nvfp4_layout.aligned_slab_bytes
            + resident_mxfp8 * mxfp8_layout.aligned_slab_bytes;

        // Placeholder pools: the real ones are allocated by `allocate_pools` once the engine's
        // hot buffers (activations, KV, scratch) exist. On Windows the driver backs whatever does
        // not fit the resident local budget with system memory, in allocation order, so the
        // large expert pools must come last (and fill hottest experts first).
        let vram_pool_nvfp4 = backend.alloc_device((1 + ring4) * nvfp4_layout.aligned_slab_bytes)?;
        let vram_pool_mxfp8 = backend.alloc_device((1 + ring8) * mxfp8_layout.aligned_slab_bytes)?;

        let total_experts = model_cfg.num_physical_layers * model_cfg.n_routed_experts;
        let gpu_slot_table = backend.alloc_device(total_experts * 4)?;

        let staging_bytes = tide_cfg.pinned_staging_mib * 1024 * 1024;
        let pinned_staging = backend.alloc_pinned(staging_bytes.max(16 * 1024 * 1024))?;

        let mut table_ring = Vec::with_capacity(TABLE_UPLOAD_RING);
        for _ in 0..TABLE_UPLOAD_RING {
            table_ring.push(TableUploadBuf {
                pinned: backend.alloc_pinned(total_experts * 4)?,
                done: backend.create_event()?,
                used: false,
            });
        }

        let compute_stream = backend.create_stream(QueueKind::Compute)?;
        let copy_stream = backend.create_stream(QueueKind::AsyncCopyH2D)?;
        let dma_event = backend.create_event()?;
        let kick_event = backend.create_event()?;
        let compute_fence = backend.create_event()?;

        let max_slab = nvfp4_layout
            .aligned_slab_bytes
            .max(mxfp8_layout.aligned_slab_bytes);
        let stage2_worker = match disk_pack {
            Some(pack) => Some(Stage2IoWorker::spawn(pack, max_slab)?),
            None => None,
        };

        let mut scheduler =
            TideScheduler::new(model_cfg, tide_cfg, unified_resident_vram_bytes, ram_budget_bytes);
        scheduler.cache.configure_scratch_rings(resident_nvfp4 as u32, resident_mxfp8 as u32, ring4 as u32, ring8 as u32);

        let mut exec = Self {
            backend,
            scheduler,
            nvfp4_layout,
            mxfp8_layout,
            vram_pool_nvfp4,
            vram_pool_mxfp8,
            gpu_slot_table,
            pinned_staging,
            staging_cursor: 0,
            table_ring,
            table_ring_cursor: 0,
            ram_slabs: HashMap::new(),
            ram_pinned_budget: 0,
            ram_pinned_bytes: 0,
            pinned_slabs: HashMap::new(),
            uploads_issued: 0,
            uploads_synced: 0,
            pinned_uploads: 0,
            rescue_stats: (0, 0.0, 0.0),
            last_rescue_ms: (0.0, 0.0),
            stage_copy: (0.0, 0),
            rescue_hist: [(0, 0.0, 0.0); 4],
            stage_disk: (0.0, 0, 0),
            stage_kick: std::env::var("SMALLTIDE_STAGE_KICK").map(|v| v.trim() != "0").unwrap_or(STAGE_KICK_DEFAULT),
            kick_event,
            stage2_worker,
            pending_batches: VecDeque::new(),
            free_events: Vec::new(),
            rescue_hold_limit: std::time::Duration::from_millis(
                std::env::var("SMALLTIDE_RESCUE_HOLD_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(250),
            ),
            copy_stall: None,
            rescue_hold_timeouts: 0,
            compute_stream,
            copy_stream,
            dma_event,
            compute_fence,
            pool_slots: (resident_nvfp4, resident_mxfp8),
            pools_allocated: false,
            mxfp8_host_mapped: false,
            mxfp8_host: None,
        };
        exec.upload_table_on_compute()?;
        Ok(exec)
    }

    /// Allocates the full NVFP4 / MXFP8 expert pools (idempotent). Call after every other
    /// long-lived device buffer exists and before the first transfer.
    pub fn allocate_pools(&mut self) -> Result<()> {
        if self.pools_allocated {
            return Ok(());
        }
        let (ring4, ring8) = rescue_ring_slots();
        self.backend.sync_stream(self.compute_stream)?;
        self.backend.free_device(self.vram_pool_nvfp4)?;
        self.backend.free_device(self.vram_pool_mxfp8)?;
        // MXFP8 first: it is the smaller pool and (with priors) its experts carry the most
        // routing mass per byte. Windows places the first-allocated memory in VRAM; whatever
        // follows past the resident cap lands in system memory (see `fit_pools_to_resident`).
        let mx_bytes = (self.pool_slots.1 + ring8) * self.mxfp8_layout.aligned_slab_bytes;
        if self.mxfp8_host_mapped {
            let (view, host) = self.backend.alloc_host_mapped(mx_bytes)?;
            self.vram_pool_mxfp8 = view;
            self.mxfp8_host = Some(host);
        } else {
            self.vram_pool_mxfp8 = self.backend.alloc_device(mx_bytes)?;
        }
        self.vram_pool_nvfp4 = self
            .backend
            .alloc_device((self.pool_slots.0 + ring4) * self.nvfp4_layout.aligned_slab_bytes)?;
        self.pools_allocated = true;
        Ok(())
    }

    /// Re-sizes the expert pools. Only valid before the first expert transfer (pools empty).
    pub fn resize_pools(&mut self, nvfp4_slots: usize, mxfp8_slots: usize) -> Result<()> {
        let (pre, core, coda) = self.scheduler.cache.count_vram_resident();
        if pre + core + coda != 0 {
            return Err(smalltide_core::SmallTideError::Backend {
                backend: "tide",
                message: "resize_pools after experts were loaded".into(),
            });
        }
        let (n4, m8) = (nvfp4_slots.max(1), mxfp8_slots.max(1));
        let (ring4, ring8) = rescue_ring_slots();
        // Free the current pools first so the new ones can take their place.
        self.backend.sync_stream(self.compute_stream)?;
        self.backend.free_device(self.vram_pool_nvfp4)?;
        self.backend.free_device(self.vram_pool_mxfp8)?;
        self.vram_pool_nvfp4 = self.backend.alloc_device((1 + ring4) * self.nvfp4_layout.aligned_slab_bytes)?;
        self.vram_pool_mxfp8 = self.backend.alloc_device((1 + ring8) * self.mxfp8_layout.aligned_slab_bytes)?;
        self.pool_slots = (n4, m8);
        self.scheduler.cache.set_unified_vram_budget(
            n4 * self.nvfp4_layout.aligned_slab_bytes + m8 * self.mxfp8_layout.aligned_slab_bytes,
        );
        self.scheduler.cache.configure_scratch_rings(n4 as u32, m8 as u32, ring4 as u32, ring8 as u32);
        self.pools_allocated = false;
        self.allocate_pools()
    }

    pub fn pool_slots(&self) -> (usize, usize) {
        self.pool_slots
    }

    /// Bytes at the start of `pool` that read at VRAM speed, scanning `win`-sized windows
    /// (device copy of the first half of each window onto its second half).
    fn fast_prefix_bytes(&self, pool: DeviceBuf, win: usize) -> Result<(usize, f64, f64)> {
        const REPS: usize = 6;
        const SLOW_GBS: f64 = 100.0;
        let s = self.compute_stream;
        let half = win / 2;
        let (mut best, mut worst) = (0.0f64, f64::INFINITY);
        let mut off = 0usize;
        while off + win <= pool.size_bytes {
            let src = pool.slice(off, half);
            let dst = pool.slice(off + half, half);
            self.backend.vec_copy(dst, src, half / 4, s)?;
            self.backend.sync_stream(s)?;
            let t0 = std::time::Instant::now();
            for _ in 0..REPS {
                self.backend.vec_copy(dst, src, half / 4, s)?;
            }
            self.backend.sync_stream(s)?;
            let gbs = (2 * REPS * half) as f64 / t0.elapsed().as_secs_f64() / 1e9;
            best = best.max(gbs);
            worst = worst.min(gbs);
            if gbs < SLOW_GBS {
                return Ok((off, best, gbs));
            }
            off += win;
        }
        Ok((pool.size_bytes, best, worst))
    }

    /// Caps each device pool's usable slots to the part the OS keeps in VRAM (measured), so no
    /// expert ever lives in system-memory-backed "VRAM". The slow tail of an allocation is simply
    /// left unused; the scratch ring follows the usable slots and so stays fast too.
    /// Returns `(nvfp4_slots, mxfp8_slots, log line)`. Only valid before the first transfer.
    pub fn fit_pools_to_resident(&mut self) -> Result<(usize, usize, String)> {
        let (ring4, ring8) = rescue_ring_slots();
        let win = 256usize << 20;
        let (n4, m8) = self.pool_slots;
        let s4 = self.nvfp4_layout.aligned_slab_bytes;
        let s8 = self.mxfp8_layout.aligned_slab_bytes;
        let mut notes = Vec::new();
        let m8_fit = if self.mxfp8_host_mapped {
            m8
        } else {
            let (fast, _best, worst) = self.fast_prefix_bytes(self.vram_pool_mxfp8, win.min(self.vram_pool_mxfp8.size_bytes & !0xFFFF))?;
            let fit = (fast / s8).saturating_sub(ring8).min(m8);
            if fit < m8 {
                notes.push(format!("mxfp8 fast {:.2} GiB (then {worst:.0} GB/s)", fast as f64 / (1u64 << 30) as f64));
            }
            fit
        };
        let (fast4, best4, worst4) = self.fast_prefix_bytes(self.vram_pool_nvfp4, win)?;
        let n4_fit = (fast4 / s4).saturating_sub(ring4).min(n4);
        if n4_fit < n4 {
            notes.push(format!("nvfp4 fast {:.2} GiB (then {worst4:.0} GB/s)", fast4 as f64 / (1u64 << 30) as f64));
        }
        let (n4_fit, m8_fit) = (n4_fit.max(1), m8_fit.max(1));
        if (n4_fit, m8_fit) != (n4, m8) {
            self.pool_slots = (n4_fit, m8_fit);
            self.scheduler.cache.set_unified_vram_budget(n4_fit * s4 + m8_fit * s8);
            self.scheduler.cache.configure_scratch_rings(n4_fit as u32, m8_fit as u32, ring4 as u32, ring8 as u32);
        }
        let line = if notes.is_empty() {
            format!("[pools] residency check: both pools fully at VRAM speed (~{best4:.0} GB/s copy)")
        } else {
            format!(
                "[pools] residency check: {} -> usable slots nvfp4 {n4} -> {n4_fit}, mxfp8 {m8} -> {m8_fit} (the slow tail stays unused)",
                notes.join(", ")
            )
        };
        Ok((n4_fit, m8_fit, line))
    }

    /// Places the MXFP8 expert pool in pinned, device-mapped host memory (call before
    /// `allocate_pools`). Useful when device memory beyond the resident budget is backed by
    /// system memory anyway: the pool can then hold every MXFP8 expert, so those layers never
    /// miss (no per-layer host sync, no rescue copies, exact routing).
    pub fn set_mxfp8_host_mapped(&mut self, on: bool) {
        assert!(!self.pools_allocated, "set_mxfp8_host_mapped after allocate_pools");
        self.mxfp8_host_mapped = on;
    }

    pub fn mxfp8_is_host_mapped(&self) -> bool {
        self.mxfp8_host_mapped
    }

    /// Writes one slab from the staging ring into its pool slot: async DMA on the copy stream
    /// for device pools, a CPU copy for the host-mapped MXFP8 pool.
    fn upload_slab(&mut self, expert: ExpertId, dst_pool: DeviceBuf, dst_off: usize, src_offset: usize, len: usize) -> Result<()> {
        if dst_off + len > dst_pool.size_bytes {
            return Err(SmallTideError::Backend {
                backend: "tide",
                message: format!(
                    "upload_slab OOB: expert={expert:?} dst_off={dst_off} len={len} pool_size={}",
                    dst_pool.size_bytes
                ),
            });
        }
        unsafe {
            let staging = self.backend.pinned_as_mut_slice(self.pinned_staging)?;
            if len >= 28 {
                let h = &staging[src_offset..src_offset + 28];
                let w13_w = u32::from_le_bytes(h[8..12].try_into().unwrap()) as usize;
                let w13_s = u32::from_le_bytes(h[12..16].try_into().unwrap()) as usize;
                let w2_w = u32::from_le_bytes(h[16..20].try_into().unwrap()) as usize;
                let w2_s = u32::from_le_bytes(h[20..24].try_into().unwrap()) as usize;
                let tag = u32::from_le_bytes(h[24..28].try_into().unwrap());
                if tag > 1 || w13_w >= len || w13_s >= len || w2_w >= len || w2_s >= len {
                    return Err(SmallTideError::Backend {
                        backend: "tide",
                        message: format!(
                            "upload_slab corrupt header for {expert:?}: tag={tag} w13=({w13_w},{w13_s}) w2=({w2_w},{w2_s}) len={len}"
                        ),
                    });
                }
            }
        }
        if self.mxfp8_host_mapped && self.scheduler.cache.expert_class(expert) == ExpertClass::Mxfp8 {
            let host = self.mxfp8_host.expect("mxfp8 host pool");
            unsafe {
                let staging = self.backend.pinned_as_mut_slice(self.pinned_staging)?;
                let src = staging[src_offset..src_offset + len].as_ptr();
                let dst = self.backend.pinned_as_mut_slice(host)?;
                std::ptr::copy_nonoverlapping(src, dst[dst_off..dst_off + len].as_mut_ptr(), len);
            }
            Ok(())
        } else {
            self.backend.memcpy_h2d_async(dst_pool, dst_off, self.pinned_staging, src_offset, len, self.copy_stream)
        }
    }

    /// Makes the runtime hand the copy stream's queued transfers to the driver now. Without it
    /// they start only when something waits for the stream: the wait after a rescue batch of
    /// 1, 2, 3 experts was 0.25, 0.49, 0.81 ms, the full transfer time each. An event query
    /// flushes the queue; it costs about 0.05 ms, so it is used only when more experts follow.
    fn kick_copy_stream(&mut self) -> Result<()> {
        self.backend.record_event(self.kick_event, self.copy_stream)?;
        let _ = self.backend.query_event(self.kick_event)?;
        Ok(())
    }

    /// Non-blocking drain of completed background Stage 2 NVMe -> RAM reads.
    /// Call once per decode token before `scheduler.plan_async_step()`!
    pub fn poll_stage2_completions(&mut self) -> Result<usize> {
        let Some(worker) = self.stage2_worker.as_mut() else {
            return Ok(0);
        };
        let completed = worker.poll_completed();
        let count = completed.len();
        for item in completed {
            let bytes = item.payload?;
            // Only accept the read if the cache still assigns this RAM slot to this expert.
            let e = self.scheduler.cache.entry(item.expert);
            if e.ram_slot == item.ram_slot {
                self.ram_put(item.class, item.ram_slot, item.expert, bytes)?;
                self.scheduler.cache.mark_ram_ready(item.expert);
            }
        }
        Ok(count)
    }

    /// Number of async promotion batches whose DMA has not yet been observed complete.
    pub fn pending_batch_count(&self) -> usize {
        self.pending_batches.len()
    }

    /// Non-blocking: publishes async promotions whose DMA has completed. Call at token boundaries
    /// (before the next forward). Returns the number of experts published.
    pub fn poll_pending(&mut self) -> Result<usize> {
        let mut published = 0usize;
        while let Some((ev, _)) = self.pending_batches.front() {
            if !self.backend.query_event(*ev)? {
                break;
            }
            let (ev, ids) = self.pending_batches.pop_front().unwrap();
            for (id, slot) in ids {
                if self.scheduler.cache.publish_vram(id, slot) {
                    published += 1;
                }
            }
            self.free_events.push(ev);
        }
        if published > 0 {
            self.upload_table_on_compute()?;
        }
        Ok(published)
    }

    /// GPU-side wait for every outstanding copy, then publish all pending experts. Used when a
    /// layer needs an expert whose async DMA is still in flight (no CPU stall).
    pub fn wait_and_publish_all_pending(&mut self) -> Result<usize> {
        self.backend.record_event(self.dma_event, self.copy_stream)?;
        self.backend
            .stream_wait_event(self.compute_stream, self.dma_event)?;
        while let Some((ev, _)) = self.pending_batches.pop_front() {
            self.free_events.push(ev);
        }
        let n = self.scheduler.cache.publish_all_pending();
        if n > 0 || self.scheduler.cache.has_dirty_gpu_table() {
            self.upload_table_on_compute()?;
        }
        Ok(n)
    }

    /// Executes a batch of Stage 1 and Stage 2 `TransferOp` commands.
    ///
    /// * `sync_compute_after = true`: the compute stream waits (GPU-side) for the copies and the
    ///   experts are published immediately — the next kernel on the compute stream sees them.
    /// * `sync_compute_after = false`: experts stay pending and are published by `poll_pending`.
    ///
    /// When `async_disk_io` is true, `DiskToRam` ops are dispatched non-blocking to `Stage2IoWorker`.
    pub fn execute_transfers(
        &mut self,
        ops: &[TransferOp],
        sync_compute_after: bool,
        async_disk_io: bool,
    ) -> Result<()> {
        if ops.is_empty() {
            return Ok(());
        }
        self.allocate_pools()?;

        let writes_vram = ops
            .iter()
            .any(|op| !matches!(op, TransferOp::DiskToRam { .. }));
        if writes_vram {
            // Invariant 2: unmap evicted / replaced experts on the compute stream first, then make
            // the copy stream wait until every kernel that could read the old mappings is done.
            self.upload_table_on_compute()?;
            self.backend
                .record_event(self.compute_fence, self.compute_stream)?;
            self.backend
                .stream_wait_event(self.copy_stream, self.compute_fence)?;
        }

        let batch = self.issue_ops(ops, async_disk_io)?;

        if !writes_vram {
            return Ok(());
        }
        if sync_compute_after {
            // Copy stream is in-order: waiting on its latest event covers every earlier async batch
            // too, so everything pending can be published.
            self.backend.record_event(self.dma_event, self.copy_stream)?;
            self.backend
                .stream_wait_event(self.compute_stream, self.dma_event)?;
            while let Some((ev, _)) = self.pending_batches.pop_front() {
                self.free_events.push(ev);
            }
            self.scheduler.cache.publish_all_pending();
            self.upload_table_on_compute()?;
        } else if !batch.is_empty() {
            let ev = match self.free_events.pop() {
                Some(e) => e,
                None => self.backend.create_event()?,
            };
            self.backend.record_event(ev, self.copy_stream)?;
            self.pending_batches.push_back((ev, batch));
        }
        Ok(())
    }

    /// Issues the copies of `ops` (Stage 2 reads, slab uploads on the copy stream); returns the
    /// (expert, slot) pairs written into VRAM.
    fn issue_ops(&mut self, ops: &[TransferOp], async_disk_io: bool) -> Result<Vec<(ExpertId, SlotIndex)>> {
        let mut batch: Vec<(ExpertId, SlotIndex)> = Vec::new();
        for (op_i, op) in ops.iter().enumerate() {
            let more_follow = op_i + 1 < ops.len();
            match *op {
                TransferOp::DiskToRam {
                    expert,
                    ram_slot,
                    evicted_from_ram: _,
                } => {
                    let cls = self.scheduler.cache.expert_class(expert);
                    let slab_len = self.slab_len_for(expert);
                    if let Some(worker) = self.stage2_worker.as_mut() {
                        if async_disk_io {
                            worker.submit_async_read(expert, cls, ram_slot, slab_len);
                        } else {
                            let bytes = worker.read_sync(expert, slab_len)?;
                            self.ram_put(cls, ram_slot, expert, bytes)?;
                            self.scheduler.cache.mark_ram_ready(expert);
                        }
                    } else {
                        self.ram_put(cls, ram_slot, expert, vec![0u8; slab_len])?;
                        self.scheduler.cache.mark_ram_ready(expert);
                    }
                }
                TransferOp::RamToVram {
                    expert,
                    src_ram_slot,
                    dst_vram_slot,
                    evicted_from_vram: _,
                    release_ram_after_upload,
                } => {
                    let cls = self.scheduler.cache.expert_class(expert);
                    let slab_len = self.slab_len_for(expert);
                    let (dst_pool, dst_off) = self.vram_target(expert, dst_vram_slot);
                    self.stage_and_upload(expert, cls, Some(src_ram_slot), dst_pool, dst_off, slab_len, more_follow)?;
                    batch.push((expert, dst_vram_slot));
                    if release_ram_after_upload {
                        self.ram_drop(cls, src_ram_slot);
                    }
                }
                TransferOp::DiskToVramDirect {
                    expert,
                    dst_vram_slot,
                    also_keep_ram_slot,
                    evicted_from_vram: _,
                } => {
                    let cls = self.scheduler.cache.expert_class(expert);
                    let slab_len = self.slab_len_for(expert);
                    let bytes = self.read_disk_sync(expert, slab_len)?;
                    let src_offset = self.reserve_staging(slab_len)?;
                    unsafe {
                        let staging = self.backend.pinned_as_mut_slice(self.pinned_staging)?;
                        staging[src_offset..src_offset + slab_len].copy_from_slice(&bytes);
                    }
                    if let Some(rs) = also_keep_ram_slot {
                        self.ram_put(cls, rs, expert, bytes)?;
                    }
                    let (dst_pool, dst_off) = self.vram_target(expert, dst_vram_slot);
                    self.upload_slab(expert, dst_pool, dst_off, src_offset, slab_len)?;
                    batch.push((expert, dst_vram_slot));
                }
                TransferOp::ScratchRescue {
                    expert,
                    src_ram_slot,
                    scratch_vram_slot,
                } => {
                    let cls = self.scheduler.cache.expert_class(expert);
                    let slab_len = self.slab_len_for(expert);
                    let (dst_pool, dst_off) = self.vram_target(expert, scratch_vram_slot);
                    self.stage_and_upload(expert, cls, src_ram_slot, dst_pool, dst_off, slab_len, more_follow)?;
                    batch.push((expert, scratch_vram_slot));
                }
            }
        }
        Ok(batch)
    }

    /// Synchronous rescue while the compute stream is parked at a GPU-side wait right before the
    /// layer that needs it (every earlier kernel finished, no later one started): copies the slab(s)
    /// on the copy stream, waits for the copy stream, publishes pending entries in the host table
    /// mirror, and returns the dirty `(flat_idx, new_slot)` updates so the caller can apply them
    /// on the **compute stream** (preserving Invariant 1 and issuing a system-scope GL2 invalidate
    /// before the MoE kernels read the newly DMA'd slabs).
    pub fn execute_rescue_now(&mut self, ops: &[TransferOp]) -> Result<Vec<(usize, u32)>> {
        self.allocate_pools()?;
        let t0 = std::time::Instant::now();
        let batch = self.issue_ops(ops, false)?;
        let t1 = std::time::Instant::now();
        // The copy stream is in order: once an event recorded behind the copies is complete,
        // every slab DMA has landed in VRAM. The GPU sits in this layer's wait kernel meanwhile,
        // so the wait is bounded (see `rescue_hold_limit`).
        let ev = match self.free_events.pop() {
            Some(e) => e,
            None => self.backend.create_event()?,
        };
        self.backend.record_event(ev, self.copy_stream)?;
        let deadline = t1 + self.rescue_hold_limit;
        let landed = loop {
            let done = self.backend.query_event(ev).map_err(|e| SmallTideError::Backend {
                backend: "tide",
                message: format!("execute_rescue_now wait for copy_stream ({} ops): {e}", ops.len()),
            })?;
            if done {
                break true;
            }
            if std::time::Instant::now() >= deadline {
                break false;
            }
            std::hint::spin_loop();
        };
        if !landed {
            // The copy is still on its way. Nothing of it is published: the experts stay pending
            // (`poll_pending` publishes them once the copy has landed) and this step skips them.
            // The returned table changes unmap what the copies overwrite, so no kernel reads a
            // slot that is being written. Inline rescues pause until the copy stream is clear.
            self.rescue_hold_timeouts += 1;
            if !batch.is_empty() {
                self.pending_batches.push_back((ev, batch));
            } else {
                self.free_events.push(ev);
            }
            let stall_ev = match self.copy_stall.take() {
                Some((_, e)) => e,
                None => match self.free_events.pop() {
                    Some(e) => e,
                    None => self.backend.create_event()?,
                },
            };
            self.backend.record_event(stall_ev, self.copy_stream)?;
            self.copy_stall = Some((t1, stall_ev));
            if self.rescue_hold_timeouts <= 5 || self.rescue_hold_timeouts % 100 == 0 {
                eprintln!(
                    "[rescue] a copy of {} expert(s) to VRAM did not land within {} ms: the GPU wait was released and \
                     those experts are skipped for this step; inline rescues pause until the copy lands ({} such stalls since start)",
                    ops.len(),
                    self.rescue_hold_limit.as_millis(),
                    self.rescue_hold_timeouts
                );
            }
            self.last_rescue_ms = ((t1 - t0).as_secs_f64() * 1000.0, t1.elapsed().as_secs_f64() * 1000.0);
            return Ok(self.scheduler.cache.take_dirty_gpu_updates());
        }
        self.free_events.push(ev);
        self.uploads_synced = self.uploads_issued;
        self.rescue_stats.0 += 1;
        self.rescue_stats.1 += (t1 - t0).as_secs_f64() * 1000.0;
        self.rescue_stats.2 += t1.elapsed().as_secs_f64() * 1000.0;
        self.last_rescue_ms = ((t1 - t0).as_secs_f64() * 1000.0, t1.elapsed().as_secs_f64() * 1000.0);
        if !ops.is_empty() {
            let h = &mut self.rescue_hist[ops.len().min(4) - 1];
            h.0 += 1;
            h.1 += self.last_rescue_ms.0;
            h.2 += self.last_rescue_ms.1;
        }
        while let Some((ev, _)) = self.pending_batches.pop_front() {
            self.free_events.push(ev);
        }
        self.scheduler.cache.publish_all_pending();
        Ok(self.scheduler.cache.take_dirty_gpu_updates())
    }

    /// True when an inline rescue must not start now: a rescue's copy that outlived the hold limit
    /// is still on its way, or the staging buffer is full and cannot be reused yet. The engine
    /// then skips the layer's rescues (as when a block's budget is spent) instead of making the
    /// GPU wait behind copies with no time limit.
    pub fn copy_stalled(&mut self) -> bool {
        if let Some((since, ev)) = self.copy_stall {
            match self.backend.query_event(ev) {
                Ok(true) => {
                    if self.rescue_hold_timeouts <= 5 || self.rescue_hold_timeouts % 100 == 0 {
                        eprintln!("[rescue] the stalled copy landed after {:.0} ms; inline rescues resume", since.elapsed().as_secs_f64() * 1000.0);
                    }
                    self.copy_stall = None;
                    self.free_events.push(ev);
                }
                // An error here is the GPU context's: the next GPU call reports it.
                _ => return true,
            }
        }
        // The staging buffer must hold this layer's copies without wrapping: a wrap waits for
        // every copy still reading the buffer, with no time limit, and the GPU would be waiting
        // behind it (this lost the GPU context once the 250 ms limit above was in place). So wrap
        // here, before anything is planned, and only if the copy stream is idle.
        let need = (DEFAULT_SCRATCH_RING_SLOTS * self.nvfp4_layout.aligned_slab_bytes.max(self.mxfp8_layout.aligned_slab_bytes))
            .min(self.pinned_staging.size_bytes);
        if self.staging_cursor + need > self.pinned_staging.size_bytes {
            if self.backend.record_event(self.kick_event, self.copy_stream).is_err() {
                return true;
            }
            // Normally idle (every rescue waits for its copies); 2 ms covers the event's own latency.
            let t = std::time::Instant::now();
            loop {
                match self.backend.query_event(self.kick_event) {
                    Ok(true) => break,
                    Ok(false) if t.elapsed() < std::time::Duration::from_millis(2) => std::hint::spin_loop(),
                    _ => return true,
                }
            }
            self.staging_cursor = 0;
            self.uploads_synced = self.uploads_issued;
        }
        false
    }

    /// Frees the Scratch Rescue Ring for direct use (unmaps occupants on the compute stream).
    pub fn release_scratch_ring(&mut self) -> Result<()> {
        self.scheduler.cache.clear_scratch_ring();
        if self.scheduler.cache.has_dirty_gpu_table() {
            self.upload_table_on_compute()?;
        }
        Ok(())
    }

    /// Copies `experts` (all of one class, at most the ring size) into consecutive Scratch Rescue
    /// slots and makes the compute stream wait for them; returns their slots. The slots are NOT
    /// published in the GPU table: the caller addresses them directly (batched prefill MoE).
    /// Call `release_scratch_ring` once before a prefill that uses this.
    pub fn stage_into_scratch(&mut self, experts: &[ExpertId]) -> Result<Vec<SlotIndex>> {
        if experts.is_empty() {
            return Ok(Vec::new());
        }
        self.fence_scratch()?;
        self.stage_into_scratch_fenced(experts)
    }

    /// Marks the point on the compute stream after which nothing queued so far uses the scratch
    /// ring: a following `stage_into_scratch_fenced` copies as soon as that point is reached, so
    /// kernels queued in between (which must not touch the ring) overlap the copy.
    pub fn fence_scratch(&mut self) -> Result<()> {
        self.backend.record_event(self.compute_fence, self.compute_stream)
    }

    /// `stage_into_scratch` behind the fence set by the last `fence_scratch`.
    pub fn stage_into_scratch_fenced(&mut self, experts: &[ExpertId]) -> Result<Vec<SlotIndex>> {
        if experts.is_empty() {
            return Ok(Vec::new());
        }
        let cls = self.scheduler.cache.expert_class(experts[0]);
        let (base, size) = self.scheduler.cache.scratch_ring(cls);
        if experts.len() > size as usize {
            return Err(SmallTideError::Backend { backend: "tide", message: "too many experts for scratch ring".into() });
        }
        // Previous users of these slots (kernels on the compute stream before the fence) must be done.
        self.backend.stream_wait_event(self.copy_stream, self.compute_fence)?;
        let mut slots = Vec::with_capacity(experts.len());
        for (i, &id) in experts.iter().enumerate() {
            debug_assert_eq!(self.scheduler.cache.expert_class(id), cls);
            let slot = SlotIndex(base + i as u32);
            let slab_len = self.slab_len_for(id);
            let e = self.scheduler.cache.entry(id);
            let ram = if self.scheduler.cache.is_ram_ready(id) { Some(e.ram_slot) } else { None };
            let (dst_pool, dst_off) = self.vram_target(id, slot);
            self.stage_and_upload(id, cls, ram, dst_pool, dst_off, slab_len, false)?;
            self.scheduler.telemetry.stage1_sync_rescues += 1;
            self.scheduler.telemetry.stage1_bytes_transferred += slab_len as u64;
            slots.push(slot);
        }
        self.backend.record_event(self.dma_event, self.copy_stream)?;
        self.backend.stream_wait_event(self.compute_stream, self.dma_event)?;
        Ok(slots)
    }

    /// Stages one expert (RAM, or the disk if it is not there) and queues its copy to the card.
    /// `more_follow`: further experts of the same batch are staged after this one, so its
    /// transfer is started now and runs while they are copied from RAM.
    fn stage_and_upload(
        &mut self,
        expert: ExpertId,
        cls: ExpertClass,
        ram_slot: Option<SlotIndex>,
        dst_pool: DeviceBuf,
        dst_off: usize,
        slab_len: usize,
        more_follow: bool,
    ) -> Result<()> {
        let host_pool = self.mxfp8_host_mapped && cls == ExpertClass::Mxfp8;
        if let (false, Some(rs)) = (host_pool, ram_slot) {
            if let Some(ps) = self.pinned_slabs.get(&(cls, rs)) {
                if ps.owner == Some(expert) && ps.len >= slab_len && dst_off + slab_len <= dst_pool.size_bytes {
                    // Page-locked RAM copy: the transfer reads it in place. With nothing to copy
                    // meanwhile there is no reason to start it early.
                    let buf = ps.buf;
                    self.backend.memcpy_h2d_async(dst_pool, dst_off, buf, 0, slab_len, self.copy_stream)?;
                    self.uploads_issued += 1;
                    self.pinned_uploads += 1;
                    self.pinned_slabs.get_mut(&(cls, rs)).unwrap().upload_seq = self.uploads_issued;
                    return Ok(());
                }
            }
        }
        let src_offset = self.reserve_staging(slab_len)?;
        self.fill_staging_from_ram_or_disk(expert, cls, ram_slot, src_offset, slab_len)?;
        self.upload_slab(expert, dst_pool, dst_off, src_offset, slab_len)?;
        if more_follow && self.stage_kick {
            self.kick_copy_stream()?;
        }
        Ok(())
    }

    /// Page-locked part of the RAM tier in bytes (`SMALLTIDE_RAM_PINNED_GB`, 0 = none). Call
    /// before any expert is loaded.
    pub fn set_ram_pinned_budget(&mut self, bytes: usize) {
        self.ram_pinned_budget = bytes;
    }

    /// (experts whose RAM copy is page-locked now, page-locked bytes allocated).
    pub fn ram_pinned_stats(&self) -> (usize, usize) {
        (self.pinned_slabs.values().filter(|p| p.owner.is_some()).count(), self.ram_pinned_bytes)
    }

    /// Stores `bytes` as `expert`'s RAM copy in slot `(cls, slot)`: in the slot's page-locked
    /// buffer when it has one (or can still get one), else on the heap.
    fn ram_put(&mut self, cls: ExpertClass, slot: SlotIndex, expert: ExpertId, bytes: Vec<u8>) -> Result<()> {
        let key = (cls, slot);
        let cap = match cls {
            ExpertClass::Nvfp4 => self.nvfp4_layout.aligned_slab_bytes,
            ExpertClass::Mxfp8 => self.mxfp8_layout.aligned_slab_bytes,
        };
        let host_pool = self.mxfp8_host_mapped && cls == ExpertClass::Mxfp8;
        if bytes.len() <= cap && !host_pool {
            if !self.pinned_slabs.contains_key(&key) && self.ram_pinned_bytes + cap <= self.ram_pinned_budget {
                if let Ok(buf) = self.backend.alloc_pinned(cap) {
                    self.ram_pinned_bytes += cap;
                    self.pinned_slabs.insert(key, PinnedSlab { owner: None, buf, len: 0, upload_seq: 0 });
                }
            }
            if let Some(ps) = self.pinned_slabs.get(&key) {
                let (buf, seq) = (ps.buf, ps.upload_seq);
                if seq > self.uploads_synced {
                    // An upload from this buffer may still be in flight: it must land first.
                    self.backend.sync_stream(self.copy_stream)?;
                    self.uploads_synced = self.uploads_issued;
                }
                unsafe {
                    let dst = self.backend.pinned_as_mut_slice(buf)?;
                    dst[..bytes.len()].copy_from_slice(&bytes);
                }
                let ps = self.pinned_slabs.get_mut(&key).unwrap();
                ps.owner = Some(expert);
                ps.len = bytes.len();
                self.ram_slabs.remove(&key);
                return Ok(());
            }
        }
        if let Some(ps) = self.pinned_slabs.get_mut(&key) {
            ps.owner = None;
        }
        self.ram_slabs.insert(key, (expert, bytes));
        Ok(())
    }

    /// Forgets the RAM copy in slot `(cls, slot)`; a page-locked buffer stays for the slot's
    /// next occupant.
    fn ram_drop(&mut self, cls: ExpertClass, slot: SlotIndex) {
        self.ram_slabs.remove(&(cls, slot));
        if let Some(ps) = self.pinned_slabs.get_mut(&(cls, slot)) {
            ps.owner = None;
        }
    }

    /// Copies an expert's slab into the staging ring from its RAM slot, verifying the slot really
    /// holds this expert; falls back to a blocking NVMe read (counted in telemetry) otherwise.
    fn fill_staging_from_ram_or_disk(
        &mut self,
        expert: ExpertId,
        cls: ExpertClass,
        ram_slot: Option<SlotIndex>,
        src_offset: usize,
        slab_len: usize,
    ) -> Result<()> {
        let hit = ram_slot
            .and_then(|rs| self.ram_slabs.get(&(cls, rs)))
            .filter(|(owner, bytes)| *owner == expert && bytes.len() >= slab_len)
            .is_some();
        if hit {
            let (_, bytes) = self.ram_slabs.get(&(cls, ram_slot.unwrap())).unwrap();
            let t = std::time::Instant::now();
            unsafe {
                let staging = self.backend.pinned_as_mut_slice(self.pinned_staging)?;
                staging[src_offset..src_offset + slab_len].copy_from_slice(&bytes[..slab_len]);
            }
            self.stage_copy.0 += t.elapsed().as_secs_f64();
            self.stage_copy.1 += 1;
            return Ok(());
        }
        if ram_slot.is_some() {
            self.scheduler.telemetry.stage1_ram_fallback_reads += 1;
            self.scheduler.telemetry.rescue_disk_reads += 1;
        }
        let t = std::time::Instant::now();
        let bytes = self.read_disk_sync(expert, slab_len)?;
        unsafe {
            let staging = self.backend.pinned_as_mut_slice(self.pinned_staging)?;
            staging[src_offset..src_offset + slab_len].copy_from_slice(&bytes[..slab_len]);
        }
        self.stage_disk.0 += t.elapsed().as_secs_f64();
        self.stage_disk.1 += 1;
        if ram_slot.is_none() {
            self.stage_disk.2 += 1;
        }
        if let Some(rs) = ram_slot {
            // Repair the RAM tier so the next promotion of this expert is a real RAM hit.
            if self.scheduler.cache.entry(expert).ram_slot == rs {
                self.ram_put(cls, rs, expert, bytes)?;
            }
        } else if let Some(rs) = self.scheduler.adopt_disk_read_into_ram(expert) {
            // It was on the disk only, or its background read has not been collected yet. Keep
            // what was just read, or every later rescue of it reads the disk again (seen: 94
            // experts read 386 times in one 12K-token request).
            self.ram_put(cls, rs, expert, bytes)?;
        }
        Ok(())
    }

    fn read_disk_sync(&self, expert: ExpertId, slab_len: usize) -> Result<Vec<u8>> {
        match &self.stage2_worker {
            Some(worker) => worker.read_sync(expert, slab_len),
            None => Err(SmallTideError::Backend {
                backend: "tide",
                message: format!(
                    "expert {:?} is not in RAM and no Stage 2 disk pack is attached",
                    expert
                ),
            }),
        }
    }

    fn slab_len_for(&self, id: ExpertId) -> usize {
        match self.scheduler.cache.expert_class(id) {
            ExpertClass::Nvfp4 => self.nvfp4_layout.aligned_slab_bytes,
            ExpertClass::Mxfp8 => self.mxfp8_layout.aligned_slab_bytes,
        }
    }

    fn vram_target(&self, id: ExpertId, slot: SlotIndex) -> (DeviceBuf, usize) {
        match self.scheduler.cache.expert_class(id) {
            ExpertClass::Nvfp4 => (
                self.vram_pool_nvfp4,
                (slot.0 as usize) * self.nvfp4_layout.aligned_slab_bytes,
            ),
            ExpertClass::Mxfp8 => (
                self.vram_pool_mxfp8,
                (slot.0 as usize) * self.mxfp8_layout.aligned_slab_bytes,
            ),
        }
    }

    fn reserve_staging(&mut self, len: usize) -> Result<usize> {
        if self.staging_cursor + len > self.pinned_staging.size_bytes {
            self.backend.sync_stream(self.copy_stream).map_err(|e| SmallTideError::Backend {
                backend: "tide",
                message: format!(
                    "reserve_staging wrap sync copy_stream (cursor={}, len={}, cap={}): {e}",
                    self.staging_cursor, len, self.pinned_staging.size_bytes
                ),
            })?;
            self.staging_cursor = 0;
        }
        let off = self.staging_cursor;
        self.staging_cursor += len;
        Ok(off)
    }

    /// Uploads the published 12 KB GPU indirection table on the **compute stream** from a
    /// dedicated pinned buffer (Invariant 1). Clears the cache's dirty set.
    pub fn upload_table_on_compute(&mut self) -> Result<()> {
        self.upload_table_on(self.compute_stream)
    }

    /// `upload_table_on_compute` on `stream`.
    pub fn upload_table_on(&mut self, stream: Stream) -> Result<()> {
        let _ = self.scheduler.cache.take_dirty_gpu_updates();
        let byte_len = self.scheduler.cache.gpu_slot_table().len() * 4;
        let i = self.table_ring_cursor;
        self.table_ring_cursor = (i + 1) % self.table_ring.len();
        if self.table_ring[i].used {
            // Practically always already complete (8 uploads ago); guards the pinned buffer reuse.
            self.backend.sync_event(self.table_ring[i].done)?;
        }
        let table = self.scheduler.cache.gpu_slot_table();
        let table_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(table.as_ptr() as *const u8, byte_len) };
        unsafe {
            let dst = self.backend.pinned_as_mut_slice(self.table_ring[i].pinned)?;
            dst[..byte_len].copy_from_slice(table_bytes);
        }
        self.backend.memcpy_h2d_async(
            self.gpu_slot_table,
            0,
            self.table_ring[i].pinned,
            0,
            byte_len,
            stream,
        )?;
        self.backend.record_event(self.table_ring[i].done, stream)?;
        self.table_ring[i].used = true;
        Ok(())
    }

    /// Back-compat alias (now stream-ordered on the compute stream).
    pub fn sync_full_slot_table_to_gpu(&mut self) -> Result<()> {
        self.upload_table_on_compute()
    }
}

impl<B: ComputeBackend> Drop for TideExecutor<B> {
    fn drop(&mut self) {
        let _ = self.backend.sync_stream(self.copy_stream);
        let _ = self.backend.sync_stream(self.compute_stream);
        for (ev, _) in self.pending_batches.drain(..) {
            let _ = self.backend.destroy_event(ev);
        }
        for ev in self.free_events.drain(..) {
            let _ = self.backend.destroy_event(ev);
        }
        for t in self.table_ring.drain(..) {
            let _ = self.backend.destroy_event(t.done);
            let _ = self.backend.free_pinned(t.pinned);
        }
        let _ = self.backend.destroy_event(self.compute_fence);
        let _ = self.backend.destroy_event(self.dma_event);
        let _ = self.backend.destroy_stream(self.copy_stream);
        let _ = self.backend.destroy_stream(self.compute_stream);
        let _ = self.backend.free_pinned(self.pinned_staging);
        let _ = self.backend.free_device(self.gpu_slot_table);
        let _ = self.backend.free_device(self.vram_pool_mxfp8);
        if let Some(h) = self.mxfp8_host.take() {
            let _ = self.backend.free_pinned(h);
        }
        let _ = self.backend.free_device(self.vram_pool_nvfp4);
    }
}
