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
const TABLE_UPLOAD_RING: usize = 8;

struct TableUploadBuf {
    pinned: PinnedBuf,
    done: Event,
    used: bool,
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
    /// Dedicated background Stage 2 NVMe I/O worker thread.
    stage2_worker: Option<Stage2IoWorker>,

    /// Async promotion batches whose DMA may still be in flight (copy-stream order).
    pending_batches: VecDeque<(Event, Vec<(ExpertId, SlotIndex)>)>,
    free_events: Vec<Event>,

    pub compute_stream: Stream,
    pub copy_stream: Stream,
    pub dma_event: Event,
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
        let scratch = DEFAULT_SCRATCH_RING_SLOTS;

        // Resident budget excludes the tail Scratch Rescue Ring slots
        let unified_resident_vram_bytes = resident_nvfp4 * nvfp4_layout.aligned_slab_bytes
            + resident_mxfp8 * mxfp8_layout.aligned_slab_bytes;

        let vram_pool_nvfp4 =
            backend.alloc_device((resident_nvfp4 + scratch) * nvfp4_layout.aligned_slab_bytes)?;
        let vram_pool_mxfp8 =
            backend.alloc_device((resident_mxfp8 + scratch) * mxfp8_layout.aligned_slab_bytes)?;

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
        scheduler.cache.configure_scratch_ring(
            resident_nvfp4 as u32,
            resident_mxfp8 as u32,
            scratch as u32,
        );

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
            stage2_worker,
            pending_batches: VecDeque::new(),
            free_events: Vec::new(),
            compute_stream,
            copy_stream,
            dma_event,
            compute_fence,
        };
        exec.upload_table_on_compute()?;
        Ok(exec)
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
                self.ram_slabs
                    .insert((item.class, item.ram_slot), (item.expert, bytes));
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

        let mut batch: Vec<(ExpertId, SlotIndex)> = Vec::new();
        for op in ops {
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
                            self.ram_slabs.insert((cls, ram_slot), (expert, bytes));
                            self.scheduler.cache.mark_ram_ready(expert);
                        }
                    } else {
                        self.ram_slabs
                            .insert((cls, ram_slot), (expert, vec![0u8; slab_len]));
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
                    let src_offset = self.reserve_staging(slab_len)?;
                    self.fill_staging_from_ram_or_disk(expert, cls, Some(src_ram_slot), src_offset, slab_len)?;
                    let (dst_pool, dst_off) = self.vram_target(expert, dst_vram_slot);
                    self.backend.memcpy_h2d_async(
                        dst_pool,
                        dst_off,
                        self.pinned_staging,
                        src_offset,
                        slab_len,
                        self.copy_stream,
                    )?;
                    batch.push((expert, dst_vram_slot));
                    if release_ram_after_upload {
                        self.ram_slabs.remove(&(cls, src_ram_slot));
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
                        self.ram_slabs.insert((cls, rs), (expert, bytes));
                    }
                    let (dst_pool, dst_off) = self.vram_target(expert, dst_vram_slot);
                    self.backend.memcpy_h2d_async(
                        dst_pool,
                        dst_off,
                        self.pinned_staging,
                        src_offset,
                        slab_len,
                        self.copy_stream,
                    )?;
                    batch.push((expert, dst_vram_slot));
                }
                TransferOp::ScratchRescue {
                    expert,
                    src_ram_slot,
                    scratch_vram_slot,
                } => {
                    let cls = self.scheduler.cache.expert_class(expert);
                    let slab_len = self.slab_len_for(expert);
                    let src_offset = self.reserve_staging(slab_len)?;
                    self.fill_staging_from_ram_or_disk(expert, cls, src_ram_slot, src_offset, slab_len)?;
                    let (dst_pool, dst_off) = self.vram_target(expert, scratch_vram_slot);
                    self.backend.memcpy_h2d_async(
                        dst_pool,
                        dst_off,
                        self.pinned_staging,
                        src_offset,
                        slab_len,
                        self.copy_stream,
                    )?;
                    batch.push((expert, scratch_vram_slot));
                }
            }
        }

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
            unsafe {
                let staging = self.backend.pinned_as_mut_slice(self.pinned_staging)?;
                staging[src_offset..src_offset + slab_len].copy_from_slice(&bytes[..slab_len]);
            }
            return Ok(());
        }
        if ram_slot.is_some() {
            self.scheduler.telemetry.stage1_ram_fallback_reads += 1;
        }
        let bytes = self.read_disk_sync(expert, slab_len)?;
        unsafe {
            let staging = self.backend.pinned_as_mut_slice(self.pinned_staging)?;
            staging[src_offset..src_offset + slab_len].copy_from_slice(&bytes[..slab_len]);
        }
        if let Some(rs) = ram_slot {
            // Repair the RAM tier so the next promotion of this expert is a real RAM hit.
            if self.scheduler.cache.entry(expert).ram_slot == rs {
                self.ram_slabs.insert((cls, rs), (expert, bytes));
            }
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
            self.backend.sync_stream(self.copy_stream)?;
            self.staging_cursor = 0;
        }
        let off = self.staging_cursor;
        self.staging_cursor += len;
        Ok(off)
    }

    /// Uploads the published 12 KB GPU indirection table on the **compute stream** from a
    /// dedicated pinned buffer (Invariant 1). Clears the cache's dirty set.
    pub fn upload_table_on_compute(&mut self) -> Result<()> {
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
            self.compute_stream,
        )?;
        self.backend
            .record_event(self.table_ring[i].done, self.compute_stream)?;
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
        let _ = self.backend.free_device(self.vram_pool_nvfp4);
    }
}
