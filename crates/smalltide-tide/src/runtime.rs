//! Backend-agnostic 2-stage runtime executor (`TideExecutor<B: ComputeBackend>`).
//!
//! Executes `TransferOp` plans emitted by `TideScheduler`:
//! - Stage 2 Async (`DiskToRam`): offloaded to dedicated `Stage2IoWorker` OS thread (`NO_BUFFERING`)
//!   so `0.96 ms` NVMe reads overlap with GPU decode compute without blocking the token loop!
//! - Stage 1 (`RamToVram`): copies RAM slab -> bounded pinned staging ring -> async H2D DMA -> VRAM
//! - Direct (`DiskToVramDirect`): blocking read via `Stage2IoWorker::read_sync` -> staging -> VRAM
//! - Transient (`ScratchRescue`): copies cold 1-off sync miss into tail Scratch Rescue Ring slot
//!   WITHOUT evicting any warm resident expert!

use crate::cache::{ExpertClass, TransferOp};
use crate::io_worker::Stage2IoWorker;
use crate::scheduler::TideScheduler;
use smalltide_core::{
    ComputeBackend, DType, DeviceBuf, Event, Inkling2Config, PinnedBuf, QueueKind, Result,
    SlotIndex, Stream, TideConfig,
};
use smalltide_io::stide_pack::{ExpertSlabLayout, StidePackReader};
use std::collections::HashMap;
use std::sync::Arc;

pub const DEFAULT_SCRATCH_RING_SLOTS: usize = 8;

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

    /// Bounded page-locked DMA staging ring (e.g. 512 MiB on Windows).
    pinned_staging: PinnedBuf,
    staging_cursor: usize,

    /// Tier 1 Host RAM warm expert slabs (indexed by `(ExpertClass, SlotIndex)`).
    ram_slabs: HashMap<(ExpertClass, SlotIndex), Vec<u8>>,
    /// Dedicated background Stage 2 NVMe I/O worker thread.
    stage2_worker: Option<Stage2IoWorker>,

    pub compute_stream: Stream,
    pub copy_stream: Stream,
    pub dma_event: Event,
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

        let compute_stream = backend.create_stream(QueueKind::Compute)?;
        let copy_stream = backend.create_stream(QueueKind::AsyncCopyH2D)?;
        let dma_event = backend.create_event()?;

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
            ram_slabs: HashMap::new(),
            stage2_worker,
            compute_stream,
            copy_stream,
            dma_event,
        };
        exec.sync_full_slot_table_to_gpu()?;
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
            self.ram_slabs.insert((item.class, item.ram_slot), bytes);
            self.scheduler.cache.mark_ram_ready(item.expert);
        }
        Ok(count)
    }

    /// Executes a batch of Stage 1 and Stage 2 `TransferOp` commands and syncs GPU indirection slots.
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

        let mut issued_dma = false;
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
                            self.ram_slabs.insert((cls, ram_slot), bytes);
                            self.scheduler.cache.mark_ram_ready(expert);
                        }
                    } else {
                        self.ram_slabs
                            .entry((cls, ram_slot))
                            .or_insert_with(|| vec![0u8; slab_len]);
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
                    if let Some(bytes) = self.ram_slabs.get(&(cls, src_ram_slot)) {
                        unsafe {
                            let staging = self.backend.pinned_as_mut_slice(self.pinned_staging)?;
                            staging[src_offset..src_offset + slab_len]
                                .copy_from_slice(&bytes[..slab_len]);
                        }
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
                    issued_dma = true;
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
                    let bytes = if let Some(worker) = &self.stage2_worker {
                        worker.read_sync(expert, slab_len)?
                    } else {
                        vec![0u8; slab_len]
                    };
                    let src_offset = self.reserve_staging(slab_len)?;
                    unsafe {
                        let staging = self.backend.pinned_as_mut_slice(self.pinned_staging)?;
                        staging[src_offset..src_offset + slab_len].copy_from_slice(&bytes);
                    }
                    if let Some(rs) = also_keep_ram_slot {
                        self.ram_slabs.insert((cls, rs), bytes);
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
                    issued_dma = true;
                }
                TransferOp::ScratchRescue {
                    expert,
                    src_ram_slot,
                    scratch_vram_slot,
                } => {
                    let cls = self.scheduler.cache.expert_class(expert);
                    let slab_len = self.slab_len_for(expert);
                    let src_offset = self.reserve_staging(slab_len)?;
                    if let Some(rs) = src_ram_slot {
                        if let Some(bytes) = self.ram_slabs.get(&(cls, rs)) {
                            unsafe {
                                let staging =
                                    self.backend.pinned_as_mut_slice(self.pinned_staging)?;
                                staging[src_offset..src_offset + slab_len]
                                    .copy_from_slice(&bytes[..slab_len]);
                            }
                        }
                    } else if let Some(worker) = &self.stage2_worker {
                        let bytes = worker.read_sync(expert, slab_len)?;
                        unsafe {
                            let staging = self.backend.pinned_as_mut_slice(self.pinned_staging)?;
                            staging[src_offset..src_offset + slab_len].copy_from_slice(&bytes);
                        }
                    }
                    let (dst_pool, dst_off) = self.vram_target(expert, scratch_vram_slot);
                    self.backend.memcpy_h2d_async(
                        dst_pool,
                        dst_off,
                        self.pinned_staging,
                        src_offset,
                        slab_len,
                        self.copy_stream,
                    )?;
                    issued_dma = true;
                }
            }
        }

        // Flush updated GPU indirection table entries
        let dirty = self.scheduler.cache.take_dirty_gpu_updates();
        if !dirty.is_empty() {
            self.sync_full_slot_table_to_gpu()?;
            issued_dma = true;
        }

        if issued_dma {
            self.backend.record_event(self.dma_event, self.copy_stream)?;
            if sync_compute_after {
                self.backend
                    .stream_wait_event(self.compute_stream, self.dma_event)?;
            }
        }
        Ok(())
    }

    fn slab_len_for(&self, id: smalltide_core::ExpertId) -> usize {
        match self.scheduler.cache.expert_class(id) {
            ExpertClass::Nvfp4 => self.nvfp4_layout.aligned_slab_bytes,
            ExpertClass::Mxfp8 => self.mxfp8_layout.aligned_slab_bytes,
        }
    }

    fn vram_target(&self, id: smalltide_core::ExpertId, slot: SlotIndex) -> (DeviceBuf, usize) {
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

    /// Uploads the entire 12 KB GPU indirection table (`3,072 * 4 B`) via pinned staging in <1 µs.
    pub fn sync_full_slot_table_to_gpu(&mut self) -> Result<()> {
        let byte_len = self.scheduler.cache.gpu_slot_table().len() * 4;
        let src_offset = self.reserve_staging(byte_len)?;
        let table = self.scheduler.cache.gpu_slot_table();
        let table_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(table.as_ptr() as *const u8, byte_len) };
        unsafe {
            let staging = self.backend.pinned_as_mut_slice(self.pinned_staging)?;
            staging[src_offset..src_offset + byte_len].copy_from_slice(table_bytes);
        }
        self.backend.memcpy_h2d_async(
            self.gpu_slot_table,
            0,
            self.pinned_staging,
            src_offset,
            byte_len,
            self.copy_stream,
        )
    }
}

impl<B: ComputeBackend> Drop for TideExecutor<B> {
    fn drop(&mut self) {
        let _ = self.backend.sync_stream(self.copy_stream);
        let _ = self.backend.sync_stream(self.compute_stream);
        let _ = self.backend.destroy_event(self.dma_event);
        let _ = self.backend.destroy_stream(self.copy_stream);
        let _ = self.backend.destroy_stream(self.compute_stream);
        let _ = self.backend.free_pinned(self.pinned_staging);
        let _ = self.backend.free_device(self.gpu_slot_table);
        let _ = self.backend.free_device(self.vram_pool_mxfp8);
        let _ = self.backend.free_device(self.vram_pool_nvfp4);
    }
}
