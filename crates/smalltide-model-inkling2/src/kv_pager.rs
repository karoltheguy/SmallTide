//! Unified Hybrid Paged KV pool: every sequence's target local sliding-window layers (40),
//! target global-attention layers (8), and DFlash 2 sliding-window layers (5) share ONE single
//! physical pool of single-layer pages (`PAGE_POS = 256` positions per page, matching
//! `KV_PAGE_SHIFT = 8` in the HIP kernels).
//!
//! Why unified?
//! - All 53 layers (40 local + 8 global + 5 DFlash 2) have identical KV head geometry (`8 x 128`).
//! - Idle lanes hold 0 physical pages (0 bytes of KV VRAM), so raising `SMALLTIDE_LANES` (e.g. to
//!   4 or 8 true parallel lanes) costs no extra upfront VRAM when fewer requests are active.
//! - Local sliding-window layers (`window = 1024`) evict expired pages back to the shared pool as
//!   the sequence advances (during prefill, layer-by-layer so peak temporary prefill overhead is
//!   only ~4 single-layer pages / ~1.7 MiB across all 40 layers).
//! - Elastic KV grows/shrinks just 4 contiguous buffers `(k, ks, v, vs)` on demand by trading
//!   VRAM with tail NVFP4 expert slots.

use smalltide_core::{ComputeBackend, DeviceBuf, PinnedBuf, Result, Stream};

/// Positions a single-layer physical page covers (must match `KV_PAGE_SHIFT = 8` in `core_ops.hip`).
pub const PAGE_POS: usize = 256;
pub const NUM_LOCAL_LAYERS: usize = 40;
pub const NUM_GLOBAL_LAYERS: usize = 8;
pub const NUM_DFLASH2_LAYERS: usize = 5;
pub const LOCAL_WINDOW: usize = 1024;
pub const DFLASH2_WINDOW: usize = 2048;

const STAGING_RING: usize = 4;

/// Device page-table slices belonging to one lane (all sliced from 3 tiny per-lane allocations).
#[derive(Clone)]
pub struct LaneTables {
    /// One `[local_ring_pages]` u32 device slice per target local layer (40).
    pub local: Vec<DeviceBuf>,
    /// One `[max_global_pages]` u32 device slice per target global layer (8).
    pub global: Vec<DeviceBuf>,
    /// One `[drafter_ring_pages]` u32 device slice per DFlash 2 layer (5).
    pub drafter: Vec<DeviceBuf>,
}

/// Ordered list of live physical page IDs for a prefix snapshot at `pos`.
#[derive(Clone, Debug)]
pub struct SnapshotPages {
    /// For each of the 40 local layers: `(logical_ring_slot, phys_page_id)` pairs.
    pub local: Vec<Vec<(usize, u32)>>,
    /// For each of the 8 global layers: `phys_page_id` for logical pages `0..n_global`.
    pub global: Vec<Vec<u32>>,
    /// For each of the 5 DFlash 2 layers (empty if not included): `(logical_ring_slot, phys_page_id)`.
    pub drafter: Vec<Vec<(usize, u32)>>,
}

impl SnapshotPages {
    pub fn total_pages(&self) -> usize {
        self.local.iter().map(Vec::len).sum::<usize>()
            + self.global.iter().map(Vec::len).sum::<usize>()
            + self.drafter.iter().map(Vec::len).sum::<usize>()
    }
}

struct LaneState {
    /// `[40][local_ring_pages]`
    local_slots: Vec<Vec<Option<u32>>>,
    /// `[8]` contiguous `0..n_mapped`
    global_slots: Vec<Vec<u32>>,
    /// `[5][drafter_ring_pages]`
    drafter_slots: Vec<Vec<Option<u32>>>,
    local_table: DeviceBuf,
    global_table: DeviceBuf,
    drafter_table: DeviceBuf,
    /// Separate 4-deep pinned staging rings per table so per-layer local uploads, global uploads,
    /// and drafter uploads never overwrite each other's in-flight H2D staging slices.
    local_staging: [PinnedBuf; STAGING_RING],
    local_stage_idx: usize,
    global_staging: [PinnedBuf; STAGING_RING],
    global_stage_idx: usize,
    drafter_staging: [PinnedBuf; STAGING_RING],
    drafter_stage_idx: usize,
}

impl LaneState {
    fn total_pages(&self) -> usize {
        let loc: usize = self.local_slots.iter().map(|v| v.iter().filter(|x| x.is_some()).count()).sum();
        let glob: usize = self.global_slots.iter().map(Vec::len).sum();
        let dr: usize = self.drafter_slots.iter().map(|v| v.iter().filter(|x| x.is_some()).count()).sum();
        loc + glob + dr
    }

    fn local_staging_buf(&mut self, advance: bool) -> PinnedBuf {
        if advance {
            self.local_stage_idx = (self.local_stage_idx + 1) % STAGING_RING;
        }
        self.local_staging[self.local_stage_idx]
    }

    fn next_global_staging(&mut self) -> PinnedBuf {
        let buf = self.global_staging[self.global_stage_idx];
        self.global_stage_idx = (self.global_stage_idx + 1) % STAGING_RING;
        buf
    }

    fn next_drafter_staging(&mut self) -> PinnedBuf {
        let buf = self.drafter_staging[self.drafter_stage_idx];
        self.drafter_stage_idx = (self.drafter_stage_idx + 1) % STAGING_RING;
        buf
    }
}

/// Logical ring slots touched by position interval `[lo, hi)` (`hi > lo`) in a ring of `ring_pages` pages.
#[inline]
pub fn ring_logical_pages(lo: usize, hi: usize, ring_pages: usize) -> Vec<usize> {
    if hi <= lo || ring_pages == 0 {
        return Vec::new();
    }
    let first = lo / PAGE_POS;
    let last = (hi - 1) / PAGE_POS;
    let mut out = Vec::with_capacity((last - first + 1).min(ring_pages));
    for abs_p in first..=last {
        let lp = abs_p % ring_pages;
        if !out.contains(&lp) {
            out.push(lp);
        }
    }
    out
}

pub struct KvPool {
    n_pages: usize,
    max_pages: usize,
    max_global_pages: usize,
    local_ring_pages: usize,
    drafter_ring_pages: usize,
    free: Vec<u32>,
    lanes: Vec<LaneState>,
    /// Lanes currently serving a request (never reclaimed when free pool runs empty).
    pub pinned: Vec<bool>,
    /// Lanes whose pages were reclaimed since the last `take_lost`.
    lost: Vec<bool>,
}

impl KvPool {
    /// Creates a unified hybrid paged KV pool of `n_pages` single-layer pages (each `PAGE_POS = 256`
    /// positions), supporting up to `max_positions` global context per lane and `prefill_chunk`-token
    /// prefill chunks for sliding-window layers.
    pub fn new(n_pages: usize, max_positions: usize, prefill_chunk: usize) -> Self {
        let max_global_pages = max_positions.div_ceil(PAGE_POS);
        // +1 extra ring page guarantees zero logical wrap collision between oldest window token
        // and newest prefill-chunk token even across unaligned boundaries.
        let local_ring_pages = (LOCAL_WINDOW + prefill_chunk).div_ceil(PAGE_POS) + 1;
        let drafter_ring_pages = (DFLASH2_WINDOW + prefill_chunk).div_ceil(PAGE_POS) + 1;
        Self {
            n_pages,
            max_pages: n_pages,
            max_global_pages,
            local_ring_pages,
            drafter_ring_pages,
            free: (0..n_pages as u32).rev().collect(),
            lanes: Vec::new(),
            pinned: Vec::new(),
            lost: Vec::new(),
        }
    }

    pub fn set_max_pages(&mut self, max_pages: usize) {
        self.max_pages = max_pages.max(self.n_pages);
    }

    pub fn pages_total(&self) -> usize {
        self.n_pages
    }

    pub fn pages_free(&self) -> usize {
        self.free.len()
    }

    pub fn pages_used(&self) -> usize {
        self.n_pages.saturating_sub(self.free.len())
    }

    pub fn local_ring_slots(&self) -> usize {
        self.local_ring_pages * PAGE_POS
    }

    pub fn drafter_ring_slots(&self) -> usize {
        self.drafter_ring_pages * PAGE_POS
    }

    pub fn local_ring_pages(&self) -> usize {
        self.local_ring_pages
    }

    pub fn drafter_ring_pages(&self) -> usize {
        self.drafter_ring_pages
    }

    pub fn max_global_pages(&self) -> usize {
        self.max_global_pages
    }

    /// Adds a lane; allocates its tiny device page tables (~18 KB) and staging rings.
    pub fn add_lane<B: ComputeBackend>(&mut self, b: &B) -> Result<LaneTables> {
        let local_bytes = NUM_LOCAL_LAYERS * self.local_ring_pages * 4;
        let global_bytes = NUM_GLOBAL_LAYERS * self.max_global_pages * 4;
        let drafter_bytes = NUM_DFLASH2_LAYERS * self.drafter_ring_pages * 4;

        let local_table = b.alloc_device(local_bytes)?;
        let global_table = b.alloc_device(global_bytes)?;
        let drafter_table = b.alloc_device(drafter_bytes)?;
        let local_staging = [
            b.alloc_pinned(local_bytes)?,
            b.alloc_pinned(local_bytes)?,
            b.alloc_pinned(local_bytes)?,
            b.alloc_pinned(local_bytes)?,
        ];
        let global_staging = [
            b.alloc_pinned(global_bytes)?,
            b.alloc_pinned(global_bytes)?,
            b.alloc_pinned(global_bytes)?,
            b.alloc_pinned(global_bytes)?,
        ];
        let drafter_staging = [
            b.alloc_pinned(drafter_bytes)?,
            b.alloc_pinned(drafter_bytes)?,
            b.alloc_pinned(drafter_bytes)?,
            b.alloc_pinned(drafter_bytes)?,
        ];

        let tables = self.slice_lane_tables(local_table, global_table, drafter_table);
        self.lanes.push(LaneState {
            local_slots: vec![vec![None; self.local_ring_pages]; NUM_LOCAL_LAYERS],
            global_slots: vec![Vec::new(); NUM_GLOBAL_LAYERS],
            drafter_slots: vec![vec![None; self.drafter_ring_pages]; NUM_DFLASH2_LAYERS],
            local_table,
            global_table,
            drafter_table,
            local_staging,
            local_stage_idx: 0,
            global_staging,
            global_stage_idx: 0,
            drafter_staging,
            drafter_stage_idx: 0,
        });
        self.pinned.push(false);
        self.lost.push(false);
        Ok(tables)
    }

    pub fn lane_tables(&self, lane: usize) -> LaneTables {
        let l = &self.lanes[lane];
        self.slice_lane_tables(l.local_table, l.global_table, l.drafter_table)
    }

    fn slice_lane_tables(&self, local_table: DeviceBuf, global_table: DeviceBuf, drafter_table: DeviceBuf) -> LaneTables {
        let l_stride = self.local_ring_pages * 4;
        let g_stride = self.max_global_pages * 4;
        let d_stride = self.drafter_ring_pages * 4;
        LaneTables {
            local: (0..NUM_LOCAL_LAYERS).map(|i| local_table.slice(i * l_stride, l_stride)).collect(),
            global: (0..NUM_GLOBAL_LAYERS).map(|i| global_table.slice(i * g_stride, g_stride)).collect(),
            drafter: (0..NUM_DFLASH2_LAYERS).map(|i| drafter_table.slice(i * d_stride, d_stride)).collect(),
        }
    }

    /// Global layer `gi` page IDs for `lane`.
    pub fn lane_global_pages(&self, lane: usize, gi: usize) -> &[u32] {
        &self.lanes[lane].global_slots[gi]
    }

    /// Allocates one physical page (lowest free ID so Elastic KV shrinks cleanly), reclaiming
    /// the unpinned idle lane holding the most pages if `free` is empty.
    fn alloc_page(&mut self, exclude_lane: usize) -> Option<u32> {
        if self.free.is_empty() {
            let victim = (0..self.lanes.len())
                .filter(|&l| l != exclude_lane && !self.pinned[l] && self.lanes[l].total_pages() > 0)
                .max_by_key(|&l| self.lanes[l].total_pages());
            let v = victim?;
            self.free_lane_all_internal(v);
            self.lost[v] = true;
        }
        let min_pos = self
            .free
            .iter()
            .enumerate()
            .min_by_key(|&(_, &p)| p)
            .map(|(i, _)| i)?;
        Some(self.free.swap_remove(min_pos))
    }

    fn free_lane_all_internal(&mut self, lane: usize) {
        let st = &mut self.lanes[lane];
        for layer in &mut st.local_slots {
            for slot in layer.iter_mut() {
                if let Some(p) = slot.take() {
                    self.free.push(p);
                }
            }
        }
        for layer in &mut st.global_slots {
            for p in layer.drain(..).rev() {
                self.free.push(p);
            }
        }
        for layer in &mut st.drafter_slots {
            for slot in layer.iter_mut() {
                if let Some(p) = slot.take() {
                    self.free.push(p);
                }
            }
        }
    }

    /// Counts how many additional physical pages are needed for `lane` to cover:
    /// - global `[0, global_end)` (if `global_end > 0`)
    /// - local ring `[local_lo, local_hi)` (across `local_layers` count, e.g. 1 or 40)
    /// - drafter ring `[drafter_lo, drafter_hi)` (across `drafter_layers` count, e.g. 0, 1, or 5)
    pub fn missing_pages(
        &self,
        lane: usize,
        global_end: usize,
        local_range: Option<(usize, usize, Option<usize>)>,
        drafter_range: Option<(usize, usize, Option<usize>)>,
    ) -> usize {
        let st = &self.lanes[lane];
        let mut need = 0usize;
        if global_end > 0 {
            let g_need = global_end.div_ceil(PAGE_POS).min(self.max_global_pages);
            for gi in 0..NUM_GLOBAL_LAYERS {
                need += g_need.saturating_sub(st.global_slots[gi].len());
            }
        }
        if let Some((lo, hi, single_layer)) = local_range {
            let lps = ring_logical_pages(lo, hi, self.local_ring_pages);
            let layers = match single_layer {
                Some(li) => li..li + 1,
                None => 0..NUM_LOCAL_LAYERS,
            };
            for li in layers {
                for &lp in &lps {
                    if st.local_slots[li][lp].is_none() {
                        need += 1;
                    }
                }
            }
        }
        if let Some((lo, hi, single_layer)) = drafter_range {
            let lps = ring_logical_pages(lo, hi, self.drafter_ring_pages);
            let layers = match single_layer {
                Some(di) => di..di + 1,
                None => 0..NUM_DFLASH2_LAYERS,
            };
            for di in layers {
                for &lp in &lps {
                    if st.drafter_slots[di][lp].is_none() {
                        need += 1;
                    }
                }
            }
        }
        need
    }

    /// Allocates missing global KV pages for all 8 global layers of `lane` up to `need` pages/layer.
    /// Returns `(complete, changed)`.
    fn ensure_global_slots(&mut self, lane: usize, need: usize) -> (bool, bool) {
        if need > self.max_global_pages {
            return (false, false);
        }
        // Check the minimum across ALL global layers (not just layer 0!): if an earlier call ran
        // out of free pages mid-way through gi=0..8 and triggered an elastic KV grow, layer 0 may
        // already have `need` pages while layers 1..7 are still short.
        let had = self.lanes[lane]
            .global_slots
            .iter()
            .map(Vec::len)
            .min()
            .unwrap_or(0);
        if had >= need {
            return (true, false);
        }
        let mut changed = false;
        for gi in 0..NUM_GLOBAL_LAYERS {
            while self.lanes[lane].global_slots[gi].len() < need {
                let Some(p) = self.alloc_page(lane) else {
                    return (false, changed);
                };
                self.lanes[lane].global_slots[gi].push(p);
                changed = true;
            }
        }
        (true, changed)
    }

    /// Ensures all 8 global layers of `lane` have pages covering `[0, end)`.
    pub fn ensure_global<B: ComputeBackend>(&mut self, b: &B, lane: usize, end: usize, stream: Stream) -> Result<bool> {
        let need = end.div_ceil(PAGE_POS);
        let (ok, changed) = self.ensure_global_slots(lane, need);
        if changed {
            self.upload_global(b, lane, stream)?;
        }
        Ok(ok)
    }

    fn upload_global<B: ComputeBackend>(&mut self, b: &B, lane: usize, stream: Stream) -> Result<()> {
        let stride = self.max_global_pages;
        let bytes = NUM_GLOBAL_LAYERS * stride * 4;
        let st = &mut self.lanes[lane];
        let stage = st.next_global_staging();
        let dev = st.global_table;
        unsafe {
            let dst = b.pinned_as_mut_slice(stage)?;
            for (gi, pages) in st.global_slots.iter().enumerate() {
                let base = gi * stride;
                for (i, &p) in pages.iter().enumerate() {
                    let off = (base + i) * 4;
                    dst[off..off + 4].copy_from_slice(&p.to_le_bytes());
                }
            }
        }
        b.memcpy_h2d_async(dev, 0, stage, 0, bytes, stream)
    }

    /// Ensures physical pages exist for local layer `li` (or all 40 if `li.is_none()`) covering `[lo, hi)`.
    pub fn ensure_local<B: ComputeBackend>(
        &mut self,
        b: &B,
        lane: usize,
        single_layer: Option<usize>,
        lo: usize,
        hi: usize,
        stream: Stream,
    ) -> Result<bool> {
        let lps = ring_logical_pages(lo, hi, self.local_ring_pages);
        if lps.is_empty() {
            return Ok(true);
        }
        let layers = match single_layer {
            Some(li) => li..li + 1,
            None => 0..NUM_LOCAL_LAYERS,
        };
        let mut changed = false;
        for li in layers.clone() {
            for &lp in &lps {
                if self.lanes[lane].local_slots[li][lp].is_none() {
                    let Some(p) = self.alloc_page(lane) else {
                        if changed {
                            self.upload_local(b, lane, single_layer, stream)?;
                        }
                        return Ok(false);
                    };
                    self.lanes[lane].local_slots[li][lp] = Some(p);
                    changed = true;
                }
            }
        }
        if changed {
            self.upload_local(b, lane, single_layer, stream)?;
        }
        Ok(true)
    }

    fn upload_local<B: ComputeBackend>(
        &mut self,
        b: &B,
        lane: usize,
        single_layer: Option<usize>,
        stream: Stream,
    ) -> Result<()> {
        let rp = self.local_ring_pages;
        let st = &mut self.lanes[lane];
        // Advance the 4-deep ring when uploading all 40 layers or starting layer 0 of a chunk;
        // layers 1..39 write to disjoint byte offsets in the same staging buffer.
        let advance = matches!(single_layer, None | Some(0));
        let stage = st.local_staging_buf(advance);
        let dev = st.local_table;
        let (byte_off, byte_len, range) = match single_layer {
            Some(li) => (li * rp * 4, rp * 4, li..li + 1),
            None => (0, NUM_LOCAL_LAYERS * rp * 4, 0..NUM_LOCAL_LAYERS),
        };
        unsafe {
            let dst = b.pinned_as_mut_slice(stage)?;
            for li in range {
                let base = li * rp;
                for (lp, opt) in st.local_slots[li].iter().enumerate() {
                    let v = opt.unwrap_or(0);
                    let off = (base + lp) * 4;
                    dst[off..off + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
        b.memcpy_h2d_async(dev, byte_off, stage, byte_off, byte_len, stream)
    }

    /// Evicts local ring pages of `lane` (one layer or all 40) that do not overlap `[keep_lo, keep_hi)`.
    pub fn evict_local(&mut self, lane: usize, single_layer: Option<usize>, keep_lo: usize, keep_hi: usize) {
        let keep = ring_logical_pages(keep_lo, keep_hi, self.local_ring_pages);
        let layers = match single_layer {
            Some(li) => li..li + 1,
            None => 0..NUM_LOCAL_LAYERS,
        };
        let st = &mut self.lanes[lane];
        for li in layers {
            for (lp, slot) in st.local_slots[li].iter_mut().enumerate() {
                if !keep.contains(&lp) {
                    if let Some(p) = slot.take() {
                        self.free.push(p);
                    }
                }
            }
        }
    }

    /// Ensures physical pages exist for DFlash 2 layer `di` (or all 5 if `None`) covering `[lo, hi)`.
    pub fn ensure_drafter<B: ComputeBackend>(
        &mut self,
        b: &B,
        lane: usize,
        single_layer: Option<usize>,
        lo: usize,
        hi: usize,
        stream: Stream,
    ) -> Result<bool> {
        let lps = ring_logical_pages(lo, hi, self.drafter_ring_pages);
        if lps.is_empty() {
            return Ok(true);
        }
        let layers = match single_layer {
            Some(di) => di..di + 1,
            None => 0..NUM_DFLASH2_LAYERS,
        };
        let mut changed = false;
        for di in layers {
            for &lp in &lps {
                if self.lanes[lane].drafter_slots[di][lp].is_none() {
                    let Some(p) = self.alloc_page(lane) else {
                        if changed {
                            self.upload_drafter(b, lane, single_layer, stream)?;
                        }
                        return Ok(false);
                    };
                    self.lanes[lane].drafter_slots[di][lp] = Some(p);
                    changed = true;
                }
            }
        }
        if changed {
            self.upload_drafter(b, lane, single_layer, stream)?;
        }
        Ok(true)
    }

    fn upload_drafter<B: ComputeBackend>(
        &mut self,
        b: &B,
        lane: usize,
        single_layer: Option<usize>,
        stream: Stream,
    ) -> Result<()> {
        let rp = self.drafter_ring_pages;
        let st = &mut self.lanes[lane];
        let stage = st.next_drafter_staging();
        let dev = st.drafter_table;
        let (byte_off, byte_len, range) = match single_layer {
            Some(di) => (di * rp * 4, rp * 4, di..di + 1),
            None => (0, NUM_DFLASH2_LAYERS * rp * 4, 0..NUM_DFLASH2_LAYERS),
        };
        unsafe {
            let dst = b.pinned_as_mut_slice(stage)?;
            for di in range {
                let base = di * rp;
                for (lp, opt) in st.drafter_slots[di].iter().enumerate() {
                    let v = opt.unwrap_or(0);
                    let off = (base + lp) * 4;
                    dst[off..off + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
        b.memcpy_h2d_async(dev, byte_off, stage, byte_off, byte_len, stream)
    }

    /// Evicts DFlash 2 ring pages of `lane` (one layer or all 5) outside `[keep_lo, keep_hi)`.
    pub fn evict_drafter(&mut self, lane: usize, single_layer: Option<usize>, keep_lo: usize, keep_hi: usize) {
        let keep = ring_logical_pages(keep_lo, keep_hi, self.drafter_ring_pages);
        let layers = match single_layer {
            Some(di) => di..di + 1,
            None => 0..NUM_DFLASH2_LAYERS,
        };
        let st = &mut self.lanes[lane];
        for di in layers {
            for (lp, slot) in st.drafter_slots[di].iter_mut().enumerate() {
                if !keep.contains(&lp) {
                    if let Some(p) = slot.take() {
                        self.free.push(p);
                    }
                }
            }
        }
    }

    /// Returns `lane`'s pages past position `keep` to the pool (when `keep == 0`, frees every page!).
    pub fn release(&mut self, lane: usize, keep: usize) {
        if keep == 0 {
            self.free_lane_all_internal(lane);
            return;
        }
        let keep_global = keep.div_ceil(PAGE_POS);
        let st = &mut self.lanes[lane];
        for layer in &mut st.global_slots {
            while layer.len() > keep_global {
                if let Some(p) = layer.pop() {
                    self.free.push(p);
                }
            }
        }
        let loc_lo = keep.saturating_sub(LOCAL_WINDOW);
        self.evict_local(lane, None, loc_lo, keep);
        let dr_lo = keep.saturating_sub(DFLASH2_WINDOW);
        self.evict_drafter(lane, None, dr_lo, keep);
    }

    /// Collects the ordered list of live physical page IDs for `lane` at position `pos`
    /// (for prefix snapshotting & restore).
    pub fn snapshot_pages(&self, lane: usize, pos: usize, include_drafter: bool) -> SnapshotPages {
        let st = &self.lanes[lane];
        let loc_lps = ring_logical_pages(pos.saturating_sub(LOCAL_WINDOW), pos, self.local_ring_pages);
        let mut local = Vec::with_capacity(NUM_LOCAL_LAYERS);
        for li in 0..NUM_LOCAL_LAYERS {
            let mut v = Vec::with_capacity(loc_lps.len());
            for &lp in &loc_lps {
                if let Some(p) = st.local_slots[li][lp] {
                    v.push((lp, p));
                }
            }
            local.push(v);
        }
        let g_cnt = pos.div_ceil(PAGE_POS);
        let mut global = Vec::with_capacity(NUM_GLOBAL_LAYERS);
        for gi in 0..NUM_GLOBAL_LAYERS {
            let take = g_cnt.min(st.global_slots[gi].len());
            global.push(st.global_slots[gi][..take].to_vec());
        }
        let mut drafter = Vec::new();
        if include_drafter {
            let dr_lps = ring_logical_pages(pos.saturating_sub(DFLASH2_WINDOW), pos, self.drafter_ring_pages);
            drafter.reserve(NUM_DFLASH2_LAYERS);
            for di in 0..NUM_DFLASH2_LAYERS {
                let mut v = Vec::with_capacity(dr_lps.len());
                for &lp in &dr_lps {
                    if let Some(p) = st.drafter_slots[di][lp] {
                        v.push((lp, p));
                    }
                }
                drafter.push(v);
            }
        }
        SnapshotPages { local, global, drafter }
    }

    /// Lanes whose pages were reclaimed since the last call (their device sequence is gone).
    pub fn take_lost(&mut self) -> Vec<usize> {
        let v: Vec<usize> = (0..self.lost.len()).filter(|&l| self.lost[l]).collect();
        for &l in &v {
            self.lost[l] = false;
        }
        v
    }

    /// True if `lane` currently holds any physical page `>= limit`.
    pub fn lane_has_pages_above(&self, lane: usize, limit: usize) -> bool {
        let limit = limit as u32;
        let l = &self.lanes[lane];
        l.local_slots.iter().any(|v| v.iter().flatten().any(|&p| p >= limit))
            || l.global_slots.iter().any(|v| v.iter().any(|&p| p >= limit))
            || l.drafter_slots.iter().any(|v| v.iter().flatten().any(|&p| p >= limit))
    }

    /// True if the pool is larger than `target_pages` and no lane currently holds any page `>= target_pages`.
    pub fn can_shrink_to(&self, target_pages: usize) -> bool {
        if self.n_pages <= target_pages {
            return false;
        }
        (0..self.lanes.len()).all(|lane| !self.lane_has_pages_above(lane, target_pages))
    }

    /// Resizes the pool's physical page count to `new_n_pages` (capped at `max_pages`).
    pub fn resize(&mut self, new_n_pages: usize) {
        let new_n = new_n_pages.min(self.max_pages);
        if new_n > self.n_pages {
            for p in (self.n_pages as u32..new_n as u32).rev() {
                self.free.push(p);
            }
            self.n_pages = new_n;
        } else if new_n < self.n_pages {
            debug_assert!(self.can_shrink_to(new_n));
            let limit = new_n as u32;
            self.free.retain(|&p| p < limit);
            self.n_pages = new_n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_lane(local_rp: usize, drafter_rp: usize) -> LaneState {
        let z = DeviceBuf { id: 0, offset_bytes: 0, size_bytes: 0 };
        let zp = PinnedBuf { id: 0, size_bytes: 0 };
        LaneState {
            local_slots: vec![vec![None; local_rp]; NUM_LOCAL_LAYERS],
            global_slots: vec![Vec::new(); NUM_GLOBAL_LAYERS],
            drafter_slots: vec![vec![None; drafter_rp]; NUM_DFLASH2_LAYERS],
            local_table: z,
            global_table: z,
            drafter_table: z,
            local_staging: [zp, zp, zp, zp],
            local_stage_idx: 0,
            global_staging: [zp, zp, zp, zp],
            global_stage_idx: 0,
            drafter_staging: [zp, zp, zp, zp],
            drafter_stage_idx: 0,
        }
    }

    #[test]
    fn unified_sliding_window_caps_pages_and_recycles() {
        let mut p = KvPool::new(512, 32_768, 1024);
        p.set_max_pages(2048);
        p.lanes.push(dummy_lane(p.local_ring_pages, p.drafter_ring_pages));
        p.pinned.push(true);
        p.lost.push(false);

        // Advance lane 0 through 10,000 tokens in 1024-token chunks: per-layer prefill borrow + evict.
        for chunk_idx in 0usize..10 {
            let pos0 = chunk_idx * 1024;
            let hi = pos0 + 1024;
            let lo = pos0.saturating_sub(LOCAL_WINDOW - 1);
            for li in 0..NUM_LOCAL_LAYERS {
                let lps = ring_logical_pages(lo, hi, p.local_ring_pages);
                for lp in lps {
                    if p.lanes[0].local_slots[li][lp].is_none() {
                        let pg = p.alloc_page(0).expect("page");
                        p.lanes[0].local_slots[li][lp] = Some(pg);
                    }
                }
                p.evict_local(0, Some(li), hi.saturating_sub(LOCAL_WINDOW), hi);
            }
        }
        // After 10,240 tokens, each of the 40 local layers holds exactly 4 pages (1024 / 256) = 160 pages total!
        assert_eq!(p.lanes[0].total_pages(), NUM_LOCAL_LAYERS * 4);
        assert_eq!(p.pages_used(), NUM_LOCAL_LAYERS * 4);

        // Reset lane to 0 -> 0 pages held!
        p.release(0, 0);
        assert_eq!(p.lanes[0].total_pages(), 0);
        assert_eq!(p.pages_free(), 512);
    }

    #[test]
    fn elastic_grow_and_shrink_packs_low_pages() {
        let mut p = KvPool::new(64, 32_768, 1024);
        p.set_max_pages(256);
        p.lanes.push(dummy_lane(p.local_ring_pages, p.drafter_ring_pages));
        p.pinned.push(true);
        p.lost.push(false);

        // Allocate 8 global layers x 8 pages = 64 pages (fills initial pool).
        for gi in 0..NUM_GLOBAL_LAYERS {
            for _ in 0..8 {
                let pg = p.alloc_page(0).unwrap();
                p.lanes[0].global_slots[gi].push(pg);
            }
        }
        assert_eq!(p.pages_free(), 0);

        // Grow 64 -> 128 pages, allocate 16 more global pages (pages 64..80).
        p.resize(128);
        assert_eq!(p.pages_total(), 128);
        for gi in 0..NUM_GLOBAL_LAYERS {
            for _ in 0..2 {
                let pg = p.alloc_page(0).unwrap();
                p.lanes[0].global_slots[gi].push(pg);
            }
        }
        assert!(!p.can_shrink_to(64));

        // Release back to 4 global pages (1024 tokens) -> high pages freed, shrink succeeds!
        p.release(0, 4 * PAGE_POS);
        assert!(p.can_shrink_to(64));
        p.resize(64);
        assert_eq!(p.pages_total(), 64);
        assert_eq!(p.pages_free(), 32);
        assert!(p.free.iter().all(|&x| x < 64));
    }

    #[test]
    fn ensure_global_partial_then_grow_allocates_all_layers() {
        let mut p = KvPool::new(100, 32_768, 1024);
        p.set_max_pages(512);
        p.lanes.push(dummy_lane(p.local_ring_pages, p.drafter_ring_pages));
        p.pinned.push(true);
        p.lost.push(false);

        // Request 34 pages/layer x 8 global layers = 272 pages when only 100 are free:
        // fills global layers 0, 1 (34 each) and partially fills layer 2 (32), leaving layers 3..7 empty.
        let (ok1, changed1) = p.ensure_global_slots(0, 34);
        assert!(!ok1);
        assert!(changed1);
        assert_eq!(p.lanes[0].global_slots[0].len(), 34);
        assert_eq!(p.lanes[0].global_slots[3].len(), 0);
        let missing = p.missing_pages(0, 34 * PAGE_POS, None, None);
        assert_eq!(missing, 272 - 100);

        // Grow elastic KV pool and retry: MUST allocate remaining pages for layers 2..7!
        p.resize(512);
        let (ok2, changed2) = p.ensure_global_slots(0, 34);
        assert!(ok2);
        assert!(changed2);
        let mut seen = std::collections::HashSet::new();
        for gi in 0..NUM_GLOBAL_LAYERS {
            assert_eq!(p.lanes[0].global_slots[gi].len(), 34, "global layer {gi} incomplete");
            for &pg in &p.lanes[0].global_slots[gi] {
                assert!(seen.insert(pg), "duplicate physical page {pg}");
            }
        }
        assert_eq!(seen.len(), 272);
    }

    #[test]
    fn lane_has_pages_above_identifies_only_blocking_lanes() {
        let mut p = KvPool::new(64, 32_768, 1024);
        p.set_max_pages(256);
        for _ in 0..2 {
            p.lanes.push(dummy_lane(p.local_ring_pages, p.drafter_ring_pages));
            p.pinned.push(false);
            p.lost.push(false);
        }
        // Lane 0: 4 global pages/layer = 32 low pages (0..32).
        assert!(p.ensure_global_slots(0, 4).0);
        // Lane 1: grow to 128 and take 6 global pages/layer = 48 pages (32..80 -> crosses 64).
        p.resize(128);
        assert!(p.ensure_global_slots(1, 6).0);
        assert!(!p.lane_has_pages_above(0, 64));
        assert!(p.lane_has_pages_above(1, 64));
        assert!(!p.can_shrink_to(64));

        // Releasing only the blocking lane (lane 1) lets the pool shrink while lane 0 stays intact.
        p.release(1, 0);
        assert!(!p.lane_has_pages_above(1, 64));
        assert!(p.can_shrink_to(64));
        p.resize(64);
        assert_eq!(p.pages_total(), 64);
        assert_eq!(p.lanes[0].total_pages(), 32);
    }
}
