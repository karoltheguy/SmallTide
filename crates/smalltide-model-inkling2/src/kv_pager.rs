//! Paged global KV: every lane's global-attention caches are paged out of one fixed pool of
//! physical pages (TideLM's `PagedKv`, here on HIP virtual memory).
//!
//! Each global cache keeps the layout the kernels address (position-major `[slot][head][..]`),
//! but per lane it is a reserved address range of the lane's full context, not an allocation: a
//! lane is backed by real pages only as far as its sequence reaches. A page is `PAGE_POS`
//! positions of every global buffer (8 layers x K, V, K scales, V scales): one physical allocation
//! per buffer, all created once at startup, so the pool's VRAM never changes - a long
//! conversation takes pages other lanes are not using, it never evicts experts.
//!
//! `ensure` maps a lane's pages up to a step's end before the step's device work; when the pool
//! is empty it reclaims every page of an idle (unpinned) lane, whose conversation then lives on
//! in the prefix cache's host snapshot. `release` returns a lane's pages past a position (after a
//! reset or a restore to a shorter prefix). Unmapping needs the device to be done with the range:
//! callers synchronize first.

use smalltide_core::{ComputeBackend, DeviceBuf, Result};

/// Positions a page covers (2 MiB of K or V a global layer, 64 KiB of its scales).
pub const PAGE_POS: usize = 2048;

struct LaneVa {
    /// (address range, bytes a page) of each global buffer, in pool buffer order
    bufs: Vec<(DeviceBuf, usize)>,
    /// pool page ids mapped at page slots 0, 1, ...
    pages: Vec<usize>,
}

pub struct KvPager {
    /// bytes a page of each buffer
    layout: Vec<usize>,
    /// physical allocations of each pool page, in buffer order
    phys: Vec<Vec<u64>>,
    free: Vec<usize>,
    lanes: Vec<LaneVa>,
    /// lanes serving a request (never reclaimed)
    pub pinned: Vec<bool>,
    /// lanes whose pages were reclaimed since the last `take_lost`
    lost: Vec<bool>,
}

impl KvPager {
    /// A pool of `n_pages` pages of buffers with `layout` bytes a page each.
    pub fn new<B: ComputeBackend>(b: &B, layout: Vec<usize>, n_pages: usize) -> Result<Self> {
        let mut phys = Vec::with_capacity(n_pages);
        for _ in 0..n_pages {
            let mut v = Vec::with_capacity(layout.len());
            for &bytes in &layout {
                v.push(b.vmm_create(bytes)?);
            }
            phys.push(v);
        }
        Ok(Self { layout, free: (0..n_pages).rev().collect(), phys, lanes: Vec::new(), pinned: Vec::new(), lost: Vec::new() })
    }

    pub fn page_bytes(&self) -> usize {
        self.layout.iter().sum()
    }

    pub fn pages_total(&self) -> usize {
        self.phys.len()
    }

    pub fn pages_free(&self) -> usize {
        self.free.len()
    }

    /// Registers the next lane's address ranges (`(range, bytes a page)` in layout order).
    pub fn add_lane(&mut self, bufs: Vec<(DeviceBuf, usize)>) {
        debug_assert_eq!(bufs.iter().map(|b| b.1).collect::<Vec<_>>(), self.layout);
        self.lanes.push(LaneVa { bufs, pages: Vec::new() });
        self.pinned.push(false);
        self.lost.push(false);
    }

    /// Positions currently backed for `lane`.
    pub fn mapped_positions(&self, lane: usize) -> usize {
        self.lanes[lane].pages.len() * PAGE_POS
    }

    pub fn lane_pages(&self, lane: usize) -> usize {
        self.lanes[lane].pages.len()
    }

    /// Touches every pool page once (maps it at lane 0's first slot and zeroes it through
    /// `zero(range, bytes)`), so the driver commits the pool before the expert pools are sized.
    pub fn touch_all<B: ComputeBackend>(&mut self, b: &B, mut zero: impl FnMut(DeviceBuf, usize) -> Result<()>,
        mut sync: impl FnMut() -> Result<()>) -> Result<()> {
        let Some(l0) = self.lanes.first() else { return Ok(()) };
        debug_assert!(l0.pages.is_empty());
        let bufs = l0.bufs.clone();
        for p in 0..self.phys.len() {
            for (i, &(va, bytes)) in bufs.iter().enumerate() {
                b.vmm_map(va, 0, self.phys[p][i])?;
                zero(va.slice(0, bytes), bytes)?;
            }
            sync()?;
            for &(va, bytes) in &bufs {
                b.vmm_unmap(va, 0, bytes)?;
            }
        }
        Ok(())
    }

    /// Maps `lane`'s pages up to position `end` (exclusive). An empty pool reclaims an idle lane
    /// (`quiesce` is called once before its ranges are unmapped). Returns false when the pool
    /// cannot back `end` positions for this lane.
    pub fn ensure<B: ComputeBackend>(&mut self, b: &B, lane: usize, end: usize, mut quiesce: impl FnMut() -> Result<()>) -> Result<bool> {
        let need = end.div_ceil(PAGE_POS);
        let mut quiesced = false;
        while self.lanes[lane].pages.len() < need {
            if self.free.is_empty() {
                // the idle lane holding the most pages gives them all back
                let victim = (0..self.lanes.len())
                    .filter(|&l| l != lane && !self.pinned[l] && !self.lanes[l].pages.is_empty())
                    .max_by_key(|&l| self.lanes[l].pages.len());
                let Some(v) = victim else { return Ok(false) };
                if !quiesced {
                    quiesce()?;
                    quiesced = true;
                }
                self.unmap_from(b, v, 0)?;
                self.lost[v] = true;
            }
            let page = self.free.pop().expect("a free page");
            let slot = self.lanes[lane].pages.len();
            let bufs = self.lanes[lane].bufs.clone();
            for (i, &(va, bytes)) in bufs.iter().enumerate() {
                b.vmm_map(va, slot * bytes, self.phys[page][i])?;
            }
            self.lanes[lane].pages.push(page);
        }
        Ok(true)
    }

    /// Returns `lane`'s pages past position `keep` to the pool. The device must be done with
    /// them (callers synchronize first).
    pub fn release<B: ComputeBackend>(&mut self, b: &B, lane: usize, keep: usize) -> Result<()> {
        self.unmap_from(b, lane, keep.div_ceil(PAGE_POS))
    }

    fn unmap_from<B: ComputeBackend>(&mut self, b: &B, lane: usize, keep_pages: usize) -> Result<()> {
        while self.lanes[lane].pages.len() > keep_pages {
            let page = self.lanes[lane].pages.pop().unwrap();
            let slot = self.lanes[lane].pages.len();
            for &(va, bytes) in &self.lanes[lane].bufs {
                b.vmm_unmap(va, slot * bytes, bytes)?;
            }
            self.free.push(page);
        }
        Ok(())
    }

    /// Lanes whose pages were reclaimed since the last call (their device sequence is gone).
    pub fn take_lost(&mut self) -> Vec<usize> {
        let v: Vec<usize> = (0..self.lost.len()).filter(|&l| self.lost[l]).collect();
        for &l in &v {
            self.lost[l] = false;
        }
        v
    }
}
