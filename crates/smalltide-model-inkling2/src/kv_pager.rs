//! Paged global KV: every sequence's global-attention caches live in one fixed pool of pages
//! shared by all lanes (TideLM's paged KV; there through CUDA virtual memory, here through page
//! tables the attention kernels read - HIP virtual memory is not usable on the Windows driver).
//!
//! The pool holds `PAGE_POS` positions a page of every global buffer (8 layers x K, V and their
//! scales); a lane owns the pages its sequence reaches, listed in its device page table
//! (`kv_phys` in the kernels maps a position through it). The pool's VRAM never changes: a long
//! conversation takes pages other lanes are not using and never evicts experts; no lane is
//! special - any may grow to the full context while the pool has pages.
//!
//! `ensure` gives a lane pages up to a step's end before the step's device work (the table upload
//! is stream-ordered on the compute stream); an empty pool reclaims every page of an idle
//! (unpinned) lane, whose conversation then lives on in the prefix cache's host snapshot.
//! `release` returns a lane's pages past a position (reset, restore to a shorter prefix).

use smalltide_core::{ComputeBackend, DeviceBuf, PinnedBuf, Result, Stream};

/// Positions a page covers (must match `KV_PAGE_SHIFT` in the kernels).
pub const PAGE_POS: usize = 2048;

struct LanePages {
    pages: Vec<u32>,
    /// device page table `[max_pages]` u32 and its pinned staging copy
    table: DeviceBuf,
    staging: PinnedBuf,
}

pub struct KvPool {
    n_pages: usize,
    max_pages: usize,
    free: Vec<u32>,
    lanes: Vec<LanePages>,
    /// lanes serving a request (never reclaimed)
    pub pinned: Vec<bool>,
    /// lanes whose pages were reclaimed since the last `take_lost`
    lost: Vec<bool>,
}

impl KvPool {
    /// A pool of `n_pages` pages; every lane may hold up to `max_positions`.
    pub fn new(n_pages: usize, max_positions: usize) -> Self {
        Self {
            n_pages,
            max_pages: max_positions.div_ceil(PAGE_POS),
            free: (0..n_pages as u32).rev().collect(),
            lanes: Vec::new(),
            pinned: Vec::new(),
            lost: Vec::new(),
        }
    }

    pub fn pages_total(&self) -> usize {
        self.n_pages
    }

    pub fn pages_free(&self) -> usize {
        self.free.len()
    }

    /// Adds a lane; returns its device page table (pass it to the kernels of its global layers).
    pub fn add_lane<B: ComputeBackend>(&mut self, b: &B) -> Result<DeviceBuf> {
        let table = b.alloc_device(self.max_pages * 4)?;
        let staging = b.alloc_pinned(self.max_pages * 4)?;
        self.lanes.push(LanePages { pages: Vec::new(), table, staging });
        self.pinned.push(false);
        self.lost.push(false);
        Ok(table)
    }

    pub fn lane_pages(&self, lane: usize) -> &[u32] {
        &self.lanes[lane].pages
    }

    /// Gives `lane` pages up to position `end` (exclusive), reclaiming an idle lane when the pool
    /// is empty; the new table is uploaded on `stream`. False when it cannot.
    pub fn ensure<B: ComputeBackend>(&mut self, b: &B, lane: usize, end: usize, stream: Stream) -> Result<bool> {
        let need = end.div_ceil(PAGE_POS);
        if need > self.max_pages {
            return Ok(false);
        }
        let had = self.lanes[lane].pages.len();
        while self.lanes[lane].pages.len() < need {
            if self.free.is_empty() {
                // the idle lane holding the most pages gives them all back
                let victim = (0..self.lanes.len())
                    .filter(|&l| l != lane && !self.pinned[l] && !self.lanes[l].pages.is_empty())
                    .max_by_key(|&l| self.lanes[l].pages.len());
                let Some(v) = victim else {
                    self.upload(b, lane, had, stream)?;
                    return Ok(false);
                };
                let taken = std::mem::take(&mut self.lanes[v].pages);
                self.free.extend(taken.into_iter().rev());
                self.lost[v] = true;
            }
            let page = self.free.pop().expect("a free page");
            self.lanes[lane].pages.push(page);
        }
        self.upload(b, lane, had, stream)?;
        Ok(true)
    }

    /// Uploads `lane`'s table entries from `from` on (stream-ordered: later kernels see them).
    fn upload<B: ComputeBackend>(&mut self, b: &B, lane: usize, from: usize, stream: Stream) -> Result<()> {
        let l = &self.lanes[lane];
        let n = l.pages.len();
        if n <= from {
            return Ok(());
        }
        unsafe {
            let dst = b.pinned_as_mut_slice(l.staging)?;
            for (i, &p) in l.pages.iter().enumerate().skip(from) {
                dst[i * 4..i * 4 + 4].copy_from_slice(&p.to_le_bytes());
            }
        }
        b.memcpy_h2d_async(l.table, from * 4, l.staging, from * 4, (n - from) * 4, stream)
    }

    /// Returns `lane`'s pages past position `keep` to the pool.
    pub fn release(&mut self, lane: usize, keep: usize) {
        let keep_pages = keep.div_ceil(PAGE_POS);
        while self.lanes[lane].pages.len() > keep_pages {
            let p = self.lanes[lane].pages.pop().unwrap();
            self.free.push(p);
        }
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
