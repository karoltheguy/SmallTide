//! Prompt prefix cache for the single-sequence server engine.
//!
//! The engine holds exactly one live sequence (`cached` tokens). Its global KV is append-only, but
//! the sliding-window KV rings and short-conv histories are overwritten as tokens are processed,
//! so the live state can only be *rewound* to a position where a snapshot of that rewindable state
//! was taken. Snapshots are taken at (a) the first user-message boundary of a prompt (shared
//! system / tool-declaration prefix across conversations) and (b) every prompt end (multi-turn
//! chats whose re-templated history diverges right after the previous prompt, e.g. when the client
//! drops `reasoning_content`).
//!
//! Invariant: every stored snapshot's position is `<= cached.len()` and describes the state after
//! `cached[..pos]`, so restoring it is valid (global KV positions `< pos` hold the same tokens).

/// How to bring the engine to a state from which `new[from..]` must be prefilled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixPlan {
    /// Live state already covers `new[..from]`.
    Extend { from: usize },
    /// Restore the snapshot at `pos` (index into the snapshot list), then prefill from `pos`.
    Restore { snap_idx: usize, pos: usize },
    /// Nothing reusable: reset and prefill everything.
    Reset,
}

#[derive(Debug, Default, Clone)]
pub struct PrefixStats {
    pub requests: u64,
    pub extend_hits: u64,
    pub restore_hits: u64,
    pub misses: u64,
    pub reused_tokens: u64,
    pub prefilled_tokens: u64,
    pub snapshots_taken: u64,
}

pub struct PrefixCache<S> {
    /// Tokens whose state is live in the engine (positions `0..cached.len()`).
    pub cached: Vec<u32>,
    snaps: Vec<(usize, u64, S)>, // (pos, last_used tick, payload), sorted by pos
    pool: Vec<S>,
    max_snaps: usize,
    tick: u64,
    pub stats: PrefixStats,
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl<S> PrefixCache<S> {
    pub fn new(max_snaps: usize) -> Self {
        Self {
            cached: Vec::new(),
            snaps: Vec::new(),
            pool: Vec::new(),
            max_snaps: max_snaps.max(1),
            tick: 0,
            stats: PrefixStats::default(),
        }
    }

    pub fn snapshot_positions(&self) -> Vec<usize> {
        self.snaps.iter().map(|s| s.0).collect()
    }

    pub fn has_snapshot_at(&self, pos: usize) -> bool {
        self.snaps.iter().any(|s| s.0 == pos)
    }

    pub fn snapshot(&self, idx: usize) -> &S {
        &self.snaps[idx].2
    }

    /// Decides how to serve `new` (at least the last prompt token is always re-processed so the
    /// engine produces fresh logits).
    pub fn plan(&self, new: &[u32]) -> PrefixPlan {
        if new.is_empty() {
            return PrefixPlan::Reset;
        }
        let lcp = common_prefix(&self.cached, new);
        let max_reuse = lcp.min(new.len() - 1);
        if max_reuse == self.cached.len() && max_reuse > 0 {
            return PrefixPlan::Extend { from: max_reuse };
        }
        match self
            .snaps
            .iter()
            .enumerate()
            .filter(|(_, s)| s.0 <= max_reuse && s.0 > 0)
            .max_by_key(|(_, s)| s.0)
        {
            Some((i, s)) => PrefixPlan::Restore { snap_idx: i, pos: s.0 },
            None => PrefixPlan::Reset,
        }
    }

    /// Records the outcome of `plan` once the engine state was rewound to `pos` (0 for Reset):
    /// truncates the live sequence and recycles snapshots that are no longer on it.
    pub fn rewound_to(&mut self, plan: &PrefixPlan, pos: usize) {
        self.stats.requests += 1;
        match plan {
            PrefixPlan::Extend { .. } => self.stats.extend_hits += 1,
            PrefixPlan::Restore { snap_idx, .. } => {
                self.stats.restore_hits += 1;
                self.tick += 1;
                self.snaps[*snap_idx].1 = self.tick;
            }
            PrefixPlan::Reset => self.stats.misses += 1,
        }
        self.stats.reused_tokens += pos as u64;
        self.cached.truncate(pos);
        let mut keep = Vec::with_capacity(self.snaps.len());
        for s in self.snaps.drain(..) {
            if s.0 <= pos {
                keep.push(s);
            } else {
                self.pool.push(s.2);
            }
        }
        self.snaps = keep;
    }

    /// Returns a payload to fill for a new snapshot at `pos` (recycled, or `None` if the caller
    /// must allocate). Evicts the least-recently-used snapshot when the cache is full.
    pub fn take_payload(&mut self) -> Option<S> {
        if self.snaps.len() >= self.max_snaps {
            if let Some((i, _)) = self.snaps.iter().enumerate().min_by_key(|(_, s)| s.1) {
                let (_, _, p) = self.snaps.remove(i);
                return Some(p);
            }
        }
        self.pool.pop()
    }

    /// Stores a filled snapshot describing the state after `cached[..pos]`.
    pub fn insert(&mut self, pos: usize, payload: S) {
        debug_assert!(pos <= self.cached.len());
        self.tick += 1;
        self.stats.snapshots_taken += 1;
        if let Some(i) = self.snaps.iter().position(|s| s.0 == pos) {
            let old = std::mem::replace(&mut self.snaps[i], (pos, self.tick, payload));
            self.pool.push(old.2);
            return;
        }
        let at = self.snaps.partition_point(|s| s.0 < pos);
        self.snaps.insert(at, (pos, self.tick, payload));
    }

    /// Drops all state (e.g. after an engine error); payloads are kept for reuse.
    pub fn invalidate(&mut self) {
        self.cached.clear();
        for s in self.snaps.drain(..) {
            self.pool.push(s.2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(pc: &mut PrefixCache<u32>, new: &[u32]) -> PrefixPlan {
        let plan = pc.plan(new);
        let pos = match plan {
            PrefixPlan::Extend { from } => from,
            PrefixPlan::Restore { pos, .. } => pos,
            PrefixPlan::Reset => 0,
        };
        pc.rewound_to(&plan, pos);
        pc.cached.extend_from_slice(&new[pos..]);
        plan
    }

    #[test]
    fn extend_restore_reset() {
        let mut pc: PrefixCache<u32> = PrefixCache::new(4);
        // Request 1: system prefix [1,2,3] then user [9,4,5]; snapshot at boundary 3 and end 6.
        assert_eq!(run(&mut pc, &[1, 2, 3, 9, 4, 5]), PrefixPlan::Reset);
        let p = pc.take_payload().unwrap_or(0);
        pc.insert(3, p);
        let p = pc.take_payload().unwrap_or(1);
        pc.insert(6, p);
        // Decode appended 7, 8 (live state now covers 8 tokens).
        pc.cached.extend_from_slice(&[7, 8]);

        // Follow-up that strictly extends the live sequence: pure extension.
        assert_eq!(run(&mut pc, &[1, 2, 3, 9, 4, 5, 7, 8, 10]), PrefixPlan::Extend { from: 8 });

        // Re-templated history diverges right after the previous prompt end -> restore at 6.
        let plan = run(&mut pc, &[1, 2, 3, 9, 4, 5, 11, 12]);
        assert_eq!(plan, PrefixPlan::Restore { snap_idx: 1, pos: 6 });
        assert_eq!(pc.snapshot_positions(), vec![3, 6]);

        // Different conversation sharing only the system prefix -> restore at boundary 3;
        // the snapshot at 6 is no longer on the live sequence and is recycled.
        let plan = run(&mut pc, &[1, 2, 3, 9, 20, 21]);
        assert_eq!(plan, PrefixPlan::Restore { snap_idx: 0, pos: 3 });
        assert_eq!(pc.snapshot_positions(), vec![3]);

        // Identical prompt: must re-process the last token, so rewinding needs a snapshot.
        pc.insert(6, 99);
        assert_eq!(run(&mut pc, &[1, 2, 3, 9, 20, 21]), PrefixPlan::Restore { snap_idx: 0, pos: 3 });

        // Nothing shared -> reset.
        assert_eq!(run(&mut pc, &[5, 5, 5]), PrefixPlan::Reset);
        assert!(pc.snapshot_positions().is_empty());
        assert_eq!(pc.stats.misses, 2);
    }

    #[test]
    fn lru_eviction_recycles_payloads() {
        let mut pc: PrefixCache<u32> = PrefixCache::new(2);
        pc.cached = (0..100).collect();
        pc.insert(10, 1);
        pc.insert(20, 2);
        // Full: taking a payload evicts the LRU snapshot (pos 10).
        assert_eq!(pc.take_payload(), Some(1));
        pc.insert(30, 1);
        assert_eq!(pc.snapshot_positions(), vec![20, 30]);
    }
}
