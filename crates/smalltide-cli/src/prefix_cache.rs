//! Prompt prefix cache for the server engine: many conversations, one live sequence.
//!
//! The engine holds exactly one live sequence (`cached` tokens). Snapshots capture the engine
//! state after a token prefix: the sliding-window KV rings, short-conv histories and the global
//! KV of that prefix. Each snapshot remembers its own tokens, so any request whose prompt starts
//! with a snapshot's tokens can resume from it - whichever conversation ran in between (chat UIs
//! interleave title / tag / autocomplete requests with the conversation's turns).
//!
//! Snapshots are taken at (a) the first user-message boundary of a prompt (shared system /
//! tool-declaration prefix across conversations) and (b) one token before every prompt end: the
//! last prompt token is always re-processed for fresh logits, so this serves exact repeats
//! (regenerate) as well as follow-up turns whose re-templated history diverges after the
//! previous prompt (e.g. when the client drops `reasoning_content`).
//!
//! Eviction: least recently used, bounded by a snapshot count and a total token budget (each
//! snapshot's host size is a fixed part for the rings plus a per-token part for the global KV).

/// How to bring the engine to a state from which `new[from..]` must be prefilled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixPlan {
    /// Live state already covers `new[..from]`.
    Extend { from: usize },
    /// Restore snapshot `snap_idx` (state after `new[..pos]`), then prefill from `pos`.
    /// `global` = the snapshot's global KV must be copied back too (its prefix is not the live
    /// sequence's, so the device global KV for positions `< pos` belongs to another sequence).
    Restore { snap_idx: usize, pos: usize, global: bool },
    /// Nothing reusable: reset and prefill everything.
    Reset,
}

#[derive(Debug, Default, Clone)]
pub struct PrefixStats {
    pub requests: u64,
    pub extend_hits: u64,
    pub restore_hits: u64,
    /// Restores from a snapshot of another sequence than the live one (global KV copied back).
    pub cross_restores: u64,
    pub misses: u64,
    pub reused_tokens: u64,
    pub prefilled_tokens: u64,
    pub snapshots_taken: u64,
    pub evictions: u64,
}

struct Snap<S> {
    tokens: Vec<u32>,
    tick: u64,
    payload: S,
}

pub struct PrefixCache<S> {
    /// Tokens whose state is live in the engine (positions `0..cached.len()`).
    pub cached: Vec<u32>,
    snaps: Vec<Snap<S>>,
    pool: Vec<S>,
    max_snaps: usize,
    max_tokens: usize,
    tick: u64,
    pub stats: PrefixStats,
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl<S> PrefixCache<S> {
    /// At most `max_snaps` snapshots holding at most `max_tokens` prefix tokens in total.
    pub fn new(max_snaps: usize, max_tokens: usize) -> Self {
        Self {
            cached: Vec::new(),
            snaps: Vec::new(),
            pool: Vec::new(),
            max_snaps: max_snaps.max(1),
            max_tokens: max_tokens.max(1),
            tick: 0,
            stats: PrefixStats::default(),
        }
    }

    pub fn snapshot_positions(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.snaps.iter().map(|s| s.tokens.len()).collect();
        v.sort_unstable();
        v
    }

    pub fn snapshot_tokens(&self) -> usize {
        self.snaps.iter().map(|s| s.tokens.len()).sum()
    }

    /// True if a snapshot of exactly `prefix` exists.
    pub fn has_snapshot_of(&self, prefix: &[u32]) -> bool {
        self.snaps.iter().any(|s| s.tokens == prefix)
    }

    pub fn snapshot(&self, idx: usize) -> &S {
        &self.snaps[idx].payload
    }

    /// Decides how to serve `new` (at least the last prompt token is always re-processed so the
    /// engine produces fresh logits).
    pub fn plan(&self, new: &[u32]) -> PrefixPlan {
        if new.is_empty() {
            return PrefixPlan::Reset;
        }
        let limit = new.len() - 1;
        let live = common_prefix(&self.cached, new).min(limit);
        // Longest snapshot whose whole prefix starts `new`.
        let best = self
            .snaps
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.tokens.is_empty() && s.tokens.len() <= limit && new.starts_with(&s.tokens))
            .max_by_key(|(_, s)| s.tokens.len());
        if live == self.cached.len() && live > 0 && best.map_or(true, |(_, s)| s.tokens.len() <= live) {
            return PrefixPlan::Extend { from: live };
        }
        match best {
            Some((i, s)) => {
                let pos = s.tokens.len();
                // The device global KV is still this prefix's iff the live sequence starts with it.
                let global = !(self.cached.len() >= pos && self.cached[..pos] == s.tokens[..]);
                PrefixPlan::Restore { snap_idx: i, pos, global }
            }
            None => PrefixPlan::Reset,
        }
    }

    /// Records the outcome of `plan` once the engine state was rewound to `pos` (0 for Reset).
    /// Snapshots of other sequences are kept: they carry their own global KV.
    pub fn rewound_to(&mut self, plan: &PrefixPlan, pos: usize) {
        self.stats.requests += 1;
        match plan {
            PrefixPlan::Extend { .. } => self.stats.extend_hits += 1,
            PrefixPlan::Restore { snap_idx, global, .. } => {
                self.stats.restore_hits += 1;
                if *global {
                    self.stats.cross_restores += 1;
                }
                self.tick += 1;
                self.snaps[*snap_idx].tick = self.tick;
                // The live sequence becomes the snapshot's prefix.
                self.cached.clear();
                self.cached.extend_from_slice(&self.snaps[*snap_idx].tokens);
            }
            PrefixPlan::Reset => self.stats.misses += 1,
        }
        self.stats.reused_tokens += pos as u64;
        self.cached.truncate(pos);
    }

    /// Returns a payload to fill for a new snapshot of `pos` tokens (recycled, or `None` if the
    /// caller must allocate), evicting least-recently-used snapshots to stay within the budgets.
    pub fn take_payload(&mut self, pos: usize) -> Option<S> {
        while !self.snaps.is_empty()
            && (self.snaps.len() >= self.max_snaps || self.snapshot_tokens() + pos > self.max_tokens)
        {
            let (i, _) = self.snaps.iter().enumerate().min_by_key(|(_, s)| s.tick).unwrap();
            let s = self.snaps.remove(i);
            self.stats.evictions += 1;
            self.pool.push(s.payload);
        }
        self.pool.pop()
    }

    /// Stores a filled snapshot describing the state after `cached[..pos]`.
    pub fn insert(&mut self, pos: usize, payload: S) {
        debug_assert!(pos <= self.cached.len());
        self.tick += 1;
        self.stats.snapshots_taken += 1;
        let tokens = self.cached[..pos].to_vec();
        if let Some(i) = self.snaps.iter().position(|s| s.tokens == tokens) {
            let old = std::mem::replace(&mut self.snaps[i], Snap { tokens, tick: self.tick, payload });
            self.pool.push(old.payload);
            return;
        }
        self.snaps.push(Snap { tokens, tick: self.tick, payload });
    }

    /// Drops the live sequence (e.g. after an engine error). Snapshots carry their own global KV
    /// and stay valid.
    pub fn invalidate(&mut self) {
        self.cached.clear();
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

    fn snap(pc: &mut PrefixCache<u32>, pos: usize, payload: u32) {
        let p = pc.take_payload(pos).unwrap_or(payload);
        pc.insert(pos, p);
    }

    #[test]
    fn extend_restore_reset() {
        let mut pc: PrefixCache<u32> = PrefixCache::new(8, 1000);
        // Request 1: system prefix [1,2,3] then user [9,4,5]; snapshots at boundary 3 and end-1 = 5.
        assert_eq!(run(&mut pc, &[1, 2, 3, 9, 4, 5]), PrefixPlan::Reset);
        snap(&mut pc, 3, 0);
        snap(&mut pc, 5, 1);
        // Decode appended 7, 8.
        pc.cached.extend_from_slice(&[7, 8]);

        // Follow-up that strictly extends the live sequence: pure extension.
        assert_eq!(run(&mut pc, &[1, 2, 3, 9, 4, 5, 7, 8, 10]), PrefixPlan::Extend { from: 8 });

        // Re-templated history diverges right after the previous prompt -> restore end-1 snapshot.
        let plan = run(&mut pc, &[1, 2, 3, 9, 4, 5, 11, 12]);
        assert_eq!(plan, PrefixPlan::Restore { snap_idx: 1, pos: 5, global: false });

        // Exact repeat of request 1 (regenerate) -> the end-1 snapshot.
        let plan = run(&mut pc, &[1, 2, 3, 9, 4, 5]);
        assert_eq!(plan, PrefixPlan::Restore { snap_idx: 1, pos: 5, global: false });

        // Nothing shared -> reset; snapshots survive.
        assert_eq!(run(&mut pc, &[5, 5, 5]), PrefixPlan::Reset);
        assert_eq!(pc.snapshot_positions(), vec![3, 5]);
        assert_eq!(pc.stats.misses, 2);
    }

    #[test]
    fn interleaved_conversations_keep_their_snapshots() {
        let mut pc: PrefixCache<u32> = PrefixCache::new(8, 1000);
        // Conversation A turn 1.
        run(&mut pc, &[1, 2, 3, 4]);
        snap(&mut pc, 3, 10);
        pc.cached.extend_from_slice(&[50, 51]); // generated
        // Unrelated side request (e.g. title generation) in between.
        assert_eq!(run(&mut pc, &[7, 7, 7, 7, 7]), PrefixPlan::Reset);
        snap(&mut pc, 4, 11);
        // Conversation A turn 2: resumes from A's snapshot, global KV must be copied back.
        let plan = run(&mut pc, &[1, 2, 3, 4, 60, 61]);
        assert_eq!(plan, PrefixPlan::Restore { snap_idx: 0, pos: 3, global: true });
        assert_eq!(pc.cached, vec![1, 2, 3, 4, 60, 61]);
        assert_eq!(pc.stats.cross_restores, 1);
    }

    #[test]
    fn lru_eviction_by_count_and_tokens() {
        let mut pc: PrefixCache<u32> = PrefixCache::new(2, 1000);
        pc.cached = (0..100).collect();
        pc.insert(10, 1);
        pc.insert(20, 2);
        // Count-full: taking a payload evicts the LRU snapshot (pos 10).
        assert_eq!(pc.take_payload(30), Some(1));
        pc.insert(30, 1);
        assert_eq!(pc.snapshot_positions(), vec![20, 30]);

        let mut pc: PrefixCache<u32> = PrefixCache::new(8, 60);
        pc.cached = (0..100).collect();
        pc.insert(20, 1);
        pc.insert(30, 2);
        // 20 + 30 + 40 > 60: the oldest goes, then 30 + 40 > 60: the next one too.
        assert_eq!(pc.take_payload(40), Some(2));
        assert_eq!(pc.snapshot_positions(), Vec::<usize>::new());
    }
}
