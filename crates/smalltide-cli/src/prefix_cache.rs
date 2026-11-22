//! Prompt prefix cache for the server engine: many conversations, one live sequence per lane.
//!
//! Each engine lane holds one live sequence (`live[lane]` tokens). Snapshots capture the engine
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
    /// Taken at a shared-prefix point (system / tool prompt before the first user message):
    /// evicted last.
    shared: bool,
    payload: S,
    /// Routing demand of the request that produced this snapshot (prompt + answer): expected
    /// routing of the conversation's next turn (prompt-aware expert loading).
    demand: Option<Vec<f32>>,
}

pub struct PrefixCache<S> {
    /// Per lane: tokens whose state is live in that lane (positions `0..live[lane].len()`).
    pub live: Vec<Vec<u32>>,
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
    /// `lanes` live sequences; at most `max_snaps` snapshots holding at most `max_tokens` prefix
    /// tokens in total.
    pub fn new(lanes: usize, max_snaps: usize, max_tokens: usize) -> Self {
        Self {
            live: vec![Vec::new(); lanes.max(1)],
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

    /// Tokens reused by `plan` (0 for Reset).
    pub fn plan_reuse(plan: &PrefixPlan) -> usize {
        match plan {
            PrefixPlan::Extend { from } => *from,
            PrefixPlan::Restore { pos, .. } => *pos,
            PrefixPlan::Reset => 0,
        }
    }

    /// Among `lanes`, the one serving `new` with the most reuse (ties: no global KV copy, then
    /// the first listed). Returns `(lane, plan)`.
    pub fn best_lane(&self, lanes: &[usize], new: &[u32]) -> Option<(usize, PrefixPlan)> {
        let mut best: Option<(usize, PrefixPlan, (usize, bool))> = None;
        for &l in lanes {
            let p = self.plan(l, new);
            let key = (Self::plan_reuse(&p), !matches!(p, PrefixPlan::Restore { global: true, .. }));
            if best.as_ref().map_or(true, |b| key > b.2) {
                best = Some((l, p, key));
            }
        }
        best.map(|(l, p, _)| (l, p))
    }

    /// Decides how to serve `new` on `lane` (at least the last prompt token is always
    /// re-processed so the engine produces fresh logits).
    pub fn plan(&self, lane: usize, new: &[u32]) -> PrefixPlan {
        if new.is_empty() {
            return PrefixPlan::Reset;
        }
        let cached = &self.live[lane];
        let limit = new.len() - 1;
        let live = common_prefix(cached, new).min(limit);
        // Longest snapshot whose whole prefix starts `new`.
        let best = self
            .snaps
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.tokens.is_empty() && s.tokens.len() <= limit && new.starts_with(&s.tokens))
            .max_by_key(|(_, s)| s.tokens.len());
        if live == cached.len() && live > 0 && best.map_or(true, |(_, s)| s.tokens.len() <= live) {
            return PrefixPlan::Extend { from: live };
        }
        match best {
            Some((i, s)) => {
                let pos = s.tokens.len();
                // The device global KV is still this prefix's iff the live sequence starts with it.
                let global = !(cached.len() >= pos && cached[..pos] == s.tokens[..]);
                PrefixPlan::Restore { snap_idx: i, pos, global }
            }
            None => PrefixPlan::Reset,
        }
    }

    /// Marks every snapshot that `prompt` starts with as used now: a shared system prompt stays
    /// fresh while conversations keep extending it in their lanes (no restore needed there).
    pub fn touch_prefixes(&mut self, prompt: &[u32]) {
        self.tick += 1;
        let t = self.tick;
        for sn in self.snaps.iter_mut() {
            if !sn.tokens.is_empty() && prompt.starts_with(&sn.tokens) {
                sn.tick = t;
            }
        }
    }

    /// Eviction order: per-turn snapshots that a longer snapshot already extends (superseded by
    /// the same conversation's later turn), then other per-turn snapshots, then shared prefixes;
    /// least recently used first within a class.
    fn eviction_victim(&self) -> usize {
        let class = |i: usize| -> u8 {
            let sn = &self.snaps[i];
            if sn.shared {
                return 2;
            }
            let superseded = self
                .snaps
                .iter()
                .enumerate()
                .any(|(j, o)| j != i && o.tokens.len() > sn.tokens.len() && o.tokens.starts_with(&sn.tokens));
            if superseded { 0 } else { 1 }
        };
        (0..self.snaps.len()).min_by_key(|&i| (class(i), self.snaps[i].tick)).unwrap()
    }

    /// Records the outcome of `plan` once the engine state was rewound to `pos` (0 for Reset).
    /// Snapshots of other sequences are kept: they carry their own global KV.
    pub fn rewound_to(&mut self, lane: usize, plan: &PrefixPlan, pos: usize) {
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
                // The lane's live sequence becomes the snapshot's prefix.
                let toks = self.snaps[*snap_idx].tokens.clone();
                self.live[lane] = toks;
            }
            PrefixPlan::Reset => self.stats.misses += 1,
        }
        self.stats.reused_tokens += pos as u64;
        self.live[lane].truncate(pos);
    }

    /// Returns a payload to fill for a new snapshot of `pos` tokens (recycled, or `None` if the
    /// caller must allocate), evicting least-recently-used snapshots to stay within the budgets.
    pub fn take_payload(&mut self, pos: usize) -> Option<S> {
        while !self.snaps.is_empty()
            && (self.snaps.len() >= self.max_snaps || self.snapshot_tokens() + pos > self.max_tokens)
        {
            let i = self.eviction_victim();
            let s = self.snaps.remove(i);
            self.stats.evictions += 1;
            self.pool.push(s.payload);
        }
        self.pool.pop()
    }

    /// Stores a filled snapshot describing `lane`'s state after `live[lane][..pos]`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn insert(&mut self, lane: usize, pos: usize, payload: S) {
        self.insert_kind(lane, pos, payload, false)
    }

    /// `insert`; `shared`: a shared-prefix point (evicted last).
    pub fn insert_kind(&mut self, lane: usize, pos: usize, payload: S, shared: bool) {
        debug_assert!(pos <= self.live[lane].len());
        self.tick += 1;
        self.stats.snapshots_taken += 1;
        let tokens = self.live[lane][..pos].to_vec();
        if let Some(i) = self.snaps.iter().position(|s| s.tokens == tokens) {
            let shared = shared || self.snaps[i].shared;
            let old = std::mem::replace(&mut self.snaps[i], Snap { tokens, tick: self.tick, shared, payload, demand: None });
            self.pool.push(old.payload);
            return;
        }
        self.snaps.push(Snap { tokens, tick: self.tick, shared, payload, demand: None });
    }

    /// Attaches `demand` to the snapshot of exactly `prefix` (if any).
    pub fn set_demand(&mut self, prefix: &[u32], demand: &[f32]) {
        if let Some(sn) = self.snaps.iter_mut().find(|s| s.tokens == prefix) {
            sn.demand = Some(demand.to_vec());
        }
    }

    /// Demand attached to snapshot `idx`.
    pub fn snapshot_demand(&self, idx: usize) -> Option<&[f32]> {
        self.snaps[idx].demand.as_deref()
    }

    /// Drops `lane`'s live sequence (e.g. after an engine error). Snapshots carry their own global
    /// KV and stay valid.
    pub fn invalidate(&mut self, lane: usize) {
        self.live[lane].clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_on(pc: &mut PrefixCache<u32>, lane: usize, new: &[u32]) -> PrefixPlan {
        let plan = pc.plan(lane, new);
        let pos = PrefixCache::<u32>::plan_reuse(&plan);
        pc.rewound_to(lane, &plan, pos);
        pc.live[lane].extend_from_slice(&new[pos..]);
        plan
    }

    fn run(pc: &mut PrefixCache<u32>, new: &[u32]) -> PrefixPlan {
        run_on(pc, 0, new)
    }

    fn snap(pc: &mut PrefixCache<u32>, pos: usize, payload: u32) {
        let p = pc.take_payload(pos).unwrap_or(payload);
        pc.insert(0, pos, p);
    }

    #[test]
    fn extend_restore_reset() {
        let mut pc: PrefixCache<u32> = PrefixCache::new(1, 8, 1000);
        // Request 1: system prefix [1,2,3] then user [9,4,5]; snapshots at boundary 3 and end-1 = 5.
        assert_eq!(run(&mut pc, &[1, 2, 3, 9, 4, 5]), PrefixPlan::Reset);
        snap(&mut pc, 3, 0);
        snap(&mut pc, 5, 1);
        // Decode appended 7, 8.
        pc.live[0].extend_from_slice(&[7, 8]);

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
        let mut pc: PrefixCache<u32> = PrefixCache::new(1, 8, 1000);
        // Conversation A turn 1.
        run(&mut pc, &[1, 2, 3, 4]);
        snap(&mut pc, 3, 10);
        pc.live[0].extend_from_slice(&[50, 51]); // generated
        // Unrelated side request (e.g. title generation) in between.
        assert_eq!(run(&mut pc, &[7, 7, 7, 7, 7]), PrefixPlan::Reset);
        snap(&mut pc, 4, 11);
        // Conversation A turn 2: resumes from A's snapshot, global KV must be copied back.
        let plan = run(&mut pc, &[1, 2, 3, 4, 60, 61]);
        assert_eq!(plan, PrefixPlan::Restore { snap_idx: 0, pos: 3, global: true });
        assert_eq!(pc.live[0], vec![1, 2, 3, 4, 60, 61]);
        assert_eq!(pc.stats.cross_restores, 1);
    }

    #[test]
    fn shared_prefix_survives_long_conversation() {
        // Budget of 3 snapshots: a system prefix (10 tokens) and a conversation whose every turn
        // adds a per-turn snapshot. The shared prefix must survive and serve a new session.
        let mut pc: PrefixCache<u32> = PrefixCache::new(2, 3, 1_000_000);
        let sys: Vec<u32> = (0..10).collect();
        let mut conv = sys.clone();
        conv.extend(100..105);
        pc.live[0] = conv.clone();
        pc.insert_kind(0, 10, 0, true);
        for turn in 0..6u32 {
            conv.extend((200 + turn * 10)..(205 + turn * 10));
            pc.touch_prefixes(&conv);
            pc.live[0] = conv.clone();
            let n = conv.len() - 1;
            let p = pc.take_payload(n).unwrap_or(turn + 1);
            pc.insert(0, n, p);
        }
        let mut other = sys.clone();
        other.extend(900..905);
        match pc.plan(1, &other) {
            PrefixPlan::Restore { pos, .. } => assert_eq!(pos, 10),
            p => panic!("expected a restore of the shared prefix, got {p:?}"),
        }
    }

    #[test]
    fn lanes_pick_best_reuse() {
        let mut pc: PrefixCache<u32> = PrefixCache::new(3, 8, 1000);
        // Lane 1 serves conversation A; lane 2 conversation B.
        run_on(&mut pc, 1, &[1, 2, 3, 4]);
        run_on(&mut pc, 2, &[8, 8, 8]);
        // A's next turn extends lane 1 best; lanes 0 / 2 would reset.
        let (lane, plan) = pc.best_lane(&[0, 1, 2], &[1, 2, 3, 4, 5]).unwrap();
        assert_eq!((lane, plan), (1, PrefixPlan::Extend { from: 4 }));
        // A snapshot of A's prefix taken on lane 1 restores on lane 0 (global KV copied).
        let p = pc.take_payload(3).unwrap_or(7);
        pc.insert(1, 3, p);
        assert_eq!(pc.plan(0, &[1, 2, 3, 9]), PrefixPlan::Restore { snap_idx: 0, pos: 3, global: true });
        // Lane 1 holds 4 live tokens: rewinding to 3 needs the snapshot, but not its global KV.
        assert_eq!(pc.plan(1, &[1, 2, 3, 9]), PrefixPlan::Restore { snap_idx: 0, pos: 3, global: false });
    }

    #[test]
    fn lru_eviction_by_count_and_tokens() {
        let mut pc: PrefixCache<u32> = PrefixCache::new(1, 2, 1000);
        pc.live[0] = (0..100).collect();
        pc.insert(0, 10, 1);
        pc.insert(0, 20, 2);
        // Count-full: taking a payload evicts the LRU snapshot (pos 10).
        assert_eq!(pc.take_payload(30), Some(1));
        pc.insert(0, 30, 1);
        assert_eq!(pc.snapshot_positions(), vec![20, 30]);

        let mut pc: PrefixCache<u32> = PrefixCache::new(1, 8, 60);
        pc.live[0] = (0..100).collect();
        pc.insert(0, 20, 1);
        pc.insert(0, 30, 2);
        // 20 + 30 + 40 > 60: the oldest goes, then 30 + 40 > 60: the next one too.
        assert_eq!(pc.take_payload(40), Some(2));
        assert_eq!(pc.snapshot_positions(), Vec::<usize>::new());
    }
}
