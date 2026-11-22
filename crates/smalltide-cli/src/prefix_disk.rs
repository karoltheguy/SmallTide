//! The disk prefix store (after TideLM's prefix.rs; `SMALLTIDE_PREFIX_DISK=0` turns it off): a
//! prompt's sequence state saved as a segment `[start, end)` of positions that extends its parent
//! segment ending at `start`, and restored by appending the rows of the deepest matching chain. A
//! conversation's next turn after a server restart, or after its snapshot left RAM, restores
//! instead of prefilling.
//!
//! A segment holds the global-attention layers' KV rows of its positions and, at its end, the
//! "tail": the sliding-window layers' rings and the short convolutions' histories (and the
//! drafter's), which exist only for the position the snapshot was taken at. So a restore ends at
//! a saved segment's end, never inside one.
//!
//! Layout on disk: `<dir>/<namespace>/index.json` and `<id>.seg` files. The namespace is a hash
//! of the snapshot layout (model, KV format, drafter), so a changed configuration starts an empty
//! store instead of reading rows it cannot use. Over the byte budget, least-recently-used leaf
//! segments go first.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use smalltide_model_inkling2::engine::SnapLayout;

const MAGIC: &[u8; 8] = b"STSEG001";

/// Two independent 64-bit hashes of a token prefix, updated a token at a time.
#[derive(Clone, Copy)]
struct Roll(u64, u64);

impl Roll {
    fn new() -> Self {
        Roll(0xcbf2_9ce4_8422_2325, 0x9E37_79B9_7F4A_7C15)
    }
    fn push(&mut self, t: u32) {
        self.0 = (self.0 ^ t as u64).wrapping_mul(0x0000_0100_0000_01B3);
        self.1 = (self.1 ^ t as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9).rotate_left(29);
    }
    fn hex(&self, len: usize) -> String {
        format!("{:016x}{:016x}{:x}", self.0, self.1 ^ (self.1 >> 31), len)
    }
}

/// The hash of `tokens[..n]` for every `n` in `ends` (ascending), in one pass.
fn prefix_hashes(tokens: &[u32], ends: &[usize]) -> Vec<String> {
    let mut out = Vec::with_capacity(ends.len());
    let mut r = Roll::new();
    let mut at = 0usize;
    for &e in ends {
        for &t in &tokens[at..e] {
            r.push(t);
        }
        at = e;
        out.push(r.hex(e));
    }
    out
}

pub fn prefix_hash(tokens: &[u32]) -> String {
    prefix_hashes(tokens, &[tokens.len()]).pop().unwrap()
}

#[derive(Clone, Debug)]
pub struct Segment {
    pub id: u64,
    pub parent: Option<u64>,
    pub start: usize,
    pub end: usize,
    /// hash of the prompt's first `end` tokens
    pub hash: String,
    pub bytes: u64,
    /// the store's clock at its last use
    pub used: u64,
}

pub struct PrefixDisk {
    dir: PathBuf,
    pub budget: u64,
    /// fewest new positions worth a save, and fewest gained positions worth a restore
    pub min_gain: usize,
    segs: BTreeMap<u64, Segment>,
    next: u64,
    clock: u64,
    layout: SnapLayout,
}

impl PrefixDisk {
    /// The store under `root` for snapshots of `layout`.
    pub fn open(root: &Path, layout: SnapLayout, budget: u64, min_gain: usize) -> std::io::Result<Self> {
        let sig = format!("v1-{:?}-{}-{}-{}", layout.per_pos, layout.n_engine, layout.fixed, layout.drafter_fixed);
        let mut r = Roll::new();
        for b in sig.bytes() {
            r.push(b as u32);
        }
        let name = format!("v1-{:016x}", r.0 ^ r.1);
        let dir = root.join(&name);
        std::fs::create_dir_all(&dir)?;
        // Stores of other snapshot layouts (another KV format, context size or prefill chunk)
        // cannot be read by this engine and would sit outside the budget: remove them.
        if let Ok(rd) = std::fs::read_dir(root) {
            for e in rd.flatten() {
                let n = e.file_name().to_string_lossy().into_owned();
                let stale = n.len() == 19 && n.starts_with("v1-") && n[3..].bytes().all(|b| b.is_ascii_hexdigit()) && n != name;
                if stale && e.path().join("index.json").is_file() && std::fs::remove_dir_all(e.path()).is_ok() {
                    println!("[prefix] disk: removed {n} (saved for a different engine configuration)");
                }
            }
        }
        let mut s = PrefixDisk { dir, budget, min_gain, segs: BTreeMap::new(), next: 0, clock: 0, layout };
        s.load_index();
        Ok(s)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn total_bytes(&self) -> u64 {
        self.segs.values().map(|s| s.bytes).sum()
    }

    pub fn segments(&self) -> usize {
        self.segs.len()
    }

    fn path(&self, id: u64) -> PathBuf {
        self.dir.join(format!("{id}.seg"))
    }

    fn load_index(&mut self) {
        let Ok(b) = std::fs::read(self.dir.join("index.json")) else { return };
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&b) else { return };
        if let Some(m) = v.get("segments").and_then(|x| x.as_object()) {
            for (k, s) in m {
                let Ok(id) = k.parse::<u64>() else { continue };
                let g = |f: &str| s.get(f).and_then(|x| x.as_u64());
                let (Some(start), Some(end), Some(bytes)) = (g("start"), g("end"), g("bytes")) else { continue };
                self.segs.insert(
                    id,
                    Segment {
                        id,
                        parent: g("parent"),
                        start: start as usize,
                        end: end as usize,
                        hash: s.get("hash").and_then(|x| x.as_str()).unwrap_or("").into(),
                        bytes,
                        used: g("used").unwrap_or(0),
                    },
                );
                self.next = self.next.max(id + 1);
            }
        }
        self.clock = v.get("clock").and_then(|x| x.as_u64()).unwrap_or(0);
        // a segment whose file is gone, or whose parent is, cannot be restored
        let ids: Vec<u64> = self.segs.keys().copied().collect();
        for id in ids {
            if !self.path(id).exists() {
                self.segs.remove(&id);
            }
        }
        loop {
            let orphans: Vec<u64> =
                self.segs.values().filter(|s| s.parent.is_some_and(|p| !self.segs.contains_key(&p))).map(|s| s.id).collect();
            if orphans.is_empty() {
                break;
            }
            for id in orphans {
                let _ = std::fs::remove_file(self.path(id));
                self.segs.remove(&id);
            }
        }
    }

    fn save_index(&self) -> std::io::Result<()> {
        let mut m = serde_json::Map::new();
        for s in self.segs.values() {
            m.insert(
                s.id.to_string(),
                serde_json::json!({"parent": s.parent, "start": s.start, "end": s.end, "hash": s.hash, "bytes": s.bytes, "used": s.used}),
            );
        }
        let tmp = self.dir.join("index.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(&serde_json::json!({"segments": m, "clock": self.clock}))?)?;
        std::fs::rename(tmp, self.dir.join("index.json"))
    }

    /// Root-first chain of segment `id`.
    fn chain(&self, id: u64) -> Vec<Segment> {
        let mut out = vec![];
        let mut cur = Some(id);
        while let Some(c) = cur {
            let Some(s) = self.segs.get(&c) else { break };
            out.push(s.clone());
            cur = s.parent;
        }
        out.reverse();
        out
    }

    /// The deepest segment whose prefix is `prompt`'s and that ends at or before `limit`.
    fn deepest(&self, prompt: &[u32], limit: usize) -> Option<Segment> {
        let limit = limit.min(prompt.len());
        let mut ends: Vec<usize> = self.segs.values().map(|s| s.end).filter(|&e| e <= limit).collect();
        ends.sort_unstable();
        ends.dedup();
        let hashes = prefix_hashes(prompt, &ends);
        let mut best: Option<&Segment> = None;
        for s in self.segs.values() {
            let Ok(i) = ends.binary_search(&s.end) else { continue };
            if hashes[i] != s.hash {
                continue;
            }
            if best.is_none_or(|b| s.end > b.end || (s.end == b.end && s.used > b.used)) {
                best = Some(s);
            }
        }
        best.cloned()
    }

    /// A restore plan for `prompt` when `have` of its positions are already at hand: the chain to
    /// load and the positions it gives, when that gains at least `min_gain` (one prompt token
    /// stays for its logits, so at most `len - 1`).
    pub fn plan_restore(&self, prompt: &[u32], have: usize) -> Option<(Vec<Segment>, usize)> {
        let seg = self.deepest(prompt, prompt.len().saturating_sub(1))?;
        if seg.end < have + self.min_gain {
            return None;
        }
        let chain = self.chain(seg.id);
        // a chain is whole: it starts at position 0 and each segment continues its parent
        let whole = chain.first().is_some_and(|s| s.start == 0) && chain.windows(2).all(|w| w[0].end == w[1].start);
        whole.then_some((chain, seg.end))
    }

    /// A save plan for the snapshot of `prompt` (all of it): the new segment for its positions
    /// past its deepest saved prefix, when at least `min_gain` are new, and the segments to evict
    /// first so the store stays in budget.
    pub fn plan_save(&mut self, prompt: &[u32], with_drafter: bool) -> Option<(Segment, Vec<u64>)> {
        let parent = self.deepest(prompt, prompt.len());
        let start = parent.as_ref().map(|p| p.end).unwrap_or(0);
        if prompt.len() < start + self.min_gain {
            return None;
        }
        let bytes = self.layout.rows_bytes(start, prompt.len(), with_drafter) as u64 + self.layout.tail_bytes(with_drafter) as u64;
        if bytes > self.budget {
            return None;
        }
        let keep: Vec<u64> = parent.as_ref().map(|p| self.chain(p.id).iter().map(|s| s.id).collect()).unwrap_or_default();
        // LRU leaves first, never the new segment's ancestors
        let mut evict = vec![];
        let mut total = self.total_bytes();
        let mut segs: Vec<Segment> = self.segs.values().cloned().collect();
        while total + bytes > self.budget {
            let v = segs
                .iter()
                .filter(|s| !keep.contains(&s.id) && !segs.iter().any(|c| c.parent == Some(s.id)))
                .min_by_key(|s| s.used)?;
            let vid = v.id;
            total -= v.bytes;
            evict.push(vid);
            segs.retain(|s| s.id != vid);
        }
        let id = self.next;
        self.next += 1;
        Some((Segment { id, parent: parent.map(|p| p.id), start, end: prompt.len(), hash: prefix_hash(prompt), bytes, used: 0 }, evict))
    }

    fn drop_segments(&mut self, ids: &[u64]) {
        for id in ids {
            let _ = std::fs::remove_file(self.path(*id));
            self.segs.remove(id);
        }
    }

    /// Writes `seg` from `host`, the bytes of a snapshot taken at `seg.end` (`with_drafter`: it
    /// holds the drafter's state), after evicting `evict`.
    pub fn save(&mut self, seg: Segment, evict: &[u64], host: &[u8], with_drafter: bool) -> std::io::Result<()> {
        self.drop_segments(evict);
        let l = &self.layout;
        let tail = l.tail_ranges(seg.end, with_drafter);
        let rows = l.row_ranges(seg.end, seg.start, seg.end, with_drafter);
        let header = serde_json::to_vec(&serde_json::json!({
            "start": seg.start, "end": seg.end, "hash": seg.hash, "drafter": with_drafter,
            "tail": tail.iter().map(|r| r.1).sum::<usize>(), "rows": rows.iter().map(|r| r.1).sum::<usize>(),
        }))?;
        let tmp = self.dir.join(format!("{}.seg.tmp", seg.id));
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(MAGIC)?;
            f.write_all(&(header.len() as u64).to_le_bytes())?;
            f.write_all(&header)?;
            for (off, len) in tail.iter().chain(&rows) {
                f.write_all(&host[*off..*off + *len])?;
            }
            f.flush()?;
        }
        std::fs::rename(&tmp, self.path(seg.id))?;
        self.clock += 1;
        let mut s = seg;
        s.used = self.clock;
        self.segs.insert(s.id, s);
        self.save_index()
    }

    /// Whether segment `id` holds the drafter's state (`None`: no such segment, or a bad file).
    pub fn segment_has_drafter(&self, id: u64) -> Option<bool> {
        self.read_header(self.segs.get(&id)?).map(|(_, d)| d)
    }

    /// Whether every segment of `chain` holds the drafter's state (read from their headers).
    pub fn chain_has_drafter(&self, chain: &[Segment]) -> bool {
        chain.iter().all(|s| self.read_header(s).is_some_and(|(_, d)| d))
    }

    fn read_header(&self, seg: &Segment) -> Option<(std::fs::File, bool)> {
        let mut f = std::fs::File::open(self.path(seg.id)).ok()?;
        let mut m = [0u8; 16];
        f.read_exact(&mut m).ok()?;
        if &m[..8] != MAGIC {
            return None;
        }
        let n = u64::from_le_bytes(m[8..16].try_into().unwrap()) as usize;
        if n > 1 << 20 {
            return None;
        }
        let mut h = vec![0u8; n];
        f.read_exact(&mut h).ok()?;
        let v: serde_json::Value = serde_json::from_slice(&h).ok()?;
        let u = |k: &str| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
        if u("start")? != seg.start || u("end")? != seg.end || v.get("hash")?.as_str()? != seg.hash {
            return None;
        }
        let d = v.get("drafter")?.as_bool()?;
        if u("tail")? != self.layout.tail_bytes(d) || u("rows")? != self.layout.rows_bytes(seg.start, seg.end, d) {
            return None;
        }
        Some((f, d))
    }

    /// Reads `chain` into `host`, the bytes of a snapshot at the chain's end (laid out
    /// `with_drafter`): every segment's rows, the last one's tail.
    pub fn load(&mut self, chain: &[Segment], host: &mut [u8], with_drafter: bool) -> std::io::Result<()> {
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
        let end = chain.last().ok_or_else(|| bad("empty chain"))?.end;
        for (i, seg) in chain.iter().enumerate() {
            let (mut f, d) = self.read_header(seg).ok_or_else(|| bad("segment file does not match the index"))?;
            if d != with_drafter {
                return Err(bad("segment and snapshot disagree about the drafter"));
            }
            let tail = self.layout.tail_ranges(end, with_drafter);
            if i + 1 == chain.len() {
                for (off, len) in tail {
                    f.read_exact(&mut host[off..off + len])?;
                }
            } else {
                let skip: usize = tail.iter().map(|r| r.1).sum();
                std::io::Seek::seek(&mut f, std::io::SeekFrom::Current(skip as i64))?;
            }
            for (off, len) in self.layout.row_ranges(end, seg.start, seg.end, with_drafter) {
                f.read_exact(&mut host[off..off + len])?;
            }
        }
        for id in chain.iter().map(|s| s.id).collect::<Vec<_>>() {
            self.clock += 1;
            if let Some(s) = self.segs.get_mut(&id) {
                s.used = self.clock;
            }
        }
        self.save_index()
    }

    /// Forgets `chain`'s last segment and its descendants (a file that failed to load).
    pub fn forget(&mut self, id: u64) {
        let mut ids = vec![id];
        let mut i = 0;
        while i < ids.len() {
            let p = ids[i];
            ids.extend(self.segs.values().filter(|s| s.parent == Some(p)).map(|s| s.id));
            i += 1;
        }
        self.drop_segments(&ids);
        let _ = self.save_index();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> SnapLayout {
        SnapLayout { fixed: 100, drafter_fixed: 10, per_pos: vec![3, 2, 1], n_engine: 2 }
    }

    /// A snapshot's bytes at `pos` whose rows depend on (region, position) only.
    fn snap(l: &SnapLayout, pos: usize, d: bool, salt: u8) -> Vec<u8> {
        let mut h = vec![0u8; l.total_bytes(pos, d)];
        for (off, len) in l.tail_ranges(pos, d) {
            h[off..off + len].fill(salt);
        }
        let regions = if d { l.per_pos.len() } else { l.n_engine };
        for r in 0..regions {
            for p in 0..pos {
                for (off, len) in l.region_range(pos, r, p, p + 1, d) {
                    for (j, b) in h[off..off + len].iter_mut().enumerate() {
                        *b = (r * 50 + p * 7 + j) as u8;
                    }
                }
            }
        }
        h
    }

    #[test]
    fn save_chain_and_restore() {
        let dir = std::env::temp_dir().join(format!("st_prefix_disk_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let l = layout();
        let mut st = PrefixDisk::open(&dir, l.clone(), 1 << 20, 4).unwrap();
        let prompt: Vec<u32> = (0..40).collect();
        // [0, 10), then [10, 30) as its child
        let (seg, ev) = st.plan_save(&prompt[..10], true).unwrap();
        assert_eq!((seg.start, seg.end, seg.parent, ev.len()), (0, 10, None, 0));
        st.save(seg, &ev, &snap(&l, 10, true, 1), true).unwrap();
        let (seg, ev) = st.plan_save(&prompt[..30], true).unwrap();
        assert_eq!((seg.start, seg.end, seg.parent), (10, 30, Some(0)));
        st.save(seg, &ev, &snap(&l, 30, true, 2), true).unwrap();
        // fewer than min_gain new positions: nothing to save
        assert!(st.plan_save(&prompt[..32], true).is_none());
        // a reopened store restores the chain into the same bytes as the snapshot at 30
        let mut st = PrefixDisk::open(&dir, l.clone(), 1 << 20, 4).unwrap();
        assert_eq!(st.segments(), 2);
        let (chain, n) = st.plan_restore(&prompt, 0).unwrap();
        assert_eq!((chain.len(), n), (2, 30));
        assert!(st.chain_has_drafter(&chain));
        let mut host = vec![0u8; l.total_bytes(30, true)];
        st.load(&chain, &mut host, true).unwrap();
        assert_eq!(host, snap(&l, 30, true, 2));
        // already holding 28 positions: 2 more are not worth a restore
        assert!(st.plan_restore(&prompt, 28).is_none());
        // a prompt that diverges at 20 restores the first segment only
        let mut other = prompt.clone();
        other[20] = 999;
        let (chain, n) = st.plan_restore(&other, 0).unwrap();
        assert_eq!((chain.len(), n), (1, 10));
        // the prompt's last token is never restored (its logits are needed)
        assert_eq!(st.plan_restore(&prompt[..30], 0).unwrap().1, 10);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn budget_evicts_least_recently_used_leaves() {
        let dir = std::env::temp_dir().join(format!("st_prefix_disk_test_b_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let l = layout();
        let one = (l.rows_bytes(0, 20, false) + l.tail_bytes(false)) as u64;
        let mut st = PrefixDisk::open(&dir, l.clone(), 2 * one + 10, 4).unwrap();
        let p = |salt: u32| -> Vec<u32> { (0..20).map(|i| i + salt * 1000).collect() };
        for s in 0..3u32 {
            let (seg, ev) = st.plan_save(&p(s), false).unwrap();
            assert_eq!(ev.len(), if s == 2 { 1 } else { 0 });
            st.save(seg, &ev, &snap(&l, 20, false, s as u8), false).unwrap();
        }
        assert_eq!(st.segments(), 2);
        assert!(st.plan_restore(&[p(0), vec![1]].concat(), 0).is_none(), "the oldest went");
        assert!(st.plan_restore(&[p(2), vec![1]].concat(), 0).is_some());
        assert!(st.total_bytes() <= st.budget);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
