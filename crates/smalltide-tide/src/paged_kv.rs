//! Paged KV Cache Manager with Prefix Caching (copy-on-write ref-counted 16-token blocks)
//! and Elastic KV <-> Expert VRAM Pool integration for concurrent requests up to 256K+ context.

use crate::cache::TideCache;
use smalltide_core::{Inkling2Config, KvDType, Result, SmallTideError};
use std::collections::HashMap;

pub const KV_PAGE_TOKENS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicalPageId(pub u32);

#[derive(Debug, Clone)]
struct PhysicalPage {
    ref_count: u32,
    prefix_hash: Option<u64>,
}

/// Manages physical global KV pages (shared across the 8 global attention layers, +1 if dspark)
/// with automatic prefix-block deduplication and Elastic VRAM reservation coupling to `TideCache`.
pub struct PagedKvManager {
    num_global_layers: usize,
    bytes_per_page_all_global_layers: usize,
    local_ring_bytes_per_seq: usize,
    max_physical_pages: usize,

    pages: Vec<PhysicalPage>,
    free_pages: Vec<PhysicalPageId>,
    /// `prefix_hash -> PhysicalPageId` for zero-copy shared system prompts across concurrent users.
    prefix_index: HashMap<u64, PhysicalPageId>,

    /// Per-sequence logical page tables (`seq_id -> Vec<PhysicalPageId>`).
    seq_tables: HashMap<u64, Vec<PhysicalPageId>>,
    /// Per-sequence current token count.
    seq_lengths: HashMap<u64, usize>,
    /// Granularity (bytes) used when notifying `TideCache::set_kv_reserved_bytes` to avoid thrashing.
    elastic_granularity_bytes: usize,
    last_reserved_bytes: usize,
}

impl PagedKvManager {
    pub fn new(
        model_cfg: &Inkling2Config,
        kv_dtype: KvDType,
        include_dspark_global_layer: bool,
        max_kv_vram_bytes: usize,
    ) -> Self {
        let local_layers_per_pass = model_cfg.local_layer_ids.len();
        // Physical global layers = num_physical_layers - local_layers_per_pass (= 4)
        // Loop core has 2 global layers (11, 17) run 3x => 1 + 2*3 + 1 = 8 logical global layers
        let global_in_loop = (model_cfg.loop_start_layer..model_cfg.loop_end_exclusive)
            .filter(|l| !model_cfg.is_local_layer(*l))
            .count();
        let global_non_loop = (model_cfg.num_physical_layers - local_layers_per_pass) - global_in_loop;
        let num_global_layers =
            global_non_loop + global_in_loop * model_cfg.loop_passes + usize::from(include_dspark_global_layer);
        let num_local_layers = model_cfg.num_logical_layers()
            - (global_non_loop + global_in_loop * model_cfg.loop_passes);

        let kv_elems_per_tok = 2 * model_cfg.num_key_value_heads * model_cfg.head_dim; // 2048
        let (bytes_per_tok_per_layer, scale_bytes_per_tok_per_layer) = match kv_dtype {
            KvDType::Fp8 => (kv_elems_per_tok, 2 * model_cfg.num_key_value_heads * 4), // FP8 + f32 scales
            KvDType::Q4 => (kv_elems_per_tok / 2, 2 * model_cfg.num_key_value_heads * 4 * 4), // 4-bit + block32 scales
            KvDType::Fp16 => (kv_elems_per_tok * 2, 0),
        };
        let bytes_per_page_one_layer =
            KV_PAGE_TOKENS * (bytes_per_tok_per_layer + scale_bytes_per_tok_per_layer);
        let bytes_per_page_all_global_layers = bytes_per_page_one_layer * num_global_layers;

        let local_ring_bytes_per_seq = num_local_layers
            * model_cfg.local_window_size
            * (bytes_per_tok_per_layer + scale_bytes_per_tok_per_layer);

        let max_physical_pages = (max_kv_vram_bytes / bytes_per_page_all_global_layers.max(1)).max(1);
        let mut free_pages = Vec::with_capacity(max_physical_pages);
        for i in (0..max_physical_pages).rev() {
            free_pages.push(PhysicalPageId(i as u32));
        }

        Self {
            num_global_layers,
            bytes_per_page_all_global_layers,
            local_ring_bytes_per_seq,
            max_physical_pages,
            pages: vec![
                PhysicalPage {
                    ref_count: 0,
                    prefix_hash: None,
                };
                max_physical_pages
            ],
            free_pages,
            prefix_index: HashMap::new(),
            seq_tables: HashMap::new(),
            seq_lengths: HashMap::new(),
            elastic_granularity_bytes: 32 * 1024 * 1024, // 32 MiB elastic steps
            last_reserved_bytes: 0,
        }
    }

    pub fn num_global_layers(&self) -> usize {
        self.num_global_layers
    }

    pub fn free_page_count(&self) -> usize {
        self.free_pages.len()
    }

    pub fn allocated_page_count(&self) -> usize {
        self.max_physical_pages - self.free_pages.len()
    }

    pub fn active_sequences(&self) -> usize {
        self.seq_tables.len()
    }

    /// Current total VRAM footprint across all active global KV pages + per-sequence local rings.
    pub fn total_kv_bytes(&self) -> usize {
        self.allocated_page_count() * self.bytes_per_page_all_global_layers
            + self.seq_tables.len() * self.local_ring_bytes_per_seq
    }

    /// Registers a new sequence, reusing any cached full-page prefix blocks via copy-on-write.
    /// Returns the number of prompt tokens already satisfied by prefix cache hits!
    pub fn register_sequence(
        &mut self,
        seq_id: u64,
        prompt_tokens: &[u32],
        tide_cache: Option<&mut TideCache>,
    ) -> Result<usize> {
        if self.seq_tables.contains_key(&seq_id) {
            return Err(SmallTideError::InvalidModel(format!(
                "sequence {seq_id} already registered"
            )));
        }

        let mut table = Vec::new();
        let mut cached_tokens = 0usize;
        let mut running_hash = 0xcbf2_9ce4_8422_2325u64;

        // Match full 16-token prefix blocks against cached physical pages
        let full_blocks = prompt_tokens.len() / KV_PAGE_TOKENS;
        for b in 0..full_blocks {
            let chunk = &prompt_tokens[b * KV_PAGE_TOKENS..(b + 1) * KV_PAGE_TOKENS];
            running_hash = hash_block(running_hash, chunk);
            if let Some(&pid) = self.prefix_index.get(&running_hash) {
                self.pages[pid.0 as usize].ref_count += 1;
                table.push(pid);
                cached_tokens += KV_PAGE_TOKENS;
            } else {
                break;
            }
        }

        self.seq_tables.insert(seq_id, table);
        self.seq_lengths.insert(seq_id, cached_tokens);
        self.sync_elastic_vram(tide_cache);
        Ok(cached_tokens)
    }

    /// Ensures `seq_id` has enough physical KV pages to hold `new_total_tokens`,
    /// allocating new pages and notifying `TideCache` elastically if KV grows.
    pub fn reserve_tokens(
        &mut self,
        seq_id: u64,
        new_total_tokens: usize,
        full_tokens_for_prefix_hash: Option<&[u32]>,
        tide_cache: Option<&mut TideCache>,
    ) -> Result<()> {
        let needed_pages = new_total_tokens.div_ceil(KV_PAGE_TOKENS);
        let current_pages = self
            .seq_tables
            .get(&seq_id)
            .ok_or_else(|| SmallTideError::InvalidModel(format!("unknown seq {seq_id}")))?
            .len();

        if needed_pages > current_pages {
            let add = needed_pages - current_pages;
            if add > self.free_pages.len() {
                return Err(SmallTideError::OutOfVram {
                    requested: add * self.bytes_per_page_all_global_layers,
                    available: self.free_pages.len() * self.bytes_per_page_all_global_layers,
                });
            }
            let table = self.seq_tables.get_mut(&seq_id).unwrap();
            for _ in 0..add {
                let pid = self.free_pages.pop().unwrap();
                self.pages[pid.0 as usize] = PhysicalPage {
                    ref_count: 1,
                    prefix_hash: None,
                };
                table.push(pid);
            }
        }

        // Index newly completed full 16-token blocks for prefix sharing across concurrent requests
        if let Some(tokens) = full_tokens_for_prefix_hash {
            let full_blocks = tokens.len() / KV_PAGE_TOKENS;
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            if let Some(table) = self.seq_tables.get(&seq_id) {
                for b in 0..full_blocks.min(table.len()) {
                    let chunk = &tokens[b * KV_PAGE_TOKENS..(b + 1) * KV_PAGE_TOKENS];
                    h = hash_block(h, chunk);
                    let pid = table[b];
                    let page = &mut self.pages[pid.0 as usize];
                    if page.prefix_hash.is_none() {
                        page.prefix_hash = Some(h);
                        self.prefix_index.entry(h).or_insert(pid);
                    }
                }
            }
        }

        self.seq_lengths.insert(seq_id, new_total_tokens);
        self.sync_elastic_vram(tide_cache);
        Ok(())
    }

    /// Releases a finished sequence's pages (decrementing prefix ref-counts) and allows
    /// the TideLM Expert Pool to expand back into reclaimed VRAM.
    pub fn release_sequence(&mut self, seq_id: u64, tide_cache: Option<&mut TideCache>) {
        self.seq_lengths.remove(&seq_id);
        if let Some(table) = self.seq_tables.remove(&seq_id) {
            for pid in table {
                let p = &mut self.pages[pid.0 as usize];
                p.ref_count = p.ref_count.saturating_sub(1);
                if p.ref_count == 0 {
                    if let Some(h) = p.prefix_hash.take() {
                        if self.prefix_index.get(&h) == Some(&pid) {
                            self.prefix_index.remove(&h);
                        }
                    }
                    self.free_pages.push(pid);
                }
            }
        }
        self.sync_elastic_vram(tide_cache);
    }

    /// Builds a padded flat GPU block table `[num_seqs, max_pages_in_batch]` of `u32` page IDs
    /// plus `seq_lens[num_seqs]` for batched PagedAttention kernel dispatch.
    pub fn build_gpu_block_table(&self, seq_ids: &[u64]) -> (Vec<u32>, Vec<u32>, usize) {
        let max_pages = seq_ids
            .iter()
            .filter_map(|id| self.seq_tables.get(id).map(|t| t.len()))
            .max()
            .unwrap_or(1)
            .max(1);
        let mut flat = vec![0u32; seq_ids.len() * max_pages];
        let mut lens = Vec::with_capacity(seq_ids.len());
        for (row, id) in seq_ids.iter().enumerate() {
            let len = self.seq_lengths.get(id).copied().unwrap_or(0) as u32;
            lens.push(len);
            if let Some(t) = self.seq_tables.get(id) {
                for (col, pid) in t.iter().enumerate() {
                    flat[row * max_pages + col] = pid.0;
                }
            }
        }
        (flat, lens, max_pages)
    }

    fn sync_elastic_vram(&mut self, tide_cache: Option<&mut TideCache>) {
        let Some(cache) = tide_cache else { return };
        let used = self.total_kv_bytes();
        let g = self.elastic_granularity_bytes;
        let rounded = used.div_ceil(g) * g;
        if rounded != self.last_reserved_bytes {
            self.last_reserved_bytes = rounded;
            let _ = cache.set_kv_reserved_bytes(rounded);
        }
    }
}

#[inline]
fn hash_block(mut h: u64, tokens: &[u32]) -> u64 {
    const PRIME: u64 = 0x1000_0000_01b3;
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}
