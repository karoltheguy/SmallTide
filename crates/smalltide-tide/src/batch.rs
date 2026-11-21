//! Continuous Batching & Concurrent Request Scheduler (`ContinuousBatchScheduler`).
//!
//! Schedules multiple concurrent users/requests into parallel GPU batches:
//! - Mixes chunked prefill tokens (`chunk_size = 512..1024`) with concurrent decode tokens
//! - Supports `dspark` speculative verification tokens inside the exact same batched path
//! - Coordinates admission control with `PagedKvManager` (automatic prefix sharing + elastic VRAM!)

use crate::cache::TideCache;
use crate::paged_kv::PagedKvManager;
use smalltide_core::Result;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeqStatus {
    Waiting,
    Prefilling,
    Decoding,
    Finished,
}

#[derive(Debug, Clone)]
pub struct SequenceRequest {
    pub seq_id: u64,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    /// Number of prompt tokens already prefilled (starts at prefix-cache hit count).
    pub prefill_cursor: usize,
    pub status: SeqStatus,
}

impl SequenceRequest {
    pub fn total_processed_tokens(&self) -> usize {
        self.prefill_cursor + self.generated_tokens.len()
    }
}

/// One token entry inside a parallel GPU execution batch.
#[derive(Debug, Clone, Copy)]
pub struct BatchTokenItem {
    pub seq_id: u64,
    pub token_id: u32,
    pub position: usize,
    /// True if this token is the last token of a sequence's chunk/step and needs logits sampled.
    pub compute_logits: bool,
}

/// A scheduled parallel execution step mixing chunked prefill + concurrent decode sequences.
#[derive(Debug, Clone, Default)]
pub struct StepBatch {
    pub items: Vec<BatchTokenItem>,
    pub active_seq_ids: Vec<u64>,
    pub num_prefill_tokens: usize,
    pub num_decode_tokens: usize,
}

impl StepBatch {
    pub fn batch_size(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

pub struct ContinuousBatchScheduler {
    max_concurrent_seqs: usize,
    max_batch_tokens: usize,
    prefill_chunk_size: usize,

    waiting: VecDeque<SequenceRequest>,
    active: Vec<SequenceRequest>,
    next_seq_id: u64,
}

impl ContinuousBatchScheduler {
    pub fn new(
        max_concurrent_seqs: usize,
        max_batch_tokens: usize,
        prefill_chunk_size: usize,
    ) -> Self {
        Self {
            max_concurrent_seqs: max_concurrent_seqs.max(1),
            max_batch_tokens: max_batch_tokens.max(1),
            prefill_chunk_size: prefill_chunk_size.max(16),
            waiting: VecDeque::new(),
            active: Vec::new(),
            next_seq_id: 1,
        }
    }

    /// Enqueues a new concurrent user request and returns its assigned `seq_id`.
    pub fn enqueue(&mut self, prompt_tokens: Vec<u32>, max_new_tokens: usize) -> u64 {
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;
        self.waiting.push_back(SequenceRequest {
            seq_id,
            prompt_tokens,
            generated_tokens: Vec::new(),
            max_new_tokens,
            prefill_cursor: 0,
            status: SeqStatus::Waiting,
        });
        seq_id
    }

    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    /// Admits waiting requests up to `max_concurrent_seqs` & KV capacity, then packs
    /// concurrent decode tokens + chunked prefill tokens into one parallel `StepBatch`.
    pub fn schedule_next_step(
        &mut self,
        kv: &mut PagedKvManager,
        mut tide_cache: Option<&mut TideCache>,
    ) -> Result<StepBatch> {
        // 1. Admit waiting requests while slots & KV pages are available
        while self.active.len() < self.max_concurrent_seqs {
            let Some(mut req) = self.waiting.pop_front() else {
                break;
            };
            match kv.register_sequence(req.seq_id, &req.prompt_tokens, tide_cache.as_deref_mut()) {
                Ok(cached_prefix) => {
                    // Always re-evaluate at least the final prompt token if 100% prefix matched
                    req.prefill_cursor = cached_prefix.min(req.prompt_tokens.len().saturating_sub(1));
                    req.status = SeqStatus::Prefilling;
                    self.active.push(req);
                }
                Err(_) => {
                    self.waiting.push_front(req);
                    break;
                }
            }
        }

        let mut batch = StepBatch::default();
        let mut budget = self.max_batch_tokens;

        // 2. Prioritize all active decoding sequences first (low latency for concurrent users!)
        for req in &mut self.active {
            if req.status != SeqStatus::Decoding || budget == 0 {
                continue;
            }
            let next_pos = req.total_processed_tokens();
            let last_tok = *req
                .generated_tokens
                .last()
                .or_else(|| req.prompt_tokens.last())
                .unwrap_or(&0);
            kv.reserve_tokens(req.seq_id, next_pos + 1, None, tide_cache.as_deref_mut())?;
            batch.items.push(BatchTokenItem {
                seq_id: req.seq_id,
                token_id: last_tok,
                position: next_pos,
                compute_logits: true,
            });
            batch.active_seq_ids.push(req.seq_id);
            batch.num_decode_tokens += 1;
            budget -= 1;
        }

        // 3. Fill remaining token budget with chunked prefill tokens (piggybacking on same weights!)
        for req in &mut self.active {
            if req.status != SeqStatus::Prefilling || budget == 0 {
                continue;
            }
            let remaining = req.prompt_tokens.len().saturating_sub(req.prefill_cursor);
            if remaining == 0 {
                req.status = SeqStatus::Decoding;
                continue;
            }
            let take = remaining.min(self.prefill_chunk_size).min(budget);
            let new_end = req.prefill_cursor + take;
            kv.reserve_tokens(
                req.seq_id,
                new_end,
                Some(&req.prompt_tokens[..new_end]),
                tide_cache.as_deref_mut(),
            )?;

            for pos in req.prefill_cursor..new_end {
                let is_last_prompt_tok = pos + 1 == req.prompt_tokens.len();
                batch.items.push(BatchTokenItem {
                    seq_id: req.seq_id,
                    token_id: req.prompt_tokens[pos],
                    position: pos,
                    compute_logits: is_last_prompt_tok,
                });
            }
            req.prefill_cursor = new_end;
            if req.prefill_cursor >= req.prompt_tokens.len() {
                req.status = SeqStatus::Decoding;
            }
            if !batch.active_seq_ids.contains(&req.seq_id) {
                batch.active_seq_ids.push(req.seq_id);
            }
            batch.num_prefill_tokens += take;
            budget -= take;
        }

        Ok(batch)
    }

    /// Appends sampled tokens for sequences that produced logits in the current step,
    /// automatically releasing KV pages (and letting Expert Tide rise!) when a sequence finishes.
    pub fn commit_sampled_tokens(
        &mut self,
        sampled: &[(u64, u32)],
        eos_token_id: u32,
        kv: &mut PagedKvManager,
        mut tide_cache: Option<&mut TideCache>,
    ) -> Vec<u64> {
        let mut finished_ids = Vec::new();
        for &(seq_id, tok) in sampled {
            if let Some(req) = self.active.iter_mut().find(|r| r.seq_id == seq_id) {
                req.generated_tokens.push(tok);
                if tok == eos_token_id || req.generated_tokens.len() >= req.max_new_tokens {
                    req.status = SeqStatus::Finished;
                    finished_ids.push(seq_id);
                }
            }
        }
        if !finished_ids.is_empty() {
            for &fid in &finished_ids {
                kv.release_sequence(fid, tide_cache.as_deref_mut());
            }
            self.active.retain(|r| r.status != SeqStatus::Finished);
        }
        finished_ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smalltide_core::{Inkling2Config, KvDType, TideConfig};

    const GIB: usize = 1024 * 1024 * 1024;

    #[test]
    fn test_concurrent_requests_prefix_sharing_and_elastic_kv() {
        let cfg = Inkling2Config::default();
        let mut tide = TideCache::new(cfg.clone(), TideConfig::default(), 11 * GIB, 16 * GIB);
        let mut kv = PagedKvManager::new(&cfg, KvDType::Fp8, false, 6 * GIB);
        let mut batcher = ContinuousBatchScheduler::new(4, 256, 64);

        // Two concurrent requests sharing a 48-token (3 full 16-token pages) system prompt!
        let mut p1: Vec<u32> = (10..58).collect();
        p1.extend_from_slice(&[1001, 1002, 1003, 1004]);
        let mut p2: Vec<u32> = (10..58).collect();
        p2.extend_from_slice(&[2001, 2002, 2003, 2004]);

        let s1 = batcher.enqueue(p1, 2);
        let b1 = batcher.schedule_next_step(&mut kv, Some(&mut tide)).unwrap();
        assert_eq!(b1.num_prefill_tokens, 52);
        let pages_after_s1 = kv.allocated_page_count();
        assert_eq!(pages_after_s1, 4); // ceil(52 / 16) = 4 pages

        // Enqueue s2 concurrently: should reuse ALL 3 shared prefix pages (zero duplicate VRAM!)
        let s2 = batcher.enqueue(p2, 2);
        let b2 = batcher.schedule_next_step(&mut kv, Some(&mut tide)).unwrap();
        assert_eq!(b2.num_decode_tokens, 1); // s1 decoding concurrently!
        assert_eq!(b2.num_prefill_tokens, 4); // s2 only prefills its 4 unique suffix tokens!
        assert_eq!(
            kv.allocated_page_count(),
            pages_after_s1 + 1,
            "s2 should only allocate 1 new physical page beyond s1's shared prefix"
        );

        // Finish both requests -> ALL physical KV pages freed!
        let fin1 = batcher.commit_sampled_tokens(&[(s1, 200006), (s2, 200006)], 200006, &mut kv, Some(&mut tide));
        assert_eq!(fin1.len(), 2);
        assert_eq!(kv.allocated_page_count(), 0);
        assert_eq!(kv.active_sequences(), 0);
    }
}
