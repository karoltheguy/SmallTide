use serde::{Deserialize, Serialize};

/// Live TideLM streaming and performance telemetry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TideTelemetry {
    pub tokens_decoded: u64,
    pub tokens_prefilled: u64,
    pub dspark_accepted_tokens: u64,
    pub dspark_drafted_tokens: u64,

    pub vram_hits_prelude: u64,
    pub vram_misses_prelude: u64,
    pub vram_hits_loop: u64,
    pub vram_misses_loop: u64,
    pub vram_hits_coda: u64,
    pub vram_misses_coda: u64,

    pub stage1_async_promotions: u64,
    pub stage1_sync_rescues: u64,
    pub stage1_skipped_tail_misses: u64,
    pub stage1_bytes_transferred: u64,
    /// RAM-tier slabs that were missing/stale at DMA time and had to be re-read from NVMe.
    #[serde(default)]
    pub stage1_ram_fallback_reads: u64,
    /// Async promotions published after their DMA completed (pending -> resident).
    #[serde(default)]
    pub stage1_async_published: u64,

    pub stage2_disk_reads: u64,
    pub stage2_bytes_read: u64,

    /// Decode routing mass seen (1.0 per token per logical layer) and the part of it that was
    /// skipped (missing experts, renormalized away): the quality cost of the resident set.
    #[serde(default)]
    pub routed_weight_total: f64,
    #[serde(default)]
    pub routed_weight_skipped: f64,
    /// Prefill routing mass (1.0 per prompt token per logical layer) and the part skipped.
    #[serde(default)]
    pub prefill_weight_total: f64,
    #[serde(default)]
    pub prefill_weight_skipped: f64,
    /// Prompt-aware async swaps (anticipated from conversation memory, between prefill chunks,
    /// at the prefill -> decode boundary).
    #[serde(default)]
    pub prompt_aware_swaps: u64,
    /// RAM -> VRAM promotions made during decode by the per-token async planner.
    #[serde(default)]
    pub decode_async_promotions: u64,
}

impl TideTelemetry {
    /// Share of decode routing weight that was skipped (0 = exact).
    pub fn skipped_weight_share(&self) -> f64 {
        if self.routed_weight_total > 0.0 {
            self.routed_weight_skipped / self.routed_weight_total
        } else {
            0.0
        }
    }

    pub fn total_vram_hit_rate(&self) -> f64 {
        let hits = self.vram_hits_prelude + self.vram_hits_loop + self.vram_hits_coda;
        let misses = self.vram_misses_prelude + self.vram_misses_loop + self.vram_misses_coda;
        let total = hits + misses;
        if total == 0 {
            1.0
        } else {
            hits as f64 / total as f64
        }
    }
}
