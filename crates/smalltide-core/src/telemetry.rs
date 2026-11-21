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

    pub stage2_disk_reads: u64,
    pub stage2_bytes_read: u64,
}

impl TideTelemetry {
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
