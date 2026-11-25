//! Inkling-2-Mini model graph:
//! - 24 physical layers -> 48 logical layers (6 prelude + 3x12 loop core + 6 coda)
//! - Gated rank-64 cross-pass message passing
//! - Hybrid attention: 40 local sliding-window (1024) + 8 global dynamic K/V conv
//! - Default FP8 E4M3 dense projections & lm_head + native NVFP4/MXFP8 routed experts
//! - Auto-enabled `dspark` speculative decoding when `mtp.safetensors` is present.
//!
//! Implemented in Phase 2 (exact forward pass) and Phase 4 (dspark integration).

pub mod dflash2;
pub mod drafter;
pub mod engine;
pub mod kv_pager;
pub mod prefill;
pub mod vision;
pub mod weights;

use smalltide_core::{Inkling2Config, LogicalLayerId, PhysicalLayerId};

/// Maps a logical execution layer (0..47) to `(physical_layer, loop_pass_index)`.
pub fn map_logical_to_physical(
    cfg: &Inkling2Config,
    logical: LogicalLayerId,
) -> (PhysicalLayerId, usize) {
    let l = logical.0 as usize;
    let loop_len = cfg.loop_end_exclusive - cfg.loop_start_layer;
    let loop_total = loop_len * cfg.loop_passes;

    if l < cfg.loop_start_layer {
        (PhysicalLayerId(l as u16), 0)
    } else if l < cfg.loop_start_layer + loop_total {
        let offset = l - cfg.loop_start_layer;
        let pass = offset / loop_len;
        let phys = cfg.loop_start_layer + (offset % loop_len);
        (PhysicalLayerId(phys as u16), pass)
    } else {
        let coda_offset = l - (cfg.loop_start_layer + loop_total);
        let phys = cfg.loop_end_exclusive + coda_offset;
        (PhysicalLayerId(phys as u16), 0)
    }
}
