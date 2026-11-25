//! 4-Mode TideLM Expert Scheduler:
//! 1. Pre-prefill warm-up (Loop-Core priority + optional priors)
//! 2. Prefill -> Decode boundary tide rebalance
//! 3. Paced async background promotion (Stage 1 RAM->VRAM + Stage 2 Disk->RAM)
//! 4. In-layer dispatch + configurable Sync Rescue (Exact / Threshold / Disabled)

use crate::cache::{TideCache, TransferOp};
use crate::expert_map::{
    RescueSnapshot, RescueZoneSnapshot, RESCUE_HIST_BINS, RESCUE_HIST_BIN_PCT,
};
use crate::priors::TidePriors;
use smalltide_core::{
    ExpertId, Inkling2Config, PhysicalLayerId, RescuePolicy, SlotIndex, TideConfig, TideTelemetry,
};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;

const RESCUE_WINDOW_CAP: usize = 128;

#[inline]
fn rescue_hist_bin(norm_w: f32) -> usize {
    let pct = (norm_w.max(0.0) * 100.0) / (RESCUE_HIST_BIN_PCT as f32);
    (pct.floor() as usize).min(RESCUE_HIST_BINS - 1)
}

#[derive(Debug, Clone)]
struct RescueStepZone {
    rescued: u16,
    budget: u16,
    capped_layers: u16,
    blocked_picks: u16,
    below_gate_picks: u16,
    blocked_weight: f32,
    below_weight: f32,
    rescued_share_sum: f32,
    rescued_share_min: f32,
    miss_hist: [u32; RESCUE_HIST_BINS],
}

impl Default for RescueStepZone {
    fn default() -> Self {
        Self {
            rescued: 0,
            budget: 0,
            capped_layers: 0,
            blocked_picks: 0,
            below_gate_picks: 0,
            blocked_weight: 0.0,
            below_weight: 0.0,
            rescued_share_sum: 0.0,
            rescued_share_min: f32::INFINITY,
            miss_hist: [0; RESCUE_HIST_BINS],
        }
    }
}

#[derive(Debug, Clone, Default)]
struct RescueStepRecord {
    verify_rows: usize,
    zones: [RescueStepZone; 3],
}

#[derive(Debug, Default)]
struct RescueTracker {
    window: VecDeque<RescueStepRecord>,
    cur: RescueStepRecord,
    step_active: bool,
    step_is_prefill: bool,
    boot_steps: u64,
    boot_rescued: [u64; 3],
    boot_capped_steps: [u64; 3],
}

impl RescueTracker {
    fn commit_current(&mut self) {
        if !self.step_active || self.step_is_prefill {
            self.step_active = false;
            return;
        }
        let rec = std::mem::take(&mut self.cur);
        self.boot_steps += 1;
        for z in 0..3 {
            let rz = &rec.zones[z];
            self.boot_rescued[z] += rz.rescued as u64;
            if rz.budget > 0 && rz.rescued >= rz.budget {
                self.boot_capped_steps[z] += 1;
            }
        }
        if self.window.len() == RESCUE_WINDOW_CAP {
            self.window.pop_front();
        }
        self.window.push_back(rec);
        self.step_active = false;
    }

    fn start_step(&mut self, verify_rows: usize, budgets: [usize; 3]) {
        self.commit_current();
        self.cur = RescueStepRecord {
            verify_rows: verify_rows.max(1),
            zones: std::array::from_fn(|z| RescueStepZone {
                budget: budgets[z].min(u16::MAX as usize) as u16,
                ..RescueStepZone::default()
            }),
        };
        self.step_active = false;
        self.step_is_prefill = false;
    }
}

/// Auto gate (`SMALLTIDE_RESCUE_AUTO_GATE=1`): under `RescuePolicy::Threshold` each block's
/// (Prelude / Loop Core / Coda) gate starts at its configured value and is then steered, one
/// small step per decode step, so that the block spends about `target` of its rescue budget:
/// a block that rescues less than that lowers its gate, one that rescues more raises it.
/// The budgets then set how much time rescues may take and the gates follow the traffic.
#[derive(Debug, Clone, Copy)]
pub struct AutoGate {
    pub enabled: bool,
    /// Share of a block's budget to spend per step, on average (0.8).
    pub target: f32,
    /// Bounds of the steered gates (routing share).
    pub min_gate: f32,
    pub max_gate: f32,
    /// The steered gate of each block; the configured gates until the first step.
    live: [f32; 3],
}

impl AutoGate {
    /// Change of ln(gate) in one step, per unit of (used / budget - target). With the 0.8 target
    /// a block that rescues nothing halves its gate in about 17 steps and one at its cap doubles
    /// it in about 70, while a block near the target moves by under 1% a step.
    const GAIN: f32 = 0.05;
}

impl Default for AutoGate {
    fn default() -> Self {
        Self { enabled: false, target: 0.8, min_gate: 0.02, max_gate: 0.5, live: [0.0; 3] }
    }
}

/// `SMALLTIDE_KEEP_EVICTED_IN_RAM=0` turns off `keep_evicted_in_ram` and
/// `adopt_disk_read_into_ram` (dev comparison).
fn keep_evicted_in_ram_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SMALLTIDE_KEEP_EVICTED_IN_RAM").map(|v| v.trim() != "0").unwrap_or(true))
}

/// Routing priors owned by the scheduler: installed at startup, updated at each request end.
pub struct PriorsStore {
    pub priors: TidePriors,
    /// Where to persist after each request (`None` = in-memory only).
    pub path: Option<PathBuf>,
    /// Weight of the newest request's vote once history exists (see `TidePriors::merge_demand`).
    pub newest_share: f32,
    /// Prior weight in expert values (heat half-life units, see `TideCache::set_prior_rates`).
    pub prior_weight: f32,
    /// Fold requests in (false = frozen priors, e.g. benchmarks).
    pub update: bool,
}

#[derive(Debug, Clone)]
pub struct ActiveExpertSlot {
    pub expert: ExpertId,
    pub vram_slot: SlotIndex,
    pub weight: f32,
}

#[derive(Debug, Clone)]
pub struct LayerDispatchPlan {
    /// Experts already resident in Tier 0 VRAM — launch immediately on Compute Stream.
    pub hit_experts: Vec<ActiveExpertSlot>,
    /// Blocking transfers required before computing rescued experts (empty in `RescuePolicy::Disabled`).
    pub sync_transfers: Vec<TransferOp>,
    /// Rescued experts to compute once `sync_transfers` complete.
    pub rescued_experts: Vec<ActiveExpertSlot>,
    /// Exact renormalization multiplier (`sum(all_topk_weights) / sum(executed_weights)`)
    /// when tail misses are skipped in `Disabled` or `Threshold` mode.
    pub weight_renorm_scale: f32,
    /// True if a top-k expert's async DMA is still in flight: the compute stream must wait on the
    /// copy stream (GPU-side) and the pending experts must be published before the MoE kernel.
    pub wait_pending: bool,
}

pub struct TideScheduler {
    pub model_cfg: Inkling2Config,
    pub tide_cfg: TideConfig,
    pub cache: TideCache,
    pub telemetry: TideTelemetry,

    /// Accumulated routing weights observed during the current prefill phase.
    prefill_scores: Vec<f32>,
    /// Pending async promotion candidates (expert -> accumulated miss priority).
    pending_async_vram: HashMap<ExpertId, f32>,
    pending_async_ram: HashMap<ExpertId, f32>,
    pub priors_store: Option<PriorsStore>,
    /// Hysteresis for decode-time (per-token) async promotions: a missed expert must be worth
    /// this many times the coldest resident it would evict. Boundary / periodic passes use
    /// `cache::SWAP_HYSTERESIS`.
    pub async_hysteresis: f64,
    /// `RescuePolicy::Threshold`: most synchronous rescues per decode token (TideLM's
    /// `RESCUE_STEP` is 2); split across Prelude / Loop Core / Coda unless `rescue_zone_budgets`
    /// is set explicitly.
    pub rescue_budget: usize,
    /// Optional explicit per-zone rescue budgets `[prelude, loop_core, coda]` per token.
    pub rescue_zone_budgets: Option<[usize; 3]>,
    /// Threshold gate for loop-core layers (their experts run once per loop pass), if it differs
    /// from the policy's gate.
    pub loop_core_gate: Option<f32>,
    /// Under `RescuePolicy::Threshold`, any missing expert whose normalized routing share is at
    /// least this threshold is *always* rescued regardless of `rescue_zone_budgets` or `auto_gate`
    /// (capped only by the layer's rescue ring size). Prevents dropping dominant domain/language
    /// experts on sudden topic shifts. `None` (or `<= 0` / `> 1`) disables.
    pub rescue_must_threshold: Option<f32>,
    /// Steers the gates towards a share of the budgets (off by default); set it with
    /// `set_auto_gate` so the steered gates start from the configured ones.
    auto_gate: AutoGate,
    rescues_this_token: usize,
    rescues_by_zone: [usize; 3],
    /// Experts rescued inline in the current decode / verify step (see `was_rescued_this_step`).
    step_rescued: Vec<ExpertId>,
    /// Residency events for the expert map (host bookkeeping; decides nothing).
    pub events: crate::expert_map::EventLog,
    /// Kind / routing share the next logged transfers get (set by each planner; 0 = not logged).
    event_kind: u8,
    event_share: f32,
    /// `decoded_tokens` at which periodic adaptive rebalance last fired (immune to multi-token spec jumps!).
    last_periodic_rebalance_tok: u64,
    /// Per-block (Prelude / Loop Core / Coda) rescue gate & budget tuning telemetry.
    rescue_tracker: RescueTracker,
    /// True during speculative draft-verify passes (row 0 is committed; later rows have lower survival).
    speculative_verify: bool,
    /// When true (default), discounts speculative tail rows in `plan_verify_layer_dispatch`.
    pub spec_row_decay: bool,
}

impl TideScheduler {
    pub fn new(
        model_cfg: Inkling2Config,
        tide_cfg: TideConfig,
        unified_vram_budget_bytes: usize,
        ram_budget_bytes: usize,
    ) -> Self {
        let total = model_cfg.num_physical_layers * model_cfg.n_routed_experts;
        let cache = TideCache::new(
            model_cfg.clone(),
            tide_cfg.clone(),
            unified_vram_budget_bytes,
            ram_budget_bytes,
        );
        Self {
            model_cfg,
            tide_cfg,
            cache,
            telemetry: TideTelemetry::default(),
            prefill_scores: vec![0.0; total],
            pending_async_vram: HashMap::new(),
            pending_async_ram: HashMap::new(),
            priors_store: None,
            async_hysteresis: 1.10,
            rescue_budget: 4,
            rescue_zone_budgets: None,
            loop_core_gate: None,
            rescue_must_threshold: None,
            auto_gate: AutoGate::default(),
            rescues_this_token: 0,
            rescues_by_zone: [0; 3],
            step_rescued: Vec::new(),
            events: crate::expert_map::EventLog::default(),
            event_kind: 0,
            event_share: 0.0,
            last_periodic_rebalance_tok: 0,
            rescue_tracker: RescueTracker::default(),
            speculative_verify: false,
            spec_row_decay: std::env::var("SMALLTIDE_SPEC_ROW_DECAY")
                .map(|v| v.trim() == "1" || v.trim().eq_ignore_ascii_case("true"))
                .unwrap_or(false),
        }
    }

    /// Installs routing priors; non-empty priors immediately drive values and warm-up.
    pub fn install_priors(&mut self, store: PriorsStore) {
        if !store.priors.is_empty() {
            self.cache.set_prior_rates(store.priors.rates(&self.model_cfg), store.prior_weight);
        }
        self.priors_store = Some(store);
    }

    /// End of a request whose routing must not teach the priors (`"learn_routing": false`): its
    /// demand is dropped.
    pub fn discard_request_demand(&mut self) {
        let _ = self.cache.take_demand();
    }

    /// End of a request: folds its routing demand into the priors as one vote, refreshes the live
    /// prior rates and persists. Returns a short log line.
    pub fn finish_request(&mut self) -> Option<String> {
        let demand = self.cache.take_demand();
        let st = self.priors_store.as_mut()?;
        if !st.update {
            return None;
        }
        st.priors.merge_demand(&demand, st.newest_share);
        if st.priors.is_empty() {
            return None;
        }
        self.cache.set_prior_rates(st.priors.rates(&self.model_cfg), st.prior_weight);
        let saved = match &st.path {
            Some(p) => match st.priors.save_to_file(p) {
                Ok(()) => format!("saved {}", p.display()),
                Err(e) => format!("NOT saved ({e})"),
            },
            None => "in memory".into(),
        };
        Some(format!("[priors] {} requests folded in; {saved}", st.priors.sessions_recorded))
    }


    /// Mode 1: Preload Before Prefill (Session Warm-up).
    /// 1) Pin Loop Core (layers 6..17) into VRAM first (up to 100% if budget permits).
    /// 2) Fill remaining VRAM with outer-layer experts (sorted by optional frequency priors).
    /// 3) Warm Tier 1 RAM with remaining unpinned outer-layer experts up to `ram_budget_bytes`.
    pub fn plan_pre_prefill(&mut self, priors: Option<&[f32]>) -> Vec<TransferOp> {
        self.event_kind = 0; // the boot placement is not an event
        if !self.tide_cfg.preload_before_prefill {
            return Vec::new();
        }
        let mut ops = Vec::new();
        let n_exp = self.model_cfg.n_routed_experts;

        if self.cache.has_priors() {
            // Priors known: load the experts with the highest expected routing mass per byte
            // (both pools), unpinned so the session can still trade them; then warm RAM.
            let total = self.model_cfg.num_physical_layers * n_exp;
            let mut ranked: Vec<(ExpertId, f64)> = (0..total)
                .map(|i| {
                    let id = self.cache.expert_from_flat(i);
                    (id, self.cache.prior_rate(id) as f64 / self.cache.expert_bytes(id) as f64)
                })
                .collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            self.cache.no_ram_copy_on_direct = true;
            for &(id, _) in &ranked {
                if self.cache.has_free_vram_room_for(id) {
                    if let Some(op) = self.cache.promote_to_vram(id, false, true) {
                        self.record_transfer_telemetry(&op, false);
                        ops.push(op);
                    }
                }
            }
            self.cache.no_ram_copy_on_direct = false;
            let order: Vec<ExpertId> = ranked.iter().map(|&(id, _)| id).collect();
            self.fill_ram(&order, &mut ops);
            return ops;
        }

        // Step 1: Loop Core (physical layers 6..17) -> VRAM (pinned)
        let mut loop_experts: Vec<(ExpertId, f32)> = Vec::new();
        for l in self.model_cfg.loop_start_layer..self.model_cfg.loop_end_exclusive {
            for e in 0..n_exp {
                let id = ExpertId {
                    layer: PhysicalLayerId(l as u16),
                    expert: e as u16,
                };
                let score = priors.map(|p| p[self.cache.flat_index(id)]).unwrap_or(1.0);
                loop_experts.push((id, score));
            }
        }
        loop_experts.sort_by(|a, b| b.1.total_cmp(&a.1));
        self.cache.no_ram_copy_on_direct = true;
        let loop_order: Vec<ExpertId> = loop_experts.iter().map(|&(id, _)| id).collect();
        for (id, _) in loop_experts {
            if !self.cache.has_free_vram_room_for(id) {
                break;
            }
            if let Some(op) = self.cache.promote_to_vram(id, true, true) {
                self.record_transfer_telemetry(&op, false);
                ops.push(op);
            }
        }

        // Step 2: Outer layers (Prelude 0..5 + Coda 18..23) -> fill remaining VRAM, then RAM
        let mut outer_experts: Vec<(ExpertId, f32)> = Vec::new();
        for l in 0..self.model_cfg.num_physical_layers {
            if self.model_cfg.is_loop_core_layer(l) {
                continue;
            }
            for e in 0..n_exp {
                let id = ExpertId {
                    layer: PhysicalLayerId(l as u16),
                    expert: e as u16,
                };
                // Mildly prefer NVFP4 outer layers (0..5, 18..19) when scores tie: 1.8x more experts per GiB!
                let base = priors.map(|p| p[self.cache.flat_index(id)]).unwrap_or(1.0);
                let density_bonus = if self.model_cfg.is_mxfp8_layer(l) { 0.85 } else { 1.0 };
                outer_experts.push((id, base * density_bonus));
            }
        }
        outer_experts.sort_by(|a, b| b.1.total_cmp(&a.1));

        for &(id, _) in &outer_experts {
            if self.cache.has_free_vram_room_for(id) {
                if let Some(op) = self.cache.promote_to_vram(id, false, true) {
                    self.record_transfer_telemetry(&op, false);
                    ops.push(op);
                }
            }
        }
        self.cache.no_ram_copy_on_direct = false;
        let mut order: Vec<ExpertId> = outer_experts.iter().map(|&(id, _)| id).collect();
        order.extend(loop_order);
        self.fill_ram(&order, &mut ops);

        ops
    }

    /// A background swap (`op`) took an expert off the card. If that expert has no copy in RAM
    /// it is on the disk only, and its next rescue would read the disk while the GPU waits:
    /// queue a background read of it now. Only where RAM has room (`ram_has_room_for`).
    fn keep_evicted_in_ram(&mut self, op: &TransferOp, ops: &mut Vec<TransferOp>) {
        if !keep_evicted_in_ram_on() {
            return;
        }
        let victim = match *op {
            TransferOp::RamToVram { evicted_from_vram: Some(v), .. } | TransferOp::DiskToVramDirect { evicted_from_vram: Some(v), .. } => v,
            _ => return,
        };
        if !self.cache.entry(victim).ram_slot.is_missing() || !self.cache.ram_has_room_for(victim) {
            return;
        }
        if let Some(read) = self.cache.promote_to_ram_with_flag(victim, true) {
            self.record_transfer_telemetry(&read, false);
            self.telemetry.evicted_kept_in_ram += 1;
            ops.push(read);
        }
    }

    /// An expert that was not ready in RAM was just read from the disk for a rescue. Returns the
    /// RAM slot the caller stores the bytes in, so its next rescue is a RAM copy: the slot its
    /// background read is still on its way to (that read is then already done), or a new one.
    /// None when RAM has no room.
    pub fn adopt_disk_read_into_ram(&mut self, id: ExpertId) -> Option<SlotIndex> {
        if !keep_evicted_in_ram_on() {
            return None;
        }
        let e = self.cache.entry(id);
        if !e.ram_slot.is_missing() {
            self.cache.mark_ram_ready(id);
            return Some(e.ram_slot);
        }
        if !self.cache.ram_has_room_for(id) {
            return None;
        }
        match self.cache.promote_to_ram_with_flag(id, false)? {
            TransferOp::DiskToRam { ram_slot, .. } => Some(ram_slot),
            _ => None,
        }
    }

    /// Startup RAM warm-up in `order` (most valuable first): experts that are NOT in VRAM first
    /// (a miss on them would otherwise read the disk), then RAM copies of VRAM-resident experts
    /// with the budget left (cheap fallback once they are evicted from VRAM). Those copies go to
    /// the least valuable residents first: they are the ones a swap evicts, and an evicted expert
    /// without a RAM copy is read from the disk the next time it is rescued - while the GPU waits.
    fn fill_ram(&mut self, order: &[ExpertId], ops: &mut Vec<TransferOp>) {
        for want_resident in [false, true] {
            let ids: Vec<ExpertId> = if want_resident { order.iter().rev().copied().collect() } else { order.to_vec() };
            for id in ids {
                let e = self.cache.entry(id);
                let in_vram = !e.vram_slot.is_missing();
                // Pinned experts are never evicted from VRAM: a RAM copy would be dead weight.
                if in_vram != want_resident || !e.ram_slot.is_missing() || e.vram_pinned {
                    continue;
                }
                if self.cache.ram_used_bytes() + self.cache.expert_bytes(id) > self.cache.ram_budget_bytes() {
                    continue;
                }
                if let Some(op) = self.cache.promote_to_ram(id) {
                    self.record_transfer_telemetry(&op, false);
                    ops.push(op);
                }
            }
        }
    }

    /// Records router activations during prefill so Mode 2 can perform a prompt-aware rebalance.
    pub fn record_prefill_routing(
        &mut self,
        layer: PhysicalLayerId,
        topk_ids: &[u16],
        topk_weights: &[f32],
    ) {
        let tot: f32 = topk_weights.iter().sum();
        let inv = if tot > 1e-8 { 1.0 / tot } else { 0.0 };
        for (&exp, &w) in topk_ids.iter().zip(topk_weights.iter()) {
            let id = ExpertId { layer, expert: exp };
            let idx = self.cache.flat_index(id);
            self.prefill_scores[idx] += w * inv;
            self.cache.touch_expert(id, w * inv);
        }
    }

    /// Mode 2: Preload Between Prefill and Decode (Boundary Tide Rebalance).
    /// Promotes the hottest prompt-activated non-resident experts into VRAM (and RAM if on disk)
    /// in one high-throughput burst before token 1 decodes.
    pub fn plan_prefill_decode_rebalance(&mut self, max_swaps: usize) -> Vec<TransferOp> {
        self.event_kind = crate::expert_map::EV_ADAPT;
        self.event_share = 0.0;
        if !self.tide_cfg.preload_prefill_decode_boundary {
            self.prefill_scores.fill(0.0);
            return Vec::new();
        }
        let ops = self.plan_prefill_progress_swaps(max_swaps);
        self.prefill_scores.fill(0.0);
        ops
    }

    /// Async swaps toward `cands` (missing experts): best value first, each must beat the eviction
    /// floor of its pool by `cache::SWAP_HYSTERESIS`. RAM-ready experts are promoted (RAM -> VRAM,
    /// copy stream); the others get an async disk -> RAM read and are picked up by later passes.
    /// Never reads disk synchronously.
    pub fn plan_async_swaps(&mut self, cands: Vec<ExpertId>, max_swaps: usize) -> Vec<TransferOp> {
        let mut ranked: Vec<(ExpertId, f64)> = cands
            .into_iter()
            .filter(|&id| {
                let e = self.cache.entry(id);
                (e.vram_slot.is_missing() || e.is_scratch_slot) && !self.cache.is_vram_pending(id)
            })
            .map(|id| (id, self.cache.value(id)))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut ops = Vec::new();
        let mut swaps = 0usize;
        let mut ram_reads = 0usize;
        let mut floors: [Option<Option<f64>>; 2] = [None, None];
        for (id, val) in ranked {
            if swaps >= max_swaps {
                break;
            }
            let cls = self.cache.expert_class(id);
            let ci = cls as usize;
            let need = self.cache.expert_bytes(id);
            let floor = *floors[ci].get_or_insert_with(|| self.cache.promotion_floor(cls, need));
            if let Some(f) = floor {
                if val <= f * self.cache.swap_hysteresis + 1e-9 {
                    continue;
                }
            }
            if !self.cache.is_ram_ready(id) {
                if ram_reads < max_swaps {
                    if let Some(op) = self.cache.promote_to_ram_with_flag(id, true) {
                        self.record_transfer_telemetry(&op, false);
                        ops.push(op);
                        ram_reads += 1;
                    }
                }
                *self.pending_async_vram.entry(id).or_insert(0.0) += 1.0;
                continue;
            }
            let pin = !self.cache.has_priors() && self.model_cfg.is_loop_core_layer(id.layer.0 as usize);
            if let Some(op) = self.cache.promote_to_vram(id, pin, false) {
                self.record_transfer_telemetry(&op, false);
                self.telemetry.prompt_aware_swaps += 1;
                ops.push(op.clone());
                self.keep_evicted_in_ram(&op, &mut ops);
                swaps += 1;
                floors[ci] = None;
            }
        }
        ops
    }

    /// Between prefill chunks: async swaps for the missing experts this prompt has routed to so
    /// far (its heat already holds the prompt's demand).
    pub fn plan_prefill_progress_swaps(&mut self, max_swaps: usize) -> Vec<TransferOp> {
        self.event_kind = crate::expert_map::EV_ADAPT;
        self.event_share = 0.0;
        let cands: Vec<ExpertId> = self
            .prefill_scores
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 0.0)
            .map(|(i, _)| self.cache.expert_from_flat(i))
            .collect();
        self.plan_async_swaps(cands, max_swaps)
    }

    /// Before a request's prefill: anticipate its routing from `demand` (e.g. the conversation's
    /// previous turn, sum of routing shares per expert): adds it as heat, then plans async swaps for
    /// the missing experts it covers.
    pub fn plan_anticipated_swaps(&mut self, demand: &[f32], max_swaps: usize) -> Vec<TransferOp> {
        self.event_kind = crate::expert_map::EV_PRELOAD;
        self.event_share = 0.0;
        let mut cands = Vec::new();
        for (i, &d) in demand.iter().enumerate() {
            if d > 0.0 {
                let id = self.cache.expert_from_flat(i);
                self.cache.add_heat(id, d);
                if self.cache.entry(id).vram_slot.is_missing() {
                    cands.push(id);
                }
            }
        }
        self.plan_async_swaps(cands, max_swaps)
    }

    /// Returns true if all `n_routed_experts` for `layer` are currently resident in VRAM.
    /// When true, the engine can skip the per-layer D2H router readback entirely!
    pub fn is_layer_fully_resident_vram(&self, layer: PhysicalLayerId) -> bool {
        let base = (layer.0 as usize) * self.model_cfg.n_routed_experts;
        let table = self.cache.gpu_slot_table();
        table[base..base + self.model_cfg.n_routed_experts]
            .iter()
            .all(|&s| s != u32::MAX)
    }

    /// Records `top_k` hits for a 100%-VRAM-resident layer without requiring CPU router readback.
    pub fn record_full_layer_hit(&mut self, layer: PhysicalLayerId, top_k: u64) {
        let l = layer.0 as usize;
        self.telemetry.routed_weight_total += 1.0;
        if self.model_cfg.is_loop_core_layer(l) {
            self.telemetry.vram_hits_loop += top_k;
        } else if l < self.model_cfg.loop_start_layer {
            self.telemetry.vram_hits_prelude += top_k;
        } else {
            self.telemetry.vram_hits_coda += top_k;
        }
    }

    /// Mode 4: Per-layer decode dispatch planner (Split-Batch Hit/Miss + Sync Rescue + Renorm).
    /// Token boundary: the per-zone rescue budgets start again.
    pub fn begin_token(&mut self) {
        self.begin_verify_step(1);
    }

    /// Step boundary for single-token decode (`verify_rows == 1`) or multi-row passes (`m > 1`).
    pub fn begin_verify_step(&mut self, verify_rows: usize) {
        self.begin_verify_step_kind(verify_rows, false);
    }

    /// Step boundary with explicit `speculative` flag (`true` for draft-verify where row 0 is
    /// guaranteed committed and tail rows are speculative; `false` for multi-lane independent rows).
    pub fn begin_verify_step_kind(&mut self, verify_rows: usize, speculative: bool) {
        self.steer_auto_gates();
        let budgets = self.effective_zone_budgets(verify_rows);
        self.rescue_tracker.start_step(verify_rows, budgets);
        self.rescues_this_token = 0;
        self.rescues_by_zone = [0; 3];
        self.step_rescued.clear();
        self.cache.clear_step_protected();
        self.speculative_verify = speculative && verify_rows > 1;
    }

    /// True if `id` was copied in by an inline rescue in the current step: the step ran it, so the
    /// expert map does not show the pick as missed even though the expert is not resident
    /// afterwards (a rescue's slot is reused within a few layers). Per physical layer: a loop-core
    /// expert rescued in one loop pass counts for the step's other passes too.
    pub fn was_rescued_this_step(&self, id: ExpertId) -> bool {
        self.step_rescued.contains(&id)
    }

    /// A step that plans no rescues (rescue off for it) starts with none recorded.
    pub fn clear_step_rescued(&mut self) {
        self.step_rescued.clear();
    }

    /// Marks the current step as prefill so prefill passes do not skew decode/verify rescue stats.
    pub fn mark_step_prefill(&mut self) {
        self.rescue_tracker.step_is_prefill = true;
    }

    /// Clears the rolling rescue-tuning window (e.g. after a live gate/budget config change).
    pub fn reset_rescue_window(&mut self) {
        self.rescue_tracker.window.clear();
        self.rescue_tracker.step_active = false;
    }

    /// Records that `layer` was short-circuited because its zone's rescue budget was already spent.
    pub fn note_zone_budget_capped_layer(&mut self, layer: PhysicalLayerId) {
        if self.rescue_tracker.step_is_prefill {
            return;
        }
        let z = self.zone_of(layer);
        self.rescue_tracker.step_active = true;
        self.rescue_tracker.cur.zones[z].capped_layers =
            self.rescue_tracker.cur.zones[z].capped_layers.saturating_add(1);
    }

    /// The configured routing-share gate for zone `z` (`0 = Prelude`, `1 = Loop Core`, `2 = Coda`).
    #[inline]
    pub fn configured_zone_gate(&self, zone: usize) -> f32 {
        match self.tide_cfg.rescue_policy {
            RescuePolicy::Exact => 0.0,
            RescuePolicy::Disabled => 1.0,
            RescuePolicy::Threshold(gate) => match self.loop_core_gate {
                Some(g) if zone == 1 => g,
                _ => gate,
            },
        }
    }

    /// The gate in effect for zone `z`: the configured one, or the steered one with the auto gate on.
    #[inline]
    pub fn effective_zone_gate(&self, zone: usize) -> f32 {
        if self.auto_gate.enabled && matches!(self.tide_cfg.rescue_policy, RescuePolicy::Threshold(_)) {
            self.auto_gate.live[zone]
        } else {
            self.configured_zone_gate(zone)
        }
    }

    pub fn auto_gate(&self) -> AutoGate {
        self.auto_gate
    }

    /// Turns the auto gate on or off (and sets its target and bounds); the steered gates start
    /// again from the configured ones.
    pub fn set_auto_gate(&mut self, cfg: AutoGate) {
        self.auto_gate = cfg;
        self.auto_gate.target = cfg.target.clamp(0.05, 1.0);
        // Never exactly 0: the gate moves by a factor per step and could not come back up from it.
        self.auto_gate.min_gate = cfg.min_gate.clamp(0.001, 1.0);
        self.auto_gate.max_gate = cfg.max_gate.clamp(self.auto_gate.min_gate, 1.0);
        self.reset_auto_gates();
    }

    /// The steered gates start again from the configured ones (after a gate or mode change).
    pub fn reset_auto_gates(&mut self) {
        self.auto_gate.live = std::array::from_fn(|z| self.configured_zone_gate(z));
    }

    /// Auto gate: one step of each block's gate from the decode / verify step that just ended.
    /// ln(gate) moves by `GAIN * (used / budget - target)`, so the gate settles where the block
    /// spends `target` of its budget on average, or at a bound when no gate gets it there (too
    /// few misses to spend the budget: the floor).
    fn steer_auto_gates(&mut self) {
        if !self.auto_gate.enabled || !matches!(self.tide_cfg.rescue_policy, RescuePolicy::Threshold(_)) {
            return;
        }
        let t = &self.rescue_tracker;
        if !t.step_active || t.step_is_prefill {
            return;
        }
        let a = &mut self.auto_gate;
        for z in 0..3 {
            let rz = &t.cur.zones[z];
            if rz.budget == 0 {
                continue;
            }
            let util = (rz.rescued as f32 / rz.budget as f32).min(1.0);
            let g = a.live[z] * (AutoGate::GAIN * (util - a.target)).exp();
            a.live[z] = g.clamp(a.min_gate, a.max_gate);
        }
    }

    /// Snapshot of per-block (`Prelude`, `Loop Core`, `Coda`) rescue gate & budget tuning metrics.
    pub fn rescue_snapshot(&self) -> RescueSnapshot {
        let policy = match self.tide_cfg.rescue_policy {
            RescuePolicy::Exact => "exact",
            RescuePolicy::Disabled => "disabled",
            RescuePolicy::Threshold(_) => "threshold",
        };
        let base_budgets = self.effective_zone_budgets(1);
        let include_cur = self.rescue_tracker.step_active && !self.rescue_tracker.step_is_prefill;
        let window_steps = self.rescue_tracker.window.len() + usize::from(include_cur);
        let total_steps = self.rescue_tracker.boot_steps + u64::from(include_cur);
        let last_verify_rows = if include_cur {
            self.rescue_tracker.cur.verify_rows.max(1)
        } else {
            self.rescue_tracker
                .window
                .back()
                .map(|r| r.verify_rows.max(1))
                .unwrap_or(1)
        };

        let meta: [(&'static str, &'static str); 3] = [
            ("Prelude", "L0–5"),
            ("Loop Core", "L6–17 ×3"),
            ("Coda", "L18–23"),
        ];

        let zones = std::array::from_fn(|z| {
            let gate = self.effective_zone_gate(z);
            let base_budget = base_budgets[z];
            let mut used_vals: Vec<usize> = Vec::with_capacity(window_steps);
            let mut sum_used = 0u64;
            let mut sum_budget = 0u64;
            let mut capped_steps = 0u64;
            let mut zero_steps = 0u64;
            let mut max_used = 0usize;
            let mut sum_capped_layers = 0u64;
            let mut sum_blocked_picks = 0u64;
            let mut sum_below_picks = 0u64;
            let mut sum_blocked_w = 0.0f64;
            let mut sum_below_w = 0.0f64;
            let mut sum_rescued_sh = 0.0f64;
            let mut min_rescued_sh = f32::INFINITY;
            let mut miss_hist = [0u32; RESCUE_HIST_BINS];

            let mut fold = |rz: &RescueStepZone| {
                let u = rz.rescued as usize;
                let b = rz.budget as usize;
                used_vals.push(u);
                sum_used += u as u64;
                sum_budget += b as u64;
                if b > 0 && u >= b {
                    capped_steps += 1;
                }
                if u == 0 {
                    zero_steps += 1;
                }
                if u > max_used {
                    max_used = u;
                }
                sum_capped_layers += rz.capped_layers as u64;
                sum_blocked_picks += rz.blocked_picks as u64;
                sum_below_picks += rz.below_gate_picks as u64;
                sum_blocked_w += rz.blocked_weight as f64;
                sum_below_w += rz.below_weight as f64;
                sum_rescued_sh += rz.rescued_share_sum as f64;
                if rz.rescued_share_min < min_rescued_sh {
                    min_rescued_sh = rz.rescued_share_min;
                }
                for bin in 0..RESCUE_HIST_BINS {
                    miss_hist[bin] = miss_hist[bin].saturating_add(rz.miss_hist[bin]);
                }
            };

            for rec in &self.rescue_tracker.window {
                fold(&rec.zones[z]);
            }
            if include_cur {
                fold(&self.rescue_tracker.cur.zones[z]);
            }

            used_vals.sort_unstable();
            let p95_used = if used_vals.is_empty() {
                0
            } else {
                let idx = ((used_vals.len() - 1) * 95) / 100;
                used_vals[idx]
            };

            let n = window_steps.max(1) as f64;
            let avg_used = if window_steps > 0 { sum_used as f64 / n } else { 0.0 };
            let avg_budget = if window_steps > 0 {
                sum_budget as f64 / n
            } else {
                base_budget as f64
            };
            let util_pct = if avg_budget > 0.0 {
                (avg_used / avg_budget) * 100.0
            } else {
                0.0
            };
            let capped_pct = if window_steps > 0 {
                (capped_steps as f64 / n) * 100.0
            } else {
                0.0
            };
            let zero_pct = if window_steps > 0 {
                (zero_steps as f64 / n) * 100.0
            } else {
                0.0
            };
            let boot_rescued = self.rescue_tracker.boot_rescued[z]
                + if include_cur {
                    self.rescue_tracker.cur.zones[z].rescued as u64
                } else {
                    0
                };
            let boot_capped_steps = self.rescue_tracker.boot_capped_steps[z]
                + if include_cur
                    && self.rescue_tracker.cur.zones[z].budget > 0
                    && self.rescue_tracker.cur.zones[z].rescued
                        >= self.rescue_tracker.cur.zones[z].budget
                {
                    1
                } else {
                    0
                };
            let boot_capped_pct = if total_steps > 0 {
                (boot_capped_steps as f64 / total_steps as f64) * 100.0
            } else {
                0.0
            };
            let avg_capped_layers = if window_steps > 0 {
                sum_capped_layers as f64 / n
            } else {
                0.0
            };
            let avg_blocked_picks = if window_steps > 0 {
                sum_blocked_picks as f64 / n
            } else {
                0.0
            };
            let avg_below_gate_picks = if window_steps > 0 {
                sum_below_picks as f64 / n
            } else {
                0.0
            };
            let avg_blocked_weight = if window_steps > 0 {
                sum_blocked_w / n
            } else {
                0.0
            };
            let avg_below_weight = if window_steps > 0 {
                sum_below_w / n
            } else {
                0.0
            };
            let rescued_share_avg = if sum_used > 0 {
                sum_rescued_sh / sum_used as f64
            } else {
                0.0
            };
            let rescued_share_min = if min_rescued_sh.is_finite() {
                min_rescued_sh as f64
            } else {
                0.0
            };

            let auto = self.auto_gate;
            let status = match self.tide_cfg.rescue_policy {
                RescuePolicy::Disabled => "off",
                RescuePolicy::Exact => "exact",
                RescuePolicy::Threshold(_) => {
                    if base_budget == 0 {
                        "off"
                    } else if window_steps == 0 {
                        "idle"
                    } else if auto.enabled {
                        // Steered: within 10 points of the target is where it should be.
                        let target_pct = auto.target as f64 * 100.0;
                        if util_pct >= target_pct + 10.0 {
                            "saturated"
                        } else if util_pct >= target_pct - 10.0 {
                            "balanced"
                        } else {
                            "underused"
                        }
                    } else if capped_pct >= 25.0 || util_pct >= 85.0 {
                        "saturated"
                    } else if capped_pct >= 8.0 || util_pct >= 65.0 {
                        "tight"
                    } else if avg_used < 0.25 && avg_below_gate_picks >= 1.5 {
                        "underused"
                    } else if avg_used < 0.05 && avg_below_gate_picks < 0.5 {
                        "idle"
                    } else if util_pct < 5.0 && base_budget >= 6 {
                        "underused"
                    } else {
                        "balanced"
                    }
                }
            };

            RescueZoneSnapshot {
                name: meta[z].0,
                layers_label: meta[z].1,
                gate,
                base_gate: self.configured_zone_gate(z),
                base_budget,
                avg_budget,
                avg_used,
                p95_used,
                max_used,
                util_pct,
                capped_pct,
                zero_pct,
                boot_capped_pct,
                boot_rescued,
                avg_capped_layers,
                avg_blocked_picks,
                avg_below_gate_picks,
                avg_blocked_weight,
                avg_below_weight,
                rescued_share_avg,
                rescued_share_min,
                miss_hist,
                status,
            }
        });

        RescueSnapshot {
            policy,
            auto_gate: self.auto_gate.enabled && policy == "threshold",
            auto_target_pct: self.auto_gate.target * 100.0,
            auto_gate_min: self.auto_gate.min_gate,
            auto_gate_max: self.auto_gate.max_gate,
            window_steps,
            total_steps,
            base_budget_total: base_budgets.iter().sum(),
            last_verify_rows,
            zones,
        }
    }

    #[inline]
    fn zone_of(&self, layer: PhysicalLayerId) -> usize {
        let l = layer.0 as usize;
        if l < self.model_cfg.loop_start_layer {
            0
        } else if self.model_cfg.is_loop_core_layer(l) {
            1
        } else {
            2
        }
    }

    /// Effective per-zone rescue budgets `[prelude, loop_core, coda]` for `verify_rows` (1 for
    /// single-token decode, 2..=8 for single-stream speculative verify, >8 for multi-stream
    /// batched verify where each 8-row stream gets one full stream budget so concurrent requests
    /// keep the same quality floor as single-stream).
    pub fn effective_zone_budgets(&self, verify_rows: usize) -> [usize; 3] {
        let base = match self.rescue_zone_budgets {
            Some(z) => z,
            None => {
                let b = self.rescue_budget;
                if b == 0 {
                    [0, 0, 0]
                } else {
                    let pre = b / 3;
                    let core = b / 4;
                    let coda = b - pre - core;
                    [pre, core, coda]
                }
            }
        };
        if verify_rows <= 1 {
            return base;
        }
        if verify_rows <= 8 {
            return [
                if base[0] > 0 { base[0] + verify_rows / 4 } else { 0 },
                if base[1] > 0 { base[1] + verify_rows / 4 } else { 0 },
                if base[2] > 0 { base[2] + verify_rows / 2 } else { 0 },
            ];
        }
        // Gentle +25% budget per extra 8-row stream (1.25x for 2, 1.50x for 3, 1.75x for 4):
        // matches MoE expert sharing across streams without letting rescues balloon to 40%+ of decode.
        let streams = verify_rows.div_ceil(8).min(8);
        let one_stream = [
            if base[0] > 0 { base[0] + 2 } else { 0 },
            if base[1] > 0 { base[1] + 2 } else { 0 },
            if base[2] > 0 { base[2] + 4 } else { 0 },
        ];
        [
            one_stream[0] * (3 + streams) / 4,
            one_stream[1] * (3 + streams) / 4,
            one_stream[2] * (3 + streams) / 4,
        ]
    }

    /// Active MUST-rescue threshold (`(0, 1]`), if configured.
    pub fn must_rescue_threshold(&self) -> Option<f32> {
        self.rescue_must_threshold.filter(|&v| v > 0.0 && v <= 1.0)
    }

    /// `RescuePolicy::Threshold`: this layer's zone has used up its rescue budget for the current
    /// token / verify step. Other zones (e.g. Coda) keep their own independent budget.
    /// Returns `false` when `rescue_must_threshold` is active so remaining layers still check
    /// for dominant misses that bypass the budget.
    pub fn rescue_budget_spent_for_layer(&self, layer: PhysicalLayerId, verify_rows: usize) -> bool {
        if !matches!(self.tide_cfg.rescue_policy, RescuePolicy::Threshold(_)) {
            return false;
        }
        if self.must_rescue_threshold().is_some() {
            return false;
        }
        let z = self.zone_of(layer);
        self.rescues_by_zone[z] >= self.effective_zone_budgets(verify_rows)[z]
    }

    /// True only once all 3 zones have exhausted their rescue budgets (and no MUST-rescue is active).
    pub fn rescue_budget_spent(&self) -> bool {
        if !matches!(self.tide_cfg.rescue_policy, RescuePolicy::Threshold(_)) {
            return false;
        }
        if self.must_rescue_threshold().is_some() {
            return false;
        }
        let lim = self.effective_zone_budgets(1);
        (0..3).all(|z| self.rescues_by_zone[z] >= lim[z])
    }

    pub fn plan_layer_dispatch(
        &mut self,
        layer: PhysicalLayerId,
        topk_ids: &[u16],
        topk_weights: &[f32],
    ) -> LayerDispatchPlan {
        let mut hit_experts = Vec::with_capacity(topk_ids.len());
        let mut sync_transfers = Vec::new();
        let mut rescued_experts = Vec::new();
        let mut wait_pending = false;

        // Bug fix: a sync rescue must never evict another expert of this same top-k.
        let protect: Vec<ExpertId> = topk_ids
            .iter()
            .map(|&e| ExpertId { layer, expert: e })
            .collect();
        self.cache.set_protected(&protect);
        if self.model_cfg.is_loop_core_layer(layer.0 as usize) {
            self.cache.add_step_protected(&protect);
        }

        let total_weight: f32 = topk_weights.iter().sum();
        let inv_total_w = if total_weight > 1e-8 { 1.0 / total_weight } else { 1.0 };
        let mut executed_weight = 0.0f32;
        self.telemetry.routed_weight_total += 1.0;
        // Threshold (TideLM's inline rescue): only the heaviest missed pick at or above the gate,
        // subject to this layer's zone rescue budget (Prelude / Loop Core / Coda).
        let zone = self.zone_of(layer);
        let zone_cap = self.effective_zone_budgets(1)[zone];
        let zone_gate = self.effective_zone_gate(zone);
        if !self.rescue_tracker.step_is_prefill {
            self.rescue_tracker.step_active = true;
            if self.rescue_tracker.cur.zones[zone].budget == 0 && zone_cap > 0 {
                self.rescue_tracker.cur.zones[zone].budget = zone_cap.min(u16::MAX as usize) as u16;
            }
        }
        let must_tau = self.must_rescue_threshold();
        if matches!(self.tide_cfg.rescue_policy, RescuePolicy::Threshold(_))
            && self.rescues_by_zone[zone] >= zone_cap
        {
            self.note_zone_budget_capped_layer(layer);
        }
        let mut selected_picks: Vec<usize> = Vec::new();
        if matches!(self.tide_cfg.rescue_policy, RescuePolicy::Threshold(_)) {
            if let Some(m_tau) = must_tau {
                let mut must_idxs: Vec<(usize, f32)> = topk_ids
                    .iter()
                    .zip(topk_weights.iter())
                    .enumerate()
                    .filter_map(|(i, (&e, &w))| {
                        let id = ExpertId { layer, expert: e };
                        let nw = w * inv_total_w;
                        (nw >= m_tau && self.cache.entry(id).vram_slot.is_missing()).then_some((i, nw))
                    })
                    .collect();
                must_idxs.sort_by(|a, b| b.1.total_cmp(&a.1));
                for (i, _) in must_idxs.into_iter().take(crate::runtime::DEFAULT_SCRATCH_RING_SLOTS) {
                    selected_picks.push(i);
                }
            }
            if self.rescues_by_zone[zone] + selected_picks.len() < zone_cap
                && selected_picks.len() < crate::runtime::DEFAULT_SCRATCH_RING_SLOTS
            {
                let tau = zone_gate;
                if let Some((i, _)) = topk_ids
                    .iter()
                    .zip(topk_weights.iter())
                    .enumerate()
                    .filter(|(i, (&e, &w))| {
                        !selected_picks.contains(i)
                            && w * inv_total_w >= tau
                            && self.cache.entry(ExpertId { layer, expert: e }).vram_slot.is_missing()
                    })
                    .max_by(|a, b| a.1 .1.total_cmp(b.1 .1))
                {
                    selected_picks.push(i);
                }
            }
        }
        if !selected_picks.is_empty() {
            self.rescues_this_token += selected_picks.len();
            self.rescues_by_zone[zone] += selected_picks.len();
        }

        for (k, (&exp, &w)) in topk_ids.iter().zip(topk_weights.iter()).enumerate() {
            let id = ExpertId { layer, expert: exp };
            let norm_w = w * inv_total_w;
            self.cache.touch_expert(id, norm_w);

            let entry = self.cache.entry(id);
            if self.cache.is_vram_pending(id) && !entry.is_scratch_slot {
                // Async promotion already in flight: wait for it on the GPU instead of re-copying.
                wait_pending = true;
            }
            if !entry.vram_slot.is_missing() {
                self.record_hit(layer);
                hit_experts.push(ActiveExpertSlot {
                    expert: id,
                    vram_slot: entry.vram_slot,
                    weight: w,
                });
                executed_weight += w;
                continue;
            }

            // VRAM Miss! Threshold tau is evaluated against normalized routing share [0..1]
            // (so it works identically regardless of route_scale = 8.0!).
            self.record_miss(layer);
            if !self.rescue_tracker.step_is_prefill {
                let bin = rescue_hist_bin(norm_w);
                self.rescue_tracker.cur.zones[zone].miss_hist[bin] += 1;
            }
            let should_sync_rescue = match self.tide_cfg.rescue_policy {
                RescuePolicy::Exact => true,
                RescuePolicy::Threshold(_) => selected_picks.contains(&k),
                RescuePolicy::Disabled => false,
            };

            if should_sync_rescue {
                self.event_share = norm_w;
                let mut rescued_ok = false;
                // If hot enough (or free space exists), promote into a resident VRAM slot;
                // otherwise use the transient Scratch Rescue Ring so cold 1-off misses
                // never thrash warm resident experts!
                if self.cache.should_promote_instead_of_scratch(id) {
                    let pin = !self.cache.has_priors() && self.model_cfg.is_loop_core_layer(layer.0 as usize);
                    if let Some(op) = self.cache.promote_to_vram(id, pin, true) {
                        self.record_transfer_telemetry(&op, true);
                        sync_transfers.push(op);
                        let new_slot = self.cache.entry(id).vram_slot;
                        rescued_experts.push(ActiveExpertSlot {
                            expert: id,
                            vram_slot: new_slot,
                            weight: w,
                        });
                        executed_weight += w;
                        rescued_ok = true;
                    } else {
                        self.telemetry.stage1_skipped_tail_misses += 1;
                        self.telemetry.routed_weight_skipped += norm_w as f64;
                    }
                } else if let Some(op) = self.cache.stage_scratch_rescue(id) {
                    self.record_transfer_telemetry(&op, true);
                    let scratch_slot = self.cache.entry(id).vram_slot;
                    sync_transfers.push(op);
                    rescued_experts.push(ActiveExpertSlot {
                        expert: id,
                        vram_slot: scratch_slot,
                        weight: w,
                    });
                    executed_weight += w;
                    rescued_ok = true;
                    // Also enqueue scratch-rescued experts for background promotion so recurring
                    // misses graduate into permanent resident VRAM slots instead of re-rescuing!
                    if self.cache.entry(id).ram_slot.is_missing() {
                        *self.pending_async_ram.entry(id).or_insert(0.0) += norm_w;
                    }
                    *self.pending_async_vram.entry(id).or_insert(0.0) += norm_w;
                } else {
                    self.telemetry.stage1_skipped_tail_misses += 1;
                    self.telemetry.routed_weight_skipped += norm_w as f64;
                }
                if rescued_ok {
                    self.step_rescued.push(id);
                }
                if !self.rescue_tracker.step_is_prefill {
                    let rz = &mut self.rescue_tracker.cur.zones[zone];
                    if rescued_ok {
                        rz.rescued = rz.rescued.saturating_add(1);
                        rz.rescued_share_sum += norm_w;
                        if norm_w < rz.rescued_share_min {
                            rz.rescued_share_min = norm_w;
                        }
                    } else if norm_w >= zone_gate {
                        rz.blocked_picks = rz.blocked_picks.saturating_add(1);
                        rz.blocked_weight += norm_w;
                    } else {
                        rz.below_gate_picks = rz.below_gate_picks.saturating_add(1);
                        rz.below_weight += norm_w;
                    }
                }
            } else {
                // Skip this miss for zero-stall decode and enqueue for paced async promotion!
                self.telemetry.stage1_skipped_tail_misses += 1;
                self.telemetry.routed_weight_skipped += norm_w as f64;
                if !self.rescue_tracker.step_is_prefill {
                    let rz = &mut self.rescue_tracker.cur.zones[zone];
                    if norm_w >= zone_gate {
                        rz.blocked_picks = rz.blocked_picks.saturating_add(1);
                        rz.blocked_weight += norm_w;
                    } else {
                        rz.below_gate_picks = rz.below_gate_picks.saturating_add(1);
                        rz.below_weight += norm_w;
                    }
                }
                let prio = norm_w;
                if entry.ram_slot.is_missing() {
                    *self.pending_async_ram.entry(id).or_insert(0.0) += prio;
                }
                *self.pending_async_vram.entry(id).or_insert(0.0) += prio;
            }
        }

        self.cache.clear_protected();
        let weight_renorm_scale = if executed_weight > 1e-8 && executed_weight < total_weight {
            total_weight / executed_weight
        } else {
            1.0
        };

        LayerDispatchPlan {
            hit_experts,
            sync_transfers,
            rescued_experts,
            weight_renorm_scale,
            wait_pending,
        }
    }

    /// Inline rescue planning for a multi-row speculative `verify` layer (`m <= 8` rows):
    /// protects every expert routed by any row in this layer, aggregates missing experts across all
    /// rows, and under `Threshold` rescues highest-impact missing experts whose max single-row
    /// normalized share meets the layer gate, using independent per-zone budgets `[prelude, core, coda]`.
    pub fn plan_verify_layer_dispatch(
        &mut self,
        layer: PhysicalLayerId,
        rows_ids: &[[u16; 8]],
        rows_ws: &[[f32; 8]],
    ) -> (Vec<TransferOp>, bool, Vec<ExpertId>) {
        let m = rows_ids.len();
        let mut sync_transfers = Vec::new();
        let mut rescued = Vec::new();
        let mut wait_pending = false;

        // Protect every expert routed by any row of this layer so a rescue never evicts one.
        let mut protect: Vec<ExpertId> = Vec::with_capacity(m * 8);
        for row in rows_ids {
            for &e in row {
                let id = ExpertId { layer, expert: e };
                if !protect.contains(&id) {
                    protect.push(id);
                }
            }
        }
        self.cache.set_protected(&protect);
        if self.model_cfg.is_loop_core_layer(layer.0 as usize) {
            self.cache.add_step_protected(&protect);
        }

        for &id in &protect {
            let entry = self.cache.entry(id);
            if self.cache.is_vram_pending(id) && !entry.is_scratch_slot {
                wait_pending = true;
            }
        }

        if let RescuePolicy::Threshold(_) = self.tide_cfg.rescue_policy {
            let zone = self.zone_of(layer);
            let zone_cap = self.effective_zone_budgets(m)[zone];
            let tau = self.effective_zone_gate(zone);
            let must_tau = self.must_rescue_threshold();
            if !self.rescue_tracker.step_is_prefill {
                self.rescue_tracker.step_active = true;
                self.rescue_tracker.cur.verify_rows = m.max(1);
                self.rescue_tracker.cur.zones[zone].budget = zone_cap.min(u16::MAX as usize) as u16;
            }
            if self.rescues_by_zone[zone] >= zone_cap {
                self.note_zone_budget_capped_layer(layer);
            }
            if self.rescues_by_zone[zone] < zone_cap || must_tau.is_some() {
                // Aggregate (effective_max_share, weighted_sum_share) per unique missing expert.
                // In speculative draft-verify passes, row 0 is always committed while tail rows are
                // progressively more likely to be rejected, so discount tail rows by survival weight
                // to avoid stalling the GPU for minor misses on discarded draft tokens.
                const SPEC_ROW_SURVIVAL: [f32; 8] = [1.00, 0.85, 0.65, 0.45, 0.35, 0.30, 0.25, 0.25];
                let decay = self.speculative_verify && self.spec_row_decay;
                let mut miss_stats: Vec<(ExpertId, f32, f32)> = Vec::new();
                for (row_idx, (r_ids, r_ws)) in rows_ids.iter().zip(rows_ws.iter()).enumerate() {
                    let tot: f32 = r_ws.iter().sum();
                    let inv = if tot > 1e-8 { 1.0 / tot } else { 1.0 };
                    let row_w = if decay {
                        SPEC_ROW_SURVIVAL[row_idx.min(SPEC_ROW_SURVIVAL.len() - 1)]
                    } else {
                        1.0
                    };
                    for (&e, &w) in r_ids.iter().zip(r_ws.iter()) {
                        let id = ExpertId { layer, expert: e };
                        if !self.cache.entry(id).vram_slot.is_missing() {
                            continue;
                        }
                        let nw = w * inv * row_w;
                        if let Some(st) = miss_stats.iter_mut().find(|s| s.0 == id) {
                            st.1 = st.1.max(nw);
                            st.2 += nw;
                        } else {
                            miss_stats.push((id, nw, nw));
                        }
                    }
                }
                let ring_cap = crate::runtime::DEFAULT_SCRATCH_RING_SLOTS;
                // Partition into MUST-rescues (bypassing both budget & auto-gate) and budgeted misses.
                let mut must_stats: Vec<(ExpertId, f32, f32)> = Vec::new();
                let mut budget_stats: Vec<(ExpertId, f32, f32)> = Vec::new();
                for s in miss_stats {
                    if must_tau.is_some_and(|mt| s.1 >= mt) {
                        must_stats.push(s);
                    } else if s.1 >= tau || (decay && s.2 >= tau * 1.25) {
                        budget_stats.push(s);
                    }
                }
                must_stats.sort_by(|a, b| b.2.total_cmp(&a.2));
                must_stats.truncate(ring_cap);
                budget_stats.sort_by(|a, b| b.2.total_cmp(&a.2));
                // 1/layer in Loop Core (36 passes); scale Prelude & Coda up to 3/layer when budget > 12.
                // With the auto gate the block's budget is the limit: the gate is what keeps the
                // block near its share of the budget, and 1-3 a layer would leave picks above the
                // gate skipped with budget to spare. A layer still takes no more than the rescue
                // ring holds (a ninth rescue would push out the layer's first).
                let max_per_layer = if self.auto_gate.enabled {
                    ring_cap
                } else if zone == 1 {
                    1
                } else if m >= 4 {
                    zone_cap.div_ceil(6).clamp(2, 3)
                } else {
                    zone_cap.div_ceil(6).clamp(1, 2)
                };
                let rem_budget = zone_cap.saturating_sub(self.rescues_by_zone[zone] + must_stats.len());
                let rem_layer = max_per_layer.saturating_sub(must_stats.len());
                let take_budget = rem_layer.min(rem_budget);
                must_stats.extend(budget_stats.into_iter().take(take_budget));
                for (id, max_share, _sum_share) in must_stats {
                    self.event_share = max_share;
                    let mut rescued_ok = false;
                    if self.cache.should_promote_instead_of_scratch(id) {
                        let pin = !self.cache.has_priors() && self.model_cfg.is_loop_core_layer(layer.0 as usize);
                        if let Some(op) = self.cache.promote_to_vram(id, pin, true) {
                            self.record_transfer_telemetry(&op, true);
                            sync_transfers.push(op);
                            rescued.push(id);
                            rescued_ok = true;
                        }
                    } else if let Some(op) = self.cache.stage_scratch_rescue(id) {
                        self.record_transfer_telemetry(&op, true);
                        sync_transfers.push(op);
                        rescued.push(id);
                        rescued_ok = true;
                        if self.cache.entry(id).ram_slot.is_missing() {
                            *self.pending_async_ram.entry(id).or_insert(0.0) += max_share;
                        }
                        *self.pending_async_vram.entry(id).or_insert(0.0) += max_share;
                    }
                    if rescued_ok {
                        self.rescues_this_token += 1;
                        self.rescues_by_zone[zone] += 1;
                        self.step_rescued.push(id);
                    }
                    if rescued_ok && !self.rescue_tracker.step_is_prefill {
                        let rz = &mut self.rescue_tracker.cur.zones[zone];
                        rz.rescued = rz.rescued.saturating_add(1);
                        rz.rescued_share_sum += max_share;
                        if max_share < rz.rescued_share_min {
                            rz.rescued_share_min = max_share;
                        }
                    }
                }
            }
        }

        self.cache.clear_protected();
        (sync_transfers, wait_pending, rescued)
    }

    /// Bookkeeping for one layer of one row of a batched decode round. Batched rounds never rescue
    /// (that would stall every lane): the kernels skipped unpublished experts, so this counts hits
    /// and misses, adds heat and skipped routing weight, and queues misses for async promotion.
    pub fn record_batch_dispatch(&mut self, layer: PhysicalLayerId, topk_ids: &[u16], topk_weights: &[f32]) {
        self.record_batch_dispatch_with_rescued(layer, topk_ids, topk_weights, &[]);
    }

    /// Like `record_batch_dispatch`, crediting experts that were rescued inline during a verify
    /// pass (even if their transient scratch slot was reused by a later layer before this runs).
    pub fn record_batch_dispatch_with_rescued(
        &mut self,
        layer: PhysicalLayerId,
        topk_ids: &[u16],
        topk_weights: &[f32],
        rescued: &[ExpertId],
    ) {
        let total: f32 = topk_weights.iter().sum();
        let inv = if total > 1e-8 { 1.0 / total } else { 1.0 };
        let zone = self.zone_of(layer);
        let zone_gate = self.effective_zone_gate(zone);
        if !self.rescue_tracker.step_is_prefill {
            self.rescue_tracker.step_active = true;
        }
        self.telemetry.routed_weight_total += 1.0;
        for (&exp, &w) in topk_ids.iter().zip(topk_weights.iter()) {
            let id = ExpertId { layer, expert: exp };
            let norm_w = w * inv;
            self.cache.touch_expert(id, norm_w);
            if self.cache.is_vram_hit(id) {
                self.record_hit(layer);
                continue;
            }
            if !self.rescue_tracker.step_is_prefill {
                let bin = rescue_hist_bin(norm_w);
                self.rescue_tracker.cur.zones[zone].miss_hist[bin] += 1;
            }
            if rescued.contains(&id) {
                self.record_miss(layer);
                continue;
            }
            self.record_miss(layer);
            self.telemetry.stage1_skipped_tail_misses += 1;
            self.telemetry.routed_weight_skipped += norm_w as f64;
            if !self.rescue_tracker.step_is_prefill {
                let rz = &mut self.rescue_tracker.cur.zones[zone];
                if norm_w >= zone_gate {
                    rz.blocked_picks = rz.blocked_picks.saturating_add(1);
                    rz.blocked_weight += norm_w;
                } else {
                    rz.below_gate_picks = rz.below_gate_picks.saturating_add(1);
                    rz.below_weight += norm_w;
                }
            }
            let entry = self.cache.entry(id);
            if entry.ram_slot.is_missing() {
                *self.pending_async_ram.entry(id).or_insert(0.0) += norm_w;
            }
            *self.pending_async_vram.entry(id).or_insert(0.0) += norm_w;
        }
    }

    /// Mode 3: Paced Async Background Tide Rise (called once per decode token).
    /// Emits up to `max_async_ram_promotions_per_token` Stage 2 (Disk->RAM) ops and
    /// `max_async_vram_promotions_per_token` Stage 1 (RAM->VRAM) ops.
    pub fn plan_async_step(&mut self) -> Vec<TransferOp> {
        self.event_kind = crate::expert_map::EV_HOT;
        self.event_share = 0.0;
        self.cache.advance_step();
        let mut ops = Vec::new();

        // Stage 2: Disk -> RAM (paced)
        if !self.pending_async_ram.is_empty() && self.tide_cfg.max_async_ram_promotions_per_token > 0
        {
            let mut cands: Vec<(ExpertId, f32)> = self.pending_async_ram.drain().collect();
            cands.sort_by(|a, b| b.1.total_cmp(&a.1));
            let mut done = 0usize;
            for (id, prio) in cands {
                if done < self.tide_cfg.max_async_ram_promotions_per_token {
                    if let Some(op) = self.cache.promote_to_ram_with_flag(id, true) {
                        self.record_transfer_telemetry(&op, false);
                        ops.push(op);
                        done += 1;
                    }
                } else {
                    // Keep top tail for next token with mild decay
                    if prio > 0.05 {
                        self.pending_async_ram.insert(id, prio * 0.9);
                    }
                }
            }
        }

        // Stage 2 repair: an expert that left the card with no copy in RAM is on the disk only,
        // and a rescue of it reads the disk while the GPU waits. Read such experts back in the
        // background, most valuable first. Only into free RAM or in place of a copy of an expert
        // that is on the card: a RAM tier full of off-card experts must not churn.
        if self.tide_cfg.max_async_ram_promotions_per_token > 0 {
            const REPAIRS_PER_STEP: usize = 8;
            let mut disk_only: Vec<(ExpertId, f64)> = Vec::new();
            let mut spare_copies = 0usize;
            for l in 0..self.model_cfg.num_physical_layers {
                for x in 0..self.model_cfg.n_routed_experts {
                    let id = ExpertId { layer: PhysicalLayerId(l as u16), expert: x as u16 };
                    let e = self.cache.entry(id);
                    let on_card = !e.vram_slot.is_missing() && !e.is_scratch_slot;
                    if e.ram_slot.is_missing() {
                        if !on_card {
                            disk_only.push((id, self.cache.value(id)));
                        }
                    } else if on_card && !e.ram_in_flight {
                        spare_copies += 1;
                    }
                }
            }
            if !disk_only.is_empty() {
                disk_only.sort_by(|a, b| b.1.total_cmp(&a.1));
                let mut free = self.cache.ram_budget_bytes().saturating_sub(self.cache.ram_used_bytes());
                for (id, _) in disk_only.into_iter().take(REPAIRS_PER_STEP) {
                    let need = self.cache.expert_bytes(id);
                    if free >= need {
                        free -= need;
                    } else if spare_copies > 0 {
                        spare_copies -= 1;
                    } else {
                        break;
                    }
                    if let Some(op) = self.cache.promote_to_ram_with_flag(id, true) {
                        self.record_transfer_telemetry(&op, false);
                        ops.push(op);
                    }
                }
            }
        }

        // Stage 1: RAM -> VRAM (paced, only promoting experts whose RAM slot is Ready!).
        // Candidates in value order; each class's eviction floor is computed once and refreshed
        // only after a promotion (a full scan per candidate cost ~10 ms of CPU per token).
        if self.tide_cfg.max_async_vram_promotions_per_token == 0 {
            self.pending_async_vram.clear();
        } else if !self.pending_async_vram.is_empty() {
            let drained: Vec<(ExpertId, f32)> = self.pending_async_vram.drain().collect();
            let mut cands: Vec<(ExpertId, f32, f64)> = drained
                .into_iter()
                .filter(|(id, _)| {
                    let e = self.cache.entry(*id);
                    e.vram_slot.is_missing() || e.is_scratch_slot
                })
                .map(|(id, prio)| (id, prio, self.cache.value(id)))
                .collect();
            cands.sort_by(|a, b| b.2.total_cmp(&a.2));
            let mut floors: [Option<Option<f64>>; 2] = [None, None];
            let mut done = 0usize;
            for (id, prio, val) in cands {
                let keep = |s: &mut Self| {
                    if prio > 0.05 {
                        s.pending_async_vram.insert(id, prio * 0.9);
                    }
                };
                if done >= self.tide_cfg.max_async_vram_promotions_per_token || !self.cache.is_ram_ready(id) {
                    keep(self);
                    continue;
                }
                let cls = self.cache.expert_class(id);
                let ci = cls as usize;
                let need = self.cache.expert_bytes(id);
                let floor = *floors[ci].get_or_insert_with(|| self.cache.promotion_floor(cls, need));
                let ok = match floor {
                    None => true,
                    Some(f) => val > f * self.async_hysteresis + 1e-9,
                };
                if !ok {
                    // Not worth a resident slot (yet): repeated misses raise its heat.
                    keep(self);
                    continue;
                }
                let pin = !self.cache.has_priors() && self.model_cfg.is_loop_core_layer(id.layer.0 as usize);
                if let Some(op) = self.cache.promote_to_vram(id, pin, false) {
                    self.record_transfer_telemetry(&op, false);
                    self.telemetry.decode_async_promotions += 1;
                    ops.push(op.clone());
                    self.keep_evicted_in_ram(&op, &mut ops);
                    done += 1;
                    floors[ci] = None;
                }
            }
            // Bound the queue (lowest claims dropped).
            if self.pending_async_vram.len() > 512 {
                let mut v: Vec<(ExpertId, f32)> = self.pending_async_vram.drain().collect();
                v.sort_by(|a, b| b.1.total_cmp(&a.1));
                v.truncate(512);
                self.pending_async_vram.extend(v);
            }
        }

        ops
    }

    /// Periodic Mid-Decode / End-of-Request Tide Rebalance:
    /// Every `interval_tokens` during long generations (or when `force = true` at request end),
    /// swaps any non-VRAM experts whose live EWMA heat beats the coldest non-pinned VRAM slot
    /// by >= 25% hysteresis.
    #[allow(clippy::too_many_arguments)]
    pub fn maybe_periodic_rebalance(
        &mut self,
        decoded_tokens: u64,
        interval_tokens: u64,
        max_swaps: usize,
        force: bool,
    ) -> Vec<TransferOp> {
        self.event_kind = crate::expert_map::EV_ADAPT;
        self.event_share = 0.0;
        if !force {
            if interval_tokens == 0 || decoded_tokens == 0 {
                return Vec::new();
            }
            if decoded_tokens < self.last_periodic_rebalance_tok {
                self.last_periodic_rebalance_tok = 0;
            }
            if decoded_tokens.saturating_sub(self.last_periodic_rebalance_tok) < interval_tokens {
                return Vec::new();
            }
        }
        self.last_periodic_rebalance_tok = decoded_tokens;
        let heat_snap = self.cache.export_heat_snapshot();
        let mut cands: Vec<(ExpertId, f64)> = heat_snap
            .iter()
            .enumerate()
            .filter_map(|(idx, &h)| {
                if h <= 1e-4 {
                    return None;
                }
                let id = self.cache.expert_from_flat(idx);
                let e = self.cache.entry(id);
                if !e.vram_slot.is_missing() && !e.is_scratch_slot {
                    return None;
                }
                Some((id, self.cache.value(id)))
            })
            .collect();
        cands.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut ops = Vec::new();
        let mut floors: [Option<Option<f64>>; 2] = [None, None];
        for (id, val) in cands {
            // swaps only: `ops` also carries the background reads of `keep_evicted_in_ram`
            if ops.iter().filter(|o| !matches!(o, TransferOp::DiskToRam { .. })).count() >= max_swaps {
                break;
            }
            let cls = self.cache.expert_class(id);
            let ci = cls as usize;
            let need = self.cache.expert_bytes(id);
            let floor = *floors[ci].get_or_insert_with(|| self.cache.promotion_floor(cls, need));
            if let Some(f) = floor {
                if val <= f * self.cache.swap_hysteresis + 1e-9 {
                    continue;
                }
            }
            floors[ci] = None;
            if !self.cache.is_ram_ready(id) {
                // Bug fix: never block decode on NVMe. Queue an async Disk->RAM read (if not
                // already in flight) and let Mode 3 promote it once it is Ready.
                if let Some(op) = self.cache.promote_to_ram_with_flag(id, true) {
                    self.record_transfer_telemetry(&op, false);
                    ops.push(op);
                }
                *self.pending_async_vram.entry(id).or_insert(0.0) += 1.0;
                continue;
            }
            let pin = !self.cache.has_priors() && self.model_cfg.is_loop_core_layer(id.layer.0 as usize);
            if let Some(op) = self.cache.promote_to_vram(id, pin, false) {
                self.record_transfer_telemetry(&op, false);
                ops.push(op.clone());
                self.keep_evicted_in_ram(&op, &mut ops);
            }
        }
        ops
    }

    /// Folds this request's routing demand into `priors` as one vote (see `TidePriors::merge_demand`)
    /// and resets the per-request demand.
    pub fn consolidate_priors(&mut self, priors: &mut TidePriors, newest_share: f32) {
        let demand = self.cache.take_demand();
        priors.merge_demand(&demand, newest_share);
    }

    fn record_hit(&mut self, layer: PhysicalLayerId) {
        let l = layer.0 as usize;
        if l < self.model_cfg.loop_start_layer {
            self.telemetry.vram_hits_prelude += 1;
        } else if l < self.model_cfg.loop_end_exclusive {
            self.telemetry.vram_hits_loop += 1;
        } else {
            self.telemetry.vram_hits_coda += 1;
        }
    }

    fn record_miss(&mut self, layer: PhysicalLayerId) {
        let l = layer.0 as usize;
        if l < self.model_cfg.loop_start_layer {
            self.telemetry.vram_misses_prelude += 1;
        } else if l < self.model_cfg.loop_end_exclusive {
            self.telemetry.vram_misses_loop += 1;
        } else {
            self.telemetry.vram_misses_coda += 1;
        }
    }

    fn record_transfer_telemetry(&mut self, op: &TransferOp, is_sync_rescue: bool) {
        match op {
            TransferOp::DiskToRam { expert, .. } => {
                self.telemetry.stage2_disk_reads += 1;
                self.telemetry.stage2_bytes_read += self.cache.expert_bytes(*expert) as u64;
            }
            TransferOp::RamToVram { expert, .. } => {
                if is_sync_rescue {
                    self.telemetry.stage1_sync_rescues += 1;
                } else {
                    self.telemetry.stage1_async_promotions += 1;
                }
                self.telemetry.stage1_bytes_transferred += self.cache.expert_bytes(*expert) as u64;
            }
            TransferOp::DiskToVramDirect { expert, .. } => {
                let b = self.cache.expert_bytes(*expert) as u64;
                self.telemetry.stage2_disk_reads += 1;
                self.telemetry.stage2_bytes_read += b;
                if is_sync_rescue {
                    self.telemetry.rescue_disk_reads += 1;
                    self.telemetry.stage1_sync_rescues += 1;
                } else {
                    self.telemetry.stage1_async_promotions += 1;
                }
                self.telemetry.stage1_bytes_transferred += b;
            }
            TransferOp::ScratchRescue {
                expert,
                src_ram_slot,
                ..
            } => {
                let b = self.cache.expert_bytes(*expert) as u64;
                if src_ram_slot.is_none() {
                    self.telemetry.stage2_disk_reads += 1;
                    self.telemetry.rescue_disk_reads += 1;
                    self.telemetry.stage2_bytes_read += b;
                }
                self.telemetry.stage1_sync_rescues += 1;
                self.telemetry.stage1_bytes_transferred += b;
            }
        }
        // Expert map: what came into VRAM, why, and whose slot it took.
        let kind = if is_sync_rescue { crate::expert_map::EV_RESCUE } else { self.event_kind };
        let ev = match *op {
            TransferOp::RamToVram { expert, evicted_from_vram, .. } | TransferOp::DiskToVramDirect { expert, evicted_from_vram, .. } => {
                Some((expert, kind, evicted_from_vram))
            }
            // a transient scratch slot: used for its pass, not kept
            TransferOp::ScratchRescue { expert, .. } => Some((expert, crate::expert_map::EV_DROPPED, None)),
            TransferOp::DiskToRam { .. } => None,
        };
        if let Some((e, k, victim)) = ev {
            if k != 0 {
                self.events.push(e.layer.0 as usize, e.expert as usize, k, self.event_share,
                    victim.map(|v| (v.layer.0 as usize, v.expert as usize)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: usize = 1024 * 1024 * 1024;

    #[test]
    fn test_rx9070xt_loop_core_100pct_and_elastic_327k_kv() {
        let model_cfg = Inkling2Config::default();
        let tide_cfg = TideConfig::default();
        // RX 9070 XT: 11.85 GiB unified KV+Expert VRAM, 16 GiB RAM budget
        let mut sched = TideScheduler::new(model_cfg, tide_cfg, (11.85 * GIB as f64) as usize, 16 * GIB);

        // Mode 1: Pre-prefill warm-up at short context (0.6 GiB KV)
        sched.cache.set_kv_reserved_bytes((0.6 * GIB as f64) as usize);
        let ops = sched.plan_pre_prefill(None);
        sched.cache.publish_all_pending(); // executor would publish once DMAs land
        assert!(!ops.is_empty());

        let (pre, loop_c, coda) = sched.cache.count_vram_resident();
        // 100% of Loop Core (12 layers x 128 = 1,536 experts) MUST be resident in VRAM!
        assert_eq!(loop_c, 1536, "Expected 100% Loop Core in VRAM");
        assert!(pre + coda > 800, "Expected >800 outer-layer experts in VRAM, got {}", pre + coda);
        // Exclusive Loop-Core pinning keeps RAM usage lean (~8.2 GiB max for outer layers, 0 for Loop Core)
        assert!(sched.cache.ram_used_bytes() < 9 * GIB);

        // Elastic KV growth to 327K FP8 KV (5.1 GiB): cold outer experts gracefully recede, Loop Core stays 100%!
        let evicted = sched.cache.set_kv_reserved_bytes((5.1 * GIB as f64) as usize);
        assert!(!evicted.is_empty());
        let (_pre2, loop_c2, _coda2) = sched.cache.count_vram_resident();
        assert_eq!(loop_c2, 1536, "Loop Core must remain 100% intact at 327K FP8 KV");
        assert!(sched.cache.vram_used_bytes() <= sched.cache.vram_expert_budget_bytes());
    }

    #[test]
    fn test_startup_ram_holds_non_resident_experts_first() {
        // 6 GiB VRAM for experts, 9 GiB RAM: RAM must hold every expert that is not in VRAM
        // before it spends anything on copies of VRAM-resident ones.
        let model_cfg = Inkling2Config::default();
        let tide_cfg = TideConfig::default();
        let mut sched = TideScheduler::new(model_cfg.clone(), tide_cfg, 6 * GIB, 9 * GIB);
        let _ = sched.plan_pre_prefill(None);
        sched.cache.publish_all_pending();
        let total = model_cfg.num_physical_layers * model_cfg.n_routed_experts;
        let (mut non_res, mut non_res_in_ram, mut res_in_ram, mut non_res_bytes) = (0usize, 0usize, 0usize, 0usize);
        for i in 0..total {
            let id = sched.cache.expert_from_flat(i);
            let e = sched.cache.entry(id);
            if e.vram_slot.is_missing() {
                non_res += 1;
                non_res_bytes += sched.cache.expert_bytes(id);
                non_res_in_ram += (!e.ram_slot.is_missing()) as usize;
            } else {
                res_in_ram += (!e.ram_slot.is_missing()) as usize;
            }
        }
        assert!(non_res > 0);
        if non_res_bytes <= 9 * GIB {
            assert_eq!(non_res_in_ram, non_res, "every non-resident expert must be in RAM");
        } else {
            assert_eq!(res_in_ram, 0, "no RAM copies of resident experts while non-resident ones are missing");
        }
    }

    #[test]
    fn test_stage2_spillover_and_four_modes() {
        let model_cfg = Inkling2Config::default();
        let mut tide_cfg = TideConfig::default();
        tide_cfg.rescue_policy = RescuePolicy::Disabled;
        // Tight RAM budget (2 GiB) forces Stage 2 Disk->RAM streaming!
        let mut sched = TideScheduler::new(model_cfg, tide_cfg, 8 * GIB, 2 * GIB);
        let _ = sched.plan_pre_prefill(None);
        sched.cache.publish_all_pending();
        assert!(sched.cache.ram_used_bytes() <= 2 * GIB);

        // Mode 2: Prefill activates specific outer-layer Coda experts (layer 22)
        let hot_ids = [100u16, 101, 102, 103, 104, 105, 106, 107];
        let weights = [0.20f32, 0.18, 0.16, 0.14, 0.12, 0.08, 0.07, 0.05];
        for _ in 0..10 {
            sched.record_prefill_routing(PhysicalLayerId(22), &hot_ids, &weights);
        }
        let rebal = sched.plan_prefill_decode_rebalance(32);
        assert!(!rebal.is_empty());
        // Async: experts not in RAM get a disk -> RAM read first (never a sync disk read);
        // once those land, the paced async steps promote them.
        for op in &rebal {
            if let TransferOp::DiskToRam { expert, .. } = op {
                sched.cache.mark_ram_ready(*expert);
            }
        }
        sched.tide_cfg.max_async_vram_promotions_per_token = 8;
        for _ in 0..4 {
            let _ = sched.plan_async_step();
        }
        sched.cache.publish_all_pending();
        for &e in &hot_ids {
            assert!(
                !sched.cache.entry(ExpertId { layer: PhysicalLayerId(22), expert: e }).vram_slot.is_missing(),
                "Rebalance should have loaded prefill-hot expert {e} into VRAM"
            );
        }

        // Mode 4 (Disabled rescue) + Mode 3 (Paced Async): touch an uncached layer 23 expert
        let cold_ids = [120u16, 121, 122, 123, 124, 125, 126, 127];
        let plan = sched.plan_layer_dispatch(PhysicalLayerId(23), &cold_ids, &weights);
        assert!(plan.sync_transfers.is_empty(), "Rescue Disabled must emit zero sync transfers");
        assert!(plan.weight_renorm_scale >= 1.0);

        // Async step 1 pulls Stage 2 Disk->RAM (paced <= 2, marked ram_in_flight=true)
        let async1 = sched.plan_async_step();
        assert!(!async1.is_empty());
        // Simulate background Stage2IoWorker completing those reads:
        for op in &async1 {
            if let TransferOp::DiskToRam { expert, .. } = op {
                sched.cache.mark_ram_ready(*expert);
            }
        }
        // Async step 2 now promotes the RAM-ready experts into VRAM!
        let async2 = sched.plan_async_step();
        assert!(!async2.is_empty());
    }

    #[test]
    fn test_scratch_rescue_preserves_hot_residents_and_periodic_rebalance() {
        let model_cfg = Inkling2Config::default();
        let mut tide_cfg = TideConfig::default();
        tide_cfg.rescue_policy = RescuePolicy::Exact;
        let mut sched = TideScheduler::new(model_cfg, tide_cfg, 8 * GIB, 16 * GIB);
        let _ = sched.plan_pre_prefill(None);
        sched.cache.publish_all_pending();

        // Warm up ALL currently VRAM-resident outer experts so they are hot (heat >= 5.0)
        for l in 0..24u16 {
            for e in 0..128u16 {
                let id = ExpertId { layer: PhysicalLayerId(l), expert: e };
                if !sched.cache.entry(id).vram_slot.is_missing() {
                    sched.cache.touch_expert(id, 5.0);
                }
            }
        }
        let resident_before = sched.cache.count_vram_resident();

        // Trigger a cold 1-off sync miss on layer 23 (heat = 0.15 << 5.0 * 1.15)
        let cold_miss = ExpertId { layer: PhysicalLayerId(23), expert: 127 };
        assert!(sched.cache.entry(cold_miss).vram_slot.is_missing());
        let plan = sched.plan_layer_dispatch(PhysicalLayerId(23), &[127], &[0.15]);
        assert_eq!(plan.sync_transfers.len(), 1);
        assert!(
            matches!(plan.sync_transfers[0], TransferOp::ScratchRescue { .. }),
            "Cold 1-off sync miss should use ScratchRescue instead of evicting hot residents"
        );
        // Zero warm residents were evicted!
        assert_eq!(sched.cache.count_vram_resident(), resident_before);

        // Now hammer expert (23, 126) until its heat >> 5.0 * 1.25 -> periodic rebalance promotes it!
        let super_hot = ExpertId { layer: PhysicalLayerId(23), expert: 126 };
        for _ in 0..20 {
            sched.cache.touch_expert(super_hot, 1.0);
        }
        let periodic = sched.maybe_periodic_rebalance(256, 256, 4, false);
        assert!(!periodic.is_empty(), "Periodic rebalance must promote super-hot expert");
        assert!(
            !sched.cache.entry(super_hot).vram_slot.is_missing()
                && !sched.cache.entry(super_hot).is_scratch_slot
        );

        // Consolidate into TidePriors at end of request
        let mut priors = TidePriors::new("Inkling-2-Mini", 24, 128);
        sched.consolidate_priors(&mut priors, 0.4);
        assert_eq!(priors.sessions_recorded, 1);
        assert!(priors.scores[23 * 128 + 126] > 0.5);
    }

    fn eid(l: u16, e: u16) -> ExpertId {
        ExpertId { layer: PhysicalLayerId(l), expert: e }
    }

    /// Review bug 2: a sync rescue must never evict another expert of the same layer's top-k.
    #[test]
    fn test_sync_rescue_never_evicts_current_topk() {
        let model_cfg = Inkling2Config::default();
        let mut tide_cfg = TideConfig::default();
        tide_cfg.rescue_policy = RescuePolicy::Exact;
        let mut sched = TideScheduler::new(model_cfg, tide_cfg, 16 * GIB, 16 * GIB);
        // Only 2 MXFP8 VRAM slots: layer 20 experts 0 and 1 end up resident.
        sched.cache.configure_scratch_ring(2560, 2, 8);
        let _ = sched.plan_pre_prefill(None);
        sched.cache.publish_all_pending();
        assert!(sched.cache.is_vram_hit(eid(20, 0)) && sched.cache.is_vram_hit(eid(20, 1)));

        // Expert 0 is the coldest resident, expert 1 is warm, and the missing expert 5 is hot.
        sched.cache.touch_expert(eid(20, 1), 5.0);
        sched.cache.touch_expert(eid(20, 5), 10.0);
        // top-k = [5 (miss), 0 (hit)]: the rescue for 5 must evict 1, not the in-use expert 0.
        let plan = sched.plan_layer_dispatch(PhysicalLayerId(20), &[5, 0], &[0.6, 0.4]);
        assert_eq!(plan.sync_transfers.len(), 1);
        assert!(
            matches!(plan.sync_transfers[0], TransferOp::RamToVram { evicted_from_vram: Some(v), .. } if v == eid(20, 1)),
            "rescue evicted the wrong expert: {:?}",
            plan.sync_transfers[0]
        );
        assert!(sched.cache.is_vram_hit(eid(20, 0)), "current top-k expert was evicted");
        sched.cache.publish_all_pending();
        assert!(sched.cache.is_vram_hit(eid(20, 5)));
    }

    /// Review bug 1 (CPU side): an in-flight promotion is not a hit and never shows in the table.
    #[test]
    fn test_pending_promotion_is_not_published() {
        let model_cfg = Inkling2Config::default();
        let tide_cfg = TideConfig::default();
        let mut sched = TideScheduler::new(model_cfg, tide_cfg, 16 * GIB, 16 * GIB);
        sched.cache.configure_scratch_ring(2560, 512, 8);
        let id = eid(21, 3);
        let _ = sched.cache.promote_to_ram(id);
        let op = sched.cache.promote_to_vram(id, false, false).expect("RAM-ready promotion");
        assert!(matches!(op, TransferOp::RamToVram { .. }));
        assert!(sched.cache.is_vram_pending(id) && !sched.cache.is_vram_hit(id));
        let flat = sched.cache.flat_index(id);
        assert_eq!(sched.cache.gpu_slot_table()[flat], SlotIndex::MISSING.0);
        // A layer that needs it must wait on the copy instead of reading a half-written slot.
        let plan = sched.plan_layer_dispatch(PhysicalLayerId(21), &[3], &[1.0]);
        assert!(plan.wait_pending && plan.sync_transfers.is_empty());
        let slot = sched.cache.entry(id).vram_slot;
        assert!(sched.cache.publish_vram(id, slot));
        assert_eq!(sched.cache.gpu_slot_table()[flat], slot.0);
    }

    /// Review bug 3: decode-time async paths never issue blocking Disk->VRAM reads.
    #[test]
    fn test_async_paths_never_block_on_disk() {
        let model_cfg = Inkling2Config::default();
        let mut tide_cfg = TideConfig::default();
        tide_cfg.preload_before_prefill = false;
        let mut sched = TideScheduler::new(model_cfg, tide_cfg, 16 * GIB, 16 * GIB);
        // Nothing is in RAM; a hot expert would previously be promoted via DiskToVramDirect.
        let hot = eid(22, 7);
        for _ in 0..10 {
            sched.cache.touch_expert(hot, 1.0);
        }
        let ops = sched.maybe_periodic_rebalance(0, 64, 8, true);
        assert!(ops.iter().all(|op| !matches!(op, TransferOp::DiskToVramDirect { .. })));
        assert!(ops.iter().any(|op| matches!(op, TransferOp::DiskToRam { expert, .. } if *expert == hot)));
        assert!(!sched.cache.is_ram_ready(hot), "read must be marked in flight");
        // Once the background read lands, Mode 3 promotes it RAM->VRAM.
        sched.cache.mark_ram_ready(hot);
        let ops2 = sched.plan_async_step();
        assert!(ops2.iter().any(|op| matches!(op, TransferOp::RamToVram { expert, .. } if *expert == hot)));
        assert!(ops2.iter().all(|op| !matches!(op, TransferOp::DiskToVramDirect { .. })));
    }

    /// An expert on the disk only is read back into RAM in the background (most valuable first),
    /// so a later rescue of it does not read the disk; never when RAM has no room for it.
    #[test]
    fn test_disk_only_experts_are_read_back_in_the_background() {
        let mut tide_cfg = TideConfig::default();
        tide_cfg.preload_before_prefill = false;
        let mut sched = TideScheduler::new(Inkling2Config::default(), tide_cfg.clone(), 16 * GIB, 16 * GIB);
        let wanted = eid(3, 9);
        for _ in 0..10 {
            sched.cache.touch_expert(wanted, 1.0);
        }
        let ops = sched.plan_async_step();
        assert!(matches!(ops.first(), Some(TransferOp::DiskToRam { expert, .. }) if *expert == wanted));
        assert!(ops.iter().all(|op| matches!(op, TransferOp::DiskToRam { .. })) && ops.len() <= 8);
        assert!(!sched.cache.is_ram_ready(wanted), "the read is in flight");
        // The same expert is not queued twice.
        let again = sched.plan_async_step();
        assert!(again.iter().all(|op| !matches!(op, TransferOp::DiskToRam { expert, .. } if *expert == wanted)));
        // No RAM at all: nothing is read.
        let mut none = TideScheduler::new(Inkling2Config::default(), tide_cfg, 16 * GIB, 0);
        assert!(none.plan_async_step().is_empty());
    }

    #[test]
    fn test_an_expert_read_from_the_disk_for_a_rescue_stays_in_ram() {
        let mut tide_cfg = TideConfig::default();
        tide_cfg.preload_before_prefill = false;
        let mut sched = TideScheduler::new(Inkling2Config::default(), tide_cfg.clone(), 16 * GIB, 16 * GIB);
        // On the disk only: it gets a RAM slot that is ready at once (the caller has the bytes).
        let a = eid(5, 17);
        assert!(sched.cache.entry(a).ram_slot.is_missing());
        let slot = sched.adopt_disk_read_into_ram(a).expect("free RAM");
        assert_eq!(sched.cache.entry(a).ram_slot, slot);
        assert!(sched.cache.is_ram_ready(a));
        // A background read of it is on its way: the rescue's read completes that slot.
        let b = eid(3, 9);
        for _ in 0..10 {
            sched.cache.touch_expert(b, 1.0);
        }
        let ops = sched.plan_async_step();
        let queued = ops.iter().find_map(|op| match op {
            TransferOp::DiskToRam { expert, ram_slot, .. } if *expert == b => Some(*ram_slot),
            _ => None,
        });
        let queued = queued.expect("background read queued");
        assert!(!sched.cache.is_ram_ready(b));
        assert_eq!(sched.adopt_disk_read_into_ram(b), Some(queued));
        assert!(sched.cache.is_ram_ready(b));
        // RAM holds one off-card expert and nothing else fits: its copy is not taken.
        let one = sched.cache.expert_bytes(a);
        let mut full = TideScheduler::new(Inkling2Config::default(), tide_cfg.clone(), 16 * GIB, one);
        assert!(full.adopt_disk_read_into_ram(a).is_some());
        assert!(full.adopt_disk_read_into_ram(eid(5, 18)).is_none());
        assert!(full.cache.is_ram_ready(a));
        // No RAM tier: nothing to keep.
        let mut none = TideScheduler::new(Inkling2Config::default(), tide_cfg, 16 * GIB, 0);
        assert!(none.adopt_disk_read_into_ram(a).is_none());
    }

    #[test]
    fn test_verify_layer_dispatch_protects_all_rows_and_ranks_shared_misses() {
        let model_cfg = Inkling2Config::default();
        let mut tide_cfg = TideConfig::default();
        tide_cfg.rescue_policy = RescuePolicy::Threshold(0.2);
        let mut sched = TideScheduler::new(model_cfg, tide_cfg, 16 * GIB, 16 * GIB);
        // 3 MXFP8 VRAM slots: layer 20 experts 0, 1, 2 start resident.
        sched.cache.configure_scratch_ring(2560, 3, 8);
        let _ = sched.plan_pre_prefill(None);
        sched.cache.publish_all_pending();
        sched.begin_token();
        sched.rescue_budget = 4;

        // 4 verify rows:
        // - expert 0 is hit in row 0 (must stay protected)
        // - expert 10 is missing in ALL 4 rows with share 0.25 each (sum = 1.0, max = 0.25 >= 0.2)
        // - expert 11 is missing in 1 row with share 0.35 (sum = 0.35, max = 0.35 >= 0.2)
        // - expert 12 is missing in 1 row with share 0.22 (sum = 0.22, max = 0.22 >= 0.2)
        // - expert 13 is missing in 4 rows with share 0.10 each (max = 0.10 < 0.2 -> skipped)
        let rows_ids = [
            [0, 10, 11, 13, 1, 2, 1, 2],
            [0, 10, 12, 13, 1, 2, 1, 2],
            [0, 10, 1,  13, 1, 2, 1, 2],
            [0, 10, 1,  13, 1, 2, 1, 2],
        ];
        let rows_ws = [
            [0.20, 0.25, 0.35, 0.10, 0.025, 0.025, 0.025, 0.025],
            [0.20, 0.25, 0.22, 0.10, 0.0575, 0.0575, 0.0575, 0.0575],
            [0.30, 0.25, 0.25, 0.10, 0.025, 0.025, 0.025, 0.025],
            [0.30, 0.25, 0.25, 0.10, 0.025, 0.025, 0.025, 0.025],
        ];
        let (ops, wait_pending, rescued) =
            sched.plan_verify_layer_dispatch(PhysicalLayerId(20), &rows_ids, &rows_ws);
        assert!(!wait_pending);
        // The step knows what it rescued (the expert map does not show those picks as missed)...
        assert!(sched.was_rescued_this_step(eid(20, 10)) && sched.was_rescued_this_step(eid(20, 11)));
        assert!(!sched.was_rescued_this_step(eid(20, 12)) && !sched.was_rescued_this_step(eid(20, 13)));
        // For m >= 4, at most 2 rescues per layer: must pick expert 10 (sum 1.0) and expert 11 (sum 0.35).
        assert_eq!(ops.len(), 2);
        assert_eq!(rescued, vec![eid(20, 10), eid(20, 11)]);
        // Protected residents 0, 1, 2 must remain in VRAM!
        assert!(sched.cache.is_vram_hit(eid(20, 0)));
        assert!(sched.cache.is_vram_hit(eid(20, 1)));
        assert!(sched.cache.is_vram_hit(eid(20, 2)));
        // ...and the next step starts with none.
        sched.begin_token();
        assert!(!sched.was_rescued_this_step(eid(20, 10)));
        // With the auto gate the budget (4) is the only limit: a layer with nothing resident gets
        // 4 rescues, not 2.
        sched.set_auto_gate(AutoGate { enabled: true, ..AutoGate::default() });
        sched.begin_token();
        let (_, _, rescued) = sched.plan_verify_layer_dispatch(PhysicalLayerId(21), &rows_ids, &rows_ws);
        assert_eq!(rescued.len(), 4);
        assert!(rescued.contains(&eid(21, 10)));
    }

    #[test]
    fn test_scratch_rescue_never_unloads_protected_in_use_expert() {
        let model_cfg = Inkling2Config::default();
        let mut tide_cfg = TideConfig::default();
        tide_cfg.rescue_policy = RescuePolicy::Threshold(0.10);
        let mut sched = TideScheduler::new(model_cfg, tide_cfg, 16 * GIB, 16 * GIB);
        // Tiny 2-slot MXFP8 scratch ring + 1 resident slot (expert 0).
        sched.cache.configure_scratch_rings(16, 1, 2, 2);
        let _ = sched.plan_pre_prefill(None);
        sched.cache.publish_all_pending();
        // Pre-populate both scratch slots with experts 20:10 and 20:11 (still in VRAM).
        assert!(sched.cache.stage_scratch_rescue(eid(20, 10)).is_some());
        assert!(sched.cache.stage_scratch_rescue(eid(20, 11)).is_some());
        sched.cache.publish_all_pending();
        sched.cache.clear_protected();

        sched.set_auto_gate(AutoGate { enabled: true, ..AutoGate::default() });
        sched.rescue_zone_budgets = Some([8, 8, 8]);
        sched.begin_verify_step_kind(2, true);
        // Row 0 (lane 0) routes to scratch experts 10 & 11 (in VRAM); Row 1 (lane 1) misses 12 & 13.
        // Because BOTH scratch slots are in use by Row 0, rescuing 12 or 13 MUST NOT unload 10 or 11!
        let rows_ids = [
            [10, 11, 0, 0, 0, 0, 0, 0],
            [12, 13, 0, 0, 0, 0, 0, 0],
        ];
        let rows_ws = [
            [0.40, 0.40, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025],
            [0.40, 0.40, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025],
        ];
        let (ops, _, rescued) = sched.plan_verify_layer_dispatch(PhysicalLayerId(20), &rows_ids, &rows_ws);
        assert!(ops.is_empty(), "must not overwrite protected scratch slots: {ops:?}");
        assert!(rescued.is_empty());
        assert!(sched.cache.is_vram_hit(eid(20, 10)), "in-use scratch expert 10 was unloaded!");
        assert!(sched.cache.is_vram_hit(eid(20, 11)), "in-use scratch expert 11 was unloaded!");
    }

    #[test]
    fn test_eviction_prefers_inactive_expert_over_active_lane_expert() {
        let model_cfg = Inkling2Config::default();
        let tide_cfg = TideConfig::default();
        let mut sched = TideScheduler::new(model_cfg, tide_cfg, 16 * GIB, 16 * GIB);
        // 2 MXFP8 resident slots: experts 20:0 and 20:1.
        sched.cache.configure_scratch_rings(16, 2, 4, 4);
        sched.cache.promote_to_vram(eid(20, 0), false, true);
        sched.cache.promote_to_vram(eid(20, 1), false, true);
        sched.cache.publish_all_pending();

        // Give expert 20:1 slightly higher heat from a finished lane 1, while active lane 0 uses 20:0.
        sched.cache.set_current_lane(1);
        sched.cache.set_lane_active(1, false);
        sched.cache.touch_expert(eid(20, 1), 2.0);
        sched.cache.set_current_lane(0);
        sched.cache.set_lane_active(0, true);
        sched.cache.touch_expert(eid(20, 0), 0.5);

        // Promoting 20:2 must evict inactive expert 20:1, keeping active lane 0's expert 20:0 in VRAM!
        let op = sched.cache.promote_to_vram(eid(20, 2), false, true).expect("promote 20:2");
        assert!(
            matches!(op, TransferOp::DiskToVramDirect { evicted_from_vram: Some(v), .. } | TransferOp::RamToVram { evicted_from_vram: Some(v), .. } if v == eid(20, 1)),
            "expected inactive expert 20:1 to be evicted before active-lane expert 20:0, got {op:?}"
        );
        assert!(sched.cache.is_vram_hit(eid(20, 0)), "active-lane expert 20:0 must remain resident!");
    }

    #[test]
    fn test_rescue_snapshot_tracks_per_zone_gate_and_budget_balance() {
        let model_cfg = Inkling2Config::default();
        let mut tide_cfg = TideConfig::default();
        tide_cfg.rescue_policy = RescuePolicy::Threshold(0.12);
        let mut sched = TideScheduler::new(model_cfg, tide_cfg, 16 * GIB, 16 * GIB);
        // Leave misses across Prelude (layer 0), Loop Core (layer 6), and Coda (layer 20).
        sched.cache.configure_scratch_ring(16, 4, 8);
        let _ = sched.plan_pre_prefill(None);
        sched.cache.publish_all_pending();
        sched.rescue_zone_budgets = Some([2, 4, 1]);
        sched.loop_core_gate = Some(0.05);

        // Step 1:
        // - Prelude (cap 2): 1 rescue (share 0.20), 1 below-gate miss (share 0.08) -> balanced (1/2)
        // - Loop Core (cap 4): 0 rescues, 2 below-gate misses (0.03, 0.04) -> underused
        // - Coda (cap 1): 1 rescue (share 0.25) hits cap 1/1 + 1 blocked miss (0.18) + 1 capped layer -> saturated
        sched.begin_token();
        let _ = sched.plan_layer_dispatch(PhysicalLayerId(0), &[120, 121], &[0.20, 0.08]); // norm: 0.714, 0.286 (wait: let's include a resident hit so norm matches!)
        // Wait: use exact sum=1.0 weights so normalized shares are exact!
        sched.begin_token();
        let _ = sched.plan_layer_dispatch(PhysicalLayerId(0), &[0, 120, 121], &[0.72, 0.20, 0.08]);
        let _ = sched.plan_layer_dispatch(PhysicalLayerId(6), &[0, 120, 121], &[0.93, 0.04, 0.03]);
        let _ = sched.plan_layer_dispatch(PhysicalLayerId(20), &[0, 120, 121], &[0.57, 0.25, 0.18]);
        sched.note_zone_budget_capped_layer(PhysicalLayerId(21));

        // Commit by starting a fresh inactive step (snapshot should see step 2 + step 1!)
        let snap = sched.rescue_snapshot();
        assert_eq!(snap.policy, "threshold");
        assert_eq!(snap.base_budget_total, 7);
        assert_eq!(snap.zones[0].name, "Prelude");
        assert!((snap.zones[0].gate - 0.12).abs() < 1e-6);
        assert_eq!(snap.zones[0].base_budget, 2);
        assert_eq!(snap.zones[1].name, "Loop Core");
        assert!((snap.zones[1].gate - 0.05).abs() < 1e-6);
        assert_eq!(snap.zones[1].base_budget, 4);
        assert_eq!(snap.zones[2].name, "Coda");
        assert!((snap.zones[2].gate - 0.12).abs() < 1e-6);
        assert_eq!(snap.zones[2].base_budget, 1);

        // Coda hit 1/1 cap on the step with misses and recorded 1 blocked >= gate pick + 1 capped layer
        assert!(snap.zones[2].capped_pct >= 50.0);
        assert!(snap.zones[2].avg_capped_layers > 0.0);
        assert!(snap.zones[2].avg_blocked_picks > 0.0);
        assert_eq!(snap.zones[2].status, "saturated");
        // Histogram recorded the missed picks across the bins
        assert!(snap.zones[2].miss_hist.iter().sum::<u32>() >= 2);
        assert!(snap.zones[1].miss_hist.iter().sum::<u32>() >= 2);
        // Auto gate off: the gates are the configured ones.
        assert!(!snap.auto_gate);
        assert!((snap.zones[1].base_gate - 0.05).abs() < 1e-6);
    }

    /// Ends a decode step in which each block rescued `used[z]` experts.
    fn end_step_with(sched: &mut TideScheduler, used: [u16; 3]) {
        sched.begin_token();
        sched.rescue_tracker.step_active = true;
        for z in 0..3 {
            sched.rescue_tracker.cur.zones[z].rescued = used[z];
        }
    }

    #[test]
    fn test_auto_gate_steers_each_block_to_its_budget_share() {
        let mut tide_cfg = TideConfig::default();
        tide_cfg.rescue_policy = RescuePolicy::Threshold(0.08);
        let mut sched = TideScheduler::new(Inkling2Config::default(), tide_cfg, 16 * GIB, 16 * GIB);
        sched.rescue_zone_budgets = Some([10, 10, 10]);
        sched.loop_core_gate = Some(0.09);

        // Off: steps leave the gates alone.
        for _ in 0..50 {
            end_step_with(&mut sched, [0, 10, 8]);
        }
        sched.begin_token();
        assert_eq!(sched.effective_zone_gate(0), 0.08);
        assert_eq!(sched.effective_zone_gate(1), 0.09);

        // On: the gates start at the configured values...
        sched.set_auto_gate(AutoGate { enabled: true, ..AutoGate::default() });
        assert_eq!(sched.effective_zone_gate(0), 0.08);
        assert_eq!(sched.effective_zone_gate(1), 0.09);
        assert_eq!(sched.effective_zone_gate(2), 0.08);
        // ...a block under its share lowers its gate, one over it raises it, one on it stays.
        for _ in 0..20 {
            end_step_with(&mut sched, [0, 10, 8]);
        }
        sched.begin_token();
        let (pre, core, coda) = (sched.effective_zone_gate(0), sched.effective_zone_gate(1), sched.effective_zone_gate(2));
        assert!(pre < 0.04 && pre > 0.02, "prelude gate {pre}");
        assert!(core > 0.105 && core < 0.12, "core gate {core}");
        assert!((coda - 0.08).abs() < 1e-4, "coda gate {coda}");
        let snap = sched.rescue_snapshot();
        assert!(snap.auto_gate);
        assert!((snap.zones[0].gate - pre).abs() < 1e-6);
        assert!((snap.zones[0].base_gate - 0.08).abs() < 1e-6);
        assert_eq!(snap.zones[2].status, "balanced");
        assert_eq!(snap.zones[0].status, "underused");

        // The gates stay within their bounds.
        for _ in 0..2000 {
            end_step_with(&mut sched, [0, 10, 8]);
        }
        sched.begin_token();
        assert_eq!(sched.effective_zone_gate(0), 0.02);
        assert_eq!(sched.effective_zone_gate(1), 0.5);
        // Prefill steps do not steer.
        end_step_with(&mut sched, [0, 0, 0]);
        sched.mark_step_prefill();
        sched.begin_token();
        assert_eq!(sched.effective_zone_gate(1), 0.5);
        // A changed configured gate restarts the steering from it.
        sched.loop_core_gate = Some(0.2);
        sched.reset_auto_gates();
        assert_eq!(sched.effective_zone_gate(1), 0.2);
        // Exact and Disabled ignore it.
        sched.tide_cfg.rescue_policy = RescuePolicy::Exact;
        assert_eq!(sched.effective_zone_gate(0), 0.0);
    }

    #[test]
    fn test_must_rescue_bypasses_budget_and_auto_gate() {
        let model_cfg = Inkling2Config::default();
        let mut tide_cfg = TideConfig::default();
        // High gate (0.45) + tiny zone budget (0) — normal threshold would rescue nothing.
        tide_cfg.rescue_policy = RescuePolicy::Threshold(0.45);
        let mut sched = TideScheduler::new(model_cfg, tide_cfg, 64 << 20, 4usize << 30);
        sched.cache.configure_scratch_ring(16, 4, 8);
        sched.rescue_zone_budgets = Some([0, 0, 0]);
        sched.rescue_must_threshold = Some(0.25);
        sched.set_auto_gate(AutoGate { enabled: true, min_gate: 0.45, max_gate: 0.50, ..AutoGate::default() });
        sched.begin_token();

        // With MUST-rescue enabled, budget-spent fast-skip must stay false so layers still check.
        let layer = PhysicalLayerId(20); // Coda
        assert!(!sched.rescue_budget_spent_for_layer(layer, 1));
        assert!(!sched.rescue_budget_spent_for_layer(layer, 4));

        // Single-token decode: two dominant misses (0.35, 0.28) bypass budget=0 and gate=0.45,
        // while a 0.20 miss stays skipped.
        let ids = [1u16, 2, 3, 4, 5, 6, 7, 8];
        let ws = [0.35f32, 0.28, 0.20, 0.05, 0.04, 0.03, 0.03, 0.02];
        let plan = sched.plan_layer_dispatch(layer, &ids, &ws);
        let rescued_ids: Vec<u16> = plan.rescued_experts.iter().map(|r| r.expert.expert).collect();
        assert_eq!(rescued_ids, vec![1, 2]);

        // Speculative verify: dominant misses (0.30, 0.26) across rows still rescue when budget=0,
        // while 0.18 stays skipped.
        sched.begin_token();
        let layer2 = PhysicalLayerId(21);
        let rows_ids = [
            [10u16, 11, 12, 13, 14, 15, 16, 17],
            [10, 18, 12, 13, 14, 15, 16, 17],
        ];
        let rows_ws = [
            [0.30f32, 0.18, 0.10, 0.10, 0.10, 0.08, 0.07, 0.07],
            [0.26f32, 0.28, 0.10, 0.10, 0.08, 0.06, 0.06, 0.06],
        ];
        let (_ops, _wait, rescued) = sched.plan_verify_layer_dispatch(layer2, &rows_ids, &rows_ws);
        let mut r_exps: Vec<u16> = rescued.into_iter().map(|e| e.expert).collect();
        r_exps.sort_unstable();
        assert_eq!(r_exps, vec![10, 18]);
    }
}
