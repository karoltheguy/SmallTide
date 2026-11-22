//! 4-Mode TideLM Expert Scheduler:
//! 1. Pre-prefill warm-up (Loop-Core priority + optional priors)
//! 2. Prefill -> Decode boundary tide rebalance
//! 3. Paced async background promotion (Stage 1 RAM->VRAM + Stage 2 Disk->RAM)
//! 4. In-layer dispatch + configurable Sync Rescue (Exact / Threshold / Disabled)

use crate::cache::{TideCache, TransferOp};
use crate::priors::TidePriors;
use smalltide_core::{
    ExpertId, Inkling2Config, PhysicalLayerId, RescuePolicy, SlotIndex, TideConfig, TideTelemetry,
};
use std::collections::HashMap;
use std::path::PathBuf;

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
    rescues_this_token: usize,
    rescues_by_zone: [usize; 3],
    /// Residency events for the expert map (host bookkeeping; decides nothing).
    pub events: crate::expert_map::EventLog,
    /// Kind / routing share the next logged transfers get (set by each planner; 0 = not logged).
    event_kind: u8,
    event_share: f32,
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
            async_hysteresis: 2.0,
            rescue_budget: 4,
            rescue_zone_budgets: None,
            loop_core_gate: None,
            rescues_this_token: 0,
            rescues_by_zone: [0; 3],
            events: crate::expert_map::EventLog::default(),
            event_kind: 0,
            event_share: 0.0,
        }
    }

    /// Installs routing priors; non-empty priors immediately drive values and warm-up.
    pub fn install_priors(&mut self, store: PriorsStore) {
        if !store.priors.is_empty() {
            self.cache.set_prior_rates(store.priors.rates(&self.model_cfg), store.prior_weight);
        }
        self.priors_store = Some(store);
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

    /// Startup RAM warm-up in `order` (most valuable first): experts that are NOT in VRAM first
    /// (a miss on them would otherwise read the disk), then RAM copies of VRAM-resident experts
    /// with the budget left (cheap fallback once they are evicted from VRAM).
    fn fill_ram(&mut self, order: &[ExpertId], ops: &mut Vec<TransferOp>) {
        for want_resident in [false, true] {
            for &id in order {
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
                if val <= f * crate::cache::SWAP_HYSTERESIS + 1e-9 {
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
                ops.push(op);
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
        self.rescues_this_token = 0;
        self.rescues_by_zone = [0; 3];
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
    /// single-token decode, 2..=8 for speculative verify).
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
        [
            if base[0] > 0 { base[0] + verify_rows / 4 } else { 0 },
            if base[1] > 0 { base[1] + verify_rows / 4 } else { 0 },
            if base[2] > 0 { base[2] + verify_rows / 2 } else { 0 },
        ]
    }

    /// `RescuePolicy::Threshold`: this layer's zone has used up its rescue budget for the current
    /// token / verify step. Other zones (e.g. Coda) keep their own independent budget.
    pub fn rescue_budget_spent_for_layer(&self, layer: PhysicalLayerId, verify_rows: usize) -> bool {
        if !matches!(self.tide_cfg.rescue_policy, RescuePolicy::Threshold(_)) {
            return false;
        }
        let z = self.zone_of(layer);
        self.rescues_by_zone[z] >= self.effective_zone_budgets(verify_rows)[z]
    }

    /// True only once all 3 zones have exhausted their rescue budgets.
    pub fn rescue_budget_spent(&self) -> bool {
        if !matches!(self.tide_cfg.rescue_policy, RescuePolicy::Threshold(_)) {
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

        let total_weight: f32 = topk_weights.iter().sum();
        let inv_total_w = if total_weight > 1e-8 { 1.0 / total_weight } else { 1.0 };
        let mut executed_weight = 0.0f32;
        self.telemetry.routed_weight_total += 1.0;
        // Threshold (TideLM's inline rescue): only the heaviest missed pick at or above the gate,
        // subject to this layer's zone rescue budget (Prelude / Loop Core / Coda).
        let zone = self.zone_of(layer);
        let zone_cap = self.effective_zone_budgets(1)[zone];
        let threshold_pick: Option<usize> = match self.tide_cfg.rescue_policy {
            RescuePolicy::Threshold(gate) if self.rescues_by_zone[zone] < zone_cap => {
                let tau = match self.loop_core_gate {
                    Some(g) if self.model_cfg.is_loop_core_layer(layer.0 as usize) => g,
                    _ => gate,
                };
                topk_ids
                .iter()
                .zip(topk_weights.iter())
                .enumerate()
                .filter(|(_, (&e, &w))| {
                    let id = ExpertId { layer, expert: e };
                    w * inv_total_w >= tau && self.cache.entry(id).vram_slot.is_missing()
                })
                .max_by(|a, b| a.1 .1.total_cmp(b.1 .1))
                .map(|(i, _)| i)
            }
            _ => None,
        };
        if threshold_pick.is_some() {
            self.rescues_this_token += 1;
            self.rescues_by_zone[zone] += 1;
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
            let should_sync_rescue = match self.tide_cfg.rescue_policy {
                RescuePolicy::Exact => true,
                RescuePolicy::Threshold(_) => threshold_pick == Some(k),
                RescuePolicy::Disabled => false,
            };

            if should_sync_rescue {
                self.event_share = norm_w;
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
                    } else {
                        self.telemetry.stage1_skipped_tail_misses += 1;
                        self.telemetry.routed_weight_skipped += norm_w as f64;
                    }
                } else {
                    let op = self.cache.stage_scratch_rescue(id);
                    self.record_transfer_telemetry(&op, true);
                    let scratch_slot = self.cache.entry(id).vram_slot;
                    sync_transfers.push(op);
                    rescued_experts.push(ActiveExpertSlot {
                        expert: id,
                        vram_slot: scratch_slot,
                        weight: w,
                    });
                    executed_weight += w;
                }
            } else {
                // Skip this miss for zero-stall decode and enqueue for paced async promotion!
                self.telemetry.stage1_skipped_tail_misses += 1;
                self.telemetry.routed_weight_skipped += norm_w as f64;
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

        for &id in &protect {
            let entry = self.cache.entry(id);
            if self.cache.is_vram_pending(id) && !entry.is_scratch_slot {
                wait_pending = true;
            }
        }

        if let RescuePolicy::Threshold(gate) = self.tide_cfg.rescue_policy {
            let zone = self.zone_of(layer);
            let zone_cap = self.effective_zone_budgets(m)[zone];
            if self.rescues_by_zone[zone] < zone_cap {
                let tau = match self.loop_core_gate {
                    Some(g) if self.model_cfg.is_loop_core_layer(layer.0 as usize) => g,
                    _ => gate,
                };
                // Aggregate (max_single_row_share, sum_share_across_rows) per unique missing expert.
                let mut miss_stats: Vec<(ExpertId, f32, f32)> = Vec::new();
                for (r_ids, r_ws) in rows_ids.iter().zip(rows_ws.iter()) {
                    let tot: f32 = r_ws.iter().sum();
                    let inv = if tot > 1e-8 { 1.0 / tot } else { 1.0 };
                    for (&e, &w) in r_ids.iter().zip(r_ws.iter()) {
                        let id = ExpertId { layer, expert: e };
                        if !self.cache.entry(id).vram_slot.is_missing() {
                            continue;
                        }
                        let nw = w * inv;
                        if let Some(st) = miss_stats.iter_mut().find(|s| s.0 == id) {
                            st.1 = st.1.max(nw);
                            st.2 += nw;
                        } else {
                            miss_stats.push((id, nw, nw));
                        }
                    }
                }
                miss_stats.retain(|s| s.1 >= tau);
                miss_stats.sort_by(|a, b| b.2.total_cmp(&a.2));
                // 1/layer in Prelude/Core so budget spreads across layers; up to 2/layer in Coda.
                let max_per_layer = if zone == 2 && m >= 4 { 2 } else { 1 };
                let rem_budget = zone_cap.saturating_sub(self.rescues_by_zone[zone]);
                let take = max_per_layer.min(rem_budget);
                for (id, max_share, _sum_share) in miss_stats.into_iter().take(take) {
                    self.rescues_this_token += 1;
                    self.rescues_by_zone[zone] += 1;
                    self.event_share = max_share;
                    if self.cache.should_promote_instead_of_scratch(id) {
                        let pin = !self.cache.has_priors() && self.model_cfg.is_loop_core_layer(layer.0 as usize);
                        if let Some(op) = self.cache.promote_to_vram(id, pin, true) {
                            self.record_transfer_telemetry(&op, true);
                            sync_transfers.push(op);
                            rescued.push(id);
                        }
                    } else {
                        let op = self.cache.stage_scratch_rescue(id);
                        self.record_transfer_telemetry(&op, true);
                        sync_transfers.push(op);
                        rescued.push(id);
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
        self.telemetry.routed_weight_total += 1.0;
        for (&exp, &w) in topk_ids.iter().zip(topk_weights.iter()) {
            let id = ExpertId { layer, expert: exp };
            let norm_w = w * inv;
            self.cache.touch_expert(id, norm_w);
            if self.cache.is_vram_hit(id) {
                self.record_hit(layer);
                continue;
            }
            if rescued.contains(&id) {
                self.record_miss(layer);
                continue;
            }
            self.record_miss(layer);
            self.telemetry.stage1_skipped_tail_misses += 1;
            self.telemetry.routed_weight_skipped += norm_w as f64;
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

        // Stage 1: RAM -> VRAM (paced, only promoting experts whose RAM slot is Ready!).
        // Candidates in value order; each class's eviction floor is computed once and refreshed
        // only after a promotion (a full scan per candidate cost ~10 ms of CPU per token).
        if !self.pending_async_vram.is_empty()
            && self.tide_cfg.max_async_vram_promotions_per_token > 0
        {
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
                    ops.push(op);
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
        if !force && (interval_tokens == 0 || decoded_tokens == 0 || decoded_tokens % interval_tokens != 0) {
            return Vec::new();
        }
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
            if ops.len() >= max_swaps {
                break;
            }
            let cls = self.cache.expert_class(id);
            let ci = cls as usize;
            let need = self.cache.expert_bytes(id);
            let floor = *floors[ci].get_or_insert_with(|| self.cache.promotion_floor(cls, need));
            if let Some(f) = floor {
                if val <= f * crate::cache::SWAP_HYSTERESIS + 1e-9 {
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
                ops.push(op);
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
        // For m >= 4, at most 2 rescues per layer: must pick expert 10 (sum 1.0) and expert 11 (sum 0.35).
        assert_eq!(ops.len(), 2);
        assert_eq!(rescued, vec![eid(20, 10), eid(20, 11)]);
        // Protected residents 0, 1, 2 must remain in VRAM!
        assert!(sched.cache.is_vram_hit(eid(20, 0)));
        assert!(sched.cache.is_vram_hit(eid(20, 1)));
        assert!(sched.cache.is_vram_hit(eid(20, 2)));
    }
}
