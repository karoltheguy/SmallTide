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
            for &(id, _) in &ranked {
                if self.cache.has_free_vram_room_for(id) {
                    if let Some(op) = self.cache.promote_to_vram(id, false, true) {
                        self.record_transfer_telemetry(&op, false);
                        ops.push(op);
                    }
                } else if self.cache.entry(id).vram_slot.is_missing()
                    && self.cache.ram_used_bytes() + self.cache.expert_bytes(id) <= self.cache.ram_budget_bytes()
                {
                    if let Some(op) = self.cache.promote_to_ram(id) {
                        self.record_transfer_telemetry(&op, false);
                        ops.push(op);
                    }
                }
            }
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
            } else if self.cache.ram_used_bytes() + self.cache.expert_bytes(id)
                <= self.cache.ram_budget_bytes()
            {
                if let Some(op) = self.cache.promote_to_ram(id) {
                    self.record_transfer_telemetry(&op, false);
                    ops.push(op);
                }
            }
        }

        ops
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
        if !self.tide_cfg.preload_prefill_decode_boundary {
            self.prefill_scores.fill(0.0);
            return Vec::new();
        }
        // Candidates: experts the prompt routed to that are not resident, best value first.
        let mut ranked: Vec<(ExpertId, f64)> = self
            .prefill_scores
            .iter()
            .enumerate()
            .filter(|(_, &s)| s > 0.0)
            .map(|(idx, _)| self.cache.expert_from_flat(idx))
            .filter(|&id| self.cache.entry(id).vram_slot.is_missing())
            .map(|id| (id, self.cache.value(id)))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut ops = Vec::new();
        let mut swaps = 0usize;
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
            floors[ci] = None;
            let pin = !self.cache.has_priors() && self.model_cfg.is_loop_core_layer(id.layer.0 as usize);
            if let Some(op) = self.cache.promote_to_vram(id, pin, true) {
                self.record_transfer_telemetry(&op, false);
                ops.push(op);
                swaps += 1;
            }
        }

        self.prefill_scores.fill(0.0);
        ops
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

        for (&exp, &w) in topk_ids.iter().zip(topk_weights.iter()) {
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
                RescuePolicy::Threshold(tau) => norm_w >= tau,
                RescuePolicy::Disabled => false,
            };

            if should_sync_rescue {
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

    /// Bookkeeping for one layer of one row of a batched decode round. Batched rounds never rescue
    /// (that would stall every lane): the kernels skipped unpublished experts, so this counts hits
    /// and misses, adds heat and skipped routing weight, and queues misses for async promotion.
    pub fn record_batch_dispatch(&mut self, layer: PhysicalLayerId, topk_ids: &[u16], topk_weights: &[f32]) {
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
    pub fn maybe_periodic_rebalance(
        &mut self,
        decoded_tokens: u64,
        interval_tokens: u64,
        max_swaps: usize,
        force: bool,
    ) -> Vec<TransferOp> {
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
        sched.cache.publish_all_pending();
        assert!(!rebal.is_empty());
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
}
