//! 3-Tier Expert Cache (Tier 0 VRAM <-> Tier 1 Host RAM <-> Tier 2 NVMe Disk)
//! with Elastic KV <-> Expert VRAM slab sharing and Exclusive Loop-Core Pinning.

use smalltide_core::{ExpertId, Inkling2Config, PhysicalLayerId, SlotIndex, TideConfig};
use std::collections::HashSet;

/// Current residency state of a single routed expert across the 3 memory tiers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertResidency {
    pub vram_slot: SlotIndex,
    pub ram_slot: SlotIndex,
    /// True while a background Stage 2 NVMe read into `ram_slot` is still in flight.
    pub ram_in_flight: bool,
    /// Permanently pinned in VRAM (e.g., Loop Core 6..17 when VRAM budget allows).
    /// When `exclusive_loop_pinning` is enabled, pinned VRAM experts do not occupy RAM slots.
    pub vram_pinned: bool,
    /// True if `vram_slot` currently points to a transient Scratch Rescue slot (not counted in resident bytes).
    pub is_scratch_slot: bool,
    /// True while the DMA into `vram_slot` may still be in flight: the slot is allocated but NOT yet
    /// published in the GPU indirection table, so the CPU must not count it as a hit and kernels
    /// never see it. Cleared by `publish_vram` once the executor knows the copy has landed.
    pub vram_pending: bool,
    /// Monotonic step when last activated.
    pub last_touch_step: u64,
    /// Loop-weighted EWMA activation heat score.
    pub heat: f32,
}

impl Default for ExpertResidency {
    fn default() -> Self {
        Self {
            vram_slot: SlotIndex::MISSING,
            ram_slot: SlotIndex::MISSING,
            ram_in_flight: false,
            vram_pinned: false,
            is_scratch_slot: false,
            vram_pending: false,
            last_touch_step: 0,
            heat: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertClass {
    Nvfp4,
    Mxfp8,
}

/// Transfer command emitted by the cache/scheduler to be executed by the I/O or GPU backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferOp {
    /// Stage 2: Read expert from Tier 2 NVMe Disk -> Tier 1 Host RAM slot.
    DiskToRam {
        expert: ExpertId,
        ram_slot: SlotIndex,
        evicted_from_ram: Option<ExpertId>,
    },
    /// Stage 1: DMA expert from Tier 1 Host RAM slot -> Tier 0 VRAM slot.
    RamToVram {
        expert: ExpertId,
        src_ram_slot: SlotIndex,
        dst_vram_slot: SlotIndex,
        evicted_from_vram: Option<ExpertId>,
        /// If true, caller may free `src_ram_slot` after DMA completes (Exclusive Loop Pinning).
        release_ram_after_upload: bool,
    },
    /// Cold-start / Rescue direct pipeline: Disk -> Staging -> VRAM (used when missing in both RAM & VRAM).
    DiskToVramDirect {
        expert: ExpertId,
        dst_vram_slot: SlotIndex,
        also_keep_ram_slot: Option<SlotIndex>,
        evicted_from_vram: Option<ExpertId>,
    },
    /// Transient Scratch Rescue Ring transfer: stages a cold 1-off sync miss into a scratch VRAM
    /// slot at the tail of the pool WITHOUT evicting any warm resident expert.
    ScratchRescue {
        expert: ExpertId,
        src_ram_slot: Option<SlotIndex>,
        scratch_vram_slot: SlotIndex,
    },
}

/// Manages Tier 0 (VRAM) and Tier 1 (Host RAM) slot tables with byte-exact budgeting
/// and Elastic KV <-> Expert Pool sharing.
pub struct TideCache {
    model_cfg: Inkling2Config,
    tide_cfg: TideConfig,

    /// Total unified VRAM budget shared between Paged KV Cache and Expert Pool.
    unified_vram_budget_bytes: usize,
    /// Bytes currently claimed by KV Cache (grows toward 327K+ context).
    kv_reserved_bytes: usize,
    /// Total Host RAM budget available for Tier 1 warm experts.
    ram_budget_bytes: usize,

    vram_used_bytes: usize,
    ram_used_bytes: usize,

    next_vram_slot_nvfp4: u32,
    next_vram_slot_mxfp8: u32,
    next_ram_slot_nvfp4: u32,
    next_ram_slot_mxfp8: u32,
    free_vram_slots_nvfp4: Vec<SlotIndex>,
    free_vram_slots_mxfp8: Vec<SlotIndex>,
    free_ram_slots_nvfp4: Vec<SlotIndex>,
    free_ram_slots_mxfp8: Vec<SlotIndex>,

    /// Flat table indexed by `layer * n_routed_experts + expert`.
    entries: Vec<ExpertResidency>,
    /// Mirror of the *published* GPU-side indirection table (`u32::MAX` = MISSING).
    /// An expert appears here only once its slab is fully resident (never while `vram_pending`).
    gpu_slot_table: Vec<u32>,
    /// Indices into `gpu_slot_table` modified since the last GPU sync.
    dirty_gpu_indices: HashSet<usize>,
    /// Flat indices that must not be evicted right now (e.g. the current layer's top-k during a
    /// sync rescue), so a rescue can never evict an expert the very same kernel is about to read.
    protected: Vec<usize>,

    /// Transient Scratch Rescue Ring state (base slot index & ring cursor per class).
    scratch_base_nvfp4: u32,
    scratch_base_mxfp8: u32,
    /// Hard physical slot capacities of the allocated NVFP4 and MXFP8 VRAM pools (`None` = byte-only budget).
    max_vram_slots_nvfp4: Option<usize>,
    max_vram_slots_mxfp8: Option<usize>,
    /// When set, a disk -> VRAM load does not also keep a RAM copy (startup placement: RAM is
    /// filled afterwards, experts that are not in VRAM first).
    pub no_ram_copy_on_direct: bool,
    used_vram_slots_nvfp4: usize,
    used_vram_slots_mxfp8: usize,
    scratch_ring_size: u32,
    scratch_cursor_nvfp4: u32,
    scratch_cursor_mxfp8: u32,
    scratch_occupants_nvfp4: Vec<Option<ExpertId>>,
    scratch_occupants_mxfp8: Vec<Option<ExpertId>>,

    step: u64,

    /// Raw routing demand of the current request (sum of normalized router shares per expert,
    /// one touch per logical pass), folded into `TidePriors` at request end.
    demand: Vec<f32>,
    /// Prior expected routing mass per token (`TidePriors::rates`), zero without priors.
    prior_rate: Vec<f32>,
    /// Weight of the prior in an expert's value, in steady-state heat units (see `value`).
    prior_heat_equiv: f32,
    /// Per-lane demand of the request in flight on that lane (conversation memory for
    /// prompt-aware loading); `current_lane` receives the touches.
    lane_demand: Vec<Vec<f32>>,
    current_lane: usize,
}

/// Per-token heat decay (half-life ~46 tokens).
pub const HEAT_DECAY: f32 = 0.985;
/// Hysteresis: a candidate must beat the coldest resident of its class by this factor.
pub const SWAP_HYSTERESIS: f64 = 1.25;

impl TideCache {
    pub fn new(
        model_cfg: Inkling2Config,
        tide_cfg: TideConfig,
        unified_vram_budget_bytes: usize,
        ram_budget_bytes: usize,
    ) -> Self {
        let total = model_cfg.num_physical_layers * model_cfg.n_routed_experts;
        Self {
            model_cfg,
            tide_cfg,
            unified_vram_budget_bytes,
            kv_reserved_bytes: 0,
            ram_budget_bytes,
            vram_used_bytes: 0,
            ram_used_bytes: 0,
            next_vram_slot_nvfp4: 0,
            next_vram_slot_mxfp8: 0,
            next_ram_slot_nvfp4: 0,
            next_ram_slot_mxfp8: 0,
            free_vram_slots_nvfp4: Vec::new(),
            free_vram_slots_mxfp8: Vec::new(),
            free_ram_slots_nvfp4: Vec::new(),
            free_ram_slots_mxfp8: Vec::new(),
            entries: vec![ExpertResidency::default(); total],
            gpu_slot_table: vec![SlotIndex::MISSING.0; total],
            dirty_gpu_indices: HashSet::new(),
            protected: Vec::new(),
            scratch_base_nvfp4: 4096,
            scratch_base_mxfp8: 4096,
            max_vram_slots_nvfp4: None,
            max_vram_slots_mxfp8: None,
            no_ram_copy_on_direct: false,
            used_vram_slots_nvfp4: 0,
            used_vram_slots_mxfp8: 0,
            scratch_ring_size: 8,
            scratch_cursor_nvfp4: 0,
            scratch_cursor_mxfp8: 0,
            scratch_occupants_nvfp4: vec![None; 8],
            scratch_occupants_mxfp8: vec![None; 8],
            step: 0,
            demand: vec![0.0; total],
            prior_rate: vec![0.0; total],
            prior_heat_equiv: 0.0,
            lane_demand: vec![vec![0.0; total]],
            current_lane: 0,
        }
    }

    /// Routing observations go to `lane`'s demand from now on.
    pub fn set_current_lane(&mut self, lane: usize) {
        let total = self.entries.len();
        while self.lane_demand.len() <= lane {
            self.lane_demand.push(vec![0.0; total]);
        }
        self.current_lane = lane;
    }

    /// `lane`'s demand since the last call (and resets it).
    pub fn take_lane_demand(&mut self, lane: usize) -> Vec<f32> {
        let total = self.entries.len();
        if lane >= self.lane_demand.len() {
            return vec![0.0; total];
        }
        std::mem::replace(&mut self.lane_demand[lane], vec![0.0; total])
    }

    /// Adds anticipated heat (expected routing shares of an upcoming request).
    pub fn add_heat(&mut self, id: ExpertId, h: f32) {
        let idx = self.flat_index(id);
        self.entries[idx].heat += h;
    }

    /// Installs prior per-token routing rates (`TidePriors::rates`). `prior_weight` is how many
    /// "heat half-lives" of observation the prior is worth (1.0 = as much as a steady stream).
    pub fn set_prior_rates(&mut self, rates: Vec<f32>, prior_weight: f32) {
        if rates.len() == self.prior_rate.len() {
            self.prior_rate = rates;
            self.prior_heat_equiv = prior_weight.max(0.0) / (1.0 - HEAT_DECAY);
        }
    }

    /// Changes the unified VRAM expert budget (pool re-sizing before any expert is resident).
    pub fn set_unified_vram_budget(&mut self, bytes: usize) {
        self.unified_vram_budget_bytes = bytes;
    }

    pub fn has_priors(&self) -> bool {
        self.prior_heat_equiv > 0.0
    }

    pub fn prior_rate(&self, id: ExpertId) -> f32 {
        self.prior_rate[self.flat_index(id)]
    }

    /// Demand of the current request (and resets it).
    pub fn take_demand(&mut self) -> Vec<f32> {
        let total = self.demand.len();
        std::mem::replace(&mut self.demand, vec![0.0; total])
    }

    /// Value of keeping an expert resident: (session heat + prior) per MiB. Heat is the decayed
    /// sum of routing shares; in steady state it is `rate / (1 - HEAT_DECAY)`, so the prior is
    /// expressed in the same units.
    pub fn value(&self, id: ExpertId) -> f64 {
        let idx = self.flat_index(id);
        let e = &self.entries[idx];
        let mib = self.expert_bytes(id) as f64 / (1u64 << 20) as f64;
        (e.heat as f64 + self.prior_heat_equiv as f64 * self.prior_rate[idx] as f64) / mib.max(1e-6)
    }

    fn class_full(&self, cls: ExpertClass) -> bool {
        match cls {
            ExpertClass::Nvfp4 => self.max_vram_slots_nvfp4.map_or(false, |cap| self.used_vram_slots_nvfp4 >= cap),
            ExpertClass::Mxfp8 => self.max_vram_slots_mxfp8.map_or(false, |cap| self.used_vram_slots_mxfp8 >= cap),
        }
    }

    /// Lowest-value evictable resident of `cls` (not pinned, pending, protected or scratch).
    fn coldest_in_class(&self, cls: ExpertClass) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if e.vram_slot.is_missing() || e.is_scratch_slot || e.vram_pinned || e.vram_pending {
                continue;
            }
            if self.protected.contains(&i) {
                continue;
            }
            let eid = self.expert_from_flat(i);
            if self.expert_class(eid) != cls {
                continue;
            }
            let age = self.step.saturating_sub(e.last_touch_step) as f64;
            let v = self.value(eid) - age * 1e-7;
            if best.map_or(true, |(_, bv)| v < bv) {
                best = Some((i, v));
            }
        }
        best
    }

    /// True if `id` would get a resident slot without evicting anything, or is worth more than the
    /// coldest evictable resident of its pool by `SWAP_HYSTERESIS`.
    pub fn worth_promoting(&self, id: ExpertId) -> bool {
        self.worth_promoting_with(id, SWAP_HYSTERESIS)
    }

    /// `worth_promoting` with an explicit hysteresis factor.
    pub fn worth_promoting_with(&self, id: ExpertId, hysteresis: f64) -> bool {
        let cls = self.expert_class(id);
        let need = self.expert_bytes(id);
        if !self.class_full(cls) && self.vram_used_bytes + need <= self.vram_expert_budget_bytes() {
            return true;
        }
        match self.eviction_floor(cls) {
            None => false,
            Some(coldest) => self.value(id) > coldest * hysteresis + 1e-9,
        }
    }

    /// `None` if an expert of `cls` (of `need` bytes) fits without eviction, else the value floor
    /// it has to beat (before hysteresis). O(experts): compute once per planning pass and only
    /// recompute after a promotion changes the resident set.
    pub fn promotion_floor(&self, cls: ExpertClass, need: usize) -> Option<f64> {
        if !self.class_full(cls) && self.vram_used_bytes + need <= self.vram_expert_budget_bytes() {
            return None;
        }
        Some(self.eviction_floor(cls).unwrap_or(f64::INFINITY))
    }

    /// Value (per MiB) of what promoting an expert of `cls` would evict: the coldest of its own
    /// pool when that pool is slot-full, otherwise the coldest resident of any pool (byte budget).
    fn eviction_floor(&self, cls: ExpertClass) -> Option<f64> {
        if self.class_full(cls) {
            self.coldest_in_class(cls).map(|(_, v)| v)
        } else {
            self.pick_vram_victim(false, Some(cls)).map(|v| self.value(v))
        }
    }

    /// Configures the exact resident capacities and Scratch Rescue Ring base indices of each VRAM pool.
    pub fn configure_scratch_ring(
        &mut self,
        nvfp4_resident_capacity: u32,
        mxfp8_resident_capacity: u32,
        ring_size: u32,
    ) {
        self.scratch_base_nvfp4 = nvfp4_resident_capacity;
        self.scratch_base_mxfp8 = mxfp8_resident_capacity;
        self.max_vram_slots_nvfp4 = Some(nvfp4_resident_capacity as usize);
        self.max_vram_slots_mxfp8 = Some(mxfp8_resident_capacity as usize);
        self.scratch_ring_size = ring_size.max(1);
        self.scratch_cursor_nvfp4 = 0;
        self.scratch_cursor_mxfp8 = 0;
        self.scratch_occupants_nvfp4 = vec![None; self.scratch_ring_size as usize];
        self.scratch_occupants_mxfp8 = vec![None; self.scratch_ring_size as usize];
    }

    pub fn has_free_vram_room_for(&self, id: ExpertId) -> bool {
        let need = self.expert_bytes(id);
        if self.vram_used_bytes + need > self.vram_expert_budget_bytes() {
            return false;
        }
        match self.expert_class(id) {
            ExpertClass::Nvfp4 => self
                .max_vram_slots_nvfp4
                .map_or(true, |cap| self.used_vram_slots_nvfp4 < cap),
            ExpertClass::Mxfp8 => self
                .max_vram_slots_mxfp8
                .map_or(true, |cap| self.used_vram_slots_mxfp8 < cap),
        }
    }

    /// Marks an expert's background Stage 2 NVMe -> RAM read as completed and ready for Stage 1 DMA.
    pub fn mark_ram_ready(&mut self, id: ExpertId) {
        let idx = self.flat_index(id);
        self.entries[idx].ram_in_flight = false;
    }

    /// Returns true if `id` is present in Tier 1 RAM and NOT still in-flight from NVMe.
    pub fn is_ram_ready(&self, id: ExpertId) -> bool {
        let e = self.entries[self.flat_index(id)];
        !e.ram_slot.is_missing() && !e.ram_in_flight
    }

    /// Exports a flat snapshot of live EWMA heat scores for `TidePriors::merge_session_heat`.
    pub fn export_heat_snapshot(&self) -> Vec<f32> {
        self.entries.iter().map(|e| e.heat).collect()
    }

    /// Value of the coldest evictable resident in `id`'s pool, or `None` if `id` fits for free.
    pub fn coldest_resident_value_for(&self, id: ExpertId) -> Option<f64> {
        let cls = self.expert_class(id);
        if !self.class_full(cls) && self.vram_used_bytes + self.expert_bytes(id) <= self.vram_expert_budget_bytes() {
            return None;
        }
        Some(self.eviction_floor(cls).unwrap_or(f64::INFINITY))
    }

    /// Returns true if a sync-rescued miss is hot enough to earn a resident VRAM slot,
    /// or false if it should run transiently in the Scratch Rescue Ring to avoid thrashing.
    pub fn should_promote_instead_of_scratch(&self, id: ExpertId) -> bool {
        // Cold 1-off misses run from the scratch ring instead of thrashing warm residents.
        self.worth_promoting(id)
    }

    /// Stages a cold 1-off sync miss into the transient Scratch Rescue Ring without evicting
    /// any resident VRAM expert.
    pub fn stage_scratch_rescue(&mut self, id: ExpertId) -> TransferOp {
        let cls = self.expert_class(id);
        let (base, prev_occupant, ring_idx) = match cls {
            ExpertClass::Nvfp4 => {
                let r = (self.scratch_cursor_nvfp4 % self.scratch_ring_size) as usize;
                self.scratch_cursor_nvfp4 = self.scratch_cursor_nvfp4.wrapping_add(1);
                let prev = self.scratch_occupants_nvfp4[r].replace(id);
                (self.scratch_base_nvfp4, prev, r)
            }
            ExpertClass::Mxfp8 => {
                let r = (self.scratch_cursor_mxfp8 % self.scratch_ring_size) as usize;
                self.scratch_cursor_mxfp8 = self.scratch_cursor_mxfp8.wrapping_add(1);
                let prev = self.scratch_occupants_mxfp8[r].replace(id);
                (self.scratch_base_mxfp8, prev, r)
            }
        };

        // Clear any previous transient occupant of this scratch slot
        if let Some(prev) = prev_occupant {
            let p_idx = self.flat_index(prev);
            if self.entries[p_idx].is_scratch_slot {
                self.entries[p_idx].vram_slot = SlotIndex::MISSING;
                self.entries[p_idx].is_scratch_slot = false;
                self.entries[p_idx].vram_pending = false;
                self.gpu_slot_table[p_idx] = SlotIndex::MISSING.0;
                self.dirty_gpu_indices.insert(p_idx);
            }
        }

        let scratch_slot = SlotIndex(base + ring_idx as u32);

        let idx = self.flat_index(id);
        self.entries[idx].vram_slot = scratch_slot;
        self.entries[idx].is_scratch_slot = true;
        self.entries[idx].vram_pending = true;

        let src_ram = if self.is_ram_ready(id) {
            Some(self.entries[idx].ram_slot)
        } else {
            None
        };
        TransferOp::ScratchRescue {
            expert: id,
            src_ram_slot: src_ram,
            scratch_vram_slot: scratch_slot,
        }
    }

    #[inline]
    pub fn flat_index(&self, id: ExpertId) -> usize {
        (id.layer.0 as usize) * self.model_cfg.n_routed_experts + (id.expert as usize)
    }

    #[inline]
    pub fn expert_from_flat(&self, idx: usize) -> ExpertId {
        let n = self.model_cfg.n_routed_experts;
        ExpertId {
            layer: PhysicalLayerId((idx / n) as u16),
            expert: (idx % n) as u16,
        }
    }

    #[inline]
    pub fn expert_class(&self, id: ExpertId) -> ExpertClass {
        if self.model_cfg.is_mxfp8_layer(id.layer.0 as usize) {
            ExpertClass::Mxfp8
        } else {
            ExpertClass::Nvfp4
        }
    }

    #[inline]
    pub fn expert_bytes(&self, id: ExpertId) -> usize {
        self.model_cfg.expert_bytes(id.layer.0 as usize)
    }

    #[inline]
    pub fn loop_priority_multiplier(&self, id: ExpertId) -> f32 {
        if self.model_cfg.is_loop_core_layer(id.layer.0 as usize) {
            self.model_cfg.loop_passes as f32
        } else {
            1.0
        }
    }

    pub fn vram_expert_budget_bytes(&self) -> usize {
        self.unified_vram_budget_bytes
            .saturating_sub(self.kv_reserved_bytes)
    }

    pub fn vram_used_bytes(&self) -> usize {
        self.vram_used_bytes
    }

    pub fn ram_used_bytes(&self) -> usize {
        self.ram_used_bytes
    }

    pub fn ram_budget_bytes(&self) -> usize {
        self.ram_budget_bytes
    }

    pub fn entry(&self, id: ExpertId) -> ExpertResidency {
        self.entries[self.flat_index(id)]
    }

    pub fn gpu_slot_table(&self) -> &[u32] {
        &self.gpu_slot_table
    }

    /// Returns `(flat_index, new_slot_u32)` updates to patch on the GPU indirection table.
    pub fn take_dirty_gpu_updates(&mut self) -> Vec<(usize, u32)> {
        let mut out = Vec::with_capacity(self.dirty_gpu_indices.len());
        for idx in self.dirty_gpu_indices.drain() {
            out.push((idx, self.gpu_slot_table[idx]));
        }
        out
    }

    pub fn advance_step(&mut self) {
        self.step += 1;
        // Mild global EWMA decay so heat adapts quickly to topic/domain shifts.
        for e in &mut self.entries {
            if e.heat > 1e-6 {
                e.heat *= HEAT_DECAY;
            }
        }
    }

    /// Records one routing observation with the expert's *normalized* share of its token's top-k
    /// (shares of one token's top-k sum to 1). Loop-core experts are touched once per pass, so
    /// their 3x use per token is counted naturally (no extra multiplier).
    pub fn touch_expert(&mut self, id: ExpertId, router_share: f32) {
        let idx = self.flat_index(id);
        self.demand[idx] += router_share;
        self.lane_demand[self.current_lane][idx] += router_share;
        let e = &mut self.entries[idx];
        e.heat += router_share;
        e.last_touch_step = self.step;
    }

    /// Elastic KV <-> Expert Pool: reserves `new_kv_bytes` out of the unified VRAM pool.
    /// Evicts coldest non-pinned outer-layer experts first (zero writeback since read-only!).
    /// Only unpins coldest Loop-Core experts if KV growth demands more than outer-layer VRAM.
    /// Returns the list of evicted experts.
    pub fn set_kv_reserved_bytes(&mut self, new_kv_bytes: usize) -> Vec<ExpertId> {
        self.kv_reserved_bytes = new_kv_bytes.min(self.unified_vram_budget_bytes);
        let budget = self.vram_expert_budget_bytes();
        let mut evicted = Vec::new();
        while self.vram_used_bytes > budget {
            if let Some(victim) = self.pick_vram_victim(false, None) {
                self.evict_from_vram(victim);
                evicted.push(victim);
            } else if let Some(victim) = self.pick_vram_victim(true, None) {
                // Only reached at extreme context (e.g., >327K FP8 KV) where KV eats into Loop Core.
                self.evict_from_vram(victim);
                evicted.push(victim);
            } else {
                break;
            }
        }
        evicted
    }

    fn pick_vram_victim(
        &self,
        allow_pinned: bool,
        prefer_class: Option<ExpertClass>,
    ) -> Option<ExpertId> {
        let mut best_idx: Option<usize> = None;
        let mut best_score = f64::INFINITY;

        for (idx, e) in self.entries.iter().enumerate() {
            if e.vram_slot.is_missing() || e.is_scratch_slot || e.vram_pending {
                continue;
            }
            if e.vram_pinned && !allow_pinned {
                continue;
            }
            if self.protected.contains(&idx) {
                continue;
            }
            let id = self.expert_from_flat(idx);
            // Lower score = colder victim. Loop Core gets natural multiplier protection.
            // Prefer matching slot class ONLY as a tiny tie-breaker (never evict a hot
            // same-class expert over a cold other-class expert!).
            let age = self.step.saturating_sub(e.last_touch_step) as f64;
            let class_tiebreak = match prefer_class {
                Some(cls) if self.expert_class(id) == cls => -1e-9,
                _ => 0.0,
            };
            let score = self.value(id) - (age * 1e-7) + class_tiebreak;
            if score < best_score {
                best_score = score;
                best_idx = Some(idx);
            }
        }
        best_idx.map(|i| self.expert_from_flat(i))
    }

    fn pick_ram_victim(&self, prefer_class: Option<ExpertClass>) -> Option<ExpertId> {
        let mut best_idx: Option<usize> = None;
        let mut best_score = f64::INFINITY;

        for (idx, e) in self.entries.iter().enumerate() {
            if e.ram_slot.is_missing() || e.ram_in_flight {
                continue;
            }
            let id = self.expert_from_flat(idx);
            // Prefer evicting experts already resident in VRAM (zero immediate need for RAM copy)
            // or coldest experts overall, with class match only as a tiny tie-breaker.
            let in_vram_bonus = if !e.vram_slot.is_missing() { -1000.0 } else { 0.0 };
            let age = self.step.saturating_sub(e.last_touch_step) as f64;
            let class_tiebreak = match prefer_class {
                Some(cls) if self.expert_class(id) == cls => -1e-6,
                _ => 0.0,
            };
            let score = self.value(id) + in_vram_bonus - (age * 1e-7) + class_tiebreak;
            if score < best_score {
                best_score = score;
                best_idx = Some(idx);
            }
        }
        best_idx.map(|i| self.expert_from_flat(i))
    }

    fn evict_from_vram(&mut self, id: ExpertId) -> SlotIndex {
        let idx = self.flat_index(id);
        let slot = self.entries[idx].vram_slot;
        debug_assert!(!slot.is_missing());
        self.entries[idx].vram_slot = SlotIndex::MISSING;
        self.entries[idx].vram_pinned = false;
        self.entries[idx].vram_pending = false;
        self.gpu_slot_table[idx] = SlotIndex::MISSING.0;
        self.dirty_gpu_indices.insert(idx);
        self.vram_used_bytes = self.vram_used_bytes.saturating_sub(self.expert_bytes(id));
        match self.expert_class(id) {
            ExpertClass::Nvfp4 => {
                self.used_vram_slots_nvfp4 = self.used_vram_slots_nvfp4.saturating_sub(1);
                self.free_vram_slots_nvfp4.push(slot);
            }
            ExpertClass::Mxfp8 => {
                self.used_vram_slots_mxfp8 = self.used_vram_slots_mxfp8.saturating_sub(1);
                self.free_vram_slots_mxfp8.push(slot);
            }
        }
        slot
    }

    fn evict_from_ram(&mut self, id: ExpertId) -> SlotIndex {
        let idx = self.flat_index(id);
        let slot = self.entries[idx].ram_slot;
        debug_assert!(!slot.is_missing());
        self.entries[idx].ram_slot = SlotIndex::MISSING;
        self.ram_used_bytes = self.ram_used_bytes.saturating_sub(self.expert_bytes(id));
        match self.expert_class(id) {
            ExpertClass::Nvfp4 => self.free_ram_slots_nvfp4.push(slot),
            ExpertClass::Mxfp8 => self.free_ram_slots_mxfp8.push(slot),
        }
        slot
    }

    fn alloc_vram_slot(&mut self, cls: ExpertClass) -> SlotIndex {
        let (free, next) = match cls {
            ExpertClass::Nvfp4 => (&mut self.free_vram_slots_nvfp4, &mut self.next_vram_slot_nvfp4),
            ExpertClass::Mxfp8 => (&mut self.free_vram_slots_mxfp8, &mut self.next_vram_slot_mxfp8),
        };
        if let Some(s) = free.pop() {
            s
        } else {
            let s = SlotIndex(*next);
            *next += 1;
            s
        }
    }

    fn alloc_ram_slot(&mut self, cls: ExpertClass) -> SlotIndex {
        let (free, next) = match cls {
            ExpertClass::Nvfp4 => (&mut self.free_ram_slots_nvfp4, &mut self.next_ram_slot_nvfp4),
            ExpertClass::Mxfp8 => (&mut self.free_ram_slots_mxfp8, &mut self.next_ram_slot_mxfp8),
        };
        if let Some(s) = free.pop() {
            s
        } else {
            let s = SlotIndex(*next);
            *next += 1;
            s
        }
    }

    /// Stage 2: Ensures `id` is allocated in Tier 1 Host RAM, evicting cold RAM experts if needed.
    /// When `mark_in_flight` is true, marks the slot pending until the background Stage 2 I/O thread completes.
    pub fn promote_to_ram_with_flag(&mut self, id: ExpertId, mark_in_flight: bool) -> Option<TransferOp> {
        let idx = self.flat_index(id);
        if !self.entries[idx].ram_slot.is_missing() {
            return None;
        }
        let need = self.expert_bytes(id);
        if need > self.ram_budget_bytes {
            return None;
        }
        let mut evicted_last = None;
        while self.ram_used_bytes + need > self.ram_budget_bytes {
            let victim = self
                .pick_ram_victim(Some(self.expert_class(id)))
                .or_else(|| self.pick_ram_victim(None))?;
            self.evict_from_ram(victim);
            evicted_last = Some(victim);
        }

        let slot = self.alloc_ram_slot(self.expert_class(id));
        self.entries[idx].ram_slot = slot;
        self.entries[idx].ram_in_flight = mark_in_flight;
        self.ram_used_bytes += need;

        Some(TransferOp::DiskToRam {
            expert: id,
            ram_slot: slot,
            evicted_from_ram: evicted_last,
        })
    }

    pub fn promote_to_ram(&mut self, id: ExpertId) -> Option<TransferOp> {
        self.promote_to_ram_with_flag(id, false)
    }

    /// Stage 1 (or Direct Stage 2->0): Promotes `id` into Tier 0 VRAM, optionally pinning it.
    ///
    /// The new slot is left `vram_pending` (unpublished) until the executor knows its DMA landed.
    /// With `allow_disk_direct = false` (every path inside the decode loop except sync rescue) an
    /// expert that is not Ready in RAM is NOT promoted (returns `None`, nothing evicted); the caller
    /// should queue an async Disk->RAM read instead of blocking the token loop on NVMe.
    pub fn promote_to_vram(
        &mut self,
        id: ExpertId,
        pin_in_vram: bool,
        allow_disk_direct: bool,
    ) -> Option<TransferOp> {
        let idx = self.flat_index(id);
        if !self.entries[idx].vram_slot.is_missing() && !self.entries[idx].is_scratch_slot {
            if pin_in_vram {
                self.entries[idx].vram_pinned = true;
            }
            return None;
        }
        if !allow_disk_direct && !self.is_ram_ready(id) {
            return None;
        }

        let need = self.expert_bytes(id);
        let budget = self.vram_expert_budget_bytes();
        if need > budget {
            return None;
        }

        let mut evicted_last = None;
        let cls = self.expert_class(id);

        // First ensure physical slot capacity of this expert's pool (must evict same class!)
        while self.class_full(cls) {
            let (v_idx, _) = self.coldest_in_class(cls)?;
            let victim = self.expert_from_flat(v_idx);
            self.evict_from_vram(victim);
            evicted_last = Some(victim);
        }

        // Then ensure unified byte budget (accounting for Elastic KV reservation)
        while self.vram_used_bytes + need > budget {
            let victim = self
                .pick_vram_victim(false, Some(cls))
                .or_else(|| self.pick_vram_victim(false, None))?;
            self.evict_from_vram(victim);
            evicted_last = Some(victim);
        }

        let dst_slot = self.alloc_vram_slot(cls);
        match cls {
            ExpertClass::Nvfp4 => self.used_vram_slots_nvfp4 += 1,
            ExpertClass::Mxfp8 => self.used_vram_slots_mxfp8 += 1,
        }
        // A scratch occupant being promoted: drop its (published) scratch mapping first.
        if self.entries[idx].is_scratch_slot {
            self.clear_scratch_occupant(id);
        }
        self.entries[idx].vram_slot = dst_slot;
        self.entries[idx].is_scratch_slot = false;
        self.entries[idx].vram_pinned = pin_in_vram;
        self.entries[idx].vram_pending = true;
        self.gpu_slot_table[idx] = SlotIndex::MISSING.0;
        self.dirty_gpu_indices.insert(idx);
        self.vram_used_bytes += need;

        let src_ram = self.entries[idx].ram_slot;
        let ram_ready = !src_ram.is_missing() && !self.entries[idx].ram_in_flight;
        let release_ram = pin_in_vram && self.tide_cfg.exclusive_loop_pinning;

        if ram_ready {
            if release_ram {
                self.evict_from_ram(id);
            }
            Some(TransferOp::RamToVram {
                expert: id,
                src_ram_slot: src_ram,
                dst_vram_slot: dst_slot,
                evicted_from_vram: evicted_last,
                release_ram_after_upload: release_ram,
            })
        } else {
            // Not ready in RAM yet: optionally keep a RAM copy unless exclusive pinned.
            let keep_ram_slot = if !release_ram
                && !self.no_ram_copy_on_direct
                && src_ram.is_missing()
                && self.ram_used_bytes + need <= self.ram_budget_bytes
            {
                let rs = self.alloc_ram_slot(self.expert_class(id));
                self.entries[idx].ram_slot = rs;
                self.entries[idx].ram_in_flight = false;
                self.ram_used_bytes += need;
                Some(rs)
            } else {
                None
            };
            Some(TransferOp::DiskToVramDirect {
                expert: id,
                dst_vram_slot: dst_slot,
                also_keep_ram_slot: keep_ram_slot,
                evicted_from_vram: evicted_last,
            })
        }
    }

    /// True if `id` is published in the GPU table (kernels will compute it) — i.e. a real VRAM hit.
    #[inline]
    pub fn is_vram_hit(&self, id: ExpertId) -> bool {
        let e = &self.entries[self.flat_index(id)];
        !e.vram_slot.is_missing() && !e.vram_pending
    }

    /// True if `id` has a VRAM slot whose DMA may still be in flight.
    #[inline]
    pub fn is_vram_pending(&self, id: ExpertId) -> bool {
        let e = &self.entries[self.flat_index(id)];
        !e.vram_slot.is_missing() && e.vram_pending
    }

    /// Publishes a pending expert into the GPU table mirror (its DMA is known complete).
    /// `expected_slot` guards against publishing a slot that was since re-assigned.
    pub fn publish_vram(&mut self, id: ExpertId, expected_slot: SlotIndex) -> bool {
        let idx = self.flat_index(id);
        let e = &mut self.entries[idx];
        if e.vram_pending && e.vram_slot == expected_slot {
            e.vram_pending = false;
            self.gpu_slot_table[idx] = expected_slot.0;
            self.dirty_gpu_indices.insert(idx);
            true
        } else {
            false
        }
    }

    /// Publishes every pending expert (caller guarantees all outstanding DMAs are complete in
    /// compute-stream order, e.g. the compute stream waited on the copy stream's latest event).
    pub fn publish_all_pending(&mut self) -> usize {
        let mut n = 0;
        for idx in 0..self.entries.len() {
            let e = &mut self.entries[idx];
            if e.vram_pending && !e.vram_slot.is_missing() {
                e.vram_pending = false;
                self.gpu_slot_table[idx] = e.vram_slot.0;
                self.dirty_gpu_indices.insert(idx);
                n += 1;
            }
        }
        n
    }

    pub fn has_dirty_gpu_table(&self) -> bool {
        !self.dirty_gpu_indices.is_empty()
    }

    /// Marks experts that must not be evicted until `clear_protected`.
    pub fn set_protected(&mut self, ids: &[ExpertId]) {
        self.protected.clear();
        for &id in ids {
            let idx = self.flat_index(id);
            self.protected.push(idx);
        }
    }

    pub fn clear_protected(&mut self) {
        self.protected.clear();
    }

    fn clear_scratch_occupant(&mut self, id: ExpertId) {
        let occupants = match self.expert_class(id) {
            ExpertClass::Nvfp4 => &mut self.scratch_occupants_nvfp4,
            ExpertClass::Mxfp8 => &mut self.scratch_occupants_mxfp8,
        };
        for o in occupants.iter_mut() {
            if *o == Some(id) {
                *o = None;
            }
        }
    }

    /// Unmaps every transient Scratch Rescue occupant (their slots are about to be reused, e.g.
    /// by batched prefill). The caller must upload the table before overwriting the slots.
    pub fn clear_scratch_ring(&mut self) {
        let occ: Vec<ExpertId> = self
            .scratch_occupants_nvfp4
            .iter_mut()
            .chain(self.scratch_occupants_mxfp8.iter_mut())
            .filter_map(|o| o.take())
            .collect();
        for id in occ {
            let idx = self.flat_index(id);
            if self.entries[idx].is_scratch_slot {
                self.entries[idx].vram_slot = SlotIndex::MISSING;
                self.entries[idx].is_scratch_slot = false;
                self.entries[idx].vram_pending = false;
                self.gpu_slot_table[idx] = SlotIndex::MISSING.0;
                self.dirty_gpu_indices.insert(idx);
            }
        }
    }

    /// First Scratch Rescue slot index and ring size of a class.
    pub fn scratch_ring(&self, cls: ExpertClass) -> (u32, u32) {
        match cls {
            ExpertClass::Nvfp4 => (self.scratch_base_nvfp4, self.scratch_ring_size),
            ExpertClass::Mxfp8 => (self.scratch_base_mxfp8, self.scratch_ring_size),
        }
    }

    pub fn count_vram_resident(&self) -> (usize, usize, usize) {
        let (mut prelude, mut loop_core, mut coda) = (0, 0, 0);
        for (idx, e) in self.entries.iter().enumerate() {
            if e.vram_slot.is_missing() || e.is_scratch_slot {
                continue;
            }
            let layer = self.expert_from_flat(idx).layer.0 as usize;
            if layer < self.model_cfg.loop_start_layer {
                prelude += 1;
            } else if layer < self.model_cfg.loop_end_exclusive {
                loop_core += 1;
            } else {
                coda += 1;
            }
        }
        (prelude, loop_core, coda)
    }

    pub fn count_ram_resident(&self) -> usize {
        self.entries.iter().filter(|e| !e.ram_slot.is_missing()).count()
    }
}
