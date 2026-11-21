//! Persisted routing-demand priors (`TidePriors`).
//!
//! For every physical layer the priors hold the observed *routing share* distribution over its
//! experts: the sum of normalized router weights an expert received, divided by the layer total
//! (each layer's shares sum to 1). With the loop structure this gives each expert's expected
//! routing mass per token, `rate = passes(layer) * share` (loop-core layers run 3 passes).
//!
//! Requests vote: at the end of each request its demand is folded in with weight
//! `max(1 / (n + 1), newest_share)`, so the first requests average evenly and later ones age the
//! history exponentially (half-life ~ ln 2 / newest_share requests).
//!
//! The priors drive (1) the value-per-byte split of the VRAM budget between the NVFP4 and MXFP8
//! pools, (2) which experts the pre-prefill warm-up loads, and (3) a floor on each expert's
//! eviction value so a short prompt cannot evict experts that are hot across workloads.

use serde::{Deserialize, Serialize};
use smalltide_core::{Inkling2Config, Result, SmallTideError};
use std::fs;
use std::path::Path;

pub const PRIORS_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TidePriors {
    pub model_name: String,
    #[serde(default)]
    pub version: u32,
    pub num_physical_layers: usize,
    pub n_routed_experts: usize,
    /// Requests (votes) folded in so far.
    pub sessions_recorded: u64,
    /// Total routed layer-passes observed (one per token per logical layer).
    #[serde(default)]
    pub passes_observed: f64,
    /// Per-layer routing share `[num_physical_layers * n_routed_experts]`, each layer sums to 1.
    pub scores: Vec<f32>,
}

impl TidePriors {
    pub fn new(model_name: impl Into<String>, num_physical_layers: usize, n_routed_experts: usize) -> Self {
        Self {
            model_name: model_name.into(),
            version: PRIORS_VERSION,
            num_physical_layers,
            n_routed_experts,
            sessions_recorded: 0,
            passes_observed: 0.0,
            scores: vec![0.0; num_physical_layers * n_routed_experts],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.sessions_recorded == 0
    }

    /// Folds one request's raw demand (sum of normalized router weights per expert, all passes)
    /// in as one vote. Layers with no observed demand keep their history.
    pub fn merge_demand(&mut self, demand: &[f32], newest_share: f32) {
        if demand.len() != self.scores.len() {
            return;
        }
        let e = self.n_routed_experts;
        let total: f64 = demand.iter().map(|&d| d as f64).sum();
        if total <= 1e-9 {
            return;
        }
        let alpha = (1.0 / (self.sessions_recorded as f32 + 1.0)).max(newest_share.clamp(0.0, 0.5));
        for l in 0..self.num_physical_layers {
            let row = &demand[l * e..(l + 1) * e];
            let sum: f32 = row.iter().sum();
            if sum <= 1e-9 {
                continue;
            }
            let dst = &mut self.scores[l * e..(l + 1) * e];
            let had_history = dst.iter().any(|&v| v > 0.0);
            let a = if had_history { alpha } else { 1.0 };
            for (d, &v) in dst.iter_mut().zip(row) {
                *d = *d * (1.0 - a) + (v / sum) * a;
            }
        }
        self.sessions_recorded += 1;
        self.passes_observed += total;
    }

    /// Expected routing mass per token for every expert: `passes(layer) * share`, with light
    /// smoothing so never-observed experts are not exactly zero.
    pub fn rates(&self, cfg: &Inkling2Config) -> Vec<f32> {
        let e = self.n_routed_experts;
        let uniform = 1.0 / e as f32;
        let mut out = vec![0.0f32; self.scores.len()];
        for l in 0..self.num_physical_layers {
            let passes = if cfg.is_loop_core_layer(l) { cfg.loop_passes as f32 } else { 1.0 };
            for x in 0..e {
                let share = 0.98 * self.scores[l * e + x] + 0.02 * uniform;
                out[l * e + x] = passes * share;
            }
        }
        out
    }

    /// Splits `budget_bytes` between the NVFP4 and MXFP8 pools by value per byte: experts are
    /// taken in descending `rate / bytes` order until the budget is spent. Returns slot counts.
    pub fn plan_pool_split(
        &self,
        cfg: &Inkling2Config,
        budget_bytes: usize,
        nvfp4_bytes: usize,
        mxfp8_bytes: usize,
    ) -> (usize, usize) {
        let rates = self.rates(cfg);
        let e = self.n_routed_experts;
        let mut items: Vec<(f64, bool)> = (0..rates.len())
            .map(|i| {
                let mx = cfg.is_mxfp8_layer(i / e);
                let b = if mx { mxfp8_bytes } else { nvfp4_bytes };
                (rates[i] as f64 / b as f64, mx)
            })
            .collect();
        items.sort_by(|a, b| b.0.total_cmp(&a.0));
        let (mut used, mut n4, mut m8) = (0usize, 0usize, 0usize);
        for (_, mx) in items {
            let b = if mx { mxfp8_bytes } else { nvfp4_bytes };
            if used + b > budget_bytes {
                // A smaller NVFP4 expert may still fit after an MXFP8 one does not.
                continue;
            }
            used += b;
            if mx {
                m8 += 1;
            } else {
                n4 += 1;
            }
        }
        (n4, m8)
    }

    /// Fraction of the routing mass per token that a resident set covers (for logs).
    pub fn coverage(&self, cfg: &Inkling2Config, resident: impl Fn(usize) -> bool) -> f64 {
        let rates = self.rates(cfg);
        let total: f64 = rates.iter().map(|&r| r as f64).sum();
        let got: f64 = rates.iter().enumerate().filter(|(i, _)| resident(*i)).map(|(_, &r)| r as f64).sum();
        if total > 0.0 { got / total } else { 0.0 }
    }

    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string(self)
            .map_err(|e| SmallTideError::InvalidModel(format!("serialize priors: {e}")))?;
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, json)?;
        fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn load_from_file(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        let p: Self = serde_json::from_slice(&bytes)
            .map_err(|e| SmallTideError::InvalidModel(format!("parse priors: {e}")))?;
        if p.version != PRIORS_VERSION {
            return Err(SmallTideError::InvalidModel(format!(
                "priors {} has version {} (expected {PRIORS_VERSION}); delete it to rebuild",
                path.display(),
                p.version
            )));
        }
        if p.scores.len() != p.num_physical_layers * p.n_routed_experts {
            return Err(SmallTideError::InvalidModel("priors: score table size mismatch".into()));
        }
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priors_merge_and_persistence() {
        let mut p = TidePriors::new("Inkling-2-Mini", 24, 128);
        let mut d = vec![0.0f32; 24 * 128];
        d[6 * 128 + 42] = 3.0;
        d[6 * 128 + 1] = 1.0;
        d[20 * 128 + 7] = 5.0;
        p.merge_demand(&d, 0.1);
        assert_eq!(p.sessions_recorded, 1);
        assert!((p.scores[6 * 128 + 42] - 0.75).abs() < 1e-6);
        assert!((p.scores[20 * 128 + 7] - 1.0).abs() < 1e-6);
        // Second vote averages evenly (alpha = 1/2).
        let mut d2 = vec![0.0f32; 24 * 128];
        d2[6 * 128 + 1] = 2.0;
        p.merge_demand(&d2, 0.1);
        assert!((p.scores[6 * 128 + 42] - 0.375).abs() < 1e-6);
        assert!((p.scores[6 * 128 + 1] - 0.625).abs() < 1e-6);
        // Layer 20 had no new demand: history kept.
        assert!((p.scores[20 * 128 + 7] - 1.0).abs() < 1e-6);

        let tmp = std::env::temp_dir().join("smalltide_priors_test.json");
        p.save_to_file(&tmp).unwrap();
        let loaded = TidePriors::load_from_file(&tmp).unwrap();
        let _ = fs::remove_file(&tmp);
        assert_eq!(loaded.sessions_recorded, 2);
        assert!((loaded.scores[6 * 128 + 1] - 0.625).abs() < 1e-6);
    }

    #[test]
    fn test_pool_split_prefers_value_per_byte() {
        let cfg = Inkling2Config::default();
        let mut p = TidePriors::new("Inkling-2-Mini", cfg.num_physical_layers, cfg.n_routed_experts);
        let e = cfg.n_routed_experts;
        let mut d = vec![0.0f32; p.scores.len()];
        // Coda MXFP8 layer 20: one dominant expert; loop-core layer 6: flat.
        d[20 * e + 3] = 100.0;
        for x in 0..e {
            d[6 * e + x] = 1.0;
        }
        p.merge_demand(&d, 0.1);
        let (n4, m8) = p.plan_pool_split(&cfg, 10 * 4_500_000 + 7_900_000, 4_500_000, 7_900_000);
        assert_eq!(m8, 1, "the dominant MXFP8 expert is worth its bytes");
        assert_eq!(n4, 10);
    }
}
