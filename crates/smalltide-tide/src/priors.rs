//! Persisted per-model hot-expert frequency priors (`TidePriors`) for Mode 1 Pre-Prefill warm-up
//! and end-of-request heat consolidation.

use serde::{Deserialize, Serialize};
use smalltide_core::{Result, SmallTideError};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TidePriors {
    pub model_name: String,
    pub num_physical_layers: usize,
    pub n_routed_experts: usize,
    pub sessions_recorded: u64,
    /// Normalized hotness scores `[num_physical_layers * n_routed_experts]`.
    pub scores: Vec<f32>,
}

impl TidePriors {
    pub fn new(model_name: impl Into<String>, num_physical_layers: usize, n_routed_experts: usize) -> Self {
        Self {
            model_name: model_name.into(),
            num_physical_layers,
            n_routed_experts,
            sessions_recorded: 0,
            scores: vec![0.0; num_physical_layers * n_routed_experts],
        }
    }

    /// Blends live EWMA expert heat into persisted priors using an exponential moving average
    /// across sessions so recent workloads gradually adapt the startup resident set.
    pub fn merge_session_heat(&mut self, live_heat: &[f32], blend_alpha: f32) {
        if live_heat.len() != self.scores.len() {
            return;
        }
        let max_live = live_heat
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        if max_live <= 1e-8 {
            return;
        }
        let alpha = if self.sessions_recorded == 0 {
            1.0
        } else {
            blend_alpha.clamp(0.05, 0.95)
        };
        for (dst, &h) in self.scores.iter_mut().zip(live_heat.iter()) {
            let norm = h / max_live;
            *dst = (*dst) * (1.0 - alpha) + norm * alpha;
        }
        self.sessions_recorded += 1;
    }

    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| SmallTideError::InvalidModel(format!("serialize priors: {e}")))?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn load_from_file(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        serde_json::from_slice(&bytes)
            .map_err(|e| SmallTideError::InvalidModel(format!("parse priors: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priors_merge_and_persistence() {
        let mut p = TidePriors::new("Inkling-2-Mini", 24, 128);
        let mut heat = vec![0.0f32; 24 * 128];
        heat[6 * 128 + 42] = 10.0;
        heat[20 * 128 + 7] = 5.0;
        p.merge_session_heat(&heat, 0.3);
        assert_eq!(p.sessions_recorded, 1);
        assert!((p.scores[6 * 128 + 42] - 1.0).abs() < 1e-6);
        assert!((p.scores[20 * 128 + 7] - 0.5).abs() < 1e-6);

        let tmp = std::env::temp_dir().join("smalltide_priors_test.json");
        p.save_to_file(&tmp).unwrap();
        let loaded = TidePriors::load_from_file(&tmp).unwrap();
        let _ = fs::remove_file(&tmp);

        assert_eq!(loaded.sessions_recorded, 1);
        assert!((loaded.scores[6 * 128 + 42] - 1.0).abs() < 1e-6);
    }
}
