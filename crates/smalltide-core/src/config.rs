use serde::{Deserialize, Serialize};

/// Architecture parameters for Thinking Machines Lab's Inkling-2-Mini
/// (matched to official `config.reference.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inkling2Config {
    pub vocab_size: usize,
    pub unpadded_vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_physical_layers: usize,
    pub loop_start_layer: usize,
    /// Exclusive end index matching `loop_config.core_end` (18 => physical layers 6..18 = 6..=17).
    pub loop_end_exclusive: usize,
    pub loop_passes: usize,
    pub loop_gate_rank: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub local_window_size: usize,
    pub sconv_kernel_size: usize,
    pub rms_norm_eps: f32,
    pub qk_l2_clip_threshold: f32,
    pub rel_l2_clip_threshold: f32,
    pub log_scaling_n_floor: usize,
    pub log_scaling_alpha: f32,
    pub final_logit_softcapping: f32,
    pub mlp_gate_softcap: f32,
    pub mlp_up_softcap: f32,
    pub route_scale: f32,
    pub use_gate_bias: bool,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_layers_per_attn_res_block: usize,
    pub max_position_embeddings: usize,
    pub local_layer_ids: Vec<usize>,
    /// Physical layers >= this index use MXFP8 experts; earlier layers use NVFP4 experts.
    pub mxfp8_start_layer: usize,
}

impl Default for Inkling2Config {
    fn default() -> Self {
        Self {
            vocab_size: 201_024,
            unpadded_vocab_size: 200_058,
            hidden_size: 2_304,
            intermediate_size: 1_152,
            num_physical_layers: 24,
            loop_start_layer: 6,
            loop_end_exclusive: 18,
            loop_passes: 3,
            loop_gate_rank: 64,
            num_attention_heads: 24,
            num_key_value_heads: 8,
            head_dim: 128,
            d_rel: 16,
            rel_extent: 1_024,
            local_window_size: 1_024,
            sconv_kernel_size: 4,
            rms_norm_eps: 1e-6,
            qk_l2_clip_threshold: 22.627_417,
            rel_l2_clip_threshold: 8.0,
            log_scaling_n_floor: 8_192,
            log_scaling_alpha: 0.1,
            final_logit_softcapping: 30.0,
            mlp_gate_softcap: 4.0,
            mlp_up_softcap: 25.0,
            route_scale: 8.0,
            use_gate_bias: true,
            n_routed_experts: 128,
            num_experts_per_tok: 8,
            n_layers_per_attn_res_block: 6,
            max_position_embeddings: 1_048_576,
            local_layer_ids: vec![
                0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22,
            ],
            mxfp8_start_layer: 20,
        }
    }
}

impl Inkling2Config {
    /// Total logical layers executed per token (48 for Inkling-2-Mini).
    pub const fn num_logical_layers(&self) -> usize {
        let loop_len = self.loop_end_exclusive - self.loop_start_layer;
        let non_loop = self.num_physical_layers - loop_len;
        non_loop + loop_len * self.loop_passes
    }

    pub const fn is_loop_core_layer(&self, physical_layer: usize) -> bool {
        physical_layer >= self.loop_start_layer && physical_layer < self.loop_end_exclusive
    }

    pub const fn is_mxfp8_layer(&self, physical_layer: usize) -> bool {
        physical_layer >= self.mxfp8_start_layer
    }

    pub fn is_local_layer(&self, physical_layer: usize) -> bool {
        self.local_layer_ids.contains(&physical_layer)
    }

    /// Exact byte size of one expert's weight + scale payload in native precision.
    /// NVFP4 (layers 0..19): 3 * 2304 * 1152 * 0.5625 = 4,478,976 bytes (~4.27 MiB)
    /// MXFP8 (layers 20..23): 3 * 2304 * 1152 * 1.03125 = 8,211,456 bytes (~7.83 MiB)
    pub const fn expert_bytes(&self, physical_layer: usize) -> usize {
        let params = 3 * self.hidden_size * self.intermediate_size;
        if self.is_mxfp8_layer(physical_layer) {
            params + (params / 32)
        } else {
            (params / 2) + (params / 16)
        }
    }
}

/// KV cache precision mode (supports 327K+ context on 16 GB VRAM).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvDType {
    /// FP8 E4M3: 16 MiB / 1K global tokens (~5.08 GiB at 327K).
    Fp8,
    /// Q4 block-32: ~9 MiB / 1K global tokens (~2.89 GiB at 327K; enables 512K+ in 16 GB VRAM).
    Q4,
    /// FP16 reference precision: 32 MiB / 1K global tokens.
    Fp16,
}

/// Sync-rescue behavior when an active routed expert misses VRAM during decode.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RescuePolicy {
    /// Rescue every missed expert via split-batch in-layer overlapped DMA (100% exact).
    Exact,
    /// TideLM threshold mode: sync-rescue only if router weight >= threshold;
    /// otherwise skip & renormalize while queuing background async promotion.
    Threshold(f32),
    /// Pure async-only mode: zero blocking sync rescues during decode (maximum tok/s).
    Disabled,
}

/// TideLM 2-stage streaming configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TideConfig {
    /// Leave at least this many MiB free in VRAM for Windows WDDM / DWM.
    pub vram_headroom_mib: usize,
    /// Maximum host RAM budget in MiB for Tier 1 expert cache (None = auto from system RAM).
    pub ram_budget_mib: Option<usize>,
    /// Size of the page-locked DMA staging ring buffer in MiB (recommended 512-1024 on Windows).
    pub pinned_staging_mib: usize,
    /// Free RAM duplicates for permanently VRAM-pinned Loop Core experts after upload.
    pub exclusive_loop_pinning: bool,
    /// Enable pre-prefill hot-set warm-up.
    pub preload_before_prefill: bool,
    /// Enable bulk tide rebalance at the prefill -> decode boundary.
    pub preload_prefill_decode_boundary: bool,
    /// Maximum number of async Stage 1 (RAM -> VRAM) expert promotions initiated per decode token.
    pub max_async_vram_promotions_per_token: usize,
    /// Maximum number of async Stage 2 (Disk -> RAM) expert reads initiated per decode token.
    pub max_async_ram_promotions_per_token: usize,
    /// Sync rescue behavior on VRAM miss.
    pub rescue_policy: RescuePolicy,
}

impl Default for TideConfig {
    fn default() -> Self {
        Self {
            vram_headroom_mib: 1_792,
            ram_budget_mib: None,
            pinned_staging_mib: 512,
            exclusive_loop_pinning: true,
            preload_before_prefill: true,
            preload_prefill_decode_boundary: true,
            max_async_vram_promotions_per_token: 2,
            max_async_ram_promotions_per_token: 2,
            rescue_policy: RescuePolicy::Disabled,
        }
    }
}

/// Top-level SmallTide engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Cast BF16 dense attention projections, loop controllers, and lm_head to FP8 E4M3 on load.
    pub fp8_dense: bool,
    /// KV cache storage precision (`Fp8` or `Q4` for 327K-512K+ in 16 GB VRAM).
    pub kv_dtype: KvDType,
    /// Enable Elastic KV <-> Expert VRAM slab sharing so expert pool automatically recedes
    /// as global KV context grows toward 327K+.
    pub elastic_kv_expert_pool: bool,
    /// Keep `embed_tokens` in host RAM (~0.86 GiB VRAM saved, 1 row lookup/tok).
    pub cpu_embed_tokens: bool,
    /// Auto-enable `dspark` speculative decoding when `mtp.safetensors` is present.
    pub enable_dspark_if_available: bool,
    /// Number of speculative draft tokens for `dspark` when active.
    pub dspark_draft_tokens: usize,
    /// Maximum context length to support.
    pub max_context_len: usize,
    /// TideLM streaming policy.
    pub tide: TideConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            fp8_dense: true,
            kv_dtype: KvDType::Fp8,
            elastic_kv_expert_pool: true,
            cpu_embed_tokens: true,
            enable_dspark_if_available: true,
            dspark_draft_tokens: 6,
            max_context_len: 262_144,
            tide: TideConfig::default(),
        }
    }
}
