use serde::{Deserialize, Serialize};

/// Supported tensor element and block-quantized formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    /// FP8 E4M3 (native RDNA4 WMMA, default for Inkling-2-Mini dense weights, lm_head, and KV cache).
    FP8E4M3,
    /// Native Inkling-2-Mini layers 20-23 experts: FP8 E4M3 with block-32 E8M0 scales (~8.25 bits/param).
    MXFP8,
    /// Native Inkling-2-Mini layers 0-19 experts: FP4 E2M1 with block-16 FP8 scales (~4.5 bits/param).
    NVFP4,
    /// Optional GGUF compatibility formats for Stage 2 testing and other MoE models.
    Q8_0,
    IQ4NL,
}

impl DType {
    /// Approximate bits per parameter including block scaling metadata.
    pub const fn bits_per_param(self) -> f64 {
        match self {
            DType::F32 => 32.0,
            DType::F16 | DType::BF16 => 16.0,
            DType::FP8E4M3 => 8.0,
            DType::MXFP8 => 8.25,
            DType::NVFP4 => 4.5,
            DType::Q8_0 => 8.5,
            DType::IQ4NL => 4.5,
        }
    }
}

/// Physical layer index (0..23 for Inkling-2-Mini).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PhysicalLayerId(pub u16);

/// Logical execution layer index (0..47 for Inkling-2-Mini: 6 prelude + 3x12 loop + 6 coda).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LogicalLayerId(pub u16);

/// Unique identifier for a routed MoE expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExpertId {
    pub layer: PhysicalLayerId,
    pub expert: u16,
}

/// Index into a pre-allocated VRAM or RAM expert slab pool.
/// `MISSING` (u32::MAX) indicates the expert is not currently resident in that tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SlotIndex(pub u32);

impl SlotIndex {
    pub const MISSING: Self = Self(u32::MAX);

    pub const fn is_missing(self) -> bool {
        self.0 == u32::MAX
    }
}
