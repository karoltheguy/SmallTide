//! Core types, model configuration, telemetry, and the backend-agnostic `ComputeBackend` trait.
//!
//! Invariant: ZERO ROCm/HIP, CUDA, or Vulkan symbols may appear in this crate.

pub mod backend;
pub mod config;
pub mod error;
pub mod telemetry;
pub mod types;

pub use backend::{
    ComputeBackend, DeviceBuf, DeviceInfo, Event, Graph, PinnedBuf, QueueKind, Stream,
};
pub use config::{EngineConfig, Inkling2Config, KvDType, RescuePolicy, TideConfig};
pub use error::{Result, SmallTideError};
pub use telemetry::TideTelemetry;
pub use types::{DType, ExpertId, LogicalLayerId, PhysicalLayerId, SlotIndex};
