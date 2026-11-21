//! 3-Tier TideLM Cache (VRAM <-> RAM <-> NVMe) and 4-mode adaptive expert scheduler:
//! 1. Pre-prefill warm-up (Loop-Core priority + persisted `TidePriors`)
//! 2. Prefill -> Decode boundary tide rebalance
//! 3. Paced async background promotion (Stage 1 RAM->VRAM + Stage 2 background `Stage2IoWorker`)
//! 4. Configurable sync rescue (Exact / Threshold / Disabled) with transient Scratch Rescue Ring.

pub mod batch;
pub mod cache;
pub mod io_worker;
pub mod paged_kv;
pub mod priors;
pub mod runtime;
pub mod scheduler;

pub use batch::{BatchTokenItem, ContinuousBatchScheduler, SeqStatus, SequenceRequest, StepBatch};
pub use cache::{ExpertClass, ExpertResidency, TideCache, TransferOp};
pub use io_worker::{CompletedDiskRead, Stage2IoWorker};
pub use paged_kv::{PagedKvManager, PhysicalPageId, KV_PAGE_TOKENS};
pub use priors::TidePriors;
pub use runtime::{TideExecutor, DEFAULT_SCRATCH_RING_SLOTS};
pub use scheduler::{ActiveExpertSlot, LayerDispatchPlan, TideScheduler};
pub use smalltide_core::{ExpertId, RescuePolicy, SlotIndex, TideConfig, TideTelemetry};
