//! Configuration from `.env` (+ process environment) and the shared engine builder for `run`
//! and `serve`.
//!
//! `.env` is read from the path in `SMALLTIDE_ENV`, else `./.env`. Variables already present in
//! the process environment win over `.env` (standard dotenv semantics), so a one-off override is
//! `SMALLTIDE_MAX_CTX=4096 smalltide-cli run ...`. See `.env.example` for every key.

use smalltide_backend_hip::HipBackend;
use smalltide_core::{ComputeBackend, DType, Inkling2Config, RescuePolicy, TideConfig};
use smalltide_io::dense_loader::Inkling2DenseHost;
use smalltide_io::stide_pack::{ExpertSlabLayout, StidePackReader};
use smalltide_io::tokenizer::BpeTokenizer;
use smalltide_model_inkling2::engine::Inkling2Engine;
use smalltide_model_inkling2::weights::Inkling2DenseGpu;
use smalltide_tide::TideExecutor;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// Loads `.env` into the process environment (existing variables are kept). Returns the file
/// that was loaded, if any. Call first thing in `main`.
pub fn load_dotenv() -> Option<PathBuf> {
    let path = std::env::var_os("SMALLTIDE_ENV")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".env"));
    let text = std::fs::read_to_string(&path).ok()?;
    for (n, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((k, v)) = line.split_once('=') else {
            eprintln!("[.env] {}:{}: ignoring line without '='", path.display(), n + 1);
            continue;
        };
        let k = k.trim();
        let mut v = v.trim();
        if (v.starts_with('"') && v.ends_with('"') && v.len() >= 2) || (v.starts_with('\'') && v.ends_with('\'') && v.len() >= 2) {
            v = &v[1..v.len() - 1];
        } else if let Some(i) = v.find(" #") {
            v = v[..i].trim_end(); // inline comment
        }
        if std::env::var_os(k).is_none() {
            std::env::set_var(k, v);
        }
    }
    Some(path)
}

fn var(key: &str) -> Option<String> {
    std::env::var(key).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

fn bad(key: &str, value: &str, expected: &str) -> ! {
    eprintln!("config error: {key}={value:?} (expected {expected})");
    std::process::exit(2);
}

fn parse<T: std::str::FromStr>(key: &str, default: T, expected: &str) -> T {
    match var(key) {
        None => default,
        Some(v) => v.parse().unwrap_or_else(|_| bad(key, &v, expected)),
    }
}

/// `auto` / unset -> None, else a number.
fn parse_auto(key: &str) -> Option<usize> {
    match var(key) {
        None => None,
        Some(v) if v.eq_ignore_ascii_case("auto") => None,
        Some(v) => Some(v.parse().unwrap_or_else(|_| bad(key, &v, "`auto` or a slot count"))),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tri {
    Auto,
    On,
    Off,
}

/// How the auto-sized device expert pools are bounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolBudget {
    /// Free device memory as reported by HIP (on Windows this includes memory the driver
    /// silently backs with system RAM).
    Device,
    /// The OS resident budget (DXGI): only memory that really stays on the card.
    Resident,
}

#[derive(Debug, Clone)]
pub struct Settings {
    pub model_dir: PathBuf,
    pub max_ctx: usize,
    pub rescue_policy: RescuePolicy,
    pub nvfp4_slots: Option<usize>,
    pub mxfp8_slots: Option<usize>,
    pub mxfp8_host: Tri,
    pub pool_budget: PoolBudget,
    pub ram_gb: f64,
    pub batched_prefill: bool,
    pub profile: bool,
    pub host: String,
    pub port: u16,
}

impl Settings {
    pub fn from_env() -> Self {
        let rescue_threshold: f32 = parse("SMALLTIDE_RESCUE_THRESHOLD", 0.25, "a number in [0, 1]");
        let rescue_policy = match var("SMALLTIDE_RESCUE_MODE").as_deref().unwrap_or("threshold") {
            "exact" => RescuePolicy::Exact,
            "disabled" | "none" => RescuePolicy::Disabled,
            "threshold" => RescuePolicy::Threshold(rescue_threshold),
            other => bad("SMALLTIDE_RESCUE_MODE", other, "exact | threshold | disabled"),
        };
        let mxfp8_host = match var("SMALLTIDE_MXFP8_HOST").as_deref().unwrap_or("auto") {
            "auto" => Tri::Auto,
            "on" | "1" | "true" => Tri::On,
            "off" | "0" | "false" => Tri::Off,
            other => bad("SMALLTIDE_MXFP8_HOST", other, "auto | on | off"),
        };
        let pool_budget = match var("SMALLTIDE_POOL_BUDGET").as_deref().unwrap_or("device") {
            "device" => PoolBudget::Device,
            "resident" => PoolBudget::Resident,
            other => bad("SMALLTIDE_POOL_BUDGET", other, "device | resident"),
        };
        let batched_prefill = match var("SMALLTIDE_PREFILL").as_deref().unwrap_or("batched") {
            "batched" => true,
            "token" => false,
            other => bad("SMALLTIDE_PREFILL", other, "batched | token"),
        };
        Self {
            model_dir: PathBuf::from(var("SMALLTIDE_MODEL_DIR").unwrap_or_else(|| "models/Inkling-2-Mini".into())),
            max_ctx: parse("SMALLTIDE_MAX_CTX", 16_384, "a token count"),
            rescue_policy,
            nvfp4_slots: parse_auto("SMALLTIDE_NVFP4_SLOTS"),
            mxfp8_slots: parse_auto("SMALLTIDE_MXFP8_SLOTS"),
            mxfp8_host,
            pool_budget,
            ram_gb: parse("SMALLTIDE_RAM_GB", 10.0, "GiB"),
            batched_prefill,
            profile: parse::<u8>("SMALLTIDE_PROFILE", 0, "0 or 1") != 0,
            host: var("SMALLTIDE_HOST").unwrap_or_else(|| "0.0.0.0".into()),
            port: parse("SMALLTIDE_PORT", 8000, "a port number"),
        }
    }
}

/// Everything `run` / `serve` need after startup.
pub struct Loaded {
    pub engine: Inkling2Engine<HipBackend>,
    pub tok: BpeTokenizer,
    pub backend: Arc<HipBackend>,
    pub nvfp4_slots: usize,
    pub mxfp8_slots: usize,
}

/// Loads tokenizer + dense weights, sizes and fills the expert pools, and prints the ready line.
pub fn build_engine(s: &Settings) -> Loaded {
    let cfg = Inkling2Config::default();
    let dir = s.model_dir.as_path();
    let t_start = Instant::now();

    let tok = BpeTokenizer::from_file(&dir.join("tokenizer.json")).expect("load tokenizer.json");
    let dense_host = Inkling2DenseHost::load_from_safetensors(&cfg, &dir.join("model-00001-of-00001.safetensors"))
        .expect("load dense weights");
    let pack = StidePackReader::open(&dir.join("experts.stide")).expect("open experts.stide");
    let backend = Arc::new(HipBackend::new_auto().expect("init HipBackend"));
    let dense_gpu = Inkling2DenseGpu::upload(backend.clone(), dense_host).expect("upload dense GPU");
    let info_after_dense = backend.device_info().expect("device_info");

    // Auto-size expert pools from free device memory after dense + KV + headroom.
    let nvfp4_slab = ExpertSlabLayout::for_inkling2(&cfg, DType::NVFP4).aligned_slab_bytes;
    let mxfp8_slab = ExpertSlabLayout::for_inkling2(&cfg, DType::MXFP8).aligned_slab_bytes;
    let est_kv_bytes = 40 * (2 * 8 * cfg.local_window_size * 128 + 2 * 8 * cfg.local_window_size * 4)
        + 8 * (2 * 8 * s.max_ctx * 128 + 2 * 8 * s.max_ctx * 4)
        + 64 * 1024 * 1024;
    let headroom_bytes = 900 * 1024 * 1024usize;
    let scratch_ring_bytes = 8 * (nvfp4_slab + mxfp8_slab);
    let mut avail = info_after_dense
        .free_vram_bytes
        .saturating_sub(est_kv_bytes + headroom_bytes + scratch_ring_bytes);
    let resident_fit = s.pool_budget == PoolBudget::Resident;
    if resident_fit {
        match backend.resident_vram_info() {
            Some((usage, resident)) => {
                // Engine buffers (~KV + scratch + 256 MiB margin) still have to fit after this.
                let margin = std::env::var("SMALLTIDE_RESIDENT_MARGIN_MB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(768) << 20;
                let left = (resident as usize).saturating_sub(usage as usize + est_kv_bytes + scratch_ring_bytes + margin);
                avail = avail.min(left);
                println!("[pools] resident budget: {:.2} GiB left for expert pools", avail as f64 / (1u64 << 30) as f64);
            }
            None => eprintln!("[pools] SMALLTIDE_POOL_BUDGET=resident but no DXGI budget is available; using device free memory"),
        }
    }
    let total_nvfp4 = cfg.mxfp8_start_layer * cfg.n_routed_experts;
    let total_mxfp8 = (cfg.num_physical_layers - cfg.mxfp8_start_layer) * cfg.n_routed_experts;
    let (nvfp4_slots, mxfp8_slots) = match (s.nvfp4_slots, s.mxfp8_slots) {
        (Some(n4), Some(m8)) => (n4.min(total_nvfp4), m8.min(total_mxfp8)),
        (Some(n4), None) => {
            let n4 = n4.min(total_nvfp4);
            (n4, (avail.saturating_sub(n4 * nvfp4_slab) / mxfp8_slab).min(total_mxfp8).max(16))
        }
        (None, Some(m8)) => {
            let m8 = m8.min(total_mxfp8);
            ((avail.saturating_sub(m8 * mxfp8_slab) / nvfp4_slab).min(total_nvfp4).max(1536), m8)
        }
        (None, None) if resident_fit => {
            // Only what really stays on the card: NVFP4 first (loop core is placed first), then
            // MXFP8 with what is left. No floors - missing experts are Tide misses.
            let n4 = (avail / nvfp4_slab).min(total_nvfp4).max(1);
            let m8 = (avail.saturating_sub(n4 * nvfp4_slab) / mxfp8_slab).min(total_mxfp8).max(1);
            (n4, m8)
        }
        (None, None) => {
            // Loop core first (1536), then the rest of NVFP4, then MXFP8 with what is left.
            let n4 = (avail.saturating_sub(128 * mxfp8_slab) / nvfp4_slab).min(total_nvfp4).max(1536);
            let m8 = (avail.saturating_sub(n4 * nvfp4_slab) / mxfp8_slab).min(total_mxfp8).max(16);
            (n4, m8)
        }
    };
    // MXFP8 pool in pinned host memory holding every MXFP8 expert ("auto": when the device pool
    // could not hold them all). Device memory past the resident budget is system memory anyway.
    let mxfp8_host_mapped = match s.mxfp8_host {
        Tri::On => true,
        Tri::Off => false,
        Tri::Auto => mxfp8_slots < total_mxfp8 && !resident_fit,
    };
    let mxfp8_slots = if mxfp8_host_mapped { total_mxfp8 } else { mxfp8_slots };

    let mut tide_cfg = TideConfig::default();
    tide_cfg.pinned_staging_mib = 256;
    tide_cfg.rescue_policy = s.rescue_policy;
    let ram_budget_bytes = (s.ram_gb * (1u64 << 30) as f64) as usize;
    let mut tide = TideExecutor::new(backend.clone(), cfg.clone(), tide_cfg, nvfp4_slots, mxfp8_slots, ram_budget_bytes, Some(pack))
        .expect("create TideExecutor");
    tide.set_mxfp8_host_mapped(mxfp8_host_mapped);

    // Engine buffers (activations, KV, prefill scratch) before the expert pools: on Windows the
    // driver backs allocations beyond the resident budget with system memory, in order.
    let mut engine = Inkling2Engine::new(cfg, dense_gpu, tide, s.max_ctx).expect("create Inkling2Engine");
    engine.tide.allocate_pools().expect("allocate expert pools");
    let preload_ops = engine.tide.scheduler.plan_pre_prefill(None);
    engine.tide.execute_transfers(&preload_ops, true, false).expect("Mode 1 pre-prefill");
    backend.sync_stream(engine.tide.copy_stream).expect("sync prefill DMA");

    let gib = (1u64 << 30) as f64;
    if let Some((usage, resident)) = backend.resident_vram_info() {
        println!(
            "[VRAM residency] process usage {:.2} GiB vs resident budget {:.2} GiB (DXGI){}",
            usage as f64 / gib,
            resident as f64 / gib,
            if usage > resident {
                format!(
                    " -> ~{:.2} GiB of 'VRAM' is backed by system memory (read over PCIe); close other GPU apps or lower SMALLTIDE_NVFP4_SLOTS",
                    (usage - resident) as f64 / gib
                )
            } else {
                String::new()
            }
        );
    }
    let info = backend.device_info().expect("device_info ready");
    let (pre, core, coda) = engine.tide.scheduler.cache.count_vram_resident();
    println!(
        "[Ready in {:.2}s] VRAM free: {:.2}/{:.2} GiB | KV({} ctx): {:.2} GiB | Experts resident: pre={}, core={}/1536, coda={} (slots nvfp4={}, mxfp8={}{}) | RAM warm: {} | Rescue: {:?}",
        t_start.elapsed().as_secs_f64(),
        info.free_vram_bytes as f64 / gib,
        info.total_vram_bytes as f64 / gib,
        s.max_ctx,
        engine.kv_vram_bytes() as f64 / gib,
        pre,
        core,
        coda,
        nvfp4_slots,
        mxfp8_slots,
        if mxfp8_host_mapped { " host-mapped" } else { "" },
        engine.tide.scheduler.cache.count_ram_resident(),
        s.rescue_policy,
    );
    Loaded { engine, tok, backend, nvfp4_slots, mxfp8_slots }
}

/// Prints where configuration came from (once, at startup).
pub fn describe_source(loaded: &Option<PathBuf>) {
    match loaded {
        Some(p) => println!("[config] loaded {}", p.display()),
        None => println!("[config] no .env found (using defaults / process environment; see .env.example)"),
    }
}
