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
use smalltide_tide::{PriorsStore, TideExecutor, TidePriors};
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
    /// Routing priors file (`None` = priors disabled).
    pub priors_path: Option<PathBuf>,
    pub priors_update: bool,
    pub prior_weight: f32,
    pub priors_newest_share: f32,
    /// Load the DSpark drafter (`mtp.safetensors`) for speculative decoding (greedy requests).
    pub spec: bool,
    /// Concurrent sequences served by `serve` (lane 0 has `max_ctx`, the others `lane_ctx`).
    pub lanes: usize,
    pub lane_ctx: usize,
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
        let model_dir = PathBuf::from(var("SMALLTIDE_MODEL_DIR").unwrap_or_else(|| "models/Inkling-2-Mini".into()));
        let priors_path = match var("SMALLTIDE_PRIORS").as_deref() {
            Some("off") | Some("0") => None,
            Some(path) => Some(PathBuf::from(path)),
            None => Some(model_dir.join("tide_priors.json")),
        };
        Self {
            lanes: parse("SMALLTIDE_LANES", 4usize, "a lane count >= 1").max(1),
            lane_ctx: parse("SMALLTIDE_LANE_CTX", 8192usize, "a token count"),
            priors_path,
            priors_update: parse::<u8>("SMALLTIDE_PRIORS_UPDATE", 1, "0 or 1") != 0,
            prior_weight: parse("SMALLTIDE_PRIOR_WEIGHT", 1.0f32, "a number >= 0"),
            priors_newest_share: parse("SMALLTIDE_PRIORS_NEWEST", 0.1f32, "a number in [0, 0.5)"),
            spec: parse::<u8>("SMALLTIDE_SPEC", 0, "0 or 1") != 0,
            model_dir,
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
pub fn build_engine(s: &Settings, lanes: usize) -> Loaded {
    let cfg = Inkling2Config::default();
    let dir = s.model_dir.as_path();
    let t_start = Instant::now();

    let tok = BpeTokenizer::from_file(&dir.join("tokenizer.json")).expect("load tokenizer.json");
    let dense_host = Inkling2DenseHost::load_from_safetensors(&cfg, &dir.join("model-00001-of-00001.safetensors"))
        .expect("load dense weights");
    let pack = StidePackReader::open(&dir.join("experts.stide")).expect("open experts.stide");
    let backend = Arc::new(HipBackend::new_auto().expect("init HipBackend"));
    let dense_gpu = Inkling2DenseGpu::upload(backend.clone(), dense_host).expect("upload dense GPU");
    // The drafter goes before the VRAM measurement so the pool budget accounts for it.
    let drafter = if s.spec {
        let path = dir.join("mtp.safetensors");
        let t = Instant::now();
        let d = smalltide_model_inkling2::drafter::Drafter::load(backend.clone(), &path, &dense_gpu, cfg.unpadded_vocab_size, s.max_ctx,
            smalltide_model_inkling2::engine::PREFILL_CHUNK).expect("load drafter (mtp.safetensors)");
        println!("[spec] DSpark drafter loaded ({:.0} MiB VRAM, {:.1}s)", d.vram_bytes as f64 / (1u64 << 20) as f64, t.elapsed().as_secs_f64());
        Some(d)
    } else {
        None
    };
    let info_after_dense = backend.device_info().expect("device_info");

    // Auto-size expert pools from free device memory after dense + KV + headroom.
    let nvfp4_slab = ExpertSlabLayout::for_inkling2(&cfg, DType::NVFP4).aligned_slab_bytes;
    let mxfp8_slab = ExpertSlabLayout::for_inkling2(&cfg, DType::MXFP8).aligned_slab_bytes;
    let est_kv_bytes = 40 * (2 * 8 * cfg.local_window_size * 128 + 2 * 8 * cfg.local_window_size * 4)
        + 8 * (2 * 8 * s.max_ctx * 128 + 2 * 8 * s.max_ctx * 4)
        + 64 * 1024 * 1024;
    // Extra lanes: their own KV (local rings + global KV of `lane_ctx`), allocated before the pools.
    let extra_lanes = lanes.max(1) - 1;
    let lane_ctx = s.lane_ctx.min(s.max_ctx).max(cfg.local_window_size);
    let est_kv_bytes = est_kv_bytes
        + extra_lanes * (40 * (2 * 8 * (cfg.local_window_size + 128) * 128 + 2 * 8 * (cfg.local_window_size + 128) * 4)
            + 8 * (2 * 8 * lane_ctx * 128 + 2 * 8 * lane_ctx * 4));
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
    let priors = s.priors_path.as_ref().map(|path| match TidePriors::load_from_file(path) {
        Ok(p) => {
            println!("[priors] loaded {} ({} requests)", path.display(), p.sessions_recorded);
            p
        }
        Err(e) => {
            if path.exists() {
                eprintln!("[priors] ignoring {}: {e}", path.display());
            } else {
                println!("[priors] none yet at {}; collecting from this session", path.display());
            }
            TidePriors::new("Inkling-2-Mini", cfg.num_physical_layers, cfg.n_routed_experts)
        }
    });
    let have_priors = priors.as_ref().is_some_and(|p| !p.is_empty());
    let total_nvfp4 = cfg.mxfp8_start_layer * cfg.n_routed_experts;
    let total_mxfp8 = (cfg.num_physical_layers - cfg.mxfp8_start_layer) * cfg.n_routed_experts;
    // Resident-budget split of `bytes` between the pools: value per byte with priors, otherwise
    // NVFP4 first (loop core is placed first), then MXFP8. No floors: missing experts are misses.
    let split_cfg = cfg.clone();
    let resident_split = |bytes: usize| -> (usize, usize) {
        if let Some(p) = priors.as_ref().filter(|p| !p.is_empty()) {
            let (n4, m8) = p.plan_pool_split(&split_cfg, bytes, nvfp4_slab, mxfp8_slab);
            (n4.max(1), m8.max(1))
        } else {
            let n4 = (bytes / nvfp4_slab).min(total_nvfp4).max(1);
            let m8 = (bytes.saturating_sub(n4 * nvfp4_slab) / mxfp8_slab).min(total_mxfp8).max(1);
            (n4, m8)
        }
    };
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
        (None, None) if resident_fit => resident_split(avail),
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
    tide_cfg.max_async_vram_promotions_per_token = parse("SMALLTIDE_ASYNC_PER_TOKEN", tide_cfg.max_async_vram_promotions_per_token, "a count per token");
    let ram_budget_bytes = (s.ram_gb * (1u64 << 30) as f64) as usize;
    let mut tide = TideExecutor::new(backend.clone(), cfg.clone(), tide_cfg, nvfp4_slots, mxfp8_slots, ram_budget_bytes, Some(pack))
        .expect("create TideExecutor");
    tide.set_mxfp8_host_mapped(mxfp8_host_mapped);

    // Engine buffers (activations, KV, prefill scratch) before the expert pools: on Windows the
    // driver backs allocations beyond the resident budget with system memory, in order.
    let mut engine = Inkling2Engine::new(cfg, dense_gpu, tide, s.max_ctx).expect("create Inkling2Engine");
    if let Some(d) = drafter {
        engine.attach_drafter(d).expect("attach drafter");
    }
    if extra_lanes > 0 {
        engine.add_lanes(extra_lanes, lane_ctx).expect("allocate lanes");
        println!("[lanes] {} concurrent sequences (lane 0: {} ctx, lanes 1..: {} ctx)", lanes, s.max_ctx, lane_ctx);
    }
    engine.preallocate_rows_buffers().expect("allocate rows buffers");
    // Re-measure now that every engine buffer exists (lanes, rows scratch, drafter): pools that
    // would push the process past the resident budget make Windows demote *some* allocation
    // (dense weights, KV, scratch) to system memory - shrink them before they are allocated.
    if resident_fit && s.nvfp4_slots.is_none() && s.mxfp8_slots.is_none() {
        if let Some((usage, resident)) = backend.resident_vram_info() {
            let tail = std::env::var("SMALLTIDE_RESIDENT_TAIL_MB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(384) << 20;
            let left = (resident as usize).saturating_sub(usage as usize + scratch_ring_bytes + tail);
            let planned = nvfp4_slots * nvfp4_slab + mxfp8_slots * mxfp8_slab;
            if planned > left {
                let (n4, m8) = resident_split(left);
                engine.tide.resize_pools(n4, m8).expect("shrink expert pools");
                println!("[pools] re-measured after engine buffers: {:.2} GiB left -> slots nvfp4 {} -> {}, mxfp8 {} -> {}",
                    left as f64 / (1u64 << 30) as f64, nvfp4_slots, n4, mxfp8_slots, m8);
            }
        }
    }
    engine.tide.allocate_pools().expect("allocate expert pools");
    backend.set_touch_on_alloc(false);
    // Measure instead of trusting the budget estimate: only the part of each pool that the OS
    // keeps in VRAM is used (Windows silently backs the excess with system memory).
    let (nvfp4_slots, mxfp8_slots) = if resident_fit || std::env::var_os("SMALLTIDE_POOL_CHECK").is_some() {
        let (n4, m8, line) = engine.tide.fit_pools_to_resident().expect("pool residency check");
        println!("{line}");
        (n4, m8)
    } else {
        (nvfp4_slots, mxfp8_slots)
    };
    let _ = &resident_split;
    if let Some(p) = priors {
        engine.tide.scheduler.install_priors(PriorsStore {
            priors: p,
            path: s.priors_path.clone(),
            newest_share: s.priors_newest_share,
            prior_weight: s.prior_weight,
            update: s.priors_update,
        });
    }
    engine.prefill_min_share = parse("SMALLTIDE_PREFILL_MIN_SHARE", 1.0f32, "tokens of routing share");
    engine.tide.scheduler.rescue_budget = parse("SMALLTIDE_RESCUE_BUDGET", 4usize, "rescues per token");
    engine.prefill_chunk_swaps = parse("SMALLTIDE_PREFILL_CHUNK_SWAPS", 96usize, "a swap count");
    engine.anticipate_swaps = parse("SMALLTIDE_ANTICIPATE_SWAPS", 256usize, "a swap count");
    engine.anticipate_wait = parse::<u8>("SMALLTIDE_ANTICIPATE_WAIT", 0, "0 or 1") != 0;
    engine.tide.scheduler.async_hysteresis = parse("SMALLTIDE_ASYNC_HYSTERESIS", 1.25f64, "a factor >= 1");
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
    if have_priors {
        let sched = &engine.tide.scheduler;
        let p = &sched.priors_store.as_ref().unwrap().priors;
        let cov = p.coverage(&cfg_for_log(), |i| {
            let id = sched.cache.expert_from_flat(i);
            !sched.cache.entry(id).vram_slot.is_missing()
        });
        println!("[priors] resident set covers {:.1}% of the expected routing mass per token", cov * 100.0);
    }
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

fn cfg_for_log() -> Inkling2Config {
    Inkling2Config::default()
}

/// Prints where configuration came from (once, at startup).
pub fn describe_source(loaded: &Option<PathBuf>) {
    match loaded {
        Some(p) => println!("[config] loaded {}", p.display()),
        None => println!("[config] no .env found (using defaults / process environment; see .env.example)"),
    }
}
