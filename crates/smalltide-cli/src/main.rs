mod chat_format;
mod prefix_cache;
mod server;
mod settings;

use smalltide_backend_cpu as cpu;
use smalltide_backend_hip::{HipBackend, HipRuntime};
use smalltide_core::{ComputeBackend, DType, Inkling2Config, TideConfig};
use smalltide_io::dense_loader::Inkling2DenseHost;
use smalltide_io::safetensors::inspect_safetensors_header;
use smalltide_io::stide_pack::{ExpertSlabHeader, ExpertSlabLayout, StidePackReader};
use smalltide_io::tokenizer::BpeTokenizer;
use smalltide_io::win_direct_io::{align_up, AlignedBuffer, DirectFile, SECTOR_ALIGN};
use smalltide_model_inkling2::engine::SamplingParams;
use smalltide_model_inkling2::weights::Inkling2DenseGpu;
use smalltide_tide::TideExecutor;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let env_file = settings::load_dotenv();
    let args: Vec<String> = env::args().collect();
    if matches!(args.get(1).map(String::as_str), Some("run") | Some("serve")) {
        settings::describe_source(&env_file);
    }
    match args.get(1).map(String::as_str) {
        Some("probe") => run_probe(),
        Some("vram-probe") => run_vram_probe(),
        Some("bench-gpu") => run_bench_gpu(),
        Some("bench-nvme") => run_bench_nvme(),
        Some("load-test") => {
            let dir = args
                .get(2)
                .map(String::as_str)
                .unwrap_or(r"models\Inkling-2-Mini");
            run_load_test(Path::new(dir));
        }
        Some("run") => {
            run_generate(&args[2..]);
        }
        Some("serve") => {
            server::run_serve(&args[2..]);
        }
        Some("pack") => {
            let (Some(src), Some(dst)) = (args.get(2), args.get(3)) else {
                eprintln!("Usage: smalltide-cli pack <model.safetensors> <out.stide>");
                std::process::exit(1);
            };
            let cfg = Inkling2Config::default();
            println!("Packing Inkling-2-Mini experts from {} -> {} ...", src, dst);
            let t0 = Instant::now();
            smalltide_io::stide_pack::StidePackReader::build_from_safetensors(
                &cfg,
                Path::new(src),
                Path::new(dst),
            )
            .expect("pack .stide");
            println!("Packed all 3,072 experts in {:.2}s!", t0.elapsed().as_secs_f64());
        }
        Some("render-chat") => {
            // Dev tool: `render-chat <cases.json>`; each case {messages, tools?, reasoning_effort?}.
            let Some(input) = args.get(2) else {
                eprintln!("Usage: smalltide-cli render-chat <cases.json>");
                std::process::exit(1);
            };
            let text = std::fs::read_to_string(input).expect("read input");
            let cases = chat_format::parse_json(&text).expect("parse cases");
            let chat_format::OJson::Arr(cases) = cases else { panic!("cases must be a list") };
            let mut out = Vec::new();
            for c in &cases {
                let msgs = match c.get("messages") {
                    Some(chat_format::OJson::Arr(m)) => m.clone(),
                    _ => Vec::new(),
                };
                let r = chat_format::render_chat(&msgs, c.get("tools"), c.get("reasoning_effort"));
                out.push(match r {
                    Ok(r) => serde_json::json!({ "ok": r.prompt }),
                    Err(e) => serde_json::json!({ "err": e }),
                });
            }
            println!("{}", serde_json::to_string(&out).unwrap());
        }
        Some("tokenize") => {
            // Dev tool: `tokenize <tokenizer.json> <in.txt>` -> JSON token ids of the whole file.
            let (Some(tj), Some(input)) = (args.get(2), args.get(3)) else {
                eprintln!("Usage: smalltide-cli tokenize <tokenizer.json> <in.txt>");
                std::process::exit(1);
            };
            let tok = BpeTokenizer::from_file(Path::new(tj)).expect("load tokenizer.json");
            let text = std::fs::read_to_string(input).expect("read input");
            let ids = tok.encode(&text);
            println!("{}", serde_json::to_string(&ids).unwrap());
        }
        Some("inspect") => {
            let Some(path) = args.get(2) else {
                eprintln!("Usage: smalltide-cli inspect <path-to.safetensors> [substring-filter]");
                std::process::exit(1);
            };
            let filter = args.get(3).map(String::as_str);
            run_inspect(Path::new(path), filter);
        }
        _ => {
            println!("SmallTide CLI");
            println!("Commands:");
            println!("  probe                         Probe AMD HIP GPU + Stage 1 H2D DMA microbenchmark");
            println!("  bench-gpu                     Verify gfx1201 kernels vs CPU + benchmark BW / hipGraph / overlap");
            println!("  bench-nvme                    Run Stage 2 Win32 NO_BUFFERING random expert read benchmark");
            println!("  pack <in.safetensors> <out.stide>  Pack 3,072 experts into 4KB-aligned .stide store");
            println!("  load-test [model-dir]         Verify real tokenizer + FP8 dense load + Mode 1 Tide pre-prefill on GPU");
            println!("  run -p <prompt> [-n N] [--temp T] [--effort E] [--raw]   Generate once (config: .env)");
            println!("  serve                         Serve OpenAI API as inkling-2-mini (config: .env)");
            println!("  vram-probe                    Measure resident vs system-memory-backed VRAM");
            println!("Configuration: .env in the working directory (or SMALLTIDE_ENV=path); see .env.example.");
            println!("  inspect <file.safetensors>    Inspect tensor names, dtypes, and shapes");
        }
    }
}

fn run_probe() {
    let hip = HipRuntime::load().expect("load HIP runtime");
    let dev = hip.select_device().expect("select HIP device");
    let info = hip.probe_device(dev).expect("probe HIP device");
    println!("Loaded HIP DLL : {}", info.loaded_dll);
    println!("HIP Runtime    : {}", info.runtime_version);
    println!("HIP Driver     : {}", info.driver_version);
    println!("Device Count   : {}", info.device_count);
    println!("Device[{dev}]      : {}", info.device_name);
    println!("Embedded HSACO : {} bytes", info.hsaco_bytes);
    println!(
        "VRAM           : {:.2} GiB free / {:.2} GiB total",
        info.free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        info.total_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    for (label, bytes) in [
        ("NVFP4 expert (4.27 MiB)", 4_478_976usize),
        ("MXFP8 expert (7.83 MiB)", 8_211_456usize),
    ] {
        let res = hip.bench_h2d_dma(bytes, 100).expect("DMA bench");
        println!(
            "DMA H2D [{}]: {:.3} ms/expert ({:.2} GB/s)",
            label, res.avg_ms, res.bandwidth_gb_s
        );
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn run_bench_gpu() {
    let hip = HipRuntime::load().expect("load HIP runtime");
    let dev = hip.select_device().expect("select HIP device");
    let info = hip.probe_device(dev).expect("probe HIP device");
    println!(
        "=== SmallTide Phase 1 gfx1201 Kernel Verify & Benchmark ({}) ===",
        info.device_name
    );
    println!("Embedded HSACO: {} bytes", info.hsaco_bytes);
    let kernels = hip.load_core_kernels().expect("load embedded .hsaco kernels");

    // --- 1. Numerical Parity Checks (GPU vs CPU Reference) ---
    let k_cols = 2304usize;
    let n_rows = 256usize;
    let x: Vec<f32> = (0..k_cols).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    let w_norm: Vec<f32> = (0..k_cols).map(|i| 0.8 + ((i as f32) * 0.07).cos() * 0.2).collect();

    let cpu_rms = cpu::rmsnorm_f32(&x, &w_norm, 1e-6);
    let gpu_rms = hip.run_rmsnorm_f32(&kernels, &x, &w_norm, 1e-6).expect("gpu rmsnorm");
    let err_rms = max_abs_diff(&cpu_rms, &gpu_rms);
    println!("[Parity] RMSNorm (2304)             max_abs_diff = {:.2e}", err_rms);
    assert!(err_rms < 1e-5, "RMSNorm parity failed: {err_rms}");

    // FP8 E4M3 GEMV parity (avoid 0x7F/0xFF NaN encodings)
    let w_fp8: Vec<u8> = (0..(n_rows * k_cols))
        .map(|i| {
            let b = ((i.wrapping_mul(37) ^ (i >> 3)) & 0xFF) as u8;
            if (b & 0x7F) == 0x7F { 0x38 } else { b }
        })
        .collect();
    let row_scale: Vec<f32> = (0..n_rows).map(|r| 0.01 + (r as f32) * 0.0002).collect();
    let cpu_fp8 = cpu::gemv_fp8_e4m3_f32(&x, &w_fp8, &row_scale, n_rows, k_cols);
    let gpu_fp8 = hip
        .run_gemv_fp8_e4m3_f32(&kernels, &x, &w_fp8, &row_scale, n_rows, k_cols)
        .expect("gpu fp8");
    let err_fp8 = max_abs_diff(&cpu_fp8, &gpu_fp8);
    println!("[Parity] FP8 E4M3 GEMV [256x2304]   max_abs_diff = {:.2e}", err_fp8);
    assert!(err_fp8 < 1e-3, "FP8 parity failed: {err_fp8}");

    // NVFP4 GEMV parity
    let w_nvfp4: Vec<u8> = (0..(n_rows * (k_cols / 2)))
        .map(|i| ((i.wrapping_mul(53) ^ (i >> 2)) & 0xFF) as u8)
        .collect();
    let s_nvfp4: Vec<u8> = (0..(n_rows * (k_cols / 16)))
        .map(|i| 0x28 + ((i % 24) as u8)) // valid positive FP8 E4M3 scales
        .collect();
    let cpu_nvfp4 = cpu::gemv_nvfp4_f32(&x, &w_nvfp4, &s_nvfp4, 0.25, n_rows, k_cols);
    let gpu_nvfp4 = hip
        .run_gemv_nvfp4_f32(&kernels, &x, &w_nvfp4, &s_nvfp4, 0.25, n_rows, k_cols)
        .expect("gpu nvfp4");
    let err_nvfp4 = max_abs_diff(&cpu_nvfp4, &gpu_nvfp4);
    println!("[Parity] NVFP4 GEMV    [256x2304]   max_abs_diff = {:.2e}", err_nvfp4);
    assert!(err_nvfp4 < 1e-3, "NVFP4 parity failed: {err_nvfp4}");

    // MXFP8 GEMV parity (verified on-disk [N, K/32] UE8M0 scales)
    let s_mxfp8: Vec<u8> = (0..(n_rows * (k_cols / 32)))
        .map(|i| 120u8 + ((i % 12) as u8)) // UE8M0 scales around 2^-7..2^4
        .collect();
    let cpu_mxfp8 = cpu::gemv_mxfp8_f32(&x, &w_fp8, &s_mxfp8, n_rows, k_cols);
    let gpu_mxfp8 = hip
        .run_gemv_mxfp8_f32(&kernels, &x, &w_fp8, &s_mxfp8, n_rows, k_cols)
        .expect("gpu mxfp8");
    let err_mxfp8 = max_abs_diff(&cpu_mxfp8, &gpu_mxfp8);
    println!("[Parity] MXFP8 GEMV    [256x2304]   max_abs_diff = {:.2e}", err_mxfp8);
    assert!(err_mxfp8 < 1e-2, "MXFP8 parity failed: {err_mxfp8}");

    // --- 2. Windows WDDM Launch Latency: Raw vs HIP Graph (240 kernels = 48 layers x 5 ops) ---
    let g = hip.bench_graph_launch(&kernels).expect("graph bench");
    println!(
        "[WDDM Launch, {} ops] Raw: {:.3} ms | HIP Graph: {:.3} ms ({:.2}x faster)",
        g.kernels_per_iter, g.raw_ms_per_iter, g.graph_ms_per_iter, g.speedup
    );

    // --- 3. Decode Kernel Effective GDDR6 Bandwidth + Async DMA Overlap ---
    let (bw_results, ov) = hip.bench_decode_kernels(&kernels).expect("bw bench");
    for r in bw_results {
        let pct_peak = (r.effective_gb_s / 640.0) * 100.0;
        println!(
            "[GDDR6 BW] {:<45} : {:6.3} ms | {:6.1} GB/s ({:.1}% of 640 GB/s)",
            r.label, r.avg_ms, r.effective_gb_s, pct_peak
        );
    }
    println!(
        "[Async Overlap] Compute: {:.3} ms | DMA (2x NVFP4): {:.3} ms | Concurrent: {:.3} ms (overlap eff: {:.1}%)",
        ov.compute_only_ms, ov.dma_only_ms, ov.overlapped_ms, ov.overlap_efficiency_pct
    );

    // --- 4. Indirect Top-8 MoE SwiGLU Parity & Latency (NVFP4 + MXFP8 + --no-rescue skip!) ---
    let cfg = Inkling2Config::default();
    for (label, dtype, skip_tail) in [
        ("Indirect Top-8 NVFP4 (8/8 hits)", DType::NVFP4, false),
        ("Indirect Top-8 NVFP4 (6/8 hits, 2 skipped+renorm)", DType::NVFP4, true),
        ("Indirect Top-8 MXFP8 (8/8 hits)", DType::MXFP8, false),
    ] {
        let layout = ExpertSlabLayout::for_inkling2(&cfg, dtype);
        let n_slots = 8usize;
        let mut pool = vec![0u8; n_slots * layout.aligned_slab_bytes];
        for s in 0..n_slots {
            let base = s * layout.aligned_slab_bytes;
            let hdr = ExpertSlabHeader {
                w13_outer_scale: 0.125,
                w2_outer_scale: 0.125,
                w13_weight_offset: layout.w13_weight_offset as u32,
                w13_scale_offset: layout.w13_scale_offset as u32,
                w2_weight_offset: layout.w2_weight_offset as u32,
                w2_scale_offset: layout.w2_scale_offset as u32,
                dtype_tag: if dtype == DType::NVFP4 { 0 } else { 1 },
                _reserved: [0; 9],
            };
            let hdr_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    (&hdr as *const ExpertSlabHeader) as *const u8,
                    std::mem::size_of::<ExpertSlabHeader>(),
                )
            };
            pool[base..base + 64].copy_from_slice(hdr_bytes);
            // Fill deterministic weights/scales (avoid FP8 NaN 0x7F/0xFF)
            for i in 0..layout.w13_weight_bytes {
                let mut b = ((i.wrapping_mul(31) ^ (s * 17)) & 0xFF) as u8;
                if dtype == DType::MXFP8 && (b & 0x7F) == 0x7F { b = 0x38; }
                pool[base + layout.w13_weight_offset + i] = b;
            }
            for i in 0..layout.w13_scale_bytes {
                pool[base + layout.w13_scale_offset + i] =
                    if dtype == DType::NVFP4 { 0x2C + ((i + s) % 12) as u8 } else { 121 + ((i + s) % 6) as u8 };
            }
            for i in 0..layout.w2_weight_bytes {
                let mut b = ((i.wrapping_mul(43) ^ (s * 29)) & 0xFF) as u8;
                if dtype == DType::MXFP8 && (b & 0x7F) == 0x7F { b = 0x38; }
                pool[base + layout.w2_weight_offset + i] = b;
            }
            for i in 0..layout.w2_scale_bytes {
                pool[base + layout.w2_scale_offset + i] =
                    if dtype == DType::NVFP4 { 0x2C + ((i + s) % 12) as u8 } else { 121 + ((i + s) % 6) as u8 };
            }
        }

        let mut slot_table = vec![u32::MAX; cfg.n_routed_experts];
        let topk_ids = [5u32, 17, 33, 49, 64, 81, 102, 120];
        let topk_weights = [0.22f32, 0.18, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05];
        for (i, &id) in topk_ids.iter().enumerate() {
            if skip_tail && i >= 6 {
                slot_table[id as usize] = u32::MAX; // Simulate --no-rescue miss!
            } else {
                slot_table[id as usize] = i as u32;
            }
        }

        let cpu_moe = cpu::moe_swiglu_indirect_ref(
            &x,
            &pool,
            &slot_table,
            &topk_ids,
            &topk_weights,
            layout.aligned_slab_bytes,
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.mlp_gate_softcap,
            cfg.mlp_up_softcap,
        );
        let (gpu_moe, moe_ms) = hip
            .run_and_bench_moe_indirect(
                &kernels,
                &x,
                &pool,
                &slot_table,
                &topk_ids,
                &topk_weights,
                layout.aligned_slab_bytes,
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.mlp_gate_softcap,
                cfg.mlp_up_softcap,
                100,
            )
            .expect("gpu moe");
        let err = max_abs_diff(&cpu_moe, &gpu_moe);
        println!(
            "[MoE Parity+Perf] {:<48} : {:6.3} ms/layer | max_abs_diff = {:.2e}",
            label, moe_ms, err
        );
        assert!(err < 1e-2, "MoE parity failed for {label}: {err}");
    }

    // --- 5. FP8 KV Split-K FlashDecoding Benchmark (1K Local + 32K + 256K Global!) ---
    for (label, seq_len, splits) in [
        ("Local 1K  FP8 KV Attn (1 layer)", 1_024usize, 4u32),
        ("Global 32K FP8 KV Attn (1 layer)", 32_768usize, 32u32),
        ("Global 256K FP8 KV Attn (1 layer)", 262_144usize, 64u32),
    ] {
        let (ms, gbs) = hip
            .bench_flash_decode_fp8(&kernels, seq_len, splits, 20)
            .expect("flash decode bench");
        println!(
            "[FP8 Attn] {:<35} : {:6.3} ms/layer | {:6.1} GB/s ({:.1}% of 640 GB/s)",
            label,
            ms,
            gbs,
            (gbs / 640.0) * 100.0
        );
    }

    hip.unload_kernels(kernels);
}

fn run_inspect(path: &Path, filter: Option<&str>) {
    match inspect_safetensors_header(path) {
        Ok(hdr) => {
            println!(
                "Safetensors header: {} bytes, {} tensors",
                hdr.header_len,
                hdr.tensors.len()
            );
            let mut shown = 0usize;
            for (name, entry) in &hdr.tensors {
                if let Some(f) = filter {
                    if !name.contains(f) {
                        continue;
                    }
                } else if shown >= 40 {
                    break;
                }
                let bytes = entry.data_offsets[1].saturating_sub(entry.data_offsets[0]);
                println!(
                    "  {:<62} {:<8} {:?} ({} bytes)",
                    name, entry.dtype, entry.shape, bytes
                );
                shown += 1;
            }
            if filter.is_none() && hdr.tensors.len() > shown {
                println!("  ... and {} more tensors (pass a substring filter to search)", hdr.tensors.len() - shown);
            }
        }
        Err(e) => eprintln!("Failed to inspect {}: {e}", path.display()),
    }
}

fn run_bench_nvme() {
    let temp_path = env::temp_dir().join("smalltide_nvme_bench.tmp");
    let file_size_bytes: usize = 256 * 1024 * 1024;
    println!(
        "Creating 256 MiB aligned benchmark file at {} ...",
        temp_path.display()
    );
    {
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&temp_path)
            .expect("create temp bench file");
        let block = vec![0xA5u8; 4 * 1024 * 1024];
        for _ in 0..(file_size_bytes / block.len()) {
            f.write_all(&block).expect("write bench block");
        }
        f.sync_all().expect("sync bench file");
    }

    let df = DirectFile::open_read(&temp_path).expect("open NO_BUFFERING file");
    for (label, raw_bytes) in [
        ("NVFP4 expert (4.27 MiB aligned)", 4_478_976usize),
        ("MXFP8 expert (7.83 MiB aligned)", 8_211_456usize),
    ] {
        let read_len = align_up(raw_bytes, SECTOR_ALIGN);
        let mut buf = AlignedBuffer::alloc(read_len).expect("alloc aligned buf");
        let max_slots = file_size_bytes / read_len;
        let iterations = 64usize;

        let start = Instant::now();
        let mut state: usize = 0x9E37_79B9;
        for _ in 0..iterations {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let slot = (state >> 16) % max_slots;
            let offset = (slot * read_len) as u64;
            let got = df
                .read_aligned_at(offset, &mut buf, read_len)
                .expect("direct read");
            assert_eq!(got, read_len);
        }
        let secs = start.elapsed().as_secs_f64();
        let avg_ms = (secs * 1000.0) / (iterations as f64);
        let gb_s = (read_len as f64 * iterations as f64 / 1e9) / secs;
        println!("NVMe Direct [{}]: {:.3} ms/expert ({:.2} GB/s)", label, avg_ms, gb_s);
    }

    let _ = std::fs::remove_file(&temp_path);
}

/// Allocates VRAM in 512 MiB chunks and measures read bandwidth of each, revealing where the
/// driver silently backs "device" allocations with system memory (WDDM over-commit).
fn run_vram_probe() {
    let backend = HipBackend::new_auto().expect("init HipBackend");
    let s = backend.create_stream(smalltide_core::QueueKind::Compute).expect("stream");
    let chunk = 512usize << 20;
    let scratch = backend.alloc_device(chunk).expect("scratch");
    let mut bufs = Vec::new();
    #[cfg(windows)]
    let _wddm = {
        let a = smalltide_backend_hip::wddm::Adapter::open_amd();
        if let Some(a) = &a {
            let gib = (1u64 << 30) as f64;
            if let Some(i) = a.local_info() {
                println!("DXGI local: budget {:.2} GiB, usage {:.2}, available-for-reservation {:.2}",
                    i.budget as f64 / gib, i.current_usage as f64 / gib, i.available_for_reservation as f64 / gib);
                let want = std::env::var("SMALLTIDE_RESERVE_GB").ok().and_then(|v| v.parse::<f64>().ok())
                    .map(|g| (g * gib) as u64).unwrap_or(i.available_for_reservation);
                if want > 0 {
                    let ok = a.reserve_local(want);
                    println!("SetVideoMemoryReservation({:.2} GiB) -> {}", want as f64 / gib, if ok { "ok" } else { "failed" });
                }
                if let Some(i) = a.local_info() {
                    println!("DXGI local after: reserved {:.2} GiB", i.current_reservation as f64 / gib);
                }
            }
        }
        a
    };
    let info0 = backend.device_info().expect("info");
    println!("free before: {:.2} GiB", info0.free_vram_bytes as f64 / (1u64 << 30) as f64);
    if let Some(hm_gb) = std::env::var("SMALLTIDE_PROBE_HOSTMAPPED_GB").ok().and_then(|v| v.parse::<usize>().ok()) {
        let (dbuf, pbuf) = backend.alloc_host_mapped(hm_gb << 30).expect("host-mapped alloc");
        backend.zero_f32(dbuf, (hm_gb << 30) / 4, s).expect("zero");
        backend.sync_stream(s).expect("sync");
        for w in 0..(hm_gb * 2) {
            let t0 = Instant::now();
            for _ in 0..4 {
                backend.vec_copy(scratch, dbuf.slice(w * chunk, chunk), chunk / 4, s).expect("copy");
            }
            backend.sync_stream(s).expect("sync");
            let gbs = (4 * chunk) as f64 / t0.elapsed().as_secs_f64() / 1e9;
            println!("host-mapped window {w:2}: read {gbs:7.1} GB/s");
        }
        let _ = backend.free_device(dbuf);
        let _ = backend.free_pinned(pbuf);
        return;
    }
    if let Some(big_gb) = std::env::var("SMALLTIDE_PROBE_BIG_GB").ok().and_then(|v| v.parse::<usize>().ok()) {
        let big = backend.alloc_device(big_gb << 30).expect("big alloc");
        backend.zero_f32(big, (big_gb << 30) / 4, s).expect("zero");
        backend.sync_stream(s).expect("sync");
        for w in 0..(big_gb * 2) {
            let t0 = Instant::now();
            for _ in 0..4 {
                backend.vec_copy(scratch, big.slice(w * chunk, chunk), chunk / 4, s).expect("copy");
            }
            backend.sync_stream(s).expect("sync");
            let gbs = (4 * chunk) as f64 / t0.elapsed().as_secs_f64() / 1e9;
            println!("big alloc window {w:2} ({:5.1} GiB offset): read {gbs:7.1} GB/s", w as f64 * 0.5);
        }
        let _ = backend.free_device(big);
        return;
    }
    #[cfg(windows)]
    let d3d = if std::env::var_os("SMALLTIDE_PROBE_D3D12").is_some() {
        let dev = _wddm.as_ref().and_then(|a| smalltide_backend_hip::d3d12::D3d12Device::create(a.raw()));
        println!("D3D12 device: {}", if dev.is_some() { "ok" } else { "FAILED" });
        dev
    } else {
        None
    };
    #[cfg(windows)]
    let mut d3d_bufs = Vec::new();
    let n_chunks = std::env::var("SMALLTIDE_PROBE_CHUNKS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(30);
    for i in 0..n_chunks {
        #[cfg(windows)]
        let b = if let Some(dev) = &d3d {
            match dev.alloc_shared_buffer(chunk as u64).map_err(|e| e).and_then(|db| {
                let r = backend.import_d3d12_buffer(&db).map_err(|e| format!("{e}"));
                d3d_bufs.push(db);
                r
            }) {
                Ok(b) => b,
                Err(e) => { println!("d3d12 alloc failed at chunk {i}: {e}"); break }
            }
        } else {
            let Ok(b) = backend.alloc_device(chunk) else { println!("alloc failed at chunk {i}"); break };
            b
        };
        #[cfg(not(windows))]
        let Ok(b) = backend.alloc_device(chunk) else { println!("alloc failed at chunk {i}"); break };
        let _ = backend.zero_f32(b, chunk / 4, s);
        let _ = backend.sync_stream(s);
        let t0 = Instant::now();
        for _ in 0..4 {
            backend.vec_copy(scratch, b, chunk / 4, s).expect("copy");
        }
        backend.sync_stream(s).expect("sync");
        let gbs = (4 * chunk) as f64 / t0.elapsed().as_secs_f64() / 1e9;
        let free = backend.device_info().map(|d| d.free_vram_bytes as f64 / (1u64 << 30) as f64).unwrap_or(0.0);
        println!("chunk {i:2} ({:5.1} GiB allocated): read {gbs:7.1} GB/s | reported free {free:.2} GiB", (i + 1) as f64 * 0.5);
        bufs.push(b);
    }
    if std::env::var_os("SMALLTIDE_PROBE_REREAD").is_some() {
        for (i, b) in bufs.iter().enumerate() {
            let t0 = Instant::now();
            for _ in 0..4 {
                backend.vec_copy(scratch, *b, chunk / 4, s).expect("copy");
            }
            backend.sync_stream(s).expect("sync");
            let gbs = (4 * chunk) as f64 / t0.elapsed().as_secs_f64() / 1e9;
            println!("reread chunk {i:2}: {gbs:7.1} GB/s");
        }
    }
    if let Some(secs) = std::env::var("SMALLTIDE_PROBE_HOLD").ok().and_then(|v| v.parse::<u64>().ok()) {
        println!("holding allocations for {secs}s (pid {})", std::process::id());
        std::thread::sleep(std::time::Duration::from_secs(secs));
    }
    for b in bufs {
        let _ = backend.free_device(b);
    }
}

fn run_load_test(model_dir: &Path) {
    let cfg = Inkling2Config::default();
    println!("=== SmallTide Step 2 Live Model Load Test ({}) ===", model_dir.display());

    // 1. Tokenizer check
    let t_tok = Instant::now();
    let tok_path = model_dir.join("tokenizer.json");
    let tok = BpeTokenizer::from_file(&tok_path).expect("load tokenizer.json");
    let sample = "<|message_user|>Hello, Inkling-2-Mini! Explain TideLM in one sentence.<|message_model|>";
    let ids = tok.encode(sample);
    let roundtrip = tok.decode(&ids);
    println!(
        "[Tokenizer] Loaded {} tokens in {:.2}s | encode={} ids | roundtrip_exact={}",
        tok.vocab_size(),
        t_tok.elapsed().as_secs_f64(),
        ids.len(),
        roundtrip == sample
    );
    assert_eq!(roundtrip, sample, "Tokenizer roundtrip mismatch: {roundtrip:?}");

    // 2. Dense --fp8-dense loader
    let t_dense = Instant::now();
    let sf_path = model_dir.join("model-00001-of-00001.safetensors");
    let dense = Inkling2DenseHost::load_from_safetensors(&cfg, &sf_path)
        .expect("load Inkling-2-Mini dense weights");
    let dense_vram_gib = dense.total_vram_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
    let embed_ram_mib = dense.embed_bf16.len() as f64 / (1024.0 * 1024.0);
    println!(
        "[Dense FP8] Loaded all 24 layers + lm_head in {:.2}s | Dense VRAM = {:.2} GiB | CPU embed = {:.1} MiB",
        t_dense.elapsed().as_secs_f64(),
        dense_vram_gib,
        embed_ram_mib
    );

    // 3. Upload FP8 dense weights to RX 9070 XT VRAM + run TideExecutor Mode 1 Pre-Prefill from experts.stide!
    let t_tide = Instant::now();
    let stide_path = model_dir.join("experts.stide");
    let pack = StidePackReader::open(&stide_path).expect("open experts.stide");
    let backend = Arc::new(HipBackend::new_auto().expect("init HipBackend"));
    let info_before = backend.device_info().expect("device_info before");

    let _dense_gpu = Inkling2DenseGpu::upload(backend.clone(), dense)
        .expect("upload FP8 dense weights to VRAM");
    let info_after_dense = backend.device_info().expect("device_info after dense");
    println!(
        "[Dense GPU] Uploaded in {:.2}s | VRAM free after dense: {:.2} GiB / {:.2} GiB",
        t_tide.elapsed().as_secs_f64(),
        info_after_dense.free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        info_after_dense.total_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    // Allocate VRAM expert pool: 1,536 Loop Core + 600 outer NVFP4 (~9.13 GiB) + 128 outer MXFP8 (~1.00 GiB) = ~10.13 GiB!
    // Leaves ~4.5 GiB VRAM free for 256K FP8 KV Cache + dspark + WDDM headroom!
    let t_preload = Instant::now();
    let mut tide_cfg = TideConfig::default();
    tide_cfg.pinned_staging_mib = 256;
    let ram_budget = 10 * 1024 * 1024 * 1024usize;
    let mut exec = TideExecutor::new(
        backend.clone(),
        cfg,
        tide_cfg,
        2_136, // NVFP4 resident slots (~9.13 GiB)
        128,   // MXFP8 resident slots (~1.00 GiB)
        ram_budget,
        Some(pack),
    )
    .expect("create TideExecutor");

    let ops = exec.scheduler.plan_pre_prefill(None);
    exec.execute_transfers(&ops, true, false)
        .expect("execute Mode 1 pre-prefill transfers");
    backend.sync_stream(exec.copy_stream).expect("sync copy stream");

    let info_after = backend.device_info().expect("device_info after");
    let (pre, loop_c, coda) = exec.scheduler.cache.count_vram_resident();
    let ram_cnt = exec.scheduler.cache.count_ram_resident();
    println!(
        "[Tide Mode 1] Pre-prefill {} transfers in {:.2}s | VRAM resident: prelude={}, loop_core={}/1536, coda={} | RAM warm: {} experts ({:.2} GiB)",
        ops.len(),
        t_preload.elapsed().as_secs_f64(),
        pre,
        loop_c,
        coda,
        ram_cnt,
        exec.scheduler.cache.ram_used_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    println!(
        "[RX 9070 XT VRAM] Start: {:.2} GiB free -> After Dense+Tide: {:.2} GiB free (total used {:.2} GiB)",
        info_before.free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        info_after.free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        (info_before.free_vram_bytes.saturating_sub(info_after.free_vram_bytes)) as f64 / (1024.0 * 1024.0 * 1024.0),
    );
}

/// Engine/server configuration lives in `.env` (see `.env.example`); only per-request inputs
/// are flags. Old configuration flags are rejected with a pointer to the variable.
fn moved_to_env(flag: &str) -> Option<&'static str> {
    Some(match flag {
        "--model" => "SMALLTIDE_MODEL_DIR",
        "--max-ctx" => "SMALLTIDE_MAX_CTX",
        "--rescue-mode" => "SMALLTIDE_RESCUE_MODE",
        "--rescue-threshold" => "SMALLTIDE_RESCUE_THRESHOLD",
        "--nvfp4-slots" => "SMALLTIDE_NVFP4_SLOTS",
        "--mxfp8-slots" => "SMALLTIDE_MXFP8_SLOTS",
        "--mxfp8-host" => "SMALLTIDE_MXFP8_HOST",
        "--ram-gb" => "SMALLTIDE_RAM_GB",
        "--prefill" => "SMALLTIDE_PREFILL",
        "--profile" => "SMALLTIDE_PROFILE",
        "--host" => "SMALLTIDE_HOST",
        "--port" => "SMALLTIDE_PORT",
        _ => return None,
    })
}

pub fn reject_config_flag(flag: &str) -> ! {
    match moved_to_env(flag) {
        Some(var) => eprintln!("{flag} is configured in .env now: set {var}=... (see .env.example)"),
        None => eprintln!("unknown flag: {flag}"),
    }
    std::process::exit(2);
}

fn run_generate(args: &[String]) {
    let mut prompt = "Explain in two concise sentences what makes Mixture-of-Experts language models fast.".to_string();
    let mut raw_prompt = false;
    let mut effort = "0".to_string();
    let mut max_tokens: usize = 128;
    let mut temperature: f32 = 0.0;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" | "-p" if i + 1 < args.len() => { prompt = args[i + 1].clone(); i += 2; }
            "--raw" => { raw_prompt = true; i += 1; }
            "--effort" if i + 1 < args.len() => { effort = args[i + 1].clone(); i += 2; }
            "--max-tokens" | "-n" if i + 1 < args.len() => {
                max_tokens = args[i + 1].parse().unwrap_or_else(|_| { eprintln!("--max-tokens needs a number"); std::process::exit(2) });
                i += 2;
            }
            "--temp" if i + 1 < args.len() => {
                temperature = args[i + 1].parse().unwrap_or_else(|_| { eprintln!("--temp needs a number"); std::process::exit(2) });
                i += 2;
            }
            other => reject_config_flag(other),
        }
    }

    let s = settings::Settings::from_env();
    let max_ctx = s.max_ctx;
    let enable_profile = s.profile;
    let prefill_mode = if s.batched_prefill { "batched" } else { "token" };
    let settings::Loaded { mut engine, tok, backend, nvfp4_slots, mxfp8_slots } = settings::build_engine(&s, if std::env::var_os("SMALLTIDE_BATCH_SELFTEST").is_some() {
        3
    } else {
        std::env::var("SMALLTIDE_BATCH_BENCH").ok().and_then(|v| v.parse::<usize>().ok()).map_or(1, |m| m + 1)
    });

    // `run` is for dev / benchmarks: it does not teach the routing priors (repeated test prompts
    // would skew them) unless SMALLTIDE_PRIORS_UPDATE is set explicitly. `serve` learns from traffic.
    if std::env::var_os("SMALLTIDE_PRIORS_UPDATE").is_none() {
        if let Some(st) = engine.tide.scheduler.priors_store.as_mut() {
            st.update = false;
        }
    }

    if std::env::var_os("SMALLTIDE_BENCH_MOE").is_some() {
        let n4 = nvfp4_slots as u32;
        let m8 = mxfp8_slots as u32;
        let mut sets: Vec<(String, Vec<u32>, bool)> = Vec::new();
        sets.push(("all 8 missing (launch floor)".to_string(), vec![u32::MAX; 8], false));
        for stride in [1u32, 2, 4, 8, 16, 64, 128] {
            if 7 * stride < n4 {
                sets.push((format!("nvfp4 8 slots stride {stride}"), (0..8).map(|i| i * stride).collect(), false));
            }
        }
        for st in (0..m8.saturating_sub(8)).step_by(160) {
            sets.push((format!("mxfp8 slots {st}..{}", st + 8), (st..st + 8).collect(), true));
        }
        for (label, slots, mx) in sets {
            let cold = engine.debug_bench_moe(&slots, mx, 10, true).expect("bench");
            let warm = engine.debug_bench_moe(&slots, mx, 10, false).expect("bench");
            println!("[bench-moe] {label:<24} cold {cold:.3} ms | warm {warm:.3} ms");
        }
    }

    let formatted = if raw_prompt {
        prompt.clone()
    } else {
        format!(
            "<|message_system|><|content_text|>Thinking effort level: {effort}<|end_message|><|message_user|><|content_text|>{prompt}<|end_message|>"
        )
    };
    let prompt_ids = tok.encode(&formatted);
    println!("[Prompt] {} tokens", prompt_ids.len());

    // Prefill
    let pf_kt = std::env::var_os("SMALLTIDE_KTIMING_PREFILL").is_some();
    if pf_kt {
        backend.set_kernel_timing(true).expect("kernel timing");
        let _ = backend.take_kernel_times();
    }
    let pf_seg = std::env::var_os("SMALLTIDE_SEGTIME_PREFILL").is_some();
    if pf_seg {
        backend.set_segment_timing(true).expect("segment timing");
        let _ = backend.take_segment_times();
    }
    let t_pf = Instant::now();
    if prefill_mode == "token" {
        for (idx, &tid) in prompt_ids.iter().enumerate() {
            let is_last = idx + 1 == prompt_ids.len();
            engine
                .forward_token(tid, idx, true, is_last)
                .expect("prefill forward_token");
        }
    } else {
        engine.prefill(&prompt_ids, 0, true).expect("batched prefill");
    }
    backend.sync_stream(engine.tide.compute_stream).expect("sync prefill");
    let pf_secs = t_pf.elapsed().as_secs_f64();
    if pf_kt {
        let mut kt = backend.take_kernel_times();
        kt.sort_by(|a, b| b.2.total_cmp(&a.2));
        let total: f64 = kt.iter().map(|k| k.2).sum();
        println!("[GPU kernels, prefill] total {total:.1} ms (wall {:.1} ms)", pf_secs * 1000.0);
        for (name, cnt, ms) in kt.iter().take(14) {
            println!("  {:<32} {:>6}x  {:>9.1} ms  ({:>7.1} us avg)", name, cnt, ms, ms * 1000.0 / *cnt as f64);
        }
        backend.set_kernel_timing(false).expect("kernel timing off");
    }
    if pf_seg {
        let st = backend.take_segment_times();
        let total: f64 = st.iter().map(|k| k.2).sum();
        println!("[GPU segments, prefill] total {total:.1} ms (wall {:.1} ms)", pf_secs * 1000.0);
        for (name, cnt, ms) in st {
            println!("  {:<32} {:>6}x  {:>9.1} ms  ({:>7.1} us avg)", name, cnt, ms, ms * 1000.0 / cnt.max(1) as f64);
        }
        backend.set_segment_timing(false).expect("segment timing off");
    }

    // Mode 2: Prefill -> Decode Boundary Rebalance
    let t_reb = Instant::now();
    let reb_swaps = engine.rebalance_after_prefill(128).expect("Mode 2 rebalance");
    let reb_ms = t_reb.elapsed().as_secs_f64() * 1000.0;

    let sampling = SamplingParams {
        temperature,
        ..SamplingParams::default()
    };
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut pos = prompt_ids.len();

    print!("\n--- Output ---\n");
    let _ = std::io::stdout().flush();

    let t_dec = Instant::now();
    if enable_profile {
        engine.profile_enabled = true;
    }
    let _ = backend.take_host_stats();
    let mut post_ms = 0.0f64;
    let ktiming = std::env::var_os("SMALLTIDE_KTIMING").is_some();
    let segtiming = std::env::var_os("SMALLTIDE_SEGTIME").is_some();
    if segtiming {
        backend.set_segment_timing(true).expect("segment timing");
        let _ = backend.take_segment_times();
    }
    if ktiming {
        backend.set_kernel_timing(true).expect("kernel timing");
        let _ = backend.take_kernel_times();
    }
    let mut next_tok = engine
        .sample_next_token(&sampling, &generated)
        .expect("sample first token");

    if let Some(iters) = std::env::var("SMALLTIDE_DRAFT_BENCH").ok().and_then(|v| v.parse::<usize>().ok()) {
        // Dev: the drafter alone (anchor = first sampled token at the prompt end), wall time per
        // draft for 1..7 drafts, then a per-kernel table for 7 drafts.
        let p0 = prompt_ids.len();
        let row = engine.last_tap_rows.max(1) - 1;
        for n in [1usize, 2, 3, 5, 7] {
            for _ in 0..3 { engine.draft(next_tok, p0, row, n).expect("draft"); }
            let t0 = Instant::now();
            for _ in 0..iters { engine.draft(next_tok, p0, row, n).expect("draft"); }
            println!("[draft-bench] {n} drafts: {:.3} ms per draft call", t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
        }
        backend.set_kernel_timing(true).expect("kt");
        let _ = backend.take_kernel_times();
        for _ in 0..iters { engine.draft(next_tok, p0, row, 7).expect("draft"); }
        let mut kt = backend.take_kernel_times();
        backend.set_kernel_timing(false).expect("kt off");
        kt.sort_by(|a, b| b.2.total_cmp(&a.2));
        for (name, cnt, ms) in kt.iter() {
            println!("    {:<30} {:>5.1}x {:>8.3} ms/call ({:.1} us avg)", name, *cnt as f64 / iters as f64, ms / iters as f64, ms * 1000.0 / *cnt as f64);
        }
        return;
    }

    if std::env::var_os("SMALLTIDE_ROWS_CMP").is_some() {
        // Dev: one token from the same state through forward_token, a 1-row batched round and a
        // 1-row verify; max |logit| differences.
        engine.tide.wait_and_publish_all_pending().expect("drain pending");
        let vocab = engine.cfg.unpadded_vocab_size;
        let p0 = prompt_ids.len();
        let read = |eng: &smalltide_model_inkling2::engine::Inkling2Engine<smalltide_backend_hip::HipBackend>, buf: smalltide_core::DeviceBuf| -> Vec<f32> {
            let mut b = vec![0u8; vocab * 4];
            eng.backend.memcpy_d2h_sync(&mut b, buf.slice(0, vocab * 4), 0, eng.tide.compute_stream).expect("d2h");
            b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        };
        let mut snap = None;
        engine.snapshot_state(p0, &mut snap).expect("snapshot");
        let mut t = next_tok;
        for step in 0..4 {
            engine.forward_token(t, p0 + step, false, true).expect("single");
            let a = read(&engine, engine.logits_buf());
            engine.restore_state(snap.as_ref().unwrap(), false).expect("restore");
            for _ in 0..step { /* state at p0 only: compare the first step from the snapshot */ }
            engine.forward_batch(&[(engine.active_lane(), t, p0 + step)]).expect("batch");
            let b = read(&engine, engine.batch_logits_buf().unwrap());
            engine.post_batch_tide_step().expect("post");
            engine.restore_state(snap.as_ref().unwrap(), false).expect("restore");
            if let Ok(d) = std::env::var("SMALLTIDE_DUMP_DIR") {
                let v = std::path::PathBuf::from(d).join("verify");
                let _ = std::fs::create_dir_all(&v);
                engine.set_dump_dir(v);
            }
            engine.verify(&[t], p0 + step).expect("verify");
            let c = read(&engine, engine.batch_logits_buf().unwrap());
            engine.post_batch_tide_step().expect("post");
            engine.restore_state(snap.as_ref().unwrap(), false).expect("restore");
            let d = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
            let am = |v: &[f32]| v.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0;
            if let Ok(dir) = std::env::var("SMALLTIDE_ROWS_CMP") {
                if dir.len() > 1 {
                    for (name, v) in [("single", &a), ("batch", &b), ("verify", &c)] {
                        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                        let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{name}.f32")), bytes);
                    }
                }
            }
            println!("[rows-cmp] token {t}: single-batch {:.4} single-verify {:.4} batch-verify {:.4} | argmax {} {} {}", d(&a, &b), d(&a, &c), d(&b, &c), am(&a), am(&b), am(&c));
            // Causality: rows must not depend on the drafts after them.
            for off in [0usize, 3, 7, 13] {
                let mut outs = Vec::new();
                for tail in [[11u32, 22, 33, 44, 55, 66, 77], [11u32, 22, 33, 999, 1234, 4321, 8888]] {
                    engine.restore_state(snap.as_ref().unwrap(), false).expect("restore");
                    // advance `off` positions with plain decode so pos0 % 16 varies
                    let mut tt = t;
                    for j in 0..off {
                        engine.forward_token(tt, p0 + j, false, true).expect("adv");
                        tt = 100 + j as u32;
                    }
                    let mut blk = vec![tt];
                    blk.extend_from_slice(&tail);
                    engine.verify(&blk, p0 + off).expect("verify");
                    let bl = engine.batch_logits_buf().unwrap();
                    let rows: Vec<Vec<f32>> = (0..8).map(|r| read(&engine, bl.slice(r * vocab * 4, vocab * 4))).collect();
                    engine.post_batch_tide_step().expect("post");
                    outs.push(rows);
                }
                let diffs: Vec<String> = (0..8).map(|r| format!("{:.3}", d(&outs[0][r], &outs[1][r]))).collect();
                println!("[rows-cmp] causality pos0={} (mod16 {}): row diffs (tails differ from row 3 on) {}", p0 + off, (p0 + off) % 16, diffs.join(" "));
            }
            engine.restore_state(snap.as_ref().unwrap(), false).expect("restore");
            t = am(&a) as u32;
            if step == 0 { break; }
        }
        return;
    }

    if let Some(n) = std::env::var("SMALLTIDE_VERIFY_SELFTEST").ok().and_then(|v| v.parse::<usize>().ok()) {
        // Speculative verify + rewind vs plain greedy decode: reference tokens from forward_token;
        // then the same span through 8-row verify steps whose drafts are the reference tokens with
        // one corrupted per block (forces a rejection and a short-conv rewind every step).
        let p0 = prompt_ids.len();
        // Same resident experts for both runs: land every queued swap first (run with
        // SMALLTIDE_ASYNC_PER_TOKEN=0 so no new ones are queued).
        engine.tide.wait_and_publish_all_pending().expect("drain pending");
        let mut snap = None;
        engine.snapshot_state(p0, &mut snap).expect("snapshot");
        let mut refs = vec![next_tok];
        let t_ref = Instant::now();
        for i in 0..n {
            engine.forward_token(refs[i], p0 + i, false, true).expect("decode");
            engine.post_decode_tide_step().expect("post");
            refs.push(engine.sample_next_token(&sampling, &[]).expect("sample"));
        }
        let ref_ms = t_ref.elapsed().as_secs_f64() * 1000.0 / n as f64;
        engine.restore_state(snap.as_ref().unwrap(), false).expect("restore");
        let vkt = std::env::var_os("SMALLTIDE_KTIMING").is_some();
        if vkt {
            backend.set_kernel_timing(true).expect("kt");
            let _ = backend.take_kernel_times();
        }
        let vseg = std::env::var_os("SMALLTIDE_SEGTIME").is_some();
        if vseg {
            backend.set_segment_timing(true).expect("seg");
            let _ = backend.take_segment_times();
        }
        let t_ver = Instant::now();
        let (mut i, mut steps, mut agree_until) = (0usize, 0usize, None::<usize>);
        let mut out = vec![refs[0]];
        let width = std::env::var("SMALLTIDE_VERIFY_WIDTH").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(8).clamp(2, 8);
        while i + width < refs.len() {
            let mut block: Vec<u32> = refs[i..i + width].to_vec();
            block[0] = *out.last().unwrap();
            let corrupt = 1 + (steps % (width - 1).max(1)).min(width - 2);
            block[corrupt] = (block[corrupt] + 7) % 100_000;
            engine.verify(&block, p0 + i).expect("verify");
            let am = engine.rows_argmax(width).expect("argmax");
            let mut c = 0usize;
            while c < width - 1 && block[c + 1] == am[c] {
                c += 1;
            }
            engine.verify_rewind(width, c + 1).expect("rewind");
            engine.post_batch_tide_step().expect("post batch");
            out.extend_from_slice(&block[1..=c]);
            out.push(am[c]);
            i += c + 1;
            steps += 1;
            if agree_until.is_none() {
                if let Some(d) = (0..out.len().min(refs.len())).find(|&k| out[k] != refs[k]) {
                    agree_until = Some(d);
                }
            }
        }
        let ver_ms = t_ver.elapsed().as_secs_f64() * 1000.0 / steps.max(1) as f64;
        if vkt {
            let mut kt = backend.take_kernel_times();
            kt.sort_by(|a, b| b.2.total_cmp(&a.2));
            for (name, cnt, ms) in kt.iter().take(40) {
                println!("    {:<32} {:>6.1}x {:>8.3} ms/step ({:.1} us avg)", name, *cnt as f64 / steps.max(1) as f64, ms / steps.max(1) as f64, ms * 1000.0 / *cnt as f64);
            }
            backend.set_kernel_timing(false).expect("kt off");
        }
        if vseg {
            let st = backend.take_segment_times();
            for (name, cnt, ms) in st.iter().take(10) {
                println!("    {:<32} {:>6.1}x {:>8.3} ms/step", name, *cnt as f64 / steps.max(1) as f64, ms / steps.max(1) as f64);
            }
            backend.set_segment_timing(false).expect("seg off");
        }
        if let Some((u, r)) = backend.resident_vram_info() {
            println!("[verify-selftest] process VRAM usage now {:.2} GiB of resident {:.2} GiB", u as f64 / (1u64 << 30) as f64, r as f64 / (1u64 << 30) as f64);
        }
        println!("[verify-selftest] distinct resident experts per verify step {:.1} (all layers)", engine.dev_unique_resident as f64 / engine.dev_rounds.max(1) as f64);
        println!("[verify-selftest] plain decode {ref_ms:.2} ms/token | 8-row verify {ver_ms:.2} ms/step ({:.2}x a decode step)", ver_ms / ref_ms);
        let cmp = out.len().min(refs.len());
        let same = (0..cmp).filter(|&k| out[k] == refs[k]).count();
        println!("[verify-selftest] {steps} verify steps, {} tokens; identical to plain decode: {same}/{cmp}; first divergence at {:?}", out.len(), agree_until);
        return;
    }

    if let Some(mrows) = std::env::var("SMALLTIDE_BATCH_BENCH").ok().and_then(|v| v.parse::<usize>().ok()) {
        // Dev: time batched decode rounds of 1..=M rows (lanes 1..=M, same prompt) with GPU segments.
        for l in 1..=mrows {
            engine.select_lane(l);
            engine.reset_sequence().expect("reset");
            engine.prefill(&prompt_ids, 0, true).expect("prefill lane");
        }
        let first = next_tok;
        for m in 1..=mrows {
            let mut pos = prompt_ids.len();
            let mut toks = vec![first; m];
            let rounds = 24usize;
            for warm in [true, false] {
                if !warm {
                    backend.set_segment_timing(true).expect("seg");
                    let _ = backend.take_segment_times();
                }
                let t0 = Instant::now();
                for _ in 0..(if warm { 4 } else { rounds }) {
                    let rows: Vec<(usize, u32, usize)> = (0..m).map(|i| (i + 1, toks[i], pos)).collect();
                    engine.forward_batch(&rows).expect("batch");
                    for i in 0..m {
                        toks[i] = engine.sample_batch_row(i, i + 1, &sampling, &[]).expect("sample");
                    }
                    engine.post_batch_tide_step().expect("post");
                    pos += 1;
                }
                if !warm {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0 / rounds as f64;
                    let st = backend.take_segment_times();
                    backend.set_segment_timing(false).expect("seg off");
                    println!("[batch-bench] {m} rows: {ms:.2} ms/round ({:.1} tok/s aggregate)", m as f64 * 1000.0 / ms);
                    let total: f64 = st.iter().map(|k| k.2).sum();
                    for (name, cnt, t) in st.iter().filter(|k| k.2 > total * 0.03) {
                        println!("    {:<30} {:>6.1}x {:>8.3} ms/round", name, *cnt as f64 / rounds as f64, t / rounds as f64);
                    }
                }
            }
        }
        return;
    }

    if let Some(n) = std::env::var("SMALLTIDE_BATCH_SELFTEST").ok().and_then(|v| v.parse::<usize>().ok()) {
        // Numerics of batched decode vs the single-row path: lane 0 decodes with forward_token,
        // lanes 1 and 2 (same prompt) together with forward_batch, all teacher-forced on lane 0's
        // greedy tokens.
        let vocab = engine.cfg.unpadded_vocab_size;
        let read = |eng: &smalltide_model_inkling2::engine::Inkling2Engine<smalltide_backend_hip::HipBackend>, buf: smalltide_core::DeviceBuf| -> Vec<f32> {
            let mut b = vec![0u8; vocab * 4];
            eng.backend.memcpy_d2h_sync(&mut b, buf, 0, eng.tide.compute_stream).expect("d2h");
            b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        };
        let argmax = |v: &[f32]| v.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0;
        // Two different prompts; single-path references first (lane 0), then both in one batch
        // (lanes 1 and 2), teacher-forced on the reference tokens.
        let other = tok.encode("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>Write a haiku about winter rain on a tin roof.<|end_message|><|message_model|>");
        let prompts = [prompt_ids.clone(), other];
        engine.tide.wait_and_publish_all_pending().expect("drain pending");
        let mut refs: Vec<(Vec<u32>, Vec<Vec<f32>>)> = Vec::new();
        for pr in &prompts {
            engine.select_lane(0);
            engine.reset_sequence().expect("reset");
            engine.prefill(pr, 0, true).expect("prefill ref");
            let mut toks = vec![argmax(&read(&engine, engine.logits_buf())) as u32];
            let mut lg = Vec::new();
            for st in 0..n {
                engine.forward_token(toks[st], pr.len() + st, false, true).expect("single");
                engine.post_decode_tide_step().expect("post");
                let a = read(&engine, engine.logits_buf());
                toks.push(argmax(&a) as u32);
                lg.push(a);
            }
            refs.push((toks, lg));
        }
        for (l, pr) in [(1usize, &prompts[0]), (2usize, &prompts[1])] {
            engine.select_lane(l);
            engine.reset_sequence().expect("reset");
            engine.prefill(pr, 0, true).expect("prefill lane");
        }
        let (mut agree, mut worst, mut total) = (0usize, 0f32, 0usize);
        for step in 0..n {
            let rows = [(1usize, refs[0].0[step], prompts[0].len() + step), (2usize, refs[1].0[step], prompts[1].len() + step)];
            engine.forward_batch(&rows).expect("batch");
            let bl = engine.batch_logits_buf().expect("batch logits");
            for r in 0..2 {
                let b = read(&engine, bl.slice(r * vocab * 4, vocab * 4));
                let a = &refs[r].1[step];
                let d = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                let (ia, ib) = (argmax(a), argmax(&b));
                agree += (ia == ib) as usize;
                total += 1;
                worst = worst.max(d);
                if step < 3 || ia != ib {
                    println!("[batch-selftest] step {step:3} row {r}: max|single-batch| {d:.4} argmax {ia} vs {ib}");
                }
            }
            engine.post_batch_tide_step().expect("post batch");
        }
        let n = total;
        println!("[batch-selftest] {n} steps: argmax agreement {agree}/{n}, worst max|dlogit| {worst:.4}");
        return;
    }

    if std::env::var_os("SMALLTIDE_GRAPH_BENCH").is_some() {
        // Dev experiment: GPU time of one decode token, eager vs replayed HIP graph (the graph
        // bakes in this token's position, so its outputs are not used).
        let s = engine.tide.compute_stream;
        let pos0 = prompt_ids.len();
        for _ in 0..3 {
            engine.forward_token(next_tok, pos0, false, true).expect("warm");
            engine.post_decode_tide_step().expect("post");
            backend.sync_stream(s).expect("sync");
        }
        let n = 20;
        let t0 = Instant::now();
        for _ in 0..n {
            engine.forward_token(next_tok, pos0, false, true).expect("eager");
            backend.sync_stream(s).expect("sync");
        }
        let eager = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        engine.post_decode_tide_step().expect("post");
        engine.run_deferred_tide_step().expect("deferred");
        backend.sync_stream(s).expect("sync");
        engine.tide.wait_and_publish_all_pending().expect("drain pending");
        backend.sync_stream(engine.tide.copy_stream).expect("sync copy");
        backend.sync_stream(s).expect("sync");
        let _ = engine.tide.poll_pending();
        engine.clear_route_log_flag();
        backend.begin_graph_capture(s).expect("begin capture");
        let cap = engine.forward_token(next_tok, pos0, false, true);
        let graph = backend.end_graph_capture(s);
        match (cap, graph) {
            (Ok(()), Ok(g)) => {
                for _ in 0..3 {
                    backend.launch_graph(g, s).expect("launch");
                }
                backend.sync_stream(s).expect("sync");
                let t0 = Instant::now();
                for _ in 0..n {
                    backend.launch_graph(g, s).expect("launch");
                    backend.sync_stream(s).expect("sync");
                }
                let replay = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
                println!("[graph-bench] one decode token: eager {eager:.2} ms | graph replay {replay:.2} ms");
            }
            (c, g) => println!("[graph-bench] capture failed: forward={c:?} graph={:?}", g.err()),
        }
        return;
    }

    const EOS_ID: u32 = 200_006;
    const EOT_ID: u32 = 199_999;

    // Speculative decoding (greedy): the drafter proposes width - 1 tokens, one verify pass
    // checks them; accepted drafts plus the target's next token are emitted per step.
    let spec = engine.drafter.is_some() && temperature <= 0.0;
    // Width: fixed (SMALLTIDE_SPEC_WIDTH=n) or adaptive (default): per step the width maximizing
    // expected tokens per ms from the recent per-draft acceptance rate.
    let fixed_width = std::env::var("SMALLTIDE_SPEC_WIDTH").ok().and_then(|v| v.parse::<usize>().ok()).map(|w| w.clamp(2, 8));
    let spec_width = fixed_width.unwrap_or(8);
    let mut acc_s = 2.0f64; // decayed accepted drafts
    let mut acc_f = 2.0f64; // decayed rejections
    let mut width_hist = [0usize; 9];
    let (mut spec_steps, mut spec_accepted, mut draft_ms, mut verify_ms) = (0usize, 0usize, 0f64, 0f64);
    let mut spec_hist = [0usize; 8];
    if spec {
        let mut tap_row = engine.last_tap_rows.max(1) - 1;
        let mut done = false;
        let emit = |t: u32, generated: &mut Vec<u32>| -> bool {
            if t == EOS_ID || t == EOT_ID {
                return false;
            }
            generated.push(t);
            print!("{}", tok.decode(&[t]));
            let _ = std::io::stdout().flush();
            generated.len() < max_tokens
        };
        if !emit(next_tok, &mut generated) {
            done = true;
        }
        while !done && pos + spec_width <= max_ctx {
            let spec_width = fixed_width.unwrap_or_else(|| {
                let a = ((acc_s + 1.0) / (acc_s + acc_f + 2.0)).min(0.98);
                (2..=8usize)
                    .max_by(|&x, &y| {
                        let gain = |w: usize| (1.0 - a.powi(w as i32)) / (1.0 - a) / (17.0 + 2.5 * (w - 1) as f64);
                        gain(x).total_cmp(&gain(y))
                    })
                    .unwrap()
            });
            width_hist[spec_width] += 1;
            let n = spec_width - 1;
            let td = Instant::now();
            let drafts = engine.draft(next_tok, pos, tap_row, n).expect("draft");
            draft_ms += td.elapsed().as_secs_f64() * 1000.0;
            let mut block = Vec::with_capacity(spec_width);
            block.push(next_tok);
            block.extend_from_slice(&drafts);
            let tv = Instant::now();
            engine.verify(&block, pos).expect("verify");
            let am = engine.rows_argmax(spec_width).expect("verify argmax");
            let mut c = 0usize;
            while c < n && block[c + 1] == am[c] {
                c += 1;
            }
            engine.verify_rewind(spec_width, c + 1).expect("verify rewind");
            engine.post_batch_tide_step().expect("post verify");
            verify_ms += tv.elapsed().as_secs_f64() * 1000.0;
            spec_steps += 1;
            spec_accepted += c;
            spec_hist[c] += 1;
            acc_s = 0.85 * acc_s + c as f64;
            acc_f = 0.85 * acc_f + if c < n { 1.0 } else { 0.0 };
            for &t in &block[1..=c] {
                if !emit(t, &mut generated) {
                    done = true;
                    break;
                }
            }
            pos += c + 1;
            tap_row = c;
            next_tok = am[c];
            if !done && !emit(next_tok, &mut generated) {
                done = true;
            }
        }
    }

    for step in 0..max_tokens {
        if spec {
            break;
        }
        if next_tok == EOS_ID || next_tok == EOT_ID {
            break;
        }
        generated.push(next_tok);
        let piece = tok.decode(&[next_tok]);
        print!("{piece}");
        let _ = std::io::stdout().flush();

        if step + 1 >= max_tokens || pos >= max_ctx {
            break;
        }
        engine
            .forward_token(next_tok, pos, false, true)
            .expect("decode forward_token");
        let tp = Instant::now();
        engine.post_decode_tide_step().expect("post_decode_tide_step");
        post_ms += tp.elapsed().as_secs_f64() * 1000.0;
        pos += 1;
        next_tok = engine
            .sample_next_token(&sampling, &generated)
            .expect("sample token");
    }
    let dec_secs = t_dec.elapsed().as_secs_f64();
    let dec_toks = generated.len();
    if spec && spec_steps > 0 {
        let hist: Vec<String> = spec_hist[..spec_width].iter().enumerate().map(|(i, c)| format!("{i}:{c}")).collect();
        if fixed_width.is_none() {
            let wh: Vec<String> = (2..=8).filter(|&w| width_hist[w] > 0).map(|w| format!("w{w}:{}", width_hist[w])).collect();
            println!("
[spec] adaptive width per step: {}", wh.join(" "));
        }
        println!(
            "\n[spec] distinct resident experts per verify step {:.1} | width {}: {spec_steps} steps, mean accepted {:.2} -> {:.2} tokens/step | draft {:.2} ms + verify {:.2} ms per step | accepted per step {}",
            engine.dev_unique_resident as f64 / engine.dev_rounds.max(1) as f64,
            fixed_width.map_or("adaptive".to_string(), |w| w.to_string()),
            spec_accepted as f64 / spec_steps as f64,
            1.0 + spec_accepted as f64 / spec_steps as f64,
            draft_ms / spec_steps as f64,
            verify_ms / spec_steps as f64,
            hist.join(" ")
        );
    }
    let hs = backend.take_host_stats();
    let tel = engine.telemetry();
    println!("\n--------------");
    println!(
        "[Perf] Prefill: {} tok in {:.2}s ({:.2} tok/s) | Boundary Rebalance: {} swaps ({:.1} ms) | Decode: {} tok in {:.2}s ({:.2} tok/s)",
        prompt_ids.len(),
        pf_secs,
        prompt_ids.len() as f64 / pf_secs.max(1e-6),
        reb_swaps,
        reb_ms,
        dec_toks,
        dec_secs,
        dec_toks as f64 / dec_secs.max(1e-6),
    );
    let pct = |h: u64, m: u64| -> f64 {
        let tot = h + m;
        if tot == 0 { 100.0 } else { (h as f64 / tot as f64) * 100.0 }
    };
    println!(
        "[Tide] Overall VRAM hit: {:.1}% (Prelude {:.1}%, LoopCore {:.1}%, Coda {:.1}%) | skipped routing weight {:.2}% (prefill {:.2}%) | Stage1 sync={}, async={}, skipped={} | decode async promotions={} | Stage2 disk->RAM={} | RAM-miss disk reads={}",
        tel.total_vram_hit_rate() * 100.0,
        pct(tel.vram_hits_prelude, tel.vram_misses_prelude),
        pct(tel.vram_hits_loop, tel.vram_misses_loop),
        pct(tel.vram_hits_coda, tel.vram_misses_coda),
        tel.skipped_weight_share() * 100.0,
        if tel.prefill_weight_total > 0.0 { 100.0 * tel.prefill_weight_skipped / tel.prefill_weight_total } else { 0.0 },
        tel.stage1_sync_rescues,
        tel.stage1_async_promotions,
        tel.stage1_skipped_tail_misses,
        tel.decode_async_promotions,
        tel.stage2_disk_reads,
        tel.stage1_ram_fallback_reads,
    );
    if let Some(line) = engine.tide.scheduler.finish_request() {
        println!("{line}");
    }
    if dec_toks > 0 {
        let n = dec_toks as f64;
        println!(
            "[Host/tok] {:.2} ms total | {:.0} launches ({:.2} ms CPU in hipModuleLaunchKernel) | {:.1} syncs ({:.2} ms blocked) | {:.2} ms other CPU (post-decode Tide {:.2} ms)",
            dec_secs * 1000.0 / n,
            hs.launches as f64 / n,
            hs.launch_ms / n,
            hs.syncs as f64 / n,
            hs.sync_wait_ms / n,
            (dec_secs * 1000.0 - hs.launch_ms - hs.sync_wait_ms) / n,
            post_ms / n,
        );
    }
    if segtiming && dec_toks > 0 {
        let n = dec_toks as f64;
        let st = backend.take_segment_times();
        let total: f64 = st.iter().map(|k| k.2).sum();
        println!("[GPU segments/tok] total {:.2} ms", total / n);
        for (name, cnt, ms) in st {
            println!("  {:<32} {:>6.1}x  {:>7.3} ms  ({:>6.1} us avg)", name, cnt as f64 / n, ms / n, ms * 1000.0 / cnt as f64);
        }
    }
    if ktiming && dec_toks > 0 {
        let n = dec_toks as f64;
        let kt = backend.take_kernel_times();
        let total: f64 = kt.iter().map(|k| k.2).sum();
        println!("[GPU kernels/tok] total {:.2} ms in {:.0} launches", total / n, kt.iter().map(|k| k.1 as f64).sum::<f64>() / n);
        for (name, cnt, ms) in kt {
            println!("  {:<28} {:>6.1}x  {:>7.3} ms  ({:>5.1} us avg)", name, cnt as f64 / n, ms / n, ms * 1000.0 / cnt as f64);
        }
    }
    if enable_profile && engine.profile.steps > 0 {
        let n = engine.profile.steps as f64;
        let p = &engine.profile;
        println!(
            "[Profile ms/tok avg over {} steps] res+ctrl={:.2} | attn_proj={:.2} | sconv+clip+rel={:.2} | flash_attn={:.2} | attn_out+moe_prep={:.2} | router+d2h={:.2} | tide_rescue={:.2} | moe={:.2} | lm_head={:.2}",
            p.steps,
            p.res_and_ctrl_ms / n,
            p.attn_proj_ms / n,
            p.sconv_clip_rel_ms / n,
            p.flash_attn_ms / n,
            p.attn_out_ms / n,
            p.router_d2h_ms / n,
            p.tide_rescue_ms / n,
            p.moe_ms / n,
            p.lm_head_ms / n,
        );
    }
}
