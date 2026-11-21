mod chat_format;
mod prefix_cache;
mod server;

use smalltide_backend_cpu as cpu;
use smalltide_backend_hip::{HipBackend, HipRuntime};
use smalltide_core::{ComputeBackend, DType, Inkling2Config, RescuePolicy, TideConfig};
use smalltide_io::dense_loader::Inkling2DenseHost;
use smalltide_io::safetensors::inspect_safetensors_header;
use smalltide_io::stide_pack::{ExpertSlabHeader, ExpertSlabLayout, StidePackReader};
use smalltide_io::tokenizer::BpeTokenizer;
use smalltide_io::win_direct_io::{align_up, AlignedBuffer, DirectFile, SECTOR_ALIGN};
use smalltide_model_inkling2::engine::{Inkling2Engine, SamplingParams};
use smalltide_model_inkling2::weights::Inkling2DenseGpu;
use smalltide_tide::TideExecutor;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
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
            println!("  run [flags]                   Run live 48-layer Inkling-2-Mini inference with TideLM 2-stage streaming");
            println!("  serve [--port 8000] [flags]   Serve OpenAI API (/v1/completions, /v1/chat/completions, /v1/models) as inkling-2-mini");
            println!("  inspect <file.safetensors>    Inspect tensor names, dtypes, and shapes");
        }
    }
}

fn run_probe() {
    let hip = HipRuntime::load().expect("load HIP runtime");
    let info = hip.probe_device(0).expect("probe device 0");
    println!("Loaded HIP DLL : {}", info.loaded_dll);
    println!("HIP Runtime    : {}", info.runtime_version);
    println!("HIP Driver     : {}", info.driver_version);
    println!("Device Count   : {}", info.device_count);
    println!("Device[0]      : {}", info.device_name);
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
    let info = hip.probe_device(0).expect("probe device 0");
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
    let backend = HipBackend::new(0).expect("init HipBackend");
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
    for i in 0..30 {
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
    let backend = Arc::new(HipBackend::new(0).expect("init HipBackend"));
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

fn run_generate(args: &[String]) {
    let mut model_dir = r"models\Inkling-2-Mini".to_string();
    let mut prompt = "Explain in two concise sentences what makes Mixture-of-Experts language models fast.".to_string();
    let mut raw_prompt = false;
    let mut effort = "0".to_string();
    let mut max_tokens: usize = 128;
    let mut max_ctx: usize = 16_384;
    let mut temperature: f32 = 0.0;
    let mut rescue_mode = "threshold".to_string();
    let mut rescue_threshold: f32 = 0.25;
    let mut nvfp4_slots_override: Option<usize> = None;
    let mut mxfp8_slots_override: Option<usize> = None;
    let mut ram_gb: f64 = 10.0;
    let mut enable_profile = false;
    let mut prefill_mode = "batched".to_string();

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--model" if i + 1 < args.len() => { model_dir = args[i + 1].clone(); i += 2; }
            "--prompt" | "-p" if i + 1 < args.len() => { prompt = args[i + 1].clone(); i += 2; }
            "--raw" => { raw_prompt = true; i += 1; }
            "--effort" if i + 1 < args.len() => { effort = args[i + 1].clone(); i += 2; }
            "--max-tokens" | "-n" if i + 1 < args.len() => { max_tokens = args[i + 1].parse().unwrap_or(128); i += 2; }
            "--max-ctx" if i + 1 < args.len() => { max_ctx = args[i + 1].parse().unwrap_or(16_384); i += 2; }
            "--temp" if i + 1 < args.len() => { temperature = args[i + 1].parse().unwrap_or(0.0); i += 2; }
            "--rescue-mode" if i + 1 < args.len() => { rescue_mode = args[i + 1].clone(); i += 2; }
            "--rescue-threshold" if i + 1 < args.len() => { rescue_threshold = args[i + 1].parse().unwrap_or(0.25); i += 2; }
            "--nvfp4-slots" if i + 1 < args.len() => { nvfp4_slots_override = args[i + 1].parse().ok(); i += 2; }
            "--mxfp8-slots" if i + 1 < args.len() => { mxfp8_slots_override = args[i + 1].parse().ok(); i += 2; }
            "--ram-gb" if i + 1 < args.len() => { ram_gb = args[i + 1].parse().unwrap_or(10.0); i += 2; }
            "--profile" => { enable_profile = true; i += 1; }
            "--prefill" if i + 1 < args.len() => { prefill_mode = args[i + 1].clone(); i += 2; }
            other => { eprintln!("Unknown flag: {other}"); i += 1; }
        }
    }

    let cfg = Inkling2Config::default();
    let dir = Path::new(&model_dir);
    let t_start = Instant::now();

    let tok = BpeTokenizer::from_file(&dir.join("tokenizer.json")).expect("load tokenizer.json");
    let dense_host = Inkling2DenseHost::load_from_safetensors(&cfg, &dir.join("model-00001-of-00001.safetensors"))
        .expect("load dense weights");
    let pack = StidePackReader::open(&dir.join("experts.stide")).expect("open experts.stide");
    let backend = Arc::new(HipBackend::new(0).expect("init HipBackend"));

    let dense_gpu = Inkling2DenseGpu::upload(backend.clone(), dense_host).expect("upload dense GPU");
    let info_after_dense = backend.device_info().expect("device_info");

    // Auto-size expert VRAM pools from remaining free VRAM after Dense + KV + WDDM headroom
    let nvfp4_slab = ExpertSlabLayout::for_inkling2(&cfg, DType::NVFP4).aligned_slab_bytes;
    let mxfp8_slab = ExpertSlabLayout::for_inkling2(&cfg, DType::MXFP8).aligned_slab_bytes;
    // 40 local layers * 1024 + 8 global layers * max_ctx (K+V FP8 + F32 scales + sconv)
    let est_kv_bytes = 40 * (2 * 8 * cfg.local_window_size * 128 + 2 * 8 * cfg.local_window_size * 4)
        + 8 * (2 * 8 * max_ctx * 128 + 2 * 8 * max_ctx * 4)
        + 64 * 1024 * 1024;
    let headroom_bytes = 900 * 1024 * 1024usize; // Scratch + WDDM safety headroom
    let scratch_ring_bytes = 8 * (nvfp4_slab + mxfp8_slab);
    let avail_expert_vram = info_after_dense
        .free_vram_bytes
        .saturating_sub(est_kv_bytes + headroom_bytes + scratch_ring_bytes);

    let total_nvfp4 = cfg.mxfp8_start_layer * cfg.n_routed_experts; // 2,560
    let total_mxfp8 = (cfg.num_physical_layers - cfg.mxfp8_start_layer) * cfg.n_routed_experts; // 512

    let (nvfp4_slots, mxfp8_slots) = match (nvfp4_slots_override, mxfp8_slots_override) {
        (Some(n4), Some(m8)) => (n4.min(total_nvfp4), m8.min(total_mxfp8)),
        (Some(n4), None) => {
            let n4 = n4.min(total_nvfp4);
            let rem = avail_expert_vram.saturating_sub(n4 * nvfp4_slab);
            (n4, (rem / mxfp8_slab).min(total_mxfp8).max(16))
        }
        (None, Some(m8)) => {
            let m8 = m8.min(total_mxfp8);
            let rem = avail_expert_vram.saturating_sub(m8 * mxfp8_slab);
            ((rem / nvfp4_slab).min(total_nvfp4).max(1536), m8)
        }
        (None, None) => {
            // Prioritize 100% Loop Core (1536) + remaining NVFP4 (up to 2560), then fill MXFP8 with leftover VRAM!
            let min_mxfp8 = 128usize;
            let max_n4_budget = avail_expert_vram.saturating_sub(min_mxfp8 * mxfp8_slab);
            let n4 = (max_n4_budget / nvfp4_slab).min(total_nvfp4).max(1536);
            let rem = avail_expert_vram.saturating_sub(n4 * nvfp4_slab);
            let m8 = (rem / mxfp8_slab).min(total_mxfp8).max(16);
            (n4, m8)
        }
    };

    let rescue_policy = match rescue_mode.as_str() {
        "exact" => RescuePolicy::Exact,
        "disabled" | "none" => RescuePolicy::Disabled,
        _ => RescuePolicy::Threshold(rescue_threshold),
    };
    let mut tide_cfg = TideConfig::default();
    tide_cfg.pinned_staging_mib = 256;
    tide_cfg.rescue_policy = rescue_policy;

    let ram_budget_bytes = (ram_gb * 1024.0 * 1024.0 * 1024.0) as usize;
    let tide = TideExecutor::new(
        backend.clone(),
        cfg.clone(),
        tide_cfg,
        nvfp4_slots,
        mxfp8_slots,
        ram_budget_bytes,
        Some(pack),
    )
    .expect("create TideExecutor");

    // Engine buffers (activations, KV, prefill scratch) before the expert pools: on Windows the
    // driver backs allocations beyond the resident budget with system memory, in order.
    let mut engine = Inkling2Engine::new(cfg.clone(), dense_gpu, tide, max_ctx)
        .expect("create Inkling2Engine");
    engine.tide.allocate_pools().expect("allocate expert pools");
    let preload_ops = engine.tide.scheduler.plan_pre_prefill(None);
    engine.tide.execute_transfers(&preload_ops, true, false).expect("Mode 1 pre-prefill");
    backend.sync_stream(engine.tide.copy_stream).expect("sync prefill DMA");

    if std::env::var_os("SMALLTIDE_BENCH_MOE").is_some() {
        let n4 = nvfp4_slots as u32;
        let m8 = mxfp8_slots as u32;
        let mut sets: Vec<(String, Vec<u32>, bool)> = Vec::new();
        sets.push(("all 8 missing (launch floor)".to_string(), vec![u32::MAX; 8], false));
        sets.push(("1 expert (slot 0)".to_string(), vec![0, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX], false));
        sets.push(("same expert x8 (slot 0)".to_string(), vec![0; 8], false));
        for st in (0..n4.saturating_sub(8)).step_by(512) {
            sets.push((format!("nvfp4 slots {st}..{}", st + 8), (st..st + 8).collect(), false));
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
    let (pre_r, core_r, coda_r) = engine.tide.scheduler.cache.count_vram_resident();
    let ram_r = engine.tide.scheduler.cache.count_ram_resident();
    if let Some((usage, resident)) = backend.resident_vram_info() {
        let gib = (1u64 << 30) as f64;
        println!(
            "[VRAM residency] process usage {:.2} GiB vs resident budget {:.2} GiB (DXGI){}",
            usage as f64 / gib,
            resident as f64 / gib,
            if usage > resident {
                format!(" -> ~{:.2} GiB of 'VRAM' is backed by system memory (read over PCIe); close other GPU apps or lower --nvfp4-slots/--mxfp8-slots", (usage - resident) as f64 / gib)
            } else {
                String::new()
            }
        );
    }
    let info_ready = backend.device_info().expect("device_info ready");
    println!(
        "[Ready in {:.2}s] VRAM free: {:.2}/{:.2} GiB | KV({} ctx): {:.2} GiB | Experts VRAM: pre={}, core={}/1536, coda={} (slots nvfp4={}, mxfp8={}) | RAM warm: {} | Rescue: {:?}",
        t_start.elapsed().as_secs_f64(),
        info_ready.free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        info_ready.total_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        max_ctx,
        engine.kv_vram_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
        pre_r,
        core_r,
        coda_r,
        nvfp4_slots,
        mxfp8_slots,
        ram_r,
        rescue_policy,
    );

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

    const EOS_ID: u32 = 200_006;
    const EOT_ID: u32 = 199_999;

    for step in 0..max_tokens {
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
        "[Tide] Overall VRAM hit: {:.1}% (Prelude {:.1}%, LoopCore {:.1}%, Coda {:.1}%) | Stage1 sync={}, async={}, skipped={} | Stage2 disk->RAM={}",
        tel.total_vram_hit_rate() * 100.0,
        pct(tel.vram_hits_prelude, tel.vram_misses_prelude),
        pct(tel.vram_hits_loop, tel.vram_misses_loop),
        pct(tel.vram_hits_coda, tel.vram_misses_coda),
        tel.stage1_sync_rescues,
        tel.stage1_async_promotions,
        tel.stage1_skipped_tail_misses,
        tel.stage2_disk_reads,
    );
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
