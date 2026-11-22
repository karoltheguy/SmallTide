# SmallTide: Engine Architecture & Implementation Plan

## 1. Goal & Target Envelope

Build **SmallTide**, a native Windows LLM inference engine in **Rust** bringing **TideLM's adaptive expert streaming** to consumer discrete GPUs with **2-stage streaming** (`NVMe Disk -> Host RAM -> GPU VRAM`), targeting **Inkling-2-Mini** first with peak **ROCm/HIP (`gfx1201`)** performance while keeping the compute backend cleanly swappable behind a Rust trait (`ComputeBackend`).

### Hardware Baseline (Reference PC)
| Resource | Spec | Realistic Usable Budget (Windows) | Effective Bandwidth |
| :--- | :--- | :--- | :--- |
| **GPU** | AMD Radeon RX 9070 XT (`gfx1201`, RDNA 4) | ~14.0–14.5 GiB VRAM (after WDDM / DWM) | 640 GB/s GDDR6 |
| **PCIe Link** | PCIe 4.0 x16 or 5.0 x16 | Stage 1 DMA (RAM -> VRAM) | ~24 GB/s (Gen4) / ~48 GB/s (Gen5) |
| **Host RAM** | 32 GB System RAM | ~18–22 GiB (keeping OS stable) | ~50–80 GB/s (DDR5) |
| **Storage** | NVMe SSD | Stage 2 I/O (Disk -> RAM) | ~3–7 GB/s seq / ~1.5–3.5 GB/s aligned random |

### Target Model Math: Inkling-2-Mini (Native Weights + FP8 Dense)
* **Looped Architecture:** 24 physical decoder layers executed as **48 logical layers** (`6 + 3 x 12 + 6`):
  * **Prelude (Layers 0–5):** Executed once.
  * **Loop Core (Layers 6–17):** Executed **3x per token**, reusing the exact same physical weights, with per-pass KV/conv caches and gated rank-64 cross-pass message passing.
  * **Coda (Layers 18–23):** Executed once.
* **Hybrid Attention:** Per 6-layer block: 5 sliding-window local attention layers (window = 1,024 tokens, 24 Q heads, 8 KV heads, relative position logits, output gates) + 1 global full-attention layer (with dynamic K/V convolutions). Only **8 of 48 logical layers** grow linearly with context length.
* **MoE Footprint:** `d_model = 2304`, `d_ff = 1152` (SwiGLU), **128 routed experts** per layer, **Top-8 active**:
  * Params per expert: `3 x 2304 x 1152 = 7.96M` params.
  * **Layers 0–19 (NVFP4, `[1, 16]` FP8 block scales + `amax / outer_scale_denominator`):** **4.27 MiB/expert** (`4,481,024 B` 4 KB-aligned `.stide` slab) -> ~0.158 ms PCIe Gen4.
  * **Layers 20–23 (MXFP8, 2D `[32, 32]` UE8M0 block scales verified in `hf_quant_config.json`):** **7.60 MiB/expert** (`7,974,912 B` 4 KB-aligned `.stide` slab) -> ~0.289 ms PCIe Gen4.
  * **Total Routed Experts (3,072 experts):** **14.48 GiB** (`10.68 GiB` NVFP4 + `3.80 GiB` MXFP8).
  * **Loop Core Alone (Layers 6–17, 1,536 experts):** Only **6.41 GiB** — yet executes **36 of 48 logical passes (75% of all MoE compute!)**.

### Finalized Design Choices (Locked)
1. **FP8 Dense & `lm_head` (Default-ON, `--fp8-dense`):** Cast native BF16 dense attention projections, loop controllers, and `lm_head` to **FP8 E4M3FN** on load.
2. **256K FP8 Context Default (`max_context_len = 262,144`) + Elastic KV <-> Expert Slab Sharing:**
   * 40 local layers fixed at `0.08 GiB` FP8; 8 global layers = `4.00 GiB` FP8 at full 256K (`4.58 GiB` with `dspark`'s 9th global layer).
   * Shared unified VRAM pool (`~11.85 GiB`): expands Expert Tide up to `~11.2 GiB` at short context and gracefully evicts cold outer experts back to RAM as context grows toward 256K+ while keeping **100% of the Loop Core (`6.41 GiB`) resident in VRAM**! (Optional `--kv-dtype q4` stretches to 512K+ in VRAM.)
3. **Optional `dspark` Speculative Decoding (Auto-enabled if `mtp.safetensors` is present):**
   * Loads the official `606.3 MiB` `mtp.safetensors` 3-layer drafter (`target_layer_ids: [5, 23, 41, 48]`, `block_size: 8`).
3. **VRAM + RAM Fit (100% Native Weights in VRAM + RAM):**
   * **VRAM (~14.0–14.5 GiB usable):** Dense FP8 (~1.5–1.8 GiB) + KV/conv cache at 32K FP8 (~0.6 GiB) + Workspace/Scratch (~0.7 GiB) + optional `dspark` (~0.65 GiB) -> leaves **~10.2–11.5 GiB for Expert Resident Pool (~70%–79% of all 14.60 GiB experts)**.
   * **Loop Core (6.41 GiB) = 100% VRAM resident**, plus **~3.8–5.1 GiB** for the 12 outer layers (`8.19 GiB` total) -> **~95%–98% effective logical-layer VRAM hit rate**.
   * **Host RAM:** Inclusive mode uses ~16–17 GiB of 32 GB RAM (~15 GB free for Windows); Exclusive Loop-Core pinning frees the 6.41 GiB RAM duplicate -> **~9.7–11 GiB RAM total**.
   * **Stage 2 (Disk -> RAM):** Cold-start only for native Inkling-2-Mini; automatically activates for `Q8_0` (~28 GB), larger MoEs (50B–200B+), or `--ram-budget <size>`.

### Target Decode Speed (RX 9070 XT, 640 GB/s Peak)
* **With FP8 Dense + `lm_head` (~4.1–4.5 GB read/tok, sync rescue OFF):**
  * At 50% BW (320 GB/s): **~71 – 78 tok/s**
  * At 70% BW (448 GB/s, tuned HIP + `hipGraph`): **~100 – 109 tok/s**
* **With `dspark` Speculative Decoding (~2.2x avg multiplier):** **~150 – 230+ tok/s**

---

## 2. Swappable Compute Backend Strategy (Rust Trait + Dynamic ROCm/HIP)

* **Strict Isolation (`smalltide-core::ComputeBackend`):** Zero ROCm/HIP types or FFI leak outside `crates/smalltide-backend-hip`. All scheduling, caching, model execution, and I/O operate exclusively on opaque handles (`DeviceBuf`, `PinnedBuf`, `Stream`, `Event`, `Graph`).
* **Dynamic `amdhip64_7.dll` + Precompiled `.hsaco` (inspired by `hipfire`):**
  * Compile `.hip` kernels offline via `hipcc --genco --offload-arch=gfx1201` into `.hsaco` code objects.
  * Load `amdhip64_7.dll` (fallback `amdhip64_6.dll` / `amdhip64.dll`) dynamically at runtime via `libloading` and launch kernels via `hipModuleLoadData` / `hipModuleLaunchKernel`.
  * Eliminates MSVC <-> `hipcc` static-link friction and allows the same binary to gracefully load `smalltide-backend-cuda` or `smalltide-backend-vulkan` on non-AMD GPUs in Phase 5!
* **RDNA 4 (`gfx1201`) Kernel Strategy:**
  * **Decode (Batch = 1):** Bandwidth-bound Wave32 fused GEMV kernels: in-register `NVFP4 -> FP16/FP32`, `MXFP8 -> FP32`, `FP8 -> FP32`, plus **`hipGraph`** capture with GPU pointer indirection (`slot_table[layer][expert_id]`) to eliminate Windows WDDM launch overhead without invalidating graphs on expert swaps.
  * **Prefill / `dspark` Verify (Batch > 1):** Unpack `NVFP4 -> FP8` in registers and feed RDNA 4's native hardware **`v_wmma_f32_16x16x16_fp8_fp8`**!

---

## 3. TideLM 2-Stage Streaming & Expert Loading Modes

All 4 TideLM expert-loading modes operate across **Stage 1 (RAM <-> VRAM)** and **Stage 2 (Disk <-> RAM)**:
1. **Preload — Before Prefill:** Seed VRAM/RAM with persisted hot-set priors, weighted by **Loop-Core 3x reuse** (Layers 6–17 get top residency priority).
2. **Preload — Between Prefill & Decode (Boundary Tide Rebalance):** Record per-layer router hit weights during prefill; bulk-evict cold VRAM slots (zero writeback) and stream prompt-hot experts before token 1 decodes.
3. **Async Load (Paced Background Tide Rise):**
   * Stage 2 worker thread: Win32 `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` / Windows 11 IoRing reads from NVMe into Tier 1 RAM (bypassing Windows OS file cache!).
   * Stage 1 copy stream: `hipMemcpyHtoDAsync` from bounded **512 MB–1 GB pinned staging ring** (`hipHostMalloc`) into Tier 0 VRAM overlapped with active compute, capped per token to prevent PCIe/TDR stalls.
4. **Sync Load (Immediate Rescue — Configurable):**
   * **Exact Mode:** Rescue all missing Top-8 experts; launch H2D DMA on copy stream immediately after router while hit experts execute concurrently on compute stream (**split-batch in-layer overlap**).
   * **Threshold / Pure-Async Mode (`--rescue-threshold` / `--no-rescue`):** Skip tail misses below threshold (renormalizing remaining resident experts) and promote asynchronously for subsequent tokens.

---

## 4. Cargo Workspace Layout

```text
smalltide/
├── Cargo.toml
├── PLAN.md
├── PROGRESS.md                     # Multi-LLM handoff & milestone progress log
├── crates/
│   ├── smalltide-core/             # Types, DType (NVFP4/MXFP8/FP8/BF16), ComputeBackend trait, telemetry
│   ├── smalltide-io/               # Win32 IoRing / NO_BUFFERING async reader, Safetensors/GGUF, aligned repacker
│   ├── smalltide-tide/             # 3-Tier TideLM Cache + Scheduler (all 4 loading modes + Loop-Core weighting)
│   ├── smalltide-model-inkling2/   # 24->48 looped graph, gated rank-64 messaging, local/global conv attn, dspark
│   ├── smalltide-backend-hip/      # Phase 1: Dynamic amdhip64_7.dll + .hsaco (gfx1201 WMMA, hipGraph, async DMA)
│   ├── smalltide-backend-cpu/      # Pure-Rust reference backend for unit tests & logit verification
│   ├── smalltide-backend-cuda/     # (Phase 5a) Swappable CUDA backend implementing ComputeBackend
│   ├── smalltide-backend-vulkan/   # (Phase 5b) Swappable Vulkan (ash) backend for AMD/NVIDIA/Intel/Apple
│   └── smalltide-cli/              # CLI runner, benchmarks, OpenAI-compatible HTTP server
└── benchmarks/                     # Standalone Phase 0 microbenchmarks
```

---

## 5. Phased Roadmap

* **Phase 0 — Environment, Model Discovery & Microbenchmarks:**
  * Verify Rust + MSVC Build Tools + ROCm HIP SDK (`hipcc`) on Windows.
  * Inspect real `Inkling-2-Mini` `config.json`, safetensors tensor names/shapes/scales, and `mtp.safetensors` (`dspark`).
  * Benchmark: (a) `amdhip64_7.dll` async `hipMemcpyHtoDAsync` overlap with compute + `hipGraph` dispatch latency; (b) Win32 `FILE_FLAG_NO_BUFFERING | OVERLAPPED` / IoRing random 4.27 MiB & 7.83 MiB NVMe latency.
* **Phase 1 — `ComputeBackend` Trait + `smalltide-backend-hip` (`gfx1201`) + CPU Reference:**
  * Lock `ComputeBackend` trait in `smalltide-core`.
  * Build `.hip -> .hsaco` kernel pipeline and dynamic `amdhip64_7.dll` loader.
  * Implement & unit-test against CPU reference: RMSNorm, RoPE, FP8 GEMV/WMMA, NVFP4 dequant-GEMV & NVFP4->FP8 WMMA, MXFP8 GEMV/WMMA, Router Top-K, fused SwiGLU, hybrid local/global conv attention.
* **Phase 2 — `smalltide-model-inkling2` Exact Forward Pass:**
  * Implement 24-physical / 48-logical loop, gated rank-64 cross-pass messaging, 40 local (1,024-window) + 8 global conv KV caches, and load-time BF16->FP8 dense/`lm_head` conversion.
  * Verify token-for-token logits against reference implementation (`llama.cpp` / `vLLM`).
* **Phase 3 — Stage 1 Streaming (RAM <-> VRAM) + TideLM Scheduler + HIP Graphs:**
  * Implement VRAM Expert Slot Pool, Scratch Rescue Ring, GPU Slot Indirection Table, Exclusive Loop-Core pinning + Inclusive outer-layer RAM backing.
  * Implement all 4 TideLM loading modes + `--no-rescue` / `--rescue-threshold` + `hipGraph` decode capture + telemetry.
* **Phase 4 — Stage 2 Streaming (Disk <-> RAM) + `dspark` Speculative Decoding:**
  * Implement Win32 `NO_BUFFERING` / IoRing async reader + aligned expert repacker (`--ram-budget` stress test).
  * Integrate `mtp.safetensors` (`dspark`) speculative drafting & batched verification when available.
* **Phase 5 — Multi-GPU Expansion & Serving:**
  * Add `smalltide-backend-cuda` and `smalltide-backend-vulkan` (`ash`) behind the unchanged `ComputeBackend` trait.
  * OpenAI-compatible HTTP server (`axum`) + optional lazy `mmproj` vision support.
