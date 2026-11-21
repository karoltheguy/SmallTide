# SmallTide Progress & Multi-LLM Handoff Log

> **Handoff Rule for Every LLM Session:** Read `PLAN.md` and this file first. Before ending a session or switching tools, update **Current Status**, **Environment & Build Commands**, **Locked Invariants**, **Benchmarks**, and **Next Actionable Steps**.

---

## Current Status
* **Active Phase:** **Step 3 (Live 48-Layer Inkling-2-Mini Inference) & Step 4b (OpenAI HTTP/SSE Server on port 8000 as `inkling-2-mini`) COMPLETE & LIVE VERIFIED!** -> Next up: Kernel Decode Speed Optimization (50 ms/tok -> ~18 ms/tok) + `dspark` (`mtp.safetensors`) Speculative Decoding + Multi-Sequence Continuous Batching in Server!
* **Date:** 2026-11-21
* **Reference Repos & Model Weights On Disk:**
  * **Inkling-2-Mini (`models/Inkling-2-Mini/`):** **100% DOWNLOADED & PACKED** (`model-00001-of-00001.safetensors` = `18.18 GiB`; **`experts.stide` = all 3,072 experts 4 KB-aligned packed in 19.03s**; `mtp.safetensors` = `635.8 MB` `dspark`; `tokenizer.json`; `config.json`; `hf_quant_config.json`; `chat_template.jinja`).
  * **TideLM Reference:** Cloned to `c:\Users\raspb\git\tideLM`.
  * **Inkling-2-Mini Math & Tensor Layout Verified (Bit-Exact against `llama.cpp` `inkling2.cpp`, `inkling2-loop.cpp`, `conversion/inkling2.py`):**
    - `w13_weight` on disk is **interleaved** (`row 2*r = gate[r]`, `row 2*r+1 = up[r]`).
    - Global `k_sconv.static_w` / `v_sconv.static_w` (`[4, 1024]`) are flipped & transposed to `[1024, 4]` on load in `dense_loader.rs`; `rel_logits_proj.proj` (`[16, 1024]`) is transposed to `[1024, 16]` on load.
    - `middle_cycle_cross_pass_queries` + `middle_cycle_cross_pass_kv_queries` are packed per core layer into `[4, 2304] = [q[2i], kv[2i], kv[2i+1], q[2i+1]]` (`[Q, K, V, FFN]`).
    - No RoPE: positions via `wr_du` + `rel_logits_proj` + per-query `attn_sink_proj`, plus global `tau = 1 + 0.1*ln(max(1, (pos+1)/8192))` on Q & R.
    - Exact **SiTU** MoE activation (`4.0*tanh(g/4.0)*sigmoid(g) * 25.0*tanh(u/25.0)`), **Sigmoid + bias-aided Top-8** router (`route_scale = 8.0`), residual `convolve` (`1/sqrt(5)` static, `1/3` dynamic), `q/rel_l2_clip` (per-channel `[n_heads, head_dim]` scale applied BEFORE L2 norm + `1e-6`), parameter-free Value RMSNorm, multi-query `attn_res` (`1/sqrt(2304)` scaled), and rank-64 Loop Core `control` (`(1-tanh(forget))*cur + tanh(gain)*msg`).
* **Completed:**
  - [x] **Step 3 — Live 48-Layer Inkling-2-Mini Engine (`crates/smalltide-model-inkling2/src/engine.rs`):**
    - 100% backend-agnostic `Inkling2Engine<B: ComputeBackend>` (zero HIP types outside `smalltide-backend-hip`).
    - Coherent, verified English + Inkling-2-Mini `<|content_thinking|>` / `<|content_text|>` generation on RX 9070 XT!
    - Zero-sync routing fast path for 100%-VRAM-resident layers (all 20 NVFP4 layers `0..19` = 44/48 logical layers on RX 9070 XT!).
    - Tunable Rescue Knob (`--rescue-mode disabled|threshold|exact`, `--rescue-threshold <0..1>` evaluated on normalized routing share `[0..1]`, default `Threshold(0.25)`).
  - [x] **Step 4b — OpenAI-Compatible HTTP/1.1 + SSE Server (`crates/smalltide-cli/src/server.rs`):**
    - `smalltide-cli serve --port 8000` serving model **`inkling-2-mini`**:
      - `GET  /v1/models` (`inkling-2-mini`)
      - `POST /v1/completions` (streaming SSE + non-streaming)
      - `POST /v1/chat/completions` (streaming SSE + non-streaming, exact `chat_template.jinja`, `reasoning_effort`, splits `<|content_thinking|>` -> `reasoning_content` & `<|content_text|>` -> `content`)
      - `GET  /v1/tide/stats` (live hit rates, Stage 1/2 counters, VRAM, tok/s)
      - `POST /v1/tide/config` (runtime rescue knob tuning without reloading weights!)
      - Exact multi-turn KV/sconv prefix reuse when follow-up requests extend the cached sequence.
  - [x] **Step 1 & Step 2 (`smalltide-io`, `smalltide-tide`, `smalltide-backend-hip`):**
    - `BpeTokenizer` (`200,058` tokens, `0.26s` load, exact round-trip).
    - `Inkling2DenseHost` -> `Inkling2DenseGpu`: all 24 layers + loop controllers + FP8 `lm_head` = **`1.10 GiB` VRAM** (`embed` in RAM = `883 MiB`).
    - Full TideLM 2-stage adaptive loader (`TideCache`, `TideScheduler`, `Stage2IoWorker`, `TidePriors`, `TideExecutor`, `PagedKvManager`, `ContinuousBatchScheduler`).
    - Complete `gfx1201` `.hsaco` kernel library (`core_ops.hip`, `moe_ops.hip`, `attn_ops.hip`, `inkling2_ops.hip`, `batched_ops.hip`) loaded dynamically via `C:\Windows\System32\amdhip64_7.dll`.
* **See `REVIEW_2026-11-21.md` for reference-checked findings, Tide runtime bugs and the fix plan.**

### Session 2026-11-21 (Claude, local RX 9070 XT) — review follow-up
* **R1 — Tide runtime fixes (review bugs 1–6) + transformers reference harness: DONE**
  - Bug 1 (async race): GPU slot table is now written only on the compute stream (dedicated pinned
    upload ring); copy stream waits on a compute fence before overwriting any slot; new experts stay
    `vram_pending` (unpublished, never a CPU hit) until their DMA event completes (`poll_pending` at
    token start) or the compute stream GPU-waits on it (sync rescue / `wait_pending`).
  - Bug 2: current top-k is `protected` from eviction during `plan_layer_dispatch`.
  - Bug 3: `promote_to_vram(.., allow_disk_direct)`; periodic rebalance + Mode 3 queue async Disk->RAM
    instead of blocking NVMe reads.
  - Bug 4: RAM slabs are tagged with their owning `ExpertId`; mismatch -> blocking re-read + counter
    `stage1_ram_fallback_reads` (never stale bytes).
  - Bug 5: router writes into a per-logical-layer routing log; one async D2H per token feeds heat /
    prefill scores for fully-resident layers too.
  - Bug 6: one 64 B D2H per partial layer (ids+weights), one 512 B D2H for argmax.
  - New unit tests: `test_sync_rescue_never_evicts_current_topk`, `test_pending_promotion_is_not_published`,
    `test_async_paths_never_block_on_disk`.
  - **Reference harness** `tools/inkling2_loop_ref.py`: runs the engine's dumped tokens
    (`SMALLTIDE_DUMP_DIR=...`) through transformers PR #51050's own `Inkling2LoopTextModel`
    (head 430050f, fp32 eager, real checkpoint, experts lazily dequantized by the PR's
    `Inkling2LoopDequantizeExperts`) and compares all 48 logical layer outputs + logits.
    Flags: `--emulate-fp8-dense`, `--emulate-fp8-kv`, `--force-engine-routes` (uses the engine's
    expert choice incl. skipped-expert mask, isolating arithmetic from near-tie router flips).
    Setup: `python -m venv refenv; pip install torch --index-url https://download.pytorch.org/whl/cpu;
    pip install safetensors numpy "transformers @ https://github.com/huggingface/transformers/archive/430050f6f71195b76004b5d61bbb446ec0508fb3.zip"`.
* **R2b — Tokenizer now matches HF `tokenizers` exactly: DONE** (found while porting the chat template)
  - Old pre-tokenizer split only on spaces/newlines and ignored `ignore_merges`: digits, whitespace
    runs and code tokenized non-canonically (Rust source: 28,457 vs 15,538 reference tokens).
  - New: hand-implemented o200k `Split` regex (same alternation/backtracking semantics) over a
    generated Unicode general-category table (`tools/gen_unicode_cats.py` -> `unicode_cats.rs`),
    id-based BPE merges, `ignore_merges`, `Utf8Stream` incremental decoding for streaming.
  - `tools/test_tokenizer.py` (HF tokenizers 0.23.3): 20/20 adversarial cases + 59 repo files
    (268K tokens) identical.
* **R3 — OpenAI tool calling: DONE**
  - `crates/smalltide-cli/src/chat_format.rs`: byte-exact port of `chat_template.jinja`
    (tool_declare with recursively sorted specs, assistant `tool_calls` as
    `<|message_model|>NAME<|content_invoke_tool_json|>{"name":..,"args":{..}}<|end_message|>`,
    tool-role headers incl. `tool_call_id` lookup, one block per content part, effort default
    **0.9** when unspecified, template refusals -> HTTP 400). Order-preserving JSON with Python
    `json.dumps` output (float repr, escapes, separators, sort_keys). String `arguments` from
    OpenAI clients are parsed first ("canonicalize upstream").
  - `AnswerParser`: streams reasoning/content, parses tool-call bodies; `StopMatcher` holds back
    partial stop strings; incremental UTF-8 (fixes `Bras�lia`).
  - Server: `tools`, `tool_choice` (`auto`/`required`=auto, `none`, named function forced by
    ending the prompt with `<|message_model|>NAME<|content_invoke_tool_json|>`),
    `parallel_tool_calls:false`, `tool_calls` in non-streaming + streaming deltas,
    `finish_reason:"tool_calls"`. `SMALLTIDE_PREFIX_DEBUG=1` logs prefix-cache decisions.
  - `tools/test_chat_template.py`: **27/27** cases byte-identical to transformers'
    `apply_chat_template` environment (incl. refusals). `tools/test_tool_calls.py` (real model,
    threshold mode): **16/16** pass — auto call, tool-result turn (prefix cache extends the
    re-templated history fully: 305/343 cached), streaming call, forced call, none, utf-8.
* **R2 — Server prompt prefix cache: DONE**
  - `Inkling2Engine::{snapshot_state, restore_state}`: rewindable state = 40 local KV rings + scales
    and all 48x4 short-conv histories (86 MiB), copied stream-ordered to pinned host RAM. Global KV
    is append-only and needs no snapshot.
  - `crates/smalltide-cli/src/prefix_cache.rs` (unit-tested planner): Extend live sequence /
    Restore latest snapshot <= LCP / Reset. Snapshots at the first `<|message_user|>` (shared system
    + tool prefix) and at every prompt end; LRU of 4. Always re-processes >= 1 prompt token.
  - Usage reports `prompt_tokens_details.cached_tokens`; `/v1/tide/stats` -> `prefix_cache`.
  - E2E `tools/test_prefix_cache.py` (server `--rescue-mode exact`): boundary restore, prompt-end
    restore (client drops reasoning) and identical-repeat rewind all produce byte-identical
    reasoning+content vs cold runs.
* **Next Actionable Steps:**
  - [ ] **Step 4a (Performance Pass #1 — 2×–2.5× Decode Speedup):**
    - Run `--profile` to inspect the exact `50 ms/tok` breakdown.
    - Parallelize `smalltide_router_topk8_f32` across 32 workgroups (currently 1 WG!), combine `topk_ids` + `topk_weights` D2H into 1 readback, fuse small elementwise/norm launches, and evaluate HIP Graph capture for the 36 pinned Loop Core layers!
    - Wire batched prefill (`batched_ops.hip` BM=8 GEMM + batched MoE) so prompt prefill scales from `20 tok/s` to `200+ tok/s`.
  - [ ] **Step 4c (`dspark` Speculative Decoding from `models/Inkling-2-Mini/mtp.safetensors`):**
    - Load `mtp.safetensors` (54 BF16 tensors, 3 dense layers, `fuse_weight [4,4]` fusing hidden states at logical layers `[5, 23, 41, 48]`, `block_size: 8`, `markov_rank: 256`).
    - Draft 5–7 tokens and batch-verify with `batched_ops.hip` for 2–3× decode speedup!
  - [ ] **Step 5 (True Multi-Sequence Continuous Batching in Server):**
    - Wire `ContinuousBatchScheduler` + `PagedKvManager` + `smalltide_paged_flash_decode_fp8_split` into `server.rs` so concurrent HTTP requests batch together on the GPU in parallel.

---

## Locked Architecture Invariants
1. **Swappable Backend Trait:** ZERO ROCm/HIP types, pointers, or DLL calls outside `crates/smalltide-backend-hip`. All engine/tide/server code depends solely on `smalltide_core::ComputeBackend`.
2. **Precision Policy:** Native experts (`NVFP4` layers `0..19`, `MXFP8` layers `20..23`) + **FP8 E4M3FN** dense layers & `lm_head` (`--fp8-dense` default ON) + **FP8 KV Cache** (default up to 256K).
3. **API Contract:** OpenAI Completion & Chat Completion API on **port `8000`**, served model name **`inkling-2-mini`**.
4. **Rescue Tuning Policy:** Start with minimal Stage 1 RAM→VRAM sync rescue (`Threshold(0.25)`), tunable via CLI (`--rescue-mode`, `--rescue-threshold`) or live HTTP (`POST /v1/tide/config`).

---

## Verified Host Environment & Build Commands (Windows)
* **GPU:** `AMD Radeon RX 9070 XT` (`gfx1201`, RDNA 4, `15.77 GiB free / 15.92 GiB total`, 640 GB/s peak)
* **HIP Runtime:** `C:\Windows\System32\amdhip64_7.dll` (`70051384`)
* **HIP Compiler:** `C:\Program Files\AMD\ROCm\7.2\bin\hipcc.exe` (`HIP 7.2.60201`, `clang 21.0.0git`)
* **Rust:** `rustc 1.100.0`, `cargo 1.100.0` (`C:\Users\raspb\.cargo\bin\cargo.exe`)
* **Working Commands:**
  ```powershell
  # Build & unit-test entire workspace (offline)
  C:\Users\raspb\.cargo\bin\cargo.exe test --offline --release --workspace
  C:\Users\raspb\.cargo\bin\cargo.exe build --offline --release -p smalltide-cli

  # Run live CLI generation (tunable rescue knob)
  .\target\release\smalltide-cli.exe run --rescue-mode threshold --rescue-threshold 0.25 --prompt "Your prompt"

  # Start OpenAI API server on port 8000 (model: inkling-2-mini)
  .\target\release\smalltide-cli.exe serve --port 8000 --rescue-mode threshold --rescue-threshold 0.25

  # Benchmarks & diagnostics
  .\target\release\smalltide-cli.exe probe
  .\target\release\smalltide-cli.exe bench-gpu
  .\target\release\smalltide-cli.exe bench-nvme
  .\target\release\smalltide-cli.exe load-test models\Inkling-2-Mini
  ```

---

## Benchmark & Verification Log

| Benchmark / Check | Measured Result | Parity / Notes |
| :--- | :--- | :--- |
| **R1 baseline (before fixes)**, `run -p "What is the capital of France?" -n 64 --temp 0` | Prefill 20.5 tok/s, decode **21.99 tok/s** | Coherent ("Paris."), 22 sync rescues, 254 skipped |
| **R1 after Tide fixes**, same command | Prefill 22.4 tok/s, decode **23.76 tok/s** | Byte-identical output, 22 sync, 250 skipped |
| **Reference check, exact rescue, 25 pos** (fp8 dense+KV emulated, engine routes) | layer rel-err 1.7e-3 (L0) -> 1.4e-2 (L47), cos >= 0.9986, argmax **6/6**, KL <= 4e-5 | Smooth growth, no jump at loop-pass/coda boundaries |
| Reference check, no emulation, free routing | argmax **6/6**, KL <= 3e-3 | Per-layer 1-7% (FP8 KV/weights + router near-tie flips) |
| **R2 prefix cache E2E** (5 chat turns, 60-98 tok prompts, greedy, exact) | cached 50/60, 60/79, 79/98 tok; warm 3.0-4.8 s vs cold 5.1-7.4 s | 5/5 byte-identical to cold; snapshot 86 MiB pinned |
| **Tide stress** (`--nvfp4-slots 1600 --mxfp8-slots 96`, threshold 0.25, 68 pos) | 15.3% experts skipped, 267 sync rescues, async churn every token; layer rel-err <= 4.0e-2, cos >= 0.9995, argmax 46/48 (2 near-ties) | GPU computed exactly the CPU-published expert set (no stale-slot reads) |
| **Live 48-Layer Decode (`exact`, RX 9070 XT)** | **18.74 tok/s** (`99.5%` VRAM hit: Pre 100%, Core 100%, Coda 95.8%) | Coherent English + thinking format, `253` Stage 1 sync rescues |
| **Live 48-Layer Decode (`threshold=0.25`, RX 9070 XT)** | **19.80 tok/s** (`99.4%` VRAM hit: Pre 100%, Core 100%, Coda 95.1%) | Coherent 2-sentence answer, clean EOS, only `26` Stage 1 sync rescues (10× fewer!) |
| **Live Prefill (`threshold=0.25`, single-token)** | **20.02 tok/s** (Mode 2 boundary rebalance: 95 swaps in `120.2 ms`) | Will jump to 200+ tok/s once batched GEMM prefill is wired |
| **Startup (Load FP8 Dense + Mode 1 Pre-Prefill)** | **18.42s** (`2560/2560` NVFP4 + `326/512` MXFP8 in VRAM, `1536` RAM warm) | `0.95 GiB` VRAM free with `16K` KV (`0.34 GiB`) |
| HIP Async H2D DMA — NVFP4 (`4.27 MiB`) | **0.158 ms/expert (28.30 GB/s)** | PCIe 4.0 x16 saturated (`amdhip64_7.dll`) |
| HIP Async H2D DMA — MXFP8 (`7.83 MiB`) | **0.289 ms/expert (28.41 GB/s)** | 6×–10× cheaper than NVMe rescue |
| NVMe Direct Read — NVFP4 (`4.27 MiB` aligned) | **0.960 ms/expert (4.67 GB/s)** | Win32 `NO_BUFFERING \| OVERLAPPED`, QD=1 |
| WDDM Kernel Launch (240 ops = 48 layers × 5) | **0.504 ms raw / 0.496 ms HIP Graph** | ROCm 7.2 WDDM overhead ≈ 2.1 µs/kernel |
| FP8 E4M3 `lm_head` `[201024×2304]` | **1.119 ms — 415.4 GB/s (64.9%)** | `max_abs_diff = 2.67e-5` vs CPU |
| NVFP4 GEMV `[147456×2304]` (192 MB L3-bypass) | **0.521 ms — 368.3 GB/s (57.5%)** | `max_abs_diff = 1.19e-6` vs CPU |
| MXFP8 GEMV `[73728×2304]` (`[N, K/32]` scales) | **0.466 ms — 365.2 GB/s (57.1%)** | `max_abs_diff = 3.91e-3` (~1e-6 rel) vs CPU |
| **Indirect Top-8 MoE NVFP4 (8/8 hits)** | **0.134–0.151 ms/layer** | `max_abs_diff = 1.79e-7` vs CPU |
| **Indirect Top-8 MoE NVFP4 (6/8, 2 skipped+renorm)** | **0.107–0.111 ms/layer** | `max_abs_diff = 2.68e-7` vs CPU |
| **Indirect Top-8 MoE MXFP8 (8/8 hits)** | **0.271–0.277 ms/layer** | `max_abs_diff = 2.44e-4` vs CPU |
| FP8 Attn — Local 1K / Global 32K / Global 256K | **0.140 ms / 0.588 ms / 3.055 ms** | GQA-fused `exp2f` Split-K FlashDecoding |
| Workspace unit tests (`smalltide-io`, `smalltide-tide`) | **8/8 PASS (0.01s)** | Concurrent prefix-sharing Paged KV, Elastic 327K KV, Loop Core 100%, Stage 2 spillover, Scratch Rescue Ring, Periodic Rebalance, TidePriors, BF16->FP8, `.stide` alignment |
