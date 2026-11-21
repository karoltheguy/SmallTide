# SmallTide Progress & Multi-LLM Handoff Log

> **Handoff Rule for Every LLM Session:** Read `PLAN.md` and this file first. Before ending a session or switching tools, update **Current Status**, **Environment & Build Commands**, **Locked Invariants**, **Benchmarks**, and **Next Actionable Steps**.

---

## Current Status
* **Active Phase (2026-11-21, end of review follow-up session):** R1-R4c done (Tide race fixes +
  transformers reference harness, prefix cache, exact tokenizer, tool calls, batched prefill,
  decode profiling/fixes, host-mapped MXFP8 pool). Decode **~27 tok/s exact** (37 ms/tok), prefill
  **~168 tok/s** on long prompts. See "Session 2026-11-21" below and the benchmark table.
* **Read first - VRAM residency:** on this desktop the process only gets ~7.6 GiB of *resident*
  VRAM (DXGI AvailableForReservation; other apps hold ~5 GiB). Device allocations beyond that are
  silently system memory read over PCIe (~28 GB/s) although hipMemGetInfo counts them as VRAM.
  Earlier "100% VRAM-resident" figures (e.g. coda/prelude experts) were mostly this zero-copy
  tier. `smalltide-cli vram-probe` and `tools/dxgi_budget.py` measure it; run/serve warn.
* **Decode time budget now (GPU, per token):** prelude/coda/MXFP8 experts over PCIe ~20 ms
  (hardware floor until more VRAM is resident), loop-core MoE ~6 ms, everything else ~11 ms.
  CPU launch cost is only ~1.5 ms/token and an injected no-op costs ~1.7 us of GPU time, so
  hipGraph / device-side pos would save at most ~3 ms here; they were deliberately not done yet.
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
* **Config moved to `.env`** (`crates/smalltide-cli/src/settings.rs`, `.env.example`): model dir,
  context, pool sizes, MXFP8 host pool, RAM tier, rescue policy, prefill mode, host/port and the
  diagnostic toggles. Process env overrides `.env`; invalid values exit with a message; old
  config flags are rejected with the variable to set. `run`/`serve` share one engine builder.
* **R4c — Host-mapped MXFP8 pool: every expert resident, exact routing, sync-free decode: DONE**
  - `--mxfp8-host auto|on|off` (auto: on when the device pool cannot hold all 512 MXFP8
    experts). The MXFP8 pool is pinned, device-mapped host memory (`alloc_host_mapped`,
    kernels read it at 26.6 GB/s - the same as the driver's silently spilled "VRAM" at
    28.6 GB/s), filled by CPU copies. All 3072 experts now have a slot: VRAM hit 100%,
    0 sync rescues, 0 skipped experts (exact even in threshold mode), no per-layer host
    syncs (2 syncs/token: routing log + argmax).
  - Decode 27 tok/s exact (previous exact config 23.8 tok/s); long-prompt prefill 1668 tok
    121 -> 168 tok/s (no scratch-wave staging). Reference: identical per-layer errors,
    argmax 6/6, 0 skipped; prefix E2E 5/5 identical (default server mode), tool E2E 16/16.
* **R4b — Decode profiling + fixes; VRAM residency finding** (decode 41.3 -> ~37 ms/tok default config)
  - **Finding (dominant):** on this desktop the process gets only ~7.6 GiB of *resident* VRAM
    (DXGI `AvailableForReservation`; other apps hold ~5 GiB). The driver silently backs every
    device allocation beyond that with system memory, which the GPU then reads over PCIe at
    ~28.6 GB/s, while `hipMemGetInfo` still reports it as VRAM. `smalltide-cli vram-probe`
    (512 MiB chunks: 290 GB/s for the first ~4-7 GiB, then 28.6 GB/s) and
    `tools/dxgi_budget.py` show it; `run`/`serve` now print a `[VRAM residency]` warning.
    With the default auto-sized pools, loop-core experts (slots 0..1535) are resident and
    prelude/coda/MXFP8 experts are effectively a zero-copy host tier: per-layer MoE 0.18 ms
    (loop core) vs 1.3 ms (NVFP4 outer) / 2.2 ms (MXFP8) -> ~16 ms/token of PCIe reads.
    That layout is still the best exact-quality one here (fast-only pools: exact 9.3 tok/s,
    threshold 32 tok/s at 59.6% hit rate vs over-committed exact 23.8 tok/s).
  - Router: single-workgroup kernel (one thread per expert, serial top-8; ~430 us) replaced by
    multi-workgroup f32 logits GEMV + block-parallel top-8 (shared with prefill): -8 ms/tok.
  - FP8 decode via gfx12 `v_cvt_f32_fp8` / `v_cvt_pk_f32_fp8`; FP4 E2M1 via a packed-register
    LUT (an indexed local array compiled to per-lane memory loads): -2.3 ms/tok.
  - Decode MoE v2 (`moe_w13_decode` / `moe_w2_decode` / `moe_post_decode`): 16 B loads, 8
    experts in parallel, renorm + post-MLP sconv + residual fused (-48 launches/token).
  - Engine buffers (activations, KV, prefill scratch) allocated before the expert pools;
    post-decode Tide CPU work deferred to overlap the next token's prelude; routing log read
    back synchronously when the stream is idle (an async D2H + event completed ~30 ms late).
  - Profiling: `SMALLTIDE_SEGTIME=1` coarse GPU timeline segments (reliable),
    `SMALLTIDE_KTIMING=1` per-kernel events (WDDM inflates each by ~40 us), `[Host/tok]` line,
    `SMALLTIDE_BENCH_MOE=1` per-pool-region MoE timing. Kernel *ablation* was misleading
    (skipping kernels changes routing/data and hides latency); use segment timing.
* **R4a — Rows-batched chunked prefill: DONE** (`crates/smalltide-model-inkling2/src/prefill.rs`,
  `kernels/prefill_ops.hip`)
  - Chunks of <= 128 tokens: FP8 rows GEMMs (BM=8), rows attn-res mix / clips (+tau) / sconv scan /
    KV store, multi-row causal attention over the FP8 KV cache (rel bias + sinks), rows router
    top-8, expert-grouped MoE (host-planned pair lists, w13+SiTU / w2 / fixed-order reduce).
    Missing experts are staged into the Scratch Rescue slots in waves: prefill is always exact.
  - Local KV rings grew to window + chunk = 1152 slots (decode attention takes `window`).
  - Verified: reference harness 25 pos (forced routes) layer err 1.7e-3 -> 1.2e-2, argmax 6/6;
    1668-token prompt batched vs token prefill: layers 0-2 <= 1.5e-4 at every position (window,
    ring and chunk boundaries), global layer 5 past 1024 median 6.9e-4; remaining divergence is
    router near-tie flips (first flip at 1.4e-4 upstream diff). Server + prefix cache use it
    (prefix E2E 5/5 identical, tool-call E2E 16/16).
  - `[Host/tok]` decode line: launches, CPU launch time, sync-blocked time (`SMALLTIDE_KTIMING=1`
    adds per-kernel event timing; event overhead on WDDM inflates small kernels ~40 us).
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
* **Next Actionable Steps (ordered by expected decode gain on this machine):**
  - [ ] **More resident VRAM** (environment): close other GPU-heavy apps (Edge, ChatGPT, Discord,
    VS Code, ... held ~5 GiB) and re-check `vram-probe` / the `[VRAM residency]` line. If the
    resident budget covers the NVFP4 pool, prelude/coda MoE drops from ~1.3 to ~0.2 ms/layer
    (~10 ms/token). Optionally make Tide residency-aware (fast vs zero-copy slots; promote hot
    outer experts into fast slots, demote cold loop-core ones).
  - [ ] **`dspark` speculative decoding** (`mtp.safetensors`): verifying k drafted tokens reuses
    the rows-batched prefill path (weights + PCIe-backed experts read once for k tokens) - the
    only lever that amortizes the ~20 ms/token PCIe expert floor. Expected 2-3x.
  - [ ] **Kernel fusion of the ~11 ms non-MoE part**: multi-segment FP8 GEMV (q,k,v,r,gate +
    f32 sinks/dyn taps in one launch; controller down/gain/forget), fused post-projection
    (sconv+clip+vnorm+tau+kv store), rel-bias inside attention, gate inside the attention
    reduce, fused post-attention (sconv+residual+mix+norm).
  - [ ] **WMMA FP8 prefill GEMMs / grouped MoE** (prefill is compute-bound at ~168 tok/s).
  - [ ] hipGraph + device-side pos: decode is now sync-free (2 syncs/token) so a static graph is
    possible, but measured upside is <= ~3 ms/token on WDDM.
  - [ ] **Step 5 (Multi-sequence continuous batching in server)** - unchanged.
---

## Locked Architecture Invariants
1. **Swappable Backend Trait:** ZERO ROCm/HIP types, pointers, or DLL calls outside `crates/smalltide-backend-hip`. All engine/tide/server code depends solely on `smalltide_core::ComputeBackend`.
2. **Precision Policy:** Native experts (`NVFP4` layers `0..19`, `MXFP8` layers `20..23`) + **FP8 E4M3FN** dense layers & `lm_head` (`--fp8-dense` default ON) + **FP8 KV Cache** (default up to 256K).
3. **API Contract:** OpenAI Completion & Chat Completion API on **port `8000`**, served model name **`inkling-2-mini`**.
4. **Rescue Tuning Policy:** Start with minimal Stage 1 RAM→VRAM sync rescue (`Threshold(0.25)`), tunable via `.env` (`SMALLTIDE_RESCUE_MODE`, `SMALLTIDE_RESCUE_THRESHOLD`) or live HTTP (`POST /v1/tide/config`).

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

  # Configuration lives in .env (copy .env.example); only per-request inputs are flags
  .\target\release\smalltide-cli.exe run -p "Your prompt" -n 128 --temp 0

  # OpenAI API server (host/port/rescue/pools from .env; model: inkling-2-mini)
  .\target\release\smalltide-cli.exe serve
  .\target\release\smalltide-cli.exe vram-probe   # resident vs system-memory-backed VRAM

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
| **R4a prefill 1668 tok** (exact) | batched **120.9 tok/s** (13.8 s) vs token **12.2 tok/s** (137.3 s) | 9.9x; 20-tok prompt 52.6 vs 22.4 tok/s |
| Decode host profile (64 tok) | 41.3 ms/tok: 2051 launches (1.36 ms CPU), 5 syncs **35.9 ms blocked** | GPU-bound: kernel count/shape, not CPU launch cost |
| **R4b decode** (default auto pools, threshold 0.25, 56-64 tok) | **26.5-27.1 tok/s** (~37 ms/tok, was 41.3) | GPU segments: MoE outer 16.0 ms (PCIe-backed experts), loop MoE 6.4, router 3.6, out/mix 3.4, pre-attn 2.0, proj 1.6, lm_head 1.5, attn 1.2 |
| **R4c decode** (default: auto pools + host-mapped MXFP8, threshold policy but nothing skipped) | **26.9-27.4 tok/s exact** (37 ms/tok), 0 rescues/skips, 2 syncs/tok | prefill 1668 tok **167.6 tok/s** |
| **Speed profile** (`SMALLTIDE_POOL_BUDGET=resident`, `MXFP8_HOST=off`, `RESCUE_MODE=disabled`; 128 tok) | 68 -> **72 tok/s** (13.8 ms/tok) at 64.8% expert hit rate, 2 syncs/tok | perm-LUT FP4 decode halves MoE (8 experts in-stream 150 -> 83 us); FP8 GEMV 2 loads in flight: lm_head 399 -> 619 GB/s |
| HIP graph replay of one decode token (speed profile) | eager 14.95 ms vs graph 14.22 ms (-5%) | per-kernel cost is GPU drain/refill between dependent kernels, not launch: fusion is the lever |
| **Speed profile, Radeon 26.10.44** (resident budget now = dedicated VRAM - other processes' resident bytes; margin 768 MiB) | **76-85 tok/s at 91.8% expert hit rate** (prelude 99%, loop core 100%, coda 35%) vs 82 tok/s at 64.9% on 26.6.2 | driver 26.6.2 capped adapter-wide residency at ~10.4 of 15.8 GiB (KMD segment bug, ROCm/TheRock#7221); 26.10.44 keeps ~12 GiB resident for one process. 256 MiB margin overshot by 70 MB -> 57 tok/s |
| Fusion after multi-segment GEMV (speed profile, 26.6.2) | attention prep fused 78 -> 80.4 tok/s; residual/mix/norm fused -> 81.9 tok/s (launches/tok 2068 -> 1037) | flash decode measured by duplication: ~1.15 ms/tok (24 us/layer) |
| **Adaptive expert placement** (speed profile, held-out eval of `tools/routing_workload.py`, 16 requests x 160 tok, priors from 37 bootstrap requests, frozen) | decode **88.4 tok/s**, prefill **319 tok/s**, skipped routing weight **2.60%** decode / 5.35% prefill vs old policy 90.3 tok/s, 182 tok/s, **8.03%** / 0% | priors (per-layer routing share, request votes) -> value-per-byte pool split (nvfp4 2445 -> 2138, mxfp8 1 -> 163), warm-up by value, eviction value = (heat + prior) / MiB with hysteresis; MXFP8 pool allocated first + measured fast prefix (it had landed in system memory: 2.5 ms per 8 experts); prefill follows the rescue policy; eviction floor computed once per pass (was ~10 ms CPU/token) |
| **Prefix cache: many conversations** (`tools/test_prefix_cache.py`, exact rescue) | 5/5 identical to cold (`cache_prompt: false`); reuse 50/60 (system prefix), 59/79 (next turn, reasoning dropped), **78/98 after an unrelated side request** (was 0: any miss recycled every snapshot), 97/98 exact repeat (was a miss without a long system prompt) | snapshots carry the global KV of their prefix (16.9 KB/token + 96 MiB rings), restore copies it only for another sequence's snapshot; end snapshot at prompt end - 1; LRU by count / tokens (`SMALLTIDE_PREFIX_SNAPSHOTS=12`, `SMALLTIDE_PREFIX_TOKENS=65536`) |
| Prefill profile (`SMALLTIDE_SEGTIME_PREFILL=1`, speed profile, 2197 tok, 277 tok/s) | per 128-row chunk and layer: MoE 4.25 ms (46%), attention 3.04 ms (33%), projections + router 1.56 ms, mixes 0.29 ms | compute-bound on scalar kernels (MoE ~2.6 TFLOPS / 135 GB/s, GEMM ~4 TFLOPS): WMMA (matrix core) kernels are the lever, not expert loading |
| VRAM residency probe (`vram-probe`) | first ~4 GiB (512 MiB chunks) / ~6.5 GiB (one 12 GiB alloc) at ~290 GB/s copy, rest **28.6 GB/s** | DXGI: budget 15.06 GiB, AvailableForReservation 7.65 GiB; other apps ~5 GiB |
| R4b verification | reference (forced routes) identical: layer err 1.7e-3 -> 1.2e-2, argmax 6/6; stress (8.7% skipped) argmax 40/40; bench-gpu parity unchanged | |
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
