# SmallTide

**Adaptive MoE inference for [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (30B total / 3.8B active) on a single 16 GB AMD Radeon RX 9070 XT on Windows.**

Inkling-2-Mini's routed experts (~14.7 GB across 3,072 NVFP4 & MXFP8 experts) plus dense weights, KV cache, and drafter (~6+ GB) exceed a 16 GB consumer GPU. SmallTide brings [TideLM](https://github.com/0xBakeer/tideLM)-style adaptive expert residency and custom RDNA 4 (`gfx1201`) HIP kernels to Windows: it keeps the highest-value experts resident in dedicated GDDR6 VRAM, promotes hot misses asynchronously from host RAM/NVMe, rescues high-weight misses inline on the GPU, and accelerates both greedy and sampled generation with Inkling-2-Mini's 3-layer DSpark speculative drafter.

Pre-built Windows releases embed the compiled `gfx1201` `.hsaco` GPU kernels inside a single `smalltide-cli.exe` — **no Rust toolchain, no ROCm SDK, and no Python environment required** to install and serve.

## Measured Performance (RX 9070 XT 16 GB, Windows 11)

Measured with `bench/bench.py` (median of 3 runs after 1 warmup; one-shots: 1 run; `temperature=0.6`, `effort=none`, DSpark speculative decoding on, 128K shared paged KV pool, rescue gate 0.15 / Loop Core 0.10, rescue budget 24 per token, Adrenalin 26.10.44):

| Workload | Prompt / output | Decode | TTFT | Skipped routing weight | Inline rescues / tok | Disk reads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `prose` (essay) | 58 / 1,024 | **68.0 tok/s** | 228 ms | 1.43% | 1.3 | 0 |
| `code` (LRU cache module + tests) | 74 / 1,024 | **68.9 tok/s** | 298 ms | 2.54% | 3.5 | 0 |
| `edit` (refactor a 1.4K-token file) | 1,425 / 1,024 | **74.7 tok/s** | 2.24 s | 3.40% | 5.5 | 0 |
| `random` (8K-token random prompt) | 8,266 / 372 | **47.9 tok/s** | 6.71 s | 8.58% | 10.8 | 0 |
| `angry-birds` (one-shot HTML game) | 76 / 8,192 | **89.1 tok/s** | 418 ms | 2.42% | 2.7 | 0 |
| `mario` (one-shot HTML game) | 66 / 6,678 | **85.7 tok/s** | 436 ms | 2.52% | 2.9 | 0 |

*VRAM hit rates on these runs: Prelude 91–93%, Loop Core 99%, Coda 80–88%; the `random` workload's prompt of random tokens is a worst case for expert residency. `angry-birds` reached the 8,192-token output limit.*

---

## Requirements

- **GPU:** AMD Radeon RX 9070 XT (16 GB, RDNA 4 `gfx1201`)
- **OS:** Windows 11 (64-bit)
- **Driver:** **AMD Software: Adrenalin Edition 26.10.44 or newer** (ships `amdhip64_7.dll` and fixes the 26.5.1–26.9.1 WDDM residency cap bug — no ROCm SDK needed)
- **System RAM:** 32 GB recommended (16 GB allocated for the warm RAM expert tier so misses rescue over PCIe instead of NVMe)
- **Disk:** ~36 GB free on NVMe (~21 GB Hugging Face checkpoint + ~14.7 GB packed `experts.stide`)

---

## Quick Start (Pre-built Release — No SDKs Needed)

1. **Download and extract** `smalltide-v0.1.0-windows-x64-gfx1201.zip` from [GitHub Releases](../../releases).
2. **Run one-time setup** in PowerShell or Command Prompt inside the extracted folder:

```powershell
.\smalltide.cmd setup
```

`setup` copies `.env.example` → `.env`, verifies your RX 9070 XT & HIP driver (`probe`), downloads [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) via built-in `curl.exe` (resumable, ~21 GB; set `$env:HF_TOKEN` only if you want higher Hugging Face rate limits), and packs all 3,072 routed experts into 4 KB-aligned `experts.stide` (~15 s).

3. **Start the server:**

```powershell
.\smalltide.cmd serve
```

4. **Use it:**
   - **OpenAI API:** `http://127.0.0.1:8000/v1` (model name: `inkling-2-mini`) — works with Open WebUI, Continue, Cline, OpenCode, and any OpenAI-compatible client.
   - **Live Expert Map:** open **`http://127.0.0.1:8000/expert-map`** in your browser to watch routed experts, VRAM residency, inline rescues, and async promotions live.
   - **One-shot CLI test:**

```powershell
.\smalltide.cmd run -p "Write a concise explanation of speculative decoding." -n 256
```

*(`smalltide.cmd` invokes `smalltide.ps1` with `-ExecutionPolicy Bypass`, so it works out of the box on stock Windows 11 without changing PowerShell execution policies.)*

---

## API Endpoints

| Method & Path | Description |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions (streaming SSE & non-streaming, `reasoning_content`, `reasoning_effort`, tool calls, `cache_prompt`) |
| `POST /v1/completions` | OpenAI Legacy Completions (streaming & non-streaming) |
| `GET  /v1/models` | Lists `inkling-2-mini` |
| `GET  /expert-map` | Live WebGL/Grid expert residency & routing visualizer |
| `GET  /v1/tide/stats` | Live TideLM telemetry (tok/s, hit rates by zone, rescues, skipped weight, VRAM) |
| `POST /v1/tide/config` | Live runtime tuning of `rescue_mode` and `rescue_threshold` |
| `GET  /health` | Health check |

Quick `curl.exe` smoke test:

```powershell
curl.exe -s http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Say hello in 10 words."}],"max_tokens":64}'
```

---

## How SmallTide Works

1. **Loop-aware adaptive residency (TideLM).** Inkling-2-Mini runs 24 physical MoE layers as 48 logical layers: 4 Prelude layers, 12 Loop Core layers executed **4 times per token**, and 8 Coda layers. SmallTide tracks every router pick into persistent routing priors (`<MODEL_DIR>/tide_priors.json`), automatically seeded on first boot from `profiles/inkling2-mini-priors.json` (aggregate routing shares only, zero user text). Value-per-byte placement prioritizes the 4× Loop Core first (~98%+ hit rate in VRAM), filling the rest of dedicated VRAM with the hottest Prelude NVFP4 and Coda MXFP8 experts.
2. **Windows DXGI residency-aware sizing.** On Windows WDDM, `hipMemGetInfo` reports shared system RAM as free VRAM, causing oversized allocations to silently drop ~10× in bandwidth over PCIe. With `SMALLTIDE_POOL_BUDGET=resident` (default), SmallTide queries DXGI's local residency budget and probes copy bandwidth across every 256 MiB window at startup (`fit_pools_to_resident`) so 100% of active expert slots stay on dedicated GDDR6.
3. **2-Stage Miss Handling + Fused GPU Spin-Wait Rescue.** Non-VRAM experts live warm in pinned/host RAM (`SMALLTIDE_RAM_GB=16`) backed by 4 KB-aligned Win32 `FILE_FLAG_NO_BUFFERING` NVMe reads (`experts.stide`). High-weight misses (`>= 0.15` in Prelude/Coda, `>= 0.10` in Loop Core, per-zone budgets `8/6/10`) are copied over PCIe on a dedicated DMA stream and synchronized on-GPU by a fused HIP spin-wait + route-patch kernel (`SMALLTIDE_RESCUE_WAIT=spin`), while lower-weight misses are skipped and renormalized and hot misses promote asynchronously in the background.
4. **Lossless DSpark Speculative Decoding.** Inkling-2-Mini's 3-layer DSpark drafter (`mtp.safetensors`, ~418 MiB VRAM at 49K ctx) drafts up to 7 tokens per step with confidence-gated adaptive depth. Verification uses GPU-side top-k/top-p filtering (`smalltide_logits_filter_rows`) and exact stochastic rejection sampling, accelerating both `temperature = 0` and `temperature > 0` without altering the target distribution.
5. **Paged K8/V4 Global KV & Prefix Caching.** Global-attention KV (8 target layers + 1 drafter layer) is paged from a shared `SMALLTIDE_KV_TOKENS` pool across lanes with asymmetric **`k8v4`** quantization (`SMALLTIDE_KV_FMT=k8v4`: FP8 E4M3 keys for exact attention scores + per-32-group asymmetric INT4 values, saving ~472 MiB at 128K for +115 more resident experts). Multi-turn conversations restore prefix snapshots (including drafter KV state) from pinned host RAM without re-prefilling.

---

## Tuning `.env`

Edit `.env` and restart `.\smalltide.cmd serve`. Key knobs:

| Setting | Default | What it controls |
| --- | --- | --- |
| `SMALLTIDE_MAX_CTX` / `SMALLTIDE_KV_TOKENS` | `131072` | Context length & shared paged global-KV pool (128K default with `k8v4` across 2 lanes; raise to `196608` or lower to `65536` to trade context for experts). |
| `SMALLTIDE_KV_FMT` | `k8v4` | Global KV format: `k8v4` (FP8 keys + Q4 values, saves ~472 MiB @ 128K) or `fp8` (full FP8 K/V). |
| `SMALLTIDE_LANES` | `2` | Concurrent serving lanes (each extra lane reserves ~124 MiB of local KV rings + slack page). |
| `SMALLTIDE_RESCUE_MODE` | `threshold` | `threshold` (balanced), `exact` (copy every miss, bit-exact MoE), or `disabled` (fastest, skip inline misses). |
| `SMALLTIDE_RESCUE_THRESHOLD` / `..._CORE` | `0.15` / `0.1` | Routing-weight gate for inline rescues (Prelude & Coda / Loop Core). |
| `SMALLTIDE_RESCUE_BUDGET_PRELUDE` / `_CORE` / `_CODA` | `8` / `6` / `10` | Per-token inline rescue caps per architectural zone. |
| `SMALLTIDE_SPEC` | `1` | Enable (`1`) or disable (`0`) DSpark speculative decoding. |
| `SMALLTIDE_RAM_GB` | `16` | Host RAM tier for non-VRAM experts (lower to `10` on 16 GB RAM PCs). |
| `SMALLTIDE_HOST` / `SMALLTIDE_PORT` | `0.0.0.0` / `8000` | Server bind address (`127.0.0.1` for local-only) and port. |

> **Getting maximum resident experts on Windows:** Closing VRAM-heavy desktop apps (browsers with many tabs, Adobe Creative Cloud, games, hardware-accelerated Electron apps) before launching `.\smalltide.cmd serve` directly increases the number of experts SmallTide can keep resident on the card. Run `.\smalltide.cmd vram-probe` anytime to inspect your card's current resident headroom.

To reset learned routing priors back to the bundled seed, delete `models\Inkling-2-Mini\tide_priors.json` and restart the server.

---

## Benchmarking

With `.\smalltide.cmd serve` running in another terminal (requires Python 3, standard library only):

```powershell
.\smalltide.cmd bench --workload prose --runs 3 --out results\prose.json
# every workload: prose, code, edit, random (--isl 8192), angry-birds, mario
foreach ($w in "prose","code","edit","random","angry-birds","mario") { .\smalltide.cmd bench --workload $w --out "results\$w.json" }
```

Each run prints decode speed, TTFT, skipped routing weight, inline rescues/token, VRAM hit rates by zone and disk reads; `--out` writes the summary JSON, and the one-shot games are saved to `results/oneshots/`.

---

## Building from Source

Only needed if you want to modify the Rust engine/HIP kernels or compile for another RDNA 4 architecture (`SMALLTIDE_HIP_ARCH`):

- **Prerequisites:** [Rust](https://rustup.rs/) (MSVC toolchain) + **AMD ROCm HIP SDK for Windows** (`hipcc.exe`, auto-detected under `C:\Program Files\AMD\ROCm\*\bin\` or `ROCM_PATH` / `SMALLTIDE_HIPCC`).

```powershell
git clone https://github.com/raspy135/SmallTide.git
cd SmallTide
.\smalltide.cmd build
.\smalltide.cmd setup
.\smalltide.cmd serve
```

To build a standalone pre-built Windows release zip (for GitHub Releases):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package-release.ps1
```

See [LIMITATIONS.md](LIMITATIONS.md) for known constraints, [CREDITS.md](CREDITS.md) for attributions, and [docs/PROGRESS.md](docs/PROGRESS.md) for engineering notes.

## License

Repository code is [MIT-licensed](LICENSE). Inkling-2-Mini model weights are governed by [Thinking Machines' license](https://huggingface.co/thinkingmachines/Inkling-2-Mini).
