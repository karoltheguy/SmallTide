# Limitations

Honest list of what SmallTide does and does not do today.

## Hardware & OS

- **Pre-built kernels target AMD Radeon RX 9070 XT / RDNA 4 (`gfx1201`) only.** The pre-built `smalltide-cli.exe` embeds compiled `gfx1201` `.hsaco` GPU kernels. Running on another AMD GPU architecture (e.g., `gfx1200` RX 9070, or RDNA 3 `gfx1100`) requires building from source with `SMALLTIDE_HIP_ARCH=<arch>` and the ROCm HIP SDK; kernels are tuned for RDNA 4 Wave32 + WMMA and are untested on older generations.
- **Windows-first.** The direct NVMe expert store uses Win32 unbuffered I/O (`FILE_FLAG_NO_BUFFERING`), and resident VRAM budgeting queries Windows DXGI (`IDXGIAdapter3::QueryVideoMemoryInfo`). Linux is not yet supported.
- **AMD driver residency cap on 26.5.1–26.9.1.** Adrenalin drivers 26.5.1 through 26.9.1 had a KMD bug ([ROCm/TheRock#7221](https://github.com/ROCm/TheRock/issues/7221)) that capped per-process resident VRAM near half of the card (~8 GiB). **Adrenalin 26.10.44 or newer** is strongly recommended (~14.5–15.0 GiB resident budget on a 16 GB RX 9070 XT). If you must stay on an older affected driver, set `SMALLTIDE_RESIDENT_CAP=reservation` in `.env`.
- **Desktop VRAM contention (WDDM).** Because Windows shares your RX 9070 XT with the desktop compositor, browsers, and background apps (e.g. Creative Cloud, Discord, Electron apps), the actual resident expert budget depends on how much VRAM other apps hold at server startup. SmallTide measures residency at startup (`fit_pools_to_resident`) and trims pools so weights never silently spill into ~10× slower PCIe system memory — freeing desktop VRAM before starting `serve` directly increases the resident expert count (e.g. from ~1,850 to ~2,016+ experts out of 3,072).

## Model & Feature Scope

- **Inkling-2-Mini only.** SmallTide's dense loader, looped-transformer forward pass, `.stide` expert pack format, and DSpark drafter are built specifically for [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini).
- **Text, thinking, and tool calling only (no vision input yet).** Multimodal image parts are not yet wired into the HTTP server.
- **No built-in API authentication.** The HTTP server binds to `0.0.0.0:8000` by default (configurable via `SMALLTIDE_HOST=127.0.0.1` in `.env`). Bind to `127.0.0.1` or place behind a reverse proxy if exposing beyond a trusted local network.
- **Threshold rescue is a speed/exactness trade-off.** Under the default `SMALLTIDE_RESCUE_MODE=threshold` (`0.15` Prelude/Coda, `0.10` Loop Core, per-zone budgets `8/6/10`), rare tail experts below the weight threshold are skipped and the remaining routed experts are renormalized (~2.5–4.2% skipped routing weight on warm traffic). For bit-exact expert execution at lower speed, set `SMALLTIDE_RESCUE_MODE=exact`.
- **Speculative decoding is single-lane.** DSpark speculative decoding (`SMALLTIDE_SPEC=1`) accelerates single active requests (both greedy `temperature=0` and stochastic `temperature>0` via exact rejection sampling). When multiple requests decode concurrently on multiple lanes, SmallTide automatically falls back to batched non-speculative decode.
- **Disk footprint.** You need ~36 GB of free NVMe space during setup: ~21 GB for the Hugging Face checkpoint (`model-00001-of-00001.safetensors` + `mtp.safetensors`) plus ~14.7 GB for the 4 KB-aligned `experts.stide` pack. Both files are kept on disk (dense weights are loaded from the safetensors file; routed experts stream from `experts.stide`).
