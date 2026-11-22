# Credits

SmallTide brings adaptive MoE expert residency and speculative decoding to consumer AMD RDNA 4 GPUs on Windows. It builds on the ideas, models, and tools of the following projects.

## Adaptive Expert Loading & TideLM

- **[TideLM](https://github.com/0xBakeer/tideLM)** (MIT License, Khaled Bakeer / 0xBakeer and Ryan Kojima) — The core insight behind SmallTide: keeping the experts your traffic actually uses resident in GPU memory, learning routing demand online into persistent priors, overlapping hot-miss promotions with decoding, and rescuing high-weight misses inline. SmallTide's live `/expert-map` visualizer and `bench/bench.py` benchmark harness are adapted from TideLM so results and diagnostics follow the same conventions.

## Model & Drafter

- **[Thinking Machines — Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** (Apache-2.0) — The 30B-total / 3.8B-active looped MoE architecture (`inkling2`: Prelude / 4× looped Core / Coda, 128 routed experts per physical layer in NVFP4 and MXFP8, sliding-window + global attention with short convolutions and attention-residual mixing, and the 3-layer DSpark speculative drafter in `mtp.safetensors`), weights, tokenizer, and chat template. Model weights are governed by Thinking Machines' license.

## Hardware & Runtime

- **AMD ROCm / HIP & Adrenalin** — RDNA 4 (`gfx1201`) ISA, `hipcc` ahead-of-time `.hsaco` compilation, and the Windows HIP runtime (`amdhip64_7.dll`) shipped with AMD Software: Adrenalin Edition.
- **Rust ecosystem** — `libloading`, `serde`, `serde_json`, `thiserror`, and zero-dependency HTTP/1.1 + SSE serving.
