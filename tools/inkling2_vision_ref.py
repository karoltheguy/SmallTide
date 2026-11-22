"""inkling2_vision_ref.py: Inkling 2's vision tower in numpy against SmallTide's rows.

`smalltide-cli vision-check` writes the patches it encoded and the feature rows it got; this
reads the same patches, runs them through a transcription of transformers' Inkling2VisionModel
(Inkling2VisionEmbeddings, Inkling2VisionRotaryEmbedding, Inkling2VisionEncoderLayer,
Inkling2VisionMerger and vision_utils' position ids and interpolation taps; from TideLM's
tools/inkling_vision_ref.py) with the checkpoint's model.visual.* weights, in fp32, and prints the
relative error and the rows' cosine. With --dump it compares the intermediates too (the engine's
SMALLTIDE_VISION_DUMP=PREFIX files).

    smalltide-cli vision-check --rows 3 --cols 5 --patches-out P.bin --out O.bin
    python -I tools/inkling2_vision_ref.py --model models/Inkling-2-Mini --patches P.bin --rows 3 --cols 5 --engine O.bin

The reference rounds every op to bf16 and the engine keeps f32 between ops, so the rows agree to
rounding. Needs numpy only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct

import numpy as np

P = "model.visual."


class Ckpt:
    def __init__(self, path: str):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.h = json.loads(self.f.read(n))
        self.base = 8 + n

    def __call__(self, name: str) -> np.ndarray:
        e = self.h[P + name]
        assert e["dtype"] == "BF16", (name, e["dtype"])
        a, b = e["data_offsets"]
        self.f.seek(self.base + a)
        bits = np.frombuffer(self.f.read(b - a), dtype=np.uint16)
        return (bits.astype(np.uint32) << 16).view(np.float32).reshape(e["shape"])


def read_bf16(path: str, width: int) -> np.ndarray:
    bits = np.fromfile(path, dtype=np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32).reshape(-1, width)


def axis_taps_weights(index, size, side):
    src = (index.astype(np.float32) + 0.5) * side / size - 0.5
    floor = np.floor(src)
    offsets = np.arange(2)
    taps = np.clip(floor.astype(np.int64)[:, None] + offsets, 0, side - 1)
    distance = np.abs(src[:, None] - floor[:, None] - offsets)
    return taps, np.clip(1 - distance, 0, None).astype(np.float32)


def block_major(h, w, merge):
    within = np.arange(h * w)
    blocks_w = w // merge
    in_col = within % merge
    in_row = (within // merge) % merge
    block_col = (within // (merge * merge)) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    return block_row * merge + in_row, block_col * merge + in_col


def layer_norm(x, w, b, eps):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


_erf = np.vectorize(math.erf, otypes=[np.float32])


def gelu(x):
    return (0.5 * x * (1.0 + _erf(x * np.float32(0.7071067811865476)))).astype(np.float32)


def rotate_pairwise(x):
    y = x.reshape(*x.shape[:-1], -1, 2)
    return np.stack((-y[..., 1], y[..., 0]), axis=-1).reshape(x.shape)


def tower(W, patches, rows, cols, cfg, theta=10000.0):
    acts = {}
    merge = cfg.get("spatial_merge_size", cfg.get("adapter_merge_size", 2))
    heads = cfg.get("num_attention_heads", 16)
    layers = cfg.get("num_hidden_layers", 23)
    eps = cfg.get("layer_norm_eps", 1e-5)
    e = "backbone.embeddings."
    table = W(e + "position_embedding")
    side = int(round(table.shape[0] ** 0.5))
    d = table.shape[1]
    hd = d // heads
    h, w = rows * merge, cols * merge

    x = patches @ W(e + "patch_embedding.weight").T
    row, col = block_major(h, w, merge)
    h_taps, h_w = axis_taps_weights(row, np.full_like(row, h), side)
    w_taps, w_w = axis_taps_weights(col, np.full_like(col, w), side)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_w[:, :, None] * w_w[:, None, :]).reshape(-1, 4)
    pos = (table[indices] * weights[:, :, None]).sum(1)
    x = x + pos
    cls = W(e + "class_embedding") + W(e + "cls_position_embedding")
    x = np.concatenate([cls[None], x]).astype(np.float32)
    acts["embed"] = x
    x = layer_norm(x, W("backbone.ln_pre.weight"), W("backbone.ln_pre.bias"), eps)
    acts["pre"] = x
    s = x.shape[0]

    # patches are 1-indexed (row, col), the class token at (0, 0); position ids are block-major
    pid = np.concatenate([np.zeros((1, 2)), np.stack([row, col], -1) + 1]).astype(np.float32)
    spatial = hd // 2
    inv_freq = (1.0 / (theta ** (np.arange(0, spatial, 2, dtype=np.float32) / spatial))).astype(np.float32)
    freqs = pid[..., None] * inv_freq                                   # [s][2][hd / 4]
    rec = lambda f: np.repeat(np.concatenate([f[:, 1], f[:, 0]], -1), 2, -1)   # width, then height
    cos, sin = rec(np.cos(freqs))[:, None, :], rec(np.sin(freqs))[:, None, :]

    for i in range(layers):
        l = f"backbone.layers.{i}."
        a = l + "self_attn."
        y = layer_norm(x, W(l + "ln_1.weight"), W(l + "ln_1.bias"), eps)
        q = (y @ W(a + "q_proj.weight").T + W(a + "q_proj.bias")).reshape(s, heads, hd)
        k = (y @ W(a + "k_proj.weight").T + W(a + "k_proj.bias")).reshape(s, heads, hd)
        v = (y @ W(a + "v_proj.weight").T + W(a + "v_proj.bias")).reshape(s, heads, hd)
        q = q * cos + rotate_pairwise(q) * sin
        k = k * cos + rotate_pairwise(k) * sin
        q, k, v = (t.transpose(1, 0, 2) for t in (q, k, v))             # [heads][s][hd]
        p = q @ k.transpose(0, 2, 1) * hd ** -0.5
        p = np.exp(p - p.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        o = (p @ v).transpose(1, 0, 2).reshape(s, -1)
        o = o @ W(a + "out_proj.weight").T + W(a + "out_proj.bias")
        x = x + W(l + "ls_1.gamma") * o
        y = layer_norm(x, W(l + "ln_2.weight"), W(l + "ln_2.bias"), eps)
        y = y @ W(l + "mlp.c_fc.weight").T + W(l + "mlp.c_fc.bias")
        y = gelu(y) @ W(l + "mlp.c_proj.weight").T + W(l + "mlp.c_proj.bias")
        x = (x + W(l + "ls_2.gamma") * y).astype(np.float32)
        acts[f"b{i}"] = x

    y = gelu(x[1:].reshape(-1, merge ** 2 * d) @ W("proj_in.weight").T)
    acts["merge"] = y
    return y @ W("proj_out.weight").T, acts


def compare(name, eng, ref):
    ref = ref.astype(np.float32).reshape(-1, ref.shape[-1])
    if eng.shape != ref.shape:
        print(f"  {name}: shapes differ, engine {eng.shape} reference {ref.shape}")
        return None
    rel = float(np.linalg.norm(eng - ref) / np.linalg.norm(ref))
    cos = (eng * ref).sum(-1) / (np.linalg.norm(eng, axis=-1) * np.linalg.norm(ref, axis=-1) + 1e-30)
    print(f"  {name}: {ref.shape[0]} rows x {ref.shape[1]}, rel err {rel:.3e}, row cosine min {cos.min():.5f} "
          f"mean {cos.mean():.5f}, max |diff| {np.abs(eng - ref).max():.3e} (max |ref| {np.abs(ref).max():.3e})")
    return float(cos.min())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="the checkpoint directory")
    ap.add_argument("--patches", required=True, help="bf16 [rows * cols * 4][1200]: vision-check's --patches-out")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--engine", required=True, help="f32 [rows * cols][hidden]: vision-check's --out")
    ap.add_argument("--dump", help="the engine's SMALLTIDE_VISION_DUMP prefix: compare the intermediates too")
    ap.add_argument("--floor", type=float, default=0.99, help="fail under this row cosine")
    a = ap.parse_args(argv)

    cfg = json.load(open(os.path.join(a.model, "config.json")))["vision_config"]
    W = Ckpt(os.path.join(a.model, "model-00001-of-00001.safetensors"))
    patches = read_bf16(a.patches, 1200)
    ref, acts = tower(W, patches, a.rows, a.cols, cfg)
    print(f"{a.rows} x {a.cols} tokens, {patches.shape[0]} patches")
    if a.dump:
        for name, t in acts.items():
            f = f"{a.dump}.{name}.bin"
            if os.path.exists(f):
                compare(name, np.fromfile(f, dtype=np.float32).reshape(-1, t.shape[-1]), t)
    low = compare("tower", np.fromfile(a.engine, dtype=np.float32).reshape(-1, ref.shape[-1]), ref)
    if low is None or not low >= a.floor:
        print(f"FAIL: row cosine under {a.floor}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
