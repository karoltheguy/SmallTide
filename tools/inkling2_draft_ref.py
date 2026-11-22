"""Offline acceptance of Inkling-2-Mini's drafter (mtp.safetensors, "DSpark") on our engine's taps.

Run the engine with SMALLTIDE_DUMP_DIR on a prompt (greedy), then:

    python -I tools/inkling2_draft_ref.py --model models/Inkling-2-Mini --dump DIR [--prompt-len N]

For every anchor position P after the prompt, the drafter (plain PyTorch transcription of the
DSpark spec, tideLM docs/inkling-dspark.md section 8, with Mini's config: 3 layers, local layers
{0, 2}, 24 / 8 heads, taps = logical layers [5, 23, 41] + the final normed hidden) drafts 7 tokens
from the context built from the engine's taps of positions < P; they are compared with the
engine's own greedy continuation. Prints the acceptance length distribution and the implied
tokens per verify step.
"""
import argparse
import glob
import json
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F

BF = torch.bfloat16
EPS = 1e-6
P_ = "model.llm."


def bf16(x):
    return x.to(BF).float()


def rms(x, w):
    xf = x.float()
    return bf16(xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float())


def rope(x, pos, theta):
    # x [rows, heads, 128], neox halves
    hd = x.shape[-1]
    inv = theta ** (-torch.arange(0, hd, 2, dtype=torch.float32) / hd)
    ang = pos.float()[:, None] * inv[None]
    c, s = torch.cos(ang)[:, None], torch.sin(ang)[:, None]
    x1, x2 = x[..., : hd // 2].float(), x[..., hd // 2:].float()
    return bf16(torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1))


def mix_rms(feats, a, w):
    # feats [rows, T, D] (bf16 values in f32), a [T]
    m = torch.zeros(feats.shape[0], feats.shape[2])
    for t in range(feats.shape[1]):
        m += feats[:, t] * a[t]
    return bf16(m * torch.rsqrt(m.pow(2).sum(-1, keepdim=True) / feats.shape[2] + EPS) * w.float())


def load_dump(d):
    rows = []
    for f in glob.glob(os.path.join(d, "tok*.bin")):
        if f.endswith(".final.bin"):
            continue
        b = open(f, "rb").read()
        magic, pos, tok, nl, h, has_logits = struct.unpack_from("<6I", b, 0)
        hid = np.frombuffer(b, dtype=np.float32, count=nl * h, offset=24).reshape(nl, h)
        fin_path = os.path.join(d, f"tok{pos}.final.bin")
        fin = np.fromfile(fin_path, dtype=np.float32) if os.path.exists(fin_path) else None
        rows.append((pos, tok, hid, fin))
    rows.sort(key=lambda r: r[0])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--prompt-len", type=int, default=0, help="first anchor position (default: first decode position)")
    ap.add_argument("--max-anchors", type=int, default=400)
    a = ap.parse_args()
    from safetensors import safe_open

    cfg = json.load(open(os.path.join(a.model, "config.json")))
    mc = cfg["mtp_config"]
    D = mc["hidden_size"]; NL = mc["num_hidden_layers"]; HQ = mc["num_attention_heads"]; HK = mc["num_key_value_heads"]
    HD = mc["head_dim"]; LOCAL = set(mc["local_layer_ids"]); W = mc["sliding_window_size"]; BLOCK = mc["block_size"]
    R = mc["markov_rank"]; THETA = float(mc["rope_theta"]); TAPS = mc["target_layer_ids"]
    VU = cfg["text_config"]["unpadded_vocab_size"]
    G = HQ // HK

    mtp = safe_open(os.path.join(a.model, "mtp.safetensors"), framework="pt")
    t = lambda n: mtp.get_tensor(n).float()
    idx = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
    tgt = safe_open(os.path.join(a.model, idx[P_ + "embed.weight"]), framework="pt")
    embed = tgt.get_tensor(P_ + "embed.weight")            # bf16 [V, D]
    embed_norm = tgt.get_tensor(P_ + "embed_norm.weight").float()
    U = tgt.get_tensor(P_ + "unembed.weight").float()        # [V, D]
    L = [{k: t(f"layers.{l}.{k}") for k in [
        "input_layernorm.weight", "post_attention_layernorm.weight", "dflare_norm.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
        "self_attn.k_proj_ctx.weight", "self_attn.v_proj_ctx.weight", "self_attn.o_proj.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.k_norm_ctx.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]} for l in range(NL)]
    norm_w = t("norm.weight"); mask_embed = t("mask_embed")
    alpha = bf16(torch.softmax(t("fuse_weight"), -1))       # [NL + 1, taps]
    mk_norm = t("markov_head.dflare_norm.weight"); w1 = t("markov_head.markov_w1.weight")
    rnn_w = t("markov_head.rnn_proj.weight"); rnn_b = t("markov_head.rnn_proj.bias")
    ho_w = t("markov_head.hidden_out.weight"); ho_b = t("markov_head.hidden_out.bias")
    fold_w = bf16(U @ ho_w)                                  # [V, R]
    fold_b = bf16(U @ ho_b)                                  # [V]

    rows = load_dump(a.dump)
    T = len(rows)
    toks = [r[1] for r in rows]
    first_decode = next((i for i, r in enumerate(rows) if i > 0 and r[3] is not None and rows[i - 1][3] is not None and False), None)
    # taps per position: logical layer k entry == output of layer k-1; 48 == final normed hidden
    feats = []
    for (pos, tok, hid, fin) in rows:
        if fin is None:
            raise SystemExit(f"position {pos} has no final hidden (re-run the engine with this build)")
        f = [torch.from_numpy(hid[k - 1].copy()) if k < 48 else torch.from_numpy(fin.copy()) for k in TAPS]
        feats.append(bf16(torch.stack(f)))
    feats = torch.stack(feats)                               # [T, taps, D]

    # context K/V for every position, per drafter layer
    pos_all = torch.arange(T)
    ctxK, ctxV = [], []
    for l in range(NL):
        c = mix_rms(feats, alpha[l], L[l]["dflare_norm.weight"])
        k = bf16(c @ L[l]["self_attn.k_proj_ctx.weight"].T).reshape(T, HK, HD)
        v = bf16(c @ L[l]["self_attn.v_proj_ctx.weight"].T).reshape(T, HK, HD)
        k = rope(rms(k, L[l]["self_attn.k_norm_ctx.weight"]), pos_all, THETA)
        ctxK.append(k); ctxV.append(v)
    f_rnn_all = mix_rms(feats, alpha[NL], mk_norm)           # [T, D]

    start = a.prompt_len if a.prompt_len > 0 else max(1, T - a.max_anchors - BLOCK)
    accepts = []
    for P in range(start, T - BLOCK):
        anchor = toks[P]
        x = torch.cat([rms(embed[anchor:anchor + 1].float(), embed_norm), bf16(mask_embed)[None].expand(BLOCK - 1, D)], 0)
        pos = P + torch.arange(BLOCK)
        res = bf16(x); xn = rms(res, L[0]["input_layernorm.weight"])
        for l in range(NL):
            Ly = L[l]
            q = bf16(xn @ Ly["self_attn.q_proj.weight"].T).reshape(BLOCK, HQ, HD)
            k = bf16(xn @ Ly["self_attn.k_proj.weight"].T).reshape(BLOCK, HK, HD)
            v = bf16(xn @ Ly["self_attn.v_proj.weight"].T).reshape(BLOCK, HK, HD)
            q = rope(rms(q, Ly["self_attn.q_norm.weight"]), pos, THETA)
            k = rope(rms(k, Ly["self_attn.k_norm.weight"]), pos, THETA)
            lo = max(0, P - W) if l in LOCAL else 0
            K = torch.cat([ctxK[l][lo:P], k]); Vv = torch.cat([ctxV[l][lo:P], v])
            o = torch.zeros(BLOCK, HQ, HD)
            for h in range(HQ):
                g = h // G
                sc = (q[:, h] @ K[:, g].T) * HD ** -0.5
                o[:, h] = torch.softmax(sc, -1) @ Vv[:, g]
            att = bf16(bf16(o).reshape(BLOCK, HQ * HD) @ Ly["self_attn.o_proj.weight"].T)
            s1 = res + att; res = bf16(s1)
            xn = rms(s1, Ly["post_attention_layernorm.weight"])
            mlp = bf16(bf16(F.silu(xn @ Ly["mlp.gate_proj.weight"].T) * (xn @ Ly["mlp.up_proj.weight"].T)) @ Ly["mlp.down_proj.weight"].T)
            s2 = res + mlp; res = bf16(s2)
            xn = rms(s2, L[l + 1]["input_layernorm.weight"] if l < NL - 1 else norm_w)
        hb = xn[1:BLOCK]
        base = bf16(hb @ U.T)
        state = torch.zeros(R); prev = anchor; drafts = []
        f_rnn = f_rnn_all[P - 1]
        for i in range(BLOCK - 1):
            z = torch.cat([state, bf16(w1[prev]), hb[i], f_rnn])
            raw = bf16(z @ rnn_w.T + rnn_b)
            g_raw, c_raw, o_raw = raw[:R], raw[R:2 * R], raw[2 * R:]
            gate = bf16(1 / (1 + torch.exp(-g_raw)))
            new = bf16(bf16(gate * state) + bf16(bf16(1 - gate) * bf16(torch.tanh(c_raw))))
            latent = bf16(torch.tanh(o_raw))
            bias = bf16(fold_b + latent @ fold_w.T)
            y = bf16(base[i] + bias)
            y = bf16(30 * torch.tanh(y / 30))
            y[VU:] = -float("inf")
            prev = int(y.argmax())
            state = new
            drafts.append(prev)
        acc = 0
        while acc < BLOCK - 1 and P + 1 + acc < T and drafts[acc] == toks[P + 1 + acc]:
            acc += 1
        accepts.append(acc)
    acc = np.array(accepts)
    hist = np.bincount(acc, minlength=BLOCK)
    print(f"anchors {len(acc)} (positions {start}..{T - BLOCK - 1})")
    print("accepted drafts per round: " + " ".join(f"{i}:{hist[i]}" for i in range(BLOCK)))
    print(f"mean accepted {acc.mean():.2f} -> {1 + acc.mean():.2f} tokens per verify step")


if __name__ == "__main__":
    main()
