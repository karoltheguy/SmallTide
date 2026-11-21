"""inkling2_loop_ref.py: check SmallTide's Inkling-2-Mini engine against the transformers reference.

Runs the prompt tokens the engine dumped (SMALLTIDE_DUMP_DIR) through the *actual* modeling code of
transformers PR #51050 (`Inkling2LoopTextModel`, eager attention, fp32) with the real checkpoint, and
compares every logical layer's output (48 = 6 prelude + 3x12 loop + 6 coda), the routed expert sets
and the final logits, position by position.

Routed experts are dequantized lazily with the PR's own `Inkling2LoopDequantizeExperts` op (only the
experts that are actually hit), so the whole model never has to fit in RAM.

    $env:SMALLTIDE_DUMP_DIR="dump"; smalltide-cli run --rescue-mode exact --prompt "..." -n 4
    python -I tools/inkling2_loop_ref.py --model models/Inkling-2-Mini --dump dump [--emulate-fp8-dense]

--emulate-fp8-dense quantizes the dense projections the engine stores as FP8 E4M3 (per-row absmax
scale) the same way, isolating the engine's arithmetic from its weight quantization.

Needs: torch, safetensors, transformers at PR #51050 head (430050f6f71195b76004b5d61bbb446ec0508fb3).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import struct
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig
from transformers.core_model_loading import Inkling2LoopDequantizeExperts
from transformers.models.inkling2_loop import modeling_inkling2_loop as M

torch.set_grad_enabled(False)
P = "model.llm."

# Checkpoint -> Inkling2LoopTextModel renames, transcribed in order from the PR's
# conversion_mapping.py ("inkling2_loop_text" + the non-expert part of "inkling2_text").
RENAMES = [
    (r"controllers\.attn\.", "q_message_gate."),
    (r"controllers\.k\.", "k_message_gate."),
    (r"controllers\.v\.", "v_message_gate."),
    (r"controllers\.mlp\.", "mlp_message_gate."),
    (r"_message_gate\.down_weight", "_message_gate.down_proj.weight"),
    (r"_message_gate\.gain_weight", "_message_gate.message_gate_proj.weight"),
    (r"_message_gate\.forget_weight", "_message_gate.forget_gate_proj.weight"),
    (r"^embed\.weight", "embed_tokens.weight"),
    (r"^embed_norm\.", "embed_tokens.embed_norm."),
    (r"mlp\.gate\.bias", "mlp.gate.e_score_correction_bias"),
    (r"\.attn\.", ".self_attn."),
    (r"self_attn\.wq_du", "self_attn.q_proj.linear"),
    (r"self_attn\.wk_dv", "self_attn.k_proj.linear"),
    (r"self_attn\.wv_dv", "self_attn.v_proj.linear"),
    (r"self_attn\.k_sconv\.", "self_attn.k_proj.sconv."),
    (r"self_attn\.v_sconv\.", "self_attn.v_proj.sconv."),
    (r"self_attn\.rel_logits_proj\.", "self_attn.relative_position_bias."),
    (r"self_attn\.wr_du", "self_attn.relative_position_bias.linear"),
    (r"self_attn\.q_l2_clip\.", "self_attn.q_proj.norm."),
    (r"self_attn\.rel_l2_clip\.", "self_attn.relative_position_bias.norm."),
    (r"self_attn\.wo_ud", "self_attn.o_proj.linear"),
    (r"self_attn\.gate_attn", "self_attn.o_proj.gate_proj"),
    (r"self_attn\.attn_sink_proj", "self_attn.sink_proj"),
    (r"attn_sconv\.", "post_attention_convolution."),
    (r"mlp_sconv\.", "post_mlp_convolution."),
    (r"(convolution|sconv)\.weight(?!_)", r"\1.conv1d.weight"),
    (r"(convolution|sconv)\.static_w", r"\1.static_weight"),
    (r"(convolution|sconv)\.weight_proj", r"\1.weight_proj.weight"),
    (r"mlp_norm", "post_attention_layernorm"),
    (r"attn_norm", "input_layernorm"),
    (r"(q|k|v)_input_norm", r"\1_input_layernorm"),
    (r"attn_res\.mlp_queries\.(\d+)", r"layers.\1.mlp_input.query"),
    (r"attn_res\.output_query", "final_input.query"),
]

# Dense matrices the engine stores as FP8 E4M3 with per-row scales (dense_loader.rs).
FP8_DENSE_PATTERNS = [
    r"q_proj\.linear\.weight$", r"k_proj\.linear\.weight$", r"v_proj\.linear\.weight$",
    r"o_proj\.linear\.weight$", r"o_proj\.gate_proj\.weight$", r"relative_position_bias\.linear\.weight$",
    r"_message_gate\.(down_proj|message_gate_proj|forget_gate_proj)\.weight$",
]


def hf_name(ck: str) -> str | None:
    if not ck.startswith(P):
        return None
    n = ck[len(P):]
    m = re.match(r"attn_res\.attn_queries\.(\d+)$", n)
    if m:  # Inkling2IndexOffsetWeightRename(offset=1)
        return f"layers.{int(m.group(1)) + 1}.attention_input.query"
    for a, b in RENAMES:
        n = re.sub(a, b, n)
    return n


def fp8_lastdim(x: torch.Tensor) -> torch.Tensor:
    """E4M3 quant-dequant with one absmax/448 scale per vector along the last dim (FP8 KV cache)."""
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    return (x / scale).to(torch.float8_e4m3fn).float() * scale


def install_fp8_kv():
    orig = M.eager_attention_forward

    def wrapped(module, query, key, value, *a, **kw):
        if isinstance(module, (M.Inkling2LoopSlidingAttention, M.Inkling2LoopFullAttention)):
            key, value = fp8_lastdim(key), fp8_lastdim(value)
        return orig(module, query, key, value, *a, **kw)

    M.eager_attention_forward = wrapped


def install_forced_routes(engine_routes: torch.Tensor):
    """Router selects the engine's experts ([48, T, 8], logical-layer call order); weights are the
    reference's own normalized sigmoid scores at those indices."""
    state = {"call": 0}

    def forward(self, hidden_states):
        hidden_states = hidden_states.float()
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        raw = engine_routes[state["call"]]
        state["call"] += 1
        skipped = (raw & 0x80000000) != 0
        indices = (raw & 0x7FFFFFFF).to(torch.long)
        weights = scores.gather(1, indices)
        # The engine's MoE kernel skips unpublished experts and renormalizes over the rest.
        weights = torch.where(skipped, torch.zeros_like(weights), weights)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices

    M.Inkling2LoopTopKRouter.forward = forward


def fp8_rows(w: torch.Tensor) -> torch.Tensor:
    amax = w.abs().amax(dim=1, keepdim=True)
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    return (w / scale).to(torch.float8_e4m3fn).float() * scale


class _F32Param:
    """Makes Inkling2LoopDequantizeExperts emit fp32 (it otherwise falls back to bf16)."""

    def get_parameter(self, _name):
        return torch.empty(0, dtype=torch.float32)


class LazyExperts(nn.Module):
    """Inkling2LoopExperts.forward with experts dequantized on demand by the PR's own op."""

    def __init__(self, sf, layer: int, cfg, log: list):
        super().__init__()
        self.sf, self.layer, self.cfg, self.log = sf, layer, cfg, log
        self.act_fn = M.Inkling2LoopSoftcappedSwiGLU(cfg)
        self.num_experts = cfg.n_routed_experts
        self.cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.op = Inkling2LoopDequantizeExperts()
        base = f"{P}layers.{layer}.mlp.experts."
        self.base = base
        keys = set(sf.keys())
        self.nvfp4 = (base + "w13_weight_amax") in keys

    def _deq(self, name: str, e: int) -> torch.Tensor:
        b = self.base + name
        w = self.sf.get_slice(b)[e:e + 1]
        sc = self.sf.get_slice(b + "_scale")[e:e + 1]
        if self.nvfp4:
            amax = self.sf.get_slice(b + "_amax")[e:e + 1]
            den = self.sf.get_tensor(b + "_outer_scale_denominator")
            src = {"w": w, "s": sc, "a": amax, "d": den}
            pats = ["w", "s", "a", "d"]
        else:
            src = {"w": w, "s": sc}
            pats = ["w", "s", "a", "d"]
        out = self.op.convert(src, pats, ["out"], model=_F32Param(), full_layer_name="x")["out"]
        return out[0].float()

    def get(self, e: int):
        # No cache: one dequantized expert is 32 MB in fp32 and up to 3072 are hit.
        w13 = self._deq("w13_weight", e)  # rows interleaved: 2r gate, 2r+1 up
        gate_up = torch.cat([w13[0::2], w13[1::2]], dim=0)  # == Interleave(dim=1)
        down = self._deq("w2_weight", e)
        return gate_up, down

    def forward(self, hidden_states, top_k_index, top_k_weights):
        self.log.append((self.layer, top_k_index.clone()))
        final = torch.zeros_like(hidden_states, dtype=torch.float32)
        for e in torch.unique(top_k_index).tolist():
            top_k_pos, token_idx = torch.where(top_k_index.T == e)
            gate_up, down = self.get(e)
            cur = hidden_states[token_idx]
            g, u = F.linear(cur, gate_up).chunk(2, dim=-1)
            h = F.linear(self.act_fn(g, u), down) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, h.float())
        return final.to(hidden_states.dtype)


def load_dumps(d: str):
    rows = []
    for f in glob.glob(os.path.join(d, "tok*.bin")):
        b = open(f, "rb").read()
        magic, pos, tok, nl, h, has_logits = struct.unpack_from("<6I", b, 0)
        assert magic == 0x53544450, f
        off = 24
        hid = np.frombuffer(b, np.float32, nl * h, off).reshape(nl, h)
        off += nl * h * 4
        routes = np.frombuffer(b, np.uint32, nl * 8, off).reshape(nl, 8)
        off += nl * 8 * 4
        logits = np.frombuffer(b, np.float32, offset=off) if has_logits else None
        rows.append((pos, tok, hid, routes, logits))
    rows.sort(key=lambda r: r[0])
    assert [r[0] for r in rows] == list(range(len(rows))), "dump positions must be 0..T-1"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--emulate-fp8-dense", action="store_true")
    ap.add_argument("--emulate-fp8-kv", action="store_true")
    ap.add_argument("--force-engine-routes", action="store_true")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    if args.emulate_fp8_kv:
        install_fp8_kv()
    rows = load_dumps(args.dump)
    tokens = [r[1] for r in rows]
    if args.force_engine_routes:
        er = torch.from_numpy(np.stack([r[3] for r in rows]).astype(np.int64))  # [T, 48, 8]
        print(f"[routes] engine skipped {(er & 0x80000000).ne(0).sum().item()} of {er.numel()} routed experts")
        install_forced_routes(er.permute(1, 0, 2).contiguous())
    T = len(tokens)
    print(f"[dump] {T} positions, tokens={tokens}")

    cfg = AutoConfig.from_pretrained(args.model).text_config
    cfg._attn_implementation = "eager"
    nl = cfg.total_num_hidden_layers if hasattr(cfg, "total_num_hidden_layers") else 48

    from safetensors import safe_open
    sf_path = os.path.join(args.model, "model-00001-of-00001.safetensors")
    sf = safe_open(sf_path, framework="pt", device="cpu")

    with torch.device("meta"):
        model = M.Inkling2LoopTextModel(cfg)
    route_log: list = []
    for i, layer in enumerate(model.layers):
        layer.mlp.experts = LazyExperts(sf, i, cfg, route_log)

    targets = dict(model.named_parameters())
    targets.update(dict(model.named_buffers()))
    state: dict[str, torch.Tensor] = {}
    for ck in sf.keys():
        if not ck.startswith(P) or ".mlp.experts." in ck:
            continue
        if ck in (P + "unembed.weight",):
            continue
        if ck.endswith("middle_cycle_cross_pass_queries") or ck.endswith("middle_cycle_cross_pass_kv_queries"):
            continue
        n = hf_name(ck)
        if n not in targets:
            print(f"  [skip] {ck} -> {n} (no such param)")
            continue
        t = sf.get_tensor(ck).float()
        if tuple(t.shape) != tuple(targets[n].shape):
            raise SystemExit(f"shape mismatch {ck} -> {n}: {tuple(t.shape)} vs {tuple(targets[n].shape)}")
        if args.emulate_fp8_dense and any(re.search(p, n) for p in FP8_DENSE_PATTERNS):
            t = fp8_rows(t)
        state[n] = t
    # Inkling2LoopMergeMessageQueries: [q[2i], kv[2i], kv[2i+1], q[2i+1]]
    q = sf.get_tensor(P + "middle_cycle_cross_pass_queries").float().unflatten(0, (-1, 2))
    kv = sf.get_tensor(P + "middle_cycle_cross_pass_kv_queries").float().unflatten(0, (-1, 2))
    state["message_input.query"] = torch.cat([q[:, :1], kv, q[:, 1:]], dim=1)

    missing = [n for n in targets if n not in state and ".mlp.experts." not in n]
    if missing:
        raise SystemExit(f"unloaded params: {missing[:20]} ... ({len(missing)})")
    model.load_state_dict(state, strict=False, assign=True)
    model.eval()

    unembed = sf.get_tensor(P + "unembed.weight")[: cfg.unpadded_vocab_size].float()
    if args.emulate_fp8_dense:
        unembed = fp8_rows(unembed)

    outs: list[torch.Tensor] = []
    hooks = []
    for layer in model.layers:
        hooks.append(layer.register_forward_hook(lambda m, i, o: outs.append(o.detach()[0])))
    ids = torch.tensor([tokens])
    res = model(input_ids=ids, use_cache=False)
    final = res.last_hidden_state[0]
    for h in hooks:
        h.remove()
    assert len(outs) == nl, f"captured {len(outs)} layer outputs, expected {nl}"
    ref = torch.stack(outs, dim=1)  # [T, 48, H]
    logits = torch.tanh(F.linear(final, unembed) / cfg.final_logit_softcapping) * cfg.final_logit_softcapping

    eng = torch.from_numpy(np.stack([r[2] for r in rows]))  # [T, 48, H]
    print("\nlayer  rel_err(mean)  rel_err(max)  cos(min)   route_overlap")
    ref_routes = [rl[1] for rl in route_log]  # in call order = logical order, each [T, 8]
    worst = 0.0
    for il in range(nl):
        d = (eng[:, il] - ref[:, il]).norm(dim=-1) / ref[:, il].norm(dim=-1).clamp_min(1e-12)
        cos = F.cosine_similarity(eng[:, il], ref[:, il], dim=-1)
        er = np.stack([r[3][il] for r in rows]) & 0x7FFFFFFF
        rr = ref_routes[il].numpy()
        ov = np.mean([len(set(er[t]) & set(rr[t])) / 8.0 for t in range(T)])
        worst = max(worst, d.max().item())
        print(f"{il:5d}  {d.mean().item():12.3e}  {d.max().item():12.3e}  {cos.min().item():8.6f}   {ov:6.3f}")

    print("\nlogits (positions the engine computed):")
    agree = 0
    n = 0
    for t, r in enumerate(rows):
        if r[4] is None:
            continue
        el = torch.from_numpy(r[4].copy())
        rl = logits[t]
        n += 1
        a, b = int(el.argmax()), int(rl.argmax())
        agree += a == b
        lpe, lpr = F.log_softmax(el, -1), F.log_softmax(rl, -1)
        kl = torch.sum(lpr.exp() * (lpr - lpe)).item()
        top5r = rl.topk(5).indices.tolist()
        print(f"  pos {t:3d}: engine argmax={a} ref argmax={b} {'OK ' if a == b else 'MISMATCH'} "
              f"max|dlogit|={(el - rl).abs().max().item():.4f} KL(ref||eng)={kl:.3e} ref top5={top5r}")
    print(f"\nsummary: worst layer rel_err={worst:.3e}; argmax agreement {agree}/{n}")
    json.dump({"worst_rel_err": worst, "argmax_agree": agree, "n_logits": n},
              open(os.path.join(args.dump, "ref_summary.json"), "w"))


if __name__ == "__main__":
    main()
