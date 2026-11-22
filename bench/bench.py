#!/usr/bin/env python3
"""
bench.py -- measure the SmallTide server (`smalltide-cli serve`, Inkling-2-Mini).

Ported from TideLM's `bench/bench.py` so workloads and metrics line up directly:
  * Captures per-request deltas from SmallTide's `/v1/tide/stats` (skipped routing
    weight %, sync rescues, PCIe rescue GB, NVMe disk reads, zone VRAM hit rates)
    alongside TTFT, TPOT, and decode tok/s.
  * Keeps TideLM's exact workloads (`prose`, `code`, `edit`, `angry-birds`, `mario`,
    `random`) and writes one-shot HTML games into `results/oneshots/`.
  * Uses `usage.completion_tokens` from the final SSE chunk (since speculative
    decoding can deliver multiple tokens per chunk).

Usage:
    python bench/bench.py --workload prose --runs 3 --out results/prose.json
    python bench/bench.py --workload code --runs 2
    python bench/bench.py --workload edit --runs 2
    python bench/bench.py --workload random --isl 8192 --osl 512 --runs 2
    python bench/bench.py --workload angry-birds --label spec --thinking --effort low
"""
import argparse
import hashlib
import json
import os
import random
import re
import statistics
import string
import sys
import time
from urllib import request as urlrequest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
ONESHOT_DIR = os.path.join(REPO_ROOT, "results", "oneshots")


def post(base, path, payload, api_key=None, timeout=3600):
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urlrequest.Request(base.rstrip("/") + path, data=data, headers=headers)
    return urlrequest.urlopen(req, timeout=timeout)


def get_json(base, path, api_key=None, timeout=30):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    req = urlrequest.Request(base.rstrip("/") + path, headers=headers)
    with urlrequest.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def parse_effort(effort_str):
    """Map --thinking / --effort into SmallTide's `reasoning_effort` value."""
    if effort_str is None:
        return None
    low = str(effort_str).strip().lower()
    if low in ("none", "minimal", "low", "medium", "high"):
        return low
    try:
        v = float(low)
        # Accept either TideLM's 0..100 integer scale or Inkling-2's 0.0..0.99 scale.
        if v > 1.0:
            v = min(0.99, v / 100.0)
        return round(max(0.0, min(0.99, v)), 3)
    except ValueError:
        return low


def build_reasoning_effort(thinking, effort):
    if not thinking:
        return "none"
    val = parse_effort(effort)
    return val if val is not None else "low"


def token_len(base, prompt, model, api_key, thinking, effort):
    """Exact rendered-prompt length. Tries /v1/debug/prompt, falls back to a 1-token dry call."""
    re_val = build_reasoning_effort(thinking, effort)
    try:
        out = post(
            base,
            "/v1/debug/prompt",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "reasoning_effort": re_val,
            },
            api_key,
            timeout=60,
        )
        return json.loads(out.read())["prompt_tokens"]
    except Exception:
        pass
    try:
        out = post(
            base,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "temperature": 0.0,
                "reasoning_effort": re_val,
                "cache_prompt": False,
                "stream": False,
            },
            api_key,
            timeout=300,
        )
        return json.loads(out.read()).get("usage", {}).get("prompt_tokens")
    except Exception:
        return None


def make_prompt(target_tokens, seed):
    """Unique, non-repetitive text so nothing is prefix-cached or run-merged."""
    rnd = random.Random(seed)
    words = []
    for i in range(target_tokens * 3 + 64):
        w = "".join(rnd.choice(string.ascii_lowercase) for _ in range(rnd.randint(3, 9)))
        words.append(f"{i}:{w}")
    return " ".join(words)


def fit_prompt(base, model, isl, seed, api_key, thinking, effort):
    """Scale a unique word list until the server's tokenizer reports ~isl."""
    pool = make_prompt(isl, seed).split()
    words = min(len(pool), max(16, isl // 3))
    best_txt, best_n = None, None
    for _ in range(6):
        txt = " ".join(pool[:words])
        n = token_len(base, txt, model, api_key, thinking, effort)
        if n is None:
            return txt, None
        if best_n is None or abs(n - isl) < abs(best_n - isl):
            best_txt, best_n = txt, n
        if abs(n - isl) <= max(16, isl // 100):  # within 1%
            return txt, n
        scaled = int(words * (isl / max(n, 1)))
        words = max(16, min(len(pool), scaled))
    return best_txt, best_n


ONE_SHOTS = {
    "angry-birds": (
        "Write a complete Angry Birds style game as a single self-contained HTML file: canvas rendering, "
        "a slingshot with drag-to-aim, projectile physics with gravity, destructible block structures, "
        "pigs as targets, a score counter, and a restart button. No external libraries or assets."
    ),
    "mario": (
        "Write a complete side-scrolling Mario style platformer as a single HTML file with inline JavaScript: "
        "keyboard controls, jumping with gravity, moving enemies you can stomp, coins, a scrolling level, "
        "and a win condition. No external libraries."
    ),
}

WORKLOADS = {
    "prose": (
        "Write a detailed, flowing essay of about 900 words on how tidal "
        "forces shaped the evolution of coastal ecosystems. Use full "
        "paragraphs and continuous prose, no bullet points or headings."
    ),
    "code": (
        "Write a complete, production-quality Python module implementing an "
        "LRU cache with a TTL per entry, thread safety, and an eviction "
        "callback. Include full docstrings, type hints, and a pytest suite "
        "covering expiry, eviction order, and concurrent access."
    ),
}

with open(os.path.join(HERE, "edit_source.py.txt"), encoding="utf-8") as _f:
    WORKLOADS["edit"] = (
        "Make these changes to the Python module below and reply with the complete updated module "
        "in one code block, with no commentary before or after it: rename the class JobQueue to "
        "RetryQueue everywhere; add a method `cancel(self, name: str) -> bool` that removes a "
        "pending job with that name and returns whether one was removed; and make `pending()` "
        "return the names of the pending jobs as a sorted list instead of a count. Keep everything "
        "else exactly as it is.\n\n```python\n" + _f.read() + "```\n"
    )
WORKLOADS.update(ONE_SHOTS)
ALL_WORKLOADS = ["random"] + sorted(WORKLOADS)

HEADLINE_STATS = [
    "expert_hit_rate",
    "skipped_weight_pct",
    "rescues_per_tok",
    "rescue_gb",
    "disk_reads",
]


def tide_stats(base, api_key):
    try:
        return get_json(base, "/v1/tide/stats", api_key, timeout=15)
    except Exception:
        return None


def compute_tide_delta(before, after, completion_tokens):
    if not before or not after:
        return {}
    d_w_tot = max(0.0, after.get("routed_weight_total", 0.0) - before.get("routed_weight_total", 0.0))
    d_w_skip = max(0.0, after.get("routed_weight_skipped", 0.0) - before.get("routed_weight_skipped", 0.0))
    d_rescues = max(0, after.get("stage1_sync_rescues", 0) - before.get("stage1_sync_rescues", 0))
    d_bytes = max(0, after.get("stage1_bytes_transferred", 0) - before.get("stage1_bytes_transferred", 0))
    d_disk = max(0, after.get("stage2_disk_reads", 0) - before.get("stage2_disk_reads", 0))
    skipped_pct = (100.0 * d_w_skip / d_w_tot) if d_w_tot > 0 else None
    return {
        "expert_hit_rate": round(after.get("vram_hit_rate_pct", 0.0) / 100.0, 4),
        "prelude_hit_pct": round(after.get("prelude_hit_rate_pct", 0.0), 2),
        "core_hit_pct": round(after.get("loop_core_hit_rate_pct", 0.0), 2),
        "coda_hit_pct": round(after.get("coda_hit_rate_pct", 0.0), 2),
        "skipped_weight_pct": round(skipped_pct, 2) if skipped_pct is not None else None,
        "sync_rescues": d_rescues,
        "rescues_per_tok": round(d_rescues / max(completion_tokens, 1), 2),
        "rescue_gb": round(d_bytes / 1e9, 3),
        "disk_reads": d_disk,
    }


def run_once(base, model, prompt, osl, api_key, thinking, effort, temperature, seed=None, cache_prompt=False):
    """One streamed chat completion. Returns timings + per-request TideLM stats."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": osl,
        "temperature": temperature,
        "top_p": 0.95,
        "stream": True,
        "cache_prompt": cache_prompt,
        "reasoning_effort": build_reasoning_effort(thinking, effort),
    }
    if seed is not None:
        body["seed"] = seed

    st_before = tide_stats(base, api_key)
    t0 = time.perf_counter()
    ttft = None
    prompt_tokens = None
    cached_tokens = None
    completion_tokens = None
    finish_reason = None
    engine_stats = {}
    content, reasoning = [], []

    resp = post(base, "/v1/chat/completions", body, api_key)
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if chunk == "[DONE]":
            break
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if obj.get("error"):
            raise RuntimeError(f"server error mid-stream: {obj['error']}")
        if obj.get("x_engine_stats"):
            engine_stats = obj["x_engine_stats"]
        if obj.get("usage"):
            prompt_tokens = obj["usage"].get("prompt_tokens")
            completion_tokens = obj["usage"].get("completion_tokens")
            ptd = obj["usage"].get("prompt_tokens_details") or {}
            cached_tokens = ptd.get("cached_tokens")
        ch = obj.get("choices") or []
        if ch:
            if ch[0].get("finish_reason"):
                finish_reason = ch[0]["finish_reason"]
            d = ch[0].get("delta") or {}
            if d.get("content"):
                content.append(d["content"])
            if d.get("reasoning_content"):
                reasoning.append(d["reasoning_content"])
            if ttft is None and (d.get("content") or d.get("reasoning_content")):
                ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    st_after = tide_stats(base, api_key)

    if not completion_tokens:
        raise RuntimeError("server returned no usage.completion_tokens -- cannot measure honestly")
    if ttft is None:
        ttft = total
    decode_s = max(total - ttft, 1e-9)
    tpot_ms = (decode_s / max(completion_tokens - 1, 1)) * 1000.0
    delta_stats = compute_tide_delta(st_before, st_after, completion_tokens)
    delta_stats.update(engine_stats)

    return {
        "prompt_tokens_actual": prompt_tokens,
        "cached_tokens": cached_tokens,
        "ttft_ms": ttft * 1000.0,
        "tpot_ms": tpot_ms,
        "decode_tok_s": (completion_tokens - 1) / decode_s,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "total_s": total,
        "x_engine_stats": delta_stats,
        "_content": "".join(content),
        "_reasoning": "".join(reasoning),
    }


_FENCE = re.compile(r"```(?:html|HTML)?\s*\n(.*?)```", re.S)


def extract_artifact(text):
    """Return (body, extension) -- a playable HTML file when one is in there."""
    for block in _FENCE.findall(text):
        if re.search(r"<!doctype html|<html|<canvas", block, re.I):
            return block.strip() + "\n", "html"
    m = re.search(r"(<!doctype html.*|<html.*)", text, re.I | re.S)
    if m:
        return m.group(1).strip() + "\n", "html"
    return text, "txt"


def save_oneshot(label, workload, run_tag, text):
    os.makedirs(ONESHOT_DIR, exist_ok=True)
    body, ext = extract_artifact(text)
    stem = f"{label}-{workload}" if run_tag in ("", "run1") else f"{label}-{workload}-{run_tag}"
    path = os.path.join(ONESHOT_DIR, f"{stem}.{ext}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return path


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark the SmallTide server (ported from TideLM bench/bench.py).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--base",
        default=os.environ.get("BENCH_BASE") or f"http://127.0.0.1:{os.environ.get('PORT', '8000')}",
    )
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--workload", default="prose", choices=ALL_WORKLOADS)
    ap.add_argument("--isl", type=int, default=8192, help="target prompt tokens (random workload only)")
    ap.add_argument("--osl", type=int, default=None, help="max output tokens (default 1024, or 8192 for one-shots)")
    ap.add_argument("--runs", type=int, default=None, help="measured runs (default 3, or 1 for a one-shot)")
    ap.add_argument("--warmup", type=int, default=None, help="unmeasured runs first (default 1, or 0 for a one-shot)")
    ap.add_argument("--thinking", action="store_true", help="enable thinking mode (default: reasoning_effort=none)")
    ap.add_argument("--effort", default=None, help="reasoning effort: low|medium|high or 0.0..0.99 / 1..100")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=None, help="sampling seed passed to the server")
    ap.add_argument("--seed-salt", default=None, help="fixes prompt seeds for A/B comparisons")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None, help="write the summary JSON here")
    ap.add_argument("--no-save-oneshot", action="store_true", help="do not write results/oneshots/<label>-<workload>.html")
    a = ap.parse_args()

    oneshot = a.workload in ONE_SHOTS
    if a.osl is None:
        a.osl = 8192 if oneshot else 1024
    if a.runs is None:
        a.runs = 1 if oneshot else 3
    if a.warmup is None:
        a.warmup = 0 if oneshot else 1

    served = get_json(a.base, "/v1/models", a.api_key)["data"][0]["id"]
    if served != a.model:
        print(f"note: server serves '{served}', using that")
        a.model = served
    try:
        health = get_json(a.base, "/health", a.api_key)
    except Exception:
        health = {}

    eff = build_reasoning_effort(a.thinking, a.effort)
    print(
        f"== {a.label}: workload={a.workload} isl={a.isl if a.workload == 'random' else 'n/a'} "
        f"osl={a.osl} runs={a.runs} warmup={a.warmup} thinking={a.thinking} effort={eff} temp={a.temperature}"
    )

    results = []
    salt_src = a.seed_salt if a.seed_salt is not None else a.label
    label_salt = int(hashlib.sha1(salt_src.encode()).hexdigest()[:6], 16) % 100000

    for i in range(a.warmup + a.runs):
        seed = 1000 + i + label_salt
        if a.workload == "random":
            prompt, _ = fit_prompt(a.base, a.model, a.isl, seed, a.api_key, a.thinking, a.effort)
        elif oneshot:
            prompt = WORKLOADS[a.workload]
        else:
            prompt = f"[req {seed}] " + WORKLOADS[a.workload]

        r = run_once(
            a.base, a.model, prompt, a.osl, a.api_key, a.thinking, a.effort, a.temperature, a.seed,
            cache_prompt=False,
        )
        measured = i >= a.warmup
        tag = "warmup" if not measured else f"run{i - a.warmup + 1}"
        st = r["x_engine_stats"] or {}
        if oneshot and not a.no_save_oneshot:
            r["artifact"] = save_oneshot(a.label, a.workload, tag, r["_content"] or r["_reasoning"])
        text = r.pop("_content")
        reasoning_text = r.pop("_reasoning")
        r["completion_chars"] = len(text)
        r["reasoning_chars"] = len(reasoning_text)
        print(
            f"  {tag:7s} isl={r['prompt_tokens_actual']} ttft={r['ttft_ms']:7.1f} ms  "
            f"tpot={r['tpot_ms']:6.2f} ms  decode={r['decode_tok_s']:6.2f} tok/s  "
            f"out={r['completion_tokens']} ({r['finish_reason']})"
        )
        print(
            f"          hit={st.get('expert_hit_rate')} (pre={st.get('prelude_hit_pct')}% "
            f"core={st.get('core_hit_pct')}% coda={st.get('coda_hit_pct')}%)  "
            f"skip_w={st.get('skipped_weight_pct')}%  rescues/tok={st.get('rescues_per_tok')}  "
            f"dma={st.get('rescue_gb')} GB  disk={st.get('disk_reads')}"
        )
        if "artifact" in r:
            print(f"          wrote {os.path.relpath(r['artifact'], REPO_ROOT)}")
        if measured:
            results.append(r)

    def med(key):
        vals = [x[key] for x in results if x.get(key) is not None]
        return round(statistics.median(vals), 4) if vals else None

    engine_meds = {}
    for k in HEADLINE_STATS:
        vals = [
            x["x_engine_stats"].get(k)
            for x in results
            if isinstance(x.get("x_engine_stats"), dict) and x["x_engine_stats"].get(k) is not None
        ]
        engine_meds[k] = round(statistics.median(vals), 4) if vals else None

    summary = {
        "label": a.label,
        "workload": a.workload,
        "seed_salt": a.seed_salt,
        "isl": a.isl if a.workload == "random" else None,
        "osl": a.osl,
        "runs": a.runs,
        "warmup": a.warmup,
        "thinking": a.thinking,
        "effort": eff,
        "temperature": a.temperature,
        "base": a.base,
        "model": a.model,
        "health": health,
        "ttft_ms_median": med("ttft_ms"),
        "tpot_ms_median": med("tpot_ms"),
        "decode_tok_s_median": med("decode_tok_s"),
        "engine_median": engine_meds,
        "raw": results,
    }
    print(
        f"  -> MEDIAN  ttft={summary['ttft_ms_median']} ms  tpot={summary['tpot_ms_median']} ms  "
        f"decode={summary['decode_tok_s_median']} tok/s"
    )
    print(
        f"  -> ENGINE  hit={engine_meds['expert_hit_rate']}  skip_w={engine_meds['skipped_weight_pct']}%  "
        f"rescues/tok={engine_meds['rescues_per_tok']}  dma={engine_meds['rescue_gb']} GB  "
        f"disk={engine_meds['disk_reads']}"
    )
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
