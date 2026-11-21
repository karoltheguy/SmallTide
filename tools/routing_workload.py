"""Mixed chat workload for expert-placement experiments against a running `smalltide-cli serve`.

Two disjoint prompt sets with the same mix of domains:
  bootstrap  - builds routing priors (the server folds each request into SMALLTIDE_PRIORS)
  eval       - held-out prompts for measuring a placement policy

Reports decode tok/s, prefill tok/s, VRAM hit rate and the skipped routing-weight share (the
quality cost of missing experts) per category and in total, from /v1/tide/stats deltas.

    python -I tools/routing_workload.py --set eval --port 8000 [--max-tokens 160] [--json out.json]
"""
import argparse
import json
import os
import sys
import time
import urllib.request

CODING = [
    "Write a Python function that parses a CSV file and returns the average of each numeric column.",
    "Implement a thread-safe LRU cache in Rust with get and put. Explain the locking choice briefly.",
    "Write a TypeScript React hook useDebounce(value, delay) and show a usage example.",
    "Explain this SQL and suggest an index: SELECT user_id, COUNT(*) FROM orders WHERE created_at > NOW() - INTERVAL '7 days' GROUP BY user_id HAVING COUNT(*) > 3;",
    "Write a C function that reverses a singly linked list in place, with a small test in main.",
    "Find the bug: def mean(xs): return sum(xs) / len(xs) - it crashes for some inputs. Fix it and add type hints.",
    "Write a Go HTTP handler that accepts JSON {\"a\":int,\"b\":int} and returns their sum, with input validation.",
    "Write a bash script that finds the 10 largest files under a directory and prints human-readable sizes.",
    "Implement binary search in Java over a sorted int array and return the insertion point when absent.",
    "Write a CUDA or HIP kernel that adds two float arrays, plus the host launch code.",
    "Refactor this into idiomatic Python: result = []\nfor i in range(len(a)):\n    if a[i] % 2 == 0:\n        result.append(a[i] * a[i])",
    "Write a regular expression that validates IPv4 addresses and explain each part.",
]
MATH = [
    "Solve for x: 3x^2 - 12x + 9 = 0. Show the steps.",
    "A train leaves at 9:00 at 80 km/h, another at 10:00 at 120 km/h on the same track. When does the second catch up?",
    "Prove that the sum of the first n odd numbers is n^2.",
    "What is the derivative of x^3 * ln(x)? Then integrate 2x * e^(x^2) dx.",
    "A bag has 5 red and 3 blue balls. Two are drawn without replacement. What is the probability both are red?",
    "Explain eigenvalues and eigenvectors with a 2x2 example and compute them.",
]
EXPLAIN = [
    "Explain how a CPU cache works, including cache lines and associativity.",
    "What causes the seasons on Earth? Correct the common misconception.",
    "Explain how TLS establishes a secure connection, step by step.",
    "How do vaccines train the immune system? Keep it accessible.",
    "Explain the difference between TCP and UDP and when to use each.",
    "What is inflation, and how do central banks try to control it?",
    "How does a transformer language model generate text? Explain attention simply.",
    "Explain how garbage collection works in Java versus reference counting in Swift.",
]
CREATIVE = [
    "Write a short poem about autumn rain in a city at night.",
    "Write the opening paragraph of a mystery novel set on a lighthouse.",
    "Write a friendly email to a colleague declining a meeting and proposing another time.",
    "Write a product description for a minimalist wooden desk lamp.",
    "Write a two-paragraph bedtime story about a fox who learns to share.",
    "Draft a LinkedIn post announcing a small open-source project release.",
]
MULTILINGUAL = [
    "캐시 메모리가 어떻게 동작하는지 쉽게 설명해 주세요.",
    "파이썬으로 리스트에서 중복을 제거하는 방법을 세 가지 알려주세요.",
    "日本の四季について短く説明してください。",
    "Explica en español qué es la fotosíntesis y por qué es importante.",
    "请用中文简单解释什么是区块链。",
    "Erkläre auf Deutsch den Unterschied zwischen Wetter und Klima.",
    "한국의 전통 음식 다섯 가지를 소개하고 각각 한 문장으로 설명해 주세요.",
    "Écris en français un court message pour remercier un ami de son aide.",
]
TOOLS_SPEC = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]},
    },
}, {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Search the internal documentation",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["query"]},
    },
}]
TOOLS = [
    "What's the weather like in Seoul right now? Use celsius.",
    "Find our documentation about configuring the rate limiter.",
    "Is it warmer in Paris or in Madrid today?",
    "Look up how to rotate the API keys in our docs, then summarize the steps.",
]


def corpus_docs(corpus_dir, name, n, chars=2500):
    path = os.path.join(corpus_dir, name)
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t = r.get("text", "")
            if len(t) > 800:
                out.append(("summarize-" + r.get("category", "doc"), "Summarize the key points of this document in five bullets:\n\n" + t[:chars]))
            if len(out) >= n:
                break
    return out


def build_sets(corpus_dir):
    cats = {"coding": CODING, "math": MATH, "explain": EXPLAIN, "creative": CREATIVE,
            "multilingual": MULTILINGUAL, "tools": TOOLS}
    boot, ev = [], []
    for c, xs in cats.items():
        for i, p in enumerate(xs):
            # Every third prompt is held out (eval); the rest bootstrap the priors.
            (ev if i % 3 == 2 else boot).append((c, p))
    boot += corpus_docs(corpus_dir, "trace_corpus_v3.jsonl", 6)
    ev += corpus_docs(corpus_dir, "heldout_quick.jsonl", 3)
    return boot, ev


def post(port, path, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--set", choices=["bootstrap", "eval", "both"], default="eval")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--corpus-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "tideLM", "corpus"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    boot, ev = build_sets(a.corpus_dir)
    work = {"bootstrap": boot, "eval": ev, "both": boot + ev}[a.set]

    per_cat = {}
    rows = []
    for i, (cat, prompt) in enumerate(work):
        s0 = post(a.port, "/v1/tide/stats")
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": a.max_tokens,
                "temperature": 0, "reasoning_effort": 0}
        if cat == "tools":
            body["tools"] = TOOLS_SPEC
        out = post(a.port, "/v1/chat/completions", body)
        s1 = post(a.port, "/v1/tide/stats")
        u, t = out["usage"], out["timings"]
        ct = u["completion_tokens"]
        dec_s = ct / max(t["predicted_per_second"], 1e-9)
        new_pt = u["prompt_tokens"] - u.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        wt = s1["routed_weight_total"] - s0["routed_weight_total"]
        ws = s1["routed_weight_skipped"] - s0["routed_weight_skipped"]
        hits_d = None
        row = {"cat": cat, "prompt_tokens": u["prompt_tokens"], "new_prompt_tokens": new_pt, "completion_tokens": ct,
               "prefill_s": t["prompt_ms"] / 1000.0, "decode_s": dec_s, "weight_total": wt, "weight_skipped": ws,
               "async": s1.get("decode_async_promotions", 0) - s0.get("decode_async_promotions", 0),
               "pf_wt": s1.get("prefill_weight_total", 0) - s0.get("prefill_weight_total", 0),
               "pf_ws": s1.get("prefill_weight_skipped", 0) - s0.get("prefill_weight_skipped", 0),
               "sync": s1["stage1_sync_rescues"] - s0["stage1_sync_rescues"]}
        rows.append(row)
        c = per_cat.setdefault(cat, {"n": 0, "ct": 0, "dec_s": 0.0, "pt": 0, "pf_s": 0.0, "wt": 0.0, "ws": 0.0,
                                     "async": 0, "sync": 0, "pf_wt": 0.0, "pf_ws": 0.0})
        c["n"] += 1; c["ct"] += ct; c["dec_s"] += dec_s; c["pt"] += new_pt; c["pf_s"] += row["prefill_s"]
        c["wt"] += wt; c["ws"] += ws; c["async"] += row["async"]; c["sync"] += row["sync"]
        c["pf_wt"] += row["pf_wt"]; c["pf_ws"] += row["pf_ws"]
        print(f"[{i + 1:2d}/{len(work)}] {cat:16s} pt={u['prompt_tokens']:5d} ct={ct:4d} decode {ct / max(dec_s, 1e-9):6.1f} tok/s "
              f"skipped weight {100.0 * ws / max(wt, 1e-9):5.2f}% async={row['async']} sync={row['sync']}", flush=True)
        _ = hits_d

    tot = {"n": 0, "ct": 0, "dec_s": 0.0, "pt": 0, "pf_s": 0.0, "wt": 0.0, "ws": 0.0, "async": 0, "sync": 0,
           "pf_wt": 0.0, "pf_ws": 0.0}

    def line(name, c):
        n = max(c["n"], 1)
        return (f"{name:18s} {c['n']:3d} {c['ct'] / max(c['dec_s'], 1e-9):12.1f} {c['pt'] / max(c['pf_s'], 1e-9):13.1f} "
                f"{100.0 * c['ws'] / max(c['wt'], 1e-9):12.2f} {100.0 * c['pf_ws'] / max(c['pf_wt'], 1e-9):11.2f} "
                f"{c['async'] / n:13.1f} {c['sync'] / n:8.1f}")

    print(f"\n{'category':18s} {'n':>3s} {'decode tok/s':>12s} {'prefill tok/s':>13s} {'dec skip wt%':>12s} "
          f"{'pf skip wt%':>11s} {'dec promo/req':>13s} {'sync/req':>8s}")
    for cat, c in per_cat.items():
        for k in tot:
            tot[k] += c[k]
        print(line(cat, c))
    print(line("TOTAL", tot))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "per_cat": per_cat, "total": tot}, f, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
