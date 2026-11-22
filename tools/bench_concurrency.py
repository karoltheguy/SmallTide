"""Aggregate decode throughput of a running `smalltide-cli serve` at 1..N concurrent requests.

    python -I tools/bench_concurrency.py [--port 8000] [--max-n 4] [--tokens 256]
"""
import argparse
import json
import threading
import time
import urllib.request

PROMPTS = [
    "Explain how a CPU cache works, including associativity and replacement policies.",
    "Write a short story about a robot learning to paint.",
    "Describe the water cycle in detail for a high-school student.",
    "Compare Python, Rust and Go for building web services.",
    "Explain how public-key cryptography works with an example.",
    "Write a detailed recipe for a vegetable lasagna.",
    "Summarize the causes and effects of the industrial revolution.",
    "Explain how transformers in machine learning use attention.",
]


def one(port, prompt, tokens, out):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": tokens, "temperature": 0,
            "reasoning_effort": 0, "cache_prompt": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read().decode("utf-8"))
    out.append((d["usage"]["completion_tokens"], d["timings"]["prompt_ms"] / 1000.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-n", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=256)
    a = ap.parse_args()
    one(a.port, "Say hi.", 8, [])  # warm-up
    print(f"{'n':>3} {'wall s':>8} {'tokens':>7} {'aggregate tok/s':>16} {'per-request tok/s':>18}")
    for n in range(1, a.max_n + 1):
        out = []
        ts = [threading.Thread(target=one, args=(a.port, PROMPTS[i % len(PROMPTS)], a.tokens, out)) for i in range(n)]
        t0 = time.time()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        wall = time.time() - t0
        toks = sum(o[0] for o in out)
        pf = sum(o[1] for o in out)
        print(f"{n:3d} {wall:8.2f} {toks:7d} {toks / wall:16.1f} {toks / wall / n:18.1f}   (prefill {pf:.2f}s total)")


if __name__ == "__main__":
    main()
