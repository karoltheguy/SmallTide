"""Multi-turn conversations against a running `smalltide-cli serve`: measures, per turn, the routing
weight skipped in prefill and decode (the quality cost of missing experts) and the speed, to
evaluate prompt-aware expert loading. Turns resend the history without reasoning, like chat
clients do, so later turns restore the conversation's prefix-cache snapshot.

    python -I tools/multiturn_workload.py [--port 8000] [--max-tokens 128]
"""
import argparse
import json
import sys
import urllib.request

CONVERSATIONS = [
    ["Explain how a hash map works.", "How does it handle collisions?", "What about resizing when it gets full?"],
    ["Write a Python function that reverses a linked list.", "Now make it recursive.", "Add type hints and a docstring."],
    ["What causes the seasons on Earth?", "Why is it hotter in summer if Earth is farther from the Sun in July?", "How do seasons differ near the equator?"],
    ["파이썬에서 리스트와 튜플의 차이는 무엇인가요?", "언제 튜플을 쓰는 것이 좋나요?", "예제 코드를 보여주세요."],
    ["Draft a short email asking my manager for a day off next Friday.", "Make it more formal.", "Now shorten it to two sentences."],
    ["Explain TCP slow start.", "How does congestion avoidance differ from it?", "What changed in TCP BBR?"],
]


def post(port, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=128)
    a = ap.parse_args()
    per_turn = {}
    for conv in CONVERSATIONS:
        msgs = []
        for ti, q in enumerate(conv):
            msgs.append({"role": "user", "content": q})
            s0 = post(a.port, "/v1/tide/stats")
            out = post(a.port, "/v1/chat/completions", {"messages": msgs, "max_tokens": a.max_tokens, "temperature": 0, "reasoning_effort": 0})
            s1 = post(a.port, "/v1/tide/stats")
            msgs.append({"role": "assistant", "content": out["choices"][0]["message"].get("content") or ""})
            d = lambda k: s1.get(k, 0) - s0.get(k, 0)
            u, t = out["usage"], out["timings"]
            r = per_turn.setdefault(ti, {"n": 0, "pf_wt": 0.0, "pf_ws": 0.0, "wt": 0.0, "ws": 0.0, "ct": 0, "dec_s": 0.0,
                                         "pt": 0, "pf_s": 0.0, "swaps": 0, "cached": 0})
            r["n"] += 1
            r["pf_wt"] += d("prefill_weight_total"); r["pf_ws"] += d("prefill_weight_skipped")
            r["wt"] += d("routed_weight_total"); r["ws"] += d("routed_weight_skipped")
            r["ct"] += u["completion_tokens"]; r["dec_s"] += u["completion_tokens"] / max(t["predicted_per_second"], 1e-9)
            new_pt = u["prompt_tokens"] - u["prompt_tokens_details"]["cached_tokens"]
            r["pt"] += new_pt; r["pf_s"] += t["prompt_ms"] / 1000.0
            r["swaps"] += d("prompt_aware_swaps"); r["cached"] += u["prompt_tokens_details"]["cached_tokens"]
    print(f"{'turn':>4} {'n':>3} {'prefill skip wt%':>16} {'decode skip wt%':>15} {'decode tok/s':>12} {'prefill tok/s':>13} {'swaps/req':>9} {'cached/req':>10}")
    for ti in sorted(per_turn):
        r = per_turn[ti]
        print(f"{ti + 1:4d} {r['n']:3d} {100 * r['pf_ws'] / max(r['pf_wt'], 1e-9):16.2f} {100 * r['ws'] / max(r['wt'], 1e-9):15.2f} "
              f"{r['ct'] / max(r['dec_s'], 1e-9):12.1f} {r['pt'] / max(r['pf_s'], 1e-9):13.1f} {r['swaps'] / r['n']:9.1f} {r['cached'] / r['n']:10.1f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
