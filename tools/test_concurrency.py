"""End-to-end check of concurrent serving against a running `smalltide-cli serve`
(start it with SMALLTIDE_RESCUE_MODE=exact so outputs do not depend on expert residency).

1. Sequential: N different prompts one after another (greedy, cache_prompt off).
2. Concurrent: the same N prompts at once, streaming.
Passes if every concurrent output is identical to its sequential one and the streams overlap
(each request's first token arrives before the other requests finish).

    python -I tools/test_concurrency.py [--port 8000] [--n 4] [--max-tokens 96]
"""
import argparse
import json
import sys
import threading
import time
import urllib.request

PROMPTS = [
    "Explain in three sentences how a hash map handles collisions.",
    "Write a haiku about a lighthouse in winter.",
    "What is 17 * 23? Answer with the number and one sentence of working.",
    "List three differences between TCP and UDP.",
    "Translate 'good morning, how are you?' into Korean and Japanese.",
    "Give a one-paragraph summary of what photosynthesis is.",
    "Write a Python one-liner that reverses a string.",
    "Name three planets and one fact about each.",
]


def body(prompt, max_tokens, stream):
    return {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
            "reasoning_effort": 0, "cache_prompt": False, "stream": stream}


def chat(port, prompt, max_tokens):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body(prompt, max_tokens, False)).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read().decode("utf-8"))
    m = out["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + " ||| " + (m.get("content") or "")


def chat_stream(port, prompt, max_tokens, rec):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body(prompt, max_tokens, True)).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    reasoning, content = [], []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            delta = d["choices"][0].get("delta", {}) if d.get("choices") else {}
            if delta.get("reasoning_content") or delta.get("content"):
                rec.setdefault("first", time.time() - t0)
            reasoning.append(delta.get("reasoning_content") or "")
            content.append(delta.get("content") or "")
    rec["end"] = time.time() - t0
    rec["text"] = "".join(reasoning) + " ||| " + "".join(content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=96)
    a = ap.parse_args()
    prompts = PROMPTS[:a.n]

    t0 = time.time()
    seq = [chat(a.port, p, a.max_tokens) for p in prompts]
    t_seq = time.time() - t0

    recs = [dict() for _ in prompts]
    threads = [threading.Thread(target=chat_stream, args=(a.port, p, a.max_tokens, r)) for p, r in zip(prompts, recs)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_con = time.time() - t0

    ok = True
    for i, (s, r) in enumerate(zip(seq, recs)):
        same = s == r.get("text")
        ok &= same
        print(f"{'OK ' if same else 'BAD'} request {i}: first token {r.get('first', -1):.2f}s, done {r.get('end', -1):.2f}s")
        if not same:
            print(f"     sequential={s!r}\n     concurrent={r.get('text')!r}")
    overlap = max(r.get("first", 1e9) for r in recs) < min(r.get("end", 0) for r in recs)
    print(f"streams overlap: {overlap} | sequential {t_seq:.2f}s vs concurrent {t_con:.2f}s for {len(prompts)} requests")
    ok &= overlap
    print("ALL IDENTICAL" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
