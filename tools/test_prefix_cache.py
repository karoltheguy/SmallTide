"""End-to-end prefix-cache check against a running `smalltide-cli serve` with SMALLTIDE_RESCUE_MODE=exact.

Sends a sequence of greedy chat requests that exercise every prefix-cache path (pure extension,
restore at the shared system-prompt boundary, restore at the previous prompt end) and verifies
each warm answer is identical to a cold answer of the same request (forced by first sending an
unrelated prompt so nothing can be reused). Exact rescue makes results residency-independent.

    python -I tools/test_prefix_cache.py [--port 8000]
"""

import argparse
import json
import sys
import time
import urllib.request

SYSTEM = ("You are a concise assistant for a geography quiz show. Answer in one short sentence, "
          "never add trivia, never ask follow-up questions, and always name the country.")


def chat(port, messages, max_tokens=160, effort=0, cache=True):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "reasoning_effort": effort, "cache_prompt": cache}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    dt = time.time() - t0
    msg = out["choices"][0]["message"]
    u = out["usage"]
    # Compare everything generated: reasoning + visible content.
    full = (msg.get("reasoning_content") or "") + " ||| " + (msg.get("content") or "")
    return msg.get("content", ""), full, u, dt


def stats(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/tide/stats", timeout=30) as r:
        return json.loads(r.read())["prefix_cache"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    sys_msg = {"role": "system", "content": SYSTEM}
    q1 = [sys_msg, {"role": "user", "content": "What is the capital of France?"}]
    q2 = [sys_msg, {"role": "user", "content": "What is the capital of Japan?"}]

    # Warm sequence
    plan = []
    c1, f1, u1, t1 = chat(a.port, q1)
    plan.append(("q1 (cold start)", q1, f1, u1, t1))
    c2, f2, u2, t2 = chat(a.port, q2)
    plan.append(("q2 (shared system prefix -> boundary restore)", q2, f2, u2, t2))
    q3 = q2 + [{"role": "assistant", "content": c2},
               {"role": "user", "content": "And of South Korea?"}]
    c3, f3, u3, t3 = chat(a.port, q3)
    plan.append(("q3 (multi-turn, reasoning dropped -> prompt-end restore)", q3, f3, u3, t3))
    q4 = q3 + [{"role": "assistant", "content": c3},
               {"role": "user", "content": "And of Brazil?"}]
    # A chat UI's side request (title generation) between turns: another sequence runs in between.
    chat(a.port, [{"role": "user", "content": "Write a 3-word title for a chat about world capitals."}], max_tokens=40)
    c4, f4, u4, t4 = chat(a.port, q4)
    plan.append(("q4 (multi-turn after an unrelated side request -> cross-sequence restore)", q4, f4, u4, t4))
    c5, f5, u5, t5 = chat(a.port, q4)
    plan.append(("q5 (identical repeat -> rewind)", q4, f5, u5, t5))
    print("prefix cache after warm sequence:", json.dumps(stats(a.port)))

    ok = True
    # Every request after the cold start must have reused something.
    for name, _, _, usage, _ in plan[1:]:
        if usage["prompt_tokens_details"]["cached_tokens"] == 0:
            ok = False
            print(f"BAD {name}: no prefix reuse")
    for name, msgs, warm, usage, dt in plan:
        _, cold, cu, cdt = chat(a.port, msgs, cache=False)
        same = warm == cold
        ok &= same
        print(f"{'OK ' if same else 'BAD'} {name}: cached={usage['prompt_tokens_details']['cached_tokens']}"
              f"/{usage['prompt_tokens']} warm {dt:.2f}s vs cold {cdt:.2f}s (cold cached="
              f"{cu['prompt_tokens_details']['cached_tokens']})\n     warm={warm!r}\n     cold={cold!r}")
    print("ALL IDENTICAL" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
