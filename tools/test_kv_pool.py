"""Paged KV pool check against a running `smalltide-cli serve` (SMALLTIDE_RESCUE_MODE=exact) whose
pool is too small for all conversations at once (e.g. SMALLTIDE_KV_TOKENS=1: lanes + 1 pages).

Three long conversations take turns, so the third must reclaim an idle lane's pages; then the
first conversation continues. Its warm answer (restored from its prefix-cache snapshot, the global
KV copied back into fresh pages) must equal a cold answer of the same request.

    python -I tools/test_kv_pool.py [--port 8012]
"""
import argparse, json, urllib.request

def chat(port, msgs, cache=True, n=48):
    body = json.dumps({"messages": msgs, "max_tokens": n, "temperature": 0, "cache": cache}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                                               {"Content-Type": "application/json"}), timeout=900))
    m = r["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "|" + (m.get("content") or ""), r["usage"]

def stats(port):
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/tide/stats", timeout=30))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8012); a = ap.parse_args()
    convs = {}
    for name, topic in [("A", "bridges"), ("B", "volcanoes"), ("C", "orchids")]:
        sysmsg = f"You are an expert on {topic}. " + " ".join(f"Fact {i} about {topic}: entry {i * 7} in the {topic} handbook." for i in range(260))
        msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": f"Name one thing about {topic}."}]
        txt, u = chat(a.port, msgs)
        convs[name] = msgs + [{"role": "assistant", "content": txt.split("|", 1)[1]}]
        print(f"conv {name}: prompt {u['prompt_tokens']} tokens, kv pool {stats(a.port).get('kv_pool')}")
    follow = convs["A"] + [{"role": "user", "content": "And another one?"}]
    warm, u = chat(a.port, follow)
    print(f"A follow-up warm: prompt {u['prompt_tokens']}, cached {u.get('prompt_tokens_details', {}).get('cached_tokens')}")
    cold, _ = chat(a.port, follow, cache=False)
    print("PASS: warm == cold" if warm == cold else f"FAIL\n warm: {warm[:300]}\n cold: {cold[:300]}")

if __name__ == "__main__":
    main()
