import json, sys, time, urllib.request
URL = sys.argv[1]
sysmsg = "You are a meticulous assistant for a hardware company. " + " ".join(f"Rule {i}: always cite the part number, check the voltage rating, and mention the warranty terms for item {i}." for i in range(120))
def chat(msgs, n=24):
    body = json.dumps({"messages": msgs, "max_tokens": n, "temperature": 0}).encode()
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=600))
    m = r["choices"][0]["message"]
    return (m.get("content") or "") + (m.get("reasoning_content") or ""), r.get("timings", {}).get("prompt_ms", -1), time.time() - t
a = [{"role": "system", "content": sysmsg}]
for turn in range(6):
    a.append({"role": "user", "content": f"Question {turn}: what is item {turn * 7} about? " + "Explain in detail. " * 20})
    txt, pms, wall = chat(a)
    a.append({"role": "assistant", "content": txt})
    print(f"session A turn {turn}: prompt {pms:7.0f} ms")
b = [{"role": "system", "content": sysmsg}, {"role": "user", "content": "New session: summarize rule 3."}]
txt, pms, wall = chat(b)
print(f"session B first turn: prompt {pms:7.0f} ms")
