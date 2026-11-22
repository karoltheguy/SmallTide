import json, sys, urllib.request
URL = sys.argv[1]
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
def call(msgs):
    body = {"messages": msgs, "tools": tools, "max_tokens": 300, "temperature": 0}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))
for city in sys.argv[2].split(","):
    msgs = [{"role": "user", "content": f"What's the weather in {city} right now? Use the tool."}]
    r = call(msgs); m = r["choices"][0]["message"]
    if not m.get("tool_calls"):
        print(city, "no tool call:", json.dumps(m)[:200]); continue
    msgs.append({"role": "assistant", "content": m.get("content"), "tool_calls": m.get("tool_calls")})
    for tc in m["tool_calls"]:
        msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps({"city": city, "temp_c": 14, "sky": "light rain"})})
    r = call(msgs); m = r["choices"][0]["message"]
    txt = (m.get("reasoning_content") or "") + "|" + (m.get("content") or "")
    words = txt.split(); bad = len(words) > 20 and len(set(words[-20:])) <= 3
    print(f"{city:10s} 2nd: finish={r['choices'][0]['finish_reason']:7s} cached={r['usage']['prompt_tokens_details']['cached_tokens']}/{r['usage']['prompt_tokens']} {'DEGENERATE' if bad else 'ok'} content={json.dumps(m.get('content'))[:80]}")
