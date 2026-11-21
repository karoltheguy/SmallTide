"""End-to-end OpenAI tool-calling checks against a running `smalltide-cli serve`.

    python -I tools/test_tool_calls.py [--port 8000]
"""

import argparse
import json
import sys
import urllib.request

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the weather forecast for a city",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "days": {"type": "integer", "description": "Number of forecast days"}},
        "required": ["city"]}}}
CLOCK = {"type": "function", "function": {
    "name": "get_time", "description": "Get the current local time in a timezone",
    "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]}}}

FAILS = []


def check(cond, what):
    print(("PASS " if cond else "FAIL ") + what)
    if not cond:
        FAILS.append(what)


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def post_stream(port, body):
    body = dict(body, stream=True)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    chunks = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    port = ap.parse_args().port
    base = {"temperature": 0, "max_tokens": 400, "reasoning_effort": "low", "tools": [WEATHER, CLOCK]}

    # 1. auto tool call
    msgs = [{"role": "user", "content": "What's the weather going to be in Paris over the next 3 days?"}]
    r = post(port, dict(base, messages=msgs))
    ch = r["choices"][0]
    m = ch["message"]
    print(json.dumps(m, ensure_ascii=False)[:600])
    calls = m.get("tool_calls") or []
    check(ch["finish_reason"] == "tool_calls", f"auto: finish_reason=tool_calls (got {ch['finish_reason']})")
    check(len(calls) >= 1 and calls[0]["function"]["name"] == "get_weather", "auto: calls get_weather")
    args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
    check("paris" in str(args.get("city", "")).lower(), f"auto: city=Paris (args={args})")

    # 2. tool result round trip (client returns reasoning + the call); prefix cache must extend
    msgs2 = msgs + [{"role": "assistant", "content": m.get("content"), "reasoning_content": m.get("reasoning_content"),
                     "tool_calls": calls},
                    {"role": "tool", "tool_call_id": calls[0]["id"] if calls else "x",
                     "content": json.dumps({"forecast": ["sunny 24C", "cloudy 19C", "rain 15C"]})}]
    r2 = post(port, dict(base, messages=msgs2))
    m2 = r2["choices"][0]["message"]
    print(json.dumps(m2, ensure_ascii=False)[:600])
    u = r2["usage"]
    check(r2["choices"][0]["finish_reason"] == "stop" and not m2.get("tool_calls"), "result: answers in text")
    txt = (m2.get("content") or "").lower()
    check(any(w in txt for w in ("sunny", "24", "rain", "cloud")), "result: uses the tool output")
    check(u["prompt_tokens_details"]["cached_tokens"] >= r["usage"]["prompt_tokens"],
          f"result: prefix cache reused the first turn ({u['prompt_tokens_details']['cached_tokens']}/{u['prompt_tokens']} cached)")

    # 3. streaming tool call
    chunks = post_stream(port, dict(base, messages=[{"role": "user", "content": "What time is it in Tokyo right now?"}]))
    tc = [c for c in chunks if c["choices"] and c["choices"][0]["delta"].get("tool_calls")]
    fin = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    check(len(tc) >= 1 and tc[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "get_time",
          "stream: tool_calls delta for get_time")
    check(fin == ["tool_calls"], f"stream: finish_reason tool_calls (got {fin})")
    if tc:
        a = json.loads(tc[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"])
        check("tokyo" in json.dumps(a).lower(), f"stream: tz mentions Tokyo ({a})")

    # 4. forced tool via tool_choice
    r4 = post(port, dict(base, messages=[{"role": "user", "content": "Tell me about Rome."}],
                         tool_choice={"type": "function", "function": {"name": "get_weather"}}))
    c4 = r4["choices"][0]["message"].get("tool_calls") or []
    check(len(c4) == 1 and c4[0]["function"]["name"] == "get_weather", f"forced: exactly one get_weather call ({len(c4)})")
    check(r4["choices"][0]["finish_reason"] == "tool_calls", "forced: finish_reason tool_calls")

    # 4b. parallel_tool_calls=false stops after the first call
    r4b = post(port, dict(base, messages=[{"role": "user", "content": "Weather in Oslo and the time in Oslo, please."}],
                          parallel_tool_calls=False))
    c4b = r4b["choices"][0]["message"].get("tool_calls") or []
    check(len(c4b) == 1, f"parallel_tool_calls=false: one call ({[c['function']['name'] for c in c4b]})")
    if c4:
        check(isinstance(json.loads(c4[0]["function"]["arguments"]), dict), f"forced: args parse ({c4[0]['function']['arguments']})")

    # 5. tool_choice none
    r5 = post(port, dict(base, messages=msgs, tool_choice="none"))
    check(not r5["choices"][0]["message"].get("tool_calls"), "none: no tool calls")

    # 6. UTF-8 integrity in streamed/decoded text
    r6 = post(port, {"temperature": 0, "max_tokens": 200, "reasoning_effort": "none",
                     "messages": [{"role": "user", "content": "Write only the word Brasília and the word Zürich, nothing else."}]})
    t6 = (r6["choices"][0]["message"].get("content") or "") + (r6["choices"][0]["message"].get("reasoning_content") or "")
    check("�" not in t6 and ("Brasília" in t6 or "Zürich" in t6), f"utf8: no U+FFFD ({t6[:80]!r})")

    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
