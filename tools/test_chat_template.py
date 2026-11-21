"""Checks SmallTide's chat_template port (`smalltide-cli render-chat`) against transformers.

Renders every case with the checkpoint's chat_template.jinja through transformers' own jinja
environment (the one `apply_chat_template` uses) and compares bytes. Cases where the template
raises must be refused by SmallTide too.

    python -I tools/test_chat_template.py --model models/Inkling-2-Mini --cli target/release/smalltide-cli.exe
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

from transformers.utils.chat_template_utils import _compile_jinja_template

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                          "required": ["city"]}}}
SEARCH = {"type": "function", "function": {"parameters": {"type": "object", "properties": {
    "q": {"type": "string", "description": "a <b>query</b> & 'more' – ünïcode ✓\n\"quoted\"\t\u0001"},
    "limit": {"type": "number", "default": 15000000000.0}, "frac": {"default": 0.1}, "tiny": {"default": 1e-07},
    "whole": {"default": 100.0}, "flag": {"default": False}, "none": {"default": None}, "Z": {}, "a": {}, "é": {}}},
    "name": "search", "strict": True}}
FLAT = {"name": "flat_tool", "description": "no function wrapper",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}
ODD = {"type": "custom", "name": "odd", "description": 0, "parameters": []}

U = "UNSET"
CASES = [
    ([{"role": "user", "content": "Hello"}], None, U),
    ([{"role": "user", "content": "Hello"}], None, None),
    ([{"role": "user", "content": "Hello"}], None, "medium"),
    ([{"role": "user", "content": "Hello"}], None, " high "),
    ([{"role": "user", "content": "Hello"}], None, 0),
    ([{"role": "user", "content": "Hello"}], None, 0.5),
    ([{"role": "user", "content": "Hello"}], None, 0.99),
    ([{"role": "user", "content": "Hello"}], None, "bogus"),
    ([{"role": "user", "content": "Hello"}], None, 1.5),
    ([{"role": "system", "content": "Be brief."}, {"role": "system", "content": "Really."},
      {"role": "user", "content": "Hello"}], None, "low"),
    ([{"role": "system", "content": "only system"}], None, U),
    ([{"role": "user", "content": "Hello"}], [WEATHER], U),
    ([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello"}],
     [WEATHER, SEARCH, FLAT, ODD], U),
    ([{"role": "user", "content": "Hello"}], [], U),
    ([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather?"},
      {"role": "assistant", "content": "Checking.", "reasoning_content": "think1", "tool_calls": [
          {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"days": 3, "city": "Paris"}'}},
          {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}}]},
      {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
      {"role": "tool", "tool_call_id": "c2", "content": "rainy"},
      {"role": "assistant", "content": "Done."}, {"role": "user", "content": "Thanks"}], [WEATHER], "low"),
    ([{"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "type": "function", "function": {
          "name": "search", "arguments": '{"q": "x <y> & \'z\' é", "n": [1, 2.5, null, true], "o": {"b": 1, "a": {}}}'}}]},
      {"role": "tool", "name": "search", "tool_call_id": "a", "content": "{\"hits\": 0}"}], None, U),
    ([{"role": "user", "content": "Hello"},
      {"role": "assistant", "content": None, "tool_calls": [
          {"id": "a", "type": "function", "function": {"name": "f", "arguments": None}},
          {"id": "b", "type": "function", "function": {"name": "g", "arguments": ""}},
          {"id": "c", "type": "function", "function": {"name": "h", "arguments": {}}}]},
      {"role": "tool", "tool_call_id": "b", "content": "1"}, {"role": "tool", "tool_call_id": "zz", "content": "2"},
      {"role": "tool", "content": "3"},
      {"role": "tool", "tool_call_id": "c", "name": "", "content": [{"type": "text", "text": "4"}, "5"]}], None, U),
    ([{"role": "user", "content": "Hello"},
      {"role": "assistant", "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "first", "arguments": {}}}]},
      {"role": "tool", "tool_call_id": "call_0", "content": "r1"},
      {"role": "assistant", "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "second", "arguments": {"k": 1}}}]},
      {"role": "tool", "tool_call_id": "call_0", "content": "r2"}], None, U),
    ([{"role": "user", "content": "Hello"},
      {"role": "assistant", "name": "f", "content": "x", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f", "arguments": {"a": 1}}}]}], None, U),
    ([{"role": "user", "content": "Hello"},
      {"role": "assistant", "name": "g", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f", "arguments": {"a": 1}}}]}], None, U),
    ([{"role": "user", "content": "Hello"},
      {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}, {"id": 1, "function": {"name": "g"}}]},
      {"role": "tool", "tool_call_id": 1.0, "content": "num id"}], None, U),
    ([{"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "input_text", "text": "two"},
                                   {"text": "untyped"}, "plain"]},
      {"role": "assistant", "reasoning_content": "r only"}], None, U),
    ([{"role": "user", "content": "multi-byte: Brasília 日本 😀"},
      {"role": "assistant", "content": "ok", "reasoning_content": ""},
      {"role": "user", "content": ""}], None, U),
    ([{"role": "user", "content": "x"}, {"role": "assistant", "tool_calls": [
        {"id": "a", "function": {"name": "f", "arguments": [1]}}]}], None, U),
    ([{"role": "user", "content": "x"}, {"role": "assistant", "tool_calls": [
        {"id": "a", "function": {"name": 7, "arguments": {}}}]}], None, U),
    ([{"role": "wizard", "content": "x"}], None, U),
    ([{"role": "user", "content": "x"}, {"role": "assistant", "tool_calls": [{"id": "a", "function": {
        "name": "f", "arguments": '{"z": {"y": 2, "x": [3, {"b": 1, "a": 0}]}, "a": 1e400, "m": -0, "f": 1.10}'}}]}], None, U),
]


def to_template_args(msgs):
    """What apply_chat_template's caller (the server) hands the template: string arguments parsed."""
    out = json.loads(json.dumps(msgs))
    for m in out:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if isinstance(fn.get("arguments"), str) and fn["arguments"].strip():
                fn["arguments"] = json.loads(fn["arguments"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cli", required=True)
    a = ap.parse_args()
    tpl = _compile_jinja_template(open(os.path.join(a.model, "chat_template.jinja"), encoding="utf-8").read())
    cases_json = []
    refs = []
    for msgs, tools, effort in CASES:
        c = {"messages": msgs}
        kw = {"messages": to_template_args(msgs), "add_generation_prompt": False}
        if tools is not None:
            c["tools"] = tools
            kw["tools"] = tools
        if effort != U:
            c["reasoning_effort"] = effort
            kw["reasoning_effort"] = effort
        cases_json.append(c)
        try:
            refs.append(("ok", tpl.render(**kw)))
        except Exception as e:  # template raised
            refs.append(("err", f"{type(e).__name__}: {e}"))
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "cases.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cases_json, f, ensure_ascii=False)
        res = json.loads(subprocess.run([a.cli, "render-chat", p], capture_output=True, text=True,
                                        encoding="utf-8", check=True).stdout)
    bad = 0
    for i, ((kind, ref), got) in enumerate(zip(refs, res)):
        if kind == "ok" and got.get("ok") == ref:
            print(f"ok   case {i}")
        elif kind == "err" and "err" in got:
            print(f"ok   case {i} (both refuse: ref={ref!r} ours={got['err']!r})")
        else:
            bad += 1
            print(f"FAIL case {i}\n  ref ({kind}): {ref!r}\n  ours: {got!r}")
    print(f"{len(CASES) - bad}/{len(CASES)} cases match transformers")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
