"""Compare SmallTide's BpeTokenizer against HuggingFace `tokenizers` on a varied corpus.

    python -I tools/test_tokenizer.py --model models/Inkling-2-Mini --cli target/release/smalltide-cli.exe

Each case is encoded by both; the script prints per-case agreement and exits non-zero on mismatch.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

from tokenizers import Tokenizer

CASES = [
    "Hello world! How's it going? I'm fine, they're here, we'll see, you'd know, it's OK.",
    "HELLO WORLD, CamelCaseWords and snake_case_words and ALLCAPSthenlower.",
    "Numbers: 1 12 123 1234 12345 3.14159 -42 1e10 0x1F 2026-11-21",
    "  leading spaces,\ttabs\t\tand\n\nnewlines\r\n\r\nwindows   trailing   ",
    "Unicode: naïve café Brasília São Paulo Zürich Ελληνικά русский язык 日本語のテキスト 中文 한국어 العربية हिन्दी",
    "Emoji 😀👍🏽🇯🇵 and symbols ©®™ — – … « » ‘quotes’ “double”",
    "Combining marks: é ä ñ and ZWJ ‍ and NBSP here",
    "def f(x):\n    return {'a': [1, 2, 3], \"b\": x ** 2}  # comment\n\nprint(f(3))\n",
    "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>What is the capital of France?<|end_message|>",
    '<|message_system|>tool_declare<|content_xml|>[{"description":"Get weather","name":"get_weather","parameters":{"properties":{"city":{"type":"string"}},"required":["city"],"type":"object"},"type":"function"}]<|end_message|>',
    "Mixed: x=3;y=4//comment /* block */ a->b c::d e!=f g<=h i>=j k&&l m||n ...",
    "URLs https://example.com/path?q=1&r=2#frag and emails a.b@c.de",
    "\n\n\n   \n  \t\n",
    "don't DON'T Don't IT'S it's 'quoted' '' we'RE they'Ll x'd 's 'll",
    "Titlecase ǅungla ǈub modifier ʰh ˆx 々々 marks-first ́abc ̀ ",
    "digits ١٢٣٤ ①② x² Ⅻ 12345678 １２３４",
    "mixed nbsp 　ideographic　space  line para vtff",
    "a/b/c//\n/\r\n!!\n\n?? ...\n  - item\n  * item\n#hash @at $dollar %pct ^caret",
    "vertical\x0btab form\x0cfeed nbsp x line sep para sep",
    "👨‍👩‍👧 𝐀𝐚 math bold",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cli", required=True)
    ap.add_argument("--extra-glob", default="")
    a = ap.parse_args()
    tj = os.path.join(a.model, "tokenizer.json")
    hf = Tokenizer.from_file(tj)
    cases = list(CASES)
    for g in a.extra_glob.split(","):
        if g:
            for f in sorted(glob.glob(g, recursive=True)):
                cases.append(open(f, encoding="utf-8").read())
    bad = 0
    tot_tokens = 0
    with tempfile.TemporaryDirectory() as td:
        for i, text in enumerate(cases):
            p = os.path.join(td, f"c{i}.txt")
            with open(p, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            out = subprocess.run([a.cli, "tokenize", tj, p], capture_output=True, text=True, check=True)
            ours = json.loads(out.stdout.strip().splitlines()[-1])
            ref = hf.encode(text, add_special_tokens=False).ids
            tot_tokens += len(ref)
            if ours != ref:
                bad += 1
                k = next((j for j, (x, y) in enumerate(zip(ours, ref)) if x != y), min(len(ours), len(ref)))
                print(f"MISMATCH case {i} ({len(ours)} vs {len(ref)} tokens) first diff at {k}: "
                      f"ours={[hf.decode([t]) for t in ours[k:k+4]]} ref={[hf.decode([t]) for t in ref[k:k+4]]}")
            else:
                print(f"ok case {i} ({len(ref)} tokens)")
    print(f"{len(cases) - bad}/{len(cases)} cases identical ({tot_tokens} reference tokens)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
