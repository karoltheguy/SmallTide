#!/usr/bin/env python3
"""Top-1 agreement of adaptive expert loading with the full model.

The engine keeps only part of the experts on the card and, under the default
`SMALLTIDE_RESCUE_MODE=threshold`, skips low-weight experts that are not there. This measures how
often that changes the most likely next token.

Two passes of `smalltide-cli run` over the same prompts (the server must be stopped: both need
the card):

  1. reference: `SMALLTIDE_RESCUE_MODE=exact SMALLTIDE_SPEC=0`. Every routed expert is copied in
     and run ("full streaming"). Greedy decoding; the tokens are saved.
  2. adaptive: your settings (.env). The same prompts, fed the reference's tokens one position at
     a time (teacher forcing). At each position its own most likely next token is compared with
     the reference's. Measured twice: through draft-verify passes (what speculative decoding
     uses) and through single-token steps.

Teacher forcing keeps every position comparable; two free-running generations say nothing after
their first difference. The prompt itself is read (prefilled) by each pass with its own settings,
so the adaptive pass also carries the experts its prefill skipped.

Default size: 4 workloads x 3 runs x 1,024 tokens, about 15 minutes on an RX 9070 XT.

  python bench/agreement.py
  python bench/agreement.py --workloads prose code --runs 1 --tokens 256     # about 2 minutes
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from bench import WORKLOADS  # noqa: E402  (the prose / code / edit prompts of bench.py)


def long_prompt():
    """A long-context prompt (about 10K tokens) built from the repository's own documents."""
    parts = []
    for rel in ("README.md", "LIMITATIONS.md", os.path.join("bench", "edit_source.py.txt")):
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            parts.append(f"===== {rel} =====\n{f.read()}")
    return (
        "Below are the documents of a software project. Write a thorough technical review of the "
        "project for a new contributor: what it does, how its parts fit together, its main "
        "trade-offs, and what you would check first. Use continuous prose.\n\n" + "\n\n".join(parts)
    )


def server_running(port):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
        return True
    except Exception:
        return False


def run_pass(exe, args, env_extra, log_path):
    env = dict(os.environ)
    env.update(env_extra)
    # `run` does not update the routing priors, so the passes do not disturb the server's.
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([exe, "run"] + args, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            log.write(line)
            if line.startswith("[agree]") or line.startswith("[Ready") or "error" in line.lower() or "fatal" in line.lower():
                print("  " + line.rstrip()[:230], flush=True)
        rc = proc.wait()
    if rc != 0:
        sys.exit(f"smalltide-cli exited with {rc}; see {log_path}")
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", default=os.path.join(ROOT, "target", "release", "smalltide-cli.exe" if os.name == "nt" else "smalltide-cli"))
    ap.add_argument("--workloads", nargs="+", default=["prose", "code", "edit", "long"], choices=["prose", "code", "edit", "long"])
    ap.add_argument("--runs", type=int, default=3, help="prompts per workload (each starts with a different request tag)")
    ap.add_argument("--tokens", type=int, default=1024, help="tokens generated per prompt")
    ap.add_argument("--width", type=int, default=4, help="reference tokens per draft-verify pass (1..8)")
    ap.add_argument("--seed-salt", default="agreement", help="fixes the request tags")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "target", "agreement"))
    ap.add_argument("--port", default=os.environ.get("PORT", "8000"))
    a = ap.parse_args()

    if server_running(a.port):
        sys.exit(f"a server answers on port {a.port}: stop it first (this test needs the card to itself)")
    os.makedirs(a.out_dir, exist_ok=True)
    salt = int(hashlib.sha1(a.seed_salt.encode()).hexdigest()[:6], 16) % 100000
    prompts = []
    for w in a.workloads:
        text = long_prompt() if w == "long" else WORKLOADS[w]
        for i in range(a.runs):
            prompts.append({"name": f"{w}-{i + 1}", "prompt": f"[req {1000 + i + salt}] " + text})
    p_file = os.path.join(a.out_dir, "prompts.json")
    r_file = os.path.join(a.out_dir, "reference.json")
    o_file = os.path.join(a.out_dir, "adaptive.json")
    with open(p_file, "w", encoding="utf-8") as f:
        json.dump(prompts, f)
    common = ["--agree-prompts", p_file, "-n", str(a.tokens), "--temp", "0"]

    print(f"== reference pass: every routed expert run ({len(prompts)} prompts, up to {a.tokens} tokens each)", flush=True)
    t_ref = run_pass(a.exe, common + ["--agree-out", r_file], {"SMALLTIDE_RESCUE_MODE": "exact", "SMALLTIDE_SPEC": "0"}, os.path.join(a.out_dir, "reference.log"))
    print(f"== adaptive pass: your settings, fed the reference's tokens", flush=True)
    t_ada = run_pass(a.exe, common + ["--agree-ref", r_file, "--agree-out", o_file], {"SMALLTIDE_AGREE_WIDTH": str(a.width)}, os.path.join(a.out_dir, "adaptive.log"))

    with open(o_file, encoding="utf-8") as f:
        rows = json.load(f)
    print()
    print(f"{'prompt':<10} {'prompt tok':>10} {'positions':>9} | {'verify passes':>13} {'skipped':>8} | {'single steps':>12} {'skipped':>8} | first token")
    by_w = {}
    for r in rows:
        n = max(r["positions"], 1)
        v, s = r["verify"], r["single"]
        print(f"{r['name']:<10} {r['prompt_tokens']:>10} {r['positions']:>9} | {100 * v['agree'] / n:>12.2f}% {100 * v['skipped_weight']:>7.2f}% | {100 * s['agree'] / n:>11.2f}% {100 * s['skipped_weight']:>7.2f}% | {'same' if r['first_token_same'] else 'different'}")
        g = by_w.setdefault(r["name"].rsplit("-", 1)[0], [0, 0, 0])
        g[0] += r["positions"]
        g[1] += v["agree"]
        g[2] += s["agree"]
    print()
    tot = [0, 0, 0]
    summary = {"tokens": a.tokens, "runs": a.runs, "width": a.width, "seconds": {"reference": t_ref, "adaptive": t_ada}, "workloads": {}, "prompts": rows}
    for w, g in by_w.items():
        print(f"{w:<10} top-1 agreement: verify passes {100 * g[1] / max(g[0], 1):.2f}% | single steps {100 * g[2] / max(g[0], 1):.2f}%  ({g[0]} positions)")
        summary["workloads"][w] = {"positions": g[0], "verify_top1": g[1] / max(g[0], 1), "single_top1": g[2] / max(g[0], 1)}
        tot = [x + y for x, y in zip(tot, g)]
    print(f"{'all':<10} top-1 agreement: verify passes {100 * tot[1] / max(tot[0], 1):.2f}% | single steps {100 * tot[2] / max(tot[0], 1):.2f}%  ({tot[0]} positions)")
    summary["all"] = {"positions": tot[0], "verify_top1": tot[1] / max(tot[0], 1), "single_top1": tot[2] / max(tot[0], 1)}
    s_file = os.path.join(a.out_dir, "summary.json")
    with open(s_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"\n{(t_ref + t_ada) / 60:.1f} minutes ({t_ref / 60:.1f} reference + {t_ada / 60:.1f} adaptive); details in {s_file}")


if __name__ == "__main__":
    main()
