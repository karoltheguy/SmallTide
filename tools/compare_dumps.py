"""Compares two SMALLTIDE_DUMP_DIR dumps (e.g. batched vs token-by-token prefill).

    python -I tools/compare_dumps.py DUMP_A DUMP_B
"""

import glob
import os
import struct
import sys

import numpy as np


def load(d):
    rows = {}
    for f in glob.glob(os.path.join(d, "tok*.bin")):
        b = open(f, "rb").read()
        magic, pos, tok, nl, h, has_logits = struct.unpack_from("<6I", b, 0)
        off = 24
        hid = np.frombuffer(b, np.float32, nl * h, off).reshape(nl, h)
        off += nl * h * 4 + nl * 8 * 4
        routes = np.frombuffer(b, np.uint32, nl * 8, 24 + nl * h * 4).reshape(nl, 8) & 0x7FFFFFFF
        logits = np.frombuffer(b, np.float32, offset=off) if has_logits else None
        rows[pos] = (tok, hid, routes, logits)
    return rows


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    common = sorted(set(a) & set(b))
    worst = 0.0
    route_diff = 0
    per_layer = None
    for p in common:
        ta, ha, ra, la = a[p]
        tb, hb, rb, lb = b[p]
        assert ta == tb, f"token mismatch at {p}"
        rel = np.linalg.norm(ha - hb, axis=1) / np.maximum(np.linalg.norm(hb, axis=1), 1e-12)
        per_layer = rel if per_layer is None else np.maximum(per_layer, rel)
        worst = max(worst, float(rel.max()))
        route_diff += int(sum(len(set(x) ^ set(y)) // 2 for x, y in zip(ra, rb)))
    print(f"{len(common)} common positions; worst layer rel diff {worst:.3e}; routed-expert differences {route_diff}")
    print("per-layer worst rel diff:", " ".join(f"{v:.1e}" for v in per_layer))
    for p in common:
        la, lb = a[p][3], b[p][3]
        if la is not None and lb is not None:
            print(f"pos {p}: argmax {int(la.argmax())} vs {int(lb.argmax())}, max|dlogit| {np.abs(la - lb).max():.4f}")


if __name__ == "__main__":
    main()
