#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = "~=3.10"
# dependencies = [
#   "numpy~=2.4",
# ]
# ///
"""
Compare two tensor dumps (index.jsonl + raw f32 files) — typically the
reference produced by scripts/llama_dump against bonsai-pot's `--mode dump`.

For every tensor name present in both dumps, reports cosine similarity and
max-abs / relative error, computed per token column (dim 1) and summarized as
the minimum cosine across columns. For `logits`, reports top-1 agreement and
mean top-5 overlap per position.

Usage:
  uv run scripts/compare.py REF_DIR OURS_DIR [--names REGEX] [--min-cos 0.999] [--min-top1 0.95]

Exit status is 1 if any compared tensor's min cosine is below --min-cos or the
logits top-1 agreement is below --min-top1.
"""
import argparse, json, os, re, sys
import numpy as np


def load_index(d):
    out = {}
    with open(os.path.join(d, "index.jsonl")) as f:
        for line in f:
            e = json.loads(line)
            out.setdefault(e["name"], e)  # first occurrence wins
    return out


def load(d, e):
    a = np.fromfile(os.path.join(d, e["file"]), dtype="<f4")
    ne = [n for n in e["ne"]]
    cols = int(np.prod(ne[1:]))
    return a.reshape(cols, ne[0])


def layer_key(name):
    m = re.match(r"(.*?)-(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("ours")
    ap.add_argument("--names", default=None, help="only compare names matching this regex")
    ap.add_argument("--min-cos", type=float, default=0.999)
    ap.add_argument("--min-top1", type=float, default=0.95)
    ap.add_argument("--quiet", action="store_true", help="only print failures and the summary")
    args = ap.parse_args()

    ref, ours = load_index(args.ref), load_index(args.ours)
    common = [n for n in ref if n in ours and n != "logits"]
    if args.names:
        rx = re.compile(args.names)
        common = [n for n in common if rx.search(n)]
    common.sort(key=lambda n: (layer_key(n)[1], layer_key(n)[0]))

    failed = False
    worst = (2.0, None)
    for n in common:
        a, b = load(args.ref, ref[n]), load(args.ours, ours[n])
        if a.shape != b.shape:
            # Allow ours to carry a different column count (e.g. only last row)
            k = min(a.shape[0], b.shape[0])
            if a.shape[1] != b.shape[1] or k == 0:
                print(f"{n:32s} SHAPE MISMATCH ref={a.shape} ours={b.shape}")
                failed = True
                continue
            a, b = a[-k:], b[-k:]
        num = np.sum(a * b, axis=1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30
        cos = num / den
        mincos = float(np.min(cos))
        maxabs = float(np.max(np.abs(a - b)))
        rel = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))
        bad = mincos < args.min_cos or not np.isfinite(b).all()
        failed |= bad
        if mincos < worst[0]:
            worst = (mincos, n)
        if bad or not args.quiet:
            print(f"{n:32s} min_cos={mincos:.6f} rel={rel:.3e} max_abs={maxabs:.3e}"
                  f" |ref|max={float(np.max(np.abs(a))):.3e}{'  <-- FAIL' if bad else ''}")

    if "logits" in ref and "logits" in ours:
        a, b = load(args.ref, ref["logits"]), load(args.ours, ours["logits"])
        k = min(a.shape[0], b.shape[0])
        a, b = a[-k:], b[-k:]
        top1 = float(np.mean(np.argmax(a, axis=1) == np.argmax(b, axis=1)))
        ta = np.argsort(-a, axis=1)[:, :5]
        tb = np.argsort(-b, axis=1)[:, :5]
        ov = float(np.mean([len(set(x) & set(y)) / 5 for x, y in zip(ta, tb)]))
        num = np.sum(a * b, axis=1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        cos = num / den
        bad = top1 < args.min_top1
        failed |= bad
        print(f"logits: positions={k} top1={top1:.4f} top5_overlap={ov:.4f}"
              f" min_cos={float(np.min(cos)):.6f}{'  <-- FAIL' if bad else ''}")

    print(f"compared {len(common)} tensors; worst min_cos={worst[0]:.6f} ({worst[1]})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
