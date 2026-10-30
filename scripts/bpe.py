#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = "~=3.10"
# dependencies = [
#   "regex",
# ]
# ///
"""
BPE-encode a prompt for the Bonsai Rust runtime.

Reads `vocab.bin`, `vocab_offsets.bin`, and `merges.txt` from a model directory
(produced by `scripts/extract.py`) and tokenizes a prompt into u32 token IDs.
Output is raw little-endian u32 bytes — pipe directly into the runtime:

  uv run scripts/bpe.py ./model "Once upon a time" \\
    | cargo run --release --features bench-internals -- ./model

Modes:
  positional `prompt` arg     -> encode that string
  no `prompt` arg, stdin tty  -> error (must specify)
  no `prompt` arg, stdin pipe -> read prompt text from stdin

Use `--out FILE` to write to a file instead of stdout. Use `--print-ids` to
also print decoded token ids (one per line) to stderr for debugging.

This script implements the GPT-2 byte-level BPE used by Qwen tokenizers, with
the pre-tokenizer regex from llama.cpp's qwen35 vocab branch. Added tokens
(`added_tokens.txt`) are matched atomically before BPE. It does NOT
depend on `gguf` or any GPU code — pure Python + the `regex` package.
"""
import argparse, os, re, struct, sys
import regex as re_u


# ---- byte-level encoding (GPT-2 / Qwen2) ------------------------------------

def gpt2_bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


# Qwen3.5 pretokenizer regex (matches llama.cpp's qwen35 branch in
# src/llama-vocab.cpp: Qwen2's regex with combining marks \p{M} folded into the
# letter classes). Uses the `regex` package for \p{L} / \p{M} / \p{N}.
QWEN35_PRETOK_RE = re_u.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
    r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|"
    r"\p{N}|"
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|"
    r"\s*[\r\n]+|"
    r"\s+(?!\S)|"
    r"\s+"
)


def build_special_re(specials):
    """Alternation over the added-token strings, longest first."""
    if not specials:
        return None
    alts = sorted(specials, key=len, reverse=True)
    return re.compile("|".join(re.escape(s) for s in alts))


def split_on_specials(text, special_re, specials):
    """Yield (kind, payload) where kind is 'text' (str) or 'special' (token id)."""
    pos = 0
    if special_re is not None:
        for m in special_re.finditer(text):
            if m.start() > pos:
                yield ("text", text[pos:m.start()])
            yield ("special", specials[m.group(0)])
            pos = m.end()
    if pos < len(text):
        yield ("text", text[pos:])


def bpe_encode(text, vocab, merges_rank):
    """Encode `text` through the BPE merge table, returning a list of token ids."""
    b2u = gpt2_bytes_to_unicode()
    ids = []
    for m in QWEN35_PRETOK_RE.finditer(text):
        chunk = m.group(0)
        word = [b2u[b] for b in chunk.encode("utf-8")]
        if not word:
            continue
        # Greedy lowest-rank-pair merge until no merges apply.
        while len(word) > 1:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            best = min(pairs, key=lambda p: merges_rank.get(p, 1 << 30))
            if best not in merges_rank:
                break
            new = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and (word[i], word[i+1]) == best:
                    new.append(word[i] + word[i+1])
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = new
        for tok in word:
            tid = vocab.get(tok)
            if tid is None:
                # Fall back to single-char tokens (always present in GPT-2 vocab).
                for ch in tok:
                    tid2 = vocab.get(ch)
                    if tid2 is None:
                        raise RuntimeError(f"BPE: unknown token piece {ch!r}")
                    ids.append(tid2)
            else:
                ids.append(tid)
    return ids


# ---- model-dir loading -------------------------------------------------------

def load_vocab(model_dir):
    """Return (vocab: str -> id, id_to_str: list[str])."""
    vocab_path = os.path.join(model_dir, "vocab.bin")
    offs_path = os.path.join(model_dir, "vocab_offsets.bin")
    with open(vocab_path, "rb") as f:
        vocab_bytes = f.read()
    with open(offs_path, "rb") as f:
        offs_raw = f.read()
    n_offs = len(offs_raw) // 4
    offs = struct.unpack(f"<{n_offs}I", offs_raw)
    n_vocab = n_offs - 1
    # vocab.bin is the GGUF-side byte-level encoded form (each token is a UTF-8
    # string of GPT-2 codepoints). bpe_encode() works in that same space, so we
    # decode the bytes as UTF-8 and use the resulting strings as merge keys.
    vocab = {}
    id_to_str = []
    for i in range(n_vocab):
        s = vocab_bytes[offs[i]:offs[i+1]].decode("utf-8", errors="replace")
        vocab[s] = i
        id_to_str.append(s)
    return vocab, id_to_str


def load_specials(model_dir, id_to_str):
    """Added (CONTROL / USER_DEFINED) tokens: literal string -> id."""
    path = os.path.join(model_dir, "added_tokens.txt")
    if not os.path.exists(path):
        sys.exit(f"missing {path}: re-run scripts/extract.py")
    specials = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tid = int(line)
                specials[id_to_str[tid]] = tid
    return specials


def load_merges(model_dir):
    merges_path = os.path.join(model_dir, "merges.txt")
    if not os.path.exists(merges_path):
        sys.exit(f"missing {merges_path}: re-run scripts/extract.py to "
                 "regenerate the model dir (older extracts didn't include it).")
    merges_rank = {}
    with open(merges_path, "r", encoding="utf-8") as f:
        for rank, line in enumerate(f):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            try:
                a, b = line.split(" ", 1)
            except ValueError:
                sys.exit(f"merges.txt:{rank+1}: malformed line: {line!r}")
            merges_rank[(a, b)] = rank
    return merges_rank


# ---- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="BPE-encode a prompt into u32 token IDs for bonsai-pot.",
    )
    ap.add_argument("model_dir", help="path to extracted model dir "
                                      "(must contain vocab.bin and merges.txt)")
    ap.add_argument("prompt", nargs="?", default=None,
                    help="prompt text. If omitted, read from stdin.")
    ap.add_argument("--out", "-o", default=None,
                    help="write u32 bytes to this file (default: stdout)")
    ap.add_argument("--print-ids", action="store_true",
                    help="also print decoded token ids to stderr")
    ap.add_argument("--no-specials", action="store_true",
                    help="byte-level-encode added-token literals (`<|im_start|>`, "
                         "`<think>`, ...) instead of emitting them as their atomic "
                         "vocab IDs (default: split on them so the chat template "
                         "renders correctly).")
    args = ap.parse_args()

    if args.prompt is None:
        if sys.stdin.isatty():
            ap.error("no prompt given and stdin is a tty; either pass a prompt "
                     "argument or pipe it in")
        text = sys.stdin.read()
        # Strip a single trailing newline (common when piping `echo "..."`).
        if text.endswith("\n"):
            text = text[:-1]
    else:
        text = args.prompt

    vocab, id_to_str = load_vocab(args.model_dir)
    merges = load_merges(args.model_dir)
    specials = load_specials(args.model_dir, id_to_str)
    special_re = build_special_re(specials)

    if args.no_specials:
        ids = bpe_encode(text, vocab, merges)
    else:
        ids = []
        for kind, payload in split_on_specials(text, special_re, specials):
            if kind == "special":
                ids.append(payload)
            else:
                ids.extend(bpe_encode(payload, vocab, merges))
    payload = struct.pack(f"<{len(ids)}I", *ids)

    if args.out:
        with open(args.out, "wb") as f:
            f.write(payload)
    else:
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()

    if args.print_ids:
        print(f"# {len(ids)} tokens", file=sys.stderr)
        for tid in ids:
            print(tid, file=sys.stderr)


if __name__ == "__main__":
    main()
