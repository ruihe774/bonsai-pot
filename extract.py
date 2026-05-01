#!/usr/bin/env python3
"""
Extract Bonsai-4B.gguf into a flat directory the Rust runtime can load.

Output layout (under --out, default ./model):
  config.json       hyperparams + tensor manifest (offsets & shapes within
                    the 5 grouped weight buffers)
  weights_attn.bin
  weights_ffn_gate_up.bin
  weights_ffn_down.bin
  weights_norms.bin
  weights_embed_lmhead.bin
  vocab.bin         flat utf-8 bytes for all tokens
  vocab_offsets.bin u32 array of length n_vocab+1: byte offsets into vocab.bin
  prompt.bin        u32 array of token IDs

Q1_0 tensors are stored as their raw 18-byte blocks (LSB-first sign bits +
FP16 d at the front). Per-row stride is (n_in / 128) * 18 bytes; rows are
contiguous, total = n_out rows. Each tensor's region is 4-byte padded.

Norms (originally F32) are written verbatim as F32.
"""
import argparse, json, os, struct, sys
sys.path.insert(0, '/tmp/llama.cpp/gguf-py')
import numpy as np
import gguf

# --- minimal byte-level BPE encoder for GPT-2-style (+ Qwen2 pre-tok) ----------

def gpt2_bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

# Qwen2 pretokenizer regex (from
# /tmp/llama.cpp/src/llama-vocab.cpp, qwen2 branch):
import regex as re_u
QWEN2_PRETOK_RE = re_u.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
    r"[^\r\n\p{L}\p{N}]?\p{L}+|"
    r"\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|"
    r"\s*[\r\n]+|"
    r"\s+(?!\S)|"
    r"\s+"
)

def bpe_encode(text: str, vocab: dict, merges_rank: dict) -> list[int]:
    """Encode one chunk's worth of bytes through the BPE merge table."""
    b2u = gpt2_bytes_to_unicode()
    ids = []
    for m in QWEN2_PRETOK_RE.finditer(text):
        chunk = m.group(0)
        # byte->unicode then split into single-char tokens
        word = [b2u[b] for b in chunk.encode("utf-8")]
        if not word:
            continue
        # BPE merge loop
        while len(word) > 1:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            best = min(pairs, key=lambda p: merges_rank.get(p, 1<<30))
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
                # fall back to single byte tokens (always present in GPT-2 vocab)
                for ch in tok:
                    tid2 = vocab.get(ch)
                    if tid2 is None:
                        raise RuntimeError(f"BPE: unknown token piece {ch!r}")
                    ids.append(tid2)
            else:
                ids.append(tid)
    return ids

# --- main extraction ----------------------------------------------------------

def field_str(f):
    if f is None: return None
    return f.contents()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", help="path to Bonsai-4B.gguf")
    ap.add_argument("--out", default="./model")
    ap.add_argument("--prompt", default="Once upon a time")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    r = gguf.GGUFReader(args.gguf)

    # ---- collect hyperparams ------------------------------------------------
    fld = lambda k: field_str(r.fields.get(k))
    cfg = {
        "n_layer":   int(fld("qwen3.block_count")),
        "n_embd":    int(fld("qwen3.embedding_length")),
        "n_ff":      int(fld("qwen3.feed_forward_length")),
        "n_head":    int(fld("qwen3.attention.head_count")),
        "n_kv_head": int(fld("qwen3.attention.head_count_kv")),
        "head_dim":  int(fld("qwen3.attention.key_length")),
        "rope_freq_base": float(fld("qwen3.rope.freq_base")),
        "rms_eps":   float(fld("qwen3.attention.layer_norm_rms_epsilon")),
        "n_vocab":   151669,
        "eos_token_id":     int(fld("tokenizer.ggml.eos_token_id")),
        "padding_token_id": int(fld("tokenizer.ggml.padding_token_id")),
        "add_bos":   bool(fld("tokenizer.ggml.add_bos_token") or False),
        "context_length": int(fld("qwen3.context_length")),
        "rope_orig_context": int(fld("qwen3.rope.scaling.original_context_length")),
    }
    cfg["n_kv_groups"] = cfg["n_head"] // cfg["n_kv_head"]
    cfg["q_dim"]  = cfg["n_head"] * cfg["head_dim"]
    cfg["kv_dim"] = cfg["n_kv_head"] * cfg["head_dim"]

    print("config:", json.dumps(cfg, indent=2))

    # ---- index tensors by name -----------------------------------------------
    by_name = {t.name: t for t in r.tensors}
    n_layer = cfg["n_layer"]

    QK1_0 = 128
    BLK_BYTES = 18
    def q1_row_bytes(n_in):
        assert n_in % QK1_0 == 0
        return (n_in // QK1_0) * BLK_BYTES

    def pad4(n):
        return (n + 3) & ~3

    manifest = {"tensors": {}}

    def write_tensor(out_f, t, name, expected_dtype):
        """Write tensor bytes to current output file with a layout suitable for
        u32-aligned reads in WGSL. For Q1_0 we split into a d-array (FP16
        scales) followed by a qs-array (raw 16-byte sign blocks). Both halves
        are u32-aligned because n_rows*(K/128) is even for all our tensors.
        For F32 / F16 we just emit FP32 contiguous."""
        shape = [int(s) for s in t.shape]
        entry = {
            "dtype": expected_dtype,
            "shape": shape,
            "buffer": os.path.basename(out_f.name),
            "offset": out_f.tell(),
        }
        data = t.data
        if t.tensor_type == gguf.GGMLQuantizationType.Q1_0:
            assert expected_dtype == "Q1_0", f"{name}: dtype mismatch"
            n_in, n_out = shape[0], shape[1] if len(shape) > 1 else 1
            nb = n_in // QK1_0
            raw = data.reshape(n_out, nb, BLK_BYTES)  # (n_out, nb, 18)
            # split: d_arr (n_out, nb, 2 bytes), qs_arr (n_out, nb, 16 bytes)
            d_arr  = np.ascontiguousarray(raw[:, :, :2])
            qs_arr = np.ascontiguousarray(raw[:, :, 2:])
            d_offset = out_f.tell()
            out_f.write(d_arr.tobytes())
            pad = pad4(out_f.tell()) - out_f.tell()
            if pad: out_f.write(b"\x00" * pad)
            qs_offset = out_f.tell()
            out_f.write(qs_arr.tobytes())
            pad = pad4(out_f.tell()) - out_f.tell()
            if pad: out_f.write(b"\x00" * pad)
            entry["d_offset"]  = d_offset
            entry["qs_offset"] = qs_offset
            entry["nb"] = nb
        elif t.tensor_type == gguf.GGMLQuantizationType.F32:
            assert expected_dtype == "F32", f"{name}: dtype mismatch"
            buf = np.ascontiguousarray(data, dtype=np.float32).tobytes()
            out_f.write(buf)
            pad = pad4(out_f.tell()) - out_f.tell()
            if pad: out_f.write(b"\x00" * pad)
        elif t.tensor_type == gguf.GGMLQuantizationType.F16:
            buf = np.ascontiguousarray(data, dtype=np.float16).astype(np.float32).tobytes()
            out_f.write(buf)
            pad = pad4(out_f.tell()) - out_f.tell()
            if pad: out_f.write(b"\x00" * pad)
        else:
            raise RuntimeError(f"{name}: unsupported tensor type {t.tensor_type}")
        entry["length"] = out_f.tell() - entry["offset"]
        manifest["tensors"][name] = entry

    # ---- group A: weights_attn (per-layer Wq, Wk, Wv, Wo) -------------------
    with open(os.path.join(args.out, "weights_attn.bin"), "wb") as f:
        for il in range(n_layer):
            for tag in ("attn_q", "attn_k", "attn_v", "attn_output"):
                name = f"blk.{il}.{tag}.weight"
                write_tensor(f, by_name[name], name, "Q1_0")

    # ---- group B: weights_ffn_gate_up (Wgate, Wup) --------------------------
    with open(os.path.join(args.out, "weights_ffn_gate_up.bin"), "wb") as f:
        for il in range(n_layer):
            for tag in ("ffn_gate", "ffn_up"):
                name = f"blk.{il}.{tag}.weight"
                write_tensor(f, by_name[name], name, "Q1_0")

    # ---- group C: weights_ffn_down -----------------------------------------
    with open(os.path.join(args.out, "weights_ffn_down.bin"), "wb") as f:
        for il in range(n_layer):
            name = f"blk.{il}.ffn_down.weight"
            write_tensor(f, by_name[name], name, "Q1_0")

    # ---- group D: weights_norms (all FP32 norms incl. q_norm/k_norm) -------
    with open(os.path.join(args.out, "weights_norms.bin"), "wb") as f:
        for il in range(n_layer):
            for tag in ("attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm"):
                name = f"blk.{il}.{tag}.weight"
                write_tensor(f, by_name[name], name, "F32")
        write_tensor(f, by_name["output_norm.weight"], "output_norm.weight", "F32")

    # ---- group E: weights_embed_lmhead -------------------------------------
    # Bonsai-4B has tied embeddings (no separate output.weight). The LM head
    # uses the same tensor as the embedding (row-gather for embed; full matvec
    # for the head, since the Q1_0 row-major layout [n_embd, n_vocab] gives
    # n_vocab rows of n_embd elements — the right shape for both ops).
    cfg["tied_embeddings"] = "output.weight" not in by_name
    with open(os.path.join(args.out, "weights_embed_lmhead.bin"), "wb") as f:
        write_tensor(f, by_name["token_embd.weight"], "token_embd.weight", "Q1_0")
        if not cfg["tied_embeddings"]:
            write_tensor(f, by_name["output.weight"], "output.weight", "Q1_0")

    # ---- vocab dump --------------------------------------------------------
    tokens_field = r.fields["tokenizer.ggml.tokens"]
    # tokens_field.data is a list of indices into tokens_field.parts;
    # each entry's bytes view is the actual UTF-8 token bytes.
    n_vocab = len(tokens_field.data)
    assert n_vocab == cfg["n_vocab"]
    vocab_bytes = bytearray()
    offsets = [0]
    token_strs = []
    for i in tokens_field.data:
        b = bytes(tokens_field.parts[i])
        token_strs.append(b.decode("utf-8", errors="replace"))
        vocab_bytes.extend(b)
        offsets.append(len(vocab_bytes))
    with open(os.path.join(args.out, "vocab.bin"), "wb") as f:
        f.write(bytes(vocab_bytes))
    with open(os.path.join(args.out, "vocab_offsets.bin"), "wb") as f:
        f.write(struct.pack(f"<{n_vocab+1}I", *offsets))

    # ---- BPE encode prompt --------------------------------------------------
    vocab_dict = {s: i for i, s in enumerate(token_strs)}
    merges_field = r.fields["tokenizer.ggml.merges"]
    merges_rank = {}
    for rank, idx in enumerate(merges_field.data):
        b = bytes(merges_field.parts[idx]).decode("utf-8")
        a, c = b.split(" ", 1)
        merges_rank[(a, c)] = rank

    prompt_ids = bpe_encode(args.prompt, vocab_dict, merges_rank)
    print(f"prompt ({args.prompt!r}) -> {len(prompt_ids)} tokens: {prompt_ids}")
    print(f"  decoded: {''.join(token_strs[t] for t in prompt_ids)!r}")

    with open(os.path.join(args.out, "prompt.bin"), "wb") as f:
        f.write(struct.pack(f"<{len(prompt_ids)}I", *prompt_ids))

    # ---- final config -------------------------------------------------------
    cfg["manifest"] = manifest["tensors"]
    cfg["prompt_text"] = args.prompt
    cfg["prompt_n"] = len(prompt_ids)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    sizes = {os.path.basename(p): os.path.getsize(os.path.join(args.out, p))
             for p in os.listdir(args.out) if p.endswith(".bin")}
    print("output bytes:", sizes)

if __name__ == "__main__":
    main()
