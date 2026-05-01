#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gguf",
#   "numpy",
# ]
#
# [tool.uv.sources]
# # PyPI's `gguf` lags upstream and (as of this writing) doesn't ship the
# # Q1_0 quantization type. Pull from llama.cpp master, which does.
# gguf = { git = "https://github.com/ggml-org/llama.cpp", rev = "a95a11e", subdirectory = "gguf-py" }
# ///
"""
Extract a Bonsai-4B Q1_0 GGUF into a flat directory the Rust runtime can load.

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
  merges.txt        BPE merges in rank order (one "a b\n" per line). Used by
                    `scripts/bpe.py` to encode prompts; not consumed by
                    the Rust runtime.

Q1_0 tensors are stored as their raw 18-byte blocks (LSB-first sign bits +
FP16 d at the front). Per-row stride is (n_in / 128) * 18 bytes; rows are
contiguous, total = n_out rows. Each tensor's region is 4-byte padded.

Norms (originally F32 in the GGUF) are downcast to F16 on disk so the
runtime can bind them as `array<f16>` without a load-time conversion.

Prompts are NOT encoded here. Run `scripts/bpe.py` for that.

Usage:
  uv run scripts/extract.py path/to/Bonsai-4B.gguf --out ./model
"""
import argparse, json, os, struct
import gguf
import numpy as np


def field_str(f):
    if f is None: return None
    return f.contents()


def main():
    ap = argparse.ArgumentParser(
        description="Extract Bonsai-4B GGUF weights/vocab into a flat model dir.",
    )
    ap.add_argument("gguf", help="path to Bonsai-4B.gguf")
    ap.add_argument("--out", default="./model",
                    help="output directory (default: ./model)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    r = gguf.GGUFReader(args.gguf)

    # ---- collect hyperparams ------------------------------------------------
    fld = lambda k: field_str(r.fields.get(k))
    tokens_field = r.fields["tokenizer.ggml.tokens"]
    cfg = {
        "n_layer":   int(fld("qwen3.block_count")),
        "n_embd":    int(fld("qwen3.embedding_length")),
        "n_ff":      int(fld("qwen3.feed_forward_length")),
        "n_head":    int(fld("qwen3.attention.head_count")),
        "n_kv_head": int(fld("qwen3.attention.head_count_kv")),
        "head_dim":  int(fld("qwen3.attention.key_length")),
        "rope_freq_base": float(fld("qwen3.rope.freq_base")),
        "rms_eps":   float(fld("qwen3.attention.layer_norm_rms_epsilon")),
        "n_vocab":   len(tokens_field.data),
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
    def pad4(n):
        return (n + 3) & ~3

    manifest = {"tensors": {}}

    def write_tensor(out_f, t, name, expected_dtype):
        """Write tensor bytes to current output file with a layout suitable for
        u32-aligned reads in WGSL. For Q1_0 we split into a d-array (FP16
        scales) followed by a qs-array (raw 16-byte sign blocks). Both halves
        are u32-aligned because n_rows*(K/128) is even for all our tensors.
        For F32 we emit f32 contiguous; for F16 we emit f16 contiguous (the
        GGUF source may be F32 or F16 — F32 sources are downcast on the fly)."""
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
            if expected_dtype == "F32":
                buf = np.ascontiguousarray(data, dtype=np.float32).tobytes()
            elif expected_dtype == "F16":
                buf = np.ascontiguousarray(data).astype(np.float16).tobytes()
            else:
                raise RuntimeError(f"{name}: F32 source cannot produce {expected_dtype}")
            out_f.write(buf)
            pad = pad4(out_f.tell()) - out_f.tell()
            if pad: out_f.write(b"\x00" * pad)
        elif t.tensor_type == gguf.GGMLQuantizationType.F16:
            assert expected_dtype == "F16", f"{name}: F16 source must produce F16"
            buf = np.ascontiguousarray(data, dtype=np.float16).tobytes()
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

    # ---- group D: weights_norms (norms downcast to F16 for the f16 runtime) ---
    with open(os.path.join(args.out, "weights_norms.bin"), "wb") as f:
        for il in range(n_layer):
            for tag in ("attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm"):
                name = f"blk.{il}.{tag}.weight"
                write_tensor(f, by_name[name], name, "F16")
        write_tensor(f, by_name["output_norm.weight"], "output_norm.weight", "F16")

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
    n_vocab = cfg["n_vocab"]
    vocab_bytes = bytearray()
    offsets = [0]
    for i in tokens_field.data:
        b = bytes(tokens_field.parts[i])
        vocab_bytes.extend(b)
        offsets.append(len(vocab_bytes))
    with open(os.path.join(args.out, "vocab.bin"), "wb") as f:
        f.write(bytes(vocab_bytes))
    with open(os.path.join(args.out, "vocab_offsets.bin"), "wb") as f:
        f.write(struct.pack(f"<{n_vocab+1}I", *offsets))

    # ---- merges (for offline tokenization via scripts/bpe.py) ----------
    # Each merge is a space-separated pair of byte-level-encoded strings, in
    # rank order (one per line). Plain UTF-8 text — no header.
    merges_field = r.fields["tokenizer.ggml.merges"]
    with open(os.path.join(args.out, "merges.txt"), "w", encoding="utf-8") as f:
        for idx in merges_field.data:
            line = bytes(merges_field.parts[idx]).decode("utf-8")
            # Sanity: a merges entry is "<a> <b>". Pass through unchanged.
            f.write(line)
            if not line.endswith("\n"):
                f.write("\n")

    # ---- final config -------------------------------------------------------
    cfg["manifest"] = manifest["tensors"]
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    sizes = {os.path.basename(p): os.path.getsize(os.path.join(args.out, p))
             for p in os.listdir(args.out) if p.endswith(".bin")}
    print("output bytes:", sizes)
    print(f"merges.txt:  {os.path.getsize(os.path.join(args.out, 'merges.txt'))} bytes")


if __name__ == "__main__":
    main()
