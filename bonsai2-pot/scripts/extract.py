#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = "~=3.10"
# dependencies = [
#   "gguf @ git+https://github.com/PrismML-Eng/llama.cpp.git#subdirectory=gguf-py",
#   "numpy~=2.4",
# ]
# ///
"""
Extract a Ternary-Bonsai-2 (qwen35 architecture, PTQ1_0) GGUF into a flat
directory the Rust runtime can load.

Output layout (under --out, default ./model):
  config.ini                hyperparams + tensor manifest (offsets & shapes
                            within the grouped weight buffers)
  weights_attn.bin          GDN attn_qkv / attn_gate / ssm_out, full-attention
                            attn_q / attn_k / attn_v / attn_output (PTQ1_0)
  weights_ffn_gate_up.bin   ffn_gate / ffn_up (PTQ1_0)
  weights_ffn_down.bin      ffn_down (PTQ1_0)
  weights_embed_lmhead.bin  token_embd / output (PTQ1_0)
  weights_small.bin         f32 norms, conv1d, dt bias, ssm_a; BF16 ssm_alpha /
                            ssm_beta (raw u16); packed Hadamard sign bits
  vocab.bin                 flat utf-8 bytes for all tokens
  vocab_offsets.bin         u32 array of length n_vocab+1: byte offsets into vocab.bin
  added_tokens.txt          ids of CONTROL / USER_DEFINED tokens (one per line);
                            tokenizers must match these atomically before BPE
  merges.txt                BPE merges in rank order (one "a b\\n" per line)

PTQ1_0 tensors are written verbatim: 28-byte blocks (qs[24], qh[2], fp16 d) of
128 weights, row-major, one row per output feature. Every region is 4-byte
aligned (28 is a multiple of 4), so the shaders read blocks as 7 u32 words.

The file must carry `prism.hadamard.*` v1 metadata: every PTQ1_0 matmul weight
is stored in a blockwise-Hadamard-rotated input basis and the runtime applies
the matching activation transform. Anything unexpected is refused.

Usage:
  uv run scripts/extract.py path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf --out ./model
"""
import argparse, os, struct
import gguf
import numpy as np

ARCH = "qwen35"
GT = gguf.GGMLQuantizationType

TOKEN_TYPE_CONTROL = 3
TOKEN_TYPE_USER_DEFINED = 4


def main():
    ap = argparse.ArgumentParser(description="Extract a Ternary-Bonsai-2 PTQ1_0 GGUF into a flat model dir.")
    ap.add_argument("gguf", help="path to Ternary-Bonsai-2-*-PTQ1_0.gguf")
    ap.add_argument("--out", default="./model", help="output directory (default: ./model)")
    args = ap.parse_args()

    r = gguf.GGUFReader(args.gguf)

    def fld(k, default=None):
        f = r.fields.get(k)
        return default if f is None else f.contents()

    def req(k):
        v = fld(k)
        if v is None:
            raise SystemExit(f"missing required GGUF key: {k}")
        return v

    arch = req("general.architecture")
    if arch != ARCH:
        raise SystemExit(f"unsupported architecture {arch!r} (expected {ARCH!r})")

    tokens = req("tokenizer.ggml.tokens")
    token_types = req("tokenizer.ggml.token_type")

    cfg = {
        "arch": ARCH,
        "n_layer": int(req(f"{ARCH}.block_count")),
        "n_embd": int(req(f"{ARCH}.embedding_length")),
        "n_ff": int(req(f"{ARCH}.feed_forward_length")),
        "n_head": int(req(f"{ARCH}.attention.head_count")),
        "n_kv_head": int(req(f"{ARCH}.attention.head_count_kv")),
        "head_dim": int(req(f"{ARCH}.attention.key_length")),
        "rope_dim": int(req(f"{ARCH}.rope.dimension_count")),
        "rope_freq_base": float(req(f"{ARCH}.rope.freq_base")),
        "rms_eps": float(req(f"{ARCH}.attention.layer_norm_rms_epsilon")),
        "full_attn_interval": int(fld(f"{ARCH}.full_attention_interval", 4)),
        "ssm_conv_kernel": int(req(f"{ARCH}.ssm.conv_kernel")),
        "ssm_head_dim": int(req(f"{ARCH}.ssm.state_size")),
        "ssm_n_k_heads": int(req(f"{ARCH}.ssm.group_count")),
        "ssm_n_v_heads": int(req(f"{ARCH}.ssm.time_step_rank")),
        "ssm_inner": int(req(f"{ARCH}.ssm.inner_size")),
        "context_length": int(req(f"{ARCH}.context_length")),
        "n_vocab": len(tokens),
        "eos_token_id": int(req("tokenizer.ggml.eos_token_id")),
        "pad_token_id": int(fld("tokenizer.ggml.padding_token_id", 0)),
        "add_bos": bool(fld("tokenizer.ggml.add_bos_token", False)),
        "sampling_top_k": int(fld("general.sampling.top_k", 20)),
        "sampling_top_p": float(fld("general.sampling.top_p", 0.95)),
        "sampling_temp": float(fld("general.sampling.temp", 1.0)),
    }
    if int(req(f"{ARCH}.attention.value_length")) != cfg["head_dim"]:
        raise SystemExit("key_length != value_length is not supported")
    if cfg["ssm_inner"] != cfg["ssm_head_dim"] * cfg["ssm_n_v_heads"]:
        raise SystemExit("ssm.inner_size != state_size * time_step_rank is not supported")
    sections = fld(f"{ARCH}.rope.dimension_sections")
    if sections is not None and sum(sections) * 2 != cfg["rope_dim"]:
        raise SystemExit(f"unexpected rope sections {sections} for rope_dim {cfg['rope_dim']}")

    n_layer = cfg["n_layer"]
    interval = cfg["full_attn_interval"]
    is_attn = [((il + 1) % interval == 0) for il in range(n_layer)]

    # ---- Hadamard metadata validation ---------------------------------------
    hb = int(req("prism.hadamard.block_size"))
    checks = {
        "prism.hadamard.version": 1,
        "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
        "prism.hadamard.axis": "input-last-dimension",
        "prism.hadamard.sign_mode": "explicit",
    }
    for k, want in checks.items():
        got = req(k)
        if got != want:
            raise SystemExit(f"{k} = {got!r}, expected {want!r}")
    if bool(fld("prism.hadamard.tied_output", False)):
        raise SystemExit("prism.hadamard.tied_output (version 2) files are not supported")
    if not bool(fld("prism.hadamard.gdn_v_grouped", False)):
        raise SystemExit("prism.hadamard.gdn_v_grouped must be true")
    if hb != 1024:
        raise SystemExit(f"prism.hadamard.block_size = {hb}, only 1024 is supported")

    expected = {"output.weight"}
    for il in range(n_layer):
        tags = (["attn_q", "attn_k", "attn_v", "attn_output"] if is_attn[il]
                else ["attn_qkv", "attn_gate", "ssm_out"])
        tags += ["ffn_gate", "ffn_up", "ffn_down"]
        expected |= {f"blk.{il}.{t}.weight" for t in tags}
    names = set(req("prism.hadamard.weight_names"))
    if names != expected:
        raise SystemExit(
            f"prism.hadamard.weight_names mismatch: missing={sorted(expected - names)[:5]} "
            f"extra={sorted(names - expected)[:5]}")
    inv = list(req("prism.hadamard.inverse_weight_names"))
    if inv != ["token_embd.weight"]:
        raise SystemExit(f"prism.hadamard.inverse_weight_names = {inv}, expected ['token_embd.weight']")

    sign_widths = [int(w) for w in req("prism.hadamard.sign_widths")]
    sign_values = [int(v) for v in req("prism.hadamard.sign_values")]
    if sum(sign_widths) != len(sign_values):
        raise SystemExit("prism.hadamard.sign_values length mismatch")
    need_widths = {cfg["n_embd"], cfg["ssm_inner"], cfg["n_head"] * cfg["head_dim"], cfg["n_ff"]}
    if not need_widths <= set(sign_widths):
        raise SystemExit(f"missing Hadamard sign widths: {sorted(need_widths - set(sign_widths))}")
    for w in sign_widths:
        if w % hb != 0 or w % 32 != 0:
            raise SystemExit(f"Hadamard width {w} not a multiple of block {hb}")
    signs = {}
    off = 0
    for w in sign_widths:
        vals = np.array(sign_values[off:off + w], dtype=np.int64)
        off += w
        if not np.all((vals == 1) | (vals == -1)):
            raise SystemExit("Hadamard sign values must be +/-1")
        signs[w] = vals
    cfg["hadamard_block"] = hb

    print("config:", cfg)

    by_name = {t.name: t for t in r.tensors}
    manifest = {}
    os.makedirs(args.out, exist_ok=True)

    def pad4(f):
        p = (-f.tell()) % 4
        if p:
            f.write(b"\x00" * p)

    def write_entry(f, name, dtype, shape, payload: bytes, **extra):
        entry = {"dtype": dtype, "shape": shape, "buffer": os.path.basename(f.name), "offset": f.tell()}
        f.write(payload)
        entry["length"] = f.tell() - entry["offset"]
        pad4(f)
        entry.update(extra)
        if f.tell() > 0xFFFF_FFFF:
            raise SystemExit(f"{entry['buffer']} exceeds 4 GiB (u32 offsets)")
        manifest[name] = entry

    def write_tensor(f, name, want_type):
        t = by_name.get(name)
        if t is None:
            raise SystemExit(f"missing tensor {name}")
        if t.tensor_type != want_type:
            raise SystemExit(f"{name}: type {t.tensor_type.name}, expected {want_type.name}")
        shape = [int(s) for s in t.shape]
        data = np.ascontiguousarray(t.data)
        if want_type == GT.PTQ1_0:
            n_in, n_out = shape[0], shape[1]
            if n_in % 128:
                raise SystemExit(f"{name}: input dim {n_in} not a multiple of 128")
            nb = n_in // 128
            payload = data.tobytes()
            if len(payload) != n_out * nb * 28:
                raise SystemExit(f"{name}: unexpected PTQ1_0 payload size")
            write_entry(f, name, "PTQ1_0", shape, payload, nb=nb)
        elif want_type == GT.F32:
            write_entry(f, name, "F32", shape, data.astype(np.float32).tobytes())
        elif want_type == GT.BF16:
            payload = data.tobytes()
            if len(payload) != int(np.prod(shape)) * 2:
                raise SystemExit(f"{name}: unexpected BF16 payload size")
            write_entry(f, name, "BF16", shape, payload)
        else:
            raise SystemExit(f"{name}: unsupported type {want_type}")

    P = GT.PTQ1_0

    # ---- weights_attn -------------------------------------------------------
    with open(os.path.join(args.out, "weights_attn.bin"), "wb") as f:
        for il in range(n_layer):
            tags = (("attn_q", "attn_k", "attn_v", "attn_output") if is_attn[il]
                    else ("attn_qkv", "attn_gate", "ssm_out"))
            for tag in tags:
                write_tensor(f, f"blk.{il}.{tag}.weight", P)

    # ---- weights_ffn_gate_up ------------------------------------------------
    with open(os.path.join(args.out, "weights_ffn_gate_up.bin"), "wb") as f:
        for il in range(n_layer):
            for tag in ("ffn_gate", "ffn_up"):
                write_tensor(f, f"blk.{il}.{tag}.weight", P)

    # ---- weights_ffn_down ---------------------------------------------------
    with open(os.path.join(args.out, "weights_ffn_down.bin"), "wb") as f:
        for il in range(n_layer):
            write_tensor(f, f"blk.{il}.ffn_down.weight", P)

    # ---- weights_embed_lmhead -----------------------------------------------
    if "output.weight" not in by_name:
        raise SystemExit("untied output.weight is required")
    with open(os.path.join(args.out, "weights_embed_lmhead.bin"), "wb") as f:
        write_tensor(f, "token_embd.weight", P)
        write_tensor(f, "output.weight", P)

    # ---- weights_small ------------------------------------------------------
    with open(os.path.join(args.out, "weights_small.bin"), "wb") as f:
        write_tensor(f, "output_norm.weight", GT.F32)
        for il in range(n_layer):
            write_tensor(f, f"blk.{il}.attn_norm.weight", GT.F32)
            write_tensor(f, f"blk.{il}.post_attention_norm.weight", GT.F32)
            if is_attn[il]:
                write_tensor(f, f"blk.{il}.attn_q_norm.weight", GT.F32)
                write_tensor(f, f"blk.{il}.attn_k_norm.weight", GT.F32)
            else:
                write_tensor(f, f"blk.{il}.ssm_norm.weight", GT.F32)
                write_tensor(f, f"blk.{il}.ssm_conv1d.weight", GT.F32)
                write_tensor(f, f"blk.{il}.ssm_dt.bias", GT.F32)
                write_tensor(f, f"blk.{il}.ssm_a", GT.F32)
                write_tensor(f, f"blk.{il}.ssm_alpha.weight", GT.BF16)
                write_tensor(f, f"blk.{il}.ssm_beta.weight", GT.BF16)
        # Hadamard signs: bit i of word i/32 set <=> sign[i] == -1.
        for w in sorted(signs):
            bits = (signs[w] == -1).astype(np.uint64)
            words = np.zeros(w // 32, dtype=np.uint32)
            for b in range(32):
                words |= (bits[b::32].astype(np.uint32) << np.uint32(b))
            write_entry(f, f"hadamard.signs.{w}", "BITS", [w], words.astype("<u4").tobytes())

    # ---- vocab --------------------------------------------------------------
    tok_field = r.fields["tokenizer.ggml.tokens"]
    vocab_bytes = bytearray()
    offsets = [0]
    for i in tok_field.data:
        vocab_bytes.extend(bytes(tok_field.parts[i]))
        offsets.append(len(vocab_bytes))
    n_vocab = cfg["n_vocab"]
    with open(os.path.join(args.out, "vocab.bin"), "wb") as f:
        f.write(bytes(vocab_bytes))
    with open(os.path.join(args.out, "vocab_offsets.bin"), "wb") as f:
        f.write(struct.pack(f"<{n_vocab + 1}I", *offsets))

    with open(os.path.join(args.out, "added_tokens.txt"), "w", newline="\n") as f:
        for i, ty in enumerate(token_types):
            if ty in (TOKEN_TYPE_CONTROL, TOKEN_TYPE_USER_DEFINED):
                f.write(f"{i}\n")

    merges_field = r.fields["tokenizer.ggml.merges"]
    with open(os.path.join(args.out, "merges.txt"), "w", encoding="utf-8", newline="\n") as f:
        for idx in merges_field.data:
            line = bytes(merges_field.parts[idx]).decode("utf-8")
            f.write(line)
            if not line.endswith("\n"):
                f.write("\n")

    # ---- config.ini ---------------------------------------------------------
    def fmt(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, list):
            return ",".join(str(x) for x in v)
        return str(v)

    with open(os.path.join(args.out, "config.ini"), "w", newline="\n") as f:
        for k, v in cfg.items():
            f.write(f"{k} = {fmt(v)}\n")
        for name, entry in manifest.items():
            f.write(f"\n[{name}]\n")
            for k, v in entry.items():
                f.write(f"{k} = {fmt(v)}\n")

    sizes = {p: os.path.getsize(os.path.join(args.out, p))
             for p in sorted(os.listdir(args.out)) if p.endswith(".bin")}
    print("output bytes:", sizes)


if __name__ == "__main__":
    main()
