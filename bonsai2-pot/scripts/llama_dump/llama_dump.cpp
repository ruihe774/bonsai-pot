// Reference dumper for bonsai2-pot: runs the PrismML llama.cpp fork on a fixed
// token sequence and writes selected intermediate tensors + logits to disk as
// raw little-endian f32, with a JSON-lines index. Used by scripts/compare.py.
//
// Usage:
//   llama_dump -m model.gguf -t tokens.u32 -o outdir [-f REGEX] [--logits-all]
//              [--gen N] [-ngl N] [-c N_CTX] [--threads N]
//
//   -t      token ids, raw u32 LE (e.g. the output of scripts/bpe.py)
//   -f      ECMAScript regex matched against the full ggml tensor name
//           (default: "l_out-[0-9]+"); dumped during the prompt decode only
//   --logits-all   request logits for every prompt position (default: last only)
//   --gen N        afterwards, greedily generate N tokens; ids -> outdir/gen.txt
//
// Output: outdir/index.jsonl with one {"name","file","ne"} object per dumped
// tensor; "logits" rows are [n_vocab, n_rows] in prompt order.

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <vector>

namespace {

struct dump_state {
    std::regex  re;
    std::string out_dir;
    FILE *      index   = nullptr;
    bool        enabled = true;
    std::map<std::string, int> seen;
};

float elem_to_f32(const uint8_t * p, ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: {
            float v;
            memcpy(&v, p, sizeof(v));
            return v;
        }
        case GGML_TYPE_F16: {
            ggml_fp16_t v;
            memcpy(&v, p, sizeof(v));
            return ggml_fp16_to_fp32(v);
        }
        case GGML_TYPE_BF16: {
            ggml_bf16_t v;
            memcpy(&v, p, sizeof(v));
            return ggml_bf16_to_fp32(v);
        }
        default:
            return 0.0f;
    }
}

std::string sanitize(const std::string & s) {
    std::string r = s;
    for (char & c : r) {
        if (!(isalnum((unsigned char) c) || c == '-' || c == '_' || c == '.')) {
            c = '_';
        }
    }
    return r;
}

void write_entry(dump_state & st, const std::string & name, const float * data, size_t n, const int64_t ne[4]) {
    const int k = st.seen[name]++;
    const std::string file = sanitize(name) + (k ? "." + std::to_string(k) : "") + ".f32";
    const std::string path = st.out_dir + "/" + file;
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        exit(1);
    }
    fwrite(data, sizeof(float), n, f);
    fclose(f);
    fprintf(st.index, "{\"name\":\"%s\",\"file\":\"%s\",\"ne\":[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]}\n",
            name.c_str(), file.c_str(), ne[0], ne[1], ne[2], ne[3]);
    fflush(st.index);
}

bool eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<dump_state *>(user_data);
    const bool match = st->enabled && std::regex_match(t->name, st->re);
    if (ask) {
        return match;
    }
    if (!match) {
        return true;
    }
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) {
        fprintf(stderr, "skipping %s: unsupported type %s\n", t->name, ggml_type_name(t->type));
        return true;
    }
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> raw(nbytes);
    if (ggml_backend_buffer_is_host(t->buffer)) {
        memcpy(raw.data(), t->data, nbytes);
    } else {
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    }
    std::vector<float> out;
    out.reserve(ggml_nelements(t));
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    const size_t off = i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1] + i0 * t->nb[0];
                    out.push_back(elem_to_f32(raw.data() + off, t->type));
                }
            }
        }
    }
    write_entry(*st, t->name, out.data(), out.size(), t->ne);
    return true;
}

[[noreturn]] void usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s -m model.gguf -t tokens.u32 -o outdir [-f REGEX] [--logits-all] [--gen N] "
            "[-ngl N] [-c N_CTX] [--threads N]\n",
            argv0);
    exit(2);
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model_path, tokens_path, out_dir, filter = "l_out-[0-9]+";
    bool        logits_all = false;
    int         n_gen = 0, ngl = 99, n_ctx = 0, n_threads = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage(argv[0]);
            }
            return argv[++i];
        };
        if (a == "-m") {
            model_path = next();
        } else if (a == "-t") {
            tokens_path = next();
        } else if (a == "-o") {
            out_dir = next();
        } else if (a == "-f") {
            filter = next();
        } else if (a == "--logits-all") {
            logits_all = true;
        } else if (a == "--gen") {
            n_gen = std::stoi(next());
        } else if (a == "-ngl") {
            ngl = std::stoi(next());
        } else if (a == "-c") {
            n_ctx = std::stoi(next());
        } else if (a == "--threads") {
            n_threads = std::stoi(next());
        } else {
            usage(argv[0]);
        }
    }
    if (model_path.empty() || tokens_path.empty() || out_dir.empty()) {
        usage(argv[0]);
    }

    std::vector<llama_token> tokens;
    {
        std::ifstream f(tokens_path, std::ios::binary);
        if (!f) {
            fprintf(stderr, "cannot read %s\n", tokens_path.c_str());
            return 1;
        }
        uint32_t v;
        while (f.read(reinterpret_cast<char *>(&v), sizeof(v))) {
            tokens.push_back((llama_token) v);
        }
    }
    if (tokens.empty()) {
        fprintf(stderr, "no tokens\n");
        return 1;
    }

    dump_state st;
    st.re      = std::regex(filter, std::regex::ECMAScript | std::regex::optimize);
    st.out_dir = out_dir;
    st.index   = fopen((out_dir + "/index.jsonl").c_str(), "w");
    if (!st.index) {
        fprintf(stderr, "cannot create %s/index.jsonl (does the directory exist?)\n", out_dir.c_str());
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = ngl;
    llama_model * model        = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    const int n_prompt = (int) tokens.size();
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = n_ctx > 0 ? n_ctx : n_prompt + n_gen + 16;
    cparams.n_batch           = n_prompt;
    cparams.n_ubatch          = n_prompt;
    cparams.cb_eval           = eval_cb;
    cparams.cb_eval_user_data = &st;
    if (n_threads > 0) {
        cparams.n_threads       = n_threads;
        cparams.n_threads_batch = n_threads;
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = logits_all || i == n_prompt - 1;
    }
    batch.n_tokens = n_prompt;
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        return 1;
    }

    {
        std::vector<float> rows;
        int                n_rows = 0;
        for (int i = 0; i < n_prompt; ++i) {
            if (!batch.logits[i]) {
                continue;
            }
            const float * l = llama_get_logits_ith(ctx, i);
            rows.insert(rows.end(), l, l + n_vocab);
            ++n_rows;
        }
        const int64_t ne[4] = { n_vocab, n_rows, 1, 1 };
        write_entry(st, "logits", rows.data(), rows.size(), ne);
    }
    llama_batch_free(batch);

    st.enabled = false;
    if (n_gen > 0) {
        FILE *      gen  = fopen((out_dir + "/gen.txt").c_str(), "w");
        llama_token last = 0;
        {
            const float * l = llama_get_logits_ith(ctx, -1);
            last            = (llama_token) (std::max_element(l, l + n_vocab) - l);
        }
        for (int s = 0; s < n_gen; ++s) {
            fprintf(gen, "%d\n", last);
            if (llama_vocab_is_eog(vocab, last)) {
                break;
            }
            llama_batch b = llama_batch_get_one(&last, 1);
            if (llama_decode(ctx, b) != 0) {
                fprintf(stderr, "llama_decode (gen) failed\n");
                return 1;
            }
            const float * l = llama_get_logits_ith(ctx, -1);
            last            = (llama_token) (std::max_element(l, l + n_vocab) - l);
        }
        fclose(gen);
    }

    fclose(st.index);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
