#![allow(clippy::panic, clippy::expect_used, reason = "build script")]

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let debug_shaders = env::var("PROFILE").as_deref() == Ok("debug");

    // Every kernel runs as a precompiled SPIR-V shader module, fed to wgpu via
    // `create_shader_module_passthrough`. naga is bypassed entirely on the
    // hot path. The three values that the WGSL versions text-substituted
    // (SUBGROUP_MIN_SIZE, MAX_CHUNKS, N_EMBD_V4) live as Vulkan specialization
    // constants (SpecId 0, 1, 2 respectively); their default operands are
    // patched at runtime by `model::spirv_set_spec_const_u32` before each
    // module is created. This keeps a single SPIR-V binary per kernel and
    // moves the per-adapter / per-model branching out of the compile pipeline.
    //
    // We use `glslangValidator` (>= 14.0 / SPV 1.5+) rather than `glslc`
    // (shaderc 2023.x ships with glslang 11, which is older than
    // `GL_EXT_integer_dot_product`). Optimisation runs as a follow-up
    // `spirv-opt -O` pass since glslangValidator has no `-O` switch.
    let shaders = [
        "embed.comp",
        "rms_norm.comp",
        "rms_norm_q8_0.comp",
        "silu_mul_q8_0.comp",
        "q_norm_rope_fused.comp",
        "kv_writeback_fused.comp",
        "attention_split.comp",
        "attention_merge.comp",
        "attention_prefill_tiled.comp",
        "matvec_q1_0.comp",
        "matvec_q1_0_silu.comp",
        "matvec_q1_0_fused_normed.comp",
        "matmul_q1_0_q8_0.comp",
        "topk_partial.comp",
        "topk_merge.comp",
    ];

    for name in shaders {
        let src_path = manifest_dir.join("src/shaders").join(name);
        let raw_path = out_dir.join(format!("{name}.raw.spv"));
        let opt_path = out_dir.join(format!("{name}.spv"));
        println!("cargo:rerun-if-changed={}", src_path.display());

        let mut glslang = Command::new("glslangValidator");
        glslang
            .arg("--target-env")
            .arg("vulkan1.3")
            .arg("-S")
            .arg("comp")
            .arg("-V");
        if debug_shaders {
            // `-g` embeds OpSource (with the full GLSL source string),
            // OpString filenames, and OpLine debug info, so RGP / Nsight /
            // RenderDoc can show the original source against profiler
            // samples. The follow-up `spirv-opt -O` preserves these
            // (it does not implicitly `--strip-debug`). The richer
            // `-gVS` (NonSemantic.Shader.DebugInfo) is *not* used because
            // its `DebugTypeArray` instructions reject our spec-constant
            // array sizes inside spirv-opt's legalisation passes.
            glslang.arg("-g");
        }
        let glslang_status = glslang
            .arg(&src_path)
            .arg("-o")
            .arg(&raw_path)
            .status()
            .unwrap_or_else(|e| {
                panic!("failed to run glslangValidator: {e} (is glslang-tools >= 14 on PATH?)")
            });
        assert!(
            glslang_status.success(),
            "glslangValidator failed for {}",
            src_path.display(),
        );

        let opt_status = Command::new("spirv-opt")
            .arg("-O")
            .arg(&raw_path)
            .arg("-o")
            .arg(&opt_path)
            .status()
            .unwrap_or_else(|e| panic!("failed to run spirv-opt: {e}"));
        assert!(
            opt_status.success(),
            "spirv-opt failed for {}",
            raw_path.display(),
        );
    }
}
