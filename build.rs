#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "build script"
)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

// Every compute kernel of the engine; each compiles to `OUT_DIR/{name}.spv`.
const SHADERS: &[&str] = &[
    "embed.comp",
    "norm_rot_q8.comp",
    "matvec_ptq1_0.comp",
    "matmul_ptq1_0.comp",
    "matmul_ptq1_0_cm.comp",
    "matmul_ptq1_0_cmi8.comp",
    "gdn_gates.comp",
    "gdn_conv.comp",
    "gdn_recurrence.comp",
    "gdn_out_rot_q8.comp",
    "attn_qk_norm_rope_kv.comp",
    "attn_split.comp",
    "attn_prefill.comp",
    "attn_out_rot_q8.comp",
    "silu_mul_rot_q8.comp",
    "topk_partial.comp",
    "topk_merge.comp",
];

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let debug_shaders = env::var_os("PROFILE").unwrap().as_encoded_bytes() == b"debug";
    build_vulkan(&manifest_dir, &out_dir, debug_shaders);
}

// ---------------------------------------------------------------------------
// Vulkan: glslangValidator → spirv-opt -O → `OUT_DIR/{name}.spv`.
//
// Every kernel runs as a precompiled SPIR-V shader module, fed to wgpu via
// `create_shader_module_passthrough`; naga is bypassed entirely.
//
// We use `glslangValidator` (>= 14.0 / SPV 1.5+) rather than `glslc`
// (shaderc 2023.x ships with glslang 11, which is older than
// `GL_EXT_integer_dot_product`). Optimisation runs as a follow-up
// `spirv-opt -O` pass since glslangValidator has no `-O` switch.
fn build_vulkan(manifest_dir: &Path, out_dir: &Path, debug_shaders: bool) {
    let lib_dir = manifest_dir.join("src/shaders");
    rerun_if_lib_changed(&lib_dir);
    for name in SHADERS {
        let src_path = manifest_dir.join("src/shaders").join(name);
        let raw_path = out_dir.join(format!("{name}.raw.spv"));
        let opt_path = out_dir.join(format!("{name}.spv"));
        println!("cargo:rerun-if-changed={}", src_path.display());

        run_glslang(&src_path, &raw_path, &lib_dir, debug_shaders);
        run_spirv_opt(manifest_dir, &raw_path, &opt_path);
    }
}

fn rerun_if_lib_changed(shaders_dir: &Path) {
    let lib_dir = shaders_dir.join("lib");
    let entries = fs::read_dir(&lib_dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", lib_dir.display()));
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("glsl") {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn run_glslang(src_path: &Path, raw_path: &Path, include_dir: &Path, debug: bool) {
    let mut glslang = Command::new("glslangValidator");
    glslang
        .arg("--target-env")
        .arg("vulkan1.3")
        .arg("-S")
        .arg("comp")
        .arg("-V")
        .arg(format!("-I{}", include_dir.display()))
        .arg("-P#extension GL_ARB_shading_language_include:require")
        .arg("-P#include\"lib/preamble.glsl\"");
    if debug {
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
    let status = glslang
        .arg(src_path)
        .arg("-o")
        .arg(raw_path)
        .status()
        .unwrap_or_else(|e| {
            panic!("failed to run glslangValidator: {e} (is glslang-tools >= 14 on PATH?)")
        });
    assert!(
        status.success(),
        "glslangValidator failed for {}",
        src_path.display(),
    );
}

fn run_spirv_opt(manifest_dir: &Path, raw_path: &Path, opt_path: &Path) {
    let status = Command::new("spirv-opt")
        .arg(format!(
            "-Oconfig={}/spirv-opt.conf",
            manifest_dir.display()
        ))
        .arg(raw_path)
        .arg("-o")
        .arg(opt_path)
        .status()
        .unwrap_or_else(|e| panic!("failed to run spirv-opt: {e}"));
    assert!(
        status.success(),
        "spirv-opt failed for {}",
        raw_path.display(),
    );
}
