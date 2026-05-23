#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "build script"
)]

#[path = "src/minify.rs"]
mod minify;
#[path = "src/msl_strip.rs"]
mod msl_strip;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

// Ordered list of every shader in the engine. The Apple build tree-shakes
// this through spirv-cross to produce MSL; the Vulkan tree just optimises
// the SPIR-V. `matmul_q1_0_q8_0.comp` is the lone Apple exception: its MSL
// counterpart is hand-rolled around `simdgroup_matrix<half,8,8>`, which has
// no GLSL surface, so on Apple we copy the hand-port verbatim instead of
// translating SPIR-V → MSL.
const SHADERS: &[&str] = &[
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

const HAND_PORTED_MSL: &[&str] = &["matmul_q1_0_q8_0.comp"];

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let debug_shaders = env::var_os("PROFILE").unwrap().as_encoded_bytes() == b"debug";
    let is_apple = env::var_os("CARGO_CFG_TARGET_VENDOR")
        .unwrap()
        .as_encoded_bytes()
        == b"apple";

    if is_apple {
        build_apple(&manifest_dir, &out_dir, debug_shaders);
    } else {
        build_vulkan(&manifest_dir, &out_dir, debug_shaders);
    }
}

// ---------------------------------------------------------------------------
// Vulkan: glslangValidator → spirv-opt -O → `OUT_DIR/{name}.spv`.
//
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
fn build_vulkan(manifest_dir: &Path, out_dir: &Path, debug_shaders: bool) {
    let lib_dir = manifest_dir.join("src/shaders");
    rerun_if_lib_changed(&lib_dir);
    for name in SHADERS {
        let src_path = manifest_dir.join("src/shaders").join(name);
        let raw_path = out_dir.join(format!("{name}.raw.spv"));
        let opt_path = out_dir.join(format!("{name}.spv"));
        println!("cargo:rerun-if-changed={}", src_path.display());

        run_glslang(&src_path, &raw_path, &[], &lib_dir, debug_shaders);
        run_spirv_opt(manifest_dir, &raw_path, &opt_path);
    }
}

// ---------------------------------------------------------------------------
// Apple: glslangValidator -DMETAL_BACKEND=1 → spirv-opt -O → spirv-cross
//        → `OUT_DIR/{name}.msl`. Hand-ported shaders bypass the pipeline
//        and are copied straight to OUT_DIR so the load-time include path
//        is uniform.
//
// `-DMETAL_BACKEND=1` lets the GLSL source select Apple-specific kernel
// variants (e.g. f16 ±-accumulate inner loops in the matvec family vs
// `dotPacked4x8EXT` on Vulkan). `spirv-opt -O` is required so the
// `[[unroll]]` annotations in the source actually unroll before
// spirv-cross sees the SPIR-V — otherwise per-iteration array indices
// stay dynamic and Apple's driver compiler spills register-resident
// state to local memory. spirv-cross emits `kernel void main0(...)` by
// default; we rename to `cs_main` so the pipeline `entry_point` stays
// the same as for the hand-ported MSL files.
fn build_apple(manifest_dir: &Path, out_dir: &Path, debug_shaders: bool) {
    let lib_dir = manifest_dir.join("src/shaders");
    rerun_if_lib_changed(&lib_dir);
    let define_metal = [("METAL_BACKEND", "1")];
    for name in SHADERS {
        let src_path = manifest_dir.join("src/shaders").join(name);
        let msl_path = out_dir.join(format!("{name}.msl"));
        println!("cargo:rerun-if-changed={}", src_path.display());

        if HAND_PORTED_MSL.contains(name) {
            // Hand-ported MSL bypass: copy the .metal sibling into OUT_DIR
            // under the same `.msl` name the load-time include macro expects.
            let metal_path = src_path.with_extension("metal");
            println!("cargo:rerun-if-changed={}", metal_path.display());
            fs::copy(&metal_path, &msl_path).unwrap_or_else(|e| {
                panic!(
                    "failed to copy hand-ported {} → {}: {e}",
                    metal_path.display(),
                    msl_path.display()
                )
            });
            minify_msl(&msl_path);
            continue;
        }

        let raw_path = out_dir.join(format!("{name}.raw.spv"));
        let opt_path = out_dir.join(format!("{name}.opt.spv"));
        run_glslang(&src_path, &raw_path, &define_metal, &lib_dir, debug_shaders);
        run_spirv_opt(manifest_dir, &raw_path, &opt_path);
        run_spirv_cross_msl(&opt_path, &msl_path);
        strip_spec_constants_msl(&msl_path);
        minify_msl(&msl_path);
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

fn run_glslang(
    src_path: &Path,
    raw_path: &Path,
    defines: &[(&str, &str)],
    include_dir: &Path,
    debug: bool,
) {
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
    for (k, v) in defines {
        glslang.arg(format!("-D{k}={v}"));
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

fn strip_spec_constants_msl(msl_path: &Path) {
    let src = fs::read_to_string(msl_path).expect("read MSL for spec-constant strip");
    let out = msl_strip::strip(&src).expect("strip spec constants");
    let tmp_path = msl_path.with_extension("msl.tmp");
    fs::write(&tmp_path, out).expect("write stripped MSL to tmp");
    fs::rename(&tmp_path, msl_path).expect("rename stripped MSL into place");
}

fn minify_msl(msl_path: &Path) {
    let src = fs::read_to_string(msl_path).expect("read MSL for minification");
    let out = minify::minify(&src).expect("minify MSL");
    let tmp_path = msl_path.with_extension("msl.tmp");
    fs::write(&tmp_path, out).expect("write minified MSL to tmp");
    fs::rename(&tmp_path, msl_path).expect("rename minified MSL into place");
}

fn run_spirv_cross_msl(spv_path: &Path, msl_path: &Path) {
    // `--msl-version 30000` matches MSL 3.0; subgroup ops require >= 2.1
    // and several kernels lean on `simdgroupShuffleXor` etc. Renaming
    // entry points keeps the pipeline `entry_point` field constant
    // across hand-ported and translated MSL.
    let status = Command::new("spirv-cross")
        .arg("--msl")
        .arg("--msl-version")
        .arg("30000")
        .arg("--msl-fixed-subgroup-size")
        .arg("32")
        .arg("--relax-nan-checks")
        // `--msl-decoration-binding` makes spirv-cross emit `[[buffer(N)]]`
        // where N is the SPIR-V `Binding` decoration. Without it, slots are
        // assigned in first-use order, which doesn't agree with wgpu's
        // Metal HAL slot allocation (push at 0, SSBOs at 1+ in binding
        // order). The push constant still lands at `[[buffer(0)]]` and
        // collides with binding 0; `msl_shift_ssbo_buffer_indices` does
        // the +1 shift at load time.
        .arg("--msl-decoration-binding")
        .arg("--rename-entry-point")
        .arg("main")
        .arg("cs_main")
        .arg("comp")
        .arg(spv_path)
        .arg("--output")
        .arg(msl_path)
        .status()
        .unwrap_or_else(|e| panic!("failed to run spirv-cross: {e} (is spirv-cross on PATH?)"));
    assert!(
        status.success(),
        "spirv-cross failed for {}",
        spv_path.display(),
    );
}
