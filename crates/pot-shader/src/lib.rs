use proc_macro::TokenStream;
use proc_macro2::Literal;
use quote::{format_ident, quote};
use syn::{Expr, LitStr, Token, parse_macro_input};

struct MacroInput {
    shader_name: LitStr,
    _comma: Token![,],
    sg_expr: Expr,
}

impl syn::parse::Parse for MacroInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(Self {
            shader_name: input.parse()?,
            _comma: input.parse()?,
            sg_expr: input.parse()?,
        })
    }
}

fn compile_spv(wgsl_src: &str, shader_file: &str, sg: u32) -> Vec<u32> {
    let module = naga::front::wgsl::Frontend::new()
        .parse(wgsl_src)
        .unwrap_or_else(|e| {
            panic!(
                "pot_shader: naga WGSL parse error in {} (sg={}):\n{}",
                shader_file,
                sg,
                e.emit_to_string(wgsl_src),
            )
        });

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| {
        panic!(
            "pot_shader: naga validation error in {} (sg={}):\n{:?}",
            shader_file, sg, e,
        )
    });

    let opts = naga::back::spv::Options {
        lang_version: (1, 3),
        flags: naga::back::spv::WriterFlags::empty(),
        zero_initialize_workgroup_memory: naga::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
        ..Default::default()
    };

    naga::back::spv::write_vec(&module, &info, &opts, None).unwrap_or_else(|e| {
        panic!(
            "pot_shader: naga SPIR-V write error in {} (sg={}):\n{:?}",
            shader_file, sg, e,
        )
    })
}

/// Compiles `shader_name` (relative to `src/shaders/`) to SPIR-V at build
/// time for each enabled `sg-N` Cargo feature, embeds the words in the binary,
/// and expands to a `match` on `subgroup_min_size_expr` that selects the right
/// variant at runtime.
///
/// Usage: `pot_shader!("foo.wgsl", subgroup_min_size)`
#[proc_macro]
pub fn pot_shader(input: TokenStream) -> TokenStream {
    let MacroInput {
        shader_name,
        sg_expr,
        ..
    } = parse_macro_input!(input as MacroInput);

    let shader_file = shader_name.value();

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is not set");
    let shader_path = std::path::PathBuf::from(&manifest_dir)
        .join("src")
        .join("shaders")
        .join(&shader_file);
    let wgsl_src = std::fs::read_to_string(&shader_path)
        .unwrap_or_else(|e| panic!("pot_shader: cannot read {}: {e}", shader_path.display()));

    let shader_path_str = shader_path
        .to_str()
        .expect("shader path is not valid UTF-8")
        .to_owned();

    // One (sg_size, feature_enabled) pair per supported SG size.
    let sg_variants: &[(u32, bool)] = &[
        (8, cfg!(feature = "sg-8")),
        (16, cfg!(feature = "sg-16")),
        (32, cfg!(feature = "sg-32")),
        (64, cfg!(feature = "sg-64")),
    ];

    let mut statics = Vec::new();
    let mut arms = Vec::new();

    for &(sg, enabled) in sg_variants {
        if !enabled {
            continue;
        }
        let src = wgsl_src
            .replace("{{SUBGROUP_MIN_SIZE}}", &sg.to_string())
            .replace("{{N_EMBD_V4}}", "1024")
            .replace("{{MAX_CHUNKS}}", "2048");

        let words = compile_spv(&src, &shader_file, sg);

        let feat = format!("sg-{sg}");
        let stat = format_ident!("SPV_{}", sg);
        statics.push(quote! {
            #[cfg(feature = #feat)]
            static #stat: &[u32] = &[#(#words),*];
        });

        let sg_lit = Literal::u32_unsuffixed(sg);
        arms.push(quote! {
            #[cfg(feature = #feat)]
            #sg_lit => #stat,
        });
    }

    quote! {
        {
            #(#statics)*
            const _: &[u8] = ::core::include_bytes!(#shader_path_str);
            match #sg_expr {
                #(#arms)*
                x => ::core::panic!(
                    "subgroup_min_size {} has no compiled shader variant; rebuild with sg-{{8,16,32,64}} feature",
                    x
                ),
            }
        }
    }
    .into()
}
