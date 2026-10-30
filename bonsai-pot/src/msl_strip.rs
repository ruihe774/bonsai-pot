//! Strip spirv-cross-emitted spec-constant scaffolding from MSL so that
//! load-time can patch values by simply prepending a `#define`.
//!
//! spirv-cross emits a Vulkan `SpecConstant` in one of two forms:
//!
//! **Form A** — `#define`-style default, guarded by `#ifndef`:
//!
//! ```text
//! #ifndef SPIRV_CROSS_CONSTANT_ID_<N>
//! #define SPIRV_CROSS_CONSTANT_ID_<N> <default>u
//! #endif
//! constant uint NAME = SPIRV_CROSS_CONSTANT_ID_<N>;
//! ```
//!
//! **Form B** — `[[function_constant(N)]]` declaration plus a ternary that
//! falls back to a default if no value is bound:
//!
//! ```text
//! constant uint NAME_tmp [[function_constant(<N>)]];
//! constant uint NAME = is_function_constant_defined(NAME_tmp) ? NAME_tmp : <default>u;
//! ```
//!
//! Neither form survives wgpu's MSL passthrough: it has no
//! `MTLFunctionConstantValues` surface, and `is_function_constant_defined`
//! is undefined without one. We pre-process the MSL at build time into a
//! uniform shape that load-time can patch by simply prepending
//! `#define SPIRV_CROSS_CONSTANT_ID_<N> <value>u`:
//!
//! * Form A: strip the `#ifndef … #endif` guard block. The
//!   `constant uint NAME = SPIRV_CROSS_CONSTANT_ID_<N>;` declaration is
//!   left in place so the prepended define resolves it.
//! * Form B: rewrite the two declarations as a single
//!   `constant uint NAME = SPIRV_CROSS_CONSTANT_ID_<N>;` line, matching
//!   the Form A shape. `NAME` is extracted from the `_tmp` declarator and
//!   `<N>` from the `function_constant(N)` attribute, so no slot-to-name
//!   table is needed.
//!
//! Identification is done by walking the tree-sitter-cpp AST
//! (`preproc_ifdef` / `declaration` / `attribute` / `init_declarator` /
//! `attributed_declarator` nodes); the only raw-text reads are on
//! individual identifier and number-literal leaves.

use tree_sitter::{Node, Parser};

pub fn strip(src: &str) -> Result<String, String> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_cpp::LANGUAGE.into())
        .map_err(|e| format!("loading C++ grammar: {e}"))?;

    let tree = parser
        .parse(src, None)
        .ok_or_else(|| "tree-sitter failed to parse MSL".to_string())?;

    let src_bytes = src.as_bytes();
    let mut ops: Vec<(usize, usize, String)> = Vec::new();

    let root = tree.root_node();
    let mut cursor = root.walk();
    let children: Vec<Node> = root.children(&mut cursor).collect();

    let mut i = 0;
    while i < children.len() {
        let child = children[i];

        // Form A guard: `#ifndef SPIRV_CROSS_CONSTANT_ID_<digits>` block.
        if child.kind() == "preproc_ifdef" && guard_targets_spirv_cross_id(child, src_bytes) {
            let end = consume_trailing_newline(src_bytes, child.end_byte());
            ops.push((child.start_byte(), end, String::new()));
            i += 1;
            continue;
        }

        // Form B: declaration with a `function_constant(N)` attribute,
        // paired with the next-sibling ternary declaration that initialises
        // the same identifier from `is_function_constant_defined(...)`.
        if let Some(slot) = function_constant_slot(child, src_bytes)
            && let Some(name_tmp) = attributed_declarator_name(child, src_bytes)
            && let Some(name) = name_tmp.strip_suffix("_tmp")
            && let Some(next) = children.get(i + 1).copied()
            && init_declarator_name(next, src_bytes).as_deref() == Some(name)
            && contains_is_function_constant_defined_call(next, src_bytes)
        {
            let tmp_end = consume_trailing_newline(src_bytes, child.end_byte());
            ops.push((child.start_byte(), tmp_end, String::new()));
            let replacement = format!("constant uint {name} = SPIRV_CROSS_CONSTANT_ID_{slot};");
            ops.push((next.start_byte(), next.end_byte(), replacement));
            i += 2;
            continue;
        }

        i += 1;
    }

    Ok(apply_ops(src, ops))
}

fn apply_ops(src: &str, mut ops: Vec<(usize, usize, String)>) -> String {
    ops.sort_by_key(|&(s, _, _)| s);
    let mut out = String::with_capacity(src.len());
    let mut cursor = 0;
    for (s, e, rep) in ops {
        out.push_str(&src[cursor..s]);
        out.push_str(&rep);
        cursor = e;
    }
    out.push_str(&src[cursor..]);
    out
}

fn consume_trailing_newline(src: &[u8], end: usize) -> usize {
    if src.get(end) == Some(&b'\n') {
        end + 1
    } else {
        end
    }
}

/// `preproc_ifdef`'s first `identifier` child is the macro being guarded.
/// Match `SPIRV_CROSS_CONSTANT_ID_` followed by one or more digits.
fn guard_targets_spirv_cross_id(node: Node, src: &[u8]) -> bool {
    let mut c = node.walk();
    for child in node.children(&mut c) {
        if child.kind() == "identifier" {
            let Ok(text) = child.utf8_text(src) else {
                return false;
            };
            let Some(rest) = text.strip_prefix("SPIRV_CROSS_CONSTANT_ID_") else {
                return false;
            };
            return !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit());
        }
    }
    false
}

/// If `node` contains an `attribute` whose first identifier child is
/// `function_constant`, return the integer slot from its `argument_list`'s
/// `number_literal`.
fn function_constant_slot(node: Node, src: &[u8]) -> Option<u32> {
    fn walk(n: Node, src: &[u8]) -> Option<u32> {
        if n.kind() == "attribute" {
            let mut c = n.walk();
            let mut children = n.children(&mut c);
            let first = children.next()?;
            if first.kind() == "identifier" && first.utf8_text(src).ok()? == "function_constant" {
                for sibling in children {
                    if sibling.kind() == "argument_list" {
                        let mut ac = sibling.walk();
                        for arg in sibling.children(&mut ac) {
                            if arg.kind() == "number_literal" {
                                return arg.utf8_text(src).ok()?.trim_end_matches('u').parse().ok();
                            }
                        }
                    }
                }
            }
        }
        let mut c = n.walk();
        for child in n.children(&mut c) {
            if let Some(slot) = walk(child, src) {
                return Some(slot);
            }
        }
        None
    }
    walk(node, src)
}

/// The first `identifier` child of any `attributed_declarator` in the
/// subtree — the variable name in Form B's `_tmp` declaration.
fn attributed_declarator_name(node: Node, src: &[u8]) -> Option<String> {
    fn walk(n: Node, src: &[u8]) -> Option<String> {
        if n.kind() == "attributed_declarator" {
            let mut c = n.walk();
            for child in n.children(&mut c) {
                if child.kind() == "identifier" {
                    return child.utf8_text(src).ok().map(String::from);
                }
            }
        }
        let mut c = n.walk();
        for child in n.children(&mut c) {
            if let Some(r) = walk(child, src) {
                return Some(r);
            }
        }
        None
    }
    walk(node, src)
}

/// The first `identifier` child of any `init_declarator` in the subtree —
/// the variable name on the LHS of `NAME = …`.
fn init_declarator_name(node: Node, src: &[u8]) -> Option<String> {
    fn walk(n: Node, src: &[u8]) -> Option<String> {
        if n.kind() == "init_declarator" {
            let mut c = n.walk();
            for child in n.children(&mut c) {
                if child.kind() == "identifier" {
                    return child.utf8_text(src).ok().map(String::from);
                }
            }
        }
        let mut c = n.walk();
        for child in n.children(&mut c) {
            if let Some(r) = walk(child, src) {
                return Some(r);
            }
        }
        None
    }
    walk(node, src)
}

/// True if any `call_expression` in the subtree invokes a function whose
/// identifier is `is_function_constant_defined`.
fn contains_is_function_constant_defined_call(node: Node, src: &[u8]) -> bool {
    if node.kind() == "call_expression" {
        let mut c = node.walk();
        if let Some(first) = node.children(&mut c).next()
            && first.kind() == "identifier"
            && first.utf8_text(src).ok() == Some("is_function_constant_defined")
        {
            return true;
        }
    }
    let mut c = node.walk();
    for child in node.children(&mut c) {
        if contains_is_function_constant_defined_call(child, src) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_form_a_guard_block() {
        let src = "#ifndef SPIRV_CROSS_CONSTANT_ID_1\n\
                   #define SPIRV_CROSS_CONSTANT_ID_1 128u\n\
                   #endif\n\
                   constant uint MAX_CHUNKS = SPIRV_CROSS_CONSTANT_ID_1;\n\
                   kernel void cs_main() {}\n";
        let out = strip(src).unwrap();
        assert!(!out.contains("#ifndef SPIRV_CROSS_CONSTANT_ID_1"));
        assert!(!out.contains("#define SPIRV_CROSS_CONSTANT_ID_1"));
        assert!(!out.contains("#endif"));
        // The dependent declaration stays in place; the prepended #define
        // at load time will resolve `SPIRV_CROSS_CONSTANT_ID_1`.
        assert!(out.contains("constant uint MAX_CHUNKS = SPIRV_CROSS_CONSTANT_ID_1;"));
        assert!(out.contains("kernel void cs_main()"));
    }

    #[test]
    fn rewrites_form_b_pair_into_form_a_shape() {
        let src = "// preamble\n\
                   constant uint SUBGROUP_SIZE_tmp [[function_constant(0)]];\n\
                   constant uint SUBGROUP_SIZE = is_function_constant_defined(SUBGROUP_SIZE_tmp) ? SUBGROUP_SIZE_tmp : 32u;\n\
                   constant bool _193 = (SUBGROUP_SIZE >= 32u);\n\
                   kernel void cs_main() {}\n";
        let out = strip(src).unwrap();
        assert!(!out.contains("function_constant("));
        assert!(!out.contains("is_function_constant_defined"));
        assert!(!out.contains("SUBGROUP_SIZE_tmp"));
        assert!(out.contains("constant uint SUBGROUP_SIZE = SPIRV_CROSS_CONSTANT_ID_0;"));
        // Surrounding context preserved.
        assert!(out.contains("// preamble"));
        assert!(out.contains("constant bool _193 = (SUBGROUP_SIZE >= 32u);"));
    }

    #[test]
    fn handles_multiple_slots_in_one_source() {
        let src = "#ifndef SPIRV_CROSS_CONSTANT_ID_2\n\
                   #define SPIRV_CROSS_CONSTANT_ID_2 640u\n\
                   #endif\n\
                   constant uint K_V4 = SPIRV_CROSS_CONSTANT_ID_2;\n\
                   #ifndef SPIRV_CROSS_CONSTANT_ID_0\n\
                   #define SPIRV_CROSS_CONSTANT_ID_0 32u\n\
                   #endif\n\
                   constant uint SUBGROUP_SIZE = SPIRV_CROSS_CONSTANT_ID_0;\n\
                   kernel void cs_main() {}\n";
        let out = strip(src).unwrap();
        assert!(!out.contains("#ifndef"));
        assert!(!out.contains("#endif"));
        assert!(out.contains("constant uint K_V4 = SPIRV_CROSS_CONSTANT_ID_2;"));
        assert!(out.contains("constant uint SUBGROUP_SIZE = SPIRV_CROSS_CONSTANT_ID_0;"));
    }

    #[test]
    fn leaves_unrelated_ifndef_blocks_alone() {
        let src = "#ifndef METAL_STDLIB\n\
                   #define METAL_STDLIB 1\n\
                   #endif\n\
                   kernel void cs_main() {}\n";
        let out = strip(src).unwrap();
        assert!(out.contains("#ifndef METAL_STDLIB"));
        assert!(out.contains("#define METAL_STDLIB 1"));
        assert!(out.contains("#endif"));
    }

    #[test]
    fn idempotent_on_already_stripped_source() {
        let src = "constant uint MAX_CHUNKS = SPIRV_CROSS_CONSTANT_ID_1;\n\
                   constant uint K_V4 = SPIRV_CROSS_CONSTANT_ID_2;\n\
                   kernel void cs_main() {}\n";
        let out = strip(src).unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn handles_minified_form_b_pair() {
        // The build pipeline currently runs strip *before* minify, but the
        // helper should be robust to either order — the AST is identical.
        let src = "constant uint X_tmp[[function_constant(0)]];constant uint X=is_function_constant_defined(X_tmp)?X_tmp:32u;kernel void cs_main(){}\n";
        let out = strip(src).unwrap();
        assert!(!out.contains("function_constant("));
        assert!(!out.contains("is_function_constant_defined"));
        assert!(out.contains("constant uint X = SPIRV_CROSS_CONSTANT_ID_0;"));
    }
}
