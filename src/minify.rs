use tree_sitter::{Node, Parser, Tree};

pub fn minify(src: &str) -> Result<String, String> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_cpp::LANGUAGE.into())
        .map_err(|e| format!("loading C++ grammar: {e}"))?;

    let tree = parser
        .parse(src, None)
        .ok_or_else(|| "tree-sitter failed to parse input".to_string())?;
    let input_errors = count_errors(&tree);

    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut last: char = '\0';
    emit(tree.root_node(), bytes, &mut out, &mut last);

    while out.ends_with('\n') {
        out.pop();
    }
    out.push('\n');

    let out_tree = parser
        .parse(&out, None)
        .ok_or_else(|| "tree-sitter failed to re-parse output".to_string())?;
    let output_errors = count_errors(&out_tree);
    if output_errors > input_errors {
        return Err(format!(
            "minification introduced new parse errors ({input_errors} -> {output_errors})"
        ));
    }

    Ok(out)
}

fn count_errors(tree: &Tree) -> usize {
    let mut n = 0;
    let mut cursor = tree.walk();
    let mut visited_self = false;
    loop {
        if !visited_self {
            let node = cursor.node();
            if node.is_error() || node.is_missing() {
                n += 1;
            }
        }
        if !visited_self && cursor.goto_first_child() {
            visited_self = false;
            continue;
        }
        if cursor.goto_next_sibling() {
            visited_self = false;
            continue;
        }
        if !cursor.goto_parent() {
            break;
        }
        visited_self = true;
    }
    n
}

fn emit(node: Node, src: &[u8], out: &mut String, last: &mut char) {
    let kind = node.kind();

    if kind == "comment" {
        return;
    }

    if kind.starts_with("preproc_") {
        if !out.is_empty() && *last != '\n' {
            out.push('\n');
            *last = '\n';
        }
        let text = node.utf8_text(src).unwrap();
        let text = text.trim_end_matches(['\n', '\r']);
        let text = strip_inline_comments_in_directive(text);
        out.push_str(&text);
        out.push('\n');
        *last = '\n';
        return;
    }

    if matches!(
        kind,
        "string_literal"
            | "raw_string_literal"
            | "char_literal"
            | "number_literal"
            | "system_lib_string"
            | "user_defined_literal"
            | "concatenated_string"
    ) {
        write_token(node.utf8_text(src).unwrap(), out, last);
        return;
    }

    if node.child_count() == 0 {
        let text = node.utf8_text(src).unwrap();
        if !text.is_empty() {
            write_token(text, out, last);
        }
        return;
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        emit(child, src, out, last);
    }
}

fn write_token(text: &str, out: &mut String, last: &mut char) {
    let Some(first) = text.chars().next() else {
        return;
    };
    if !out.is_empty() && *last != '\n' && needs_separator(*last, first) {
        out.push(' ');
    }
    out.push_str(text);
    *last = text.chars().last().unwrap();
}

fn needs_separator(prev: char, next: char) -> bool {
    let word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    if word(prev) && word(next) {
        return true;
    }
    matches!(
        (prev, next),
        ('+', '+' | '=')
            | ('-', '-' | '=' | '>')
            | (
                '*' | '/' | '%' | '<' | '>' | '=' | '!' | '&' | '|' | '^',
                '='
            )
            | ('*' | '/', '/')
            | ('/', '*')
            | ('%' | '>', '>')
            | ('<', '<' | ':' | '%')
            | ('&', '&')
            | ('|', '|')
            | (':', ':')
            | ('.', '.')
            | ('#', '#')
    )
}

fn strip_inline_comments_in_directive(line: &str) -> String {
    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    let mut in_str: Option<u8> = None;
    while i < bytes.len() {
        let c = bytes[i];
        if let Some(q) = in_str {
            out.push(c as char);
            if c == b'\\' && i + 1 < bytes.len() {
                out.push(bytes[i + 1] as char);
                i += 2;
                continue;
            }
            if c == q {
                in_str = None;
            }
            i += 1;
            continue;
        }
        if c == b'"' || c == b'\'' || c == b'<' {
            // '<' only opens an angle-string after #include; cheap heuristic:
            // treat it as quote when the line begins with '#' and 'include' appears before.
            if c == b'<' {
                let prefix = &line[..i];
                if prefix.trim_start().starts_with('#') && prefix.contains("include") {
                    out.push('<');
                    in_str = Some(b'>');
                    i += 1;
                    continue;
                }
            } else {
                out.push(c as char);
                in_str = Some(c);
                i += 1;
                continue;
            }
        }
        if c == b'/' && i + 1 < bytes.len() {
            if bytes[i + 1] == b'/' {
                break;
            }
            if bytes[i + 1] == b'*' {
                if let Some(end) = line[i + 2..].find("*/") {
                    i += 2 + end + 2;
                } else {
                    i = bytes.len();
                }
                if !out.ends_with(' ') {
                    out.push(' ');
                }
                continue;
            }
        }
        out.push(c as char);
        i += 1;
    }

    out.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_simple_function() {
        let src = r#"
            // load a single fp16 word
            static inline float load_w_f16(const uint* w, uint b) {
                uint word = w[b >> 2];          // word load
                uint h = (word >> ((b & 2u) * 8u)) & 0xFFFFu;
                return float(h);
            }
        "#;
        let out = minify(src).expect("minify");
        // comments gone
        assert!(!out.contains("//"));
        assert!(!out.contains("/*"));
        // tokens preserved
        for tok in [
            "static",
            "inline",
            "float",
            "load_w_f16",
            "uint",
            "word",
            ">>",
            "0xFFFFu",
            "return",
        ] {
            assert!(out.contains(tok), "missing {tok} in {out}");
        }
        // adjacent identifiers/keywords still separated
        assert!(out.contains("static inline float"));
        // idempotent
        let out2 = minify(&out).expect("re-minify");
        assert_eq!(out, out2);
    }

    #[test]
    fn preserves_preprocessor_lines() {
        let src = "#include <metal_stdlib>\n#define FOO 1\nint x = FOO;\n";
        let out = minify(src).unwrap();
        assert!(out.contains("#include <metal_stdlib>\n"));
        assert!(out.contains("#define FOO 1\n"));
        assert!(
            out.contains("int x=FOO;")
                || out.contains("int x =FOO;")
                || out.contains("int x= FOO;")
        );
    }

    #[test]
    fn preserves_string_literals_with_spaces() {
        let src = r#"const char* s = "hello world";"#;
        let out = minify(src).unwrap();
        assert!(out.contains(r#""hello world""#));
    }

    #[test]
    fn handles_attributes_and_templates() {
        let src = r#"
            kernel void k(
                constant int& p [[buffer(0)]],
                device half* y [[buffer(1)]],
                uint tid [[thread_index_in_threadgroup]]) {
                simdgroup_matrix<float, 8, 8> C;
            }
        "#;
        let out = minify(src).unwrap();
        assert!(out.contains("[[buffer(0)]]"));
        assert!(out.contains("[[thread_index_in_threadgroup]]"));
        assert!(
            out.contains("simdgroup_matrix<float,8,8>")
                || out.contains("simdgroup_matrix<float, 8, 8>")
        );
    }
}
