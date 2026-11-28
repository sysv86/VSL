use logos::Logos;
use smol_str::SmolStr;
use vsl_diagnostics::{Diagnostic, FileId, Span};

/// Token stream element produced by the `logos` 0.16 DFA lexer.
#[derive(Logos, Debug, Clone, PartialEq, Eq)]
#[logos(skip r"[ \t\r\n\f]+")]
#[logos(skip(r"//[^\n]*", allow_greedy = true))]
#[logos(skip r"/\*([^*]|\*+[^*/])*\*+/")]
pub enum Token {
    // Module & item keywords
    #[token("module")]
    KwModule,
    #[token("import")]
    KwImport,
    #[token("pub")]
    KwPub,
    #[token("const")]
    KwConst,
    #[token("struct")]
    KwStruct,
    #[token("trait")]
    KwTrait,
    #[token("impl")]
    KwImpl,
    #[token("fn")]
    KwFn,
    #[token("let")]
    KwLet,
    #[token("mut")]
    KwMut,
    #[token("inout")]
    KwInout,
    #[token("out")]
    KwOut,
    #[token("workgroup")]
    KwWorkgroup,
    #[token("var")]
    KwVar,

    // Control-flow keywords
    #[token("if")]
    KwIf,
    #[token("else")]
    KwElse,
    #[token("for")]
    KwFor,
    #[token("in")]
    KwIn,
    #[token("while")]
    KwWhile,
    #[token("loop")]
    KwLoop,
    #[token("switch")]
    KwSwitch,
    #[token("case")]
    KwCase,
    #[token("default")]
    KwDefault,
    #[token("break")]
    KwBreak,
    #[token("continue")]
    KwContinue,
    #[token("return")]
    KwReturn,

    // Literals & `pack.vsl` DSL keywords
    #[token("true")]
    KwTrue,
    #[token("false")]
    KwFalse,
    #[token("none")]
    KwNone,
    #[token("pack")]
    KwPack,
    #[token("settings")]
    KwSettings,
    #[token("category")]
    KwCategory,
    #[token("param")]
    KwParam,
    #[token("spec")]
    KwSpec,
    #[token("enum")]
    KwEnum,
    #[token("presets")]
    KwPresets,
    #[token("preset")]
    KwPreset,
    #[token("pipeline")]
    KwPipeline,
    #[token("extend")]
    KwExtend,
    #[token("custom")]
    KwCustom,

    // Multi-character operators & punctuation
    #[token("::")]
    ColonColon,
    #[token("..=")]
    DotDotEq,
    #[token("..")]
    DotDot,
    #[token("->")]
    Arrow,
    #[token("+=")]
    PlusEq,
    #[token("-=")]
    MinusEq,
    #[token("*=")]
    StarEq,
    #[token("/=")]
    SlashEq,
    #[token("%=")]
    PercentEq,
    #[token("&=")]
    AmpEq,
    #[token("|=")]
    PipeEq,
    #[token("^=")]
    CaretEq,
    #[token("==")]
    EqEq,
    #[token("!=")]
    BangEq,
    #[token("<=")]
    LtEq,
    #[token(">=")]
    GtEq,
    #[token("&&")]
    AmpAmp,
    #[token("||")]
    PipePipe,
    #[token("<<")]
    Shl,
    #[token(">>")]
    Shr,

    // Single-character punctuation & operators
    #[token("@")]
    At,
    #[token("#")]
    Hash,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token(",")]
    Comma,
    #[token(":")]
    Colon,
    #[token(";")]
    Semi,
    #[token(".")]
    Dot,
    #[token("=")]
    Eq,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("!")]
    Bang,
    #[token("~")]
    Tilde,
    #[token("&")]
    Amp,
    #[token("|")]
    Pipe,
    #[token("^")]
    Caret,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,

    // Identifiers and literals
    #[regex(r"[A-Za-z_][A-Za-z0-9_]*", |lex| SmolStr::new(lex.slice()))]
    Ident(SmolStr),

    #[regex(r"0[xX][0-9a-fA-F_]+(?:u64|i64|u32|i32|u|i)?|0[bB][01_]+(?:u64|i64|u32|i32|u|i)?|0[oO][0-7_]+(?:u64|i64|u32|i32|u|i)?", |lex| SmolStr::new(lex.slice()))]
    HexLit(SmolStr),

    #[regex(r"[0-9][0-9_]*(?:\.[0-9][0-9_]*)?[eE][+-]?[0-9][0-9_]*(?:f16|f32|f)?|[0-9][0-9_]*\.[0-9][0-9_]*(?:f16|f32|f)?|[0-9][0-9_]*(?:f16|f32|f)", |lex| SmolStr::new(lex.slice()))]
    FloatLit(SmolStr),

    #[regex(r"[0-9][0-9_]*(?:u64|i64|u32|i32|u|i)?", |lex| SmolStr::new(lex.slice()))]
    IntLit(SmolStr),

    #[regex(r#""([^"\\]|\\.)*""#, |lex| unescape_string_literal(lex.slice()))]
    StringLit(SmolStr),
}

fn unescape_string_literal(raw: &str) -> SmolStr {
    let inner = &raw[1..raw.len().saturating_sub(1)];
    if !inner.contains('\\') {
        return SmolStr::new(inner);
    }
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some('\\') | None => out.push('\\'),
                Some('"') => out.push('"'),
                Some('0') => out.push('\0'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
            }
        } else {
            out.push(ch);
        }
    }
    SmolStr::new(out)
}

/// Token paired with its source [`Span`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpannedToken {
    pub token: Token,
    pub span: Span,
}

fn numeric_literal_error(token: &Token, source: &str, end: usize) -> Option<&'static str> {
    let (Token::IntLit(raw) | Token::HexLit(raw) | Token::FloatLit(raw)) = token else {
        return None;
    };
    if source
        .as_bytes()
        .get(end)
        .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
    {
        return Some("invalid numeric literal or suffix");
    }
    match token {
        Token::IntLit(_) | Token::HexLit(_) => {
            let is_32 = raw.ends_with("u32") || raw.ends_with("i32") || raw.ends_with(['u', 'i']);
            let clean = raw
                .trim_end_matches("u64")
                .trim_end_matches("i64")
                .trim_end_matches("u32")
                .trim_end_matches("i32")
                .trim_end_matches(['u', 'i'])
                .replace('_', "");
            let (digits, radix) = if let Some(rest) = clean
                .strip_prefix("0x")
                .or_else(|| clean.strip_prefix("0X"))
            {
                (rest, 16)
            } else if let Some(rest) = clean
                .strip_prefix("0b")
                .or_else(|| clean.strip_prefix("0B"))
            {
                (rest, 2)
            } else if let Some(rest) = clean
                .strip_prefix("0o")
                .or_else(|| clean.strip_prefix("0O"))
            {
                (rest, 8)
            } else {
                (clean.as_str(), 10)
            };
            match u64::from_str_radix(digits, radix) {
                Ok(value) if is_32 && u32::try_from(value).is_err() => {
                    Some("numeric literal exceeds 32-bit integer range")
                }
                Ok(_) => None,
                Err(_) => Some("invalid or out-of-range 64-bit integer literal"),
            }
        }
        Token::FloatLit(_) => {
            let clean = raw
                .trim_end_matches("f32")
                .trim_end_matches("f16")
                .trim_end_matches('f')
                .replace('_', "");
            // The manifest uses unsuffixed f64 values; explicit shader suffixes select f32/f16.
            if raw.ends_with("f16") || raw.ends_with("f32") || raw.ends_with('f') {
                if clean.parse::<f32>().is_ok_and(f32::is_finite) {
                    None
                } else {
                    Some("invalid or out-of-range floating-point literal")
                }
            } else if clean.parse::<f64>().is_ok_and(f64::is_finite) {
                None
            } else {
                Some("invalid or out-of-range floating-point literal")
            }
        }
        _ => None,
    }
}

/// Tokenizes `source` into a flat vector of [`SpannedToken`]s and any lexical diagnostics.
pub fn lex(file: FileId, source: &str) -> (Vec<SpannedToken>, Vec<Diagnostic>) {
    let mut tokens = Vec::new();
    let mut diagnostics = Vec::new();

    for (result, range) in Token::lexer(source).spanned() {
        let span = Span::from_range(file, &range);
        if let Ok(token) = result {
            if let Some(message) = numeric_literal_error(&token, source, range.end) {
                diagnostics.push(Diagnostic::error("E0001", message, span));
            }
            tokens.push(SpannedToken { token, span });
        } else {
            let bad = source.get(span.to_range()).unwrap_or("");
            diagnostics.push(
                Diagnostic::error(
                    "E0001",
                    format!("unexpected character sequence `{bad}`"),
                    span,
                )
                .with_primary_label(span, "invalid token"),
            );
        }
    }

    (tokens, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_malformed_numeric_literals_and_preserves_valid_boundaries() {
        for source in [
            "0x_",
            "0b102",
            "0o89",
            "0x",
            "1e+",
            "10u128",
            "1f32foo",
            "4294967296u",
            "18446744073709551616u64",
            "1e100f32",
        ] {
            let (_, diagnostics) = lex(FileId(0), source);
            assert!(!diagnostics.is_empty(), "{source}");
        }
        let (tokens, diagnostics) = lex(
            FileId(0),
            "0xffffffffu 0xffffffffffffffffu64 1e+1_0 1..2 0b1_0u 0o17i",
        );
        assert!(diagnostics.is_empty(), "{diagnostics:?}");
        assert!(tokens.iter().any(|token| token.token == Token::DotDot));
    }

    #[test]
    fn lexes_vsl_attributes_and_literals() {
        let src = r"
            #[cap(int64_atomics)]
            @stage(compute(8, 8, 1))
            pub const EMPTY_KEY: u64 = 0u64;
            let eps = 1e-6;
        ";
        let (tokens, diags) = lex(FileId(0), src);
        assert_eq!(diags, Vec::new());
        assert!(tokens.iter().any(|t| t.token == Token::At));
        assert!(tokens.iter().any(|t| t.token == Token::Hash));
        assert!(
            tokens
                .iter()
                .any(|t| t.token == Token::FloatLit(SmolStr::new_static("1e-6")))
        );
    }

    #[test]
    fn lexes_integer_form_floats_and_underscore_exponents() {
        let src = r"
            let a = 1f32;
            let b = 1f16;
            let c = 1f;
            let d = 1e+1_0;
            let e = 42u32;
            let f = 0x1f;
        ";
        let (tokens, diags) = lex(FileId(0), src);
        assert_eq!(diags, Vec::new());
        assert!(
            tokens
                .iter()
                .any(|t| t.token == Token::FloatLit(SmolStr::new_static("1f32")))
        );
        assert!(
            tokens
                .iter()
                .any(|t| t.token == Token::FloatLit(SmolStr::new_static("1f16")))
        );
        assert!(
            tokens
                .iter()
                .any(|t| t.token == Token::FloatLit(SmolStr::new_static("1f")))
        );
        assert!(
            tokens
                .iter()
                .any(|t| t.token == Token::FloatLit(SmolStr::new_static("1e+1_0")))
        );
        assert!(
            tokens
                .iter()
                .any(|t| t.token == Token::IntLit(SmolStr::new_static("42u32")))
        );
        assert!(
            tokens
                .iter()
                .any(|t| t.token == Token::HexLit(SmolStr::new_static("0x1f")))
        );
    }
}
