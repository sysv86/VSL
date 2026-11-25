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

/// Tokenizes `source` into a flat vector of [`SpannedToken`]s and any lexical diagnostics.
pub fn lex(file: FileId, source: &str) -> (Vec<SpannedToken>, Vec<Diagnostic>) {
    let mut tokens = Vec::new();
    let mut diagnostics = Vec::new();

    for (result, range) in Token::lexer(source).spanned() {
        let span = Span::from_range(file, &range);
        if let Ok(token) = result {
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
