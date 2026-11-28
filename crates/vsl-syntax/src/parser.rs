use crate::ast::{
    Attribute, AttributeEntry, BinaryOp, Block, CallArg, ConstDecl, Expr, FnDecl, FnParam,
    GenericParam, Ident, ImplBlock, ImportDecl, ImportTree, Item, ModuleDecl, PackDecl, PackField,
    ParamMode, Path, PipelineDecl, PipelineModeAst, PresetDecl, PresetsDecl, SettingCategoryDecl,
    SettingEntryDecl, SettingKind, SettingTypeAst, SettingsDecl, SourceUnit, Stmt, StructDecl,
    StructField, SwitchCase, TraitDecl, TypeArg, TypeExpr, UnaryOp, WorkgroupVarDecl,
};
use crate::token::{SpannedToken, Token, lex};
use smol_str::SmolStr;
use vsl_diagnostics::{Diagnostic, FileId, Span};

/// Result of lexing and parsing a `.vsl` or `pack.vsl` file.
#[derive(Debug, Clone)]
pub struct ParseResult {
    pub unit: SourceUnit,
    pub diagnostics: Vec<Diagnostic>,
}

impl ParseResult {
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(Diagnostic::is_error)
    }
}

/// Lexes and parses `source` into a [`SourceUnit`] with panic-mode error recovery.
pub fn parse_source(file: FileId, source: &str) -> ParseResult {
    let (tokens, mut diagnostics) = lex(file, source);
    let mut parser = Parser {
        file,
        tokens,
        pos: 0,
        allow_struct_lit: true,
        source_end: source.len() as u32,
        nesting: 0,
        diagnostics: Vec::new(),
    };
    let unit = parser.parse_unit();
    diagnostics.extend(parser.diagnostics);
    ParseResult { unit, diagnostics }
}

enum ParsedStmt {
    Stmt(Stmt),
    TailExpr(Expr),
}

struct Parser {
    file: FileId,
    tokens: Vec<SpannedToken>,
    pos: usize,
    allow_struct_lit: bool,
    source_end: u32,
    nesting: usize,
    diagnostics: Vec<Diagnostic>,
}

impl Parser {
    // Bound recursive grammar entry points before allocating or dropping a deeply nested AST.
    fn with_nesting<T>(&mut self, parse: impl FnOnce(&mut Self) -> Option<T>) -> Option<T> {
        if self.nesting >= 64 {
            self.diagnostics.push(Diagnostic::error(
                "E0110",
                "syntax nesting limit exceeded",
                self.current_span(),
            ));
            return None;
        }
        self.nesting += 1;
        let result = parse(self);
        self.nesting -= 1;
        result
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|t| &t.token)
    }

    fn peek_n(&self, offset: usize) -> Option<&Token> {
        self.tokens.get(self.pos + offset).map(|t| &t.token)
    }

    fn current_span(&self) -> Span {
        if let Some(tok) = self.tokens.get(self.pos) {
            tok.span
        } else {
            Span::new(self.file, self.source_end, self.source_end)
        }
    }

    fn bump(&mut self) -> Option<SpannedToken> {
        if self.pos < self.tokens.len() {
            let tok = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(tok)
        } else {
            None
        }
    }

    fn eat(&mut self, expected: &Token) -> Option<Span> {
        if self.peek() == Some(expected) {
            self.bump().map(|t| t.span)
        } else {
            None
        }
    }

    fn expect(&mut self, expected: &Token, label: &str) -> Option<Span> {
        if self.peek() == Some(expected) {
            self.bump().map(|t| t.span)
        } else {
            let span = self.current_span();
            self.diagnostics.push(
                Diagnostic::error("E0100", format!("expected {label}"), span)
                    .with_primary_label(span, format!("expected {label} here")),
            );
            None
        }
    }

    /// Consumes a closing `>` in generic parameter or type-argument lists, splitting `>>`
    /// (`Token::Shr`) in place so nested generics like `ptr<Reservoir<S>>` parse cleanly.
    fn expect_gt(&mut self) -> Option<Span> {
        match self.peek() {
            Some(Token::Gt) => self.bump().map(|t| t.span),
            Some(Token::Shr) => {
                let full_span = self.tokens[self.pos].span;
                let first_span = Span::new(full_span.file, full_span.start, full_span.start + 1);
                let second_span = Span::new(full_span.file, full_span.start + 1, full_span.end);
                self.tokens[self.pos] = SpannedToken {
                    token: Token::Gt,
                    span: second_span,
                };
                Some(first_span)
            }
            _ => {
                let span = self.current_span();
                self.diagnostics.push(
                    Diagnostic::error("E0100", "expected `>`", span)
                        .with_primary_label(span, "expected `>` here"),
                );
                None
            }
        }
    }

    fn at_closing_gt(&self) -> bool {
        matches!(self.peek(), Some(Token::Gt | Token::Shr))
    }

    /// Parses an identifier, also permitting contextual keywords used as field/param names.
    fn parse_ident(&mut self) -> Option<Ident> {
        let Some(tok) = self.tokens.get(self.pos) else {
            self.diagnostics.push(Diagnostic::error(
                "E0101",
                "expected identifier",
                self.current_span(),
            ));
            return None;
        };
        let name = match &tok.token {
            Token::Ident(s) => s.clone(),
            Token::KwPack => SmolStr::new_static("pack"),
            Token::KwParam => SmolStr::new_static("param"),
            Token::KwSpec => SmolStr::new_static("spec"),
            Token::KwDefault => SmolStr::new_static("default"),
            Token::KwExtend => SmolStr::new_static("extend"),
            Token::KwCustom => SmolStr::new_static("custom"),
            Token::KwOut => SmolStr::new_static("out"),
            Token::KwIn => SmolStr::new_static("in"),
            Token::KwVar => SmolStr::new_static("var"),
            Token::KwCategory => SmolStr::new_static("category"),
            Token::KwPreset => SmolStr::new_static("preset"),
            Token::KwPresets => SmolStr::new_static("presets"),
            Token::KwSettings => SmolStr::new_static("settings"),
            Token::KwPipeline => SmolStr::new_static("pipeline"),
            Token::KwEnum => SmolStr::new_static("enum"),
            Token::KwWorkgroup => SmolStr::new_static("workgroup"),
            Token::KwSwitch => SmolStr::new_static("switch"),
            Token::KwCase => SmolStr::new_static("case"),
            _ => {
                let span = tok.span;
                self.diagnostics.push(
                    Diagnostic::error("E0101", "expected identifier", span)
                        .with_primary_label(span, "expected an identifier"),
                );
                return None;
            }
        };
        let span = tok.span;
        self.pos += 1;
        Some(Ident { name, span })
    }

    fn is_contextual_ident(tok: Option<&Token>) -> bool {
        matches!(
            tok,
            Some(
                Token::Ident(_)
                    | Token::KwPack
                    | Token::KwParam
                    | Token::KwSpec
                    | Token::KwDefault
                    | Token::KwExtend
                    | Token::KwCustom
                    | Token::KwOut
                    | Token::KwIn
                    | Token::KwVar
                    | Token::KwCategory
                    | Token::KwPreset
                    | Token::KwPresets
                    | Token::KwSettings
                    | Token::KwPipeline
                    | Token::KwEnum
                    | Token::KwWorkgroup
                    | Token::KwSwitch
                    | Token::KwCase
            )
        )
    }

    fn parse_string_lit(&mut self) -> Option<(SmolStr, Span)> {
        if let Some(SpannedToken {
            token: Token::StringLit(s),
            span,
        }) = self.tokens.get(self.pos)
        {
            let res = (s.clone(), *span);
            self.pos += 1;
            Some(res)
        } else {
            let span = self.current_span();
            self.diagnostics.push(
                Diagnostic::error("E0102", "expected string literal", span)
                    .with_primary_label(span, "expected string literal"),
            );
            None
        }
    }

    fn parse_path(&mut self) -> Option<Path> {
        let first = self.parse_ident()?;
        let mut span = first.span;
        let mut segments = vec![first];
        while self.eat(&Token::ColonColon).is_some() {
            if matches!(self.peek(), Some(Token::LBrace | Token::Star | Token::Lt)) {
                self.pos -= 1;
                break;
            }
            if let Some(next) = self.parse_ident() {
                span = span.merge(next.span);
                segments.push(next);
            } else {
                break;
            }
        }
        Some(Path { segments, span })
    }

    fn synchronize_item(&mut self) {
        while !self.is_eof() {
            if matches!(
                self.peek(),
                Some(
                    Token::KwPub
                        | Token::KwFn
                        | Token::KwStruct
                        | Token::KwTrait
                        | Token::KwImpl
                        | Token::KwConst
                        | Token::KwWorkgroup
                        | Token::KwPack
                        | Token::KwSettings
                        | Token::KwPresets
                        | Token::KwPipeline
                        | Token::Hash
                        | Token::At
                )
            ) {
                break;
            }
            if self.eat(&Token::Semi).is_some() || self.eat(&Token::RBrace).is_some() {
                break;
            }
            self.bump();
        }
    }

    fn parse_unit(&mut self) -> SourceUnit {
        let mut module_decl = None;
        let mut imports = Vec::new();
        let mut items = Vec::new();

        if self.peek() == Some(&Token::KwModule) {
            module_decl = self.parse_module_decl();
        }

        while self.peek() == Some(&Token::KwImport) {
            if let Some(imp) = self.parse_import_decl() {
                imports.push(imp);
            } else {
                self.synchronize_item();
            }
        }

        while !self.is_eof() {
            let start_pos = self.pos;
            if let Some(item) = self.parse_item() {
                items.push(item);
            } else {
                self.synchronize_item();
                if self.pos == start_pos {
                    self.bump();
                }
            }
        }

        SourceUnit {
            module_decl,
            imports,
            items,
        }
    }

    fn parse_module_decl(&mut self) -> Option<ModuleDecl> {
        let start = self.expect(&Token::KwModule, "`module`")?;
        let path = self.parse_path()?;
        let end = self.expect(&Token::Semi, "`;`").unwrap_or(path.span);
        Some(ModuleDecl {
            path,
            span: start.merge(end),
        })
    }

    fn parse_import_decl(&mut self) -> Option<ImportDecl> {
        let start = self.expect(&Token::KwImport, "`import`")?;
        let prefix = self.parse_path()?;
        let tree = if self.eat(&Token::ColonColon).is_some() {
            if self.eat(&Token::LBrace).is_some() {
                let mut names = Vec::new();
                while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
                    if let Some(ident) = self.parse_ident() {
                        names.push(ident);
                    } else {
                        break;
                    }
                    if self.eat(&Token::Comma).is_none() {
                        break;
                    }
                }
                self.expect(&Token::RBrace, "`}`");
                ImportTree::Group(names)
            } else if self.eat(&Token::Star).is_some() {
                ImportTree::Glob
            } else {
                ImportTree::Single
            }
        } else {
            ImportTree::Single
        };
        let end = self.expect(&Token::Semi, "`;`").unwrap_or(prefix.span);
        Some(ImportDecl {
            prefix,
            tree,
            span: start.merge(end),
        })
    }

    fn parse_attributes(&mut self) -> Vec<Attribute> {
        let mut attrs = Vec::new();
        loop {
            if let Some(start) = self.eat(&Token::Hash) {
                self.expect(&Token::LBracket, "`[`");
                let mut entries = Vec::new();
                while !self.is_eof() && self.peek() != Some(&Token::RBracket) {
                    if let Some(entry) = self.parse_attr_entry() {
                        entries.push(entry);
                    } else {
                        break;
                    }
                    if self.eat(&Token::Comma).is_none() {
                        break;
                    }
                }
                let end = self.expect(&Token::RBracket, "`]`").unwrap_or(start);
                attrs.push(Attribute {
                    is_at_decorator: false,
                    entries,
                    span: start.merge(end),
                });
            } else if let Some(start) = self.eat(&Token::At) {
                if let Some(entry) = self.parse_attr_entry() {
                    let span = start.merge(entry.span);
                    attrs.push(Attribute {
                        is_at_decorator: true,
                        entries: vec![entry],
                        span,
                    });
                }
            } else {
                break;
            }
        }
        attrs
    }

    fn parse_attr_entry(&mut self) -> Option<AttributeEntry> {
        let name = self.parse_ident()?;
        let mut span = name.span;
        let mut args = Vec::new();
        if self.eat(&Token::LParen).is_some() {
            while !self.is_eof() && self.peek() != Some(&Token::RParen) {
                if let Some(expr) = self.parse_expr() {
                    args.push(expr);
                } else {
                    break;
                }
                if self.eat(&Token::Comma).is_none() {
                    break;
                }
            }
            if let Some(end) = self.expect(&Token::RParen, "`)`") {
                span = span.merge(end);
            }
        }
        Some(AttributeEntry { name, args, span })
    }

    fn parse_item(&mut self) -> Option<Item> {
        let attrs = self.parse_attributes();
        let is_pub = self.eat(&Token::KwPub).is_some();

        match self.peek() {
            Some(Token::KwConst) => self.parse_const_decl(attrs, is_pub).map(Item::Const),
            Some(Token::KwWorkgroup) => self.parse_workgroup_var(attrs).map(Item::WorkgroupVar),
            Some(Token::KwStruct) => self.parse_struct_decl(attrs, is_pub).map(Item::Struct),
            Some(Token::KwTrait) => self.parse_trait_decl(attrs, is_pub).map(Item::Trait),
            Some(Token::KwImpl) => self.parse_impl_block(attrs).map(Item::Impl),
            Some(Token::KwFn) => self.parse_fn_decl(attrs, is_pub).map(Item::Function),
            Some(Token::KwPack) => self.parse_pack_decl().map(Item::Pack),
            Some(Token::KwSettings) => self.parse_settings_decl().map(Item::Settings),
            Some(Token::KwPresets) => self.parse_presets_decl().map(Item::Presets),
            Some(Token::KwPipeline) => self.parse_pipeline_decl().map(Item::Pipeline),
            _ => {
                let span = self.current_span();
                self.diagnostics.push(
                    Diagnostic::error("E0103", "unexpected token at item position", span)
                        .with_primary_label(span, "expected item declaration"),
                );
                None
            }
        }
    }

    fn parse_const_decl(&mut self, attrs: Vec<Attribute>, is_pub: bool) -> Option<ConstDecl> {
        let start = self.expect(&Token::KwConst, "`const`")?;
        let name = self.parse_ident()?;
        self.expect(&Token::Colon, "`:`")?;
        let ty = self.parse_type_expr()?;
        self.expect(&Token::Eq, "`=`")?;
        let value = self.parse_expr()?;
        let end = self.expect(&Token::Semi, "`;`").unwrap_or(value.span());
        Some(ConstDecl {
            attrs,
            is_pub,
            name,
            ty,
            value,
            span: start.merge(end),
        })
    }

    fn parse_workgroup_var(&mut self, attrs: Vec<Attribute>) -> Option<WorkgroupVarDecl> {
        let start = self.expect(&Token::KwWorkgroup, "`workgroup`")?;
        self.expect(&Token::KwVar, "`var`")?;
        let name = self.parse_ident()?;
        self.expect(&Token::Colon, "`:`")?;
        let ty = self.parse_type_expr()?;
        let end = self.expect(&Token::Semi, "`;`").unwrap_or(name.span);
        Some(WorkgroupVarDecl {
            attrs,
            name,
            ty,
            span: start.merge(end),
        })
    }

    fn parse_generics(&mut self) -> Vec<GenericParam> {
        let mut params = Vec::new();
        if self.eat(&Token::Lt).is_none() {
            return params;
        }
        while !self.is_eof() && !self.at_closing_gt() {
            let Some(name) = self.parse_ident() else {
                break;
            };
            let mut span = name.span;
            let bound = if self.eat(&Token::Colon).is_some() {
                let b = self.parse_path();
                if let Some(ref path) = b {
                    span = span.merge(path.span);
                }
                b
            } else {
                None
            };
            params.push(GenericParam { name, bound, span });
            if self.eat(&Token::Comma).is_none() {
                break;
            }
        }
        self.expect_gt();
        params
    }

    fn parse_struct_decl(&mut self, attrs: Vec<Attribute>, is_pub: bool) -> Option<StructDecl> {
        let start = self.expect(&Token::KwStruct, "`struct`")?;
        let name = self.parse_ident()?;
        let generics = self.parse_generics();
        self.expect(&Token::LBrace, "`{`")?;
        let mut fields = Vec::new();
        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            let field_attrs = self.parse_attributes();
            let field_pub = self.eat(&Token::KwPub).is_some();
            let Some(field_name) = self.parse_ident() else {
                break;
            };
            self.expect(&Token::Colon, "`:`")?;
            let field_ty = self.parse_type_expr()?;
            let field_span = field_name.span;
            fields.push(StructField {
                attrs: field_attrs,
                is_pub: field_pub,
                name: field_name,
                ty: field_ty,
                span: field_span,
            });
            if self.eat(&Token::Comma).is_none() {
                break;
            }
        }
        let end = self.expect(&Token::RBrace, "`}`").unwrap_or(start);
        Some(StructDecl {
            attrs,
            is_pub,
            name,
            generics,
            fields,
            span: start.merge(end),
        })
    }

    fn parse_trait_decl(&mut self, attrs: Vec<Attribute>, is_pub: bool) -> Option<TraitDecl> {
        let start = self.expect(&Token::KwTrait, "`trait`")?;
        let name = self.parse_ident()?;
        self.expect(&Token::LBrace, "`{`")?;
        let mut methods = Vec::new();
        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            let m_attrs = self.parse_attributes();
            let m_pub = self.eat(&Token::KwPub).is_some();
            if let Some(m) = self.parse_fn_decl(m_attrs, m_pub) {
                methods.push(m);
            } else {
                break;
            }
        }
        let end = self.expect(&Token::RBrace, "`}`").unwrap_or(start);
        Some(TraitDecl {
            attrs,
            is_pub,
            name,
            methods,
            span: start.merge(end),
        })
    }

    fn parse_impl_block(&mut self, attrs: Vec<Attribute>) -> Option<ImplBlock> {
        let start = self.expect(&Token::KwImpl, "`impl`")?;
        let generics = self.parse_generics();
        let first_ty = self.parse_type_expr()?;
        let (trait_path, target) = if self.eat(&Token::KwFor).is_some() {
            let trait_path = match first_ty {
                TypeExpr::Named { path, .. } => Some(path),
                TypeExpr::Array { .. } => None,
            };
            let target = self.parse_type_expr()?;
            (trait_path, target)
        } else {
            (None, first_ty)
        };
        self.expect(&Token::LBrace, "`{`")?;
        let mut methods = Vec::new();
        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            let m_attrs = self.parse_attributes();
            let m_pub = self.eat(&Token::KwPub).is_some();
            if let Some(m) = self.parse_fn_decl(m_attrs, m_pub) {
                methods.push(m);
            } else {
                break;
            }
        }
        let end = self.expect(&Token::RBrace, "`}`").unwrap_or(start);
        Some(ImplBlock {
            attrs,
            generics,
            trait_path,
            target,
            methods,
            span: start.merge(end),
        })
    }

    fn parse_fn_params(&mut self) -> Vec<FnParam> {
        let mut params = Vec::new();
        self.expect(&Token::LParen, "`(`");
        while !self.is_eof() && self.peek() != Some(&Token::RParen) {
            let p_attrs = self.parse_attributes();
            let mut mode = if self.eat(&Token::KwInout).is_some() {
                ParamMode::InOut
            } else if self.eat(&Token::KwOut).is_some() {
                ParamMode::Out
            } else if self.eat(&Token::KwMut).is_some() {
                ParamMode::Mut
            } else {
                ParamMode::In
            };
            let Some(name) = self.parse_ident() else {
                break;
            };
            let mut span = name.span;
            let ty = if self.eat(&Token::Colon).is_some() {
                // Support `g: mut mc::GraphBuilder` in `pipeline` signatures.
                if self.eat(&Token::KwMut).is_some() && mode == ParamMode::In {
                    mode = ParamMode::Mut;
                }
                let parsed = self.parse_type_expr();
                if let Some(ref t) = parsed {
                    span = span.merge(type_span(t));
                }
                parsed
            } else {
                None
            };
            params.push(FnParam {
                attrs: p_attrs,
                mode,
                name,
                ty,
                span,
            });
            if self.eat(&Token::Comma).is_none() {
                break;
            }
        }
        self.expect(&Token::RParen, "`)`");
        params
    }

    fn parse_fn_decl(&mut self, attrs: Vec<Attribute>, is_pub: bool) -> Option<FnDecl> {
        let start = self.expect(&Token::KwFn, "`fn`")?;
        let name = self.parse_ident()?;
        let generics = self.parse_generics();
        let params = self.parse_fn_params();
        let return_ty = if self.eat(&Token::Arrow).is_some() {
            self.parse_type_expr()
        } else {
            None
        };
        let (body, end) = if self.peek() == Some(&Token::LBrace) {
            let b = self.parse_block()?;
            let s = b.span;
            (Some(b), s)
        } else {
            let s = self.expect(&Token::Semi, "`;` or `{`").unwrap_or(name.span);
            (None, s)
        };
        Some(FnDecl {
            attrs,
            is_pub,
            name,
            generics,
            params,
            return_ty,
            body,
            span: start.merge(end),
        })
    }

    fn parse_type_expr(&mut self) -> Option<TypeExpr> {
        self.with_nesting(Self::parse_type_expr_inner)
    }

    fn parse_type_expr_inner(&mut self) -> Option<TypeExpr> {
        if let Some(start) = self.eat(&Token::LBracket) {
            let element = self.parse_type_expr()?;
            self.expect(&Token::Semi, "`;`")?;
            let len = self.parse_expr()?;
            let end = self.expect(&Token::RBracket, "`]`").unwrap_or(start);
            return Some(TypeExpr::Array {
                element: Box::new(element),
                len: Box::new(len),
                span: start.merge(end),
            });
        }

        let path = self.parse_path()?;
        let mut span = path.span;
        let mut args = Vec::new();
        if self.eat(&Token::Lt).is_some() {
            args = self.parse_type_args();
            if let Some(end) = self.expect_gt() {
                span = span.merge(end);
            }
        }

        let mut unsized_array = false;
        if self.peek() == Some(&Token::LBracket) && self.peek_n(1) == Some(&Token::RBracket) {
            self.bump();
            if let Some(end) = self.bump() {
                span = span.merge(end.span);
            }
            unsized_array = true;
        }

        Some(TypeExpr::Named {
            path,
            args,
            unsized_array,
            span,
        })
    }

    fn parse_type_args(&mut self) -> Vec<TypeArg> {
        let mut args = Vec::new();
        while !self.is_eof() && !self.at_closing_gt() {
            if Self::is_contextual_ident(self.peek()) && self.peek_n(1) == Some(&Token::Eq) {
                let Some(name) = self.parse_ident() else {
                    break;
                };
                self.bump(); // `=`
                // Bind above shift precedence (`Token::Shr` has l_bp = 18) so nested generics
                // such as `Outer<ptr<u32, align = 16>>` split `>>` into two closing `>` tokens
                // instead of parsing it as a bitwise right-shift operator.
                if let Some(value) = self.parse_expr_bp(19) {
                    let span = name.span.merge(value.span());
                    args.push(TypeArg::NamedExpr { name, value, span });
                }
            } else if let Some(ty) = self.parse_type_expr() {
                args.push(TypeArg::Type(ty));
            } else {
                break;
            }
            if self.eat(&Token::Comma).is_none() {
                break;
            }
        }
        args
    }

    fn parse_block(&mut self) -> Option<Block> {
        self.with_nesting(Self::parse_block_inner)
    }

    fn parse_block_inner(&mut self) -> Option<Block> {
        let start = self.expect(&Token::LBrace, "`{`")?;
        let mut stmts = Vec::new();
        let mut tail_expr = None;

        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            if matches!(
                self.peek(),
                Some(
                    Token::KwFn
                        | Token::KwPub
                        | Token::KwStruct
                        | Token::KwTrait
                        | Token::KwImpl
                        | Token::KwConst
                        | Token::KwModule
                        | Token::KwImport
                )
            ) {
                break;
            }
            let before = self.pos;
            match self.parse_stmt(true) {
                Some(ParsedStmt::Stmt(stmt)) => stmts.push(stmt),
                Some(ParsedStmt::TailExpr(expr)) => {
                    tail_expr = Some(Box::new(expr));
                    break;
                }
                None => {}
            }
            if self.pos == before {
                self.bump();
            }
        }

        let end = self.expect(&Token::RBrace, "`}`").unwrap_or(start);
        Some(Block {
            stmts,
            tail_expr,
            span: start.merge(end),
        })
    }

    fn parse_stmt(&mut self, allow_tail: bool) -> Option<ParsedStmt> {
        match self.peek() {
            Some(Token::KwLet) => self.parse_let_stmt().map(ParsedStmt::Stmt),
            Some(Token::KwReturn) => self.parse_return_stmt().map(ParsedStmt::Stmt),
            Some(Token::KwBreak) => {
                let s = self.bump()?.span;
                let end = self.eat(&Token::Semi).unwrap_or(s);
                Some(ParsedStmt::Stmt(Stmt::Break(s.merge(end))))
            }
            Some(Token::KwContinue) => {
                let s = self.bump()?.span;
                let end = self.eat(&Token::Semi).unwrap_or(s);
                Some(ParsedStmt::Stmt(Stmt::Continue(s.merge(end))))
            }
            Some(Token::Hash | Token::At) => {
                let attrs = self.parse_attributes();
                match self.peek() {
                    Some(Token::KwFor) => self.parse_for_stmt(attrs).map(ParsedStmt::Stmt),
                    Some(Token::KwWhile) => self.parse_while_stmt(attrs).map(ParsedStmt::Stmt),
                    Some(Token::KwLoop) => self.parse_loop_stmt(attrs).map(ParsedStmt::Stmt),
                    _ => {
                        let span = self.current_span();
                        self.diagnostics.push(Diagnostic::error(
                            "E0104",
                            "attributes inside blocks are only valid on `for`, `while`, or `loop` statements",
                            span,
                        ));
                        None
                    }
                }
            }
            Some(Token::KwFor) => self.parse_for_stmt(Vec::new()).map(ParsedStmt::Stmt),
            Some(Token::KwWhile) => self.parse_while_stmt(Vec::new()).map(ParsedStmt::Stmt),
            Some(Token::KwLoop) => self.parse_loop_stmt(Vec::new()).map(ParsedStmt::Stmt),
            Some(Token::KwSwitch) => self.parse_switch_stmt().map(ParsedStmt::Stmt),
            Some(Token::KwIf) => {
                // Parse statement-level `if` directly so a trailing `(` or `[` on the next
                // statement is never misparsed as a postfix call or index, while still
                // allowing member access and infix continuations such as
                // `if c { 1 } else { 2 } + 1`.
                let mut expr = self.parse_if_expr()?;
                let has_continuation = self.peek() == Some(&Token::Dot)
                    || self.peek().and_then(infix_binding_power).is_some();
                if has_continuation {
                    expr = self.parse_expr_bp_with_lhs(expr, 0)?;
                }
                let has_semi = self.eat(&Token::Semi).is_some();
                // An `if` / `else if` chain without a final `else` branch produces no value
                // and must remain a statement rather than becoming `Block::tail_expr`.
                let is_value_if = has_continuation || if_expr_has_else(&expr);
                if allow_tail && !has_semi && is_value_if && self.peek() == Some(&Token::RBrace) {
                    return Some(ParsedStmt::TailExpr(expr));
                }
                let span = expr.span();
                Some(ParsedStmt::Stmt(Stmt::Expr {
                    expr,
                    has_semi,
                    span,
                }))
            }
            Some(Token::LBrace) => {
                // Parse statement-level blocks directly so a following `(` or `[` on the next
                // statement is never misparsed as a postfix call or index, while still allowing
                // member access and infix continuations such as `{ 1 } + 2`.
                let mut expr = Expr::Block(self.parse_block()?);
                let has_continuation = self.peek() == Some(&Token::Dot)
                    || self.peek().and_then(infix_binding_power).is_some();
                if has_continuation {
                    expr = self.parse_expr_bp_with_lhs(expr, 0)?;
                }
                let has_semi = self.eat(&Token::Semi).is_some();
                if allow_tail && !has_semi && self.peek() == Some(&Token::RBrace) {
                    return Some(ParsedStmt::TailExpr(expr));
                }
                let span = expr.span();
                Some(ParsedStmt::Stmt(Stmt::Expr {
                    expr,
                    has_semi,
                    span,
                }))
            }
            _ => {
                let expr = self.parse_expr()?;
                let has_semi = self.eat(&Token::Semi).is_some();
                if allow_tail && !has_semi && self.peek() == Some(&Token::RBrace) {
                    return Some(ParsedStmt::TailExpr(expr));
                }
                if !has_semi {
                    self.expect(&Token::Semi, "`;`");
                }
                let span = expr.span();
                Some(ParsedStmt::Stmt(Stmt::Expr {
                    expr,
                    has_semi,
                    span,
                }))
            }
        }
    }

    fn parse_let_stmt(&mut self) -> Option<Stmt> {
        let start = self.expect(&Token::KwLet, "`let`")?;
        let is_mut = self.eat(&Token::KwMut).is_some();
        let name = self.parse_ident()?;
        let ty = if self.eat(&Token::Colon).is_some() {
            self.parse_type_expr()
        } else {
            None
        };
        let init = if self.eat(&Token::Eq).is_some() {
            self.parse_expr()
        } else {
            None
        };
        if ty.is_none() && init.is_none() {
            self.diagnostics.push(Diagnostic::error(
                "E0100",
                "local declaration requires a type or initializer",
                name.span,
            ));
        }
        let end = self.expect(&Token::Semi, "`;`").unwrap_or(name.span);
        Some(Stmt::Let {
            is_mut,
            name,
            ty,
            init,
            span: start.merge(end),
        })
    }

    fn parse_return_stmt(&mut self) -> Option<Stmt> {
        let start = self.expect(&Token::KwReturn, "`return`")?;
        let value = if self.peek() != Some(&Token::Semi) && self.peek() != Some(&Token::RBrace) {
            self.parse_expr()
        } else {
            None
        };
        let end = self.expect(&Token::Semi, "`;`").unwrap_or(start);
        Some(Stmt::Return {
            value,
            span: start.merge(end),
        })
    }

    fn parse_for_stmt(&mut self, attrs: Vec<Attribute>) -> Option<Stmt> {
        let start = self.expect(&Token::KwFor, "`for`")?;
        let binding = self.parse_ident()?;
        self.expect(&Token::KwIn, "`in`")?;
        let iter = self.parse_expr_no_struct_lit()?;
        let body = self.parse_block()?;
        let span = start.merge(body.span);
        Some(Stmt::For {
            attrs,
            binding,
            iter,
            body,
            span,
        })
    }

    fn parse_while_stmt(&mut self, attrs: Vec<Attribute>) -> Option<Stmt> {
        let start = self.expect(&Token::KwWhile, "`while`")?;
        let cond = self.parse_expr_no_struct_lit()?;
        let body = self.parse_block()?;
        let span = start.merge(body.span);
        Some(Stmt::While {
            attrs,
            cond,
            body,
            span,
        })
    }

    fn parse_loop_stmt(&mut self, attrs: Vec<Attribute>) -> Option<Stmt> {
        let start = self.expect(&Token::KwLoop, "`loop`")?;
        let body = self.parse_block()?;
        let span = start.merge(body.span);
        Some(Stmt::Loop { attrs, body, span })
    }

    fn parse_switch_stmt(&mut self) -> Option<Stmt> {
        let start = self.expect(&Token::KwSwitch, "`switch`")?;
        let selector = self.parse_expr_no_struct_lit()?;
        self.expect(&Token::LBrace, "`{`")?;
        let mut cases = Vec::new();
        let mut default = None;

        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            if let Some(case_start) = self.eat(&Token::KwCase) {
                let value = self.parse_expr_no_struct_lit()?;
                self.expect(&Token::Colon, "`:`")?;
                let body = self.parse_switch_arm_body(case_start)?;
                let span = case_start.merge(body.span);
                cases.push(SwitchCase { value, body, span });
            } else if let Some(def_start) = self.eat(&Token::KwDefault) {
                self.expect(&Token::Colon, "`:`")?;
                let body = self.parse_switch_arm_body(def_start);
                if default.is_some() {
                    self.diagnostics.push(Diagnostic::error(
                        "E0108",
                        "duplicate default arm in switch",
                        def_start,
                    ));
                } else {
                    default = body;
                }
            } else {
                let span = self.current_span();
                self.diagnostics.push(Diagnostic::error(
                    "E0108",
                    "expected `case` or `default` inside `switch` block",
                    span,
                ));
                self.bump();
            }
        }

        let end = self.expect(&Token::RBrace, "`}`").unwrap_or(start);
        Some(Stmt::Switch {
            selector,
            cases,
            default,
            span: start.merge(end),
        })
    }

    fn parse_switch_arm_body(&mut self, start: Span) -> Option<Block> {
        if self.peek() == Some(&Token::LBrace) {
            return self.parse_block();
        }
        let mut stmts = Vec::new();
        let mut end = start;
        while !self.is_eof()
            && !matches!(
                self.peek(),
                Some(Token::KwCase | Token::KwDefault | Token::RBrace)
            )
        {
            let before = self.pos;
            if let Some(ParsedStmt::Stmt(stmt)) = self.parse_stmt(false) {
                end = stmt_span(&stmt);
                stmts.push(stmt);
            }
            if self.pos == before {
                self.bump();
            }
        }
        Some(Block {
            stmts,
            tail_expr: None,
            span: start.merge(end),
        })
    }

    fn parse_expr_no_struct_lit(&mut self) -> Option<Expr> {
        let prev = self.allow_struct_lit;
        self.allow_struct_lit = false;
        let res = self.parse_expr();
        self.allow_struct_lit = prev;
        res
    }

    fn parse_expr(&mut self) -> Option<Expr> {
        self.parse_expr_bp(0)
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> Option<Expr> {
        self.with_nesting(|parser| {
            let lhs = parser.parse_prefix_expr()?;
            parser.parse_expr_bp_with_lhs(lhs, min_bp)
        })
    }

    fn parse_expr_bp_with_lhs(&mut self, mut lhs: Expr, min_bp: u8) -> Option<Expr> {
        let mut chain_length = 0;
        loop {
            let continues = matches!(
                self.peek(),
                Some(Token::Dot | Token::LParen | Token::LBracket)
            ) || self
                .peek()
                .and_then(infix_binding_power)
                .is_some_and(|(_, bp, _)| bp >= min_bp);
            if !continues {
                break;
            }
            if chain_length >= 256 {
                self.diagnostics.push(Diagnostic::error(
                    "E0110",
                    "expression chain limit exceeded",
                    self.current_span(),
                ));
                return None;
            }
            chain_length += 1;
            match self.peek() {
                Some(Token::Dot) => {
                    self.bump();
                    let field = self.parse_ident()?;
                    let (turbofish, turbofish_span) = if self.peek() == Some(&Token::ColonColon)
                        && self.peek_n(1) == Some(&Token::Lt)
                    {
                        let tf_start = self.bump()?.span;
                        self.bump();
                        let args = self.parse_type_args();
                        let tf_end = self.expect_gt().unwrap_or(tf_start);
                        (args, Some(tf_start.merge(tf_end)))
                    } else {
                        (Vec::new(), None)
                    };
                    if self.peek() == Some(&Token::LParen) {
                        let (args, end) = self.parse_call_args();
                        let span = lhs.span().merge(end);
                        lhs = Expr::MethodCall {
                            receiver: Box::new(lhs),
                            method: field,
                            turbofish,
                            args,
                            span,
                        };
                    } else {
                        if let Some(tf_span) = turbofish_span {
                            self.diagnostics.push(
                                Diagnostic::error(
                                    "E0109",
                                    "generic arguments are not allowed on field access; expected `(` for a method call",
                                    tf_span,
                                )
                                .with_primary_label(
                                    tf_span,
                                    "unexpected generic arguments on field access",
                                ),
                            );
                        }
                        let span = lhs.span().merge(field.span);
                        lhs = Expr::Field {
                            receiver: Box::new(lhs),
                            field,
                            span,
                        };
                    }
                    continue;
                }
                Some(Token::LParen) => {
                    let (args, end) = self.parse_call_args();
                    let span = lhs.span().merge(end);
                    lhs = Expr::Call {
                        callee: Box::new(lhs),
                        args,
                        span,
                    };
                    continue;
                }
                Some(Token::LBracket) => {
                    self.bump();
                    let index = self.parse_expr()?;
                    let end = self.expect(&Token::RBracket, "`]`").unwrap_or(index.span());
                    let span = lhs.span().merge(end);
                    lhs = Expr::Index {
                        receiver: Box::new(lhs),
                        index: Box::new(index),
                        span,
                    };
                    continue;
                }
                _ => {}
            }

            let Some((op, l_bp, r_bp)) = self.peek().and_then(infix_binding_power) else {
                break;
            };
            if l_bp < min_bp {
                break;
            }
            self.bump();
            let rhs = self.parse_expr_bp(r_bp)?;
            let span = lhs.span().merge(rhs.span());
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span,
            };
        }

        Some(lhs)
    }

    fn parse_prefix_expr(&mut self) -> Option<Expr> {
        // Prefix binding power 23 sits strictly above Mul/Div/Rem (21) so `-a * b` parses as `(-a) * b`.
        const PREFIX_BP: u8 = 23;

        match self.peek() {
            Some(Token::Minus) => {
                let s = self.bump()?.span;
                let operand = self.parse_expr_bp(PREFIX_BP)?;
                let span = s.merge(operand.span());
                Some(Expr::Unary {
                    op: UnaryOp::Neg,
                    operand: Box::new(operand),
                    span,
                })
            }
            Some(Token::Bang) => {
                let s = self.bump()?.span;
                let operand = self.parse_expr_bp(PREFIX_BP)?;
                let span = s.merge(operand.span());
                Some(Expr::Unary {
                    op: UnaryOp::Not,
                    operand: Box::new(operand),
                    span,
                })
            }
            Some(Token::Tilde) => {
                let s = self.bump()?.span;
                let operand = self.parse_expr_bp(PREFIX_BP)?;
                let span = s.merge(operand.span());
                Some(Expr::Unary {
                    op: UnaryOp::BitNot,
                    operand: Box::new(operand),
                    span,
                })
            }
            Some(Token::Amp) => {
                let s = self.bump()?.span;
                let operand = self.parse_expr_bp(PREFIX_BP)?;
                let span = s.merge(operand.span());
                Some(Expr::Unary {
                    op: UnaryOp::AddrOf,
                    operand: Box::new(operand),
                    span,
                })
            }
            Some(Token::IntLit(_) | Token::HexLit(_)) => {
                let tok = self.bump()?;
                let (Token::IntLit(raw) | Token::HexLit(raw)) = tok.token else {
                    unreachable!()
                };
                Some(Expr::IntLit(raw, tok.span))
            }
            Some(Token::FloatLit(_)) => {
                let tok = self.bump()?;
                let Token::FloatLit(v) = tok.token else {
                    unreachable!()
                };
                Some(Expr::FloatLit(v, tok.span))
            }
            Some(Token::KwTrue) => Some(Expr::BoolLit(true, self.bump()?.span)),
            Some(Token::KwFalse) => Some(Expr::BoolLit(false, self.bump()?.span)),
            Some(Token::KwNone) => Some(Expr::NoneLit(self.bump()?.span)),
            Some(Token::StringLit(_)) => {
                let (s, span) = self.parse_string_lit()?;
                Some(Expr::StringLit(s, span))
            }
            Some(Token::LParen) => {
                self.bump();
                let prev = self.allow_struct_lit;
                self.allow_struct_lit = true;
                let inner = self.parse_expr();
                self.allow_struct_lit = prev;
                self.expect(&Token::RParen, "`)`");
                inner
            }
            Some(Token::LBracket) => {
                let start = self.bump()?.span;
                let mut elements = Vec::new();
                while !self.is_eof() && self.peek() != Some(&Token::RBracket) {
                    if let Some(el) = self.parse_expr() {
                        elements.push(el);
                    } else {
                        break;
                    }
                    if self.eat(&Token::Comma).is_none() {
                        break;
                    }
                }
                let end = self.expect(&Token::RBracket, "`]`").unwrap_or(start);
                Some(Expr::ArrayLit {
                    elements,
                    span: start.merge(end),
                })
            }
            Some(Token::KwIf) => self.parse_if_expr(),
            Some(Token::LBrace) => self.parse_block().map(Expr::Block),
            tok if Self::is_contextual_ident(tok) => {
                let path = self.parse_path()?;
                let mut span = path.span;
                let turbofish = if self.peek() == Some(&Token::ColonColon)
                    && self.peek_n(1) == Some(&Token::Lt)
                {
                    self.bump();
                    self.bump();
                    let args = self.parse_type_args();
                    if let Some(end) = self.expect_gt() {
                        span = span.merge(end);
                    }
                    args
                } else {
                    Vec::new()
                };
                // Allow turbofish on generic struct literals such as
                // `Reservoir::<LightSample> { ... }`. `Expr::StructLit` keeps its existing
                // fields for downstream compatibility with `vsl-hir`, which destructures
                // `Expr::StructLit { path, fields, span }` without `..`.
                if self.allow_struct_lit && self.peek() == Some(&Token::LBrace) {
                    self.bump(); // `{`
                    let mut fields = Vec::new();
                    while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
                        let Some(field_name) = self.parse_ident() else {
                            break;
                        };
                        self.expect(&Token::Colon, "`:`");
                        let Some(field_val) = self.parse_expr() else {
                            break;
                        };
                        fields.push((field_name, field_val));
                        if self.eat(&Token::Comma).is_none() {
                            break;
                        }
                    }
                    let end = self.expect(&Token::RBrace, "`}`").unwrap_or(span);
                    return Some(Expr::StructLit {
                        path,
                        fields,
                        span: span.merge(end),
                    });
                }
                Some(Expr::Path {
                    path,
                    turbofish,
                    span,
                })
            }
            _ => {
                let span = self.current_span();
                self.diagnostics.push(
                    Diagnostic::error("E0105", "expected expression", span)
                        .with_primary_label(span, "expected expression here"),
                );
                None
            }
        }
    }

    fn parse_if_expr(&mut self) -> Option<Expr> {
        self.with_nesting(Self::parse_if_expr_inner)
    }

    fn parse_if_expr_inner(&mut self) -> Option<Expr> {
        let start = self.expect(&Token::KwIf, "`if`")?;
        let cond = self.parse_expr_no_struct_lit()?;
        let then_branch = self.parse_block()?;
        let mut span = start.merge(then_branch.span);
        let else_branch = if self.eat(&Token::KwElse).is_some() {
            let else_expr = if self.peek() == Some(&Token::KwIf) {
                self.parse_if_expr()?
            } else {
                Expr::Block(self.parse_block()?)
            };
            span = span.merge(else_expr.span());
            Some(Box::new(else_expr))
        } else {
            None
        };
        Some(Expr::If {
            cond: Box::new(cond),
            then_branch,
            else_branch,
            span,
        })
    }

    fn parse_call_args(&mut self) -> (Vec<CallArg>, Span) {
        let start = self
            .expect(&Token::LParen, "`(`")
            .unwrap_or_else(|| self.current_span());
        let mut args = Vec::new();
        while !self.is_eof() && self.peek() != Some(&Token::RParen) {
            let is_named =
                Self::is_contextual_ident(self.peek()) && self.peek_n(1) == Some(&Token::Eq);

            if is_named {
                let Some(name) = self.parse_ident() else {
                    break;
                };
                self.bump(); // `=`
                if let Some(value) = self.parse_expr() {
                    let span = name.span.merge(value.span());
                    args.push(CallArg {
                        name: Some(name),
                        value,
                        span,
                    });
                }
            } else if let Some(value) = self.parse_expr() {
                let span = value.span();
                args.push(CallArg {
                    name: None,
                    value,
                    span,
                });
            } else {
                break;
            }
            if self.eat(&Token::Comma).is_none() {
                break;
            }
        }
        let end = self.expect(&Token::RParen, "`)`").unwrap_or(start);
        (args, end)
    }

    // ========================================================================
    // `pack.vsl` DSL Parsers
    // ========================================================================

    fn parse_key_value_fields(&mut self, closing: &Token) -> Vec<PackField> {
        let mut fields = Vec::new();
        while !self.is_eof() && self.peek() != Some(closing) {
            let Some(key) = self.parse_ident() else {
                break;
            };
            if self.eat(&Token::Colon).is_none() && self.eat(&Token::Eq).is_none() {
                self.expect(&Token::Colon, "`:` or `=`");
                break;
            }
            let Some(value) = self.parse_expr() else {
                break;
            };
            let span = key.span.merge(value.span());
            fields.push(PackField { key, value, span });
            if self.eat(&Token::Comma).is_none() {
                break;
            }
        }
        fields
    }

    fn parse_pack_decl(&mut self) -> Option<PackDecl> {
        let start = self.expect(&Token::KwPack, "`pack`")?;
        let (id, _) = self.parse_string_lit()?;
        self.expect(&Token::LBrace, "`{`")?;
        let fields = self.parse_key_value_fields(&Token::RBrace);
        let end = self.expect(&Token::RBrace, "`}`").unwrap_or(start);
        Some(PackDecl {
            id,
            fields,
            span: start.merge(end),
        })
    }

    fn parse_settings_decl(&mut self) -> Option<SettingsDecl> {
        let start = self.expect(&Token::KwSettings, "`settings`")?;
        self.expect(&Token::LBrace, "`{`")?;
        let mut categories = Vec::new();
        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            let cat_start = self.expect(&Token::KwCategory, "`category`")?;
            let (name, _) = self.parse_string_lit()?;
            self.expect(&Token::LBrace, "`{`")?;
            let mut entries = Vec::new();
            while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
                if let Some(entry) = self.parse_setting_entry() {
                    entries.push(entry);
                } else {
                    break;
                }
            }
            let cat_end = self.expect(&Token::RBrace, "`}`").unwrap_or(cat_start);
            categories.push(SettingCategoryDecl {
                name,
                entries,
                span: cat_start.merge(cat_end),
            });
        }
        let end = self.expect(&Token::RBrace, "`}`").unwrap_or(start);
        Some(SettingsDecl {
            categories,
            span: start.merge(end),
        })
    }

    fn parse_setting_entry(&mut self) -> Option<SettingEntryDecl> {
        let (kind, start) = match self.peek() {
            Some(Token::KwParam) => (SettingKind::Param, self.bump()?.span),
            Some(Token::KwSpec) => (SettingKind::Spec, self.bump()?.span),
            _ => {
                let span = self.current_span();
                self.diagnostics.push(Diagnostic::error(
                    "E0106",
                    "expected `param` or `spec` inside `category`",
                    span,
                ));
                return None;
            }
        };
        let name = self.parse_ident()?;
        self.expect(&Token::Colon, "`:`")?;
        let ty = if let Some(enum_start) = self.eat(&Token::KwEnum) {
            self.expect(&Token::LBrace, "`{`")?;
            let mut variants = Vec::new();
            while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
                if let Some(v) = self.parse_ident() {
                    variants.push(v);
                } else {
                    break;
                }
                if self.eat(&Token::Comma).is_none() {
                    break;
                }
            }
            let enum_end = self.expect(&Token::RBrace, "`}`").unwrap_or(enum_start);
            SettingTypeAst::Enum {
                variants,
                span: enum_start.merge(enum_end),
            }
        } else {
            SettingTypeAst::Named(self.parse_ident()?)
        };

        self.expect(&Token::Eq, "`=`")?;
        // The following `{ ... }` is entry metadata, not a struct literal.
        let default_value = self.parse_expr_no_struct_lit()?;
        let mut end = default_value.span();
        let metadata = if self.eat(&Token::LBrace).is_some() {
            let fields = self.parse_key_value_fields(&Token::RBrace);
            end = self.expect(&Token::RBrace, "`}`").unwrap_or(end);
            fields
        } else {
            Vec::new()
        };
        let _ = self.eat(&Token::Semi);

        Some(SettingEntryDecl {
            kind,
            name,
            ty,
            default_value,
            metadata,
            span: start.merge(end),
        })
    }

    fn parse_presets_decl(&mut self) -> Option<PresetsDecl> {
        let start = self.expect(&Token::KwPresets, "`presets`")?;
        self.expect(&Token::LBrace, "`{`")?;
        let mut presets = Vec::new();
        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            let p_start = self.expect(&Token::KwPreset, "`preset`")?;
            let name = self.parse_ident()?;
            let metadata = if self.eat(&Token::LParen).is_some() {
                let m = self.parse_key_value_fields(&Token::RParen);
                self.expect(&Token::RParen, "`)`");
                m
            } else {
                Vec::new()
            };
            self.expect(&Token::LBrace, "`{`")?;
            let mut assignments = Vec::new();
            while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
                let Some(key) = self.parse_ident() else {
                    break;
                };
                self.expect(&Token::Eq, "`=`")?;
                let Some(val) = self.parse_expr() else {
                    break;
                };
                let _ = self.eat(&Token::Semi);
                assignments.push((key, val));
            }
            let p_end = self.expect(&Token::RBrace, "`}`").unwrap_or(p_start);
            presets.push(PresetDecl {
                name,
                metadata,
                assignments,
                span: p_start.merge(p_end),
            });
        }
        let end = self.expect(&Token::RBrace, "`}`").unwrap_or(start);
        Some(PresetsDecl {
            presets,
            span: start.merge(end),
        })
    }

    fn parse_pipeline_decl(&mut self) -> Option<PipelineDecl> {
        let start = self.expect(&Token::KwPipeline, "`pipeline`")?;
        let mode = if self.eat(&Token::KwExtend).is_some() {
            PipelineModeAst::Extend
        } else if self.eat(&Token::KwCustom).is_some() {
            PipelineModeAst::Custom
        } else {
            let span = self.current_span();
            self.diagnostics.push(Diagnostic::error(
                "E0107",
                "expected `extend` or `custom` after `pipeline`",
                span,
            ));
            return None;
        };
        let params = self.parse_fn_params();
        let body = self.parse_block()?;
        let span = start.merge(body.span);
        Some(PipelineDecl {
            mode,
            params,
            body,
            span,
        })
    }
}

const fn stmt_span(stmt: &Stmt) -> Span {
    match stmt {
        Stmt::Let { span, .. }
        | Stmt::Return { span, .. }
        | Stmt::For { span, .. }
        | Stmt::While { span, .. }
        | Stmt::Loop { span, .. }
        | Stmt::Switch { span, .. }
        | Stmt::Break(span)
        | Stmt::Continue(span)
        | Stmt::Expr { span, .. } => *span,
    }
}

fn if_expr_has_else(expr: &Expr) -> bool {
    match expr {
        Expr::If {
            else_branch: Some(eb),
            ..
        } => match eb.as_ref() {
            Expr::If { .. } => if_expr_has_else(eb),
            _ => true,
        },
        _ => false,
    }
}

fn type_span(ty: &TypeExpr) -> Span {
    match ty {
        TypeExpr::Named { span, .. } | TypeExpr::Array { span, .. } => *span,
    }
}

fn infix_binding_power(token: &Token) -> Option<(BinaryOp, u8, u8)> {
    let bp = match token {
        Token::Eq => (BinaryOp::Assign, 2, 1),
        Token::PlusEq => (BinaryOp::AddAssign, 2, 1),
        Token::MinusEq => (BinaryOp::SubAssign, 2, 1),
        Token::StarEq => (BinaryOp::MulAssign, 2, 1),
        Token::SlashEq => (BinaryOp::DivAssign, 2, 1),
        Token::PercentEq => (BinaryOp::RemAssign, 2, 1),
        Token::AmpEq => (BinaryOp::BitAndAssign, 2, 1),
        Token::PipeEq => (BinaryOp::BitOrAssign, 2, 1),
        Token::CaretEq => (BinaryOp::BitXorAssign, 2, 1),
        Token::DotDot => (BinaryOp::Range, 4, 5),
        Token::DotDotEq => (BinaryOp::RangeInclusive, 4, 5),
        Token::PipePipe => (BinaryOp::Or, 6, 7),
        Token::AmpAmp => (BinaryOp::And, 8, 9),
        Token::EqEq => (BinaryOp::Eq, 9, 10),
        Token::BangEq => (BinaryOp::Ne, 9, 10),
        Token::Lt => (BinaryOp::Lt, 11, 12),
        Token::LtEq => (BinaryOp::Le, 11, 12),
        Token::Gt => (BinaryOp::Gt, 11, 12),
        Token::GtEq => (BinaryOp::Ge, 11, 12),
        Token::Pipe => (BinaryOp::BitOr, 12, 13),
        Token::Caret => (BinaryOp::BitXor, 14, 15),
        Token::Amp => (BinaryOp::BitAnd, 16, 17),
        Token::Shl => (BinaryOp::Shl, 18, 19),
        Token::Shr => (BinaryOp::Shr, 18, 19),
        Token::Plus => (BinaryOp::Add, 19, 20),
        Token::Minus => (BinaryOp::Sub, 19, 20),
        Token::Star => (BinaryOp::Mul, 21, 22),
        Token::Slash => (BinaryOp::Div, 21, 22),
        Token::Percent => (BinaryOp::Rem, 21, 22),
        _ => return None,
    };
    Some(bp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_incomplete_declarations_and_recovers_following_functions() {
        let parsed = parse_source(FileId(0), "fn   ");
        assert!(parsed.has_errors());
        assert_eq!(
            parsed.diagnostics[0].primary_span,
            Span::new(FileId(0), 5, 5)
        );

        let parsed = parse_source(FileId(0), "fn first() { let x = 1u; fn second() {}");
        assert!(parsed.has_errors());
        assert!(
            parsed
                .unit
                .items
                .iter()
                .any(|item| matches!(item, Item::Function(f) if f.name.name == "second"))
        );

        for source in [
            "fn bad() { 1u 2u; }",
            "fn bad() { let a; }",
            "fn bad(x: u32) { switch x { default: break; default: break; } }",
        ] {
            assert!(parse_source(FileId(0), source).has_errors(), "{source}");
        }
    }

    #[test]
    fn rejects_deeply_nested_and_unbounded_expression_chains() {
        let deep = format!(
            "fn f() {{ let a = {}1u{}; }}",
            "(".repeat(500),
            ")".repeat(500)
        );
        let chain = format!("fn f() {{ let a = 0u{}; }}", " + 1u".repeat(500));
        for source in [deep, chain] {
            let parsed = parse_source(FileId(0), &source);
            assert!(parsed.diagnostics.iter().any(|diag| diag.code == "E0110"));
        }
    }

    #[test]
    fn parses_rayquery_compute_shader() {
        let src = r"
            module pack::reflections;
            import mc::{world, tlas, bindless};

            #[tier(2..=3)]
            #[cap(ray_query)]
            @stage(compute(8, 8, 1))
            pub fn trace_glossy_reflections(
                #[builtin(global_invocation_id)] gid: uvec3,
                gbuffer_normal: Texture2D<f32>,
                out_refl: RWImage2D<rgba16f>,
            ) {
                let mut rq: RayQuery;
                while rq.proceed() {}
                out_refl.store(ivec2(gid.xy), vec4(1.0));
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
        assert_eq!(parsed.unit.items.len(), 1);
    }

    #[test]
    fn parses_nested_generics_and_trait_impl() {
        let src = r"
            pub trait ResampleTarget {
                fn target_pdf(self, world_pos: vec3) -> f32;
            }
            impl ResampleTarget for LightSample {
                fn target_pdf(self, world_pos: vec3) -> f32 {
                    return -1.0 * 2.0;
                }
            }
            pub fn test_nested(p: ptr<Reservoir<LightSample>>) {}
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
        assert_eq!(parsed.unit.items.len(), 3);
    }

    #[test]
    fn parses_pack_vsl_manifest() {
        let src = r#"
            pack "hyaloclast-sunlit-vanilla" {
                name: "Sunlit Vanilla",
                version: "2.0.0",
                tiers: 0..=3,
            }
            settings {
                category "Lighting" {
                    param sun_intensity: f32 = 2.75 {
                        label: "Sun Intensity",
                        range: 0.5..=8.0,
                    }
                }
            }
            presets {
                preset high (order = 2, tiers: 0..=3, label: "High") {
                    sun_intensity = 3.0;
                }
            }
            pipeline extend(g: mut mc::GraphBuilder, cfg: mc::PipelineConfig) {
                g.hook(clouds, hooks::clouds::vanilla_volumetric_clouds);
            }
        "#;
        let parsed = parse_source(FileId(0), src);
        assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
        assert_eq!(parsed.unit.items.len(), 4);
    }

    #[test]
    fn parses_non_block_switch_arm_with_loops_and_attributes() {
        let src = r"
            pub fn dispatch(mode: u32, mut acc: u32) -> u32 {
                switch mode {
                    case 0u:
                        #[unroll]
                        for i in 0u..4u {
                            acc += i;
                        }
                        while acc < 10u {
                            acc += 1u;
                        }
                        break;
                    default:
                        loop {
                            break;
                        }
                        switch acc {
                            case 0u:
                                acc = 1u;
                                break;
                            default:
                                break;
                        }
                        break;
                }
                return acc;
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
        assert_eq!(parsed.unit.items.len(), 1);

        let Item::Function(func) = &parsed.unit.items[0] else {
            panic!("expected function item");
        };
        let body = func.body.as_ref().expect("function body");
        let Stmt::Switch { cases, default, .. } = &body.stmts[0] else {
            panic!("expected switch statement");
        };
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].body.stmts.len(), 3);
        assert!(matches!(
            &cases[0].body.stmts[0],
            Stmt::For { attrs, .. } if attrs.len() == 1
        ));
        assert!(matches!(&cases[0].body.stmts[1], Stmt::While { .. }));
        assert!(matches!(&cases[0].body.stmts[2], Stmt::Break(_)));

        let def_block = default.as_ref().expect("default arm");
        assert_eq!(def_block.stmts.len(), 3);
        assert!(matches!(&def_block.stmts[0], Stmt::Loop { .. }));
        assert!(matches!(&def_block.stmts[1], Stmt::Switch { .. }));
        assert!(matches!(&def_block.stmts[2], Stmt::Break(_)));
    }

    #[test]
    fn parses_if_block_followed_by_array_literal_as_two_statements() {
        let src = r"
            pub fn branch_then_array(c: bool) -> [u32; 2] {
                if c {
                    let x = 1u;
                }
                [1u, 2u]
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
        assert_eq!(parsed.unit.items.len(), 1);

        let Item::Function(func) = &parsed.unit.items[0] else {
            panic!("expected function item");
        };
        let body = func.body.as_ref().expect("function body");
        assert_eq!(body.stmts.len(), 1);
        assert!(matches!(
            &body.stmts[0],
            Stmt::Expr {
                expr: Expr::If { .. },
                ..
            }
        ));
        assert!(matches!(
            body.tail_expr.as_deref(),
            Some(Expr::ArrayLit { elements, .. }) if elements.len() == 2
        ));
    }

    #[test]
    fn else_less_if_does_not_become_tail_expr() {
        let src = r"
            pub fn only_if(c: bool) {
                if c {
                    let x = 1u;
                }
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
        let Item::Function(func) = &parsed.unit.items[0] else {
            panic!("expected function item");
        };
        let body = func.body.as_ref().expect("function body");
        assert!(
            body.tail_expr.is_none(),
            "else-less `if` must not become tail_expr"
        );
        assert_eq!(body.stmts.len(), 1);
        assert!(matches!(
            &body.stmts[0],
            Stmt::Expr {
                expr: Expr::If {
                    else_branch: None,
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn parses_nested_generic_with_named_align_closing_shr() {
        let src = r"
            pub struct Wrapper {
                p: Outer<ptr<u32, align = 16>>,
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
        let Item::Struct(st) = &parsed.unit.items[0] else {
            panic!("expected struct item");
        };
        assert_eq!(st.fields.len(), 1);
        let TypeExpr::Named { path, args, .. } = &st.fields[0].ty else {
            panic!("expected Outer<...>");
        };
        assert_eq!(path.joined(), "Outer");
        assert_eq!(args.len(), 1);
        let TypeArg::Type(TypeExpr::Named {
            path: inner_path,
            args: inner_args,
            ..
        }) = &args[0]
        else {
            panic!("expected nested ptr<u32, align = 16>");
        };
        assert_eq!(inner_path.joined(), "ptr");
        assert_eq!(inner_args.len(), 2);
        assert!(matches!(
            &inner_args[1],
            TypeArg::NamedExpr { name, .. } if name.name == "align"
        ));
    }

    #[test]
    fn parses_standalone_block_followed_by_bracket_and_paren() {
        let src = r"
            pub fn blocks_then_exprs(a: u32) -> [u32; 2] {
                {
                    let x = a;
                }
                (a + 1u);
                {
                    let y = a;
                }
                [1u, 2u]
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);

        let Item::Function(func) = &parsed.unit.items[0] else {
            panic!("expected function item");
        };
        let body = func.body.as_ref().expect("function body");
        assert_eq!(body.stmts.len(), 3);
        assert!(matches!(
            &body.stmts[0],
            Stmt::Expr {
                expr: Expr::Block(_),
                ..
            }
        ));
        assert!(matches!(
            &body.stmts[1],
            Stmt::Expr {
                expr: Expr::Binary { .. },
                has_semi: true,
                ..
            }
        ));
        assert!(matches!(
            &body.stmts[2],
            Stmt::Expr {
                expr: Expr::Block(_),
                ..
            }
        ));
        assert!(matches!(
            body.tail_expr.as_deref(),
            Some(Expr::ArrayLit { elements, .. }) if elements.len() == 2
        ));
    }

    #[test]
    fn rejects_turbofish_on_non_call_field_access() {
        let src = r"
            pub fn bad_field(x: Foo) {
                let y = x.field::<f32>;
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(parsed.has_errors());
        assert!(parsed.diagnostics.iter().any(|d| d.code == "E0109"));
    }

    #[test]
    fn parses_generic_struct_literal_with_turbofish() {
        let src = r"
            pub fn make_res(s: LightSample) -> Reservoir<LightSample> {
                Reservoir::<LightSample> { sample: s }
            }
        ";
        let parsed = parse_source(FileId(0), src);
        assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);

        let Item::Function(func) = &parsed.unit.items[0] else {
            panic!("expected function item");
        };
        let body = func.body.as_ref().expect("function body");
        let Some(Expr::StructLit { path, fields, .. }) = body.tail_expr.as_deref() else {
            panic!("expected struct literal tail expression");
        };
        assert_eq!(path.joined(), "Reservoir");
        assert_eq!(fields.len(), 1);
    }
}
