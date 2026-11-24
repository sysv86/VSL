use std::collections::HashMap;
use tokio::sync::RwLock;
use tower_lsp_server::jsonrpc::Result;
use tower_lsp_server::ls_types::{
    CompletionItem, CompletionItemKind, CompletionOptions, CompletionParams, CompletionResponse,
    Diagnostic as LspDiagnostic, DiagnosticSeverity, DidChangeTextDocumentParams,
    DidCloseTextDocumentParams, DidOpenTextDocumentParams, Hover, HoverContents, HoverParams,
    HoverProviderCapability, InitializeParams, InitializeResult, InitializedParams, MarkupContent,
    MarkupKind, MessageType, Position, PositionEncodingKind, Range, ServerCapabilities,
    TextDocumentSyncCapability, TextDocumentSyncKind, Uri,
};
use tower_lsp_server::{Client, LanguageServer, LspService, Server};
use vsl_diagnostics::{Diagnostic, FileId, Severity, SourceFile, Span};
use vsl_hir::validate_unit;
use vsl_mc::{BUILTIN_RESOURCES, BuiltinResourceId, CANONICAL_PASSES, CanonicalPass, HookStage};
use vsl_pack::{EvalConfig, evaluate_pack_vsl, validate_and_schedule};
use vsl_syntax::parse_source;

#[derive(Debug)]
struct VslLanguageServer {
    client: Client,
    documents: RwLock<HashMap<String, String>>,
}

impl VslLanguageServer {
    fn new(client: Client) -> Self {
        Self {
            client,
            documents: RwLock::new(HashMap::new()),
        }
    }

    async fn publish_file_diagnostics(&self, uri: Uri, text: &str) {
        self.documents
            .write()
            .await
            .insert(uri.as_str().to_owned(), text.to_owned());

        let file_id = FileId(0);
        let source_file = SourceFile::new(file_id, uri.as_str(), text);
        let parsed = parse_source(file_id, text);
        let mut all_diags = parsed.diagnostics;

        if is_pack_vsl_uri(uri.as_str()) {
            let (ir, eval_diags) = evaluate_pack_vsl(&parsed.unit, &EvalConfig::default());
            all_diags.extend(eval_diags.into_vec());
            if let Err(sched_errors) = validate_and_schedule(&ir) {
                for msg in sched_errors {
                    all_diags.push(Diagnostic::error("E0450", msg, Span::new(file_id, 0, 0)));
                }
            }
        } else {
            // Validate across all tiers without pruning `#[tier(0..=1)]` functions in the editor.
            let (_, hir_diags) = validate_unit(&parsed.unit);
            all_diags.extend(hir_diags.into_vec());
        }

        let lsp_diags = all_diags
            .iter()
            .map(|d| to_lsp_diagnostic(&source_file, d))
            .collect();

        self.client.publish_diagnostics(uri, lsp_diags, None).await;
    }
}

/// Returns `true` only when the URI basename is strictly `pack.vsl` (never `unpack.vsl`).
fn is_pack_vsl_uri(uri: &str) -> bool {
    let path = uri.split(['?', '#']).next().unwrap_or(uri);
    path == "pack.vsl" || path.ends_with("/pack.vsl") || path.ends_with("\\pack.vsl")
}

fn to_lsp_diagnostic(file: &SourceFile, diag: &Diagnostic) -> LspDiagnostic {
    let start = file.line_col_utf16(diag.primary_span.start);
    let end = file.line_col_utf16(diag.primary_span.end);
    let severity = match diag.severity {
        Severity::Error => DiagnosticSeverity::ERROR,
        Severity::Warning => DiagnosticSeverity::WARNING,
        Severity::Advice => DiagnosticSeverity::HINT,
    };

    LspDiagnostic {
        range: Range {
            start: Position {
                line: start.line,
                character: start.col,
            },
            end: Position {
                line: end.line,
                character: end.col,
            },
        },
        severity: Some(severity),
        source: Some("vsl2".to_owned()),
        message: format!("[{}] {}", diag.code, diag.message),
        ..LspDiagnostic::default()
    }
}

/// Converts an LSP UTF-16 character column into a UTF-8 byte offset within `line`.
fn utf16_col_to_byte_offset(line: &str, utf16_col: u32) -> usize {
    let mut utf16_seen = 0u32;
    for (byte_idx, ch) in line.char_indices() {
        if utf16_seen >= utf16_col {
            return byte_idx;
        }
        utf16_seen = utf16_seen.saturating_add(ch.len_utf16() as u32);
    }
    line.len()
}

/// Extracts `(prefix_before_word, word)` at the given LSP UTF-16 `Position`.
fn word_with_prefix_at_position(text: &str, pos: Position) -> Option<(&str, &str)> {
    let line = text.lines().nth(pos.line as usize)?;
    let bytes = line.as_bytes();
    let idx = utf16_col_to_byte_offset(line, pos.character).min(bytes.len());
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';

    let mut start = idx;
    while start > 0 && is_ident(bytes[start - 1]) {
        start -= 1;
    }
    let mut end = idx;
    while end < bytes.len() && is_ident(bytes[end]) {
        end += 1;
    }
    if start < end {
        Some((&line[..start], &line[start..end]))
    } else {
        None
    }
}

impl LanguageServer for VslLanguageServer {
    fn initialize(
        &self,
        _: InitializeParams,
    ) -> impl std::future::Future<Output = Result<InitializeResult>> {
        std::future::ready(Ok(InitializeResult {
            capabilities: ServerCapabilities {
                position_encoding: Some(PositionEncodingKind::UTF16),
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                completion_provider: Some(CompletionOptions {
                    trigger_characters: Some(vec![":".to_owned(), ".".to_owned(), "@".to_owned()]),
                    ..CompletionOptions::default()
                }),
                ..ServerCapabilities::default()
            },
            ..InitializeResult::default()
        }))
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "vsl-lsp initialized (VSL 2.0)")
            .await;
    }

    fn shutdown(&self) -> impl std::future::Future<Output = Result<()>> {
        std::future::ready(Ok(()))
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.publish_file_diagnostics(params.text_document.uri, &params.text_document.text)
            .await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        if let Some(change) = params.content_changes.into_iter().last() {
            self.publish_file_diagnostics(params.text_document.uri, &change.text)
                .await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        self.documents
            .write()
            .await
            .remove(params.text_document.uri.as_str());
        self.client
            .publish_diagnostics(params.text_document.uri, Vec::new(), None)
            .await;
    }

    // These suffixes are VSL graph-builder method names, not filesystem extensions.
    #[allow(clippy::case_sensitive_file_extension_comparisons)]
    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;
        let docs = self.documents.read().await;
        let Some(text) = docs.get(uri.as_str()) else {
            return Ok(None);
        };
        let Some((prefix, word)) = word_with_prefix_at_position(text, pos) else {
            return Ok(None);
        };
        let trimmed = prefix.trim_end();

        // Only surface engine hovers in valid namespace / decorator / graph-builder contexts
        // (tolerating whitespace before or after `(`) so unrelated locals are never misidentified.
        let before_paren = trimmed.strip_suffix('(').map(str::trim_end);
        let is_pass_ctx = trimmed.ends_with("mc::passes::")
            || before_paren.is_some_and(|p| {
                p.ends_with(".after") || p.ends_with(".before") || p.ends_with(".replaces")
            });
        let is_target_ctx = trimmed.ends_with("mc::targets::") || word.starts_with("RES_");
        let is_hook_ctx =
            before_paren.is_some_and(|p| p.ends_with("@hook") || p.ends_with(".hook"));

        let markdown = if is_pass_ctx && let Some(pass) = CanonicalPass::from_name(word) {
            format!(
                "**Hyaloclast Canonical Pass** `mc::passes::{}`\n\nRegistered in `vulkanshaders-core::RenderGraph`.",
                pass.as_str()
            )
        } else if is_target_ctx && let Some(res) = BuiltinResourceId::from_name(word) {
            format!(
                "**Hyaloclast Built-in Resource** `mc::targets::{word}`\n\n`ResourceId({})`",
                res.raw_id()
            )
        } else if is_hook_ctx && let Some(hook) = HookStage::from_name(word) {
            format!(
                "**VSL 2.0 Hook Stage** `@hook({})`\n\nLowering ABI: `{:?}`{}",
                hook.as_str(),
                hook.abi_route(),
                if hook.uses_alpha_over_blending() {
                    "\n\nReturns `vec4` (alpha-over composited across chains)."
                } else {
                    ""
                }
            )
        } else {
            return Ok(None);
        };

        Ok(Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: markdown,
            }),
            range: None,
        }))
    }

    fn completion(
        &self,
        _: CompletionParams,
    ) -> impl std::future::Future<Output = Result<Option<CompletionResponse>>> {
        let mut items = Vec::with_capacity(
            CANONICAL_PASSES.len() + HookStage::ALL.len() + BUILTIN_RESOURCES.len(),
        );

        for pass in CANONICAL_PASSES {
            items.push(CompletionItem {
                label: format!("mc::passes::{}", pass.as_str()),
                kind: Some(CompletionItemKind::CONSTANT),
                detail: Some("Hyaloclast canonical RenderGraph pass".to_owned()),
                ..CompletionItem::default()
            });
        }

        for &(target, res) in BUILTIN_RESOURCES {
            items.push(CompletionItem {
                label: format!("mc::targets::{target}"),
                kind: Some(CompletionItemKind::CONSTANT),
                detail: Some(format!(
                    "Hyaloclast RenderGraph ResourceId({})",
                    res.raw_id()
                )),
                ..CompletionItem::default()
            });
        }

        for hook in HookStage::ALL {
            items.push(CompletionItem {
                label: hook.as_str().to_owned(),
                kind: Some(CompletionItemKind::ENUM_MEMBER),
                detail: Some(format!("VSL 2.0 @hook stage ({:?})", hook.abi_route())),
                ..CompletionItem::default()
            });
        }

        std::future::ready(Ok(Some(CompletionResponse::Array(items))))
    }
}

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let (service, socket) = LspService::new(VslLanguageServer::new);
    Server::new(stdin, stdout, socket).serve(service).await;
}

#[cfg(test)]
mod tests {
    use super::is_pack_vsl_uri;

    #[test]
    fn matches_pack_vsl_basename_strictly() {
        assert!(is_pack_vsl_uri("file:///example/pack/pack.vsl"));
        assert!(is_pack_vsl_uri("pack.vsl"));
        assert!(is_pack_vsl_uri("file:///C:/packs/pack.vsl?v=1"));
        assert!(!is_pack_vsl_uri("file:///example/pack/unpack.vsl"));
        assert!(!is_pack_vsl_uri("file:///example/pack/my_pack.vsl"));
    }
}
