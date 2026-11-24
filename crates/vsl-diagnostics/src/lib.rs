//! Source spans, file table, structured diagnostics, and terminal rendering for VSL 2.0.

mod render;
mod span;

pub use render::render_diagnostics;
pub use span::{FileId, LineCol, SourceDatabase, SourceFile, Span};

use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

/// Severity level of a compiler diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Error,
    Warning,
    Advice,
}

/// Secondary or primary source annotation attached to a diagnostic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticLabel {
    pub span: Span,
    pub message: String,
    pub is_primary: bool,
}

/// Structured compiler diagnostic shared across lexer, parser, HIR, `pack.vsl`, IR, and LSP.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: SmolStr,
    pub severity: Severity,
    pub message: String,
    pub primary_span: Span,
    pub labels: Vec<DiagnosticLabel>,
    pub notes: Vec<String>,
}

impl Diagnostic {
    /// Creates an error diagnostic anchored at `span`.
    pub fn error(code: impl Into<SmolStr>, message: impl Into<String>, span: Span) -> Self {
        Self {
            code: code.into(),
            severity: Severity::Error,
            message: message.into(),
            primary_span: span,
            labels: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// Creates a warning diagnostic anchored at `span`.
    pub fn warning(code: impl Into<SmolStr>, message: impl Into<String>, span: Span) -> Self {
        Self {
            code: code.into(),
            severity: Severity::Warning,
            message: message.into(),
            primary_span: span,
            labels: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// Appends a primary highlighted span label.
    #[must_use]
    pub fn with_primary_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push(DiagnosticLabel {
            span,
            message: message.into(),
            is_primary: true,
        });
        self
    }

    /// Appends a secondary contextual span label.
    #[must_use]
    pub fn with_secondary_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push(DiagnosticLabel {
            span,
            message: message.into(),
            is_primary: false,
        });
        self
    }

    /// Appends a trailing explanatory note.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Returns `true` if this diagnostic is an error.
    pub const fn is_error(&self) -> bool {
        matches!(self.severity, Severity::Error)
    }
}

/// Accumulator for diagnostics emitted during a compilation stage.
#[derive(Debug, Default, Clone)]
pub struct DiagnosticBag {
    items: Vec<Diagnostic>,
}

impl DiagnosticBag {
    pub const fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn push(&mut self, diagnostic: Diagnostic) {
        self.items.push(diagnostic);
    }

    pub fn extend(&mut self, other: impl IntoIterator<Item = Diagnostic>) {
        self.items.extend(other);
    }

    pub fn has_errors(&self) -> bool {
        self.items.iter().any(Diagnostic::is_error)
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn as_slice(&self) -> &[Diagnostic] {
        &self.items
    }

    pub fn into_vec(self) -> Vec<Diagnostic> {
        self.items
    }
}
