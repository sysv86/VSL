use crate::{Diagnostic, FileId, Severity, SourceDatabase};
use ariadne::{Cache, Color, Config, IndexType, Label, Report, ReportKind, Source};
use std::collections::HashMap;
use std::fmt;
use std::io;

struct DbCache<'a> {
    db: &'a SourceDatabase,
    sources: HashMap<FileId, Source<&'a str>>,
}

impl<'a> DbCache<'a> {
    fn new(db: &'a SourceDatabase) -> Self {
        let mut sources = HashMap::with_capacity(db.files().len() + 1);
        for file in db.files() {
            sources.insert(file.id, Source::from(file.text.as_str()));
        }
        // Provide an empty backing source for synthetic spans so `ariadne` never fails `fetch`.
        sources.insert(FileId::SYNTHETIC, Source::from(""));
        Self { db, sources }
    }
}

impl<'a> Cache<FileId> for DbCache<'a> {
    type Storage = &'a str;

    fn fetch(&mut self, id: &FileId) -> Result<&Source<Self::Storage>, impl fmt::Debug> {
        self.sources
            .get(id)
            .ok_or_else(|| format!("unknown file id {}", id.0))
    }

    fn display<'b>(&self, id: &'b FileId) -> Option<impl fmt::Display + 'b> {
        let name = self
            .db
            .get(*id)
            .map_or("<synthetic>", |file| file.name.as_str());
        Some(name.to_owned())
    }
}

/// Renders a slice of diagnostics into `writer` using `ariadne` 0.6 multi-file reports
/// configured with UTF-8 byte indexing matching [`crate::Span`].
pub fn render_diagnostics(
    db: &SourceDatabase,
    diagnostics: &[Diagnostic],
    writer: &mut impl io::Write,
) -> io::Result<()> {
    let mut cache = DbCache::new(db);
    let config = Config::default().with_index_type(IndexType::Byte);

    for diag in diagnostics {
        let (kind, primary_color) = match diag.severity {
            Severity::Error => (ReportKind::Error, Color::Red),
            Severity::Warning => (ReportKind::Warning, Color::Yellow),
            Severity::Advice => (ReportKind::Advice, Color::Cyan),
        };

        let mut builder =
            Report::build(kind, (diag.primary_span.file, diag.primary_span.to_range()))
                .with_config(config)
                .with_code(diag.code.as_str())
                .with_message(&diag.message);

        if diag.labels.is_empty() {
            builder = builder.with_label(
                Label::new((diag.primary_span.file, diag.primary_span.to_range()))
                    .with_message(&diag.message)
                    .with_color(primary_color),
            );
        } else {
            for label in &diag.labels {
                let color = if label.is_primary {
                    primary_color
                } else {
                    Color::Blue
                };
                builder = builder.with_label(
                    Label::new((label.span.file, label.span.to_range()))
                        .with_message(&label.message)
                        .with_color(color),
                );
            }
        }

        for note in &diag.notes {
            builder = builder.with_note(note);
        }

        builder.finish().write(&mut cache, &mut *writer)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Span;

    #[test]
    fn renders_ariadne_report_and_synthetic_spans() {
        let mut db = SourceDatabase::new();
        let file = db.add_file("sample.vsl", "let x: f32 = true;\n");
        let d1 = Diagnostic::error("E0301", "type mismatch", Span::new(file, 13, 17))
            .with_primary_label(Span::new(file, 13, 17), "expected `f32`, found `bool`");
        let d2 = Diagnostic::warning("W0001", "synthetic notice", Span::synthetic());

        let mut out = Vec::new();
        render_diagnostics(&db, &[d1, d2], &mut out).expect("render succeeds");
        let rendered = String::from_utf8(out).expect("utf-8 output");
        assert!(rendered.contains("E0301"));
        assert!(rendered.contains("W0001"));
    }
}
