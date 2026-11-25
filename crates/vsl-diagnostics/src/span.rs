use serde::{Deserialize, Serialize};
use std::ops::Range;

/// Dense identifier for a source file registered in [`SourceDatabase`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FileId(pub u32);

impl FileId {
    /// Sentinel file ID for synthetic compiler-generated nodes.
    pub const SYNTHETIC: Self = Self(u32::MAX);

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Half-open UTF-8 byte span `[start, end)` within a specific [`FileId`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    pub file: FileId,
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub const fn new(file: FileId, start: u32, end: u32) -> Self {
        Self { file, start, end }
    }

    pub const fn synthetic() -> Self {
        Self {
            file: FileId::SYNTHETIC,
            start: 0,
            end: 0,
        }
    }

    pub fn from_range(file: FileId, range: &Range<usize>) -> Self {
        Self {
            file,
            start: range.start as u32,
            end: range.end as u32,
        }
    }

    pub const fn to_range(self) -> Range<usize> {
        (self.start as usize)..(self.end as usize)
    }

    /// Merges two spans from the same file into the minimal enclosing span.
    ///
    /// Synthetic spans yield to real file spans; cross-file merges return `self`
    /// without mixing byte offsets from different files.
    #[must_use]
    pub const fn merge(self, other: Self) -> Self {
        if other.file.0 == FileId::SYNTHETIC.0 {
            return self;
        }
        if self.file.0 == FileId::SYNTHETIC.0 {
            return other;
        }
        if self.file.0 != other.file.0 {
            return self;
        }
        let start = if self.start < other.start {
            self.start
        } else {
            other.start
        };
        let end = if self.end > other.end {
            self.end
        } else {
            other.end
        };
        Self {
            file: self.file,
            start,
            end,
        }
    }
}

/// Zero-based line and column coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineCol {
    pub line: u32,
    pub col: u32,
}

/// Immutable source text paired with precomputed line-start byte offsets.
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub id: FileId,
    pub name: String,
    pub text: String,
    line_starts: Vec<u32>,
}

impl SourceFile {
    pub fn new(id: FileId, name: impl Into<String>, text: impl Into<String>) -> Self {
        let text = text.into();
        let mut line_starts = vec![0u32];
        for (idx, byte) in text.bytes().enumerate() {
            if byte == b'\n' {
                line_starts.push((idx + 1) as u32);
            }
        }
        Self {
            id,
            name: name.into(),
            text,
            line_starts,
        }
    }

    /// Resolves a byte offset to a zero-based `(line, utf8_byte_col)` coordinate.
    pub fn line_col(&self, offset: u32) -> LineCol {
        let clamped = offset.min(self.text.len() as u32);
        let line = match self.line_starts.binary_search(&clamped) {
            Ok(exact) => exact,
            Err(insert_pos) => insert_pos.saturating_sub(1),
        };
        let line_start = self.line_starts[line];
        LineCol {
            line: line as u32,
            col: clamped.saturating_sub(line_start),
        }
    }

    /// Resolves a UTF-8 byte offset to a zero-based `(line, utf16_code_unit_col)` coordinate
    /// compliant with default LSP UTF-16 positioning.
    pub fn line_col_utf16(&self, offset: u32) -> LineCol {
        let clamped = (offset as usize).min(self.text.len());
        let line = match self.line_starts.binary_search(&(clamped as u32)) {
            Ok(exact) => exact,
            Err(insert_pos) => insert_pos.saturating_sub(1),
        };
        let line_start = self.line_starts[line] as usize;
        let mut safe_end = clamped;
        while safe_end > line_start && !self.text.is_char_boundary(safe_end) {
            safe_end -= 1;
        }
        let utf16_col = self.text[line_start..safe_end]
            .chars()
            .map(char::len_utf16)
            .sum::<usize>();
        LineCol {
            line: line as u32,
            col: utf16_col as u32,
        }
    }

    /// Clamps `span` to `0..=self.text.len()`, snaps both endpoints down to valid UTF-8
    /// character boundaries, and guarantees `start <= end`.
    pub(crate) fn clamp_range(&self, span: Span) -> Range<usize> {
        let mut start = (span.start as usize).min(self.text.len());
        while start > 0 && !self.text.is_char_boundary(start) {
            start -= 1;
        }
        let mut end = (span.end as usize).clamp(start, self.text.len());
        while end > start && !self.text.is_char_boundary(end) {
            end -= 1;
        }
        start..end
    }

    pub fn slice(&self, span: Span) -> &str {
        &self.text[self.clamp_range(span)]
    }
}

/// In-memory source file registry used by CLI, driver, and LSP.
#[derive(Debug, Default, Clone)]
pub struct SourceDatabase {
    files: Vec<SourceFile>,
}

impl SourceDatabase {
    pub const fn new() -> Self {
        Self { files: Vec::new() }
    }

    pub fn add_file(&mut self, name: impl Into<String>, text: impl Into<String>) -> FileId {
        let id = FileId(self.files.len() as u32);
        self.files.push(SourceFile::new(id, name, text));
        id
    }

    pub fn get(&self, id: FileId) -> Option<&SourceFile> {
        self.files.get(id.index())
    }

    pub fn files(&self) -> &[SourceFile] {
        &self.files
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_col_tracks_newlines_and_utf16() {
        let mut db = SourceDatabase::new();
        let id = db.add_file("test.vsl", "module a;\nlet x = 1;\n");
        let file = db.get(id).expect("file registered");
        assert_eq!(file.line_col(0), LineCol { line: 0, col: 0 });
        assert_eq!(file.line_col(10), LineCol { line: 1, col: 0 });
        assert_eq!(file.line_col_utf16(14), LineCol { line: 1, col: 4 });
    }

    #[test]
    fn slice_snaps_to_utf8_char_boundaries() {
        let mut db = SourceDatabase::new();
        // 'α' occupies bytes 4..6 and 'β' occupies bytes 9..11.
        let id = db.add_file("utf8.vsl", "let α = β;");
        let file = db.get(id).expect("file registered");
        assert_eq!(file.slice(Span::new(id, 5, 6)), "α");
        assert_eq!(file.slice(Span::new(id, 4, 5)), "");
        assert_eq!(file.slice(Span::new(id, 5, 10)), "α = ");
        assert_eq!(file.slice(Span::new(id, 100, 200)), "");
        assert_eq!(file.slice(Span::new(id, 10, 4)), "");
    }

    #[test]
    fn span_merge_guards_synthetic_and_cross_file() {
        let f0 = FileId(0);
        let f1 = FileId(1);
        let a = Span::new(f0, 4, 8);
        let b = Span::new(f0, 2, 10);
        let c = Span::new(f1, 0, 100);
        let syn = Span::synthetic();

        assert_eq!(a.merge(b), Span::new(f0, 2, 10));
        assert_eq!(a.merge(syn), a);
        assert_eq!(syn.merge(c), c);
        assert_eq!(a.merge(c), a);
    }
}
