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
    #[must_use]
    pub const fn merge(self, other: Self) -> Self {
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

    pub fn slice(&self, span: Span) -> &str {
        let start = (span.start as usize).min(self.text.len());
        let end = (span.end as usize).clamp(start, self.text.len());
        &self.text[start..end]
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
}
