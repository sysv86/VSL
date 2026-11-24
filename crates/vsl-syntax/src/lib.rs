//! Shared lexer, AST, and parser for VSL 2.0 shader modules (`.vsl`) and manifests (`pack.vsl`).

pub mod ast;
mod parser;
pub mod token;

pub use parser::{ParseResult, parse_source};
pub use token::{SpannedToken, Token, lex};
