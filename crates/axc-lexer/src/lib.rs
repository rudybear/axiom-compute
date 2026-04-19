//! AXIOM-Compute Lexer — tokenizes `.axc` source into a stream of typed tokens.
//!
//! Design principles:
//! - Every token carries a `Span` (byte offset range into the raw source)
//! - Error recovery: invalid chars produce `TokenKind::Error` tokens AND push a
//!   `LexError`; lexing always continues to EOF
//! - Annotations (`@name`) and optimization holes (`?name`) are first-class tokens
//! - M1-reserved keywords (`let`, `if`, …) are dedicated `TokenKind` variants so
//!   the parser can match them in its keyword-deny-list check (§3.3 of spec)
//! - BOM detection is NOT done here; that is the driver's responsibility

pub mod token;
pub mod lexer;

pub use token::{Token, TokenKind, Span, Spanned, IntBase, IntSuffix, FloatSuffix, LineIndex};
pub use lexer::{Lexer, LexError};

/// Convenience wrapper: lex `source` from start to EOF.
///
/// Always runs to EOF regardless of errors; never short-circuits.
/// Returns all tokens (including `TokenKind::Error` in-stream) plus a
/// side-channel `Vec<LexError>` for callers that need structured errors.
pub fn tokenize(source: &str) -> (Vec<Token>, Vec<LexError>) {
    Lexer::new(source).tokenize()
}
