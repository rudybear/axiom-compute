//! AXIOM-Compute parser — produces a typed AST from a token stream.
//!
//! M0 grammar was minimal: only `@kernel fn name() -> void { return; }`.
//! M1.1 adds let/let-mut bindings, assignments, and Pratt expressions.
//! M1.2 adds buffer parameters, index expressions, and buffer writes.
//! M1.3 adds structured control flow: if/else, for-range, while, break, continue.

pub mod ast;
pub mod parser;

pub use ast::{Module, Item, KernelDecl, Annotation, AnnotationArg, Block, Stmt, Expr, TypeRef, Param,
              BinOp, UnaryOp, ShortCircuitOp, ElseArm};
pub use parser::{Parser, ParseError, M1_1_RESERVED_KEYWORDS, M1_3_RESERVED_KEYWORDS};

use axc_lexer::LexError;

/// Convenience: lex + parse. Always runs both phases fully.
///
/// Returns `(module, lex_errors, parse_errors)`. All three outputs are valid
/// regardless of errors — the module may be partial (empty items list) when
/// parsing fails, but it is never `None`.
pub fn parse(source: &str) -> (Module, Vec<LexError>, Vec<ParseError>) {
    let (tokens, lex_errors): (Vec<axc_lexer::Token>, Vec<LexError>) = axc_lexer::tokenize(source);
    let mut p: Parser = Parser::new(&tokens);
    let (module, parse_errors): (Module, Vec<ParseError>) = p.parse_module();
    (module, lex_errors, parse_errors)
}
