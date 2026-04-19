//! Recursive-descent parser for AXIOM-Compute M0 grammar.
//!
//! Collects ALL errors — never short-circuits on the first error (anti-pattern #6).
//! `TokenKind::Error` tokens from the lexer are skipped silently (already reported
//! in the `LexError` channel; re-reporting would cause double-counting).
//!
//! The §3.3 M1-reserved keyword pre-check in `parse_stmt` produces
//! `ParseError::UnsupportedInM0` with a human-readable `detail` string so that
//! users see an M1-roadmap-aware hint rather than a generic "unexpected token".

use axc_lexer::{Token, TokenKind, Span, Spanned, LexError};
use crate::ast::{
    Module, Item, KernelDecl, Annotation, AnnotationArg, Block, Stmt, Expr, TypeRef, Param,
};

/// Error produced by the parser.
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum ParseError {
    #[error("expected {expected}, found {found}")]
    Unexpected {
        expected: String,
        found: String,
        #[label("here")]
        span: Span,
    },
    #[error("unknown top-level item; only `@kernel fn ...` is supported in M0")]
    UnknownItem {
        #[label("here")]
        span: Span,
    },
    #[error("unsupported syntax in M0: {detail}")]
    UnsupportedInM0 {
        detail: String,
        #[label("here")]
        span: Span,
    },
    #[error("unterminated annotation argument list")]
    UnterminatedAnnotationArgs {
        #[label("here")]
        span: Span,
    },
    #[error(transparent)]
    LexerError(#[from] LexError),
}

/// Recursive-descent parser.
pub struct Parser<'tok> {
    tokens: &'tok [Token],
    pos: usize,
    errors: Vec<ParseError>,
}

impl<'tok> Parser<'tok> {
    /// Create a new parser over a slice of tokens (must end with `Eof`).
    pub fn new(tokens: &'tok [Token]) -> Self {
        Self { tokens, pos: 0, errors: Vec::new() }
    }

    /// Parse the entire token stream into a `Module`.
    ///
    /// Always returns a (possibly empty/partial) module plus all collected errors.
    pub fn parse_module(&mut self) -> (Module, Vec<ParseError>) {
        let mut items: Vec<Spanned<Item>> = Vec::new();

        while !self.is_at_end() {
            // Skip Error tokens from the lexer silently (already in LexError channel)
            if self.peek_kind().is_error() {
                self.advance();
                continue;
            }

            if self.peek_kind() == &TokenKind::Eof {
                break;
            }

            // Only `@kernel fn …` is valid at top level in M0
            if matches!(self.peek_kind(), TokenKind::Annotation(_)) {
                match self.parse_kernel_decl() {
                    Some(kd) => {
                        let span: Span = kd.name.span;
                        items.push(Spanned::new(Item::Kernel(kd), span));
                    }
                    None => {
                        // Recovery: parse_kernel_decl already pushed errors; skip to next `@`
                        self.recover_to_next_annotation();
                    }
                }
            } else {
                // Unknown top-level item
                let tok: &Token = self.current_token();
                self.errors.push(ParseError::UnknownItem { span: tok.span });
                self.recover_to_next_annotation();
            }
        }

        let errors: Vec<ParseError> = std::mem::take(&mut self.errors);
        (Module { items }, errors)
    }

    // ── Kernel declaration ───────────────────────────────────────────────────

    fn parse_kernel_decl(&mut self) -> Option<KernelDecl> {
        // 1. Collect all leading annotations
        let annotations: Vec<Spanned<Annotation>> = self.parse_annotations();

        // 2. Verify at least one annotation is `@kernel`
        let has_kernel_annotation: bool = annotations.iter().any(|a| a.node.name.node == "kernel");
        if !has_kernel_annotation {
            // This is a top-level item that starts with annotations but is not @kernel
            let span: Span = annotations.first().map(|a| a.span).unwrap_or(self.peek_span());
            self.errors.push(ParseError::UnknownItem { span });
            return None;
        }

        // 3. `fn` keyword
        if !self.expect_token(TokenKind::Fn, "fn") {
            return None;
        }

        // 4. Kernel name
        let name_tok: &Token = self.current_token();
        let name_span: Span = name_tok.span;
        let name: String = match &name_tok.kind {
            TokenKind::Ident(n) => n.clone(),
            _ => {
                self.errors.push(ParseError::Unexpected {
                    expected: "kernel name (identifier)".into(),
                    found: format!("{:?}", name_tok.kind),
                    span: name_tok.span,
                });
                return None;
            }
        };
        self.advance();

        // 5. Parameter list `( params? )`
        if !self.expect_token(TokenKind::LParen, "(") {
            return None;
        }
        let params: Vec<Spanned<Param>> = self.parse_params();
        if !self.expect_token(TokenKind::RParen, ")") {
            return None;
        }

        // 6. Return type `-> type_ref`
        if !self.expect_token(TokenKind::Arrow, "->") {
            return None;
        }
        let ret_ty: Spanned<TypeRef> = self.parse_type_ref()?;

        // 7. Body `{ stmts }`
        let body: Spanned<Block> = self.parse_block()?;

        Some(KernelDecl {
            annotations,
            name: Spanned::new(name, name_span),
            params,
            return_type: ret_ty,
            body,
        })
    }

    // ── Annotations ──────────────────────────────────────────────────────────

    fn parse_annotations(&mut self) -> Vec<Spanned<Annotation>> {
        let mut result: Vec<Spanned<Annotation>> = Vec::new();
        while matches!(self.peek_kind(), TokenKind::Annotation(_)) {
            let ann_start: Span = self.peek_span();
            let name_str: String = match &self.peek_kind().clone() {
                TokenKind::Annotation(n) => n.clone(),
                _ => unreachable!(), // guarded by while condition
            };
            self.advance();

            let args: Vec<Spanned<AnnotationArg>> = if self.peek_kind() == &TokenKind::LParen {
                self.advance(); // consume `(`
                match self.parse_annotation_args() {
                    Some(a) => a,
                    None => return result, // error already pushed
                }
            } else {
                Vec::new()
            };

            let span: Span = ann_start.merge(self.last_span());
            result.push(Spanned::new(
                Annotation {
                    name: Spanned::new(name_str, ann_start),
                    args,
                },
                span,
            ));
        }
        result
    }

    fn parse_annotation_args(&mut self) -> Option<Vec<Spanned<AnnotationArg>>> {
        let mut args: Vec<Spanned<AnnotationArg>> = Vec::new();
        loop {
            // Skip lexer error tokens inside annotation args
            while self.peek_kind().is_error() {
                self.advance();
            }
            if self.peek_kind() == &TokenKind::RParen || self.is_at_end() {
                break;
            }
            let arg: Spanned<AnnotationArg> = match self.parse_annotation_arg() {
                Some(a) => a,
                None => {
                    // Error already pushed; try to close the arg list
                    self.errors.push(ParseError::UnterminatedAnnotationArgs {
                        span: self.peek_span(),
                    });
                    return Some(args);
                }
            };
            args.push(arg);
            // Optional trailing comma
            if self.peek_kind() == &TokenKind::Comma {
                self.advance();
            }
        }
        // Consume closing `)`
        if self.peek_kind() == &TokenKind::RParen {
            self.advance();
        }
        Some(args)
    }

    fn parse_annotation_arg(&mut self) -> Option<Spanned<AnnotationArg>> {
        let span_start: Span = self.peek_span();
        let kind: &TokenKind = self.peek_kind();

        match kind.clone() {
            // Unary minus followed by an integer literal: treat as a negative integer.
            TokenKind::Minus => {
                self.advance(); // consume `-`
                if let TokenKind::IntLiteral { value, .. } = self.peek_kind().clone() {
                    let span_end: Span = self.peek_span();
                    self.advance();
                    let v: i64 = i64::try_from(value.wrapping_neg()).unwrap_or(i64::MIN);
                    Some(Spanned::new(AnnotationArg::Int(v), span_start.merge(span_end)))
                } else {
                    self.errors.push(ParseError::Unexpected {
                        expected: "integer literal after `-`".into(),
                        found: format!("{:?}", self.peek_kind()),
                        span: self.peek_span(),
                    });
                    None
                }
            }
            TokenKind::IntLiteral { value, .. } => {
                self.advance();
                let v: i64 = i64::try_from(value).unwrap_or(i64::MAX);
                Some(Spanned::new(AnnotationArg::Int(v), span_start))
            }
            TokenKind::BoolLiteral(b) => {
                self.advance();
                Some(Spanned::new(AnnotationArg::Bool(b), span_start))
            }
            TokenKind::StringLiteral(s) => {
                self.advance();
                Some(Spanned::new(AnnotationArg::String(s.clone()), span_start))
            }
            TokenKind::Ident(name) => {
                self.advance();
                // Is this a call like `O(1)` or `Theta(n)`?
                if self.peek_kind() == &TokenKind::LParen {
                    self.advance(); // consume `(`
                    let inner_args: Vec<Spanned<AnnotationArg>> = self.parse_annotation_args()?;
                    let span: Span = span_start.merge(self.last_span());
                    Some(Spanned::new(AnnotationArg::Call { name, args: inner_args }, span))
                } else {
                    Some(Spanned::new(AnnotationArg::Ident(name), span_start))
                }
            }
            _ => {
                self.errors.push(ParseError::Unexpected {
                    expected: "annotation argument (integer, string, bool, identifier, or call)".into(),
                    found: format!("{:?}", kind),
                    span: span_start,
                });
                None
            }
        }
    }

    // ── Parameters ───────────────────────────────────────────────────────────

    fn parse_params(&mut self) -> Vec<Spanned<Param>> {
        // M0 rejects any params with UnsupportedInM0
        if self.peek_kind() == &TokenKind::RParen {
            return Vec::new(); // empty param list is fine
        }
        // Any non-`)` token means params are present — reject
        let span: Span = self.peek_span();
        self.errors.push(ParseError::UnsupportedInM0 {
            detail: "kernel parameters".into(),
            span,
        });
        // Recovery: skip to the matching `)` (or EOF)
        let mut depth: usize = 0;
        loop {
            match self.peek_kind() {
                TokenKind::RParen if depth == 0 => break,
                TokenKind::LParen => { depth += 1; self.advance(); }
                TokenKind::RParen => { depth -= 1; self.advance(); }
                TokenKind::Eof => break,
                _ => { self.advance(); }
            }
        }
        Vec::new()
    }

    // ── Type reference ───────────────────────────────────────────────────────

    fn parse_type_ref(&mut self) -> Option<Spanned<TypeRef>> {
        let span: Span = self.peek_span();
        let ty: TypeRef = match self.peek_kind().clone() {
            TokenKind::Void => { self.advance(); TypeRef::Void }
            TokenKind::Bool => { self.advance(); TypeRef::Bool }
            TokenKind::I32  => { self.advance(); TypeRef::I32 }
            TokenKind::U32  => { self.advance(); TypeRef::U32 }
            TokenKind::F32  => { self.advance(); TypeRef::F32 }
            other => {
                self.errors.push(ParseError::Unexpected {
                    expected: "type (void, bool, i32, u32, f32)".into(),
                    found: format!("{:?}", other),
                    span,
                });
                return None;
            }
        };
        Some(Spanned::new(ty, span))
    }

    // ── Block ────────────────────────────────────────────────────────────────

    fn parse_block(&mut self) -> Option<Spanned<Block>> {
        let block_start: Span = self.peek_span();
        if !self.expect_token(TokenKind::LBrace, "{") {
            return None;
        }

        let mut stmts: Vec<Spanned<Stmt>> = Vec::new();

        while self.peek_kind() != &TokenKind::RBrace && !self.is_at_end() {
            // Skip lexer error tokens silently
            if self.peek_kind().is_error() {
                self.advance();
                continue;
            }
            if let Some(stmt) = self.parse_stmt() {
                stmts.push(stmt);
            }
            // parse_stmt either succeeded or recovered; continue with next statement
        }

        let block_end_span: Span = self.peek_span();
        if !self.expect_token(TokenKind::RBrace, "}") {
            return None;
        }

        let span: Span = block_start.merge(block_end_span);
        Some(Spanned::new(Block { stmts }, span))
    }

    // ── Statement ────────────────────────────────────────────────────────────

    /// Parse one statement inside a kernel body (§3.3 M1-reserved keyword pre-check).
    ///
    /// Flow:
    /// 1. Peek next token.
    /// 2. If it is in the M1-reserved deny-list, emit `UnsupportedInM0` with a
    ///    per-keyword detail string and recover to the next `;` or `}`.
    /// 3. Otherwise, expect `return`, optional expression, `;`.
    fn parse_stmt(&mut self) -> Option<Spanned<Stmt>> {
        let tok: &Token = self.current_token();
        let span: Span = tok.span;

        // §3.3: M1-reserved keyword pre-check
        if let Some(detail) = tok.kind.m1_reserved_detail() {
            self.errors.push(ParseError::UnsupportedInM0 {
                detail: detail.to_owned(),
                span,
            });
            // Recovery: skip tokens until `;` or `}` (end of statement / block)
            self.advance(); // consume the reserved keyword
            loop {
                match self.peek_kind() {
                    TokenKind::Semicolon => { self.advance(); break; }
                    TokenKind::RBrace | TokenKind::Eof => break,
                    _ => { self.advance(); }
                }
            }
            return None;
        }

        // Expect `return`
        if self.peek_kind() != &TokenKind::Return {
            let found: String = format!("{:?}", self.peek_kind());
            self.errors.push(ParseError::Unexpected {
                expected: "return".into(),
                found,
                span,
            });
            // Recovery: skip to next `;` or `}`
            loop {
                match self.peek_kind() {
                    TokenKind::Semicolon => { self.advance(); break; }
                    TokenKind::RBrace | TokenKind::Eof => break,
                    _ => { self.advance(); }
                }
            }
            return None;
        }
        let stmt_start: Span = self.peek_span();
        self.advance(); // consume `return`

        // Optional expression (only int or bool literals in M0)
        let expr: Option<Spanned<Expr>> = if self.peek_kind() == &TokenKind::Semicolon
            || self.peek_kind() == &TokenKind::RBrace
            || self.is_at_end()
        {
            None
        } else {
            self.parse_expr()
        };

        // Consume `;`
        let stmt_end: Span = self.peek_span();
        if !self.expect_token(TokenKind::Semicolon, ";") {
            return None;
        }

        let span: Span = stmt_start.merge(stmt_end);
        Some(Spanned::new(Stmt::Return(expr), span))
    }

    fn parse_expr(&mut self) -> Option<Spanned<Expr>> {
        let span: Span = self.peek_span();
        match self.peek_kind().clone() {
            TokenKind::BoolLiteral(b) => {
                self.advance();
                Some(Spanned::new(Expr::BoolLit(b), span))
            }
            TokenKind::IntLiteral { value, .. } => {
                self.advance();
                Some(Spanned::new(Expr::IntLit(value), span))
            }
            _ => None,
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Consume the current token if it matches `expected_kind`; push an error otherwise.
    /// Returns `true` if the token was consumed.
    fn expect_token(&mut self, expected_kind: TokenKind, expected_label: &str) -> bool {
        if self.peek_kind() == &expected_kind {
            self.advance();
            return true;
        }
        let found: String = format!("{:?}", self.peek_kind());
        self.errors.push(ParseError::Unexpected {
            expected: expected_label.into(),
            found,
            span: self.peek_span(),
        });
        false
    }

    /// Skip tokens until the next `Annotation` token (start of a new item) or EOF.
    fn recover_to_next_annotation(&mut self) {
        loop {
            match self.peek_kind() {
                TokenKind::Annotation(_) | TokenKind::Eof => break,
                _ => { self.advance(); }
            }
        }
    }

    fn peek_kind(&self) -> &TokenKind {
        &self.tokens[self.pos.min(self.tokens.len() - 1)].kind
    }

    fn peek_span(&self) -> Span {
        self.tokens[self.pos.min(self.tokens.len() - 1)].span
    }

    fn current_token(&self) -> &Token {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn last_span(&self) -> Span {
        let prev: usize = self.pos.saturating_sub(1).min(self.tokens.len() - 1);
        self.tokens[prev].span
    }

    fn advance(&mut self) -> &Token {
        let tok: &Token = &self.tokens[self.pos.min(self.tokens.len() - 1)];
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        tok
    }

    fn is_at_end(&self) -> bool {
        self.peek_kind() == &TokenKind::Eof
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axc_lexer::tokenize;
    use crate::ast::{AnnotationArg, TypeRef};

    fn parse_src(src: &str) -> (Module, Vec<ParseError>) {
        let (tokens, _) = tokenize(src);
        let mut p = Parser::new(&tokens);
        p.parse_module()
    }

    // ── Happy path ───────────────────────────────────────────────────────────

    #[test]
    fn happy_empty_kernel() {
        let src = "@kernel @workgroup(64,1,1) fn empty() -> void { return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        assert_eq!(module.items.len(), 1);
    }

    #[test]
    fn happy_multi_annotation() {
        let src = concat!(
            "@kernel @workgroup(64,1,1) @intent(\"test\") @precondition(true) ",
            "fn k() -> void { return; }",
        );
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        let item = &module.items[0];
        let crate::ast::Item::Kernel(ref kd) = item.node;
        assert_eq!(kd.annotations.len(), 4);
    }

    #[test]
    fn happy_return_with_value() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { return 0; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        assert_eq!(kd.body.node.stmts.len(), 1);
    }

    // ── AT-6: unknown top-level item ─────────────────────────────────────────

    #[test]
    fn unknown_item_struct() {
        let src = "struct Foo {}";
        let (_, errors) = parse_src(src);
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| matches!(e, ParseError::UnknownItem { .. })));
    }

    // ── AT-9: M1-reserved keyword pre-check ──────────────────────────────────

    #[test]
    fn nonempty_body_rejected_let() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { let x: i32 = 0; return; }";
        let (_, errors) = parse_src(src);
        let m0_err = errors.iter().find(|e| matches!(e, ParseError::UnsupportedInM0 { detail, .. } if detail == "let statement"));
        assert!(m0_err.is_some(), "expected UnsupportedInM0 with detail 'let statement', got: {errors:?}");
    }

    #[test]
    fn nonempty_body_rejected_if() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { if true { return; } return; }";
        let (_, errors) = parse_src(src);
        let m0_err = errors.iter().find(|e| matches!(e, ParseError::UnsupportedInM0 { detail, .. } if detail == "if statement"));
        assert!(m0_err.is_some(), "expected UnsupportedInM0 with detail 'if statement', got: {errors:?}");
    }

    #[test]
    fn nonempty_body_rejected_for() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { for i in 0..1 {} return; }";
        let (_, errors) = parse_src(src);
        let m0_err = errors.iter().find(|e| matches!(e, ParseError::UnsupportedInM0 { detail, .. } if detail == "for loop"));
        assert!(m0_err.is_some(), "expected UnsupportedInM0 with detail 'for loop', got: {errors:?}");
    }

    // ── Kernel params are rejected ────────────────────────────────────────────

    #[test]
    fn kernel_params_rejected() {
        let src = "@kernel @workgroup(64,1,1) fn k(x: i32) -> void { return; }";
        let (_, errors) = parse_src(src);
        let m0_err = errors.iter().find(|e| matches!(e, ParseError::UnsupportedInM0 { detail, .. } if detail == "kernel parameters"));
        assert!(m0_err.is_some(), "expected UnsupportedInM0 for kernel parameters: {errors:?}");
    }

    // ── Missing @kernel annotation → UnknownItem ──────────────────────────────

    #[test]
    fn fn_without_kernel_annotation_is_unknown_item() {
        let src = "@workgroup(64,1,1) fn k() -> void { return; }";
        let (_, errors) = parse_src(src);
        assert!(errors.iter().any(|e| matches!(e, ParseError::UnknownItem { .. })),
            "expected UnknownItem: {errors:?}");
    }

    // ── Error-token skip (no double-reporting) ────────────────────────────────

    #[test]
    fn lex_error_not_double_reported_as_parse_error() {
        // Source with an emoji (lex error) followed by struct (parse error).
        // Parser should skip the Error token silently; only UnknownItem for struct.
        let src = "💥 struct Foo {}";
        let (tokens, lex_errors) = tokenize(src);
        assert!(!lex_errors.is_empty(), "expected lex error for emoji");
        let mut p = Parser::new(&tokens);
        let (_, parse_errors) = p.parse_module();
        // No ParseError should re-report the lex error
        let has_lex_replication = parse_errors.iter().any(|e| matches!(e, ParseError::LexerError(_)));
        assert!(!has_lex_replication, "lex error was double-reported as ParseError");
    }

    // ── @complexity(O(1)) parses to Call AST (anti-pattern #7 guard) ─────────

    #[test]
    fn complexity_o1_parses_to_call_not_string() {
        let src = "@kernel @workgroup(64,1,1) @complexity(O(1)) fn k() -> void { return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        let complexity_ann = kd.annotations.iter().find(|a| a.node.name.node == "complexity")
            .expect("@complexity annotation missing");
        assert_eq!(complexity_ann.node.args.len(), 1);
        if let AnnotationArg::Call { ref name, ref args } = complexity_ann.node.args[0].node {
            assert_eq!(name, "O");
            assert_eq!(args.len(), 1);
            assert!(matches!(args[0].node, AnnotationArg::Int(1)));
        } else {
            panic!("expected Call arg, got: {:?}", complexity_ann.node.args[0].node);
        }
    }

    // ── Trailing comma in annotations ─────────────────────────────────────────

    #[test]
    fn trailing_comma_in_annotation_args() {
        let src = "@kernel @workgroup(64, 1, 1,) fn empty() -> void { return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        assert!(!module.items.is_empty());
    }

    // ── TypeRef variants parse ───────────────────────────────────────────────

    #[test]
    fn return_type_void_parses() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        assert_eq!(kd.return_type.node, TypeRef::Void);
    }
}
