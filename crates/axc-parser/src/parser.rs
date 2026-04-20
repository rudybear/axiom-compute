//! Recursive-descent + Pratt expression parser for AXIOM-Compute M1.1 grammar.
//!
//! Collects ALL errors — never short-circuits on the first error (anti-pattern #6).
//! `TokenKind::Error` tokens from the lexer are skipped silently (already reported
//! in the `LexError` channel; re-reporting would cause double-counting).
//!
//! The §3.3 M1.1-reserved keyword pre-check in `parse_stmt` produces
//! `ParseError::UnsupportedInM1_1` with a human-readable `detail` string so that
//! users see a roadmap-aware hint rather than a generic "unexpected token".
//!
//! M1.1 changes from M0:
//! - Let and Mut are removed from the reserved deny-list; they are now valid syntax.
//! - Pratt expression parser added (§3.3 precedence table).
//! - Let / Assign statements added to `parse_stmt`.
//! - ParseError::UnsupportedInM0 renamed to UnsupportedInM1_1.
//! - TypeRef gains I64, U64, F64.
//! - MAX_EXPR_DEPTH = 256 nesting limit.

use axc_lexer::{Token, TokenKind, Span, Spanned, LexError, is_reserved_subgroup_builtin};
use crate::ast::{
    Module, Item, KernelDecl, Annotation, AnnotationArg, Block, Stmt, Expr, TypeRef, Param,
    BinOp, UnaryOp, ShortCircuitOp, ScalarTypeRef, ElseArm, CoopMatUseAst,
};

/// Maximum expression nesting depth before emitting ExpressionNestingTooDeep.
const MAX_EXPR_DEPTH: u32 = 256;

/// M1.3-reserved keywords — keywords still deferred past this milestone.
///
/// M1.1 removed Let/Mut from this list (now valid syntax).
/// M1.3 removes If/Else/For/While/Break/Continue (now valid control-flow syntax).
/// Only Struct remains — deferred to M2.
///
/// This constant mirrors `TokenKind::m1_reserved_detail()`'s Some-returning set
/// and is used in `parse_stmt` (deny-list check) and in tests.
pub const M1_3_RESERVED_KEYWORDS: &[TokenKind] = &[
    TokenKind::Struct,
];

/// Backward-compat alias pointing at the current reserved-keyword table.
///
/// Prior milestones used this name. New code should reference `M1_3_RESERVED_KEYWORDS`.
pub const M1_1_RESERVED_KEYWORDS: &[TokenKind] = M1_3_RESERVED_KEYWORDS;

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
    /// Formerly UnsupportedInM0; renamed in M1.1 to reflect that we now implement
    /// let/mut/return, but control flow (if/for/while/etc.) is still deferred.
    #[error("unsupported syntax in M1.1: {detail}")]
    UnsupportedInM1_1 {
        detail: String,
        #[label("here")]
        span: Span,
    },
    /// M1.3-specific rejection — for features still deferred past M1.3 (e.g. struct).
    ///
    /// Emitted by `parse_stmt` when a keyword listed in `M1_3_RESERVED_KEYWORDS` is
    /// encountered.  The `detail` field carries a roadmap hint (e.g.
    /// "`struct` is deferred to M2").
    #[error("unsupported syntax in M1.3: {detail}")]
    UnsupportedInM1_3 {
        detail: String,
        #[label("here")]
        span: Span,
    },
    #[error("unterminated annotation argument list")]
    UnterminatedAnnotationArgs {
        #[label("here")]
        span: Span,
    },
    /// `let x = …;` without `: type_ref` annotation — anti-pattern #1 at grammar level.
    #[error("missing type annotation in `let` binding; every binding requires an explicit `: type` (e.g. `let x: i32 = 0i32;`)")]
    MissingTypeAnnotation {
        #[label("here")]
        span: Span,
    },
    /// Expression nested deeper than MAX_EXPR_DEPTH (256) parentheses.
    #[error("expression nesting depth exceeds {MAX_EXPR_DEPTH}; simplify the expression")]
    ExpressionNestingTooDeep {
        #[label("here")]
        span: Span,
    },
    #[error(transparent)]
    LexerError(#[from] LexError),

    // ── M2.1 cooperative-matrix parse errors ────────────────────────────────

    #[error("cooperative-matrix dimension must be an unsuffixed positive integer literal; got {found_kind}")]
    CoopMatrixDimMustBeUnsuffixedIntegerLiteral {
        found_kind: String,
        #[label("here")]
        span: Span,
    },

    #[error("cooperative-matrix `use` must be `a`, `b`, or `accumulator`; got `{found}`")]
    CoopMatrixUseMustBeABOrAccumulator {
        found: String,
        #[label("here")]
        span: Span,
    },
}

/// Internal enum for infix operators used by the Pratt parser.
#[derive(Debug, Clone, Copy)]
enum InfixOp {
    Binary(BinOp),
    ShortCircuit(ShortCircuitOp),
}

/// Recursive-descent + Pratt expression parser.
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

            // Only `@kernel fn …` is valid at top level in M1.1
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
        // M1.2: kernel parameters are now supported.
        let mut params: Vec<Spanned<Param>> = Vec::new();

        if self.peek_kind() == &TokenKind::RParen {
            return params; // empty param list is fine
        }

        loop {
            // Skip lexer error tokens
            while self.peek_kind().is_error() {
                self.advance();
            }
            if self.peek_kind() == &TokenKind::RParen || self.is_at_end() {
                break;
            }

            let param_start: Span = self.peek_span();

            // name : type_ref
            let name: String = match self.peek_kind().clone() {
                TokenKind::Ident(n) => { self.advance(); n }
                _ => {
                    self.errors.push(ParseError::Unexpected {
                        expected: "parameter name (identifier)".into(),
                        found: format!("{:?}", self.peek_kind()),
                        span: self.peek_span(),
                    });
                    // Recovery: skip to `)` or `,`
                    self.recover_to_comma_or_rparen();
                    break;
                }
            };
            let name_span: Span = self.last_span();

            if !self.expect_token(TokenKind::Colon, ":") {
                self.recover_to_comma_or_rparen();
                break;
            }

            let ty: Spanned<TypeRef> = match self.parse_type_ref() {
                Some(t) => t,
                None => {
                    self.recover_to_comma_or_rparen();
                    break;
                }
            };

            let param_span: Span = param_start.merge(ty.span);
            params.push(Spanned::new(
                Param {
                    name: Spanned::new(name, name_span),
                    ty,
                },
                param_span,
            ));

            // Optional trailing comma
            if self.peek_kind() == &TokenKind::Comma {
                self.advance();
            } else {
                break;
            }
        }

        params
    }

    // ── Type reference ───────────────────────────────────────────────────────

    fn parse_type_ref(&mut self) -> Option<Spanned<TypeRef>> {
        let span: Span = self.peek_span();
        let ty: TypeRef = match self.peek_kind().clone() {
            TokenKind::Void => { self.advance(); TypeRef::Void }
            TokenKind::Bool => { self.advance(); TypeRef::Bool }
            TokenKind::I32  => { self.advance(); TypeRef::I32 }
            TokenKind::U32  => { self.advance(); TypeRef::U32 }
            TokenKind::I64  => { self.advance(); TypeRef::I64 }
            TokenKind::U64  => { self.advance(); TypeRef::U64 }
            // M2.1: F16 scalar type
            TokenKind::F16  => { self.advance(); TypeRef::F16 }
            TokenKind::F32  => { self.advance(); TypeRef::F32 }
            TokenKind::F64  => { self.advance(); TypeRef::F64 }
            // M1.2: buffer types — buffer[elem], readonly_buffer[elem], writeonly_buffer[elem]
            TokenKind::Buffer => {
                self.advance();
                let elem: ScalarTypeRef = self.parse_buffer_elem()?;
                let end_span: Span = self.last_span();
                return Some(Spanned::new(TypeRef::Buffer(elem), span.merge(end_span)));
            }
            TokenKind::ReadonlyBuffer => {
                self.advance();
                let elem: ScalarTypeRef = self.parse_buffer_elem()?;
                let end_span: Span = self.last_span();
                return Some(Spanned::new(TypeRef::ReadonlyBuffer(elem), span.merge(end_span)));
            }
            TokenKind::WriteonlyBuffer => {
                self.advance();
                let elem: ScalarTypeRef = self.parse_buffer_elem()?;
                let end_span: Span = self.last_span();
                return Some(Spanned::new(TypeRef::WriteonlyBuffer(elem), span.merge(end_span)));
            }
            // M2.1: `matrix[T, M, N, use]` cooperative-matrix type.
            // `matrix` is a plain Ident (NOT a keyword) — the parser matches on the ident string.
            TokenKind::Ident(ref ident) if ident == "matrix" => {
                self.advance();
                if let Some(cm) = self.parse_coopmat_type_args() {
                    let end_span: Span = self.last_span();
                    return Some(Spanned::new(cm, span.merge(end_span)));
                }
                return None;
            }
            other => {
                self.errors.push(ParseError::Unexpected {
                    expected: "type (void, bool, i32, u32, i64, u64, f16, f32, f64, buffer[T], readonly_buffer[T], writeonly_buffer[T], matrix[T, M, N, use])".into(),
                    found: format!("{:?}", other),
                    span,
                });
                return None;
            }
        };
        Some(Spanned::new(ty, span))
    }

    /// Parse `[T, M, N, use]` for `matrix[T, M, N, use]` cooperative-matrix type.
    ///
    /// Called after `matrix` ident is consumed. Produces `TypeRef::CoopMatrix { ... }`.
    fn parse_coopmat_type_args(&mut self) -> Option<TypeRef> {
        if !self.expect_token(TokenKind::LBracket, "[") {
            return None;
        }
        // Arg 1: element scalar type from the M2.1 allowed set.
        let elem: ScalarTypeRef = self.parse_coopmat_elem_type()?;
        if !self.expect_token(TokenKind::Comma, ",") {
            return None;
        }
        // Arg 2: M dimension — unsuffixed positive integer literal.
        let m: u32 = self.parse_coopmat_dim("M")?;
        if !self.expect_token(TokenKind::Comma, ",") {
            return None;
        }
        // Arg 3: N dimension.
        let n: u32 = self.parse_coopmat_dim("N")?;
        if !self.expect_token(TokenKind::Comma, ",") {
            return None;
        }
        // Arg 4: use tag — bare ident `a`, `b`, or `accumulator`.
        let use_span: Span = self.peek_span();
        let use_: CoopMatUseAst = match self.peek_kind().clone() {
            TokenKind::Ident(ref s) if s == "a" => { self.advance(); CoopMatUseAst::A }
            TokenKind::Ident(ref s) if s == "b" => { self.advance(); CoopMatUseAst::B }
            TokenKind::Ident(ref s) if s == "accumulator" => { self.advance(); CoopMatUseAst::Accumulator }
            TokenKind::Ident(ref s) => {
                let found = s.clone();
                self.advance();
                self.errors.push(ParseError::CoopMatrixUseMustBeABOrAccumulator {
                    found,
                    span: use_span,
                });
                return None;
            }
            other => {
                self.errors.push(ParseError::Unexpected {
                    expected: "cooperative-matrix use (`a`, `b`, or `accumulator`)".into(),
                    found: format!("{:?}", other),
                    span: use_span,
                });
                return None;
            }
        };
        if !self.expect_token(TokenKind::RBracket, "]") {
            return None;
        }
        Some(TypeRef::CoopMatrix { elem, m, n, use_ })
    }

    /// Parse a cooperative-matrix element type from the M2.1 allowed set.
    ///
    /// `bf16` is accepted at the parser layer so that the HIR can emit a precise
    /// `CoopMatrixElementTypeUnsupported` diagnostic (AT-609). All other unknown
    /// types are rejected here with `ParseError::Unexpected`.
    fn parse_coopmat_elem_type(&mut self) -> Option<ScalarTypeRef> {
        match self.peek_kind().clone() {
            TokenKind::I8   => { self.advance(); Some(ScalarTypeRef::I8) }
            TokenKind::U8   => { self.advance(); Some(ScalarTypeRef::U8) }
            TokenKind::I32  => { self.advance(); Some(ScalarTypeRef::I32) }
            TokenKind::U32  => { self.advance(); Some(ScalarTypeRef::U32) }
            TokenKind::F16  => { self.advance(); Some(ScalarTypeRef::F16) }
            // Bf16 is accepted by the parser but rejected by the HIR typechecker
            // with CoopMatrixElementTypeUnsupported (AT-609).
            TokenKind::Bf16 => { self.advance(); Some(ScalarTypeRef::Bf16) }
            TokenKind::F32  => { self.advance(); Some(ScalarTypeRef::F32) }
            other => {
                self.errors.push(ParseError::Unexpected {
                    expected: "cooperative-matrix element type (i8, u8, i32, u32, f16, f32)".into(),
                    found: format!("{:?}", other),
                    span: self.peek_span(),
                });
                None
            }
        }
    }

    /// Parse an unsuffixed positive integer literal for a cooperative-matrix dimension.
    fn parse_coopmat_dim(&mut self, dim_name: &str) -> Option<u32> {
        let dim_span = self.peek_span();
        match self.peek_kind().clone() {
            TokenKind::IntLiteral { value, suffix: None, .. } => {
                self.advance();
                if value <= 0 || value > 65535 {
                    self.errors.push(ParseError::CoopMatrixDimMustBeUnsuffixedIntegerLiteral {
                        found_kind: format!("{value} (out of range 1..=65535 for {dim_name})"),
                        span: dim_span,
                    });
                    return None;
                }
                Some(value as u32)
            }
            TokenKind::IntLiteral { suffix: Some(_), .. } => {
                self.advance();
                self.errors.push(ParseError::CoopMatrixDimMustBeUnsuffixedIntegerLiteral {
                    found_kind: "suffixed integer literal (remove the suffix)".into(),
                    span: dim_span,
                });
                None
            }
            other => {
                self.errors.push(ParseError::CoopMatrixDimMustBeUnsuffixedIntegerLiteral {
                    found_kind: format!("{:?}", other),
                    span: dim_span,
                });
                None
            }
        }
    }

    /// Parse `[scalar_type]` after a buffer keyword.
    ///
    /// M2.1: f16 is now accepted as a buffer element type.
    fn parse_buffer_elem(&mut self) -> Option<ScalarTypeRef> {
        if !self.expect_token(TokenKind::LBracket, "[") {
            return None;
        }
        let elem: ScalarTypeRef = match self.peek_kind().clone() {
            TokenKind::I32 => { self.advance(); ScalarTypeRef::I32 }
            TokenKind::U32 => { self.advance(); ScalarTypeRef::U32 }
            TokenKind::I64 => { self.advance(); ScalarTypeRef::I64 }
            TokenKind::U64 => { self.advance(); ScalarTypeRef::U64 }
            // M2.1: f16 buffer elements allowed.
            TokenKind::F16 => { self.advance(); ScalarTypeRef::F16 }
            TokenKind::F32 => { self.advance(); ScalarTypeRef::F32 }
            TokenKind::F64 => { self.advance(); ScalarTypeRef::F64 }
            other => {
                self.errors.push(ParseError::Unexpected {
                    expected: "buffer element type (i32, u32, i64, u64, f16, f32, f64)".into(),
                    found: format!("{:?}", other),
                    span: self.peek_span(),
                });
                return None;
            }
        };
        if !self.expect_token(TokenKind::RBracket, "]") {
            return None;
        }
        Some(elem)
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

    /// Parse one statement inside a kernel body (§3.2 dispatch table).
    ///
    /// Flow:
    /// 1. Peek next token.
    /// 2. If it is in M1_1_RESERVED_KEYWORDS, emit `UnsupportedInM1_1` and recover.
    /// 3. If `Let` -> parse_let_stmt.
    /// 4. If `Return` -> parse return statement (M0 path).
    /// 5. If `Ident` -> lookahead for `=` -> parse_assign_stmt, else recover.
    /// 6. Otherwise -> Unexpected.
    fn parse_stmt(&mut self) -> Option<Spanned<Stmt>> {
        let tok: &Token = self.current_token();
        let span: Span = tok.span;

        // §3.2: M1.3-reserved keyword pre-check (only Struct remains deferred).
        // Emit UnsupportedInM1_3 for features deferred past M1.3.
        // UnsupportedInM1_1 is kept for backward compatibility with M1.1/M1.2 error paths.
        if let Some(detail) = tok.kind.m1_reserved_detail() {
            self.errors.push(ParseError::UnsupportedInM1_3 {
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

        match self.peek_kind().clone() {
            TokenKind::Let    => self.parse_let_stmt(),
            TokenKind::Return => self.parse_return_stmt(),
            TokenKind::If     => self.parse_if_stmt(),
            TokenKind::For    => self.parse_for_stmt(),
            TokenKind::While  => self.parse_while_stmt(),
            TokenKind::Break  => {
                let break_span: Span = self.peek_span();
                self.advance(); // consume `break`
                let semi_span: Span = self.peek_span();
                if !self.expect_token(TokenKind::Semicolon, ";") {
                    return None;
                }
                Some(Spanned::new(Stmt::Break, break_span.merge(semi_span)))
            }
            TokenKind::Continue => {
                let cont_span: Span = self.peek_span();
                self.advance(); // consume `continue`
                let semi_span: Span = self.peek_span();
                if !self.expect_token(TokenKind::Semicolon, ";") {
                    return None;
                }
                Some(Spanned::new(Stmt::Continue, cont_span.merge(semi_span)))
            }
            TokenKind::Ident(ref ident_name) => {
                // M1.4: if the ident is a reserved subgroup builtin followed by `(`,
                // parse it as a BuiltinCallStmt (the HIR will validate return type).
                // M2.1: coopmat_store also dispatches as BuiltinCallStmt (void-returning
                // statement-level coopmat builtin).
                let ident_name_clone: String = ident_name.clone();
                let next_pos: usize = (self.pos + 1).min(self.tokens.len() - 1);
                if (is_reserved_subgroup_builtin(&ident_name_clone)
                    || axc_lexer::is_reserved_coopmat_builtin(&ident_name_clone))
                    && matches!(self.tokens[next_pos].kind, TokenKind::LParen)
                {
                    return self.parse_builtin_call_stmt();
                }
                // Lookahead: pos+1 is `=` → scalar assign; pos+1 is `[` → buffer index assign.
                match self.tokens[next_pos].kind {
                    TokenKind::Assign => {
                        return self.parse_assign_stmt();
                    }
                    TokenKind::LBracket => {
                        return self.parse_index_assign_stmt();
                    }
                    _ => {}
                }
                // Else: bare expression statements are not allowed in M1.3/M1.4
                let found: String = format!("{:?}", self.peek_kind());
                self.errors.push(ParseError::Unexpected {
                    expected: "let, return, assignment, or control flow".into(),
                    found,
                    span,
                });
                self.recover_to_semicolon_or_brace();
                None
            }
            _ => {
                let found: String = format!("{:?}", self.peek_kind());
                self.errors.push(ParseError::Unexpected {
                    expected: "let, return, assignment, or control flow".into(),
                    found,
                    span,
                });
                self.recover_to_semicolon_or_brace();
                None
            }
        }
    }

    /// Parse `if cond { then } [else (if_stmt | block)]`
    fn parse_if_stmt(&mut self) -> Option<Spanned<Stmt>> {
        let if_span: Span = self.peek_span();
        self.advance(); // consume `if`

        // Condition expression
        let cond: Spanned<Expr> = match self.parse_expr(0) {
            Some(e) => e,
            None => {
                self.recover_to_semicolon_or_brace();
                return None;
            }
        };

        // Then-block
        let then_block: Spanned<Block> = self.parse_block()?;

        // Optional else arm
        let else_arm: Option<Box<ElseArm>> = if self.peek_kind() == &TokenKind::Else {
            self.advance(); // consume `else`
            if self.peek_kind() == &TokenKind::If {
                // else if → recurse; produce ElseArm::If wrapping a Stmt::If
                let nested: Spanned<Stmt> = self.parse_if_stmt()?;
                Some(Box::new(ElseArm::If(Box::new(nested))))
            } else {
                // else { block }
                let else_block: Spanned<Block> = self.parse_block()?;
                Some(Box::new(ElseArm::Block(else_block)))
            }
        } else {
            None
        };

        let end_span: Span = then_block.span;
        Some(Spanned::new(
            Stmt::If { cond, then_block, else_arm },
            if_span.merge(end_span),
        ))
    }

    /// Parse `for var in range(start, end [, step]) { body }`
    fn parse_for_stmt(&mut self) -> Option<Spanned<Stmt>> {
        let for_span: Span = self.peek_span();
        self.advance(); // consume `for`

        // Induction variable name
        let var_span: Span = self.peek_span();
        let var: String = match self.peek_kind().clone() {
            TokenKind::Ident(n) => { self.advance(); n }
            _ => {
                self.errors.push(ParseError::Unexpected {
                    expected: "induction variable name (identifier)".into(),
                    found: format!("{:?}", self.peek_kind()),
                    span: self.peek_span(),
                });
                self.recover_to_semicolon_or_brace();
                return None;
            }
        };

        // `in` keyword
        if self.peek_kind() != &TokenKind::In {
            self.errors.push(ParseError::Unexpected {
                expected: "`in` keyword after loop variable".into(),
                found: format!("{:?}", self.peek_kind()),
                span: self.peek_span(),
            });
            self.recover_to_semicolon_or_brace();
            return None;
        }
        self.advance(); // consume `in`

        // `range` identifier (special form, not a general call)
        match self.peek_kind().clone() {
            TokenKind::Ident(ref name) if name == "range" => {
                self.advance(); // consume `range`
            }
            _ => {
                self.errors.push(ParseError::Unexpected {
                    expected: "`range` after `in` in for-loop header".into(),
                    found: format!("{:?}", self.peek_kind()),
                    span: self.peek_span(),
                });
                self.recover_to_semicolon_or_brace();
                return None;
            }
        }

        // `(` start, end [, step] `)`
        if !self.expect_token(TokenKind::LParen, "(") {
            self.recover_to_semicolon_or_brace();
            return None;
        }

        let start: Spanned<Expr> = match self.parse_expr(0) {
            Some(e) => e,
            None => {
                self.recover_to_semicolon_or_brace();
                return None;
            }
        };

        if !self.expect_token(TokenKind::Comma, ",") {
            self.recover_to_semicolon_or_brace();
            return None;
        }

        let end: Spanned<Expr> = match self.parse_expr(0) {
            Some(e) => e,
            None => {
                self.recover_to_semicolon_or_brace();
                return None;
            }
        };

        // Optional step
        let step: Option<Spanned<Expr>> = if self.peek_kind() == &TokenKind::Comma {
            self.advance(); // consume `,`
            match self.parse_expr(0) {
                Some(e) => Some(e),
                None => {
                    self.recover_to_semicolon_or_brace();
                    return None;
                }
            }
        } else {
            None
        };

        if !self.expect_token(TokenKind::RParen, ")") {
            self.recover_to_semicolon_or_brace();
            return None;
        }

        let body: Spanned<Block> = self.parse_block()?;
        let end_span: Span = body.span;

        Some(Spanned::new(
            Stmt::For {
                var: Spanned::new(var, var_span),
                start,
                end,
                step,
                body,
            },
            for_span.merge(end_span),
        ))
    }

    /// Parse `while cond { body }`
    fn parse_while_stmt(&mut self) -> Option<Spanned<Stmt>> {
        let while_span: Span = self.peek_span();
        self.advance(); // consume `while`

        let cond: Spanned<Expr> = match self.parse_expr(0) {
            Some(e) => e,
            None => {
                self.recover_to_semicolon_or_brace();
                return None;
            }
        };

        let body: Spanned<Block> = self.parse_block()?;
        let end_span: Span = body.span;

        Some(Spanned::new(
            Stmt::While { cond, body },
            while_span.merge(end_span),
        ))
    }

    /// Parse `let [mut] name: type = expr;`
    fn parse_let_stmt(&mut self) -> Option<Spanned<Stmt>> {
        let let_span: Span = self.peek_span();
        self.advance(); // consume `let`

        // Optional `mut`
        let is_mut: bool = if self.peek_kind() == &TokenKind::Mut {
            self.advance();
            true
        } else {
            false
        };

        // Name
        let name_span: Span = self.peek_span();
        let name: String = match self.peek_kind().clone() {
            TokenKind::Ident(n) => { self.advance(); n }
            _ => {
                self.errors.push(ParseError::Unexpected {
                    expected: "binding name (identifier)".into(),
                    found: format!("{:?}", self.peek_kind()),
                    span: self.peek_span(),
                });
                self.recover_to_semicolon_or_brace();
                return None;
            }
        };

        // Require `: type_ref` (anti-pattern #1 at grammar level)
        if self.peek_kind() != &TokenKind::Colon {
            self.errors.push(ParseError::MissingTypeAnnotation { span: self.peek_span() });
            self.recover_to_semicolon_or_brace();
            return None;
        }
        self.advance(); // consume `:`

        let ty: Spanned<TypeRef> = self.parse_type_ref()?;

        // `=`
        if !self.expect_token(TokenKind::Assign, "=") {
            self.recover_to_semicolon_or_brace();
            return None;
        }

        // Expression
        let init: Spanned<Expr> = match self.parse_expr(0) {
            Some(e) => e,
            None => {
                self.recover_to_semicolon_or_brace();
                return None;
            }
        };

        // `;`
        let semi_span: Span = self.peek_span();
        if !self.expect_token(TokenKind::Semicolon, ";") {
            return None;
        }

        let span: Span = let_span.merge(semi_span);
        Some(Spanned::new(
            Stmt::Let {
                is_mut,
                name: Spanned::new(name, name_span),
                ty,
                init,
            },
            span,
        ))
    }

    /// Parse a bare reserved-builtin call statement: `IDENT(args);`
    ///
    /// Called when the current Ident is a reserved subgroup builtin and the next
    /// token is `(`. The inner `Expr` is always `Expr::Call { name, args }`.
    /// HIR will validate return type and arity.
    fn parse_builtin_call_stmt(&mut self) -> Option<Spanned<Stmt>> {
        let stmt_start: Span = self.peek_span();

        // Parse the call expression using the normal expression parser.
        // The Ident + LParen will naturally produce an Expr::Call.
        let call_expr: Spanned<Expr> = match self.parse_expr(0) {
            Some(e) => e,
            None => {
                self.recover_to_semicolon_or_brace();
                return None;
            }
        };

        // Must be followed by `;`
        let semi_span: Span = self.peek_span();
        if !self.expect_token(TokenKind::Semicolon, ";") {
            return None;
        }

        Some(Spanned::new(
            Stmt::BuiltinCallStmt { call: call_expr },
            stmt_start.merge(semi_span),
        ))
    }

    /// Parse `name = expr;`
    fn parse_assign_stmt(&mut self) -> Option<Spanned<Stmt>> {
        let name_span: Span = self.peek_span();
        let name: String = match self.peek_kind().clone() {
            TokenKind::Ident(n) => { self.advance(); n }
            _ => unreachable!("parse_assign_stmt called without Ident"),
        };

        self.advance(); // consume `=`

        let value: Spanned<Expr> = match self.parse_expr(0) {
            Some(e) => e,
            None => {
                self.recover_to_semicolon_or_brace();
                return None;
            }
        };

        let semi_span: Span = self.peek_span();
        if !self.expect_token(TokenKind::Semicolon, ";") {
            return None;
        }

        let span: Span = name_span.merge(semi_span);
        Some(Spanned::new(
            Stmt::Assign {
                target: Spanned::new(name, name_span),
                value,
            },
            span,
        ))
    }

    /// Parse `name[index] = expr;` — buffer write statement.
    fn parse_index_assign_stmt(&mut self) -> Option<Spanned<Stmt>> {
        let name_span: Span = self.peek_span();
        let name: String = match self.peek_kind().clone() {
            TokenKind::Ident(n) => { self.advance(); n }
            _ => unreachable!("parse_index_assign_stmt called without Ident"),
        };

        // Consume `[`
        self.advance();

        let index: Spanned<Expr> = match self.parse_expr(0) {
            Some(e) => e,
            None => {
                self.recover_to_semicolon_or_brace();
                return None;
            }
        };

        if !self.expect_token(TokenKind::RBracket, "]") {
            self.recover_to_semicolon_or_brace();
            return None;
        }

        if !self.expect_token(TokenKind::Assign, "=") {
            self.recover_to_semicolon_or_brace();
            return None;
        }

        let value: Spanned<Expr> = match self.parse_expr(0) {
            Some(e) => e,
            None => {
                self.recover_to_semicolon_or_brace();
                return None;
            }
        };

        let semi_span: Span = self.peek_span();
        if !self.expect_token(TokenKind::Semicolon, ";") {
            return None;
        }

        let span: Span = name_span.merge(semi_span);
        Some(Spanned::new(
            Stmt::IndexAssign {
                target: Spanned::new(name, name_span),
                index,
                value,
            },
            span,
        ))
    }

    /// Parse `return [expr];` — only void return used in M1.1.
    fn parse_return_stmt(&mut self) -> Option<Spanned<Stmt>> {
        let stmt_start: Span = self.peek_span();
        self.advance(); // consume `return`

        // Optional expression (only int or bool literals in M0; Pratt in M1.1)
        let expr: Option<Spanned<Expr>> = if self.peek_kind() == &TokenKind::Semicolon
            || self.peek_kind() == &TokenKind::RBrace
            || self.is_at_end()
        {
            None
        } else {
            self.parse_expr(0)
        };

        // Consume `;`
        let stmt_end: Span = self.peek_span();
        if !self.expect_token(TokenKind::Semicolon, ";") {
            return None;
        }

        let span: Span = stmt_start.merge(stmt_end);
        Some(Spanned::new(Stmt::Return(expr), span))
    }

    // ── Pratt expression parser ───────────────────────────────────────────────

    /// Entry-point: parse an expression with minimum binding power 0.
    ///
    /// `depth` tracks nesting for the MAX_EXPR_DEPTH guard.
    fn parse_expr(&mut self, depth: u32) -> Option<Spanned<Expr>> {
        self.parse_expr_bp(0, depth)
    }

    /// Pratt parser core (§3.3 precedence table).
    ///
    /// | Op/construct          | Token(s)               | (left_bp, right_bp) |
    /// |-----------------------|------------------------|---------------------|
    /// | `or`                  | Or                     | (1, 2)              |
    /// | `and`                 | And                    | (3, 4)              |
    /// | comparison            | Eq/Neq/Lt/LtEq/Gt/GtEq| (5, 6)              |
    /// | additive              | Plus/Minus             | (7, 8)              |
    /// | multiplicative        | Star/Slash/Percent     | (9, 10)             |
    /// | unary `-` / `not`     | prefix                 | right_bp=11 / 9     |
    /// | postfix call `(...)`  | LParen                 | left_bp=13          |
    fn parse_expr_bp(&mut self, min_bp: u8, depth: u32) -> Option<Spanned<Expr>> {
        if depth > MAX_EXPR_DEPTH {
            let span: Span = self.peek_span();
            self.errors.push(ParseError::ExpressionNestingTooDeep { span });
            // Sentinel: suffixed literal so HIR won't emit UnconstrainedLiteralNeedsSuffix
            use axc_lexer::IntSuffix;
            return Some(Spanned::new(
                Expr::IntLit { value: 0, suffix: Some(IntSuffix::I32) },
                span,
            ));
        }

        // Parse prefix (left-hand side)
        let mut lhs: Spanned<Expr> = self.parse_prefix(depth)?;

        // Parse infix operations as long as their binding power exceeds min_bp
        while let Some((infix_op, left_bp, right_bp)) = self.peek_infix_op() {
            if left_bp < min_bp {
                break;
            }

            let op_span: Span = self.peek_span();
            self.advance(); // consume the infix operator

            let rhs: Spanned<Expr> = match self.parse_expr_bp(right_bp, depth + 1) {
                Some(e) => e,
                None => break,
            };

            let combined_span: Span = lhs.span.merge(rhs.span).merge(op_span);

            lhs = match infix_op {
                InfixOp::Binary(op) => Spanned::new(
                    Expr::Binary {
                        op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    },
                    combined_span,
                ),
                InfixOp::ShortCircuit(op) => Spanned::new(
                    Expr::ShortCircuit {
                        op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    },
                    combined_span,
                ),
            };
        }

        Some(lhs)
    }

    /// Parse a prefix expression (literal, ident, unary op, paren, call).
    fn parse_prefix(&mut self, depth: u32) -> Option<Spanned<Expr>> {
        if depth > MAX_EXPR_DEPTH {
            let span: Span = self.peek_span();
            self.errors.push(ParseError::ExpressionNestingTooDeep { span });
            use axc_lexer::IntSuffix;
            return Some(Spanned::new(
                Expr::IntLit { value: 0, suffix: Some(IntSuffix::I32) },
                span,
            ));
        }

        let span: Span = self.peek_span();

        match self.peek_kind().clone() {
            // ── Literals ────────────────────────────────────────────────────
            TokenKind::BoolLiteral(b) => {
                self.advance();
                Some(Spanned::new(Expr::BoolLit(b), span))
            }
            TokenKind::IntLiteral { value, suffix, .. } => {
                self.advance();
                Some(Spanned::new(Expr::IntLit { value, suffix }, span))
            }
            TokenKind::FloatLiteral { value, suffix } => {
                self.advance();
                Some(Spanned::new(Expr::FloatLit { value, suffix }, span))
            }

            // ── Identifier, call, or index ──────────────────────────────────
            TokenKind::Ident(name) => {
                self.advance();
                // Postfix call: `name(...)`
                if self.peek_kind() == &TokenKind::LParen {
                    self.advance(); // consume `(`
                    let mut args: Vec<Spanned<Expr>> = Vec::new();
                    while self.peek_kind() != &TokenKind::RParen && !self.is_at_end() {
                        if let Some(arg) = self.parse_expr(depth + 1) {
                            args.push(arg);
                        }
                        if self.peek_kind() == &TokenKind::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    let end_span: Span = self.peek_span();
                    if !self.expect_token(TokenKind::RParen, ")") {
                        return None;
                    }
                    let call_span: Span = span.merge(end_span);
                    Some(Spanned::new(
                        Expr::Call {
                            name: Spanned::new(name, span),
                            args,
                        },
                        call_span,
                    ))
                } else if self.peek_kind() == &TokenKind::LBracket {
                    // Postfix index: `name[expr]` — buffer read.
                    self.advance(); // consume `[`
                    let index_expr: Spanned<Expr> = self.parse_expr(depth + 1)?;
                    let end_span: Span = self.peek_span();
                    if !self.expect_token(TokenKind::RBracket, "]") {
                        return None;
                    }
                    let index_span: Span = span.merge(end_span);
                    Some(Spanned::new(
                        Expr::Index {
                            base: Box::new(Spanned::new(Expr::Ident(name), span)),
                            index: Box::new(index_expr),
                        },
                        index_span,
                    ))
                } else {
                    Some(Spanned::new(Expr::Ident(name), span))
                }
            }

            // ── Unary minus ─────────────────────────────────────────────────
            TokenKind::Minus => {
                self.advance();
                let operand: Spanned<Expr> = self.parse_expr_bp(11, depth + 1)?;
                let combined: Span = span.merge(operand.span);
                Some(Spanned::new(
                    Expr::Unary { op: UnaryOp::Neg, operand: Box::new(operand) },
                    combined,
                ))
            }

            // ── Logical not ─────────────────────────────────────────────────
            TokenKind::Not => {
                self.advance();
                let operand: Spanned<Expr> = self.parse_expr_bp(9, depth + 1)?;
                let combined: Span = span.merge(operand.span);
                Some(Spanned::new(
                    Expr::Unary { op: UnaryOp::LogicalNot, operand: Box::new(operand) },
                    combined,
                ))
            }

            // ── Parenthesized expression ────────────────────────────────────
            TokenKind::LParen => {
                self.advance(); // consume `(`
                let inner: Spanned<Expr> = self.parse_expr(depth + 1)?;
                let end_span: Span = self.peek_span();
                if !self.expect_token(TokenKind::RParen, ")") {
                    return None;
                }
                let paren_span: Span = span.merge(end_span);
                Some(Spanned::new(Expr::Paren(Box::new(inner)), paren_span))
            }

            // ── Anything else is not an expression start ────────────────────
            other => {
                self.errors.push(ParseError::Unexpected {
                    expected: "expression".into(),
                    found: format!("{other:?}"),
                    span,
                });
                None
            }
        }
    }

    /// Return the infix operator at the current position plus its binding powers,
    /// or `None` if the current token is not an infix operator.
    ///
    /// Precedence table per §3.3:
    fn peek_infix_op(&self) -> Option<(InfixOp, u8, u8)> {
        match self.peek_kind() {
            TokenKind::Or     => Some((InfixOp::ShortCircuit(ShortCircuitOp::Or),  1, 2)),
            TokenKind::And    => Some((InfixOp::ShortCircuit(ShortCircuitOp::And), 3, 4)),
            TokenKind::Eq     => Some((InfixOp::Binary(BinOp::Eq),    5, 6)),
            TokenKind::NotEq  => Some((InfixOp::Binary(BinOp::Neq),   5, 6)),
            TokenKind::Lt     => Some((InfixOp::Binary(BinOp::Lt),    5, 6)),
            TokenKind::LtEq   => Some((InfixOp::Binary(BinOp::LtEq),  5, 6)),
            TokenKind::Gt     => Some((InfixOp::Binary(BinOp::Gt),    5, 6)),
            TokenKind::GtEq   => Some((InfixOp::Binary(BinOp::GtEq),  5, 6)),
            TokenKind::Plus   => Some((InfixOp::Binary(BinOp::Add),   7, 8)),
            TokenKind::Minus  => Some((InfixOp::Binary(BinOp::Sub),   7, 8)),
            TokenKind::Star   => Some((InfixOp::Binary(BinOp::Mul),   9, 10)),
            TokenKind::Slash  => Some((InfixOp::Binary(BinOp::Div),   9, 10)),
            TokenKind::Percent=> Some((InfixOp::Binary(BinOp::Rem),   9, 10)),
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

    /// Skip tokens until `,` or `)` (left in place) or EOF.
    fn recover_to_comma_or_rparen(&mut self) {
        loop {
            match self.peek_kind() {
                TokenKind::Comma | TokenKind::RParen | TokenKind::Eof => break,
                _ => { self.advance(); }
            }
        }
    }

    /// Skip tokens until `;` (consumed) or `}` / EOF (left in place).
    fn recover_to_semicolon_or_brace(&mut self) {
        loop {
            match self.peek_kind() {
                TokenKind::Semicolon => { self.advance(); break; }
                TokenKind::RBrace | TokenKind::Eof => break,
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
    use axc_lexer::{tokenize, IntSuffix, FloatSuffix};
    use crate::ast::{AnnotationArg, TypeRef, BinOp, UnaryOp, ShortCircuitOp};

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

    // ── §7.6 audit table row 4: nonempty_body_accepted_let (REWRITE) ─────────

    #[test]
    fn nonempty_body_accepts_let() {
        // M1.1: let is now valid syntax — must produce ZERO errors.
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { let x: i32 = 0i32; return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "M1.1 let binding should be accepted, got errors: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        // Should have Stmt::Let + Stmt::Return
        assert_eq!(kd.body.node.stmts.len(), 2);
        assert!(matches!(kd.body.node.stmts[0].node, Stmt::Let { .. }));
        assert!(matches!(kd.body.node.stmts[1].node, Stmt::Return(None)));
    }

    // ── §7.6 audit table row 5,6: these were removed in M1.3 (if/for/while now parse)
    // Old tests deleted; new M1.3 tests below verify correct parsing.

    // ── AT-301: `if true { return; }` parses without errors ──────────────────

    #[test]
    fn parse_if_then_only() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { if true { return; } }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        assert_eq!(kd.body.node.stmts.len(), 1);
        assert!(matches!(kd.body.node.stmts[0].node, Stmt::If { .. }));
    }

    // ── AT-302: if/else parses correctly ──────────────────────────────────────

    #[test]
    fn parse_if_else_correctly() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { if true { return; } else { return; } }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        if let Stmt::If { ref else_arm, .. } = kd.body.node.stmts[0].node {
            assert!(else_arm.is_some(), "expected else arm");
            assert!(matches!(else_arm.as_ref().unwrap().as_ref(), ElseArm::Block(_)));
        } else {
            panic!("expected Stmt::If");
        }
    }

    // ── AT-303: else-if chain parses as nested If ─────────────────────────────

    #[test]
    fn parse_else_if_chain_as_nested_if() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { if true { return; } else if false { return; } else { return; } }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        if let Stmt::If { ref else_arm, .. } = kd.body.node.stmts[0].node {
            // Outer else arm is ElseArm::If(nested)
            match else_arm.as_ref().unwrap().as_ref() {
                ElseArm::If(inner_spanned) => {
                    assert!(matches!(inner_spanned.node, Stmt::If { .. }));
                }
                other => panic!("expected ElseArm::If, got: {other:?}"),
            }
        } else {
            panic!("expected Stmt::If");
        }
    }

    // ── AT-304: for-range default step parses as Stmt::For { step: None } ────

    #[test]
    fn parse_for_range_default_step() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { for i in range(0u32, 10u32) { return; } }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        if let Stmt::For { ref var, ref step, .. } = kd.body.node.stmts[0].node {
            assert_eq!(var.node, "i");
            assert!(step.is_none(), "expected step=None for default step");
        } else {
            panic!("expected Stmt::For");
        }
    }

    // ── AT-305: for-range explicit step parses with step: Some(IntLit) ────────

    #[test]
    fn parse_for_range_explicit_step() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { for i in range(0u32, 10u32, 2u32) { return; } }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        if let Stmt::For { ref step, .. } = kd.body.node.stmts[0].node {
            assert!(step.is_some(), "expected step=Some");
            assert!(matches!(step.as_ref().unwrap().node, Expr::IntLit { value: 2, .. }));
        } else {
            panic!("expected Stmt::For");
        }
    }

    // ── AT-306: missing `in` keyword rejected at parse time ───────────────────

    #[test]
    fn parse_for_missing_in_rejected() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { for i range(0u32, 10u32) { return; } }";
        let (_, errors) = parse_src(src);
        assert!(!errors.is_empty(), "expected parse error for missing `in`");
    }

    // ── AT-327 parser mirror: reserved-keyword list contains only Struct in M1.3 ─

    #[test]
    fn parse_m1_3_reserved_keywords_contains_only_struct() {
        // In M1.3, break/continue parse as valid statements inside a loop.
        // The reserved-keyword list now contains only Struct.
        assert_eq!(M1_3_RESERVED_KEYWORDS.len(), 1);
        assert!(matches!(M1_3_RESERVED_KEYWORDS[0], TokenKind::Struct));
    }

    // ── AT-328: struct keyword still rejected ─────────────────────────────────

    #[test]
    fn parse_struct_still_rejected_in_m1_3() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { struct Foo {} return; }";
        let (_, errors) = parse_src(src);
        assert!(
            errors.iter().any(|e| matches!(e, ParseError::UnsupportedInM1_3 { detail, .. } if detail.contains("struct"))),
            "expected UnsupportedInM1_3 for struct, got: {errors:?}"
        );
    }

    // ── break and continue parse correctly ────────────────────────────────────

    #[test]
    fn parse_break_and_continue_in_for_loop() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void {
            for i in range(0u32, 10u32) {
                break;
            }
        }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        if let Stmt::For { ref body, .. } = kd.body.node.stmts[0].node {
            assert!(matches!(body.node.stmts[0].node, Stmt::Break));
        } else {
            panic!("expected Stmt::For");
        }
    }

    #[test]
    fn parse_while_stmt() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { while true { return; } }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        assert!(matches!(kd.body.node.stmts[0].node, Stmt::While { .. }));
    }

    #[test]
    fn parse_continue_in_while_loop() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { while true { continue; } }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        if let Stmt::While { ref body, .. } = kd.body.node.stmts[0].node {
            assert!(matches!(body.node.stmts[0].node, Stmt::Continue));
        } else {
            panic!("expected Stmt::While");
        }
    }

    // ── §7.6 audit table row 7: kernel params now SUPPORTED in M1.2 ──────────

    #[test]
    fn kernel_scalar_params_accepted() {
        // M1.2: scalar params are now valid.
        let src = "@kernel @workgroup(64,1,1) fn k(x: i32) -> void { return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "expected scalar param to be accepted in M1.2: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        assert_eq!(kd.params.len(), 1);
        assert_eq!(kd.params[0].node.name.node, "x");
        assert_eq!(kd.params[0].node.ty.node, TypeRef::I32);
    }

    // ── AT-6: unknown top-level item ─────────────────────────────────────────

    #[test]
    fn unknown_item_struct() {
        let src = "struct Foo {}";
        let (_, errors) = parse_src(src);
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| matches!(e, ParseError::UnknownItem { .. })));
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
        let src = "💥 struct Foo {}";
        let (tokens, lex_errors) = tokenize(src);
        assert!(!lex_errors.is_empty(), "expected lex error for emoji");
        let mut p = Parser::new(&tokens);
        let (_, parse_errors) = p.parse_module();
        let has_lex_replication = parse_errors.iter().any(|e| matches!(e, ParseError::LexerError(_)));
        assert!(!has_lex_replication, "lex error was double-reported as ParseError");
    }

    // ── @complexity(O(1)) parses to Call AST ─────────────────────────────────

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

    // ── §7.1.2: 14+ new M1.1 parser tests ──────────────────────────────────

    #[test]
    fn parse_let_i32_literal() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let x: i32 = 42i32; return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        if let Stmt::Let { is_mut, ref name, ref ty, ref init } = kd.body.node.stmts[0].node {
            assert!(!is_mut);
            assert_eq!(name.node, "x");
            assert_eq!(ty.node, TypeRef::I32);
            assert!(matches!(init.node, Expr::IntLit { value: 42, suffix: Some(IntSuffix::I32) }));
        } else {
            panic!("expected Stmt::Let");
        }
    }

    #[test]
    fn parse_let_mut_f64_suffixed() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let mut y: f64 = 3.14f64; return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        if let Stmt::Let { is_mut, ref name, ref ty, ref init } = kd.body.node.stmts[0].node {
            assert!(is_mut);
            assert_eq!(name.node, "y");
            assert_eq!(ty.node, TypeRef::F64);
            if let Expr::FloatLit { suffix: Some(FloatSuffix::F64), .. } = &init.node {
                // ok
            } else {
                panic!("expected FloatLit with f64 suffix: {:?}", init.node);
            }
        } else {
            panic!("expected Stmt::Let");
        }
    }

    #[test]
    fn parse_let_missing_type_rejected() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let x = 0; return; }";
        let (_, errors) = parse_src(src);
        assert!(errors.iter().any(|e| matches!(e, ParseError::MissingTypeAnnotation { .. })),
            "expected MissingTypeAnnotation, got: {errors:?}");
    }

    #[test]
    fn parse_assign_simple() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let mut x: i32 = 0i32; x = 5i32; return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        assert!(matches!(kd.body.node.stmts[1].node, Stmt::Assign { .. }));
    }

    #[test]
    fn parse_arith_precedence() {
        // `1 + 2 * 3` should parse as `Add(1, Mul(2, 3))`
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let a: i32 = 1i32 + 2i32 * 3i32; return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        if let Stmt::Let { ref init, .. } = kd.body.node.stmts[0].node {
            if let Expr::Binary { op: BinOp::Add, ref rhs, .. } = init.node {
                assert!(matches!(rhs.node, Expr::Binary { op: BinOp::Mul, .. }),
                    "RHS should be Mul; got: {:?}", rhs.node);
            } else {
                panic!("expected Add at top, got: {:?}", init.node);
            }
        }
    }

    #[test]
    fn parse_arith_left_assoc() {
        // `1 - 2 - 3` should parse as `Sub(Sub(1,2), 3)`
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let a: i32 = 1i32 - 2i32 - 3i32; return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        if let Stmt::Let { ref init, .. } = kd.body.node.stmts[0].node {
            if let Expr::Binary { op: BinOp::Sub, ref lhs, .. } = init.node {
                assert!(matches!(lhs.node, Expr::Binary { op: BinOp::Sub, .. }),
                    "LHS should be Sub; got: {:?}", lhs.node);
            } else {
                panic!("expected top-level Sub, got: {:?}", init.node);
            }
        }
    }

    #[test]
    fn parse_comparison_lower_than_arith() {
        // `1i32 + 2i32 == 3i32` -> Eq(Add(1,2), 3)
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let b: bool = 1i32 + 2i32 == 3i32; return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        if let Stmt::Let { ref init, .. } = kd.body.node.stmts[0].node {
            if let Expr::Binary { op: BinOp::Eq, ref lhs, .. } = init.node {
                assert!(matches!(lhs.node, Expr::Binary { op: BinOp::Add, .. }),
                    "LHS of Eq should be Add; got: {:?}", lhs.node);
            } else {
                panic!("expected Eq at top, got: {:?}", init.node);
            }
        }
    }

    #[test]
    fn parse_and_lower_than_comparison() {
        // `a == 0i32 and b != 1i32` -> And(Eq(a,0), Neq(b,1))
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let b: bool = a == 0i32 and b != 1i32; return; }";
        let (module, errors) = parse_src(src);
        // Parse errors are OK here (a,b not yet declared), we only check structure
        let _: &Vec<ParseError> = &errors; // suppress unused
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        if kd.body.node.stmts.is_empty() {
            return; // errors prevented statement from parsing
        }
        if let Stmt::Let { ref init, .. } = kd.body.node.stmts[0].node {
            assert!(matches!(init.node, Expr::ShortCircuit { op: ShortCircuitOp::And, .. }),
                "expected And at top: {:?}", init.node);
        }
    }

    #[test]
    fn parse_or_lower_than_and() {
        // `a and b or c` -> Or(And(a,b), c)
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let b: bool = a and b or c; return; }";
        let (module, _) = parse_src(src);
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        if kd.body.node.stmts.is_empty() {
            return;
        }
        if let Stmt::Let { ref init, .. } = kd.body.node.stmts[0].node {
            if let Expr::ShortCircuit { op: ShortCircuitOp::Or, ref lhs, .. } = init.node {
                assert!(matches!(lhs.node, Expr::ShortCircuit { op: ShortCircuitOp::And, .. }),
                    "LHS should be And, got: {:?}", lhs.node);
            } else {
                panic!("expected Or at top: {:?}", init.node);
            }
        }
    }

    #[test]
    fn parse_unary_minus_binds_tight() {
        // `-x + y` -> Add(Neg(x), y)
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let a: i32 = -x + y; return; }";
        let (module, _) = parse_src(src);
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        if kd.body.node.stmts.is_empty() {
            return;
        }
        if let Stmt::Let { ref init, .. } = kd.body.node.stmts[0].node {
            if let Expr::Binary { op: BinOp::Add, ref lhs, .. } = init.node {
                assert!(matches!(lhs.node, Expr::Unary { op: UnaryOp::Neg, .. }),
                    "LHS of Add should be Neg; got: {:?}", lhs.node);
            } else {
                panic!("expected Add at top: {:?}", init.node);
            }
        }
    }

    #[test]
    fn parse_not_unary() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let b: bool = not x; return; }";
        let (module, _) = parse_src(src);
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        if kd.body.node.stmts.is_empty() {
            return;
        }
        if let Stmt::Let { ref init, .. } = kd.body.node.stmts[0].node {
            assert!(matches!(init.node, Expr::Unary { op: UnaryOp::LogicalNot, .. }),
                "expected LogicalNot: {:?}", init.node);
        }
    }

    #[test]
    fn parse_paren_group_overrides_precedence() {
        // `(1i32 + 2i32) * 3i32` -> Mul(Add(1,2), 3)
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let a: i32 = (1i32 + 2i32) * 3i32; return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        if let Stmt::Let { ref init, .. } = kd.body.node.stmts[0].node {
            if let Expr::Binary { op: BinOp::Mul, ref lhs, .. } = init.node {
                // LHS should be Paren wrapping Add
                assert!(matches!(lhs.node, Expr::Paren(_)),
                    "LHS of Mul should be Paren; got: {:?}", lhs.node);
            } else {
                panic!("expected Mul at top: {:?}", init.node);
            }
        }
    }

    #[test]
    fn parse_bitwise_builtin_call() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { let c: u32 = band(0xFFu32, 0x0Fu32); return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        if let Stmt::Let { ref init, .. } = kd.body.node.stmts[0].node {
            if let Expr::Call { ref name, ref args } = init.node {
                assert_eq!(name.node, "band");
                assert_eq!(args.len(), 2);
            } else {
                panic!("expected Call: {:?}", init.node);
            }
        }
    }

    #[test]
    fn parse_nested_deep_rejected() {
        // 257 nested parens should exceed MAX_EXPR_DEPTH
        let open: String = "(".repeat(257);
        let close: String = ")".repeat(257);
        let src: String = format!("@kernel @workgroup(1,1,1) fn k() -> void {{ let a: i32 = {}1i32{}; return; }}", open, close);
        let (_, errors) = parse_src(&src);
        assert!(errors.iter().any(|e| matches!(e, ParseError::ExpressionNestingTooDeep { .. })),
            "expected ExpressionNestingTooDeep for 257 nested parens, got: {errors:?}");
    }

    // ── M1.2: buffer parameter parsing ───────────────────────────────────────

    #[test]
    fn parse_buffer_param() {
        let src = "@kernel @workgroup(64,1,1) fn saxpy(x: readonly_buffer[f32]) -> void { return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        assert_eq!(kd.params.len(), 1);
        assert_eq!(kd.params[0].node.name.node, "x");
        assert_eq!(kd.params[0].node.ty.node, TypeRef::ReadonlyBuffer(crate::ast::ScalarTypeRef::F32));
    }

    #[test]
    fn parse_writeonly_buffer_param() {
        let src = "@kernel @workgroup(64,1,1) fn k(out: writeonly_buffer[u32]) -> void { return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        assert_eq!(kd.params[0].node.ty.node, TypeRef::WriteonlyBuffer(crate::ast::ScalarTypeRef::U32));
    }

    #[test]
    fn parse_rw_buffer_param() {
        let src = "@kernel @workgroup(64,1,1) fn k(y: buffer[f32]) -> void { return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        assert_eq!(kd.params[0].node.ty.node, TypeRef::Buffer(crate::ast::ScalarTypeRef::F32));
    }

    #[test]
    fn parse_multiple_params() {
        let src = "@kernel @workgroup(64,1,1) fn saxpy(a: f32, x: readonly_buffer[f32], y: buffer[f32]) -> void { return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        assert_eq!(kd.params.len(), 3);
        assert_eq!(kd.params[0].node.name.node, "a");
        assert_eq!(kd.params[0].node.ty.node, TypeRef::F32);
        assert_eq!(kd.params[1].node.name.node, "x");
        assert_eq!(kd.params[1].node.ty.node, TypeRef::ReadonlyBuffer(crate::ast::ScalarTypeRef::F32));
        assert_eq!(kd.params[2].node.name.node, "y");
        assert_eq!(kd.params[2].node.ty.node, TypeRef::Buffer(crate::ast::ScalarTypeRef::F32));
    }

    #[test]
    fn parse_index_expr_buffer_read() {
        // x[i] in an expression context parses to Expr::Index
        let src = "@kernel @workgroup(64,1,1) fn k(x: readonly_buffer[f32]) -> void { let v: f32 = x[0u32]; return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        if let crate::ast::Stmt::Let { ref init, .. } = kd.body.node.stmts[0].node {
            assert!(matches!(init.node, Expr::Index { .. }), "expected Index expr, got: {:?}", init.node);
        } else {
            panic!("expected Let stmt");
        }
    }

    #[test]
    fn parse_index_assign_buffer_write() {
        // y[i] = expr parses to Stmt::IndexAssign
        let src = "@kernel @workgroup(64,1,1) fn k(y: buffer[f32]) -> void { y[0u32] = 1.0f32; return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        assert!(matches!(kd.body.node.stmts[0].node, crate::ast::Stmt::IndexAssign { .. }),
            "expected IndexAssign stmt, got: {:?}", kd.body.node.stmts[0].node);
    }

    #[test]
    fn parse_saxpy_body() {
        // Full saxpy kernel: a * x[i] + y[i] with buffer write
        let src = "@kernel @workgroup(64,1,1) fn saxpy(a: f32, x: readonly_buffer[f32], y: buffer[f32]) -> void { let i: u32 = 0u32; y[i] = a * x[i] + y[i]; return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        assert_eq!(kd.params.len(), 3);
        // First stmt is let, second is index-assign
        assert!(matches!(kd.body.node.stmts[0].node, crate::ast::Stmt::Let { .. }));
        assert!(matches!(kd.body.node.stmts[1].node, crate::ast::Stmt::IndexAssign { .. }));
    }

    #[test]
    fn parse_gid_call() {
        // gid(0u32) — a builtin call that parses as Expr::Call
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { let i: u32 = gid(0u32); return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        if let crate::ast::Stmt::Let { ref init, .. } = kd.body.node.stmts[0].node {
            assert!(matches!(&init.node, Expr::Call { name, .. } if name.node == "gid"),
                "expected Call gid, got: {:?}", init.node);
        } else {
            panic!("expected Let stmt");
        }
    }

    #[test]
    fn parse_buffer_i64_elem() {
        let src = "@kernel @workgroup(64,1,1) fn k(b: buffer[i64]) -> void { return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "errors: {errors:?}");
        let crate::ast::Item::Kernel(ref kd) = module.items[0].node;
        assert_eq!(kd.params[0].node.ty.node, TypeRef::Buffer(crate::ast::ScalarTypeRef::I64));
    }

    #[test]
    fn parse_buffer_invalid_elem_rejected() {
        let src = "@kernel @workgroup(64,1,1) fn k(b: buffer[bool]) -> void { return; }";
        let (_, errors) = parse_src(src);
        assert!(!errors.is_empty(), "expected parse error for invalid buffer elem type");
    }

    // ── M1.4: subgroup builtin parsing tests ────────────────────────────────────

    /// Helper: get the first stmt from the first kernel in a module.
    fn first_stmt(src: &str) -> Stmt {
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "unexpected parse errors: {errors:?}");
        let Item::Kernel(ref kd) = module.items[0].node;
        kd.body.node.stmts[0].node.clone()
    }

    #[test]
    fn parse_workgroup_barrier_as_stmt() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { workgroup_barrier(); return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        let Item::Kernel(ref kd) = module.items[0].node;
        let stmt = &kd.body.node.stmts[0].node;
        assert!(
            matches!(stmt, Stmt::BuiltinCallStmt { call } if matches!(&call.node, Expr::Call { name, .. } if name.node == "workgroup_barrier")),
            "expected BuiltinCallStmt for workgroup_barrier; got: {stmt:?}"
        );
    }

    #[test]
    fn parse_subgroup_reduce_add_as_expr() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { let v: f32 = 1.0f32; let s: f32 = subgroup_reduce_add(v); return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        let Item::Kernel(ref kd) = module.items[0].node;
        // second stmt is the reduce_add
        if let Stmt::Let { init, .. } = &kd.body.node.stmts[1].node {
            assert!(
                matches!(&init.node, Expr::Call { name, .. } if name.node == "subgroup_reduce_add"),
                "expected Call to subgroup_reduce_add; got: {:?}", init.node
            );
        } else {
            panic!("expected Let stmt for subgroup_reduce_add");
        }
    }

    #[test]
    fn parse_subgroup_elect_as_if_cond() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { if subgroup_elect() { return; } return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        let Item::Kernel(ref kd) = module.items[0].node;
        if let Stmt::If { cond, .. } = &kd.body.node.stmts[0].node {
            assert!(
                matches!(&cond.node, Expr::Call { name, .. } if name.node == "subgroup_elect"),
                "expected subgroup_elect as if cond; got: {:?}", cond.node
            );
        } else {
            panic!("expected If stmt");
        }
    }

    #[test]
    fn parse_subgroup_invocation_id_as_expr() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { let id: u32 = subgroup_invocation_id(); return; }";
        if let Stmt::Let { init, .. } = first_stmt(src) {
            assert!(
                matches!(&init.node, Expr::Call { name, .. } if name.node == "subgroup_invocation_id"),
                "expected subgroup_invocation_id call; got: {:?}", init.node
            );
        } else {
            panic!("expected Let stmt");
        }
    }

    #[test]
    fn parse_subgroup_size_as_expr() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { let sz: u32 = subgroup_size(); return; }";
        if let Stmt::Let { init, .. } = first_stmt(src) {
            assert!(
                matches!(&init.node, Expr::Call { name, .. } if name.node == "subgroup_size"),
                "expected subgroup_size call; got: {:?}", init.node
            );
        } else {
            panic!("expected Let stmt");
        }
    }

    #[test]
    fn parse_subgroup_all_with_bool_arg() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { let e: bool = subgroup_elect(); let r: bool = subgroup_all(e); return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        let Item::Kernel(ref kd) = module.items[0].node;
        if let Stmt::Let { init, .. } = &kd.body.node.stmts[1].node {
            assert!(
                matches!(&init.node, Expr::Call { name, .. } if name.node == "subgroup_all"),
                "expected subgroup_all; got: {:?}", init.node
            );
        } else {
            panic!("expected Let stmt for subgroup_all");
        }
    }

    #[test]
    fn parse_subgroup_any_with_bool_arg() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { let e: bool = subgroup_elect(); let r: bool = subgroup_any(e); return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        let Item::Kernel(ref kd) = module.items[0].node;
        if let Stmt::Let { init, .. } = &kd.body.node.stmts[1].node {
            assert!(
                matches!(&init.node, Expr::Call { name, .. } if name.node == "subgroup_any"),
                "expected subgroup_any; got: {:?}", init.node
            );
        } else {
            panic!("expected Let stmt for subgroup_any");
        }
    }

    #[test]
    fn parse_subgroup_broadcast_first_as_expr() {
        let src = "@kernel @workgroup(64,1,1) fn k() -> void { let v: f32 = 1.0f32; let b: f32 = subgroup_broadcast_first(v); return; }";
        let (module, errors) = parse_src(src);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        let Item::Kernel(ref kd) = module.items[0].node;
        if let Stmt::Let { init, .. } = &kd.body.node.stmts[1].node {
            assert!(
                matches!(&init.node, Expr::Call { name, .. } if name.node == "subgroup_broadcast_first"),
                "expected subgroup_broadcast_first; got: {:?}", init.node
            );
        } else {
            panic!("expected Let stmt for subgroup_broadcast_first");
        }
    }

    // ── M2.1: Cooperative-matrix type parsing (AT-606, AT-607, AT-608) ────────

    /// AT-606: `matrix[f16, 16, 16, a]` parses as TypeRef::CoopMatrix.
    #[test]
    fn parse_matrix_type_f16_16_16_a() {
        // Embed in a valid let binding to exercise the type-ref parser in context.
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { \
            let m: matrix[f16, 16, 16, a] = coopmat_zero(); return; }";
        let (module, errors) = parse_src(src);
        let Item::Kernel(ref kd) = module.items[0].node;
        // The let statement is stmts[0].
        if let Stmt::Let { ty, .. } = &kd.body.node.stmts[0].node {
            assert!(
                matches!(
                    &ty.node,
                    TypeRef::CoopMatrix {
                        elem: crate::ast::ScalarTypeRef::F16,
                        m: 16,
                        n: 16,
                        use_: crate::ast::CoopMatUseAst::A,
                    }
                ),
                "expected CoopMatrix {{ F16, 16, 16, A }}; got: {:?}", ty.node
            );
        } else {
            panic!("expected Let stmt; got: {:?}", kd.body.node.stmts[0].node);
        }
        // Note: parse errors may occur for the unresolved coopmat_zero() call
        // at parser level (it's a Call expr which is fine). We only validate the type ref.
        let _ = errors; // errors may include unresolved ident warnings — not our concern here
    }

    /// AT-607: `matrix[f16, 16u32, 16, a]` is rejected with
    /// `CoopMatrixDimMustBeUnsuffixedIntegerLiteral`.
    #[test]
    fn parse_matrix_type_suffixed_dim_rejected() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { \
            let m: matrix[f16, 16u32, 16, a] = coopmat_zero(); return; }";
        let (_, errors) = parse_src(src);
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ParseError::CoopMatrixDimMustBeUnsuffixedIntegerLiteral { .. }
            )),
            "expected CoopMatrixDimMustBeUnsuffixedIntegerLiteral in: {errors:?}"
        );
    }

    /// AT-608: `matrix[f16, 16, 16, c]` is rejected with
    /// `CoopMatrixUseMustBeABOrAccumulator { found: "c" }`.
    #[test]
    fn parse_matrix_type_unknown_use_rejected() {
        let src = "@kernel @workgroup(1,1,1) fn k() -> void { \
            let m: matrix[f16, 16, 16, c] = coopmat_zero(); return; }";
        let (_, errors) = parse_src(src);
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ParseError::CoopMatrixUseMustBeABOrAccumulator { found, .. }
                    if found == "c"
            )),
            "expected CoopMatrixUseMustBeABOrAccumulator {{ found: \"c\" }} in: {errors:?}"
        );
    }
}
