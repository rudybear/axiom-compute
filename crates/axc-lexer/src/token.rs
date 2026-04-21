//! Token types for AXIOM-Compute (.axc) source files.
//!
//! Ported from axiom-lexer/token.rs with GPU-specific additions and removal
//! of CPU-only variants (noalias, nsw, fast-math, arena, lifetime, etc.).
//! M1-reserved keywords are included as dedicated variants so the parser's
//! §3.3 deny-list check can match them with a typed `match` rather than
//! string comparison.

/// Byte offset span in source text (half-open: `start..end`).
///
/// Spans are raw-file-aligned: no BOM stripping, no whitespace trimming.
/// The driver rejects BOM-prefixed files before any tokenization, so
/// offset 0 always refers to the first non-BOM byte of the source.
///
/// `From<Span> for miette::SourceSpan` is implemented so that `#[label]`
/// attributes in `thiserror` + `miette::Diagnostic` derive macros work correctly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl From<Span> for miette::SourceSpan {
    fn from(s: Span) -> Self {
        miette::SourceSpan::new(
            miette::SourceOffset::from(s.start as usize),
            s.end as usize - s.start as usize,
        )
    }
}

impl Default for Span {
    /// Returns the zero span `Span { start: 0, end: 0 }`.
    ///
    /// Used as the serde-skip default for `Span` fields in HIR binding plan types.
    /// Serde calls `Default::default()` automatically for skipped fields on deserialization.
    fn default() -> Self {
        Self { start: 0, end: 0 }
    }
}

impl Span {
    /// Create a new span from inclusive `start` to exclusive `end` byte offsets.
    pub fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    /// Extend this span to also cover `other` (smallest covering span).
    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    /// Number of bytes covered by this span.
    pub fn len(&self) -> u32 {
        self.end - self.start
    }

    /// True when the span covers zero bytes.
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Generic AST/IR node with attached source location.
///
/// Using a newtype wrapper rather than an (T, Span) tuple to keep
/// the field names explicit and searchable in IDE tooling.
#[derive(Debug, Clone, PartialEq)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    /// Wrap `node` with a source `span`.
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }

    /// Transform the inner node, preserving the span.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Spanned<U> {
        Spanned { node: f(self.node), span: self.span }
    }
}

/// A single lexed token with its kind and position.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    /// Create a new token with the given kind and span.
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }
}

/// The base (radix) of an integer literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntBase {
    /// Decimal (base 10), e.g. `42`
    Decimal,
    /// Hexadecimal (base 16), e.g. `0xFF`
    Hex,
    /// Binary (base 2), e.g. `0b1010`
    Binary,
    /// Octal (base 8), e.g. `0o77`
    Octal,
}

/// Width suffix for integer literals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntSuffix {
    I8, I16, I32, I64,
    U8, U16, U32, U64,
}

/// Width suffix for float literals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatSuffix {
    F16,
    Bf16,
    F32,
    F64,
}

/// All token kinds emitted by the AXIOM-Compute lexer.
///
/// Note on M1-reserved keywords: `Let`, `Mut`, `If`, `Else`, `For`, `While`,
/// `Break`, `Continue`, `Struct` are included as dedicated variants even though
/// M0 does not generate them in valid programs. This lets the parser's `parse_stmt`
/// pre-check (§3.3) match on the token kind directly instead of comparing strings.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // ── Literals ────────────────────────────────────────
    IntLiteral { value: i128, suffix: Option<IntSuffix>, base: IntBase },
    FloatLiteral { value: f64, suffix: Option<FloatSuffix> },
    StringLiteral(String),
    BoolLiteral(bool),

    // ── Identifiers & Special Prefixes ──────────────────
    Ident(String),
    /// `@annotation_name` — the `@` is consumed; token carries the bare name.
    Annotation(String),
    /// `?opt_hole_name` — optimization hole for M2+ autotuner.
    OptHole(String),

    // ── Keywords ────────────────────────────────────────
    Fn,
    // M1-reserved keywords — present now so parse_stmt deny-list can match them.
    Let,
    Mut,
    Return,
    If,
    Else,
    For,
    While,
    In,
    Struct,
    Break,
    Continue,
    // Logical operators (keyword form, not operator tokens)
    And,
    Or,
    Not,
    // GPU-specific entry-point keyword (surface-level; also recognized as @kernel annotation)
    Void,
    // GPU built-in type keywords — needed for `-> void` return-type parsing
    Kernel,
    Buffer,
    ReadonlyBuffer,
    WriteonlyBuffer,
    Shared,
    Barrier,
    SubgroupUniform,

    // ── Primitive type keywords ──────────────────────────
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    F16, Bf16, F32, F64,
    Bool,

    // ── Operators ───────────────────────────────────────
    Plus,       // +
    Minus,      // -
    Star,       // *
    Slash,      // /
    Percent,    // %
    Eq,         // ==
    NotEq,      // !=
    Lt,         // <
    Gt,         // >
    LtEq,       // <=
    GtEq,       // >=
    Assign,     // =
    Arrow,      // ->
    FatArrow,   // =>
    Dot,        // .
    Colon,      // :
    ColonColon, // ::
    Comma,      // ,
    Semicolon,  // ;
    Pipe,       // |

    // ── Delimiters ──────────────────────────────────────
    LParen,   // (
    RParen,   // )
    LBracket, // [
    RBracket, // ]
    LBrace,   // {
    RBrace,   // }

    // ── Special ─────────────────────────────────────────
    Eof,
    /// Error token — lexer recovered and continued. The string is the error message.
    Error(String),
    /// `?` — M2.3 strategy-hole sigil (bare question mark, before `[` or at EOF).
    ///
    /// Emitted by the lexer when `?` is followed by `[` (declaration site) so that
    /// the parser can handle `?[...]` as a hole-candidate list. `?ident` is still
    /// lexed as `OptHole(String)` for the reference-site form.
    Question,
}

impl TokenKind {
    /// Map an identifier string to the corresponding keyword `TokenKind`, if any.
    ///
    /// Returns `None` for non-keywords so the caller falls back to `Ident`.
    pub fn keyword_from_str(s: &str) -> Option<TokenKind> {
        match s {
            "fn"       => Some(TokenKind::Fn),
            "let"      => Some(TokenKind::Let),
            "mut"      => Some(TokenKind::Mut),
            "return"   => Some(TokenKind::Return),
            "if"       => Some(TokenKind::If),
            "else"     => Some(TokenKind::Else),
            "for"      => Some(TokenKind::For),
            "while"    => Some(TokenKind::While),
            "in"       => Some(TokenKind::In),
            "struct"   => Some(TokenKind::Struct),
            "break"    => Some(TokenKind::Break),
            "continue" => Some(TokenKind::Continue),
            "and"      => Some(TokenKind::And),
            "or"       => Some(TokenKind::Or),
            "not"      => Some(TokenKind::Not),
            "void"             => Some(TokenKind::Void),
            "kernel"           => Some(TokenKind::Kernel),
            "buffer"           => Some(TokenKind::Buffer),
            "readonly_buffer"  => Some(TokenKind::ReadonlyBuffer),
            "writeonly_buffer" => Some(TokenKind::WriteonlyBuffer),
            "shared"           => Some(TokenKind::Shared),
            "barrier"          => Some(TokenKind::Barrier),
            "subgroup_uniform" => Some(TokenKind::SubgroupUniform),
            "true"             => Some(TokenKind::BoolLiteral(true)),
            "false"            => Some(TokenKind::BoolLiteral(false)),
            // Primitive types
            "i8"   => Some(TokenKind::I8),
            "i16"  => Some(TokenKind::I16),
            "i32"  => Some(TokenKind::I32),
            "i64"  => Some(TokenKind::I64),
            "u8"   => Some(TokenKind::U8),
            "u16"  => Some(TokenKind::U16),
            "u32"  => Some(TokenKind::U32),
            "u64"  => Some(TokenKind::U64),
            "f16"  => Some(TokenKind::F16),
            "bf16" => Some(TokenKind::Bf16),
            "f32"  => Some(TokenKind::F32),
            "f64"  => Some(TokenKind::F64),
            "bool" => Some(TokenKind::Bool),
            _ => None,
        }
    }

    /// Returns `true` for `TokenKind::Error(_)`.
    pub fn is_error(&self) -> bool {
        matches!(self, TokenKind::Error(_))
    }

    /// Returns the human-readable M1.3+-reserved statement description for this token,
    /// or `None` if the token is not in the reserved deny-list for post-M1.3 features.
    ///
    /// Used by `parse_stmt` (§3.3) to produce `ParseError::UnsupportedInM1_1 { detail }`.
    /// Keeping the mapping here lets the parser stay free of literal strings.
    ///
    /// M1.1: `Let` and `Mut` were delisted from reserved (now valid syntax).
    /// M1.3: `If`, `Else`, `For`, `While`, `Break`, `Continue` are delisted (now valid
    ///        control-flow syntax). Only `Struct` remains reserved.
    pub fn m1_reserved_detail(&self) -> Option<&'static str> {
        match self {
            // Let and Mut are NO LONGER reserved — M1.1 implements them.
            TokenKind::Let      => None,
            TokenKind::Mut      => None,
            // If/Else/For/While/Break/Continue are NO LONGER reserved — M1.3 implements them.
            TokenKind::If       => None,
            TokenKind::Else     => None,
            TokenKind::For      => None,
            TokenKind::While    => None,
            TokenKind::Break    => None,
            TokenKind::Continue => None,
            // Struct remains reserved until M2.
            TokenKind::Struct   => Some("struct declaration"),
            // All other tokens are not M1-reserved
            TokenKind::IntLiteral { .. }
            | TokenKind::FloatLiteral { .. }
            | TokenKind::StringLiteral(_)
            | TokenKind::BoolLiteral(_)
            | TokenKind::Ident(_)
            | TokenKind::Annotation(_)
            | TokenKind::OptHole(_)
            | TokenKind::Fn
            | TokenKind::Return
            | TokenKind::In
            | TokenKind::And
            | TokenKind::Or
            | TokenKind::Not
            | TokenKind::Void
            | TokenKind::Kernel
            | TokenKind::Buffer
            | TokenKind::ReadonlyBuffer
            | TokenKind::WriteonlyBuffer
            | TokenKind::Shared
            | TokenKind::Barrier
            | TokenKind::SubgroupUniform
            | TokenKind::I8
            | TokenKind::I16
            | TokenKind::I32
            | TokenKind::I64
            | TokenKind::U8
            | TokenKind::U16
            | TokenKind::U32
            | TokenKind::U64
            | TokenKind::F16
            | TokenKind::Bf16
            | TokenKind::F32
            | TokenKind::F64
            | TokenKind::Bool
            | TokenKind::Plus
            | TokenKind::Minus
            | TokenKind::Star
            | TokenKind::Slash
            | TokenKind::Percent
            | TokenKind::Eq
            | TokenKind::NotEq
            | TokenKind::Lt
            | TokenKind::Gt
            | TokenKind::LtEq
            | TokenKind::GtEq
            | TokenKind::Assign
            | TokenKind::Arrow
            | TokenKind::FatArrow
            | TokenKind::Dot
            | TokenKind::Colon
            | TokenKind::ColonColon
            | TokenKind::Comma
            | TokenKind::Semicolon
            | TokenKind::Pipe
            | TokenKind::LParen
            | TokenKind::RParen
            | TokenKind::LBracket
            | TokenKind::RBracket
            | TokenKind::LBrace
            | TokenKind::RBrace
            | TokenKind::Eof
            | TokenKind::Error(_)
            | TokenKind::Question => None,
        }
    }
}

/// Sorted list of reserved subgroup builtin names for binary_search lookup.
///
/// These are plain identifiers (not keywords) recognized by the HIR typechecker
/// in `check_call` and `check_builtin_call_stmt`. The lexer does NOT create
/// dedicated token kinds for these; they remain `TokenKind::Ident`.
///
/// Sorted lexicographically so `binary_search` works correctly.
/// `subgroup_ballot` is DEFERRED to M1.5 and NOT included here.
pub const RESERVED_SUBGROUP_BUILTIN_NAMES: &[&str] = &[
    "subgroup_all",
    "subgroup_any",
    "subgroup_broadcast_first",
    "subgroup_elect",
    "subgroup_invocation_id",
    "subgroup_reduce_add",
    "subgroup_reduce_max",
    "subgroup_reduce_min",
    "subgroup_size",
    "workgroup_barrier",
];

/// Returns `true` if `name` is a reserved subgroup builtin identifier.
///
/// Uses binary search on the sorted `RESERVED_SUBGROUP_BUILTIN_NAMES` slice.
pub fn is_reserved_subgroup_builtin(name: &str) -> bool {
    RESERVED_SUBGROUP_BUILTIN_NAMES.binary_search(&name).is_ok()
}

/// Sorted-for-binary-search list of the four cooperative-matrix builtin call names (M2.1).
///
/// These are plain identifiers (not keywords) recognized by the parser and HIR typechecker.
/// The lexer does NOT create dedicated token kinds for these; they remain `TokenKind::Ident`.
///
/// Sorted lexicographically so `binary_search` works correctly.
/// The HIR crate (`axc_hir::coopmat::RESERVED_COOPMAT_BUILTIN_NAMES`) must stay in sync.
pub const RESERVED_COOPMAT_BUILTIN_NAMES: &[&str] = &[
    "coopmat_load",
    "coopmat_mul_add",
    "coopmat_store",
    "coopmat_zero",
];

/// Returns `true` if `name` is a reserved cooperative-matrix builtin identifier.
///
/// Uses binary search on the sorted `RESERVED_COOPMAT_BUILTIN_NAMES` slice.
pub fn is_reserved_coopmat_builtin(name: &str) -> bool {
    RESERVED_COOPMAT_BUILTIN_NAMES.binary_search(&name).is_ok()
}

/// Lookup table mapping byte offsets to (line, column) positions.
///
/// Built on demand for diagnostic rendering; not part of the hot tokenize path.
pub struct LineIndex {
    /// Byte offset of the first character of each line (0-indexed).
    line_starts: Vec<u32>,
}

impl LineIndex {
    /// Build a `LineIndex` from the source text.
    ///
    /// # Examples
    /// ```
    /// use axc_lexer::LineIndex;
    /// let idx = LineIndex::new("hello\nworld\n");
    /// assert_eq!(idx.line_col(0), (0, 0));
    /// assert_eq!(idx.line_col(6), (1, 0));
    /// ```
    pub fn new(source: &str) -> Self {
        let mut line_starts: Vec<u32> = vec![0u32];
        for (i, b) in source.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push((i + 1) as u32);
            }
        }
        Self { line_starts }
    }

    /// Convert a byte offset to a 0-based `(line, column)` pair.
    pub fn line_col(&self, offset: u32) -> (u32, u32) {
        let line: usize = match self.line_starts.binary_search(&offset) {
            Ok(exact) => exact,
            Err(insertion) => insertion.saturating_sub(1),
        };
        let col: u32 = offset.saturating_sub(self.line_starts[line]);
        (line as u32, col)
    }

    /// Total number of lines in the source (always >= 1).
    pub fn line_count(&self) -> u32 {
        self.line_starts.len() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_new_and_len() {
        let s: Span = Span::new(3, 7);
        assert_eq!(s.start, 3);
        assert_eq!(s.end, 7);
        assert_eq!(s.len(), 4);
        assert!(!s.is_empty());
    }

    /// AT-508a: Span::default returns Span { start: 0, end: 0 }.
    ///
    /// This is required so that serde-skipped Span fields on HIR binding-plan
    /// types (BufferBindingSlot, ScalarPushConstantSlot) deserialize correctly
    /// via `serde(skip)` — serde calls Default::default() for skipped fields.
    #[test]
    fn at_508a_span_default_impl_returns_zero_zero() {
        let d: Span = Span::default();
        assert_eq!(d, Span::new(0, 0), "Span::default() must equal Span::new(0, 0)");
        assert_eq!(d.start, 0);
        assert_eq!(d.end, 0);
        assert!(d.is_empty());
    }

    #[test]
    fn span_empty() {
        let s: Span = Span::new(5, 5);
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn span_merge() {
        let a: Span = Span::new(2, 5);
        let b: Span = Span::new(4, 9);
        let m: Span = a.merge(b);
        assert_eq!(m.start, 2);
        assert_eq!(m.end, 9);
    }

    #[test]
    fn keyword_from_str_recognizes_m1_reserved() {
        assert_eq!(TokenKind::keyword_from_str("let"), Some(TokenKind::Let));
        assert_eq!(TokenKind::keyword_from_str("if"), Some(TokenKind::If));
        assert_eq!(TokenKind::keyword_from_str("for"), Some(TokenKind::For));
        assert_eq!(TokenKind::keyword_from_str("struct"), Some(TokenKind::Struct));
        assert_eq!(TokenKind::keyword_from_str("break"), Some(TokenKind::Break));
        assert_eq!(TokenKind::keyword_from_str("continue"), Some(TokenKind::Continue));
    }

    // ── M1.2: buffer keywords in keyword_from_str ────────────────────────────

    #[test]
    fn keyword_from_str_buffer_keywords() {
        // All buffer-related keywords must map to distinct token kinds.
        assert_eq!(TokenKind::keyword_from_str("buffer"), Some(TokenKind::Buffer));
        assert_eq!(TokenKind::keyword_from_str("readonly_buffer"), Some(TokenKind::ReadonlyBuffer));
        assert_eq!(TokenKind::keyword_from_str("writeonly_buffer"), Some(TokenKind::WriteonlyBuffer));
        assert_eq!(TokenKind::keyword_from_str("shared"), Some(TokenKind::Shared));
        assert_eq!(TokenKind::keyword_from_str("barrier"), Some(TokenKind::Barrier));
        assert_eq!(TokenKind::keyword_from_str("kernel"), Some(TokenKind::Kernel));
        assert_eq!(TokenKind::keyword_from_str("subgroup_uniform"), Some(TokenKind::SubgroupUniform));
    }

    #[test]
    fn buffer_keywords_are_distinct_tokens() {
        // All three buffer access modes produce distinct tokens.
        use crate::Lexer;
        let (tokens, errs) = Lexer::new("buffer readonly_buffer writeonly_buffer").tokenize();
        assert!(errs.is_empty(), "unexpected lex errors: {errs:?}");
        assert!(matches!(tokens[0].kind, TokenKind::Buffer));
        assert!(matches!(tokens[1].kind, TokenKind::ReadonlyBuffer));
        assert!(matches!(tokens[2].kind, TokenKind::WriteonlyBuffer));
    }

    #[test]
    fn buffer_bracket_elem_lexes_correctly() {
        // `buffer[f32]` should lex as Buffer, LBracket, F32, RBracket.
        use crate::Lexer;
        let (tokens, errs) = Lexer::new("buffer[f32]").tokenize();
        assert!(errs.is_empty(), "unexpected lex errors: {errs:?}");
        assert!(matches!(tokens[0].kind, TokenKind::Buffer));
        assert!(matches!(tokens[1].kind, TokenKind::LBracket));
        assert!(matches!(tokens[2].kind, TokenKind::F32));
        assert!(matches!(tokens[3].kind, TokenKind::RBracket));
    }

    #[test]
    fn readonly_buffer_bracket_elem_lexes_correctly() {
        use crate::Lexer;
        let (tokens, errs) = Lexer::new("readonly_buffer[u32]").tokenize();
        assert!(errs.is_empty(), "unexpected lex errors: {errs:?}");
        assert!(matches!(tokens[0].kind, TokenKind::ReadonlyBuffer));
        assert!(matches!(tokens[1].kind, TokenKind::LBracket));
        assert!(matches!(tokens[2].kind, TokenKind::U32));
        assert!(matches!(tokens[3].kind, TokenKind::RBracket));
    }

    #[test]
    fn keyword_from_str_non_keyword() {
        assert_eq!(TokenKind::keyword_from_str("myident"), None);
        assert_eq!(TokenKind::keyword_from_str(""), None);
    }

    #[test]
    fn m1_reserved_detail_control_flow_unreserved_after_m1_3() {
        // M1.1 delists Let and Mut — they are now valid syntax, not reserved.
        assert_eq!(TokenKind::Let.m1_reserved_detail(), None);
        assert_eq!(TokenKind::Mut.m1_reserved_detail(), None);
        // M1.3 delists If/Else/For/While/Break/Continue — now valid control-flow syntax.
        assert_eq!(TokenKind::If.m1_reserved_detail(), None);
        assert_eq!(TokenKind::For.m1_reserved_detail(), None);
        assert_eq!(TokenKind::While.m1_reserved_detail(), None);
        assert_eq!(TokenKind::Break.m1_reserved_detail(), None);
        assert_eq!(TokenKind::Continue.m1_reserved_detail(), None);
        assert_eq!(TokenKind::Else.m1_reserved_detail(), None);
        // Struct still remains reserved (deferred to M2+):
        assert_eq!(TokenKind::Struct.m1_reserved_detail(), Some("struct declaration"));
    }

    #[test]
    fn m1_3_in_token_is_not_reserved() {
        // `in` was never reserved; remains non-reserved in M1.3.
        assert_eq!(TokenKind::In.m1_reserved_detail(), None);
    }

    #[test]
    fn m1_reserved_detail_non_reserved() {
        // Non-M1-reserved tokens must return None (not accidentally match)
        assert_eq!(TokenKind::Return.m1_reserved_detail(), None);
        assert_eq!(TokenKind::Fn.m1_reserved_detail(), None);
        assert_eq!(TokenKind::Void.m1_reserved_detail(), None);
        assert_eq!(TokenKind::Eof.m1_reserved_detail(), None);
    }

    // ── M1.4: reserved subgroup builtin names ────────────────────────────────

    #[test]
    fn reserved_subgroup_builtin_names_contains_subgroup_invocation_id() {
        assert!(is_reserved_subgroup_builtin("subgroup_invocation_id"));
    }

    #[test]
    fn reserved_subgroup_builtin_names_contains_subgroup_size() {
        assert!(is_reserved_subgroup_builtin("subgroup_size"));
    }

    #[test]
    fn reserved_subgroup_builtin_names_contains_subgroup_elect() {
        assert!(is_reserved_subgroup_builtin("subgroup_elect"));
    }

    #[test]
    fn reserved_subgroup_builtin_names_contains_subgroup_reduce_add() {
        assert!(is_reserved_subgroup_builtin("subgroup_reduce_add"));
    }

    #[test]
    fn reserved_subgroup_builtin_names_contains_subgroup_reduce_min() {
        assert!(is_reserved_subgroup_builtin("subgroup_reduce_min"));
    }

    #[test]
    fn reserved_subgroup_builtin_names_contains_subgroup_reduce_max() {
        assert!(is_reserved_subgroup_builtin("subgroup_reduce_max"));
    }

    #[test]
    fn reserved_subgroup_builtin_names_contains_subgroup_broadcast_first() {
        assert!(is_reserved_subgroup_builtin("subgroup_broadcast_first"));
    }

    #[test]
    fn reserved_subgroup_builtin_names_contains_subgroup_all() {
        assert!(is_reserved_subgroup_builtin("subgroup_all"));
    }

    #[test]
    fn reserved_subgroup_builtin_names_contains_subgroup_any() {
        assert!(is_reserved_subgroup_builtin("subgroup_any"));
    }

    #[test]
    fn reserved_subgroup_builtin_names_contains_workgroup_barrier() {
        assert!(is_reserved_subgroup_builtin("workgroup_barrier"));
    }

    #[test]
    fn reserved_subgroup_builtin_names_rejects_unknown_names_via_binary_search() {
        // Deferred names must NOT be in the M1.4 list.
        assert!(!is_reserved_subgroup_builtin("subgroup_ballot"), "subgroup_ballot is deferred to M1.5");
        assert!(!is_reserved_subgroup_builtin("foo"), "foo is not a reserved name");
        assert!(!is_reserved_subgroup_builtin("gid"), "gid is not a subgroup reserved name");
        // Verify sort invariant: RESERVED_SUBGROUP_BUILTIN_NAMES must be sorted lexicographically.
        let mut sorted = RESERVED_SUBGROUP_BUILTIN_NAMES.to_vec();
        sorted.sort_unstable();
        assert_eq!(
            RESERVED_SUBGROUP_BUILTIN_NAMES, sorted.as_slice(),
            "RESERVED_SUBGROUP_BUILTIN_NAMES must be sorted lexicographically for binary_search to work"
        );
    }

    #[test]
    fn is_error_variants() {
        assert!(TokenKind::Error("bad".into()).is_error());
        assert!(!TokenKind::Fn.is_error());
        assert!(!TokenKind::Return.is_error());
    }

    #[test]
    fn line_index_basic() {
        let idx: LineIndex = LineIndex::new("hello\nworld\nfoo");
        assert_eq!(idx.line_col(0), (0, 0));
        assert_eq!(idx.line_col(5), (0, 5)); // the '\n' character
        assert_eq!(idx.line_col(6), (1, 0));
        assert_eq!(idx.line_col(12), (2, 0));
        assert_eq!(idx.line_count(), 3);
    }

    #[test]
    fn line_index_empty_source() {
        let idx: LineIndex = LineIndex::new("");
        assert_eq!(idx.line_col(0), (0, 0));
        assert_eq!(idx.line_count(), 1);
    }

    #[test]
    fn spanned_map() {
        let s: Spanned<i32> = Spanned::new(42_i32, Span::new(0, 2));
        let s2: Spanned<String> = s.map(|n| n.to_string());
        assert_eq!(s2.node, "42");
        assert_eq!(s2.span, Span::new(0, 2));
    }
}
