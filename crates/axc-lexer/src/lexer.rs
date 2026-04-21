//! Hand-written byte-oriented lexer for AXIOM-Compute (.axc) source files.
//!
//! Ported from axiom-lexer/lexer.rs with GPU-keyword additions and removal of
//! CPU-only token kinds (CharLiteral, i128 suffix, DotDot, Ellipsis, Caret,
//! wrapping operators, etc.).
//!
//! Invariant: `tokenize()` always ends with a `TokenKind::Eof` token.
//! Even on errors, lexing continues to EOF (no short-circuit).
//!
//! BOM handling: the lexer does NOT strip or reject BOMs. If a BOM is present,
//! offset 0 is the first BOM byte; the lexer will emit `UnexpectedChar` for it.
//! In practice the driver rejects BOM-prefixed files before calling the lexer,
//! so this situation does not arise in the normal pipeline.

use crate::token::{Token, TokenKind, Span, IntBase, IntSuffix, FloatSuffix};

/// Diagnostic error produced by the lexer.
///
/// Every `LexError` corresponds to exactly one `TokenKind::Error` in the
/// token stream. Consumers that only need the token stream can ignore this
/// side-channel; consumers that need structured diagnostics use it.
#[derive(Debug, Clone, thiserror::Error, miette::Diagnostic)]
pub enum LexError {
    #[error("unexpected character {ch:?}")]
    UnexpectedChar {
        ch: char,
        #[label("here")]
        span: Span,
    },
    #[error("unterminated string literal")]
    UnterminatedString {
        #[label("string started here")]
        span: Span,
    },
    #[error("unterminated block comment")]
    UnterminatedBlockComment {
        #[label("comment started here")]
        span: Span,
    },
    #[error("invalid integer literal {lit:?}")]
    InvalidIntLiteral {
        lit: String,
        #[label("here")]
        span: Span,
    },
}

/// Lexer for AXIOM-Compute source.
pub struct Lexer<'src> {
    source: &'src [u8],
    pos: u32,
    /// Pending error from an unterminated block comment (emitted on next token call).
    pending_error: Option<(Token, LexError)>,
}

impl<'src> Lexer<'src> {
    /// Create a new lexer for the given UTF-8 source string.
    pub fn new(source: &'src str) -> Self {
        Self {
            source: source.as_bytes(),
            pos: 0,
            pending_error: None,
        }
    }

    /// Tokenize the entire source, returning `(tokens, errors)`.
    ///
    /// The token stream always ends with `TokenKind::Eof`.
    /// `errors` is a structured side-channel; the stream already embeds
    /// `TokenKind::Error` at the corresponding positions.
    pub fn tokenize(mut self) -> (Vec<Token>, Vec<LexError>) {
        let mut tokens: Vec<Token> = Vec::new();
        let mut errors: Vec<LexError> = Vec::new();
        loop {
            let (tok, maybe_err) = self.next_token();
            let is_eof: bool = tok.kind == TokenKind::Eof;
            if let Some(err) = maybe_err {
                errors.push(err);
            }
            tokens.push(tok);
            if is_eof {
                break;
            }
        }
        (tokens, errors)
    }

    /// Produce the next `Token` plus an optional structured `LexError`.
    /// Returns `(token, Some(err))` for error tokens, `(token, None)` otherwise.
    fn next_token(&mut self) -> (Token, Option<LexError>) {
        // Drain any pending error from an unterminated block comment
        if let Some((tok, err)) = self.pending_error.take() {
            return (tok, Some(err));
        }

        self.skip_whitespace_and_comments();

        // A pending error may have been set by skip_whitespace_and_comments
        if let Some((tok, err)) = self.pending_error.take() {
            return (tok, Some(err));
        }

        if self.is_at_end() {
            return (Token::new(TokenKind::Eof, Span::new(self.pos, self.pos)), None);
        }

        let start: u32 = self.pos;
        let ch: u8 = self.advance();

        match ch {
            // ── Single-char delimiters ───────────────────────────────────
            b'(' => (Token::new(TokenKind::LParen,   Span::new(start, self.pos)), None),
            b')' => (Token::new(TokenKind::RParen,   Span::new(start, self.pos)), None),
            b'[' => (Token::new(TokenKind::LBracket, Span::new(start, self.pos)), None),
            b']' => (Token::new(TokenKind::RBracket, Span::new(start, self.pos)), None),
            b'{' => (Token::new(TokenKind::LBrace,   Span::new(start, self.pos)), None),
            b'}' => (Token::new(TokenKind::RBrace,   Span::new(start, self.pos)), None),
            b',' => (Token::new(TokenKind::Comma,     Span::new(start, self.pos)), None),
            b';' => (Token::new(TokenKind::Semicolon, Span::new(start, self.pos)), None),
            b'|' => (Token::new(TokenKind::Pipe,      Span::new(start, self.pos)), None),
            b'%' => (Token::new(TokenKind::Percent,   Span::new(start, self.pos)), None),
            b'*' => (Token::new(TokenKind::Star,      Span::new(start, self.pos)), None),

            // ── Multi-char operators ─────────────────────────────────────
            b'+' => (Token::new(TokenKind::Plus,  Span::new(start, self.pos)), None),
            b'-' => {
                if self.match_byte(b'>') {
                    (Token::new(TokenKind::Arrow, Span::new(start, self.pos)), None)
                } else {
                    (Token::new(TokenKind::Minus, Span::new(start, self.pos)), None)
                }
            }
            b'/' => (Token::new(TokenKind::Slash, Span::new(start, self.pos)), None),
            b'=' => {
                if self.match_byte(b'=') {
                    (Token::new(TokenKind::Eq,      Span::new(start, self.pos)), None)
                } else if self.match_byte(b'>') {
                    (Token::new(TokenKind::FatArrow, Span::new(start, self.pos)), None)
                } else {
                    (Token::new(TokenKind::Assign,  Span::new(start, self.pos)), None)
                }
            }
            b'!' => {
                if self.match_byte(b'=') {
                    (Token::new(TokenKind::NotEq, Span::new(start, self.pos)), None)
                } else {
                    let span: Span = Span::new(start, self.pos);
                    let err: LexError = LexError::UnexpectedChar { ch: '!', span };
                    let tok: Token = Token::new(TokenKind::Error("unexpected character '!'".into()), span);
                    (tok, Some(err))
                }
            }
            b'<' => {
                if self.match_byte(b'=') {
                    (Token::new(TokenKind::LtEq, Span::new(start, self.pos)), None)
                } else {
                    (Token::new(TokenKind::Lt, Span::new(start, self.pos)), None)
                }
            }
            b'>' => {
                if self.match_byte(b'=') {
                    (Token::new(TokenKind::GtEq, Span::new(start, self.pos)), None)
                } else {
                    (Token::new(TokenKind::Gt, Span::new(start, self.pos)), None)
                }
            }
            b'.' => (Token::new(TokenKind::Dot, Span::new(start, self.pos)), None),
            b':' => {
                if self.match_byte(b':') {
                    (Token::new(TokenKind::ColonColon, Span::new(start, self.pos)), None)
                } else {
                    (Token::new(TokenKind::Colon, Span::new(start, self.pos)), None)
                }
            }

            // ── Annotation: @name ────────────────────────────────────────
            b'@' => self.lex_annotation(start),

            // ── Optimization hole: ?name ─────────────────────────────────
            b'?' => self.lex_opt_hole(start),

            // ── String literal ───────────────────────────────────────────
            b'"' => self.lex_string(start),

            // ── Number literal ───────────────────────────────────────────
            c if c.is_ascii_digit() => {
                // Back up so lex_number can re-read the first digit
                self.pos = start;
                self.lex_number()
            }

            // ── Identifier or keyword ────────────────────────────────────
            c if c.is_ascii_alphabetic() || c == b'_' => {
                self.pos = start;
                self.lex_ident_or_keyword()
            }

            // ── Unknown character — emit error, continue ─────────────────
            _ => {
                // Re-interpret as a UTF-8 character for a meaningful error message.
                // Because `source` is valid UTF-8 (Rust &str requirement), this is safe.
                let byte_at_start: u8 = self.source[start as usize];
                let display_ch: char = byte_at_start as char;
                let span: Span = Span::new(start, self.pos);
                let err: LexError = LexError::UnexpectedChar { ch: display_ch, span };
                let tok: Token = Token::new(
                    TokenKind::Error(format!("unexpected character {:?}", display_ch)),
                    span,
                );
                (tok, Some(err))
            }
        }
    }

    // ── Token-family helpers ─────────────────────────────────────────────────

    fn lex_annotation(&mut self, start: u32) -> (Token, Option<LexError>) {
        let name_start: u32 = self.pos;
        while !self.is_at_end() && (self.peek().is_ascii_alphanumeric() || self.peek() == b'_') {
            self.advance();
        }
        let name: String = String::from_utf8_lossy(
            &self.source[name_start as usize..self.pos as usize],
        ).into_owned();
        if name.is_empty() {
            let span: Span = Span::new(start, self.pos);
            let err: LexError = LexError::UnexpectedChar { ch: '@', span };
            (Token::new(TokenKind::Error("expected annotation name after '@'".into()), span), Some(err))
        } else {
            (Token::new(TokenKind::Annotation(name), Span::new(start, self.pos)), None)
        }
    }

    fn lex_opt_hole(&mut self, start: u32) -> (Token, Option<LexError>) {
        // M2.3: `?` followed by `[` → emit Question token so the parser can
        // construct an AnnotationArg::Hole from the bracket-delimited list.
        // `?` followed by an identifier → emit OptHole(name) as before.
        // `?` followed by anything else (including EOF) → emit Question and let
        // the parser produce ParseError::QuestionMarkExpectsBracketOrIdent.
        if !self.is_at_end() && self.peek() == b'[' {
            // Leave `[` for the parser to consume as LBracket.
            return (Token::new(TokenKind::Question, Span::new(start, self.pos)), None);
        }

        let name_start: u32 = self.pos;
        while !self.is_at_end() && (self.peek().is_ascii_alphanumeric() || self.peek() == b'_') {
            self.advance();
        }
        let name: String = String::from_utf8_lossy(
            &self.source[name_start as usize..self.pos as usize],
        ).into_owned();
        if name.is_empty() {
            // `?` at EOF or before a non-identifier, non-`[` token.
            // Emit Question so the parser can produce a structured error.
            let span: Span = Span::new(start, self.pos);
            (Token::new(TokenKind::Question, span), None)
        } else {
            (Token::new(TokenKind::OptHole(name), Span::new(start, self.pos)), None)
        }
    }

    fn lex_string(&mut self, start: u32) -> (Token, Option<LexError>) {
        let mut value: String = String::new();
        while !self.is_at_end() && self.peek() != b'"' {
            if self.peek() == b'\\' {
                self.advance(); // consume backslash
                if !self.is_at_end() {
                    match self.advance() {
                        b'n'  => value.push('\n'),
                        b't'  => value.push('\t'),
                        b'r'  => value.push('\r'),
                        b'\\' => value.push('\\'),
                        b'"'  => value.push('"'),
                        b'0'  => value.push('\0'),
                        b'x'  => {
                            match self.lex_hex_escape(2) {
                                Ok(byte_val) => value.push(byte_val as char),
                                Err(msg) => {
                                    // Skip to closing quote, then emit error
                                    while !self.is_at_end() && self.peek() != b'"' {
                                        self.advance();
                                    }
                                    if !self.is_at_end() {
                                        self.advance();
                                    }
                                    let span: Span = Span::new(start, self.pos);
                                    let err: LexError = LexError::InvalidIntLiteral { lit: msg.clone(), span };
                                    return (Token::new(TokenKind::Error(msg), span), Some(err));
                                }
                            }
                        }
                        _ => { /* ignore unknown escapes — not an error in M0 */ }
                    }
                }
            } else {
                value.push(self.advance() as char);
            }
        }
        if self.is_at_end() {
            let span: Span = Span::new(start, self.pos);
            let err: LexError = LexError::UnterminatedString { span };
            (Token::new(TokenKind::Error("unterminated string literal".into()), span), Some(err))
        } else {
            self.advance(); // consume closing `"`
            (Token::new(TokenKind::StringLiteral(value), Span::new(start, self.pos)), None)
        }
    }

    /// Parse exactly `count` hex digits and return the byte value, or an error string.
    fn lex_hex_escape(&mut self, count: usize) -> Result<u8, String> {
        let mut val: u8 = 0;
        for _ in 0..count {
            if self.is_at_end() {
                return Err("incomplete hex escape sequence".into());
            }
            let b: u8 = self.peek();
            let digit: u8 = match b {
                b'0'..=b'9' => b - b'0',
                b'a'..=b'f' => b - b'a' + 10,
                b'A'..=b'F' => b - b'A' + 10,
                _ => return Err(format!("invalid hex digit {:?} in escape sequence", b as char)),
            };
            self.advance();
            val = val * 16 + digit;
        }
        Ok(val)
    }

    fn lex_number(&mut self) -> (Token, Option<LexError>) {
        let start: u32 = self.pos;
        let first: u8 = self.advance();
        let mut is_float: bool = false;
        let mut base: IntBase = IntBase::Decimal;

        // Check for base prefix after leading `0`
        if first == b'0' && !self.is_at_end() {
            match self.peek() {
                b'x' | b'X' => {
                    base = IntBase::Hex;
                    self.advance();
                    return self.lex_int_with_base(start, base, is_hex_digit);
                }
                b'b' | b'B' => {
                    base = IntBase::Binary;
                    self.advance();
                    return self.lex_int_with_base(start, base, is_binary_digit);
                }
                b'o' | b'O' => {
                    base = IntBase::Octal;
                    self.advance();
                    return self.lex_int_with_base(start, base, is_octal_digit);
                }
                _ => {}
            }
        }

        // Consume remaining decimal digits
        while !self.is_at_end() && (self.peek().is_ascii_digit() || self.peek() == b'_') {
            self.advance();
        }

        // Check for decimal point (but not `..` range operator or `.method`)
        if !self.is_at_end()
            && self.peek() == b'.'
            && (self.pos + 1) < self.source.len() as u32
            && self.source[(self.pos + 1) as usize] != b'.'
            && self.source[(self.pos + 1) as usize].is_ascii_digit()
        {
            is_float = true;
            self.advance(); // consume `.`
            while !self.is_at_end() && (self.peek().is_ascii_digit() || self.peek() == b'_') {
                self.advance();
            }
        }

        // Check for scientific notation
        if !self.is_at_end() && (self.peek() == b'e' || self.peek() == b'E') {
            let saved: u32 = self.pos;
            self.advance();
            if !self.is_at_end() && (self.peek() == b'+' || self.peek() == b'-') {
                self.advance();
            }
            if !self.is_at_end() && self.peek().is_ascii_digit() {
                is_float = true;
                while !self.is_at_end() && (self.peek().is_ascii_digit() || self.peek() == b'_') {
                    self.advance();
                }
            } else {
                // Not a valid exponent — back up
                self.pos = saved;
            }
        }

        // Build numeric text with underscores stripped
        let raw: &[u8] = &self.source[start as usize..self.pos as usize];
        let text: String = raw.iter().filter(|&&b| b != b'_').map(|&b| b as char).collect();

        if is_float {
            let float_suffix: Option<FloatSuffix> = self.try_consume_float_suffix();
            let span: Span = Span::new(start, self.pos);
            return match text.parse::<f64>() {
                Ok(v) => (Token::new(TokenKind::FloatLiteral { value: v, suffix: float_suffix }, span), None),
                Err(e) => {
                    let msg: String = format!("invalid float: {e}");
                    let err: LexError = LexError::InvalidIntLiteral { lit: msg.clone(), span };
                    (Token::new(TokenKind::Error(msg), span), Some(err))
                }
            };
        }

        // Integer: try float suffix first (e.g. `42f32`)
        if let Some(fs) = self.try_consume_float_suffix() {
            let span: Span = Span::new(start, self.pos);
            return match text.parse::<f64>() {
                Ok(v) => (Token::new(TokenKind::FloatLiteral { value: v, suffix: Some(fs) }, span), None),
                Err(e) => {
                    let msg: String = format!("invalid float: {e}");
                    let err: LexError = LexError::InvalidIntLiteral { lit: msg.clone(), span };
                    (Token::new(TokenKind::Error(msg), span), Some(err))
                }
            };
        }

        let int_suffix: Option<IntSuffix> = self.try_consume_int_suffix();
        let span: Span = Span::new(start, self.pos);
        match text.parse::<i128>() {
            Ok(v) => (Token::new(TokenKind::IntLiteral { value: v, suffix: int_suffix, base }, span), None),
            Err(e) => {
                let msg: String = format!("invalid integer: {e}");
                let err: LexError = LexError::InvalidIntLiteral { lit: msg.clone(), span };
                (Token::new(TokenKind::Error(msg), span), Some(err))
            }
        }
    }

    fn lex_int_with_base(
        &mut self,
        start: u32,
        base: IntBase,
        is_valid_digit: fn(u8) -> bool,
    ) -> (Token, Option<LexError>) {
        let digit_start: u32 = self.pos;
        while !self.is_at_end() && (is_valid_digit(self.peek()) || self.peek() == b'_') {
            self.advance();
        }

        if self.pos == digit_start {
            let base_name: &str = match base {
                IntBase::Hex     => "hexadecimal",
                IntBase::Binary  => "binary",
                IntBase::Octal   => "octal",
                IntBase::Decimal => "decimal",
            };
            let span: Span = Span::new(start, self.pos);
            let msg: String = format!("expected {base_name} digits after prefix");
            let err: LexError = LexError::InvalidIntLiteral { lit: msg.clone(), span };
            return (Token::new(TokenKind::Error(msg), span), Some(err));
        }

        let digits: String = self.source[digit_start as usize..self.pos as usize]
            .iter()
            .filter(|&&b| b != b'_')
            .map(|&b| b as char)
            .collect();

        let radix: u32 = match base {
            IntBase::Hex     => 16,
            IntBase::Binary  => 2,
            IntBase::Octal   => 8,
            IntBase::Decimal => 10,
        };

        let int_suffix: Option<IntSuffix> = self.try_consume_int_suffix();
        let span: Span = Span::new(start, self.pos);
        match i128::from_str_radix(&digits, radix) {
            Ok(v) => (Token::new(TokenKind::IntLiteral { value: v, suffix: int_suffix, base }, span), None),
            Err(e) => {
                let msg: String = format!("invalid integer: {e}");
                let err: LexError = LexError::InvalidIntLiteral { lit: msg.clone(), span };
                (Token::new(TokenKind::Error(msg), span), Some(err))
            }
        }
    }

    fn lex_ident_or_keyword(&mut self) -> (Token, Option<LexError>) {
        let start: u32 = self.pos;
        while !self.is_at_end() && (self.peek().is_ascii_alphanumeric() || self.peek() == b'_') {
            self.advance();
        }
        let text: String = String::from_utf8_lossy(
            &self.source[start as usize..self.pos as usize],
        ).into_owned();
        let kind: TokenKind = TokenKind::keyword_from_str(&text)
            .unwrap_or(TokenKind::Ident(text));
        (Token::new(kind, Span::new(start, self.pos)), None)
    }

    // ── Suffix consumers ─────────────────────────────────────────────────────

    fn try_consume_float_suffix(&mut self) -> Option<FloatSuffix> {
        let remaining: &[u8] = &self.source[self.pos as usize..];

        // Check `bf16` first (longest match wins)
        if remaining.len() >= 4
            && &remaining[..4] == b"bf16"
            && !remaining.get(4).is_some_and(|&b| b.is_ascii_alphanumeric() || b == b'_')
        {
            self.pos += 4;
            return Some(FloatSuffix::Bf16);
        }

        let suffixes: &[(&[u8], FloatSuffix)] = &[
            (b"f16", FloatSuffix::F16),
            (b"f32", FloatSuffix::F32),
            (b"f64", FloatSuffix::F64),
        ];
        for &(pat, suf) in suffixes {
            if remaining.len() >= pat.len()
                && &remaining[..pat.len()] == pat
                && !remaining.get(pat.len()).is_some_and(|&b| b.is_ascii_alphanumeric() || b == b'_')
            {
                self.pos += pat.len() as u32;
                return Some(suf);
            }
        }
        None
    }

    fn try_consume_int_suffix(&mut self) -> Option<IntSuffix> {
        let remaining: &[u8] = &self.source[self.pos as usize..];

        // Longer patterns first to avoid prefix aliasing (e.g. `i16` before `i1`)
        let suffixes: &[(&[u8], IntSuffix)] = &[
            (b"i16", IntSuffix::I16),
            (b"i32", IntSuffix::I32),
            (b"i64", IntSuffix::I64),
            (b"i8",  IntSuffix::I8),
            (b"u16", IntSuffix::U16),
            (b"u32", IntSuffix::U32),
            (b"u64", IntSuffix::U64),
            (b"u8",  IntSuffix::U8),
        ];
        for &(pat, suf) in suffixes {
            if remaining.len() >= pat.len()
                && &remaining[..pat.len()] == pat
                && !remaining.get(pat.len()).is_some_and(|&b| b.is_ascii_alphanumeric() || b == b'_')
            {
                self.pos += pat.len() as u32;
                return Some(suf);
            }
        }
        None
    }

    // ── Whitespace / comment skipping ────────────────────────────────────────

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // Whitespace
            while !self.is_at_end() && self.peek().is_ascii_whitespace() {
                self.advance();
            }

            // Line comment `// …`
            if self.remaining() >= 2
                && self.peek() == b'/'
                && self.source[(self.pos + 1) as usize] == b'/'
            {
                while !self.is_at_end() && self.peek() != b'\n' {
                    self.advance();
                }
                continue;
            }

            // Block comment `/* … */` (nestable depth tracking)
            if self.remaining() >= 2
                && self.peek() == b'/'
                && self.source[(self.pos + 1) as usize] == b'*'
            {
                let comment_start: u32 = self.pos;
                self.advance(); // `/`
                self.advance(); // `*`
                let mut depth: u32 = 1;
                while !self.is_at_end() && depth > 0 {
                    if self.remaining() >= 2
                        && self.peek() == b'/'
                        && self.source[(self.pos + 1) as usize] == b'*'
                    {
                        self.advance();
                        self.advance();
                        depth += 1;
                    } else if self.remaining() >= 2
                        && self.peek() == b'*'
                        && self.source[(self.pos + 1) as usize] == b'/'
                    {
                        self.advance();
                        self.advance();
                        depth -= 1;
                    } else {
                        self.advance();
                    }
                }
                if depth > 0 {
                    // Unterminated block comment
                    let span: Span = Span::new(comment_start, self.pos);
                    let err: LexError = LexError::UnterminatedBlockComment { span };
                    let tok: Token = Token::new(TokenKind::Error("unterminated block comment".into()), span);
                    self.pending_error = Some((tok, err));
                    return;
                }
                continue;
            }

            break;
        }
    }

    // ── Low-level cursor helpers ─────────────────────────────────────────────

    fn is_at_end(&self) -> bool {
        self.pos >= self.source.len() as u32
    }

    fn remaining(&self) -> u32 {
        (self.source.len() as u32).saturating_sub(self.pos)
    }

    fn peek(&self) -> u8 {
        self.source[self.pos as usize]
    }

    fn advance(&mut self) -> u8 {
        let ch: u8 = self.source[self.pos as usize];
        self.pos += 1;
        ch
    }

    fn match_byte(&mut self, expected: u8) -> bool {
        if !self.is_at_end() && self.peek() == expected {
            self.advance();
            true
        } else {
            false
        }
    }
}

fn is_hex_digit(b: u8) -> bool {
    b.is_ascii_hexdigit()
}

fn is_binary_digit(b: u8) -> bool {
    b == b'0' || b == b'1'
}

fn is_octal_digit(b: u8) -> bool {
    (b'0'..=b'7').contains(&b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::LineIndex;

    fn lex_kinds(src: &str) -> Vec<TokenKind> {
        let (tokens, _) = Lexer::new(src).tokenize();
        tokens.into_iter().map(|t| t.kind).collect()
    }

    fn lex_with_errors(src: &str) -> (Vec<TokenKind>, Vec<LexError>) {
        let (tokens, errors) = Lexer::new(src).tokenize();
        let kinds: Vec<TokenKind> = tokens.into_iter().map(|t| t.kind).collect();
        (kinds, errors)
    }

    // ── Always-reaches-EOF invariant ────────────────────────────────────────

    #[test]
    fn always_ends_with_eof() {
        // Empty source
        let kinds: Vec<TokenKind> = lex_kinds("");
        assert_eq!(kinds.last(), Some(&TokenKind::Eof));

        // Error source
        let (kinds, _): (Vec<TokenKind>, Vec<LexError>) = lex_with_errors("@");
        assert_eq!(kinds.last(), Some(&TokenKind::Eof));
    }

    // ── Annotation tokens ────────────────────────────────────────────────────

    #[test]
    fn annotation_kernel_workgroup_intent() {
        let kinds: Vec<TokenKind> = lex_kinds("@kernel @workgroup @intent");
        assert_eq!(kinds, vec![
            TokenKind::Annotation("kernel".into()),
            TokenKind::Annotation("workgroup".into()),
            TokenKind::Annotation("intent".into()),
            TokenKind::Eof,
        ]);
    }

    #[test]
    fn annotation_gpu_specific() {
        let kinds: Vec<TokenKind> = lex_kinds("@complexity @precondition @subgroup_uniform");
        assert_eq!(kinds, vec![
            TokenKind::Annotation("complexity".into()),
            TokenKind::Annotation("precondition".into()),
            TokenKind::Annotation("subgroup_uniform".into()),
            TokenKind::Eof,
        ]);
    }

    // ── M1-reserved keywords ─────────────────────────────────────────────────

    #[test]
    fn m1_reserved_keywords_are_distinct_tokens() {
        let kinds: Vec<TokenKind> = lex_kinds("let mut if else for while break continue struct");
        assert_eq!(kinds, vec![
            TokenKind::Let, TokenKind::Mut,
            TokenKind::If, TokenKind::Else,
            TokenKind::For, TokenKind::While,
            TokenKind::Break, TokenKind::Continue,
            TokenKind::Struct,
            TokenKind::Eof,
        ]);
    }

    // ── Integer literals ─────────────────────────────────────────────────────

    #[test]
    fn integer_literals_decimal() {
        let kinds: Vec<TokenKind> = lex_kinds("42 0 100");
        assert_eq!(kinds, vec![
            TokenKind::IntLiteral { value: 42, suffix: None, base: IntBase::Decimal },
            TokenKind::IntLiteral { value: 0,  suffix: None, base: IntBase::Decimal },
            TokenKind::IntLiteral { value: 100, suffix: None, base: IntBase::Decimal },
            TokenKind::Eof,
        ]);
    }

    #[test]
    fn integer_literal_hex() {
        let kinds: Vec<TokenKind> = lex_kinds("0xFF 0x1A");
        assert_eq!(kinds, vec![
            TokenKind::IntLiteral { value: 255, suffix: None, base: IntBase::Hex },
            TokenKind::IntLiteral { value: 0x1A, suffix: None, base: IntBase::Hex },
            TokenKind::Eof,
        ]);
    }

    // ── String literals ──────────────────────────────────────────────────────

    #[test]
    fn string_literal_basic() {
        let kinds: Vec<TokenKind> = lex_kinds(r#""hello world""#);
        assert_eq!(kinds, vec![
            TokenKind::StringLiteral("hello world".into()),
            TokenKind::Eof,
        ]);
    }

    #[test]
    fn string_literal_with_escapes() {
        let kinds: Vec<TokenKind> = lex_kinds(r#""\n\t\\""#);
        assert_eq!(kinds, vec![
            TokenKind::StringLiteral("\n\t\\".into()),
            TokenKind::Eof,
        ]);
    }

    // ── Error recovery ───────────────────────────────────────────────────────

    #[test]
    fn error_recovery_unterminated_string() {
        let (kinds, errors): (Vec<TokenKind>, Vec<LexError>) = lex_with_errors(r#""unterminated"#);
        assert!(!errors.is_empty(), "expected at least one LexError");
        assert!(matches!(kinds.first(), Some(TokenKind::Error(_))));
        assert_eq!(kinds.last(), Some(&TokenKind::Eof));
    }

    #[test]
    fn error_recovery_unexpected_char_lexing_continues() {
        // After the bad char, `fn` and `return` must still be lexed
        let (kinds, errors): (Vec<TokenKind>, Vec<LexError>) = lex_with_errors("fn ! return");
        assert!(!errors.is_empty());
        assert!(kinds.iter().any(|k| k == &TokenKind::Fn));
        assert!(kinds.iter().any(|k| k == &TokenKind::Return));
        assert_eq!(kinds.last(), Some(&TokenKind::Eof));
    }

    #[test]
    fn error_recovery_unterminated_block_comment() {
        let (kinds, errors): (Vec<TokenKind>, Vec<LexError>) = lex_with_errors("fn /* unterminated");
        assert_eq!(errors.len(), 1);
        assert!(matches!(&kinds[0], TokenKind::Fn));
        assert!(matches!(&kinds[1], TokenKind::Error(_)));
        assert_eq!(kinds.last(), Some(&TokenKind::Eof));
    }

    // ── Span accuracy ────────────────────────────────────────────────────────

    #[test]
    fn span_accuracy() {
        let (tokens, _): (Vec<Token>, Vec<LexError>) = Lexer::new("fn empty").tokenize();
        assert_eq!(tokens[0].span, Span::new(0, 2));
        assert_eq!(tokens[0].kind, TokenKind::Fn);
        assert_eq!(tokens[1].span, Span::new(3, 8));
        assert!(matches!(tokens[1].kind, TokenKind::Ident(_)));
    }

    // ── LineIndex ────────────────────────────────────────────────────────────

    #[test]
    fn line_index_works() {
        let idx: LineIndex = LineIndex::new("a\nb\nc");
        assert_eq!(idx.line_col(0), (0, 0));
        assert_eq!(idx.line_col(2), (1, 0));
        assert_eq!(idx.line_col(4), (2, 0));
        assert_eq!(idx.line_count(), 3);
    }

    // ── Keywords ─────────────────────────────────────────────────────────────

    #[test]
    fn void_is_keyword() {
        let kinds: Vec<TokenKind> = lex_kinds("void");
        assert_eq!(kinds, vec![TokenKind::Void, TokenKind::Eof]);
    }

    #[test]
    fn bool_literals() {
        let kinds: Vec<TokenKind> = lex_kinds("true false");
        assert_eq!(kinds, vec![TokenKind::BoolLiteral(true), TokenKind::BoolLiteral(false), TokenKind::Eof]);
    }

    // ── Operators and delimiters ─────────────────────────────────────────────

    #[test]
    fn arrow_and_fat_arrow() {
        let kinds: Vec<TokenKind> = lex_kinds("-> =>");
        assert_eq!(kinds, vec![TokenKind::Arrow, TokenKind::FatArrow, TokenKind::Eof]);
    }

    #[test]
    fn colon_colon_colon() {
        let kinds: Vec<TokenKind> = lex_kinds(": ::");
        assert_eq!(kinds, vec![TokenKind::Colon, TokenKind::ColonColon, TokenKind::Eof]);
    }

    // ── m1_reserved_detail (spec §7.1 sub-assertion) ─────────────────────────

    #[test]
    fn reserved_detail_after_m1_1_delists_let_mut() {
        // M1.1 removes Let and Mut from the reserved deny-list (they are now valid syntax).
        assert_eq!(TokenKind::Let.m1_reserved_detail(),      None);
        assert_eq!(TokenKind::Mut.m1_reserved_detail(),      None);
        // M1.3 removes If, For, While, Break, Continue from the reserved deny-list (now valid syntax).
        assert_eq!(TokenKind::If.m1_reserved_detail(),       None);
        assert_eq!(TokenKind::For.m1_reserved_detail(),      None);
        assert_eq!(TokenKind::Return.m1_reserved_detail(),   None);
        assert_eq!(TokenKind::Fn.m1_reserved_detail(),       None);
    }

    // ── M2.1: F16 suffix and matrix ident (AT-604, AT-605) ───────────────────

    /// AT-604: The lexer tokenizes `0.5f16` as FloatLiteral with F16 suffix.
    #[test]
    fn float_literal_f16_suffix_parses() {
        let kinds = lex_kinds("0.5f16");
        assert_eq!(
            kinds,
            vec![
                TokenKind::FloatLiteral {
                    value: 0.5,
                    suffix: Some(FloatSuffix::F16),
                },
                TokenKind::Eof,
            ],
            "0.5f16 should lex as FloatLiteral {{ value: 0.5, suffix: Some(F16) }}"
        );
    }

    /// AT-605: `matrix` is a plain Ident at the lexer layer — not a keyword.
    ///
    /// `matrix[T, M, N, use]` is parsed entirely by the parser; the lexer does
    /// NOT reserve `matrix` as a token. This test verifies the lexer contract.
    #[test]
    fn ident_matrix_is_plain_ident() {
        let kinds = lex_kinds("matrix");
        assert_eq!(
            kinds,
            vec![TokenKind::Ident("matrix".to_string()), TokenKind::Eof],
            "matrix must be a plain Ident at the lexer layer, not a keyword"
        );
    }
}
