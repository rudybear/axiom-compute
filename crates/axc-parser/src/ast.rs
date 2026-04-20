//! Abstract Syntax Tree node types for AXIOM-Compute.
//!
//! Every node is wrapped in `Spanned<T>` so that diagnostic messages can
//! point at the exact source location. M1.1 adds scalar types, let/let-mut
//! bindings, assignments, and a full Pratt-parsed expression tree.
//! M1.3 adds structured control flow: if/else, for-range, while, break, continue.

use axc_lexer::{Spanned, IntSuffix, FloatSuffix};

/// Top-level module: a sequence of items.
#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub items: Vec<Spanned<Item>>,
}

/// A top-level declaration.
///
/// M0 only supports kernels; other forms produce `ParseError::UnknownItem`.
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Kernel(KernelDecl),
}

/// A `@kernel fn name() -> void { … }` declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelDecl {
    pub annotations: Vec<Spanned<Annotation>>,
    pub name: Spanned<String>,
    pub params: Vec<Spanned<Param>>,
    pub return_type: Spanned<TypeRef>,
    pub body: Spanned<Block>,
}

/// A single function parameter (`name: type`).
///
/// M0 rejects any params with `ParseError::UnsupportedInM1_1 { detail: "kernel parameters" }`.
/// The field is kept for forward-compatibility so the AST shape is stable into M1.
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: Spanned<String>,
    pub ty: Spanned<TypeRef>,
}

/// Scalar element type for buffer parameters.
///
/// Mirrors the subset of `ScalarTy` that is valid as a buffer element in M1.2.
/// I8/I16/U8/U16/F16/Bf16 are excluded until the narrow-type milestone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalarTypeRef {
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
}

/// A type reference in source (return type of a kernel declaration, let binding type,
/// or kernel parameter type).
///
/// M0 only permitted `void`. M1.1 adds scalar types (§3.1 subset).
/// M1.2 adds buffer types with an element type.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeRef {
    Void,
    Bool,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
    /// `buffer[elem]` — readable and writable SSBO.
    Buffer(ScalarTypeRef),
    /// `readonly_buffer[elem]` — read-only SSBO.
    ReadonlyBuffer(ScalarTypeRef),
    /// `writeonly_buffer[elem]` — write-only SSBO.
    WriteonlyBuffer(ScalarTypeRef),
}

/// An annotation on a kernel: `@name` or `@name(arg, …)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Annotation {
    /// The bare annotation name (e.g. `"kernel"` for `@kernel`).
    pub name: Spanned<String>,
    pub args: Vec<Spanned<AnnotationArg>>,
}

/// A single argument inside an annotation argument list.
///
/// `Call` is used for compound forms like `O(1)` inside `@complexity(O(1))`.
#[derive(Debug, Clone, PartialEq)]
pub enum AnnotationArg {
    Int(i64),
    String(String),
    Bool(bool),
    Ident(String),
    Call { name: String, args: Vec<Spanned<AnnotationArg>> },
}

/// The body of a function: a braced list of statements.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub stmts: Vec<Spanned<Stmt>>,
}

/// The else-arm of an if statement.
///
/// `Block` is a plain else-block; `If` is an else-if (recursive nesting).
/// The `If` variant's inner `Stmt` is ALWAYS `Stmt::If`.
#[derive(Debug, Clone, PartialEq)]
pub enum ElseArm {
    /// `else { block }`
    Block(Spanned<Block>),
    /// `else if cond { block } ...` — stored as the nested `Stmt::If`, boxed to avoid
    /// infinite recursive sizing in the enum.
    If(Box<Spanned<Stmt>>),
}

/// A statement inside a kernel body.
///
/// M1.1 adds Let and Assign. Return remains from M0.
/// M1.2 adds IndexAssign for buffer writes.
/// M1.3 adds If, For, While, Break, Continue for structured control flow.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// `return;` or `return expr;` (only void return is used in M1.1).
    Return(Option<Spanned<Expr>>),
    /// `let [mut] name: type = expr;`
    Let {
        is_mut: bool,
        name: Spanned<String>,
        ty: Spanned<TypeRef>,
        init: Spanned<Expr>,
    },
    /// `name = expr;` (assignment to existing scalar binding).
    Assign {
        target: Spanned<String>,
        value: Spanned<Expr>,
    },
    /// `name[index] = expr;` (write to a buffer parameter).
    IndexAssign {
        target: Spanned<String>,
        index: Spanned<Expr>,
        value: Spanned<Expr>,
    },
    /// `if cond { then } [else { else } | else if cond { ... }]`
    If {
        cond: Spanned<Expr>,
        then_block: Spanned<Block>,
        else_arm: Option<Box<ElseArm>>,
    },
    /// `for var in range(start, end [, step]) { body }`
    ///
    /// `range` is a special form recognized by the parser when immediately
    /// following the `in` keyword; it is NOT a general function call.
    For {
        var: Spanned<String>,
        start: Spanned<Expr>,
        end: Spanned<Expr>,
        /// `None` means no explicit step was written → HIR uses step=1.
        step: Option<Spanned<Expr>>,
        body: Spanned<Block>,
    },
    /// `while cond { body }`
    While {
        cond: Spanned<Expr>,
        body: Spanned<Block>,
    },
    /// `break;`
    Break,
    /// `continue;`
    Continue,
    /// A bare `IDENT(args);` call to a reserved subgroup builtin at statement position.
    ///
    /// Only `workgroup_barrier()` returns void and is the canonical use here.
    /// All other reserved builtin names at statement position are rejected at HIR
    /// typecheck with `NonVoidSubgroupCallAsStatement`. The parser does NOT check
    /// return type; that is HIR's responsibility.
    ///
    /// The inner `Expr` is always `Expr::Call { name, args }`.
    BuiltinCallStmt { call: Spanned<Expr> },
}

/// Binary arithmetic / comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Neq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

/// Unary operator (prefix).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Arithmetic negation (`-x`).
    Neg,
    /// Logical NOT (`not x`); operand must be bool.
    LogicalNot,
}

/// Short-circuit logical operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShortCircuitOp {
    /// `a and b` — false if LHS is false (RHS not evaluated).
    And,
    /// `a or b` — true if LHS is true (RHS not evaluated).
    Or,
}

/// An expression.
///
/// M1.1 adds a full expression tree. The `IntLit` shape changes from M0's
/// `IntLit(i128)` to `IntLit { value, suffix }` to carry suffix information.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    BoolLit(bool),
    /// Integer literal with optional type suffix.
    IntLit {
        value: i128,
        suffix: Option<IntSuffix>,
    },
    /// Float literal with optional type suffix.
    FloatLit {
        value: f64,
        suffix: Option<FloatSuffix>,
    },
    /// Identifier reference (variable read).
    Ident(String),
    /// Unary prefix operator.
    Unary {
        op: UnaryOp,
        operand: Box<Spanned<Expr>>,
    },
    /// Binary infix operator (arithmetic, comparison).
    Binary {
        op: BinOp,
        lhs: Box<Spanned<Expr>>,
        rhs: Box<Spanned<Expr>>,
    },
    /// Short-circuit logical operator (and / or).
    ShortCircuit {
        op: ShortCircuitOp,
        lhs: Box<Spanned<Expr>>,
        rhs: Box<Spanned<Expr>>,
    },
    /// Built-in function call: `band(a, b)`, `shl(a, n)`, etc.
    Call {
        name: Spanned<String>,
        args: Vec<Spanned<Expr>>,
    },
    /// Parenthesized expression: `(expr)`.
    Paren(Box<Spanned<Expr>>),
    /// Array index expression: `expr[expr]`.
    ///
    /// Used for buffer reads: `buf[i]`.
    Index {
        base: Box<Spanned<Expr>>,
        index: Box<Spanned<Expr>>,
    },
}
