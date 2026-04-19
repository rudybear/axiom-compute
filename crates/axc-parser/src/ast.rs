//! Abstract Syntax Tree node types for AXIOM-Compute.
//!
//! Every node is wrapped in `Spanned<T>` so that diagnostic messages can
//! point at the exact source location. In M0 only `@kernel` functions with
//! annotations and a `return;` body are representable.

use axc_lexer::Spanned;

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
/// M0 rejects any params with `ParseError::UnsupportedInM0 { detail: "kernel parameters" }`.
/// The field is kept for forward-compatibility so the AST shape is stable into M1.
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: Spanned<String>,
    pub ty: Spanned<TypeRef>,
}

/// A type reference in source (return type of a kernel declaration).
///
/// M0 only permits `void`. Other types parse successfully but HIR rejects them
/// with `HirError::BadKernelReturnType`.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeRef {
    Void,
    Bool,
    I32,
    U32,
    F32,
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

/// A statement inside a kernel body.
///
/// M0 only allows `return;` or `return <lit>;`. All other statement forms are
/// caught by the §3.3 M1-reserved keyword pre-check in `parse_stmt`.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Return(Option<Spanned<Expr>>),
}

/// A simple expression (only literals in M0).
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    BoolLit(bool),
    IntLit(i128),
}
