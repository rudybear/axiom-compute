//! Loop-context stack used during typecheck and codegen.
//!
//! A `HirLoopStack` is threaded through statement typechecking.
//! It is pushed on `for`/`while` entry and popped on exit.
//! `break` and `continue` call `is_in_loop()` to validate their position.
//! Assignment to a for-induction variable is detected via
//! `contains_induction_binding()`.
//!
//! Codegen pushes `LoopCodegenCtx` (with pre-allocated SPIR-V block IDs)
//! onto a separate `loop_stack: Vec<LoopCodegenCtx>`.  The typecheck-only
//! `HirLoopStack` does NOT carry SPIR-V IDs; that data lives in `body.rs`.

use crate::expr::BindingId;

/// Typecheck-side loop context: tracks induction variables so that
/// `AssignToForInductionVar` can be detected.
///
/// Only for-loops have an induction variable. While-loops push an entry
/// with `induction: None`.
#[derive(Debug, Clone)]
pub struct HirLoopFrame {
    /// The induction binding for this loop, if it is a for-range loop.
    /// `None` for while-loops.
    pub induction: Option<BindingId>,
}

/// Stack of loop frames threaded through typecheck.
///
/// Using a `Vec` (not `HashMap`) so that frame order is deterministic
/// and inner loops shadow outer ones correctly.
#[derive(Debug, Default, Clone)]
pub struct HirLoopStack {
    frames: Vec<HirLoopFrame>,
}

impl HirLoopStack {
    /// Create an empty loop stack.
    pub fn new() -> Self {
        Self { frames: Vec::new() }
    }

    /// Push a new loop frame on entry to a for/while loop.
    pub fn push(&mut self, induction: Option<BindingId>) {
        self.frames.push(HirLoopFrame { induction });
    }

    /// Pop the innermost loop frame on exit from a for/while loop.
    ///
    /// Panics if the stack is empty (compiler bug: mismatched push/pop).
    pub fn pop(&mut self) {
        self.frames.pop().expect("HirLoopStack::pop on empty stack — compiler bug");
    }

    /// True if there is at least one enclosing loop.
    pub fn is_in_loop(&self) -> bool {
        !self.frames.is_empty()
    }

    /// True if `binding_id` is the induction variable of any enclosing for-loop.
    ///
    /// This gates `AssignToForInductionVar` detection: if the user writes
    /// `i = expr` and `i` is an induction variable, this returns true.
    pub fn contains_induction_binding(&self, binding_id: BindingId) -> bool {
        self.frames.iter().any(|f| f.induction == Some(binding_id))
    }
}

/// Typecheck-side scope frame for name resolution.
///
/// Each block (if-then, if-else, loop body, etc.) opens a new scope frame.
/// Bindings declared inside a frame are invisible after the frame is popped.
/// BindingIds remain in `KernelBodyTyped.bindings` (codegen allocates storage
/// for every BindingId regardless of scope exit).
#[derive(Debug, Default, Clone)]
pub struct ScopeFrame {
    /// Mapping name → index in the typechecker's global `bindings` vec.
    ///
    /// Stored as a `Vec` of `(name, index)` pairs (not a `HashMap`) to
    /// preserve deterministic order and avoid hash-map ordering non-determinism.
    entries: Vec<(String, usize)>,
}

impl ScopeFrame {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Add a name → binding-index mapping to this frame.
    pub fn insert(&mut self, name: String, binding_idx: usize) {
        self.entries.push((name, binding_idx));
    }

    /// Look up a name in this frame only (does not traverse outer frames).
    pub fn get(&self, name: &str) -> Option<usize> {
        // Linear scan — frames are small (per-block let bindings only).
        self.entries.iter().rev().find_map(|(n, idx)| {
            if n == name {
                Some(*idx)
            } else {
                None
            }
        })
    }
}

/// Scoped name-resolution stack: a stack of `ScopeFrame`s.
///
/// Ident lookup traverses frames from innermost to outermost.
/// On scope exit the innermost frame is popped.
#[derive(Debug, Default, Clone)]
pub struct ScopeStack {
    frames: Vec<ScopeFrame>,
}

impl ScopeStack {
    pub fn new() -> Self {
        Self { frames: Vec::new() }
    }

    /// Push a new (initially empty) scope frame.
    pub fn push_frame(&mut self) {
        self.frames.push(ScopeFrame::new());
    }

    /// Pop the innermost scope frame (discarding its name bindings).
    ///
    /// Panics if called on an empty stack (compiler bug).
    pub fn pop_frame(&mut self) {
        self.frames.pop().expect("ScopeStack::pop_frame on empty stack — compiler bug");
    }

    /// Register a name → binding-index in the innermost frame.
    ///
    /// Panics if there are no frames (caller must push at least one).
    pub fn insert(&mut self, name: String, binding_idx: usize) {
        self.frames.last_mut()
            .expect("ScopeStack::insert with no frames — compiler bug")
            .insert(name, binding_idx);
    }

    /// Look up a name from innermost scope to outermost.
    pub fn get(&self, name: &str) -> Option<usize> {
        for frame in self.frames.iter().rev() {
            if let Some(idx) = frame.get(name) {
                return Some(idx);
            }
        }
        None
    }

    /// Look up a name in the CURRENT (innermost) frame only.
    ///
    /// Used by duplicate-binding detection in `register_binding`: shadowing is
    /// allowed across frames, but re-declaring in the SAME scope is not.
    pub fn get_in_current_frame(&self, name: &str) -> Option<usize> {
        self.frames.last().and_then(|f| f.get(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hir_loop_stack_is_in_loop_false_when_empty() {
        let stack = HirLoopStack::new();
        assert!(!stack.is_in_loop());
    }

    #[test]
    fn hir_loop_stack_is_in_loop_true_after_push() {
        let mut stack = HirLoopStack::new();
        stack.push(None);
        assert!(stack.is_in_loop());
        stack.pop();
        assert!(!stack.is_in_loop());
    }

    #[test]
    fn hir_loop_stack_contains_induction_binding_for_loop() {
        let mut stack = HirLoopStack::new();
        let bid = BindingId(42);
        stack.push(Some(bid));
        assert!(stack.contains_induction_binding(bid));
        assert!(!stack.contains_induction_binding(BindingId(99)));
        stack.pop();
        assert!(!stack.contains_induction_binding(bid));
    }

    #[test]
    fn hir_loop_stack_while_loop_no_induction() {
        let mut stack = HirLoopStack::new();
        stack.push(None); // while loop
        assert!(stack.is_in_loop());
        assert!(!stack.contains_induction_binding(BindingId(0)));
        stack.pop();
    }

    #[test]
    fn hir_loop_stack_nested_loops_inner_shadows() {
        let mut stack = HirLoopStack::new();
        let outer_id = BindingId(1);
        let inner_id = BindingId(2);
        stack.push(Some(outer_id));
        stack.push(Some(inner_id));
        assert!(stack.contains_induction_binding(outer_id)); // outer visible through stack
        assert!(stack.contains_induction_binding(inner_id));
        stack.pop();
        // After popping inner, outer still visible
        assert!(stack.contains_induction_binding(outer_id));
        assert!(!stack.contains_induction_binding(inner_id));
        stack.pop();
    }

    #[test]
    fn scope_stack_get_innermost_wins() {
        let mut s = ScopeStack::new();
        s.push_frame();
        s.insert("x".to_owned(), 0);
        s.push_frame();
        s.insert("x".to_owned(), 1); // shadows outer
        assert_eq!(s.get("x"), Some(1));
        s.pop_frame();
        // After pop, outer x is visible again
        assert_eq!(s.get("x"), Some(0));
        s.pop_frame();
        assert_eq!(s.get("x"), None);
    }

    #[test]
    fn scope_stack_unknown_name_returns_none() {
        let mut s = ScopeStack::new();
        s.push_frame();
        assert_eq!(s.get("z"), None);
        s.pop_frame();
    }

    #[test]
    fn scope_stack_get_in_current_frame_finds_local() {
        let mut s = ScopeStack::new();
        s.push_frame();
        s.insert("a".to_owned(), 7);
        assert_eq!(s.get_in_current_frame("a"), Some(7));
        s.pop_frame();
    }

    #[test]
    fn scope_stack_get_in_current_frame_misses_outer() {
        let mut s = ScopeStack::new();
        s.push_frame();
        s.insert("x".to_owned(), 0);
        s.push_frame();
        // x is in the outer frame, not the current frame
        assert_eq!(s.get_in_current_frame("x"), None);
        s.pop_frame();
        s.pop_frame();
    }

    #[test]
    fn scope_stack_shadowing_does_not_conflict_in_inner_frame() {
        // Inner frame can shadow outer without triggering same-frame conflict.
        let mut s = ScopeStack::new();
        s.push_frame();
        s.insert("i".to_owned(), 0); // outer induction var
        s.push_frame();
        assert_eq!(s.get_in_current_frame("i"), None); // not in inner frame yet
        s.insert("i".to_owned(), 1); // inner induction var (shadowing)
        assert_eq!(s.get_in_current_frame("i"), Some(1)); // finds inner
        assert_eq!(s.get("i"), Some(1)); // full search also finds inner first
        s.pop_frame();
        assert_eq!(s.get("i"), Some(0)); // outer now visible again
        s.pop_frame();
    }

    #[test]
    fn scope_stack_multiple_entries_in_frame() {
        let mut s = ScopeStack::new();
        s.push_frame();
        s.insert("a".to_owned(), 0);
        s.insert("b".to_owned(), 1);
        s.insert("c".to_owned(), 2);
        assert_eq!(s.get("a"), Some(0));
        assert_eq!(s.get("b"), Some(1));
        assert_eq!(s.get("c"), Some(2));
        assert_eq!(s.get("d"), None);
        s.pop_frame();
    }

    #[test]
    fn hir_loop_stack_empty_does_not_contain_any_induction() {
        let stack = HirLoopStack::new();
        assert!(!stack.contains_induction_binding(BindingId(0)));
        assert!(!stack.contains_induction_binding(BindingId(999)));
    }

    #[test]
    fn hir_loop_stack_two_while_loops_nested() {
        let mut stack = HirLoopStack::new();
        stack.push(None); // outer while
        stack.push(None); // inner while
        assert!(stack.is_in_loop());
        assert!(!stack.contains_induction_binding(BindingId(5)));
        stack.pop();
        assert!(stack.is_in_loop());
        stack.pop();
        assert!(!stack.is_in_loop());
    }

    #[test]
    fn hir_loop_stack_for_inside_while_tracks_induction() {
        let mut stack = HirLoopStack::new();
        let for_id = BindingId(3);
        stack.push(None);           // outer while
        stack.push(Some(for_id));   // inner for
        assert!(stack.contains_induction_binding(for_id));
        stack.pop();
        // After popping for, induction no longer tracked
        assert!(!stack.contains_induction_binding(for_id));
        stack.pop();
    }
}
