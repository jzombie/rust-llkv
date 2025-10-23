//! Generic iterative traversal utilities for AST-like structures.
//!
//! This module provides postorder traversal patterns that avoid stack overflow
//! on deeply nested expression trees. The approach uses explicit work and result
//! stacks instead of recursion, which is critical for handling complex SQL queries
//! (e.g., 55,000+ line files with deeply nested WHERE clauses).
//!
//! # Two Approaches
//!
//! ## 1. Trait-Based Traversal (for borrowed trees)
//!
//! Use [`Traversable`] and [`traverse_postorder`] when you have a tree of borrowed
//! nodes and want to compute a result without consuming the tree.
//!
//! ## 2. Manual Frame Pattern (for owned/consumed trees)
//!
//! When transforming owned trees (consuming input, producing output), use the
//! manual Frame pattern with these helpers:
//!
//! - Define a `Frame` enum with `Enter(InputNode)` and `Exit*` variants
//! - Use `Vec<Frame>` as work_stack, `Vec<OutputNode>` as result_stack
//! - Process frames in a loop, building results in postorder
//!
//! See existing implementations in:
//! - `llkv-sql/src/sql_engine.rs`: `translate_condition_with_context`
//! - `llkv-executor/src/translation/expression.rs`: `translate_predicate_with`
//!
//! # Example (Trait-Based)
//!
//! ```
//! use llkv_plan::traversal::{Traversable, traverse_postorder};
//! use llkv_result::Result as LlkvResult;
//!
//! enum MyExpr {
//!     Add(Box<MyExpr>, Box<MyExpr>),
//!     Value(i32),
//! }
//!
//! impl Traversable for MyExpr {
//!     type Output = i32;
//!
//!     fn visit_children(&self) -> LlkvResult<Vec<&Self>> {
//!         match self {
//!             MyExpr::Add(left, right) => Ok(vec![left.as_ref(), right.as_ref()]),
//!             MyExpr::Value(_) => Ok(vec![]),
//!         }
//!     }
//!
//!     fn construct(&self, children: Vec<i32>) -> LlkvResult<i32> {
//!         match self {
//!             MyExpr::Add(_, _) => Ok(children[0] + children[1]),
//!             MyExpr::Value(v) => Ok(*v),
//!         }
//!     }
//! }
//!
//! let expr = MyExpr::Add(
//!     Box::new(MyExpr::Value(2)),
//!     Box::new(MyExpr::Value(3)),
//! );
//! assert_eq!(traverse_postorder(&expr).unwrap(), 5);
//! ```
//!
//! # Example (TransformFrame Pattern)
//!
//! ```
//! use llkv_plan::TransformFrame;
//! 
//! #[derive(Clone)]
//! enum InputExpr {
//!     Binary { left: Box<InputExpr>, op: BinaryOp, right: Box<InputExpr> },
//!     Unary { op: UnaryOp, expr: Box<InputExpr> },
//!     Value(i32),
//! }
//! 
//! #[derive(Clone)]
//! enum BinaryOp { Add, Subtract }
//! 
//! #[derive(Clone)]
//! enum UnaryOp { Negate }
//! 
//! enum OutputExpr {
//!     Binary { left: Box<OutputExpr>, op: BinaryOp, right: Box<OutputExpr> },
//!     Unary { op: UnaryOp, expr: Box<OutputExpr> },
//!     Value(i32),
//! }
//! 
//! enum ExprExitContext {
//!     Binary(BinaryOp),
//!     Unary,
//! }
//! 
//! type ExprFrame<'a> = TransformFrame<'a, InputExpr, OutputExpr, ExprExitContext>;
//! 
//! fn transform_expr(root_expr: &InputExpr) -> OutputExpr {
//!     let mut work_stack = vec![ExprFrame::Enter(root_expr)];
//!     let mut result_stack = Vec::new();
//! 
//!     while let Some(frame) = work_stack.pop() {
//!         match frame {
//!             ExprFrame::Enter(node) => match node {
//!                 InputExpr::Binary { left, op, right } => {
//!                     work_stack.push(ExprFrame::Exit(ExprExitContext::Binary(op.clone())));
//!                     work_stack.push(ExprFrame::Enter(right));
//!                     work_stack.push(ExprFrame::Enter(left));
//!                 }
//!                 InputExpr::Unary { op: _, expr } => {
//!                     work_stack.push(ExprFrame::Exit(ExprExitContext::Unary));
//!                     work_stack.push(ExprFrame::Enter(expr));
//!                 }
//!                 InputExpr::Value(v) => {
//!                     work_stack.push(ExprFrame::Leaf(OutputExpr::Value(*v)));
//!                 }
//!             },
//!             ExprFrame::Exit(exit_context) => match exit_context {
//!                 ExprExitContext::Binary(op) => {
//!                     let right = result_stack.pop().unwrap();
//!                     let left = result_stack.pop().unwrap();
//!                     result_stack.push(OutputExpr::Binary {
//!                         left: Box::new(left),
//!                         op,
//!                         right: Box::new(right),
//!                     });
//!                 }
//!                 ExprExitContext::Unary => {
//!                     let expr = result_stack.pop().unwrap();
//!                     result_stack.push(OutputExpr::Unary {
//!                         op: UnaryOp::Negate,
//!                         expr: Box::new(expr),
//!                     });
//!                 }
//!             },
//!             ExprFrame::Leaf(output) => {
//!                 result_stack.push(output);
//!             }
//!         }
//!     }
//! 
//!     result_stack.pop().unwrap()
//! }
//! # transform_expr(&InputExpr::Value(42));
//! ```

use llkv_result::Result as LlkvResult;

/// Internal frame type used by [`traverse_postorder`] for borrowed tree traversal.
///
/// This is distinct from [`TransformFrame`] because it's used internally by the
/// trait-based traversal function and tracks both the node reference and expected
/// child count for the Exit variant.
///
/// Not exported - users should use [`TransformFrame`] for custom traversals.
enum TraversalFrame<'a, T> {
    Enter(&'a T),
    Exit(&'a T, usize), // node and expected child count
}

/// A frame in the traversal work stack for tree transformations.
///
/// Used in manual iterative traversals that consume input nodes and produce
/// transformed output nodes. This enum provides the common structure while
/// allowing implementations to define custom exit variants for operation-specific
/// context (e.g., which binary operator triggered the exit).
///
/// The pattern avoids stack overflow on deeply nested trees (50k+ nodes) by using
/// explicit work and result stacks instead of recursion.
///
/// # Type Parameters
/// * `Input` - The input node type being consumed during traversal
/// * `Output` - The result type being built on the result stack
/// * `ExitContext` - Custom enum carrying operation-specific exit state
///
/// # Traversal Pattern
///
/// ```rust
/// use llkv_plan::TransformFrame;
/// # use llkv_result::Result;
/// # type SqlExpr = String;
/// # type OutputExpr = i32;
/// # #[derive(Clone)]
/// # enum BinaryOp { Add, Subtract }
///
/// // Define operation-specific exit variants
/// enum ScalarExitContext {
///     BinaryOp { op: BinaryOp },
///     UnaryMinus,
/// }
///
/// type ScalarTransformFrame<'a> = TransformFrame<'a, SqlExpr, OutputExpr, ScalarExitContext>;
///
/// fn transform_scalar(expr: &SqlExpr) -> Result<OutputExpr> {
///     let mut work_stack: Vec<ScalarTransformFrame> = vec![ScalarTransformFrame::Enter(expr)];
///     let mut result_stack: Vec<OutputExpr> = Vec::new();
///
///     while let Some(frame) = work_stack.pop() {
///         match frame {
///             ScalarTransformFrame::Enter(node) => {
///                 // Decompose node, push exit frame, then push children
/// #               work_stack.push(ScalarTransformFrame::Leaf(42));
///             }
///             ScalarTransformFrame::Exit(ScalarExitContext::BinaryOp { op }) => {
///                 // Pop children from result_stack, combine, push result
/// #               let right = result_stack.pop().unwrap();
/// #               let left = result_stack.pop().unwrap();
/// #               result_stack.push(left + right);
///             }
///             ScalarTransformFrame::Exit(ScalarExitContext::UnaryMinus) => {
///                 // Pop single child, transform, push result
/// #               let val = result_stack.pop().unwrap();
/// #               result_stack.push(-val);
///             }
///             ScalarTransformFrame::Leaf(output) => {
///                 result_stack.push(output);
///             }
///         }
///     }
///
///     Ok(result_stack.pop().unwrap())
/// }
/// # transform_scalar(&"test".to_string()).unwrap();
/// ```
///
/// # See Also
///
/// Concrete implementations using this pattern:
/// - `llkv-sql/src/sql_engine.rs`: `translate_condition_with_context`, `translate_scalar`
/// - `llkv-executor/src/translation/expression.rs`: `translate_predicate_with`
pub enum TransformFrame<'a, Input, Output, ExitContext> {
    /// Begin visiting an input node (descend to children).
    Enter(&'a Input),
    
    /// Complete processing of a node (combine child results).
    ///
    /// The `ExitContext` carries operation-specific state needed to combine
    /// child results into the parent node's result. For example, in arithmetic
    /// expression evaluation, this might carry which operator to apply.
    Exit(ExitContext),
    
    /// A leaf node that has been fully transformed to output.
    ///
    /// Used when a node can be immediately converted without further traversal
    /// (e.g., literals, pre-resolved identifiers).
    Leaf(Output),
}

/// Trait for types that support iterative postorder traversal.
///
/// Implementors define:
/// 1. How to decompose a node into children (in `visit_children`)
/// 2. How to reconstruct a result from child results (in `construct`)
///
/// The traversal guarantees postorder: children are fully processed before
/// their parent's `construct` is called.
pub trait Traversable: Sized {
    /// The result type produced by traversing this tree.
    type Output;

    /// Visit each child of this node.
    ///
    /// Called when first encountering a node. Implementations should:
    /// - Return an iterator over child references
    /// - Return an empty iterator for leaf nodes
    ///
    /// # Returns
    /// An iterator over references to child nodes, or an error if the node is malformed.
    fn visit_children(&self) -> LlkvResult<Vec<&Self>>;

    /// Reconstruct this node's result from processed children.
    ///
    /// Called after all children have been processed. The `children` vec
    /// contains child results in the same order they were visited.
    ///
    /// # Arguments
    /// * `children` - Results from child nodes, in visit order
    ///
    /// # Returns
    /// The reconstructed result for this node.
    fn construct(&self, children: Vec<Self::Output>) -> LlkvResult<Self::Output>;
}

/// Performs iterative postorder traversal of a tree structure.
///
/// This function avoids stack overflow by using explicit work and result stacks
/// instead of recursion. It's designed for deeply nested ASTs common in complex
/// SQL queries.
///
/// # Type Parameters
/// * `T` - The node type (must implement [`Traversable`])
///
/// # Arguments
/// * `root` - The root node to traverse
///
/// # Returns
/// The result produced by traversing the entire tree.
///
/// # Errors
/// Returns errors from `visit_children` or `construct` implementations, or if the
/// traversal encounters an inconsistent state (e.g., result stack underflow).
///
/// # Performance
/// Stack usage is O(tree depth), memory usage is O(tree depth + node count).
/// For very deep trees (50k+ nesting), thread stack size should be >= 16MB.
pub fn traverse_postorder<T>(root: &T) -> LlkvResult<T::Output>
where
    T: Traversable,
{
    let mut work_stack: Vec<TraversalFrame<T>> = vec![TraversalFrame::Enter(root)];
    let mut result_stack: Vec<T::Output> = Vec::new();

    while let Some(frame) = work_stack.pop() {
        match frame {
            TraversalFrame::Enter(node) => {
                let children = node.visit_children()?;
                let child_count = children.len();

                // Schedule exit frame first
                work_stack.push(TraversalFrame::Exit(node, child_count));

                // Then push children in reverse order for correct postorder
                for child in children.into_iter().rev() {
                    work_stack.push(TraversalFrame::Enter(child));
                }
            }
            TraversalFrame::Exit(node, child_count) => {
                // Collect child results from result stack
                if result_stack.len() < child_count {
                    return Err(llkv_result::Error::Internal(
                        "traverse_postorder: result stack underflow".into(),
                    ));
                }

                let start = result_stack.len() - child_count;
                let child_results: Vec<T::Output> = result_stack.drain(start..).collect();

                // Process this node with its children's results
                let output = node.construct(child_results)?;
                result_stack.push(output);
            }
        }
    }

    // Should have exactly one result
    if result_stack.len() != 1 {
        return Err(llkv_result::Error::Internal(format!(
            "traverse_postorder: expected 1 result, got {}",
            result_stack.len()
        )));
    }

    result_stack.pop().ok_or_else(|| {
        llkv_result::Error::Internal("traverse_postorder: empty result stack".into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple expression tree for testing
    #[derive(Debug, Clone, PartialEq)]
    enum TestExpr {
        Add(Box<TestExpr>, Box<TestExpr>),
        Mul(Box<TestExpr>, Box<TestExpr>),
        Value(i32),
    }

    impl Traversable for TestExpr {
        type Output = i32;

        fn visit_children(&self) -> LlkvResult<Vec<&Self>> {
            match self {
                TestExpr::Add(left, right) | TestExpr::Mul(left, right) => {
                    Ok(vec![left.as_ref(), right.as_ref()])
                }
                TestExpr::Value(_) => Ok(vec![]),
            }
        }

        fn construct(&self, children: Vec<i32>) -> LlkvResult<i32> {
            match self {
                TestExpr::Add(_, _) => Ok(children[0] + children[1]),
                TestExpr::Mul(_, _) => Ok(children[0] * children[1]),
                TestExpr::Value(v) => Ok(*v),
            }
        }
    }

    #[test]
    fn test_simple_evaluation() {
        // 2 + 3 = 5
        let expr = TestExpr::Add(
            Box::new(TestExpr::Value(2)),
            Box::new(TestExpr::Value(3)),
        );
        assert_eq!(traverse_postorder(&expr).unwrap(), 5);
    }

    #[test]
    fn test_nested_evaluation() {
        // (2 + 3) * 4 = 20
        let expr = TestExpr::Mul(
            Box::new(TestExpr::Add(
                Box::new(TestExpr::Value(2)),
                Box::new(TestExpr::Value(3)),
            )),
            Box::new(TestExpr::Value(4)),
        );
        assert_eq!(traverse_postorder(&expr).unwrap(), 20);
    }

    #[test]
    fn test_deeply_nested() {
        // Build a deeply nested expression: 1 + 1 + 1 + ... (1000 times)
        let mut expr = TestExpr::Value(1);
        for _ in 0..1000 {
            expr = TestExpr::Add(Box::new(expr), Box::new(TestExpr::Value(1)));
        }
        assert_eq!(traverse_postorder(&expr).unwrap(), 1001);
    }
}
