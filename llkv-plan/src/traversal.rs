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
//! # Example (Manual Frame Pattern)
//!
//! ```ignore
//! enum Frame {
//!     Enter(InputExpr),
//!     ExitBinary(BinaryOp), // remember op, pop 2 children
//!     ExitUnary,            // pop 1 child
//! }
//!
//! let mut work_stack = vec![Frame::Enter(root_expr)];
//! let mut result_stack = Vec::new();
//!
//! while let Some(frame) = work_stack.pop() {
//!     match frame {
//!         Frame::Enter(node) => {
//!             // Decompose node, push Exit frame, then push children
//!         }
//!         Frame::ExitBinary(op) => {
//!             // Pop 2 results, combine them, push back
//!             let right = result_stack.pop().unwrap();
//!             let left = result_stack.pop().unwrap();
//!             result_stack.push(combine(left, op, right));
//!         }
//!         Frame::ExitUnary => {
//!             // Pop 1 result, transform it, push back
//!         }
//!     }
//! }
//!
//! result_stack.pop() // final result
//! ```

use llkv_result::Result as LlkvResult;

/// A frame in the traversal work stack.
///
/// Frames represent visit states during postorder traversal:
/// - `Enter`: Begin visiting a node (descend to children)
/// - `Exit`: Complete a node visit (combine child results)
enum Frame<'a, T> {
    Enter(&'a T),
    Exit(&'a T, usize), // node and expected child count
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
    let mut work_stack: Vec<Frame<T>> = vec![Frame::Enter(root)];
    let mut result_stack: Vec<T::Output> = Vec::new();

    while let Some(frame) = work_stack.pop() {
        match frame {
            Frame::Enter(node) => {
                let children = node.visit_children()?;
                let child_count = children.len();

                // Schedule exit frame first
                work_stack.push(Frame::Exit(node, child_count));

                // Then push children in reverse order for correct postorder
                for child in children.into_iter().rev() {
                    work_stack.push(Frame::Enter(child));
                }
            }
            Frame::Exit(node, child_count) => {
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
