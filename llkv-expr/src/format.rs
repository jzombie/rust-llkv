//! Lightweight formatting helpers for expression enums.

use crate::{BinaryOp, CompareOp};

impl BinaryOp {
    /// Render the operator as a human-readable symbol/keyword.
    pub fn as_str(&self) -> &'static str {
        match self {
            BinaryOp::Add => "+",
            BinaryOp::Subtract => "-",
            BinaryOp::Multiply => "*",
            BinaryOp::Divide => "/",
            BinaryOp::Modulo => "%",
            BinaryOp::And => "AND",
            BinaryOp::Or => "OR",
            BinaryOp::BitwiseShiftLeft => "<<",
            BinaryOp::BitwiseShiftRight => ">>",
        }
    }
}

impl CompareOp {
    /// Render the operator as a human-readable symbol.
    pub fn as_str(&self) -> &'static str {
        match self {
            CompareOp::Eq => "=",
            CompareOp::NotEq => "!=",
            CompareOp::Lt => "<",
            CompareOp::LtEq => "<=",
            CompareOp::Gt => ">",
            CompareOp::GtEq => ">=",
        }
    }
}
