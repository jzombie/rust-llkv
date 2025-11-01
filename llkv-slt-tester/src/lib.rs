//! SLT runner utilities for LLKV.
//!
//! This crate provides a high-level API for executing sqllogictest (SLT) files against LLKV.
//!
//! # Primary API
//!
//! [`LlkvSltRunner`] is the main entry point for running SLT tests. It provides methods for:
//! - Running individual `.slt` files or `.slturl` pointer files
//! - Running all SLT files in a directory
//! - Running SLT scripts from strings, readers, or URLs
//!
//! # Test Harness API
//!
//! For integration with `libtest-mimic` test harnesses:
//! - [`run_slt_harness`] and [`run_slt_harness_with_args`] for test discovery and execution
//! - [`make_in_memory_factory_factory`] for creating in-memory test backends

mod parser;
mod runner;
mod slt_test_engine;

// Primary public API - the main entry point for users
pub use runner::{LlkvSltRunner, RuntimeKind};

// Test harness API - for libtest integration
pub use runner::{run_slt_harness, run_slt_harness_with_args};
pub use slt_test_engine::make_in_memory_factory_factory;

// Parser utilities - only exported for testing purposes
pub use parser::expand_loops_with_mapping;
