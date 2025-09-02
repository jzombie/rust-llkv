// TODO: No magic types or numbers! Everything should be aliased!

pub mod errors;

pub mod bplus_tree;
pub use bplus_tree::*;

pub mod codecs;
pub mod traits;

pub mod iter;
pub use iter::*; // re-export iterator + scan traits

pub mod pager;

pub mod node_cache;
pub mod views;

pub mod prelude {
    pub use crate::traits::BTree as _;
}
