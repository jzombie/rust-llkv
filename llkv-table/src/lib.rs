#![forbid(unsafe_code)]

pub mod expr {
    pub use llkv_expr::expr::*;
}

pub mod btree {

    pub mod errors {
        pub use llkv_btree::errors::Error;
    }

    pub mod pager {
        pub use llkv_btree::define_mem_pager;
        pub use llkv_btree::pager::Pager;
        pub use llkv_btree::pager::SharedPager;
    }
}

pub mod codecs;
pub mod constants;
pub mod table;
pub mod types;

pub use table::{Table, TableCfg};
pub use types::{FieldId, RowId, RowIdCmp};
