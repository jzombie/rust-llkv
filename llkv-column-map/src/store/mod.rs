//! The main ColumnStore API.

use crate::error::{Error, Result};
use crate::serialization::{deserialize_array, serialize_array};
use crate::storage::pager::{BatchGet, BatchPut, GetResult, Pager};
use crate::store::catalog::ColumnCatalog;
use crate::store::descriptor::{
    ChunkMetadata, ColumnDescriptor, DescriptorIterator, DescriptorPageHeader,
};
use crate::types::{CATALOG_ROOT_PKEY, LogicalFieldId, PhysicalKey};
use arrow::array::{Array, ArrayRef, BooleanArray, UInt32Array, UInt64Array, make_array};
use arrow::compute::{self, SortColumn, lexsort_to_indices};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;

use rustc_hash::{FxHashMap, FxHashSet};
use simd_r_drive_entry_handle::EntryHandle;
use std::sync::{Arc, RwLock};

mod config;
use config::*;

mod constants;
use constants::*;

pub mod core;
pub use core::*;

pub mod catalog;
pub mod debug;
pub mod descriptor;

mod ingest;
use ingest::*;

pub mod layout;
pub use layout::*;

pub mod rowid;
pub use rowid::*;

pub mod scan;
pub use scan::*;

mod slicing;
use slicing::*;

mod dtype_cache;
use dtype_cache::DTypeCache;

pub mod indexing;
pub use indexing::*;
