use llkv_storage::types::PhysicalKey;
use llkv_types::ids::LogicalFieldId;

/// Statistics for a single descriptor page.
#[derive(Debug, Clone)]
pub struct DescriptorPageStats {
    pub page_pk: PhysicalKey,
    pub entry_count: u32,
    pub page_size_bytes: usize,
}

/// Aggregated layout statistics for a single column.
#[derive(Debug, Clone)]
pub struct ColumnLayoutStats {
    pub field_id: LogicalFieldId,
    pub total_rows: u64,
    pub total_chunks: u64,
    pub pages: Vec<DescriptorPageStats>,
}
