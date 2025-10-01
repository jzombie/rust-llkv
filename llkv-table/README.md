# LLKV Table

**Work in Progress**

Columnar table using the [LLKV](https://github.com/jzombie/rust-llkv) toolkit.
This crate is designed to work directly with Arrow `RecordBatch` and does not provide any additional abstraction over the batch data model beyond how batches are queried and streamed. Data is fed into tables and retrieved from tables as batches of `RecordBatch`.

## License

Licensed under the [Apache-2.0 License](../LICENSE).
