use regex::Regex;
use std::sync::OnceLock;

/// Strip TPC-H `CONNECT TO database;` statements from the input SQL script.
///
/// The upstream TPC-H referential integrity scripts use a multi-database syntax
/// that LLKV does not implement. Treat these directives as no-ops so the
/// remainder of the script can be parsed and executed without modification.
pub fn strip_tpch_connect_statements(sql: &str) -> String {
    static CONNECT_REGEX: OnceLock<Regex> = OnceLock::new();
    let re = CONNECT_REGEX.get_or_init(|| {
        Regex::new(r#"(?im)^\s*CONNECT\s+TO\s+(?:[A-Za-z0-9_]+|'[^']+'|"[^"]+")\s*;\s*"#)
            .expect("valid CONNECT TO regex")
    });
    re.replace_all(sql, "").to_string()
}
