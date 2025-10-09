use llkv_test_utils::slt::expand_loops_with_mapping;

#[test]
fn expand_loops_basic() {
    let lines = vec![
        "create table t(a int);".to_string(),
        "loop i 1 3".to_string(),
        "insert into t values($i);".to_string(),
        "endloop".to_string(),
    ];
    let (expanded, mapping) = expand_loops_with_mapping(&lines, 0).expect("expand");
    assert_eq!(expanded.len(), 4);
    assert_eq!(mapping.len(), 4);
}

// Note: run_slt_file_with_factory is exercised indirectly by llkv-sql tests.
