/// Test that .slturl pointer files can be discovered and executed.
/// This test verifies the remote URL fetching functionality works end-to-end.
use llkv_slt_tester::LlkvSltRunner;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
#[ignore] // Run with --ignored to test network-dependent functionality
fn test_slturl_pointer_file() {
    let runner = LlkvSltRunner::in_memory();
    
    // Test a single .slturl pointer file
    let result = runner.run_file("tests/slt/sqlite/index/in/10/slt_good_0.slturl");
    
    // This test is expected to pass if the URL is reachable
    // In CI or without network, use --ignored to skip
    assert!(result.is_ok(), "Failed to run .slturl pointer: {:?}", result.err());
}

#[test]
fn test_slturl_file_not_found() {
    let runner = LlkvSltRunner::in_memory();
    
    // Test with a non-existent .slturl file
    let result = runner.run_file("tests/slt/nonexistent_file.slturl");
    
    // Should fail because the file doesn't exist
    assert!(result.is_err(), "Expected error for non-existent .slturl file");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("failed to read .slturl file") || err_msg.contains("No such file"),
        "Expected file read error, got: {}", err_msg
    );
}

#[test]
#[ignore] // Run with --ignored to test network-dependent functionality
fn test_slturl_invalid_url() {
    let runner = LlkvSltRunner::in_memory();
    
    // Create a temporary .slturl file with an invalid/unreachable URL
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    temp_file.write_all(b"https://invalid.example.com/nonexistent/test.slt").expect("Failed to write");
    
    // Rename to have .slturl extension
    let temp_path = temp_file.path().with_extension("slturl");
    std::fs::copy(temp_file.path(), &temp_path).expect("Failed to copy temp file");
    
    let result = runner.run_file(&temp_path);
    
    // Clean up
    let _ = std::fs::remove_file(&temp_path);
    
    // Should fail because the URL is not reachable
    assert!(result.is_err(), "Expected error for invalid URL");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("failed to fetch SLT URL") || err_msg.contains("dns error") || err_msg.contains("error sending request"),
        "Expected network/fetch error, got: {}", err_msg
    );
}

#[test]
#[ignore] // Run with --ignored to test network-dependent functionality  
fn test_slturl_valid_url_but_404() {
    let runner = LlkvSltRunner::in_memory();
    
    // Create a temporary .slturl file with a valid URL that returns 404
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    temp_file.write_all(b"https://raw.githubusercontent.com/jzombie/sqlite-sqllogictest-corpus/refs/heads/main/test/nonexistent.test").expect("Failed to write");
    
    // Rename to have .slturl extension
    let temp_path = temp_file.path().with_extension("slturl");
    std::fs::copy(temp_file.path(), &temp_path).expect("Failed to copy temp file");
    
    let result = runner.run_file(&temp_path);
    
    // Clean up
    let _ = std::fs::remove_file(&temp_path);
    
    // Should fail because the content doesn't exist (404 or similar)
    // Note: GitHub raw returns 404 as a successful response with error content
    // so this may succeed but fail during SLT parsing
    assert!(result.is_err(), "Expected error for 404 URL");
}
