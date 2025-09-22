use line_ending::LineEnding;
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;

/// Canonicalize a DOT graph by sorting body lines while preserving header/footer.
pub fn canonicalize_dot(dot: &str) -> String {
    let lines: Vec<&str> = dot.lines().collect();
    if lines.is_empty() {
        return String::new();
    }
    // Keep first line (digraph ...) and last line (}) in place if present.
    let (head, tail, body_start, body_end) =
        if lines.first().unwrap().trim_start().starts_with("digraph ") {
            let last = lines.len().saturating_sub(1);
            let has_tail = last > 0 && lines[last].trim() == "}";
            (
                Some(lines[0]),
                if has_tail { Some(lines[last]) } else { None },
                1,
                if has_tail { last } else { lines.len() },
            )
        } else {
            (None, None, 0, lines.len())
        };
    let mut body: Vec<&str> = lines[body_start..body_end].to_vec();
    // Strip empty lines and sort for stability.
    body.retain(|l| !l.trim().is_empty());
    body.sort_unstable();
    let mut out = String::new();
    if let Some(h) = head {
        out.push_str(h);
        out.push_str(LineEnding::from_current_platform().as_str());
    }
    for l in body {
        out.push_str(l);
        out.push_str(LineEnding::from_current_platform().as_str());
    }
    if let Some(t) = tail {
        out.push_str(t);
        out.push_str(LineEnding::from_current_platform().as_str());
    }
    out
}

/// Assert a string matches or updates a golden file under `tests/snapshots/`.
/// - If UPDATE_SNAPSHOTS=1, overwrites the golden and passes.
/// - If golden missing, creates it and passes.
/// - Else, compares and panics on diff.
pub fn assert_matches_golden(content: &str, rel_path: &str) {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = base.join(rel_path);
    let update = std::env::var("UPDATE_SNAPSHOTS").ok().as_deref() == Some("1");
    if update || !path.exists() {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).unwrap();
        }
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        return;
    }
    let mut existing = String::new();
    fs::File::open(&path)
        .unwrap()
        .read_to_string(&mut existing)
        .unwrap();
    if existing != content {
        panic!("snapshot mismatch: {}", path.display());
    }
}
