// sqllogictest Runner isn't used by this in-place interpreter.

// Reuse the existing EngineHarness implementation in slt.rs by declaring a module.
mod slt;

#[tokio::test]
async fn run_all_slt_files() {
    let base_dir = std::path::PathBuf::from("tests/slt");

    // Collect .slt files recursively, sort them for deterministic order.
    let mut slt_files: Vec<std::path::PathBuf> = Vec::new();
    fn collect(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let path = e.path();
                if path.is_dir() {
                    collect(&path, out);
                } else if path.extension().and_then(|s| s.to_str()) == Some("slt") {
                    out.push(path);
                }
            }
        }
    }

    collect(&base_dir, &mut slt_files);
    slt_files.sort();

    for path in slt_files {
        tracing::info!("Running slt: {}", path.display());

        let content = std::fs::read_to_string(&path).expect("read slt file");

        // Expand loop directives (supports nested loops) and substitute $vars.
        fn expand_loops(lines: &[String]) -> Vec<String> {
            let mut out: Vec<String> = Vec::new();
            let mut i = 0usize;
            while i < lines.len() {
                let line = lines[i].trim_start().to_string();
                if line.starts_with("loop ") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() < 4 {
                        panic!("malformed loop directive: {}", line);
                    }
                    let var = parts[1];
                    let start: i64 = parts[2].parse().expect("invalid loop start");
                    let count: i64 = parts[3].parse().expect("invalid loop count");

                    // find matching endloop
                    let mut j = i + 1;
                    while j < lines.len() && lines[j].trim_start() != "endloop" {
                        j += 1;
                    }
                    if j >= lines.len() {
                        panic!("unterminated loop in slt");
                    }

                    let inner = &lines[i + 1..j];
                    for k in 0..count {
                        let val = (start + k).to_string();
                        // substitute and recursively expand in case of nested loops
                        let substituted: Vec<String> = inner
                            .iter()
                            .map(|l| l.replace(&format!("${}", var), &val))
                            .collect();
                        let rec = expand_loops(&substituted);
                        out.extend(rec);
                    }

                    i = j + 1;
                } else {
                    out.push(lines[i].clone());
                    i += 1;
                }
            }
            out
        }

        let raw_lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
        let expanded_lines = expand_loops(&raw_lines);

        // Split into records separated by blank lines.
        let mut records: Vec<Vec<String>> = Vec::new();
        let mut current: Vec<String> = Vec::new();
        for l in expanded_lines {
            if l.trim().is_empty() {
                if !current.is_empty() {
                    records.push(current);
                    current = Vec::new();
                }
            } else {
                current.push(l);
            }
        }
        if !current.is_empty() {
            records.push(current);
        }

        // Execute records using the EngineHarness directly.
        use sqllogictest::AsyncDB;
        let mut harness = slt::EngineHarness::new();

        for rec in records {
            // find first meaningful line
            let mut idx = 0usize;
            while idx < rec.len() && rec[idx].trim_start().starts_with('#') {
                idx += 1;
            }
            if idx >= rec.len() {
                continue;
            }
            let header = rec[idx].trim_start();

            if header.starts_with("statement") {
                // expect next non-empty line to be SQL
                let mut si = idx + 1;
                while si < rec.len() && rec[si].trim().is_empty() {
                    si += 1;
                }
                if si >= rec.len() {
                    panic!("statement record with no SQL in {}", path.display());
                }
                let sql = rec[si].trim();

                match <slt::EngineHarness as AsyncDB>::run(&mut harness, sql).await {
                    Ok(sqllogictest::DBOutput::StatementComplete(_))
                    | Ok(sqllogictest::DBOutput::Rows { .. }) => {
                        // success
                    }
                    Ok(_) => {}
                    Err(e) => panic!(
                        "  FAIL: {}: statement failed: {}\n[SQL] {}",
                        path.display(),
                        e,
                        sql
                    ),
                }
            } else if header.starts_with("query") {
                // next non-empty line is SQL, then a line with ----, then expected rows
                let mut qi = idx + 1;
                while qi < rec.len() && rec[qi].trim().is_empty() {
                    qi += 1;
                }
                if qi >= rec.len() {
                    panic!("query record with no SQL in {}", path.display());
                }
                let sql = rec[qi].trim();
                // find ---- separator
                let mut sep = qi + 1;
                while sep < rec.len() && rec[sep].trim() != "----" {
                    sep += 1;
                }
                if sep >= rec.len() {
                    panic!("query record missing ---- separator in {}", path.display());
                }
                let mut expected: Vec<String> = Vec::new();
                for r in rec.iter().skip(sep + 1) {
                    expected.push(r.clone());
                }

                let got = match <slt::EngineHarness as AsyncDB>::run(&mut harness, sql).await {
                    Ok(sqllogictest::DBOutput::Rows { rows, .. }) => rows,
                    Ok(_) => panic!(
                        "  FAIL: {}: expected rows but got non-row DBOutput for {}",
                        path.display(),
                        sql
                    ),
                    Err(e) => panic!(
                        "  FAIL: {}: query failed: {}\n[SQL] {}",
                        path.display(),
                        e,
                        sql
                    ),
                };

                let normalize = sqllogictest::runner::default_normalizer;
                let got_lines: Vec<String> = got
                    .into_iter()
                    .map(|row| {
                        row.into_iter()
                            .map(|cell| normalize(&cell))
                            .collect::<Vec<String>>()
                            .join(" ")
                    })
                    .collect();
                let exp_lines: Vec<String> = expected
                    .into_iter()
                    .map(|s| normalize(&s.trim().to_string()))
                    .collect();
                if got_lines != exp_lines {
                    panic!(
                        "  FAIL: {}: query mismatch\n[SQL] {}\nexpected: {:?}\ngot: {:?}",
                        path.display(),
                        sql,
                        exp_lines,
                        got_lines
                    );
                }
            } else {
                // ignore other headers (name:, description:, group:, comments)
            }
        }

        tracing::info!("  PASS: {}", path.display());
    }
}
