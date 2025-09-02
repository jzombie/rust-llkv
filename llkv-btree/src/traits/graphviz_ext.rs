use crate::bplus_tree::{BPlusTree, Node, SharedBPlusTree};
use crate::codecs::{IdCodec, KeyCodec};
use crate::errors::Error;
use crate::pager::Pager;

pub trait GraphvizExt {
    fn to_dot(&self) -> Result<String, Error>;

    /// page-id differences do not break snapshot comparisons.
    fn to_canonicalized_dot(&self) -> Result<String, Error> {
        use std::collections::BTreeMap;

        // Own the String; do NOT take &self.to_dot()? as &str.
        let dot = self.to_dot()?;

        // 1) Normalize line endings to LF.
        let mut lf = String::with_capacity(dot.len());
        let mut it = dot.chars().peekable();
        while let Some(ch) = it.next() {
            if ch == '\r' {
                if matches!(it.peek(), Some('\n')) {
                    it.next(); // swallow the '\n' of CRLF
                }
                lf.push('\n');
            } else {
                lf.push(ch);
            }
        }

        // 2) Canonicalize 'node_<...>' to N0, N1, ...
        let b = lf.as_bytes();
        let mut out = String::with_capacity(lf.len());
        let mut map: BTreeMap<String, String> = BTreeMap::new();
        let mut next = 0usize;
        let mut i = 0usize;

        while i < b.len() {
            if i + 5 <= b.len() && &b[i..i + 5] == b"node_" {
                let mut j = i + 5;
                while j < b.len() {
                    let c = b[j];
                    let ok = (c as char).is_ascii_alphanumeric() || c == b'_';
                    if !ok {
                        break;
                    }
                    j += 1;
                }
                let tag = &lf[i..j];
                let name = map.entry(tag.to_string()).or_insert_with(|| {
                    let s = format!("N{}", next);
                    next += 1;
                    s
                });
                out.push_str(name);
                i = j;
            } else {
                out.push(b[i] as char);
                i += 1;
            }
        }

        // 3) Collapse horizontal whitespace, preserve newlines.
        let mut norm = String::with_capacity(out.len());
        let mut in_hspace = false;
        for ch in out.chars() {
            if ch == '\n' {
                norm.push('\n');
                in_hspace = false;
            } else if ch.is_ascii_whitespace() {
                if !in_hspace {
                    norm.push(' ');
                    in_hspace = true;
                }
            } else {
                norm.push(ch);
                in_hspace = false;
            }
        }

        Ok(norm)
    }
}

impl<P, KC, IC> GraphvizExt for BPlusTree<P, KC, IC>
where
    P: Pager,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    fn to_dot(&self) -> Result<String, Error> {
        let mut dot = String::new();
        dot.push_str("digraph BPlusTree {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=record, style=filled];\n\n");

        let mut visited = rustc_hash::FxHashSet::default();
        if self.read_node(&self.root)?.entry_count() == 0 {
            dot.push_str("}\n");
            return Ok(dot);
        }

        fn walk<P, KC, IC>(
            tree: &BPlusTree<P, KC, IC>,
            node_id: &P::Id,
            dot: &mut String,
            visited: &mut rustc_hash::FxHashSet<P::Id>,
        ) -> Result<(), Error>
        where
            P: Pager,
            KC: KeyCodec,
            IC: IdCodec<Id = P::Id>,
        {
            if !visited.insert(node_id.clone()) {
                return Ok(());
            }

            let node = &tree.read_node(node_id)?;
            let node_name = format!("node_{:?}", node_id).replace('\"', "");

            match &node {
                Node::Internal { entries } => {
                    let keys_str = entries
                        .iter()
                        .map(|(k, _)| format!("<{:?}> {:?}", k, k))
                        .collect::<Vec<_>>()
                        .join(" | ");

                    dot.push_str(&format!(
                        "  {} [label=\"{}\", fillcolor=lightblue];\n",
                        node_name, keys_str
                    ));

                    for (k, child_id) in entries {
                        let child_name = format!("node_{:?}", child_id).replace('\"', "");
                        dot.push_str(&format!(
                            "  {}:\"<{:?}>\" -> {};\n",
                            node_name, k, child_name
                        ));
                        walk(&tree, child_id, dot, visited)?;
                    }
                }
                Node::Leaf { entries, next } => {
                    let entries_str = entries
                        .iter()
                        .map(|(k, _)| format!("{{{:?} | (data)}}", k))
                        .collect::<Vec<_>>()
                        .join(" | ");
                    dot.push_str(&format!(
                        "  {} [label=\"{}\", fillcolor=lightgreen];\n",
                        node_name, entries_str
                    ));
                    if let Some(next_id) = next {
                        let next_name = format!("node_{:?}", next_id).replace('\"', "");
                        dot.push_str(&format!(
                            "  {} -> {} [style=dashed, constraint=false];\n",
                            node_name, next_name
                        ));
                    }
                }
            }
            Ok(())
        }

        walk(self, &self.root, &mut dot, &mut visited)?;

        dot.push_str("}\n");
        Ok(dot)
    }
}

impl<P, KC, IC> GraphvizExt for SharedBPlusTree<P, KC, IC>
where
    P: Pager + Clone,
    P::Page: Send + Sync + 'static,
    P::Id: Send + Sync + 'static,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    fn to_dot(&self) -> Result<String, Error> {
        let snap = self.snapshot();
        // Call the BPlusTree impl explicitly.
        <BPlusTree<P, KC, IC> as GraphvizExt>::to_dot(&snap)
    }
}
