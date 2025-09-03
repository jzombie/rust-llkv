#[cfg(feature = "debug")]
use crate::{
    bplus_tree::{BPlusTree, Node, SharedBPlusTree},
    codecs::{IdCodec, KeyCodec},
    errors::Error,
    pager::Pager,
};

#[cfg(feature = "debug")]
use line_ending::LineEnding;

#[cfg(feature = "debug")]
pub trait GraphvizExt {
    fn to_dot(&self) -> Result<String, Error>;

    /// page-id differences do not break snapshot comparisons.
    fn to_canonicalized_dot(&self) -> Result<String, Error> {
        use std::collections::BTreeMap;

        // Own the String and normalize line endings via the crate.
        let dot_raw = self.to_dot()?;
        let dot = LineEnding::normalize(&dot_raw);

        // Canonicalize 'node_<...>' to N0, N1, ...
        let b = dot.as_bytes();
        let mut out = String::with_capacity(dot.len());
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
                let tag = &dot[i..j];
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

        // Collapse horizontal whitespace, preserve '\n'.
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

#[cfg(feature = "debug")]
impl<P, KC, IC> GraphvizExt for BPlusTree<P, KC, IC>
where
    P: Pager,
    KC: KeyCodec,
    IC: IdCodec<Id = P::Id>,
{
    fn to_dot(&self) -> Result<String, Error> {
        // Snapshot the current root while holding the lock briefly,
        // then drop the lock before any calls that may re-lock state.
        let root_id = {
            let s = self.state.lock().unwrap();
            s.root.clone()
        };

        let mut dot = String::new();
        dot.push_str("digraph BPlusTree {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=record, style=filled];\n\n");

        let mut visited = rustc_hash::FxHashSet::default();
        if self.read_node(&root_id)?.entry_count() == 0 {
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
                        walk(tree, child_id, dot, visited)?;
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

        walk(self, &root_id, &mut dot, &mut visited)?;

        dot.push_str("}\n");
        Ok(dot)
    }
}

#[cfg(feature = "debug")]
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
