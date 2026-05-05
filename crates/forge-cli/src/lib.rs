//! Runtime helpers for the `forge` CLI: run, fork, and diff over the run
//! graph.

use std::time::{SystemTime, UNIX_EPOCH};

use forge_core::agent::Agent;
use forge_core::{NodeHash, Step, StepKind};
use forge_storage::Storage;

/// Drive `agent` to completion, persisting every step into `storage`.
/// Returns the chain of node hashes in emission order.
pub async fn run_agent<A: Agent + ?Sized, S: Storage + ?Sized>(
    agent: &mut A,
    storage: &S,
) -> anyhow::Result<Vec<NodeHash>> {
    let mut parent: Option<NodeHash> = None;
    let mut chain = Vec::new();

    while let Some(kind) = agent.next_step(parent.clone()).await {
        let ts = now_ms();
        let step = Step::new(parent.clone(), kind, ts);
        let id = storage.put(step).await?;
        chain.push(id.clone());
        parent = Some(id);
    }

    Ok(chain)
}

/// Print a chain of steps in DAG-walk order.
pub async fn print_chain<S: Storage + ?Sized>(
    storage: &S,
    chain: &[NodeHash],
) -> anyhow::Result<()> {
    for (i, hash) in chain.iter().enumerate() {
        let step = storage
            .get(hash)
            .await?
            .ok_or_else(|| anyhow::anyhow!("step missing from storage: {hash}"))?;
        let parent_short = step
            .parent
            .as_ref()
            .map(|p| short(&p.0))
            .unwrap_or_else(|| "(root)".into());
        println!(
            "{i:>3}  {id}  parent={parent_short}  {label}",
            id = short(&hash.0),
            label = label(&step.kind),
        );
    }
    Ok(())
}

/// Fork a chain at the step matching `at_prefix`, replacing its text content
/// with `rewrite_text`. The new chain shares the prefix up to (but not
/// including) the rewritten step; everything after is dropped — the user
/// drives a fresh agent forward later.
///
/// Only `Prompt` and `Message` step kinds are rewritable in v0; tool-call
/// rewriting needs the tool-use loop to land first.
pub async fn fork_chain<S: Storage + ?Sized>(
    storage: &S,
    chain: &[Step],
    at_prefix: &str,
    rewrite_text: &str,
) -> anyhow::Result<Vec<NodeHash>> {
    let matches: Vec<usize> = chain
        .iter()
        .enumerate()
        .filter(|(_, s)| s.id.0.starts_with(at_prefix))
        .map(|(i, _)| i)
        .collect();
    let pos = match matches.as_slice() {
        [] => anyhow::bail!("no step in chain matches prefix {at_prefix:?}"),
        [p] => *p,
        _ => anyhow::bail!(
            "ambiguous --at prefix {at_prefix:?}: matches {} steps",
            matches.len()
        ),
    };

    let parent = if pos == 0 {
        None
    } else {
        Some(chain[pos - 1].id.clone())
    };
    let new_kind = rewrite_kind_text(&chain[pos].kind, rewrite_text)?;
    let new_step = Step::new(parent, new_kind, now_ms());
    let new_id = storage.put(new_step.clone()).await?;

    let mut new_chain: Vec<NodeHash> = chain[..pos].iter().map(|s| s.id.clone()).collect();
    new_chain.push(new_id);
    Ok(new_chain)
}

fn rewrite_kind_text(kind: &StepKind, new_text: &str) -> anyhow::Result<StepKind> {
    match kind {
        StepKind::Prompt { model, .. } => Ok(StepKind::Prompt {
            model: model.clone(),
            content: new_text.into(),
        }),
        StepKind::Message { role, .. } => Ok(StepKind::Message {
            role: role.clone(),
            content: new_text.into(),
        }),
        StepKind::ToolCall { .. } | StepKind::ToolResult { .. } => {
            anyhow::bail!("--rewrite-text only supports prompt or message steps in v0")
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiffResult {
    pub common_prefix_len: usize,
    pub a_tail: Vec<Step>,
    pub b_tail: Vec<Step>,
}

/// Walk two chains pairwise, find the first divergent step. The shared
/// prefix is content-addressed equal (same hashes), so equality is cheap.
pub fn diff_chains(a: &[Step], b: &[Step]) -> DiffResult {
    let mut common = 0;
    for (sa, sb) in a.iter().zip(b.iter()) {
        if sa.id == sb.id {
            common += 1;
        } else {
            break;
        }
    }
    DiffResult {
        common_prefix_len: common,
        a_tail: a[common..].to_vec(),
        b_tail: b[common..].to_vec(),
    }
}

pub fn render_diff(diff: &DiffResult) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "shared prefix: {} step(s)\n",
        diff.common_prefix_len
    ));
    out.push_str("\n--- only in A ---\n");
    if diff.a_tail.is_empty() {
        out.push_str("(empty)\n");
    } else {
        for s in &diff.a_tail {
            out.push_str(&format!("{}  {}\n", short(&s.id.0), label(&s.kind)));
            out.push_str(&format!("        {}\n", brief(&s.kind)));
        }
    }
    out.push_str("\n--- only in B ---\n");
    if diff.b_tail.is_empty() {
        out.push_str("(empty)\n");
    } else {
        for s in &diff.b_tail {
            out.push_str(&format!("{}  {}\n", short(&s.id.0), label(&s.kind)));
            out.push_str(&format!("        {}\n", brief(&s.kind)));
        }
    }
    out
}

fn label(kind: &StepKind) -> String {
    match kind {
        StepKind::Prompt { model, .. } => format!("prompt[{model}]"),
        StepKind::Message { role, .. } => format!("message[{role}]"),
        StepKind::ToolCall { name, .. } => format!("tool_call[{name}]"),
        StepKind::ToolResult { call_id, .. } => format!("tool_result[{call_id}]"),
    }
}

/// One-line content preview, truncated.
fn brief(kind: &StepKind) -> String {
    let raw = match kind {
        StepKind::Prompt { content, .. } => content.clone(),
        StepKind::Message { content, .. } => content.clone(),
        StepKind::ToolCall { input, .. } => input.to_string(),
        StepKind::ToolResult { output, .. } => output.to_string(),
    };
    let trimmed = raw.replace('\n', " ");
    if trimmed.chars().count() > 80 {
        let head: String = trimmed.chars().take(77).collect();
        format!("{head}...")
    } else {
        trimmed
    }
}

fn short(s: &str) -> String {
    s.chars().take(10).collect()
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_core::StepKind;
    use forge_storage::MemoryStorage;

    fn msg(role: &str, content: &str) -> StepKind {
        StepKind::Message {
            role: role.into(),
            content: content.into(),
        }
    }

    async fn build_chain(storage: &MemoryStorage, kinds: Vec<StepKind>) -> Vec<Step> {
        let mut parent: Option<NodeHash> = None;
        let mut steps = Vec::new();
        for kind in kinds {
            let step = Step::new(parent.clone(), kind, 0);
            storage.put(step.clone()).await.unwrap();
            parent = Some(step.id.clone());
            steps.push(step);
        }
        steps
    }

    #[tokio::test]
    async fn fork_shares_prefix_and_rewrites_step() {
        let storage = MemoryStorage::new();
        let chain = build_chain(&storage, vec![msg("user", "hi"), msg("assistant", "hello")]).await;
        let at_prefix: String = chain[1].id.0.chars().take(8).collect();
        let new_chain = fork_chain(&storage, &chain, &at_prefix, "rewritten")
            .await
            .unwrap();

        assert_eq!(new_chain.len(), 2);
        assert_eq!(new_chain[0], chain[0].id, "prefix should be shared");
        assert_ne!(new_chain[1], chain[1].id, "rewritten step has new hash");

        let new_step = storage.get(&new_chain[1]).await.unwrap().unwrap();
        if let StepKind::Message { content, .. } = new_step.kind {
            assert_eq!(content, "rewritten");
        } else {
            panic!("expected Message after rewrite");
        }
    }

    #[tokio::test]
    async fn fork_at_root_drops_parent() {
        let storage = MemoryStorage::new();
        let chain = build_chain(&storage, vec![msg("user", "hi")]).await;
        let at_prefix: String = chain[0].id.0.chars().take(8).collect();
        let new_chain = fork_chain(&storage, &chain, &at_prefix, "different")
            .await
            .unwrap();
        let new_step = storage.get(&new_chain[0]).await.unwrap().unwrap();
        assert!(new_step.parent.is_none());
        assert_ne!(new_chain[0], chain[0].id);
    }

    #[tokio::test]
    async fn fork_rejects_unknown_prefix() {
        let storage = MemoryStorage::new();
        let chain = build_chain(&storage, vec![msg("user", "hi")]).await;
        let err = fork_chain(&storage, &chain, "deadbeef", "x")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("no step"));
    }

    #[tokio::test]
    async fn diff_finds_first_divergence() {
        let storage = MemoryStorage::new();
        let a = build_chain(
            &storage,
            vec![
                msg("user", "hi"),
                msg("assistant", "hello"),
                msg("user", "and now?"),
            ],
        )
        .await;
        // Same prefix, different second step content -> divergence at index 1.
        let b_first = a[0].clone();
        let b_second_kind = msg("assistant", "different reply");
        let b_second = Step::new(Some(b_first.id.clone()), b_second_kind, 0);
        storage.put(b_second.clone()).await.unwrap();
        let b = vec![b_first, b_second];

        let diff = diff_chains(&a, &b);
        assert_eq!(diff.common_prefix_len, 1);
        assert_eq!(diff.a_tail.len(), 2);
        assert_eq!(diff.b_tail.len(), 1);
    }

    #[tokio::test]
    async fn diff_full_match_has_empty_tails() {
        let storage = MemoryStorage::new();
        let a = build_chain(&storage, vec![msg("user", "hi")]).await;
        let diff = diff_chains(&a, &a);
        assert_eq!(diff.common_prefix_len, 1);
        assert!(diff.a_tail.is_empty());
        assert!(diff.b_tail.is_empty());
    }
}
