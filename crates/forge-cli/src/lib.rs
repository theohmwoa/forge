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
    run_agent_from(None, agent, storage).await
}

/// Like [`run_agent`], but the first emitted step's `parent` is the given
/// `starting_parent`. Used by `forge fork --continue` to thread a continuation
/// onto the rewritten step.
///
/// Only the newly emitted step hashes are returned; the prefix stays implicit.
pub async fn run_agent_from<A: Agent + ?Sized, S: Storage + ?Sized>(
    starting_parent: Option<NodeHash>,
    agent: &mut A,
    storage: &S,
) -> anyhow::Result<Vec<NodeHash>> {
    let mut parent = starting_parent;
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

/// Fork a chain at the step matching `at_prefix`, replacing its content with
/// `rewrite`. The new chain shares the prefix up to (but not including) the
/// rewritten step; everything after is dropped.
///
/// Interpretation of `rewrite` depends on the target step's kind:
/// - `Prompt` / `Message`: raw text replaces the content field.
/// - `ToolCall`: parsed as JSON, replaces the `input` field. The tool name
///   and `call_id` are preserved.
/// - `ToolResult`: parsed as JSON, replaces the `output` field. The
///   `call_id` is preserved.
pub async fn fork_chain<S: Storage + ?Sized>(
    storage: &S,
    chain: &[Step],
    at_prefix: &str,
    rewrite: &str,
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
    let new_kind = rewrite_kind(&chain[pos].kind, rewrite)?;
    let new_step = Step::new(parent, new_kind, now_ms());
    let new_id = storage.put(new_step.clone()).await?;

    let mut new_chain: Vec<NodeHash> = chain[..pos].iter().map(|s| s.id.clone()).collect();
    new_chain.push(new_id);
    Ok(new_chain)
}

fn rewrite_kind(kind: &StepKind, raw: &str) -> anyhow::Result<StepKind> {
    match kind {
        StepKind::Prompt { model, .. } => Ok(StepKind::Prompt {
            model: model.clone(),
            content: raw.into(),
        }),
        StepKind::Message { role, .. } => Ok(StepKind::Message {
            role: role.clone(),
            content: raw.into(),
        }),
        StepKind::ToolCall { call_id, name, .. } => {
            let input: serde_json::Value = serde_json::from_str(raw).map_err(|e| {
                anyhow::anyhow!("--rewrite must be valid JSON for tool_call steps: {e}")
            })?;
            Ok(StepKind::ToolCall {
                call_id: call_id.clone(),
                name: name.clone(),
                input,
            })
        }
        StepKind::ToolResult { call_id, .. } => {
            let output: serde_json::Value = serde_json::from_str(raw).map_err(|e| {
                anyhow::anyhow!("--rewrite must be valid JSON for tool_result steps: {e}")
            })?;
            Ok(StepKind::ToolResult {
                call_id: call_id.clone(),
                output,
            })
        }
    }
}

/// After a fork, append a tool execution if the rewritten step was a
/// `ToolCall`. Looks up `name` in `tools`, runs it with the new input, and
/// returns the new `ToolResult` step's hash. No-op for non-tool-call kinds.
///
/// Returns `Ok(None)` when the last step in `chain` isn't a `ToolCall`.
pub async fn auto_run_tool_after_fork<S: Storage + ?Sized>(
    storage: &S,
    chain: &mut Vec<NodeHash>,
    tools: &[std::sync::Arc<dyn forge_core::tool::Tool>],
) -> anyhow::Result<Option<NodeHash>> {
    let last_hash = chain
        .last()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("chain is empty; nothing to extend"))?;
    let last_step = storage
        .get(&last_hash)
        .await?
        .ok_or_else(|| anyhow::anyhow!("step missing: {last_hash}"))?;

    let (call_id, name, input) = match &last_step.kind {
        StepKind::ToolCall {
            call_id,
            name,
            input,
        } => (call_id.clone(), name.clone(), input.clone()),
        _ => return Ok(None),
    };

    let tool = tools
        .iter()
        .find(|t| t.name() == name)
        .ok_or_else(|| anyhow::anyhow!("tool {name:?} not registered; pass --tools {name}"))?;
    let output = match tool.run(&input).await {
        Ok(v) => v,
        Err(e) => serde_json::json!(e.to_string()),
    };
    let result_step = Step::new(
        Some(last_hash),
        StepKind::ToolResult { call_id, output },
        now_ms(),
    );
    let result_id = storage.put(result_step).await?;
    chain.push(result_id.clone());
    Ok(Some(result_id))
}

#[derive(Debug, Clone)]
pub struct DiffResult {
    pub common_prefix_len: usize,
    pub aligned: Vec<AlignedStep>,
}

#[derive(Debug, Clone)]
pub enum AlignedStep {
    /// Same hash on both sides (rare in the divergent tail; would be in
    /// the prefix if the parents lined up too).
    Match(Step, Step),
    /// Same structural signature (kind + name/role), different content.
    Modified(Step, Step),
    OnlyA(Step),
    OnlyB(Step),
}

/// Diff two chains. Phase 1 uses content-addressed equality to find the
/// shared prefix (cheap). Phase 2 walks the divergent tails and aligns by
/// step *signature* (kind + name/role) using LCS, then emits per-position
/// modifications. A step in A and a step in B with the same signature but
/// different content show up as `Modified`; otherwise as `OnlyA` / `OnlyB`.
pub fn diff_chains(a: &[Step], b: &[Step]) -> DiffResult {
    let mut common = 0;
    for (sa, sb) in a.iter().zip(b.iter()) {
        if sa.id == sb.id {
            common += 1;
        } else {
            break;
        }
    }
    let a_tail = &a[common..];
    let b_tail = &b[common..];

    let sigs_a: Vec<String> = a_tail.iter().map(|s| signature(&s.kind)).collect();
    let sigs_b: Vec<String> = b_tail.iter().map(|s| signature(&s.kind)).collect();
    let pairs = lcs_pairs(&sigs_a, &sigs_b);

    let aligned = pairs
        .into_iter()
        .map(|p| match p {
            (Some(i), Some(j)) => {
                if a_tail[i].id == b_tail[j].id {
                    AlignedStep::Match(a_tail[i].clone(), b_tail[j].clone())
                } else {
                    AlignedStep::Modified(a_tail[i].clone(), b_tail[j].clone())
                }
            }
            (Some(i), None) => AlignedStep::OnlyA(a_tail[i].clone()),
            (None, Some(j)) => AlignedStep::OnlyB(b_tail[j].clone()),
            (None, None) => unreachable!(),
        })
        .collect();

    DiffResult {
        common_prefix_len: common,
        aligned,
    }
}

/// Step signature for alignment: collapses content but keeps structural
/// identity (kind + tool name + message role). Two steps with the same
/// signature are "the same kind of action," and worth aligning.
fn signature(kind: &StepKind) -> String {
    match kind {
        StepKind::Prompt { .. } => "prompt".into(),
        StepKind::Message { role, .. } => format!("message:{role}"),
        StepKind::ToolCall { name, .. } => format!("tool_call:{name}"),
        StepKind::ToolResult { .. } => "tool_result".into(),
    }
}

/// LCS alignment with forward-greedy traceback. The forward direction makes
/// the algorithm prefer the *earliest* viable match in A when there's a tie —
/// which matches user intuition: if a chain was forked at step 1, B's single
/// message should align with A's first message, not the last one of the same
/// kind. O(N*M) time and space; fine for typical agent chain lengths.
fn lcs_pairs<T: Eq>(a: &[T], b: &[T]) -> Vec<(Option<usize>, Option<usize>)> {
    let n = a.len();
    let m = b.len();
    // dp[i][j] = LCS length of a[i..] and b[j..]
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i][j] = if a[i] == b[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n && j < m {
        if a[i] == b[j] {
            out.push((Some(i), Some(j)));
            i += 1;
            j += 1;
        } else if dp[i + 1][j] >= dp[i][j + 1] {
            out.push((Some(i), None));
            i += 1;
        } else {
            out.push((None, Some(j)));
            j += 1;
        }
    }
    while i < n {
        out.push((Some(i), None));
        i += 1;
    }
    while j < m {
        out.push((None, Some(j)));
        j += 1;
    }
    out
}

pub fn render_diff(diff: &DiffResult) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "shared prefix: {} step(s)\n",
        diff.common_prefix_len
    ));
    if diff.aligned.is_empty() {
        out.push_str("\nno divergence — runs are identical\n");
        return out;
    }
    out.push_str("\nlegend: =  same   ~  modified   -  only in A   +  only in B\n\n");

    for entry in &diff.aligned {
        match entry {
            AlignedStep::Match(a, _b) => {
                out.push_str(&format!("=  {}\n", label(&a.kind)));
            }
            AlignedStep::Modified(a, b) => {
                out.push_str(&format!("~  {}\n", label(&a.kind)));
                out.push_str(&format!("-    {}\n", brief(&a.kind)));
                out.push_str(&format!("+    {}\n", brief(&b.kind)));
            }
            AlignedStep::OnlyA(s) => {
                out.push_str(&format!("-  {}\n", label(&s.kind)));
                out.push_str(&format!("-    {}\n", brief(&s.kind)));
            }
            AlignedStep::OnlyB(s) => {
                out.push_str(&format!("+  {}\n", label(&s.kind)));
                out.push_str(&format!("+    {}\n", brief(&s.kind)));
            }
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
    async fn fork_rewrites_tool_call_input_as_json() {
        let storage = MemoryStorage::new();
        let chain = build_chain(
            &storage,
            vec![
                msg("user", "do math"),
                StepKind::ToolCall {
                    call_id: "t1".into(),
                    name: "calculator".into(),
                    input: serde_json::json!({"op":"add","a":1,"b":2}),
                },
            ],
        )
        .await;
        let at: String = chain[1].id.0.chars().take(8).collect();
        let new_chain = fork_chain(&storage, &chain, &at, r#"{"op":"mul","a":7,"b":8}"#)
            .await
            .unwrap();
        let new_step = storage.get(&new_chain[1]).await.unwrap().unwrap();
        match new_step.kind {
            StepKind::ToolCall { name, input, .. } => {
                assert_eq!(name, "calculator");
                assert_eq!(input["op"], "mul");
                assert_eq!(input["a"], 7);
            }
            _ => panic!("expected ToolCall after rewrite"),
        }
    }

    #[tokio::test]
    async fn fork_rejects_invalid_json_for_tool_steps() {
        let storage = MemoryStorage::new();
        let chain = build_chain(
            &storage,
            vec![StepKind::ToolResult {
                call_id: "t1".into(),
                output: serde_json::json!(5),
            }],
        )
        .await;
        let at: String = chain[0].id.0.chars().take(8).collect();
        let err = fork_chain(&storage, &chain, &at, "not json at all")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("valid JSON"));
    }

    #[tokio::test]
    async fn auto_run_tool_extends_chain_with_fresh_result() {
        use forge_core::tool::Calculator;
        let storage = MemoryStorage::new();
        let chain = build_chain(
            &storage,
            vec![StepKind::ToolCall {
                call_id: "t1".into(),
                name: "calculator".into(),
                input: serde_json::json!({"op":"mul","a":7,"b":8}),
            }],
        )
        .await;
        let mut hashes: Vec<NodeHash> = chain.iter().map(|s| s.id.clone()).collect();
        let tools: Vec<std::sync::Arc<dyn forge_core::tool::Tool>> =
            vec![std::sync::Arc::new(Calculator)];
        let appended = auto_run_tool_after_fork(&storage, &mut hashes, &tools)
            .await
            .unwrap();
        assert!(appended.is_some());
        assert_eq!(hashes.len(), 2);
        let result = storage.get(hashes.last().unwrap()).await.unwrap().unwrap();
        match result.kind {
            StepKind::ToolResult { output, .. } => assert_eq!(output, serde_json::json!(56.0)),
            _ => panic!("expected ToolResult"),
        }
    }

    #[tokio::test]
    async fn auto_run_tool_is_noop_for_non_tool_call_tail() {
        let storage = MemoryStorage::new();
        let chain = build_chain(&storage, vec![msg("user", "hi")]).await;
        let mut hashes: Vec<NodeHash> = chain.iter().map(|s| s.id.clone()).collect();
        let tools: Vec<std::sync::Arc<dyn forge_core::tool::Tool>> = vec![];
        let appended = auto_run_tool_after_fork(&storage, &mut hashes, &tools)
            .await
            .unwrap();
        assert!(appended.is_none());
        assert_eq!(hashes.len(), 1);
    }

    #[tokio::test]
    async fn diff_aligns_modified_step_then_extra_a() {
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
        // Same prefix, then assistant says something else, B stops there.
        let b_first = a[0].clone();
        let b_second = Step::new(
            Some(b_first.id.clone()),
            msg("assistant", "different reply"),
            0,
        );
        storage.put(b_second.clone()).await.unwrap();
        let b = vec![b_first, b_second];

        let diff = diff_chains(&a, &b);
        assert_eq!(diff.common_prefix_len, 1);
        // Tail A: [assistant("hello"), user("and now?")]
        // Tail B: [assistant("different reply")]
        // Alignment by signature: assistant matches assistant (Modified),
        // A's user("and now?") has no peer (OnlyA).
        assert_eq!(diff.aligned.len(), 2);
        assert!(matches!(diff.aligned[0], AlignedStep::Modified(_, _)));
        assert!(matches!(diff.aligned[1], AlignedStep::OnlyA(_)));
    }

    #[tokio::test]
    async fn diff_full_match_has_no_aligned_entries() {
        let storage = MemoryStorage::new();
        let a = build_chain(&storage, vec![msg("user", "hi")]).await;
        let diff = diff_chains(&a, &a);
        assert_eq!(diff.common_prefix_len, 1);
        assert!(diff.aligned.is_empty());
    }

    #[tokio::test]
    async fn diff_aligns_across_inserted_step() {
        let storage = MemoryStorage::new();
        // A: user -> assistant (1 step)
        // B: user -> assistant (1 step) -> user (extra)
        // After common prefix of 0, signature LCS pairs assistant<->assistant
        // and surfaces B's extra user as OnlyB.
        let a = build_chain(
            &storage,
            vec![msg("user", "hi"), msg("assistant", "hi back")],
        )
        .await;
        let b = build_chain(
            &storage,
            vec![
                msg("user", "hi"),
                msg("assistant", "hi back"),
                msg("user", "follow up"),
            ],
        )
        .await;
        let diff = diff_chains(&a, &b);
        // Both share user("hi") + assistant("hi back") (same hashes)
        assert_eq!(diff.common_prefix_len, 2);
        assert_eq!(diff.aligned.len(), 1);
        assert!(matches!(diff.aligned[0], AlignedStep::OnlyB(_)));
    }

    #[test]
    fn lcs_simple() {
        let a = vec!["a", "b", "c"];
        let b = vec!["a", "x", "c"];
        let pairs = lcs_pairs(&a, &b);
        assert_eq!(pairs.len(), 4);
        assert_eq!(pairs[0], (Some(0), Some(0)));
        assert_eq!(pairs[3], (Some(2), Some(2)));
    }

    #[test]
    fn lcs_picks_earliest_match_in_a() {
        // a has "m" at index 0 and 3; b has "m" once.
        // Forward greedy traceback should pair b[0] with a[0], not a[3].
        let a = vec!["m", "t", "r", "m"];
        let b = vec!["m"];
        let pairs = lcs_pairs(&a, &b);
        assert_eq!(pairs.len(), 4);
        assert_eq!(pairs[0], (Some(0), Some(0)));
        assert_eq!(pairs[1], (Some(1), None));
        assert_eq!(pairs[2], (Some(2), None));
        assert_eq!(pairs[3], (Some(3), None));
    }
}
