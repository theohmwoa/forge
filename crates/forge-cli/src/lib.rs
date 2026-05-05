//! Runtime helpers for the `forge` CLI: run, fork, diff, and bisect over the
//! run graph.

use std::time::{SystemTime, UNIX_EPOCH};

use forge_core::agent::Agent;
use forge_core::{NodeHash, Step, StepKind};
use forge_storage::{RunMeta, Storage};

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

// -- bisect ----------------------------------------------------------------
//
// `git bisect` for agent runs. Given two chains that share a prefix — one
// that succeeds (`good`) and one that fails (`bad`) — walk the divergent
// tail step-by-step. At each substitutable divergence, fork `bad` at that
// step replacing the value with `good`'s, run the agent forward, and ask:
// did it recover? The first step where forking → success is the step that
// caused the failure.
//
// Distinct from `forge diff`: diff *describes* divergence, bisect *isolates
// causation*. Distinct from cassette-based replay: only possible because
// Forge's content-addressed graph makes per-step forks free.

#[derive(Debug, Clone)]
pub struct BisectOutcome {
    /// Index in the *bad* chain at which we forked.
    pub bad_index: usize,
    pub bad_step: Step,
    pub good_step: Step,
    /// Head of the chain produced by forking bad at `bad_index` with
    /// `good_step`'s content and replaying forward.
    pub fork_head: NodeHash,
    pub recovered: bool,
    /// Last assistant message text in the forked chain, if any. Useful for
    /// the rendering layer.
    pub final_text: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BisectResult {
    pub common_prefix_len: usize,
    pub trials: Vec<BisectOutcome>,
    /// Index into `trials` of the first step that, when substituted with
    /// `good`'s value, made the run succeed. `None` if no single
    /// substitution recovered the run (multiple things broke, or the check
    /// is too strict).
    pub first_recoverable: Option<usize>,
}

/// Default success criterion: chain ends with a non-empty `assistant`
/// message. Pass a custom `check` to `bisect_chains` for tighter assertions
/// (e.g. substring match against `good`'s answer).
pub fn default_bisect_check(steps: &[Step]) -> bool {
    steps.last().is_some_and(|s| match &s.kind {
        StepKind::Message { role, content } => role == "assistant" && !content.trim().is_empty(),
        _ => false,
    })
}

/// Walk the divergent tail of `chain_bad`. At each step where `bad` and
/// `good` have substitutable kinds, fork the bad chain at that step with
/// `good`'s value, run the agent forward, and check if the result passes.
///
/// `build_agent` is called once per trial with the rewritten prefix so the
/// caller can construct a *continuing* agent (Anthropic, OpenAI, Gemini, or
/// a test fake) bound to that prefix's history.
///
/// Bisect stops at the first recovered trial — the tweet-worthy case is one
/// step; if you need an exhaustive scan, call this once per substitutable
/// divergence with `take_until_recovered = false`. (Not yet exposed; the
/// internal walk hits the first recoverable step and returns.)
pub async fn bisect_chains<S, AgentBuilder, Check>(
    storage: &S,
    chain_good: &[Step],
    chain_bad: &[Step],
    mut build_agent: AgentBuilder,
    check: Check,
) -> anyhow::Result<BisectResult>
where
    S: Storage + ?Sized,
    AgentBuilder: FnMut(&[Step]) -> anyhow::Result<Box<dyn Agent>>,
    Check: Fn(&[Step]) -> bool,
{
    // Length of the content-addressed shared prefix.
    let mut common = 0;
    for (a, b) in chain_good.iter().zip(chain_bad.iter()) {
        if a.id == b.id {
            common += 1;
        } else {
            break;
        }
    }

    // Sanity: the bad run shouldn't already pass and the good run should.
    // We don't bail on these — print as warnings via tracing and let the
    // caller decide — but they often indicate a misconfigured `--expect`.
    if check(chain_bad) {
        tracing::warn!("bisect: the BAD chain already satisfies the check; nothing to bisect");
    }
    if !check(chain_good) {
        tracing::warn!("bisect: the GOOD chain doesn't satisfy the check — bisect may be vacuous");
    }

    let mut trials: Vec<BisectOutcome> = Vec::new();
    let mut first_recoverable: Option<usize> = None;

    let walk_end = chain_bad.len().min(chain_good.len());
    let chain_bad_owned: Vec<Step> = chain_bad.to_vec();

    for i in common..walk_end {
        let bad_step = chain_bad[i].clone();
        let good_step = chain_good[i].clone();

        // Substitution requires matching kind shapes — otherwise the rewrite
        // can't be interpreted correctly (text vs JSON, etc).
        if !same_kind_signature(&bad_step.kind, &good_step.kind) {
            continue;
        }

        // Use the full hash so fork_chain doesn't accidentally match a
        // different step at the same prefix.
        let rewrite = step_rewrite_content(&good_step);
        let new_chain = fork_chain(storage, &chain_bad_owned, &bad_step.id.0, &rewrite).await?;

        let mut prefix_steps: Vec<Step> = Vec::with_capacity(new_chain.len());
        for h in &new_chain {
            let s = storage
                .get(h)
                .await?
                .ok_or_else(|| anyhow::anyhow!("forked step missing: {h}"))?;
            prefix_steps.push(s);
        }
        let last_hash = new_chain.last().cloned().unwrap();

        // Drive the agent forward from the rewritten step. The agent's own
        // `max_turns` cap (set up by the caller) bounds runtime.
        let mut agent = build_agent(&prefix_steps)?;
        let appended = run_agent_from(Some(last_hash.clone()), agent.as_mut(), storage).await?;

        let mut all_steps = prefix_steps.clone();
        for h in &appended {
            let s = storage
                .get(h)
                .await?
                .ok_or_else(|| anyhow::anyhow!("appended step missing: {h}"))?;
            all_steps.push(s);
        }

        let recovered = check(&all_steps);
        let final_text = all_steps.last().and_then(|s| match &s.kind {
            StepKind::Message { role, content } if role == "assistant" => Some(content.clone()),
            _ => None,
        });

        let fork_head = appended.last().cloned().unwrap_or(last_hash);

        // Persist the trial so users can `forge view` / `forge web` it later.
        // Tagged so they're easy to filter or clean up.
        let root = chain_bad_owned[0].id.clone();
        let _ = storage
            .record_run(&RunMeta {
                head: fork_head.clone(),
                root,
                recorded_at_ms: now_ms(),
                tag: Some(format!("bisect-step-{i}")),
            })
            .await;

        trials.push(BisectOutcome {
            bad_index: i,
            bad_step,
            good_step,
            fork_head,
            recovered,
            final_text,
        });

        if recovered {
            first_recoverable = Some(trials.len() - 1);
            break;
        }
    }

    Ok(BisectResult {
        common_prefix_len: common,
        trials,
        first_recoverable,
    })
}

fn same_kind_signature(a: &StepKind, b: &StepKind) -> bool {
    use StepKind::*;
    matches!(
        (a, b),
        (Prompt { .. }, Prompt { .. })
            | (Message { .. }, Message { .. })
            | (ToolCall { .. }, ToolCall { .. })
            | (ToolResult { .. }, ToolResult { .. })
    )
}

fn step_rewrite_content(step: &Step) -> String {
    match &step.kind {
        StepKind::Prompt { content, .. } => content.clone(),
        StepKind::Message { content, .. } => content.clone(),
        StepKind::ToolCall { input, .. } => input.to_string(),
        StepKind::ToolResult { output, .. } => output.to_string(),
    }
}

pub fn render_bisect(result: &BisectResult) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "shared prefix: {} step(s)\n",
        result.common_prefix_len
    ));
    out.push_str(&format!("trials run:    {}\n\n", result.trials.len()));

    for t in &result.trials {
        let marker = if t.recovered {
            "RECOVERED"
        } else {
            "still failed"
        };
        out.push_str(&format!(
            "  step {:>2}  {:<24}  {}\n",
            t.bad_index,
            label(&t.bad_step.kind),
            marker
        ));
        if t.recovered {
            if let Some(text) = &t.final_text {
                let head: String = text.lines().next().unwrap_or("").chars().take(80).collect();
                out.push_str(&format!("            final: {head}\n"));
            }
        }
    }

    out.push('\n');
    if let Some(idx) = result.first_recoverable {
        let t = &result.trials[idx];
        out.push_str(&format!(
            "first recoverable divergence: step {} ({})\n",
            t.bad_index,
            label(&t.bad_step.kind)
        ));
        out.push_str(&format!("  bad:  {}\n", brief(&t.bad_step.kind)));
        out.push_str(&format!("  good: {}\n", brief(&t.good_step.kind)));
        out.push_str(&format!("  fork: {}\n", short(&t.fork_head.0)));
        out.push('\n');
        out.push_str(
            "the bad value at this step prevented recovery; substituting good's value succeeded.\n",
        );
    } else {
        out.push_str("no single substitution recovered the run.\n");
        out.push_str(
            "multiple steps may be implicated, or the check is too strict for the agent.\n",
        );
    }

    out
}

// -- audit -----------------------------------------------------------------
//
// "Am I overpaying?" 2026's most-cited LLM ops question. Teams default to
// frontier models during prototyping and never downgrade. Routing tools
// exist; audit tools don't. `forge audit` replays each recorded run through
// a cheaper target model, uses a strong model as judge to compare answers,
// and reports which prompts are safe to downgrade.
//
// The judge prompt is intentionally tight: produces a single verdict token
// (EQUIVALENT or DIFFERENT) followed by one short reason line. We parse the
// first token; the reason is for human review.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditVerdict {
    /// Cheaper model produced an answer the judge considers equivalent.
    Equivalent,
    /// Judge flagged a meaningful difference.
    Different,
    /// Judge response wasn't parseable; treat as failure to be conservative.
    Inconclusive,
}

#[derive(Debug, Clone)]
pub struct AuditOutcome {
    pub original_head: NodeHash,
    pub original_prompt: String,
    pub original_answer: String,
    pub candidate_answer: String,
    /// Head of the recorded candidate run (replay against the cheaper model).
    pub candidate_head: NodeHash,
    pub verdict: AuditVerdict,
    pub judge_reason: String,
}

#[derive(Debug, Clone, Default)]
pub struct AuditSummary {
    pub outcomes: Vec<AuditOutcome>,
}

impl AuditSummary {
    pub fn equivalent_count(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|o| o.verdict == AuditVerdict::Equivalent)
            .count()
    }
    pub fn different_count(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|o| o.verdict == AuditVerdict::Different)
            .count()
    }
    pub fn inconclusive_count(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|o| o.verdict == AuditVerdict::Inconclusive)
            .count()
    }
}

/// Pull the user-facing prompt and final assistant answer from a recorded
/// chain. Skips system messages; takes the first `Prompt` step and the
/// last assistant `Message`. Returns `None` for runs that don't match this
/// shape (e.g. ones that ended mid-tool-call).
pub fn extract_audit_pair(chain: &[Step]) -> Option<(String, String)> {
    let prompt = chain.iter().find_map(|s| match &s.kind {
        StepKind::Prompt { content, .. } => Some(content.clone()),
        _ => None,
    })?;
    let answer = chain.iter().rev().find_map(|s| match &s.kind {
        StepKind::Message { role, content } if role == "assistant" => Some(content.clone()),
        _ => None,
    })?;
    Some((prompt, answer))
}

/// Audit one recorded run against a cheaper model.
///
/// `build_target` produces a fresh agent for the cheaper model, given the
/// original prompt. `build_judge` produces a fresh agent for the judge,
/// given the assembled judge prompt. Both are async because building real
/// agents reads `from_env`.
///
/// The candidate run is persisted with tag `audit-cheap-<short>` so users
/// can browse it in `forge web` and verify the judge's call.
pub async fn audit_run<S, BuildTarget, BuildJudge>(
    storage: &S,
    chain: &[Step],
    original_head: NodeHash,
    target_label: &str,
    mut build_target: BuildTarget,
    mut build_judge: BuildJudge,
) -> anyhow::Result<Option<AuditOutcome>>
where
    S: Storage + ?Sized,
    BuildTarget: FnMut(&str) -> anyhow::Result<Box<dyn Agent>>,
    BuildJudge: FnMut(&str) -> anyhow::Result<Box<dyn Agent>>,
{
    let Some((original_prompt, original_answer)) = extract_audit_pair(chain) else {
        return Ok(None);
    };

    // 1) Run the cheaper model on the same prompt.
    let mut target = build_target(&original_prompt)?;
    let candidate_chain = run_agent(target.as_mut(), storage).await?;
    if candidate_chain.is_empty() {
        anyhow::bail!("audit: candidate model produced no steps");
    }
    let candidate_head = candidate_chain.last().cloned().unwrap();
    let candidate_steps: Vec<Step> = {
        let mut acc = Vec::with_capacity(candidate_chain.len());
        for h in &candidate_chain {
            acc.push(
                storage
                    .get(h)
                    .await?
                    .ok_or_else(|| anyhow::anyhow!("candidate step missing: {h}"))?,
            );
        }
        acc
    };
    let candidate_answer = candidate_steps
        .iter()
        .rev()
        .find_map(|s| match &s.kind {
            StepKind::Message { role, content } if role == "assistant" => Some(content.clone()),
            _ => None,
        })
        .unwrap_or_default();

    // Persist the candidate trial so it's reviewable in forge web.
    let root = candidate_steps[0].id.clone();
    let _ = storage
        .record_run(&RunMeta {
            head: candidate_head.clone(),
            root,
            recorded_at_ms: now_ms(),
            tag: Some(format!("audit-{target_label}")),
        })
        .await;

    // 2) Ask the judge.
    let judge_prompt = build_judge_prompt(&original_prompt, &original_answer, &candidate_answer);
    let mut judge = build_judge(&judge_prompt)?;
    let judge_chain = run_agent(judge.as_mut(), storage).await?;
    let judge_steps: Vec<Step> = {
        let mut acc = Vec::with_capacity(judge_chain.len());
        for h in &judge_chain {
            acc.push(
                storage
                    .get(h)
                    .await?
                    .ok_or_else(|| anyhow::anyhow!("judge step missing: {h}"))?,
            );
        }
        acc
    };
    let judge_answer = judge_steps
        .iter()
        .rev()
        .find_map(|s| match &s.kind {
            StepKind::Message { role, content } if role == "assistant" => Some(content.clone()),
            _ => None,
        })
        .unwrap_or_default();

    let (verdict, judge_reason) = parse_judge_verdict(&judge_answer);

    Ok(Some(AuditOutcome {
        original_head,
        original_prompt,
        original_answer,
        candidate_answer,
        candidate_head,
        verdict,
        judge_reason,
    }))
}

/// Audit many runs. Caller filters runs (e.g. by tag) and passes the
/// resulting heads in. Stops early on `--limit`. Errors on individual runs
/// are recorded as `Inconclusive` rather than aborting the whole audit.
pub async fn audit_runs<S, BuildTarget, BuildJudge>(
    storage: &S,
    heads: &[NodeHash],
    target_label: &str,
    mut build_target: BuildTarget,
    mut build_judge: BuildJudge,
) -> anyhow::Result<AuditSummary>
where
    S: Storage + ?Sized,
    BuildTarget: FnMut(&str) -> anyhow::Result<Box<dyn Agent>>,
    BuildJudge: FnMut(&str) -> anyhow::Result<Box<dyn Agent>>,
{
    let mut summary = AuditSummary::default();
    for head in heads {
        let chain = match storage.chain_to(head).await {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, head = %head, "audit: failed to load chain");
                continue;
            }
        };
        let result = audit_run(
            storage,
            &chain,
            head.clone(),
            target_label,
            &mut build_target,
            &mut build_judge,
        )
        .await;
        match result {
            Ok(Some(outcome)) => summary.outcomes.push(outcome),
            Ok(None) => {
                tracing::debug!(head = %head, "audit: skipping run with no Prompt+Message pair");
            }
            Err(e) => {
                tracing::warn!(error = %e, head = %head, "audit: trial failed; recording as inconclusive");
                summary.outcomes.push(AuditOutcome {
                    original_head: head.clone(),
                    original_prompt: String::new(),
                    original_answer: String::new(),
                    candidate_answer: String::new(),
                    candidate_head: head.clone(),
                    verdict: AuditVerdict::Inconclusive,
                    judge_reason: format!("trial errored: {e}"),
                });
            }
        }
    }
    Ok(summary)
}

/// The judge prompt is the load-bearing piece. Designed to be tight and to
/// produce a parseable single-token verdict. Equivalence is task-agnostic —
/// "would a user notice the difference?" — which biases toward DIFFERENT
/// when in doubt (the conservative direction for a cost-cut decision).
fn build_judge_prompt(prompt: &str, original: &str, candidate: &str) -> String {
    format!(
        "You are an impartial judge comparing two answers to the same user request.\n\
        \n\
        USER REQUEST:\n{prompt}\n\
        \n\
        ANSWER A (reference):\n{original}\n\
        \n\
        ANSWER B (candidate):\n{candidate}\n\
        \n\
        Would a user notice a meaningful difference between A and B for the request? \
        Consider correctness, completeness, structure, and whether key facts match. \
        Stylistic differences alone are not meaningful.\n\
        \n\
        Respond with exactly one word on the first line: EQUIVALENT or DIFFERENT.\n\
        On the second line, give a single short sentence explaining why."
    )
}

fn parse_judge_verdict(judge_text: &str) -> (AuditVerdict, String) {
    let mut lines = judge_text.lines();
    let first = lines.next().unwrap_or("").trim().to_uppercase();
    let reason = lines
        .next()
        .map(|l| l.trim().to_string())
        .unwrap_or_default();
    let verdict = if first.starts_with("EQUIVALENT") {
        AuditVerdict::Equivalent
    } else if first.starts_with("DIFFERENT") {
        AuditVerdict::Different
    } else {
        // Some judges put the verdict mid-line; do a lenient scan.
        let upper = judge_text.to_uppercase();
        if upper.contains("EQUIVALENT") && !upper.contains("DIFFERENT") {
            AuditVerdict::Equivalent
        } else if upper.contains("DIFFERENT") {
            AuditVerdict::Different
        } else {
            AuditVerdict::Inconclusive
        }
    };
    (verdict, reason)
}

pub fn render_audit(summary: &AuditSummary, target_label: &str) -> String {
    let total = summary.outcomes.len();
    let eq = summary.equivalent_count();
    let diff = summary.different_count();
    let inc = summary.inconclusive_count();

    let mut out = String::new();
    out.push_str(&format!(
        "audited {total} run(s) against {target_label}.\n\n"
    ));
    out.push_str(&format!("  ✓ {eq:>3} EQUIVALENT  — safe to downgrade\n"));
    out.push_str(&format!(
        "  ⚠ {diff:>3} DIFFERENT   — keep on the original model\n"
    ));
    if inc > 0 {
        out.push_str(&format!(
            "  ? {inc:>3} INCONCLUSIVE — judge couldn't decide\n"
        ));
    }
    out.push('\n');

    if diff > 0 {
        out.push_str("differences:\n");
        for o in &summary.outcomes {
            if o.verdict == AuditVerdict::Different {
                out.push_str(&format!(
                    "  • {}  {}\n",
                    short(&o.original_head.0),
                    one_line(&o.judge_reason, 80)
                ));
            }
        }
        out.push('\n');
    }

    if eq > 0 {
        out.push_str(&format!(
            "review the {eq} downgrades in forge web (tag: audit-{target_label}).\n"
        ));
    }
    out
}

fn one_line(s: &str, max: usize) -> String {
    let collapsed = s.lines().next().unwrap_or("").trim();
    if collapsed.chars().count() > max {
        let head: String = collapsed.chars().take(max - 3).collect();
        format!("{head}...")
    } else {
        collapsed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use forge_core::StepKind;
    use forge_storage::MemoryStorage;

    fn msg(role: &str, content: &str) -> StepKind {
        StepKind::Message {
            role: role.into(),
            content: content.into(),
        }
    }

    type RespondFn = Box<dyn FnMut(&[Step]) -> String + Send>;

    /// Test-only agent: emits one assistant message whose content depends on
    /// the prefix it sees. Exposes a closure so individual tests can pin the
    /// behavior they want.
    struct ScriptedAgent {
        respond: RespondFn,
        prefix: Vec<Step>,
        emitted: bool,
    }

    impl ScriptedAgent {
        fn new(prefix: Vec<Step>, respond: impl FnMut(&[Step]) -> String + Send + 'static) -> Self {
            Self {
                respond: Box::new(respond),
                prefix,
                emitted: false,
            }
        }
    }

    #[async_trait]
    impl Agent for ScriptedAgent {
        async fn next_step(&mut self, _parent: Option<NodeHash>) -> Option<StepKind> {
            if self.emitted {
                return None;
            }
            self.emitted = true;
            let content = (self.respond)(&self.prefix);
            if content.is_empty() {
                return None; // closure returning "" means "say nothing".
            }
            Some(StepKind::Message {
                role: "assistant".into(),
                content,
            })
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

    #[tokio::test]
    async fn bisect_finds_step_that_caused_failure() {
        let storage = MemoryStorage::new();

        // good: prompt -> assistant("Paris is the capital of France.")
        let good = build_chain(
            &storage,
            vec![
                StepKind::Prompt {
                    model: "test".into(),
                    content: "What is the capital of France?".into(),
                },
                msg("assistant", "Paris is the capital of France."),
            ],
        )
        .await;

        // bad: SAME prompt, but assistant got it wrong.
        // Different content at index 1 produces a different hash, so the
        // chains share a 1-step prefix (the prompt) and diverge at step 1.
        let bad = build_chain(
            &storage,
            vec![
                StepKind::Prompt {
                    model: "test".into(),
                    content: "What is the capital of France?".into(),
                },
                msg("assistant", "Lyon."),
            ],
        )
        .await;

        // Sanity: shared prefix is the prompt only.
        assert_eq!(good[0].id, bad[0].id);
        assert_ne!(good[1].id, bad[1].id);

        // Build agent that, after we fork bad with good's content, doesn't
        // need to do anything — the chain already ends in a Message. So
        // build_agent returns an "emit nothing" agent.
        let build_agent = |_prefix: &[Step]| -> anyhow::Result<Box<dyn Agent>> {
            Ok(Box::new(ScriptedAgent::new(Vec::new(), |_| String::new())) as Box<dyn Agent>)
        };

        // Custom check: the answer must contain "Paris".
        let check = |steps: &[Step]| {
            steps.last().is_some_and(|s| match &s.kind {
                StepKind::Message { role, content } => {
                    role == "assistant" && content.contains("Paris")
                }
                _ => false,
            })
        };

        let result = bisect_chains(&storage, &good, &bad, build_agent, check)
            .await
            .unwrap();

        assert_eq!(result.common_prefix_len, 1);
        assert_eq!(result.trials.len(), 1);
        assert!(result.trials[0].recovered);
        assert_eq!(result.trials[0].bad_index, 1);
        assert_eq!(result.first_recoverable, Some(0));
    }

    #[tokio::test]
    async fn bisect_skips_steps_with_mismatched_kinds() {
        let storage = MemoryStorage::new();

        let good = build_chain(
            &storage,
            vec![
                StepKind::Prompt {
                    model: "test".into(),
                    content: "go".into(),
                },
                StepKind::ToolCall {
                    call_id: "t1".into(),
                    name: "calc".into(),
                    input: serde_json::json!({"a": 1}),
                },
                msg("assistant", "done"),
            ],
        )
        .await;

        // bad has a Message at index 1 where good has a ToolCall — kinds
        // differ, so bisect should skip step 1, walk to step 2 (Message vs
        // Message), substitute, and recover.
        let bad = build_chain(
            &storage,
            vec![
                StepKind::Prompt {
                    model: "test".into(),
                    content: "go".into(),
                },
                msg("assistant", "I refuse to use a tool"),
                msg("assistant", "wrong final"),
            ],
        )
        .await;

        let build_agent = |_prefix: &[Step]| -> anyhow::Result<Box<dyn Agent>> {
            Ok(Box::new(ScriptedAgent::new(Vec::new(), |_| String::new())) as Box<dyn Agent>)
        };
        let check = |steps: &[Step]| {
            steps.last().is_some_and(|s| match &s.kind {
                StepKind::Message { role, content } => role == "assistant" && content == "done",
                _ => false,
            })
        };

        let result = bisect_chains(&storage, &good, &bad, build_agent, check)
            .await
            .unwrap();
        // Step 1 was skipped (ToolCall vs Message), step 2 substituted.
        assert_eq!(result.trials.len(), 1);
        assert_eq!(result.trials[0].bad_index, 2);
        assert!(result.trials[0].recovered);
    }

    #[tokio::test]
    async fn bisect_reports_no_recovery_when_no_substitution_helps() {
        let storage = MemoryStorage::new();

        let good = build_chain(
            &storage,
            vec![
                StepKind::Prompt {
                    model: "test".into(),
                    content: "go".into(),
                },
                msg("assistant", "good answer"),
            ],
        )
        .await;
        let bad = build_chain(
            &storage,
            vec![
                StepKind::Prompt {
                    model: "test".into(),
                    content: "go".into(),
                },
                msg("assistant", "bad answer"),
            ],
        )
        .await;

        // Agent does nothing; check requires impossible content.
        let build_agent = |_prefix: &[Step]| -> anyhow::Result<Box<dyn Agent>> {
            Ok(Box::new(ScriptedAgent::new(Vec::new(), |_| String::new())) as Box<dyn Agent>)
        };
        let check = |steps: &[Step]| {
            steps.last().is_some_and(|s| match &s.kind {
                StepKind::Message { role, content } => {
                    role == "assistant" && content.contains("THIS WILL NEVER MATCH")
                }
                _ => false,
            })
        };

        let result = bisect_chains(&storage, &good, &bad, build_agent, check)
            .await
            .unwrap();
        assert_eq!(result.first_recoverable, None);
    }

    #[tokio::test]
    async fn bisect_drives_agent_forward_when_fork_chain_is_short() {
        // good has 3 steps, bad diverges at step 1 and is also 2 steps.
        // After substituting good's step 1, the rewritten chain is only 2
        // steps; the agent must extend it to 3 to satisfy the check.
        let storage = MemoryStorage::new();

        let good = build_chain(
            &storage,
            vec![
                StepKind::Prompt {
                    model: "test".into(),
                    content: "compute then summarize".into(),
                },
                msg("assistant", "Let me compute."),
                msg("assistant", "result: 42"),
            ],
        )
        .await;
        let bad = build_chain(
            &storage,
            vec![
                StepKind::Prompt {
                    model: "test".into(),
                    content: "compute then summarize".into(),
                },
                msg("assistant", "I will not compute."),
            ],
        )
        .await;

        // Agent emits "result: 42" once, then stops.
        let build_agent = |_prefix: &[Step]| -> anyhow::Result<Box<dyn Agent>> {
            Ok(Box::new(ScriptedAgent::new(Vec::new(), |_| "result: 42".into())) as Box<dyn Agent>)
        };
        let check = |steps: &[Step]| {
            steps.last().is_some_and(|s| match &s.kind {
                StepKind::Message { role, content } => {
                    role == "assistant" && content.contains("42")
                }
                _ => false,
            })
        };

        let result = bisect_chains(&storage, &good, &bad, build_agent, check)
            .await
            .unwrap();
        assert!(result.first_recoverable.is_some());
        let trial = &result.trials[result.first_recoverable.unwrap()];
        assert_eq!(trial.bad_index, 1);
        assert_eq!(trial.final_text.as_deref(), Some("result: 42"));
    }

    #[test]
    fn parse_judge_verdict_handles_canonical_forms() {
        let (v, r) = parse_judge_verdict("EQUIVALENT\nthey say the same thing");
        assert_eq!(v, AuditVerdict::Equivalent);
        assert_eq!(r, "they say the same thing");

        let (v, _) = parse_judge_verdict("DIFFERENT\nthe candidate omits a key fact");
        assert_eq!(v, AuditVerdict::Different);

        // Lenient mid-line scan for less obedient judges.
        let (v, _) = parse_judge_verdict("After review, the answers are EQUIVALENT.");
        assert_eq!(v, AuditVerdict::Equivalent);

        // Falls through to inconclusive when neither verdict appears.
        let (v, _) = parse_judge_verdict("I'm not sure.");
        assert_eq!(v, AuditVerdict::Inconclusive);
    }

    #[test]
    fn extract_audit_pair_pulls_first_prompt_and_last_assistant() {
        let chain = vec![
            Step::new(
                None,
                StepKind::Prompt {
                    model: "m".into(),
                    content: "what is 2+3?".into(),
                },
                0,
            ),
            Step::new(None, msg("assistant", "I will compute."), 0),
            Step::new(None, msg("assistant", "5"), 0),
        ];
        let (p, a) = extract_audit_pair(&chain).unwrap();
        assert_eq!(p, "what is 2+3?");
        assert_eq!(a, "5");

        // Chains without an assistant message return None.
        let chain = vec![Step::new(
            None,
            StepKind::Prompt {
                model: "m".into(),
                content: "x".into(),
            },
            0,
        )];
        assert!(extract_audit_pair(&chain).is_none());
    }

    #[tokio::test]
    async fn audit_run_records_equivalent_when_judge_says_so() {
        let storage = MemoryStorage::new();
        // Recorded run: prompt + assistant answer.
        let chain = build_chain(
            &storage,
            vec![
                StepKind::Prompt {
                    model: "sonnet".into(),
                    content: "What's the capital of France?".into(),
                },
                msg("assistant", "Paris."),
            ],
        )
        .await;
        let head = chain.last().unwrap().id.clone();

        // Target (cheap model) gives an equivalent answer.
        let target_builder = |_prompt: &str| -> anyhow::Result<Box<dyn Agent>> {
            Ok(Box::new(ScriptedAgent::new(Vec::new(), |_| {
                "Paris is the capital.".into()
            })) as Box<dyn Agent>)
        };
        // Judge says EQUIVALENT.
        let judge_builder = |_prompt: &str| -> anyhow::Result<Box<dyn Agent>> {
            Ok(Box::new(ScriptedAgent::new(Vec::new(), |_| {
                "EQUIVALENT\nboth identify Paris".into()
            })) as Box<dyn Agent>)
        };

        let outcome = audit_run(
            &storage,
            &chain,
            head.clone(),
            "haiku",
            target_builder,
            judge_builder,
        )
        .await
        .unwrap()
        .unwrap();

        assert_eq!(outcome.verdict, AuditVerdict::Equivalent);
        assert_eq!(outcome.original_answer, "Paris.");
        assert_eq!(outcome.candidate_answer, "Paris is the capital.");
        assert!(outcome.judge_reason.contains("Paris"));

        // Candidate trial was persisted with audit tag.
        let runs = storage.list_runs().await.unwrap();
        assert!(runs.iter().any(|r| r.tag.as_deref() == Some("audit-haiku")));
    }

    #[tokio::test]
    async fn audit_runs_aggregates_mixed_verdicts() {
        let storage = MemoryStorage::new();

        let head_a = build_chain(
            &storage,
            vec![
                StepKind::Prompt {
                    model: "sonnet".into(),
                    content: "easy task".into(),
                },
                msg("assistant", "A1"),
            ],
        )
        .await
        .last()
        .unwrap()
        .id
        .clone();
        let head_b = build_chain(
            &storage,
            vec![
                StepKind::Prompt {
                    model: "sonnet".into(),
                    content: "hard task".into(),
                },
                msg("assistant", "B1"),
            ],
        )
        .await
        .last()
        .unwrap()
        .id
        .clone();

        let target_builder = |_prompt: &str| -> anyhow::Result<Box<dyn Agent>> {
            Ok(Box::new(ScriptedAgent::new(Vec::new(), |_| {
                "candidate answer".into()
            })) as Box<dyn Agent>)
        };
        // Judge: alternates between EQUIVALENT and DIFFERENT to cover both.
        let mut call = 0u8;
        let judge_builder = move |_prompt: &str| -> anyhow::Result<Box<dyn Agent>> {
            call += 1;
            let verdict = if call % 2 == 1 {
                "EQUIVALENT\nfine"
            } else {
                "DIFFERENT\nmissed a fact"
            };
            let resp = verdict.to_string();
            Ok(Box::new(ScriptedAgent::new(Vec::new(), move |_| resp.clone())) as Box<dyn Agent>)
        };

        let summary = audit_runs(
            &storage,
            &[head_a, head_b],
            "haiku",
            target_builder,
            judge_builder,
        )
        .await
        .unwrap();

        assert_eq!(summary.outcomes.len(), 2);
        assert_eq!(summary.equivalent_count(), 1);
        assert_eq!(summary.different_count(), 1);
        assert_eq!(summary.inconclusive_count(), 0);
    }
}
