//! Runtime helpers for `forge run`.
//!
//! Drives an `Agent` against a `Storage`, hashing each emitted step and
//! threading parent IDs to form a linear DAG (for now — branching comes with
//! `forge fork`).

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

/// Print a chain of steps in DAG-walk order. Compact, two-line-per-step.
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

fn label(kind: &StepKind) -> String {
    match kind {
        StepKind::Prompt { model, .. } => format!("prompt[{model}]"),
        StepKind::Message { role, .. } => format!("message[{role}]"),
        StepKind::ToolCall { name, .. } => format!("tool_call[{name}]"),
        StepKind::ToolResult { call_id, .. } => format!("tool_result[{call_id}]"),
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
