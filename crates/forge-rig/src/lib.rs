//! Record a [Rig](https://rig.rs) agent's conversation into Forge.
//!
//! Rig already handles the API call; this adapter just translates the
//! resulting `rig::completion::Message` history into a Forge step chain so
//! you can replay, fork, and diff it like any other run.
//!
//! # Two ways to use Forge with Rig
//!
//! 1. **Recorder proxy (zero code).** Point Rig's HTTP client at
//!    `forge serve`'s proxy via `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL`. No
//!    integration code needed; works for any Rig agent.
//! 2. **This crate.** Call [`record_messages`] after your Rig agent finishes
//!    a conversation. Useful when you don't want to route traffic through
//!    a proxy or when you want to record locally constructed message logs.
//!
//! ```no_run
//! # use forge_rig::record_messages;
//! # use forge_storage::SledStorage;
//! # async fn ex(messages: Vec<rig::completion::Message>) -> anyhow::Result<()> {
//! let storage = SledStorage::open("./forge.db")?;
//! let head = record_messages(&messages, "claude-sonnet-4-6", &storage).await?;
//! println!("recorded run head: {head}");
//! # Ok(()) }
//! ```

use std::time::{SystemTime, UNIX_EPOCH};

use forge_core::{NodeHash, Step, StepKind};
use forge_storage::{RunMeta, Storage};
use rig::completion::Message;
use rig::message::{AssistantContent, UserContent};
use rig::OneOrMany;

/// Walk a Rig conversation and write it into the Forge graph as a chain of
/// content-addressed steps. Returns the head hash of the recorded run.
///
/// `model` is stamped on the synthesized [`StepKind::Prompt`] so `forge runs`
/// shows what was used. Pass the model name you configured Rig with.
pub async fn record_messages(
    messages: &[Message],
    model: &str,
    storage: &dyn Storage,
) -> anyhow::Result<NodeHash> {
    if messages.is_empty() {
        anyhow::bail!("record_messages: empty conversation; nothing to record");
    }

    let mut parent: Option<NodeHash> = None;
    let mut chain: Vec<NodeHash> = Vec::new();
    let mut first_was_user_text = false;
    let now = now_ms();

    for (i, msg) in messages.iter().enumerate() {
        match msg {
            Message::User { content } => {
                for block in iter(content) {
                    match block {
                        UserContent::Text(t) => {
                            // Treat the very first user-text block as the run's
                            // Prompt; subsequent user-text blocks are normal
                            // user messages (e.g. follow-ups, tool feedback).
                            let kind = if i == 0 && !first_was_user_text {
                                first_was_user_text = true;
                                StepKind::Prompt {
                                    model: model.to_string(),
                                    content: t.text.clone(),
                                }
                            } else {
                                StepKind::Message {
                                    role: "user".into(),
                                    content: t.text.clone(),
                                }
                            };
                            parent = Some(write_step(storage, parent, kind, now).await?);
                            chain.push(parent.clone().unwrap());
                        }
                        UserContent::ToolResult(tr) => {
                            let output = tool_result_payload(tr);
                            parent = Some(
                                write_step(
                                    storage,
                                    parent,
                                    StepKind::ToolResult {
                                        call_id: tr.id.clone(),
                                        output,
                                    },
                                    now,
                                )
                                .await?,
                            );
                            chain.push(parent.clone().unwrap());
                        }
                        // Image / Audio / Document — not modeled in Forge core
                        // yet. Skip silently rather than fail the whole record.
                        _ => {
                            tracing::debug!("forge-rig: skipping non-text user content");
                        }
                    }
                }
            }
            Message::System { content } => {
                parent = Some(
                    write_step(
                        storage,
                        parent,
                        StepKind::Message {
                            role: "system".into(),
                            content: content.clone(),
                        },
                        now,
                    )
                    .await?,
                );
                chain.push(parent.clone().unwrap());
            }
            Message::Assistant { content, .. } => {
                for block in iter(content) {
                    match block {
                        AssistantContent::Text(t) => {
                            parent = Some(
                                write_step(
                                    storage,
                                    parent,
                                    StepKind::Message {
                                        role: "assistant".into(),
                                        content: t.text.clone(),
                                    },
                                    now,
                                )
                                .await?,
                            );
                            chain.push(parent.clone().unwrap());
                        }
                        AssistantContent::ToolCall(tc) => {
                            parent = Some(
                                write_step(
                                    storage,
                                    parent,
                                    StepKind::ToolCall {
                                        call_id: tc.id.clone(),
                                        name: tc.function.name.clone(),
                                        input: tc.function.arguments.clone(),
                                    },
                                    now,
                                )
                                .await?,
                            );
                            chain.push(parent.clone().unwrap());
                        }
                        // Reasoning blocks (Anthropic extended thinking) are
                        // not part of the Forge step model yet — record them
                        // as plain assistant messages so the chain stays
                        // walkable.
                        _ => {
                            tracing::debug!(
                                "forge-rig: skipping non-text/tool-call assistant content"
                            );
                        }
                    }
                }
            }
        }
    }

    let head = chain
        .last()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("record_messages: produced no steps from conversation"))?;
    let root = chain[0].clone();
    storage
        .record_run(&RunMeta {
            head: head.clone(),
            root,
            recorded_at_ms: now,
            tag: None,
        })
        .await?;
    Ok(head)
}

async fn write_step(
    storage: &dyn Storage,
    parent: Option<NodeHash>,
    kind: StepKind,
    now: u64,
) -> anyhow::Result<NodeHash> {
    let step = Step::new(parent, kind, now);
    storage.put(step).await
}

/// Walk a `OneOrMany<T>` as a slice-like iterator. `OneOrMany::iter` requires
/// `T: Clone` (it materializes a backing Vec); the bound is satisfied by Rig's
/// content enums.
fn iter<T: Clone>(o: &OneOrMany<T>) -> impl Iterator<Item = &T> {
    o.iter()
}

/// Best-effort extraction of a tool result's content into a JSON `Value` for
/// `StepKind::ToolResult`. Rig models tool results as a sequence of typed
/// content blocks; we collapse them to a single string when they're all text,
/// otherwise emit a JSON array preserving the structure.
fn tool_result_payload(tr: &rig::message::ToolResult) -> serde_json::Value {
    let parts: Vec<serde_json::Value> = tr
        .content
        .iter()
        .map(|c| match c {
            rig::message::ToolResultContent::Text(t) => serde_json::Value::String(t.text.clone()),
            rig::message::ToolResultContent::Image(_) => serde_json::json!({"type": "image"}),
        })
        .collect();
    match parts.as_slice() {
        [serde_json::Value::String(s)] => serde_json::Value::String(s.clone()),
        _ => serde_json::Value::Array(parts),
    }
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
    use forge_storage::MemoryStorage;
    use rig::completion::Message;
    use rig::message::{ToolCall, ToolFunction};

    #[tokio::test]
    async fn round_trips_a_simple_conversation() {
        let storage = MemoryStorage::new();
        let messages = vec![
            Message::user("what is 2 + 3?"),
            Message::assistant("The answer is 5."),
        ];
        let head = record_messages(&messages, "test-model", &storage)
            .await
            .unwrap();
        let chain = storage.chain_to(&head).await.unwrap();
        assert_eq!(chain.len(), 2);
        match &chain[0].kind {
            StepKind::Prompt { content, model } => {
                assert_eq!(content, "what is 2 + 3?");
                assert_eq!(model, "test-model");
            }
            _ => panic!("first step should be Prompt"),
        }
        match &chain[1].kind {
            StepKind::Message { role, content } => {
                assert_eq!(role, "assistant");
                assert_eq!(content, "The answer is 5.");
            }
            _ => panic!("second step should be assistant Message"),
        }
    }

    #[tokio::test]
    async fn records_tool_call_and_result() {
        let storage = MemoryStorage::new();
        let tool_call = ToolCall {
            id: "call-1".into(),
            call_id: None,
            function: ToolFunction {
                name: "calculator".into(),
                arguments: serde_json::json!({"op": "add", "a": 2, "b": 3}),
            },
            additional_params: None,
            signature: None,
        };
        let messages = vec![
            Message::user("what is 2 + 3?"),
            Message::Assistant {
                id: None,
                content: OneOrMany::one(AssistantContent::ToolCall(tool_call)),
            },
            Message::User {
                content: OneOrMany::one(UserContent::tool_result(
                    "call-1",
                    OneOrMany::one(rig::message::ToolResultContent::text("5")),
                )),
            },
        ];
        let head = record_messages(&messages, "test-model", &storage)
            .await
            .unwrap();
        let chain = storage.chain_to(&head).await.unwrap();
        assert_eq!(chain.len(), 3);
        match &chain[1].kind {
            StepKind::ToolCall { name, call_id, .. } => {
                assert_eq!(name, "calculator");
                assert_eq!(call_id, "call-1");
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
        match &chain[2].kind {
            StepKind::ToolResult { call_id, output } => {
                assert_eq!(call_id, "call-1");
                assert_eq!(output, &serde_json::json!("5"));
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn empty_conversation_errors() {
        let storage = MemoryStorage::new();
        let err = record_messages(&[], "m", &storage).await.unwrap_err();
        assert!(err.to_string().contains("empty conversation"));
    }
}
