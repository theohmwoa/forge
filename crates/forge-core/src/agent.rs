//! The agent abstraction.
//!
//! An `Agent` produces a sequence of `StepKind` values, each conceptually
//! emitted from a parent node in the run graph. The runtime is responsible for
//! hashing, persisting, and threading parent IDs through.

use std::collections::VecDeque;

use async_trait::async_trait;
use serde_json::json;

use crate::{NodeHash, StepKind};

#[async_trait]
pub trait Agent: Send {
    /// Produce the next step given the current parent node, or `None` when the
    /// agent is done. The parent is provided so future agents (real LLM ones)
    /// can read the conversation prefix from storage if they want to.
    async fn next_step(&mut self, parent: Option<NodeHash>) -> Option<StepKind>;
}

/// A demo agent that emits a fixed scripted sequence. Useful for tests and as
/// a smoke-test target for `forge run` until a real adapter exists.
pub struct FakeAgent {
    remaining: VecDeque<StepKind>,
}

impl FakeAgent {
    pub fn scripted() -> Self {
        let mut remaining = VecDeque::new();
        remaining.push_back(StepKind::Prompt {
            model: "fake-model-v1".into(),
            content: "Sum two numbers, 2 and 3.".into(),
        });
        remaining.push_back(StepKind::Message {
            role: "assistant".into(),
            content: "I'll use the calculator tool.".into(),
        });
        remaining.push_back(StepKind::ToolCall {
            call_id: "call-1".into(),
            name: "calculator".into(),
            input: json!({ "op": "add", "a": 2, "b": 3 }),
        });
        remaining.push_back(StepKind::ToolResult {
            call_id: "call-1".into(),
            output: json!(5),
        });
        remaining.push_back(StepKind::Message {
            role: "assistant".into(),
            content: "The sum is 5.".into(),
        });
        Self { remaining }
    }
}

#[async_trait]
impl Agent for FakeAgent {
    async fn next_step(&mut self, _parent: Option<NodeHash>) -> Option<StepKind> {
        self.remaining.pop_front()
    }
}

/// A matcher inspects an in-flight step before it is committed to the graph.
/// Returning a non-`Continue` decision lets the runtime fork: persist the
/// original under a `would_have_been` edge, then apply the decision.
///
/// Concrete matchers (regex, JSON-path, semantic, LLM-judge) live in their
/// own crates. This trait is intentionally minimal so it can sit upstream
/// of any of them.
#[async_trait]
pub trait Matcher: Send + Sync {
    async fn inspect(&self, step: &StepKind) -> MatcherDecision;
}

#[derive(Debug, Clone)]
pub enum MatcherDecision {
    /// Pass through unchanged.
    Continue,
    /// Replace this step's content with a rewritten variant.
    Rewrite(StepKind),
    /// Abort this step; the runtime should not commit it.
    Abort { reason: String },
    /// Pause the run and surface a handle for human review (Mirage-style).
    Pause { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fake_agent_emits_a_well_formed_conversation() {
        let mut agent = FakeAgent::scripted();
        let mut steps = Vec::new();
        while let Some(s) = agent.next_step(None).await {
            steps.push(s);
        }
        assert!(steps.len() >= 3, "should emit a multi-step conversation");
        assert!(matches!(steps.first(), Some(StepKind::Prompt { .. })));
        assert!(steps.iter().any(|s| matches!(s, StepKind::ToolCall { .. })));
        assert!(steps
            .iter()
            .any(|s| matches!(s, StepKind::ToolResult { .. })));
    }
}
