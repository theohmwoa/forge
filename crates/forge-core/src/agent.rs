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
        remaining.push_back(StepKind::ToolCall {
            call_id: "call-1".into(),
            name: "calculator".into(),
            input: json!({ "op": "add", "a": 2, "b": 3 }),
        });
        remaining.push_back(StepKind::ToolResult {
            call_id: "call-1".into(),
            output: json!(5),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fake_agent_emits_three_steps() {
        let mut agent = FakeAgent::scripted();
        let mut count = 0;
        while agent.next_step(None).await.is_some() {
            count += 1;
        }
        assert_eq!(count, 3);
    }
}
