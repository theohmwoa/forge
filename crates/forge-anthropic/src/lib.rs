//! An [`Agent`] backed by the Anthropic Messages API.
//!
//! Two construction paths:
//! - [`AnthropicAgent::new`] — fresh run from a user prompt.
//! - [`AnthropicAgent::continuing`] — resume from a Forge step prefix
//!   (translated into the Messages API conversation shape).
//!
//! Both modes drive the multi-turn tool-use loop in `fire()`. Streaming is
//! not yet implemented — each turn is a complete request.

use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use forge_core::agent::Agent;
use forge_core::tool::Tool;
use forge_core::{NodeHash, Step, StepKind};
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::{json, Value};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TURNS: usize = 16;

#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    pub api_key: String,
    pub model: String,
    pub max_tokens: u32,
}

impl AnthropicConfig {
    pub fn from_env(model: impl Into<String>) -> anyhow::Result<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY not set"))?;
        Ok(Self {
            api_key,
            model: model.into(),
            max_tokens: 1024,
        })
    }
}

pub struct AnthropicAgent {
    config: AnthropicConfig,
    client: reqwest::Client,
    tools: Vec<Arc<dyn Tool>>,
    pending: VecDeque<StepKind>,
    fired: bool,
    /// `Some` for fresh runs — emitted as a Prompt step before the first turn.
    /// `None` for continuations: the prefix already contains the prompt.
    user_prompt: Option<String>,
    initial_history: Vec<Value>,
    /// Cap on inner-loop turns. Useful for handoffs: run model A for 1 turn,
    /// then model B continues.
    max_turns: usize,
}

impl AnthropicAgent {
    pub fn new(config: AnthropicConfig, user_prompt: impl Into<String>) -> Self {
        let user_prompt = user_prompt.into();
        let initial_history = vec![json!({
            "role": "user",
            "content": user_prompt.clone(),
        })];
        Self {
            config,
            client: reqwest::Client::new(),
            tools: Vec::new(),
            pending: VecDeque::new(),
            fired: false,
            user_prompt: Some(user_prompt),
            initial_history,
            max_turns: DEFAULT_MAX_TURNS,
        }
    }

    /// Continue a previously-recorded conversation. The prefix is translated
    /// into the Anthropic Messages API shape; no new Prompt step is emitted.
    pub fn continuing(config: AnthropicConfig, prefix: &[Step]) -> Self {
        let initial_history = prefix_to_history(prefix);
        Self {
            config,
            client: reqwest::Client::new(),
            tools: Vec::new(),
            pending: VecDeque::new(),
            fired: false,
            user_prompt: None,
            initial_history,
            max_turns: DEFAULT_MAX_TURNS,
        }
    }

    pub fn with_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = tools;
        self
    }

    /// Cap the number of API turns this agent will run. After the cap, the
    /// agent stops cleanly even if `stop_reason == "tool_use"`. Used by the
    /// CLI to hand off between models mid-run.
    pub fn with_max_turns(mut self, n: usize) -> Self {
        self.max_turns = n.max(1);
        self
    }

    fn tool_schemas(&self) -> Vec<Value> {
        self.tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name(),
                    "description": t.description(),
                    "input_schema": t.schema()
                })
            })
            .collect()
    }

    async fn fire(&mut self) -> anyhow::Result<()> {
        if let Some(prompt) = &self.user_prompt {
            self.pending.push_back(StepKind::Prompt {
                model: self.config.model.clone(),
                content: prompt.clone(),
            });
        }

        let mut history = self.initial_history.clone();

        for turn in 0..self.max_turns {
            tracing::debug!(turn, model = %self.config.model, "anthropic turn");
            let response = self.call_api(&history).await?;
            let content = response["content"].as_array().cloned().unwrap_or_default();
            let stop_reason = response["stop_reason"].as_str().unwrap_or("").to_string();

            let mut tool_results: Vec<Value> = Vec::new();
            for block in &content {
                match block["type"].as_str() {
                    Some("text") => {
                        let text = block["text"].as_str().unwrap_or("").to_string();
                        self.pending.push_back(StepKind::Message {
                            role: "assistant".into(),
                            content: text,
                        });
                    }
                    Some("tool_use") => {
                        let id = block["id"].as_str().unwrap_or("").to_string();
                        let name = block["name"].as_str().unwrap_or("").to_string();
                        let input = block["input"].clone();
                        self.pending.push_back(StepKind::ToolCall {
                            call_id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                        });

                        let (output, is_error) = match self.tools.iter().find(|t| t.name() == name)
                        {
                            Some(tool) => match tool.run(&input).await {
                                Ok(v) => (v, false),
                                Err(e) => (json!(e.to_string()), true),
                            },
                            None => (json!(format!("unknown tool: {name}")), true),
                        };

                        self.pending.push_back(StepKind::ToolResult {
                            call_id: id.clone(),
                            output: output.clone(),
                        });
                        tool_results.push(json!({
                            "type": "tool_result",
                            "tool_use_id": id,
                            "content": output.to_string(),
                            "is_error": is_error
                        }));
                    }
                    _ => {}
                }
            }

            history.push(json!({ "role": "assistant", "content": content }));

            if stop_reason != "tool_use" {
                return Ok(());
            }
            history.push(json!({ "role": "user", "content": tool_results }));
        }

        // Reached the cap; not an error — the runtime may want to swap models
        // and call `continuing` to keep going.
        tracing::info!(
            max_turns = self.max_turns,
            "anthropic agent stopped at turn cap"
        );
        Ok(())
    }

    async fn call_api(&self, history: &[Value]) -> anyhow::Result<Value> {
        let mut req = json!({
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": history,
        });
        let schemas = self.tool_schemas();
        if !schemas.is_empty() {
            req["tools"] = json!(schemas);
        }

        let mut headers = HeaderMap::new();
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.config.api_key)
                .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY contained invalid characters"))?,
        );
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );
        headers.insert("content-type", HeaderValue::from_static("application/json"));

        let resp = self
            .client
            .post(API_URL)
            .headers(headers)
            .json(&req)
            .send()
            .await?;
        let status = resp.status();
        let body: Value = resp.json().await?;
        if !status.is_success() {
            anyhow::bail!("anthropic api error ({status}): {body}");
        }
        Ok(body)
    }
}

#[async_trait]
impl Agent for AnthropicAgent {
    async fn next_step(&mut self, _parent: Option<NodeHash>) -> Option<StepKind> {
        if !self.fired {
            self.fired = true;
            if let Err(err) = self.fire().await {
                tracing::error!(?err, "anthropic agent failed; emitting nothing");
                return None;
            }
        }
        self.pending.pop_front()
    }
}

/// Translate a Forge step prefix into the Anthropic Messages API conversation
/// shape. Consecutive same-role steps are coalesced into one message with a
/// `content` array of typed blocks.
pub fn prefix_to_history(prefix: &[Step]) -> Vec<Value> {
    let mut history: Vec<Value> = Vec::new();
    let mut current_role: Option<String> = None;
    let mut current_blocks: Vec<Value> = Vec::new();

    fn flush(history: &mut Vec<Value>, role: &mut Option<String>, blocks: &mut Vec<Value>) {
        if let Some(r) = role.take() {
            if !blocks.is_empty() {
                history.push(json!({ "role": r, "content": std::mem::take(blocks) }));
            }
        }
    }

    for step in prefix {
        let (role, block) = match &step.kind {
            StepKind::Prompt { content, .. } => (
                "user".to_string(),
                json!({ "type": "text", "text": content }),
            ),
            StepKind::Message { role, content } => {
                (role.clone(), json!({ "type": "text", "text": content }))
            }
            StepKind::ToolCall {
                call_id,
                name,
                input,
            } => (
                "assistant".to_string(),
                json!({
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": input
                }),
            ),
            StepKind::ToolResult { call_id, output } => (
                "user".to_string(),
                json!({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": output.to_string()
                }),
            ),
        };

        if current_role.as_deref() != Some(role.as_str()) {
            flush(&mut history, &mut current_role, &mut current_blocks);
            current_role = Some(role);
        }
        current_blocks.push(block);
    }
    flush(&mut history, &mut current_role, &mut current_blocks);
    history
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_core::tool::Calculator;

    fn step(parent: Option<NodeHash>, kind: StepKind) -> Step {
        Step::new(parent, kind, 0)
    }

    #[test]
    fn tool_schemas_serialize() {
        let agent = AnthropicAgent::new(
            AnthropicConfig {
                api_key: "x".into(),
                model: "claude-sonnet-4-6".into(),
                max_tokens: 1024,
            },
            "what is 2 + 3?",
        )
        .with_tools(vec![Arc::new(Calculator)]);
        let schemas = agent.tool_schemas();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0]["name"], "calculator");
    }

    #[test]
    fn prefix_with_only_a_prompt() {
        let s = step(
            None,
            StepKind::Prompt {
                model: "m".into(),
                content: "hello".into(),
            },
        );
        let h = prefix_to_history(&[s]);
        assert_eq!(h.len(), 1);
        assert_eq!(h[0]["role"], "user");
        assert_eq!(h[0]["content"][0]["type"], "text");
        assert_eq!(h[0]["content"][0]["text"], "hello");
    }

    #[test]
    fn prefix_coalesces_consecutive_same_role() {
        let s0 = step(
            None,
            StepKind::Prompt {
                model: "m".into(),
                content: "p".into(),
            },
        );
        let s1 = step(
            Some(s0.id.clone()),
            StepKind::Message {
                role: "assistant".into(),
                content: "I'll use a tool.".into(),
            },
        );
        let s2 = step(
            Some(s1.id.clone()),
            StepKind::ToolCall {
                call_id: "t1".into(),
                name: "calculator".into(),
                input: json!({"op": "add", "a": 1, "b": 2}),
            },
        );
        let s3 = step(
            Some(s2.id.clone()),
            StepKind::ToolResult {
                call_id: "t1".into(),
                output: json!(3),
            },
        );

        let h = prefix_to_history(&[s0, s1, s2, s3]);
        // user(prompt) -> assistant(text + tool_use) -> user(tool_result)
        assert_eq!(h.len(), 3);
        assert_eq!(h[0]["role"], "user");
        assert_eq!(h[1]["role"], "assistant");
        assert_eq!(h[1]["content"].as_array().unwrap().len(), 2);
        assert_eq!(h[1]["content"][0]["type"], "text");
        assert_eq!(h[1]["content"][1]["type"], "tool_use");
        assert_eq!(h[2]["role"], "user");
        assert_eq!(h[2]["content"][0]["type"], "tool_result");
    }
}
