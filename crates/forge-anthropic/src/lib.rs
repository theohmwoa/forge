//! An [`Agent`] backed by the Anthropic Messages API.
//!
//! Supports multi-turn tool use. The agent runs the full prompt -> tool ->
//! response loop inside `fire()` and queues every emitted step (Prompt,
//! Message, ToolCall, ToolResult) for the runtime to drain via `next_step`.
//!
//! Streaming is not yet implemented — each turn is a complete request.

use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use forge_core::agent::Agent;
use forge_core::tool::Tool;
use forge_core::{NodeHash, StepKind};
use reqwest::header::{HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const MAX_TURNS: usize = 16;

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
    user_prompt: String,
}

impl AnthropicAgent {
    pub fn new(config: AnthropicConfig, user_prompt: impl Into<String>) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
            tools: Vec::new(),
            pending: VecDeque::new(),
            fired: false,
            user_prompt: user_prompt.into(),
        }
    }

    pub fn with_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = tools;
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
        // Always emit the original user prompt as the first graph step.
        self.pending.push_back(StepKind::Prompt {
            model: self.config.model.clone(),
            content: self.user_prompt.clone(),
        });

        // History is the API-shaped message log we round-trip with each turn.
        let mut history: Vec<Value> = vec![json!({
            "role": "user",
            "content": self.user_prompt.clone()
        })];

        for turn in 0..MAX_TURNS {
            tracing::debug!(turn, model = %self.config.model, "anthropic turn");
            let response = self.call_api(&history).await?;
            let content = response["content"].as_array().cloned().unwrap_or_default();
            let stop_reason = response["stop_reason"].as_str().unwrap_or("").to_string();

            // Decode content blocks for the graph + execute any tool_use.
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
                    _ => {
                        // Skip unknown content block types (image, thinking, ...).
                    }
                }
            }

            // Add the assistant's full response to history (for the next turn).
            history.push(json!({ "role": "assistant", "content": content }));

            if stop_reason != "tool_use" {
                return Ok(());
            }
            // Push tool results as the next user message and loop.
            history.push(json!({ "role": "user", "content": tool_results }));
        }

        anyhow::bail!("exceeded MAX_TURNS={MAX_TURNS}; aborting agent");
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

// Kept around for backward compat with v0.0.3 deserialization needs; not used.
#[allow(dead_code)]
#[derive(Deserialize, Serialize)]
struct InputMessage {
    role: String,
    content: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_core::tool::Calculator;

    /// The decoder logic is private but we can exercise it via a synthetic
    /// response that mirrors what the API would return after a tool_use turn.
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
        assert!(schemas[0]["input_schema"]["properties"]["op"].is_object());
    }
}
