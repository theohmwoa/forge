//! An [`Agent`] backed by the OpenAI Chat Completions API.
//!
//! Mirrors `forge-anthropic`: `OpenAIAgent::new` for fresh runs,
//! `OpenAIAgent::continuing` for resuming from a Forge step prefix. The
//! multi-turn tool-call loop runs inside `fire()`. Streaming is not yet
//! implemented.

use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use forge_core::agent::Agent;
use forge_core::tool::Tool;
use forge_core::{NodeHash, Step, StepKind};
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::{json, Value};

const API_URL: &str = "https://api.openai.com/v1/chat/completions";
const DEFAULT_MAX_TURNS: usize = 16;

#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    pub api_key: String,
    pub model: String,
    pub max_tokens: u32,
}

impl OpenAIConfig {
    pub fn from_env(model: impl Into<String>) -> anyhow::Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY not set"))?;
        Ok(Self {
            api_key,
            model: model.into(),
            max_tokens: 1024,
        })
    }
}

pub struct OpenAIAgent {
    config: OpenAIConfig,
    client: reqwest::Client,
    tools: Vec<Arc<dyn Tool>>,
    pending: VecDeque<StepKind>,
    fired: bool,
    user_prompt: Option<String>,
    initial_history: Vec<Value>,
    max_turns: usize,
    stream: bool,
}

impl OpenAIAgent {
    pub fn new(config: OpenAIConfig, user_prompt: impl Into<String>) -> Self {
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
            stream: false,
        }
    }

    pub fn continuing(config: OpenAIConfig, prefix: &[Step]) -> Self {
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
            stream: false,
        }
    }

    pub fn with_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_max_turns(mut self, n: usize) -> Self {
        self.max_turns = n.max(1);
        self
    }

    /// Enable SSE streaming. Text deltas are printed to stderr in real time
    /// during a turn; the graph still gets atomic Step entries.
    pub fn with_streaming(mut self, on: bool) -> Self {
        self.stream = on;
        self
    }

    fn tool_schemas(&self) -> Vec<Value> {
        self.tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name(),
                        "description": t.description(),
                        "parameters": t.schema()
                    }
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
            tracing::debug!(turn, model = %self.config.model, "openai turn");
            let response = if self.stream {
                self.call_api_stream(&history).await?
            } else {
                self.call_api(&history).await?
            };
            let choice = response["choices"][0].clone();
            let message = choice["message"].clone();
            let finish_reason = choice["finish_reason"].as_str().unwrap_or("").to_string();

            if let Some(text) = message["content"].as_str() {
                if !text.is_empty() {
                    self.pending.push_back(StepKind::Message {
                        role: "assistant".into(),
                        content: text.to_string(),
                    });
                }
            }

            // Push the assistant turn as-is into history for the next call.
            history.push(message.clone());

            let tool_calls = message["tool_calls"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            if tool_calls.is_empty() {
                return Ok(());
            }

            for tc in &tool_calls {
                let id = tc["id"].as_str().unwrap_or("").to_string();
                let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                // OpenAI returns arguments as a JSON STRING; parse before storing.
                let args_str = tc["function"]["arguments"].as_str().unwrap_or("");
                let input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));

                self.pending.push_back(StepKind::ToolCall {
                    call_id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                });

                let output = match self.tools.iter().find(|t| t.name() == name) {
                    Some(tool) => match tool.run(&input).await {
                        Ok(v) => v,
                        Err(e) => json!(e.to_string()),
                    },
                    None => json!(format!("unknown tool: {name}")),
                };
                self.pending.push_back(StepKind::ToolResult {
                    call_id: id.clone(),
                    output: output.clone(),
                });

                history.push(json!({
                    "role": "tool",
                    "tool_call_id": id,
                    "content": output.to_string()
                }));
            }

            if finish_reason != "tool_calls" {
                return Ok(());
            }
        }

        tracing::info!(
            max_turns = self.max_turns,
            "openai agent stopped at turn cap"
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
            "authorization",
            HeaderValue::from_str(&format!("Bearer {}", self.config.api_key))
                .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY contained invalid characters"))?,
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
            anyhow::bail!("openai api error ({status}): {body}");
        }
        Ok(body)
    }

    /// SSE streaming variant. Reduces the streamed `chat.completion.chunk`
    /// events back to the same `{choices: [...]}` shape as a non-streaming
    /// response so the caller can stay shape-agnostic. Text deltas are
    /// printed to stderr in real time.
    async fn call_api_stream(&self, history: &[Value]) -> anyhow::Result<Value> {
        use futures_util::StreamExt;
        use std::io::Write;

        let mut req = json!({
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": history,
            "stream": true,
        });
        let schemas = self.tool_schemas();
        if !schemas.is_empty() {
            req["tools"] = json!(schemas);
        }

        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer {}", self.config.api_key))
                .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY contained invalid characters"))?,
        );
        headers.insert("content-type", HeaderValue::from_static("application/json"));
        headers.insert("accept", HeaderValue::from_static("text/event-stream"));

        let resp = self
            .client
            .post(API_URL)
            .headers(headers)
            .json(&req)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body: Value = resp.json().await.unwrap_or_else(|_| json!({}));
            anyhow::bail!("openai api error ({status}): {body}");
        }

        // Accumulator for the assembled message.
        let mut content = String::new();
        let mut finish_reason = String::new();
        // tool_calls indexed by their `index` field, since deltas can fragment
        // the arguments string and only the first delta carries id/name.
        let mut tool_calls: Vec<Value> = Vec::new();

        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(idx) = buffer.find("\n\n") {
                let event_block: String = buffer.drain(..idx + 2).collect();
                let data: String = event_block
                    .lines()
                    .filter(|l| l.starts_with("data:"))
                    .map(|l| l.trim_start_matches("data:").trim())
                    .collect::<Vec<_>>()
                    .join("");
                if data.is_empty() || data == "[DONE]" {
                    continue;
                }
                let event: Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let delta = &event["choices"][0]["delta"];
                if let Some(text) = delta["content"].as_str() {
                    content.push_str(text);
                    eprint!("{text}");
                    let _ = std::io::stderr().flush();
                }
                if let Some(deltas) = delta["tool_calls"].as_array() {
                    for d in deltas {
                        let idx = d["index"].as_u64().unwrap_or(0) as usize;
                        while tool_calls.len() <= idx {
                            tool_calls.push(json!({
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            }));
                        }
                        if let Some(id) = d["id"].as_str() {
                            tool_calls[idx]["id"] = json!(id);
                        }
                        if let Some(name) = d["function"]["name"].as_str() {
                            tool_calls[idx]["function"]["name"] = json!(name);
                        }
                        if let Some(args) = d["function"]["arguments"].as_str() {
                            let existing = tool_calls[idx]["function"]["arguments"]
                                .as_str()
                                .unwrap_or("")
                                .to_string();
                            tool_calls[idx]["function"]["arguments"] =
                                json!(format!("{existing}{args}"));
                        }
                    }
                }
                if let Some(fr) = event["choices"][0]["finish_reason"].as_str() {
                    finish_reason = fr.to_string();
                }
            }
        }
        let _ = writeln!(std::io::stderr());
        let _ = std::io::stderr().flush();

        let mut message = json!({ "role": "assistant", "content": content });
        if !tool_calls.is_empty() {
            message["tool_calls"] = json!(tool_calls);
            message["content"] = Value::Null;
        }
        Ok(json!({
            "choices": [{
                "message": message,
                "finish_reason": finish_reason,
            }]
        }))
    }
}

#[async_trait]
impl Agent for OpenAIAgent {
    async fn next_step(&mut self, _parent: Option<NodeHash>) -> Option<StepKind> {
        if !self.fired {
            self.fired = true;
            if let Err(err) = self.fire().await {
                tracing::error!(?err, "openai agent failed; emitting nothing");
                return None;
            }
        }
        self.pending.pop_front()
    }
}

/// Translate a Forge step prefix into the OpenAI Chat Completions message
/// shape. Tool calls live inside the assistant message; tool results are
/// separate `role: "tool"` messages keyed by `tool_call_id`.
pub fn prefix_to_history(prefix: &[Step]) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    // Buffer state for building the current assistant message.
    let mut pending_assistant: Option<Value> = None;

    fn flush_assistant(out: &mut Vec<Value>, pending: &mut Option<Value>) {
        if let Some(v) = pending.take() {
            out.push(v);
        }
    }

    for step in prefix {
        match &step.kind {
            StepKind::Prompt { content, .. } => {
                flush_assistant(&mut out, &mut pending_assistant);
                out.push(json!({ "role": "user", "content": content }));
            }
            StepKind::Message { role, content } if role == "assistant" => {
                flush_assistant(&mut out, &mut pending_assistant);
                pending_assistant = Some(json!({
                    "role": "assistant",
                    "content": content
                }));
            }
            StepKind::Message { role, content } => {
                flush_assistant(&mut out, &mut pending_assistant);
                out.push(json!({ "role": role, "content": content }));
            }
            StepKind::ToolCall {
                call_id,
                name,
                input,
            } => {
                let tc = json!({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": input.to_string()
                    }
                });
                match pending_assistant.as_mut() {
                    Some(msg) => {
                        let arr = msg["tool_calls"]
                            .as_array_mut()
                            .map(std::mem::take)
                            .unwrap_or_default();
                        let mut arr = arr;
                        arr.push(tc);
                        msg["tool_calls"] = Value::Array(arr);
                    }
                    None => {
                        pending_assistant = Some(json!({
                            "role": "assistant",
                            "content": Value::Null,
                            "tool_calls": [tc]
                        }));
                    }
                }
            }
            StepKind::ToolResult { call_id, output } => {
                flush_assistant(&mut out, &mut pending_assistant);
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output.to_string()
                }));
            }
        }
    }
    flush_assistant(&mut out, &mut pending_assistant);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_core::tool::Calculator;

    fn step(parent: Option<NodeHash>, kind: StepKind) -> Step {
        Step::new(parent, kind, 0)
    }

    #[test]
    fn tool_schemas_use_function_envelope() {
        let agent = OpenAIAgent::new(
            OpenAIConfig {
                api_key: "x".into(),
                model: "gpt-5".into(),
                max_tokens: 1024,
            },
            "what is 2 + 3?",
        )
        .with_tools(vec![Arc::new(Calculator)]);
        let schemas = agent.tool_schemas();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0]["type"], "function");
        assert_eq!(schemas[0]["function"]["name"], "calculator");
    }

    #[test]
    fn prefix_packs_tool_calls_into_assistant_message() {
        let s0 = step(
            None,
            StepKind::Prompt {
                model: "m".into(),
                content: "what is 2 + 3?".into(),
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
                call_id: "call_abc".into(),
                name: "calculator".into(),
                input: json!({"op": "add", "a": 2, "b": 3}),
            },
        );
        let s3 = step(
            Some(s2.id.clone()),
            StepKind::ToolResult {
                call_id: "call_abc".into(),
                output: json!(5),
            },
        );

        let h = prefix_to_history(&[s0, s1, s2, s3]);
        // user, assistant(content+tool_calls), tool
        assert_eq!(h.len(), 3);
        assert_eq!(h[0]["role"], "user");
        assert_eq!(h[1]["role"], "assistant");
        assert_eq!(h[1]["content"], "I'll use a tool.");
        assert_eq!(h[1]["tool_calls"][0]["id"], "call_abc");
        assert_eq!(h[1]["tool_calls"][0]["function"]["name"], "calculator");
        assert_eq!(h[2]["role"], "tool");
        assert_eq!(h[2]["tool_call_id"], "call_abc");
    }

    #[test]
    fn prefix_handles_tool_call_without_text() {
        let s0 = step(
            None,
            StepKind::Prompt {
                model: "m".into(),
                content: "go".into(),
            },
        );
        let s1 = step(
            Some(s0.id.clone()),
            StepKind::ToolCall {
                call_id: "x".into(),
                name: "t".into(),
                input: json!({}),
            },
        );
        let h = prefix_to_history(&[s0, s1]);
        assert_eq!(h.len(), 2);
        assert_eq!(h[1]["role"], "assistant");
        assert!(h[1]["content"].is_null());
        assert_eq!(h[1]["tool_calls"][0]["id"], "x");
    }
}
