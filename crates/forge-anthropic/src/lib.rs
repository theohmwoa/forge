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
    /// Optional system prompt. When set, sent with `cache_control: ephemeral`
    /// so subsequent calls hit the cache instead of re-paying the input tokens.
    pub system: Option<String>,
}

impl AnthropicConfig {
    pub fn from_env(model: impl Into<String>) -> anyhow::Result<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY not set"))?;
        Ok(Self {
            api_key,
            model: model.into(),
            max_tokens: 1024,
            system: None,
        })
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
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
    /// When true, use the SSE streaming endpoint and print text deltas to
    /// stderr as they arrive. Steps are still emitted as atomic units when
    /// each content block completes.
    stream: bool,
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
            stream: false,
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
            stream: false,
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
            let response = if self.stream {
                self.call_api_stream(&history).await?
            } else {
                self.call_api(&history).await?
            };
            let content = response["content"].as_array().cloned().unwrap_or_default();
            let stop_reason = response["stop_reason"].as_str().unwrap_or("").to_string();
            log_usage(turn, &response["usage"]);

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
        apply_prompt_caching(&mut req, self.config.system.as_deref());
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

    /// SSE streaming variant of `call_api`. Prints text deltas to stderr as
    /// they arrive and reconstructs an equivalent of the non-streaming
    /// response (`{ content, stop_reason }`) for the caller.
    async fn call_api_stream(&self, history: &[Value]) -> anyhow::Result<Value> {
        use futures_util::StreamExt;
        use std::io::Write;

        let mut req = json!({
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": history,
            "stream": true,
        });
        apply_prompt_caching(&mut req, self.config.system.as_deref());
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
            anyhow::bail!("anthropic api error ({status}): {body}");
        }

        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();
        let mut blocks: Vec<Value> = Vec::new();
        let mut partial_json: Vec<String> = Vec::new();
        let mut stop_reason = String::new();
        let mut usage: Value = json!({});

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // SSE events are separated by a blank line.
            while let Some(idx) = buffer.find("\n\n") {
                let event_block: String = buffer.drain(..idx + 2).collect();
                let data_payload = event_block
                    .lines()
                    .filter(|l| l.starts_with("data:"))
                    .map(|l| l.trim_start_matches("data:").trim())
                    .collect::<Vec<_>>()
                    .join("");
                if data_payload.is_empty() {
                    continue;
                }
                let event: Value = match serde_json::from_str(&data_payload) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                merge_usage_from_event(&mut usage, &event);
                if let Some(delta) = handle_anthropic_sse_event(
                    &event,
                    &mut blocks,
                    &mut partial_json,
                    &mut stop_reason,
                ) {
                    eprint!("{delta}");
                    let _ = std::io::stderr().flush();
                }
            }
        }
        // Final newline so the next CLI line starts cleanly.
        let _ = writeln!(std::io::stderr());
        let _ = std::io::stderr().flush();

        // Finalize any tool_use blocks whose JSON didn't get a content_block_stop
        // before the connection ended (defensive — Anthropic always sends it).
        for (i, raw) in partial_json.iter().enumerate() {
            if i < blocks.len()
                && blocks[i]["type"] == "tool_use"
                && blocks[i]["input"].is_object()
                && blocks[i]["input"].as_object().is_some_and(|o| o.is_empty())
            {
                if let Ok(parsed) = serde_json::from_str::<Value>(raw) {
                    blocks[i]["input"] = parsed;
                }
            }
        }

        Ok(json!({
            "content": blocks,
            "stop_reason": stop_reason,
            "usage": usage,
        }))
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

/// Log the per-turn token usage from an Anthropic response. Surfaces the
/// `cache_creation_input_tokens` / `cache_read_input_tokens` fields so users
/// can confirm prompt caching is actually saving them money. Skipped silently
/// when the response shape doesn't include usage (e.g. an error path).
fn log_usage(turn: usize, usage: &Value) {
    if !usage.is_object() {
        return;
    }
    let input = usage["input_tokens"].as_u64().unwrap_or(0);
    let output = usage["output_tokens"].as_u64().unwrap_or(0);
    let cache_create = usage["cache_creation_input_tokens"].as_u64().unwrap_or(0);
    let cache_read = usage["cache_read_input_tokens"].as_u64().unwrap_or(0);
    if input == 0 && output == 0 && cache_create == 0 && cache_read == 0 {
        return;
    }
    tracing::info!(
        turn,
        input_tokens = input,
        output_tokens = output,
        cache_creation_input_tokens = cache_create,
        cache_read_input_tokens = cache_read,
        "anthropic usage"
    );
}

/// Merge the `usage` payload from an SSE event into a running accumulator.
/// `message_start.message.usage` carries the prompt-side counts (including
/// cache reads/creates); `message_delta.usage` updates the output token
/// count as the response streams in.
fn merge_usage_from_event(acc: &mut Value, event: &Value) {
    let etype = event["type"].as_str().unwrap_or("");
    let src = match etype {
        "message_start" => &event["message"]["usage"],
        "message_delta" => &event["usage"],
        _ => return,
    };
    let Some(obj) = src.as_object() else {
        return;
    };
    if !acc.is_object() {
        *acc = json!({});
    }
    let acc_obj = acc.as_object_mut().expect("just-created object");
    for (k, v) in obj {
        acc_obj.insert(k.clone(), v.clone());
    }
}

/// Mark the most recent input block with `cache_control: ephemeral` so the
/// Anthropic prompt cache can match this prefix on subsequent calls. The win
/// is biggest on multi-turn tool use loops (every turn after the first reads
/// the cached prefix instead of re-paying input tokens) and on continuations
/// (the prefix is already known to repeat).
///
/// Also injects the system prompt with cache_control when one is configured.
fn apply_prompt_caching(req: &mut Value, system: Option<&str>) {
    if let Some(sys) = system {
        req["system"] = json!([{
            "type": "text",
            "text": sys,
            "cache_control": {"type": "ephemeral"}
        }]);
    }

    let Some(messages) = req["messages"].as_array_mut() else {
        return;
    };
    let Some(last_msg) = messages.last_mut() else {
        return;
    };
    let content = &mut last_msg["content"];
    // The Messages API accepts either a plain string or an array of typed
    // content blocks. Normalize both into an array we can tag.
    if let Some(s) = content.as_str() {
        let s = s.to_string();
        last_msg["content"] = json!([{
            "type": "text",
            "text": s,
            "cache_control": {"type": "ephemeral"}
        }]);
    } else if let Some(arr) = content.as_array_mut() {
        if let Some(last) = arr.last_mut() {
            if let Some(obj) = last.as_object_mut() {
                obj.insert("cache_control".into(), json!({"type": "ephemeral"}));
            }
        }
    }
}

/// Parse a complete Anthropic SSE response body (concatenated event stream)
/// into the same `{ content, stop_reason }` shape as a non-streaming response.
/// Useful for the recorder, which tees the upstream SSE stream to its client
/// and then has the full body to convert into Forge steps.
pub fn parse_anthropic_sse_body(body: &str) -> serde_json::Value {
    let mut blocks: Vec<Value> = Vec::new();
    let mut partial_json: Vec<String> = Vec::new();
    let mut stop_reason = String::new();
    let mut usage: Value = json!({});

    for chunk in body.split("\n\n") {
        let data: String = chunk
            .lines()
            .filter(|l| l.starts_with("data:"))
            .map(|l| l.trim_start_matches("data:").trim())
            .collect::<Vec<_>>()
            .join("");
        if data.is_empty() {
            continue;
        }
        let event: Value = match serde_json::from_str(&data) {
            Ok(v) => v,
            Err(_) => continue,
        };
        merge_usage_from_event(&mut usage, &event);
        handle_anthropic_sse_event(&event, &mut blocks, &mut partial_json, &mut stop_reason);
    }
    json!({ "content": blocks, "stop_reason": stop_reason, "usage": usage })
}

/// Apply one parsed SSE event to the in-progress block buffer.
/// Returns the text delta string when the event is a `text_delta`, so callers
/// that want to stream output (e.g. the agent printing to stderr) can do so;
/// callers that don't (e.g. the recorder) just ignore the return value.
fn handle_anthropic_sse_event(
    event: &Value,
    blocks: &mut Vec<Value>,
    partial_json: &mut Vec<String>,
    stop_reason: &mut String,
) -> Option<String> {
    let etype = event["type"].as_str().unwrap_or("");
    match etype {
        "content_block_start" => {
            let idx = event["index"].as_u64().unwrap_or(0) as usize;
            while blocks.len() <= idx {
                blocks.push(json!({}));
                partial_json.push(String::new());
            }
            let mut block = event["content_block"].clone();
            if block["type"] == "text" && !block["text"].is_string() {
                block["text"] = json!("");
            }
            if block["type"] == "tool_use" {
                block["input"] = json!({});
                partial_json[idx].clear();
            }
            blocks[idx] = block;
            None
        }
        "content_block_delta" => {
            let idx = event["index"].as_u64().unwrap_or(0) as usize;
            if idx >= blocks.len() {
                return None;
            }
            let delta = &event["delta"];
            match delta["type"].as_str() {
                Some("text_delta") => {
                    let text = delta["text"].as_str().unwrap_or("").to_string();
                    let existing = blocks[idx]["text"].as_str().unwrap_or("").to_string();
                    blocks[idx]["text"] = json!(format!("{existing}{text}"));
                    Some(text)
                }
                Some("input_json_delta") => {
                    let partial = delta["partial_json"].as_str().unwrap_or("");
                    partial_json[idx].push_str(partial);
                    None
                }
                _ => None,
            }
        }
        "content_block_stop" => {
            let idx = event["index"].as_u64().unwrap_or(0) as usize;
            if idx < blocks.len() && blocks[idx]["type"] == "tool_use" {
                if let Ok(parsed) = serde_json::from_str::<Value>(&partial_json[idx]) {
                    blocks[idx]["input"] = parsed;
                }
            }
            None
        }
        "message_delta" => {
            if let Some(sr) = event["delta"]["stop_reason"].as_str() {
                *stop_reason = sr.to_string();
            }
            None
        }
        _ => None,
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
                system: None,
            },
            "what is 2 + 3?",
        )
        .with_tools(vec![Arc::new(Calculator)]);
        let schemas = agent.tool_schemas();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0]["name"], "calculator");
    }

    #[test]
    fn cache_control_tags_last_block_in_string_content() {
        let mut req = json!({
            "messages": [{"role": "user", "content": "hello"}]
        });
        apply_prompt_caching(&mut req, None);
        let last = &req["messages"][0]["content"][0];
        assert_eq!(last["type"], "text");
        assert_eq!(last["text"], "hello");
        assert_eq!(last["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn cache_control_tags_last_block_in_array_content() {
        let mut req = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "5"},
                    {"type": "text", "text": "next"}
                ]
            }]
        });
        apply_prompt_caching(&mut req, None);
        let blocks = req["messages"][0]["content"].as_array().unwrap();
        // First block left alone; last one is tagged.
        assert!(blocks[0].get("cache_control").is_none());
        assert_eq!(blocks[1]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn merge_usage_picks_up_message_start_and_message_delta() {
        let mut acc = json!({});
        merge_usage_from_event(
            &mut acc,
            &json!({
                "type": "message_start",
                "message": {"usage": {
                    "input_tokens": 12,
                    "cache_creation_input_tokens": 100,
                    "cache_read_input_tokens": 50,
                    "output_tokens": 1
                }}
            }),
        );
        merge_usage_from_event(
            &mut acc,
            &json!({
                "type": "message_delta",
                "usage": {"output_tokens": 42}
            }),
        );
        assert_eq!(acc["input_tokens"], 12);
        assert_eq!(acc["cache_creation_input_tokens"], 100);
        assert_eq!(acc["cache_read_input_tokens"], 50);
        // message_delta replaces the streaming output count.
        assert_eq!(acc["output_tokens"], 42);
    }

    #[test]
    fn cache_control_attaches_system_prompt() {
        let mut req = json!({
            "messages": [{"role": "user", "content": "hi"}]
        });
        apply_prompt_caching(&mut req, Some("you are a helpful assistant"));
        assert_eq!(req["system"][0]["type"], "text");
        assert_eq!(req["system"][0]["text"], "you are a helpful assistant");
        assert_eq!(req["system"][0]["cache_control"]["type"], "ephemeral");
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
