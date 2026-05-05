//! An [`Agent`] backed by Google's Gemini `generateContent` API.
//!
//! Mirrors `forge-anthropic` and `forge-openai`: `GeminiAgent::new` for fresh
//! runs, `GeminiAgent::continuing` for resuming from a Forge step prefix. The
//! multi-turn function-calling loop runs inside `fire()`, with optional SSE
//! streaming via `with_streaming(true)`.
//!
//! Wire-format notes (the API differs from Anthropic/OpenAI):
//! - The conversation is `contents: [{role: "user"|"model", parts: [...]}]`.
//! - Function calls are emitted as `{functionCall: {name, args}}` parts inside
//!   a model turn, and answered with `{functionResponse: {name, response}}`
//!   parts in a user turn.
//! - Gemini does not assign call ids to function calls. We synthesize them as
//!   `gemini-call-{turn}-{idx}` so Forge's `ToolCall.call_id` /
//!   `ToolResult.call_id` link is preserved; when continuing, we resolve the
//!   tool name from the matching prior `ToolCall` step in the prefix.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use async_trait::async_trait;
use forge_core::agent::Agent;
use forge_core::tool::Tool;
use forge_core::{NodeHash, Step, StepKind};
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::{json, Value};

const API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";
const DEFAULT_MAX_TURNS: usize = 16;

#[derive(Debug, Clone)]
pub struct GeminiConfig {
    pub api_key: String,
    pub model: String,
    pub max_output_tokens: u32,
    /// Optional system instruction. Sent as Gemini's `systemInstruction`.
    pub system: Option<String>,
}

impl GeminiConfig {
    pub fn from_env(model: impl Into<String>) -> anyhow::Result<Self> {
        let api_key = std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_API_KEY"))
            .map_err(|_| anyhow::anyhow!("GEMINI_API_KEY (or GOOGLE_API_KEY) not set"))?;
        Ok(Self {
            api_key,
            model: model.into(),
            max_output_tokens: 1024,
            system: None,
        })
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }
}

pub struct GeminiAgent {
    config: GeminiConfig,
    client: reqwest::Client,
    tools: Vec<Arc<dyn Tool>>,
    pending: VecDeque<StepKind>,
    fired: bool,
    user_prompt: Option<String>,
    initial_history: Vec<Value>,
    /// Map of synthesized call_id -> tool name, learned either from the Forge
    /// prefix (continuations) or from this fire()'s own emitted ToolCalls.
    /// Needed because Gemini's `functionResponse` is keyed by name, not id.
    call_names: HashMap<String, String>,
    max_turns: usize,
    stream: bool,
}

impl GeminiAgent {
    pub fn new(config: GeminiConfig, user_prompt: impl Into<String>) -> Self {
        let user_prompt = user_prompt.into();
        let initial_history = vec![json!({
            "role": "user",
            "parts": [{"text": user_prompt.clone()}]
        })];
        Self {
            config,
            client: reqwest::Client::new(),
            tools: Vec::new(),
            pending: VecDeque::new(),
            fired: false,
            user_prompt: Some(user_prompt),
            initial_history,
            call_names: HashMap::new(),
            max_turns: DEFAULT_MAX_TURNS,
            stream: false,
        }
    }

    pub fn continuing(config: GeminiConfig, prefix: &[Step]) -> Self {
        let mut call_names = HashMap::new();
        for s in prefix {
            if let StepKind::ToolCall { call_id, name, .. } = &s.kind {
                call_names.insert(call_id.clone(), name.clone());
            }
        }
        let initial_history = prefix_to_contents(prefix, &call_names);
        Self {
            config,
            client: reqwest::Client::new(),
            tools: Vec::new(),
            pending: VecDeque::new(),
            fired: false,
            user_prompt: None,
            initial_history,
            call_names,
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

    pub fn with_streaming(mut self, on: bool) -> Self {
        self.stream = on;
        self
    }

    fn tool_declarations(&self) -> Vec<Value> {
        // Gemini wraps all function declarations in a single tools entry.
        self.tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name(),
                    "description": t.description(),
                    "parameters": t.schema()
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
            tracing::debug!(turn, model = %self.config.model, "gemini turn");
            let response = if self.stream {
                self.call_api_stream(&history).await?
            } else {
                self.call_api(&history).await?
            };
            let candidate = response["candidates"][0].clone();
            let content = candidate["content"].clone();
            let parts = content["parts"].as_array().cloned().unwrap_or_default();
            let finish_reason = candidate["finishReason"].as_str().unwrap_or("").to_string();

            // Push the model turn back into history verbatim for the next call.
            history.push(json!({
                "role": "model",
                "parts": parts.clone()
            }));

            let mut function_responses: Vec<Value> = Vec::new();
            let mut had_call = false;

            for (idx, part) in parts.iter().enumerate() {
                if let Some(text) = part["text"].as_str() {
                    if !text.is_empty() {
                        self.pending.push_back(StepKind::Message {
                            role: "assistant".into(),
                            content: text.to_string(),
                        });
                    }
                } else if let Some(call) = part.get("functionCall") {
                    had_call = true;
                    let name = call["name"].as_str().unwrap_or("").to_string();
                    let args = call["args"].clone();
                    let call_id = format!("gemini-call-{turn}-{idx}");
                    self.call_names.insert(call_id.clone(), name.clone());

                    self.pending.push_back(StepKind::ToolCall {
                        call_id: call_id.clone(),
                        name: name.clone(),
                        input: args.clone(),
                    });

                    let output = match self.tools.iter().find(|t| t.name() == name) {
                        Some(tool) => match tool.run(&args).await {
                            Ok(v) => v,
                            Err(e) => json!(e.to_string()),
                        },
                        None => json!(format!("unknown tool: {name}")),
                    };
                    self.pending.push_back(StepKind::ToolResult {
                        call_id: call_id.clone(),
                        output: output.clone(),
                    });

                    function_responses.push(json!({
                        "functionResponse": {
                            "name": name,
                            "response": wrap_response(&output),
                        }
                    }));
                }
            }

            if !had_call {
                return Ok(());
            }

            // Send all tool results back to the model in a single user turn.
            history.push(json!({
                "role": "user",
                "parts": function_responses
            }));

            if finish_reason != "STOP" && !finish_reason.is_empty() && finish_reason != "TOOL_USE" {
                // Anything other than a normal stop or a tool-call indication
                // (different SDKs use different strings) — bail.
                return Ok(());
            }
        }

        tracing::info!(
            max_turns = self.max_turns,
            "gemini agent stopped at turn cap"
        );
        Ok(())
    }

    fn build_request(&self, history: &[Value]) -> Value {
        let mut req = json!({
            "contents": history,
            "generationConfig": {
                "maxOutputTokens": self.config.max_output_tokens
            }
        });
        let decls = self.tool_declarations();
        if !decls.is_empty() {
            req["tools"] = json!([{ "functionDeclarations": decls }]);
        }
        if let Some(sys) = &self.config.system {
            req["systemInstruction"] = json!({
                "parts": [{"text": sys}]
            });
        }
        req
    }

    fn auth_headers(&self) -> anyhow::Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-goog-api-key",
            HeaderValue::from_str(&self.config.api_key)
                .map_err(|_| anyhow::anyhow!("GEMINI_API_KEY contained invalid characters"))?,
        );
        headers.insert("content-type", HeaderValue::from_static("application/json"));
        Ok(headers)
    }

    async fn call_api(&self, history: &[Value]) -> anyhow::Result<Value> {
        let url = format!("{API_BASE}/{}:generateContent", self.config.model);
        let req = self.build_request(history);
        let resp = self
            .client
            .post(&url)
            .headers(self.auth_headers()?)
            .json(&req)
            .send()
            .await?;
        let status = resp.status();
        let body: Value = resp.json().await?;
        if !status.is_success() {
            anyhow::bail!("gemini api error ({status}): {body}");
        }
        Ok(body)
    }

    /// SSE streaming variant. Gemini emits whole `GenerateContentResponse`
    /// objects as SSE `data:` lines; we accumulate text parts and capture
    /// functionCall parts (which arrive complete in one event).
    async fn call_api_stream(&self, history: &[Value]) -> anyhow::Result<Value> {
        use futures_util::StreamExt;
        use std::io::Write;

        let url = format!(
            "{API_BASE}/{}:streamGenerateContent?alt=sse",
            self.config.model
        );
        let req = self.build_request(history);
        let mut headers = self.auth_headers()?;
        headers.insert("accept", HeaderValue::from_static("text/event-stream"));

        let resp = self
            .client
            .post(&url)
            .headers(headers)
            .json(&req)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body: Value = resp.json().await.unwrap_or_else(|_| json!({}));
            anyhow::bail!("gemini api error ({status}): {body}");
        }

        let mut text_acc = String::new();
        let mut function_calls: Vec<Value> = Vec::new();
        let mut finish_reason = String::new();

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
                if data.is_empty() {
                    continue;
                }
                let event: Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let parts = event["candidates"][0]["content"]["parts"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
                for p in parts {
                    if let Some(t) = p["text"].as_str() {
                        text_acc.push_str(t);
                        eprint!("{t}");
                        let _ = std::io::stderr().flush();
                    } else if p.get("functionCall").is_some() {
                        function_calls.push(p);
                    }
                }
                if let Some(fr) = event["candidates"][0]["finishReason"].as_str() {
                    finish_reason = fr.to_string();
                }
            }
        }
        let _ = writeln!(std::io::stderr());
        let _ = std::io::stderr().flush();

        let mut parts: Vec<Value> = Vec::new();
        if !text_acc.is_empty() {
            parts.push(json!({"text": text_acc}));
        }
        parts.extend(function_calls);

        Ok(json!({
            "candidates": [{
                "content": {"role": "model", "parts": parts},
                "finishReason": finish_reason,
            }]
        }))
    }
}

#[async_trait]
impl Agent for GeminiAgent {
    async fn next_step(&mut self, _parent: Option<NodeHash>) -> Option<StepKind> {
        if !self.fired {
            self.fired = true;
            if let Err(err) = self.fire().await {
                tracing::error!(?err, "gemini agent failed; emitting nothing");
                return None;
            }
        }
        self.pending.pop_front()
    }
}

/// Gemini's `functionResponse.response` field expects a JSON object. If the
/// tool returned a scalar (number, string, etc.), wrap it in `{result: ...}`
/// so the API accepts it.
fn wrap_response(v: &Value) -> Value {
    if v.is_object() {
        v.clone()
    } else {
        json!({"result": v})
    }
}

/// Translate a Forge step prefix into the Gemini `contents` shape. Consecutive
/// same-role steps are coalesced into one content with multiple parts.
///
/// `call_names` is a map of synthesized call_id -> tool name learned from the
/// prefix's ToolCall steps; we use it to resolve the name to attach to each
/// `functionResponse` part.
pub fn prefix_to_contents(prefix: &[Step], call_names: &HashMap<String, String>) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    let mut current_role: Option<String> = None;
    let mut current_parts: Vec<Value> = Vec::new();

    fn flush(out: &mut Vec<Value>, role: &mut Option<String>, parts: &mut Vec<Value>) {
        if let Some(r) = role.take() {
            if !parts.is_empty() {
                out.push(json!({ "role": r, "parts": std::mem::take(parts) }));
            }
        }
    }

    for step in prefix {
        let (role, part) = match &step.kind {
            StepKind::Prompt { content, .. } => ("user".to_string(), json!({"text": content})),
            StepKind::Message { role, content } => {
                // Gemini uses "model" for assistant turns; "user" otherwise.
                let mapped = if role == "assistant" { "model" } else { "user" };
                (mapped.to_string(), json!({"text": content}))
            }
            StepKind::ToolCall { name, input, .. } => (
                "model".to_string(),
                json!({
                    "functionCall": {
                        "name": name,
                        "args": input
                    }
                }),
            ),
            StepKind::ToolResult { call_id, output } => {
                let name = call_names
                    .get(call_id)
                    .cloned()
                    .unwrap_or_else(|| call_id.clone());
                (
                    "user".to_string(),
                    json!({
                        "functionResponse": {
                            "name": name,
                            "response": wrap_response(output)
                        }
                    }),
                )
            }
        };

        if current_role.as_deref() != Some(role.as_str()) {
            flush(&mut out, &mut current_role, &mut current_parts);
            current_role = Some(role);
        }
        current_parts.push(part);
    }
    flush(&mut out, &mut current_role, &mut current_parts);
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
    fn tool_declarations_serialize_in_function_envelope() {
        let agent = GeminiAgent::new(
            GeminiConfig {
                api_key: "x".into(),
                model: "gemini-2.5-flash".into(),
                max_output_tokens: 1024,
                system: None,
            },
            "what is 2 + 3?",
        )
        .with_tools(vec![Arc::new(Calculator)]);
        let decls = agent.tool_declarations();
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0]["name"], "calculator");
        assert!(decls[0]["parameters"].is_object());
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
        let h = prefix_to_contents(&[s], &HashMap::new());
        assert_eq!(h.len(), 1);
        assert_eq!(h[0]["role"], "user");
        assert_eq!(h[0]["parts"][0]["text"], "hello");
    }

    #[test]
    fn prefix_maps_assistant_to_model_role() {
        let s0 = step(
            None,
            StepKind::Prompt {
                model: "m".into(),
                content: "hi".into(),
            },
        );
        let s1 = step(
            Some(s0.id.clone()),
            StepKind::Message {
                role: "assistant".into(),
                content: "hello".into(),
            },
        );
        let h = prefix_to_contents(&[s0, s1], &HashMap::new());
        assert_eq!(h.len(), 2);
        assert_eq!(h[1]["role"], "model");
    }

    #[test]
    fn prefix_packs_function_call_and_response() {
        let s0 = step(
            None,
            StepKind::Prompt {
                model: "m".into(),
                content: "what is 2 + 3?".into(),
            },
        );
        let s1 = step(
            Some(s0.id.clone()),
            StepKind::ToolCall {
                call_id: "gemini-call-0-0".into(),
                name: "calculator".into(),
                input: json!({"op": "add", "a": 2, "b": 3}),
            },
        );
        let s2 = step(
            Some(s1.id.clone()),
            StepKind::ToolResult {
                call_id: "gemini-call-0-0".into(),
                output: json!(5),
            },
        );

        let mut names = HashMap::new();
        names.insert("gemini-call-0-0".into(), "calculator".into());
        let h = prefix_to_contents(&[s0, s1, s2], &names);

        // user(prompt) -> model(functionCall) -> user(functionResponse)
        assert_eq!(h.len(), 3);
        assert_eq!(h[0]["role"], "user");
        assert_eq!(h[1]["role"], "model");
        assert_eq!(h[1]["parts"][0]["functionCall"]["name"], "calculator");
        assert_eq!(h[2]["role"], "user");
        assert_eq!(h[2]["parts"][0]["functionResponse"]["name"], "calculator");
        // Scalar response wrapped in {result: ...}.
        assert_eq!(
            h[2]["parts"][0]["functionResponse"]["response"]["result"],
            5
        );
    }

    #[test]
    fn wrap_response_preserves_objects() {
        let v = json!({"a": 1, "b": "two"});
        assert_eq!(wrap_response(&v), v);
    }

    #[test]
    fn wrap_response_wraps_scalars() {
        assert_eq!(wrap_response(&json!(42)), json!({"result": 42}));
        assert_eq!(wrap_response(&json!("hi")), json!({"result": "hi"}));
    }
}
