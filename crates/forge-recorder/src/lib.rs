//! Drop-in HTTP recorder for LLM API calls.
//!
//! The recorder runs as a local proxy that the user's existing agent points
//! at instead of the real provider URL. It forwards each request upstream,
//! returns the response unchanged, and records the round-trip into Forge.
//!
//! v0 supports Anthropic's `/v1/messages` and OpenAI's `/v1/chat/completions`.
//! Streaming requests are forwarded but not recorded yet (the SSE response
//! body would need teeing); for now we strip `stream:true` and warn.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use forge_core::{NodeHash, Step, StepKind};
use forge_storage::{RunMeta, SledStorage, Storage};
use futures_util::{stream, StreamExt};
use serde_json::{json, Value};

const ANTHROPIC_UPSTREAM: &str = "https://api.anthropic.com/v1/messages";
const OPENAI_UPSTREAM: &str = "https://api.openai.com/v1/chat/completions";

pub struct RecorderState {
    pub storage: Arc<SledStorage>,
    pub client: reqwest::Client,
}

pub fn router(state: Arc<RecorderState>) -> Router {
    Router::new()
        .route("/v1/messages", post(proxy_anthropic))
        .route("/v1/chat/completions", post(proxy_openai))
        .with_state(state)
}

async fn proxy_anthropic(
    State(state): State<Arc<RecorderState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, (StatusCode, String)> {
    let mut req = state.client.post(ANTHROPIC_UPSTREAM);
    for name in &["x-api-key", "anthropic-version", "anthropic-beta"] {
        if let Some(v) = headers.get(*name) {
            req = req.header(*name, v);
        }
    }
    let upstream = req
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
    let status = upstream.status();
    if !status.is_success() {
        let body: Value = upstream.json().await.unwrap_or_else(|_| json!({}));
        return Err((
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            body.to_string(),
        ));
    }

    let is_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if is_stream {
        // Tee: forward chunks to the client AS they arrive, accumulate the
        // raw SSE body, and parse + record once the stream completes.
        let captured: Arc<std::sync::Mutex<Vec<u8>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured_for_stream = Arc::clone(&captured);
        let storage = Arc::clone(&state.storage);
        let request_for_finalize = body.clone();

        let upstream_stream = upstream.bytes_stream().map(move |chunk| match chunk {
            Ok(bytes) => {
                if let Ok(mut buf) = captured_for_stream.lock() {
                    buf.extend_from_slice(&bytes);
                }
                Ok::<_, std::io::Error>(bytes)
            }
            Err(e) => Err(std::io::Error::other(e)),
        });

        // Append a no-op final chunk that triggers the recorder once the
        // stream is exhausted.
        let captured_for_finalize = Arc::clone(&captured);
        let finalize = stream::once(async move {
            let bytes = captured_for_finalize
                .lock()
                .map(|b| b.clone())
                .unwrap_or_default();
            let raw = String::from_utf8_lossy(&bytes).to_string();
            let response_body = forge_anthropic::parse_anthropic_sse_body(&raw);
            if let Err(err) =
                record_anthropic(&storage, &request_for_finalize, &response_body).await
            {
                tracing::warn!(?err, "failed to record streamed anthropic run");
            }
            Ok::<_, std::io::Error>(bytes::Bytes::new())
        });

        let body_stream = upstream_stream.chain(finalize);
        return Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .body(Body::from_stream(body_stream))
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
    }

    let response_body: Value = upstream
        .json()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;

    if let Err(err) = record_anthropic(&state.storage, &body, &response_body).await {
        tracing::warn!(?err, "failed to record anthropic round-trip");
    }
    Ok(Json(response_body).into_response())
}

async fn proxy_openai(
    State(state): State<Arc<RecorderState>>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> Result<Json<Value>, (StatusCode, String)> {
    if body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        tracing::warn!("recorder does not yet tee streaming responses; stripping stream=true");
        body["stream"] = json!(false);
    }
    let mut req = state.client.post(OPENAI_UPSTREAM);
    if let Some(v) = headers.get("authorization") {
        req = req.header("authorization", v);
    }
    if let Some(v) = headers.get("openai-organization") {
        req = req.header("openai-organization", v);
    }
    let resp = req
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
    let status = resp.status();
    let response_body: Value = resp
        .json()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
    if !status.is_success() {
        return Err((
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            response_body.to_string(),
        ));
    }

    if let Err(err) = record_openai(&state.storage, &body, &response_body).await {
        tracing::warn!(?err, "failed to record openai round-trip");
    }
    Ok(Json(response_body))
}

/// Convert an Anthropic request+response into a Forge run. Multi-call
/// conversations are auto-threaded: each request's `messages` array is
/// content-addressed, and any prefix that already exists in storage is
/// reused as-is. Only the new continuation steps get written. The final
/// chain (existing prefix + new continuation + response) is recorded as
/// the run's head, which naturally extends previous heads of the same
/// conversation.
async fn record_anthropic(
    storage: &SledStorage,
    request: &Value,
    response: &Value,
) -> anyhow::Result<()> {
    let model = request["model"].as_str().unwrap_or("unknown").to_string();
    let messages = request["messages"].as_array().cloned().unwrap_or_default();
    let now = now_ms();
    let mut chain: Vec<NodeHash> = Vec::new();
    let mut parent: Option<NodeHash> = None;
    let mut first_user_seen = false;

    let mut request_kinds: Vec<StepKind> = Vec::new();
    for msg in &messages {
        let role = msg["role"].as_str().unwrap_or("user").to_string();
        let kinds = anthropic_message_to_steps(&role, &msg["content"], &model, !first_user_seen);
        if role == "user" {
            first_user_seen = true;
        }
        request_kinds.extend(kinds);
    }

    // Walk the request kinds, hashing each. Steps that already exist in
    // storage are reused; the first one that doesn't exist marks the
    // divergence point and everything from there is new.
    for kind in &request_kinds {
        let candidate_id = Step::compute_id(&parent, kind);
        match storage.get(&candidate_id).await? {
            Some(_) => {
                chain.push(candidate_id.clone());
                parent = Some(candidate_id);
            }
            None => {
                let step = Step::new(parent.clone(), kind.clone(), now);
                let id = storage.put(step).await?;
                chain.push(id.clone());
                parent = Some(id);
            }
        }
    }

    // Response steps are always new (the model just produced them).
    let response_content = response["content"].as_array().cloned().unwrap_or_default();
    for block in &response_content {
        if let Some(kind) = anthropic_block_to_step(block) {
            let step = Step::new(parent.clone(), kind, now);
            let id = storage.put(step).await?;
            chain.push(id.clone());
            parent = Some(id);
        }
    }

    if let (Some(head), Some(root)) = (chain.last(), chain.first()) {
        storage.record_run(&RunMeta {
            head: head.clone(),
            root: root.clone(),
            recorded_at_ms: now,
        })?;
        tracing::info!(
            steps = chain.len(),
            head = %head,
            "recorded anthropic run (auto-threaded)"
        );
    }
    Ok(())
}

fn anthropic_message_to_steps(
    role: &str,
    content: &Value,
    model: &str,
    is_first_user: bool,
) -> Vec<StepKind> {
    let mut out = Vec::new();
    if let Some(text) = content.as_str() {
        if role == "user" && is_first_user {
            out.push(StepKind::Prompt {
                model: model.into(),
                content: text.to_string(),
            });
        } else {
            out.push(StepKind::Message {
                role: role.into(),
                content: text.to_string(),
            });
        }
        return out;
    }
    if let Some(blocks) = content.as_array() {
        let mut emitted_first_text_as_prompt = false;
        for block in blocks {
            match block["type"].as_str() {
                Some("text") => {
                    let text = block["text"].as_str().unwrap_or("").to_string();
                    if role == "user" && is_first_user && !emitted_first_text_as_prompt {
                        out.push(StepKind::Prompt {
                            model: model.into(),
                            content: text,
                        });
                        emitted_first_text_as_prompt = true;
                    } else {
                        out.push(StepKind::Message {
                            role: role.into(),
                            content: text,
                        });
                    }
                }
                Some("tool_use") => {
                    if let Some(kind) = anthropic_block_to_step(block) {
                        out.push(kind);
                    }
                }
                Some("tool_result") => {
                    out.push(StepKind::ToolResult {
                        call_id: block["tool_use_id"].as_str().unwrap_or("").to_string(),
                        output: block.get("content").cloned().unwrap_or(Value::Null),
                    });
                }
                _ => {}
            }
        }
    }
    out
}

fn anthropic_block_to_step(block: &Value) -> Option<StepKind> {
    match block["type"].as_str()? {
        "text" => Some(StepKind::Message {
            role: "assistant".into(),
            content: block["text"].as_str()?.to_string(),
        }),
        "tool_use" => Some(StepKind::ToolCall {
            call_id: block["id"].as_str()?.to_string(),
            name: block["name"].as_str()?.to_string(),
            input: block["input"].clone(),
        }),
        _ => None,
    }
}

async fn record_openai(
    storage: &SledStorage,
    request: &Value,
    response: &Value,
) -> anyhow::Result<()> {
    let model = request["model"].as_str().unwrap_or("unknown").to_string();
    let messages = request["messages"].as_array().cloned().unwrap_or_default();
    let now = now_ms();
    let mut chain: Vec<NodeHash> = Vec::new();
    let mut parent: Option<NodeHash> = None;
    let mut first_user_seen = false;

    let mut request_kinds: Vec<StepKind> = Vec::new();
    for msg in &messages {
        let role = msg["role"].as_str().unwrap_or("user").to_string();
        let kinds = openai_message_to_steps(&role, msg, &model, !first_user_seen);
        if role == "user" {
            first_user_seen = true;
        }
        request_kinds.extend(kinds);
    }
    for kind in &request_kinds {
        let candidate_id = Step::compute_id(&parent, kind);
        match storage.get(&candidate_id).await? {
            Some(_) => {
                chain.push(candidate_id.clone());
                parent = Some(candidate_id);
            }
            None => {
                let step = Step::new(parent.clone(), kind.clone(), now);
                let id = storage.put(step).await?;
                chain.push(id.clone());
                parent = Some(id);
            }
        }
    }

    let assistant = &response["choices"][0]["message"];
    if let Some(text) = assistant["content"].as_str() {
        if !text.is_empty() {
            let step = Step::new(
                parent.clone(),
                StepKind::Message {
                    role: "assistant".into(),
                    content: text.to_string(),
                },
                now,
            );
            let id = storage.put(step).await?;
            parent = Some(id.clone());
            chain.push(id);
        }
    }
    if let Some(tool_calls) = assistant["tool_calls"].as_array() {
        for tc in tool_calls {
            let id_s = tc["id"].as_str().unwrap_or("").to_string();
            let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
            let args_str = tc["function"]["arguments"].as_str().unwrap_or("");
            let input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
            let step = Step::new(
                parent.clone(),
                StepKind::ToolCall {
                    call_id: id_s,
                    name,
                    input,
                },
                now,
            );
            let id = storage.put(step).await?;
            parent = Some(id.clone());
            chain.push(id);
        }
    }

    if let (Some(head), Some(root)) = (chain.last(), chain.first()) {
        storage.record_run(&RunMeta {
            head: head.clone(),
            root: root.clone(),
            recorded_at_ms: now,
        })?;
        tracing::info!(steps = chain.len(), head = %head, "recorded openai run");
    }
    Ok(())
}

fn openai_message_to_steps(
    role: &str,
    msg: &Value,
    model: &str,
    is_first_user: bool,
) -> Vec<StepKind> {
    let mut out = Vec::new();
    match role {
        "user" => {
            if let Some(text) = msg["content"].as_str() {
                if is_first_user {
                    out.push(StepKind::Prompt {
                        model: model.into(),
                        content: text.to_string(),
                    });
                } else {
                    out.push(StepKind::Message {
                        role: "user".into(),
                        content: text.to_string(),
                    });
                }
            }
        }
        "assistant" => {
            if let Some(text) = msg["content"].as_str() {
                if !text.is_empty() {
                    out.push(StepKind::Message {
                        role: "assistant".into(),
                        content: text.to_string(),
                    });
                }
            }
            if let Some(tool_calls) = msg["tool_calls"].as_array() {
                for tc in tool_calls {
                    let id = tc["id"].as_str().unwrap_or("").to_string();
                    let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                    let args_str = tc["function"]["arguments"].as_str().unwrap_or("");
                    let input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
                    out.push(StepKind::ToolCall {
                        call_id: id,
                        name,
                        input,
                    });
                }
            }
        }
        "tool" => {
            let id = msg["tool_call_id"].as_str().unwrap_or("").to_string();
            let content = msg["content"].clone();
            let output = if let Some(s) = content.as_str() {
                serde_json::from_str(s).unwrap_or(Value::String(s.to_string()))
            } else {
                content
            };
            out.push(StepKind::ToolResult {
                call_id: id,
                output,
            });
        }
        _ => {}
    }
    out
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
    use serde_json::json;

    #[tokio::test]
    async fn anthropic_recording_emits_prompt_then_assistant() {
        let dir = tempfile::tempdir().unwrap();
        let storage = SledStorage::open(dir.path()).unwrap();
        let request = json!({
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": "what is 2+3?"}
            ]
        });
        let response = json!({
            "content": [
                {"type": "text", "text": "5"}
            ],
            "stop_reason": "end_turn"
        });
        record_anthropic(&storage, &request, &response)
            .await
            .unwrap();
        let runs = storage.list_runs().unwrap();
        assert_eq!(runs.len(), 1);
        let chain = storage.chain_to(&runs[0].head).unwrap();
        assert_eq!(chain.len(), 2);
        assert!(matches!(chain[0].kind, StepKind::Prompt { .. }));
        assert!(matches!(chain[1].kind, StepKind::Message { ref role, .. } if role == "assistant"));
    }

    #[tokio::test]
    async fn anthropic_recording_threads_a_multi_call_conversation() {
        // Two API calls that share the same prefix should produce ONE
        // tree, with the second call's response branching from the first
        // assistant message — i.e., shared prefix length 2.
        let dir = tempfile::tempdir().unwrap();
        let storage = SledStorage::open(dir.path()).unwrap();

        let req1 = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "what is 2+3?"}]
        });
        let resp1 = json!({
            "content": [{"type": "text", "text": "5"}],
            "stop_reason": "end_turn"
        });
        record_anthropic(&storage, &req1, &resp1).await.unwrap();

        // Second call includes the prefix + a follow-up.
        let req2 = json!({
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": "what is 2+3?"},
                {"role": "assistant", "content": [{"type": "text", "text": "5"}]},
                {"role": "user", "content": "and 5+5?"}
            ]
        });
        let resp2 = json!({
            "content": [{"type": "text", "text": "10"}],
            "stop_reason": "end_turn"
        });
        record_anthropic(&storage, &req2, &resp2).await.unwrap();

        let runs = storage.list_runs().unwrap();
        assert_eq!(runs.len(), 2, "two heads, one per call");

        // Order-independent: pick the shorter chain as the prefix and assert
        // it lines up with the longer chain step-for-step.
        let a = storage.chain_to(&runs[0].head).unwrap();
        let b = storage.chain_to(&runs[1].head).unwrap();
        let (short, long) = if a.len() < b.len() { (a, b) } else { (b, a) };
        for i in 0..short.len() {
            assert_eq!(short[i].id, long[i].id, "step {i} should be shared");
        }
        assert!(long.len() > short.len(), "second call extends first");
    }

    #[tokio::test]
    async fn anthropic_recording_branches_when_continuation_diverges() {
        // Two different follow-ups from the same prefix should produce two
        // distinct heads sharing the prefix steps.
        let dir = tempfile::tempdir().unwrap();
        let storage = SledStorage::open(dir.path()).unwrap();

        let req1 = json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "what is 2+3?"},
                {"role": "assistant", "content": [{"type": "text", "text": "5"}]},
                {"role": "user", "content": "and 5+5?"}
            ]
        });
        let resp1 = json!({
            "content": [{"type": "text", "text": "10"}], "stop_reason": "end_turn"
        });
        let req2 = json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "what is 2+3?"},
                {"role": "assistant", "content": [{"type": "text", "text": "5"}]},
                {"role": "user", "content": "what about 7+8?"}
            ]
        });
        let resp2 = json!({
            "content": [{"type": "text", "text": "15"}], "stop_reason": "end_turn"
        });

        record_anthropic(&storage, &req1, &resp1).await.unwrap();
        record_anthropic(&storage, &req2, &resp2).await.unwrap();

        let runs = storage.list_runs().unwrap();
        assert_eq!(runs.len(), 2, "two heads, one per branch");
        let c1 = storage.chain_to(&runs[0].head).unwrap();
        let c2 = storage.chain_to(&runs[1].head).unwrap();
        assert_eq!(c1[0].id, c2[0].id, "prompt is shared");
        assert_eq!(c1[1].id, c2[1].id, "first assistant message is shared");
        assert_ne!(c1[2].id, c2[2].id, "follow-up diverges");

        // The first shared assistant message should have two children.
        let kids = storage.children(&c1[1].id).await.unwrap();
        assert_eq!(kids.len(), 2, "branching produces two children");
    }

    #[tokio::test]
    async fn anthropic_recording_idempotent_replay() {
        // Recording the same exact request+response twice should not
        // grow the chain — every step's hash already exists.
        let dir = tempfile::tempdir().unwrap();
        let storage = SledStorage::open(dir.path()).unwrap();
        let req = json!({"model": "m", "messages": [{"role":"user","content":"hi"}]});
        let resp = json!({"content":[{"type":"text","text":"hi back"}], "stop_reason":"end_turn"});

        record_anthropic(&storage, &req, &resp).await.unwrap();
        let runs1 = storage.list_runs().unwrap();
        assert_eq!(runs1.len(), 1);
        let chain1 = storage.chain_to(&runs1[0].head).unwrap();

        record_anthropic(&storage, &req, &resp).await.unwrap();
        let runs2 = storage.list_runs().unwrap();
        assert_eq!(runs2.len(), 1, "replay should not create a new run");
        assert_eq!(runs2[0].head, runs1[0].head);
        let chain2 = storage.chain_to(&runs2[0].head).unwrap();
        assert_eq!(chain2.len(), chain1.len());
    }

    #[tokio::test]
    async fn anthropic_recording_preserves_tool_call_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let storage = SledStorage::open(dir.path()).unwrap();

        // First request: user asks something, model uses tool.
        let req1 = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "compute 7*8"}]
        });
        let resp1 = json!({
            "content": [
                {"type": "text", "text": "Using calculator."},
                {"type": "tool_use", "id": "toolu_01", "name": "calculator",
                 "input": {"op": "mul", "a": 7, "b": 8}}
            ],
            "stop_reason": "tool_use"
        });
        record_anthropic(&storage, &req1, &resp1).await.unwrap();

        // Second request: client sends tool_result, model replies.
        let req2 = json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "compute 7*8"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Using calculator."},
                    {"type": "tool_use", "id": "toolu_01", "name": "calculator",
                     "input": {"op": "mul", "a": 7, "b": 8}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01", "content": "56"}
                ]}
            ]
        });
        let resp2 = json!({
            "content": [{"type": "text", "text": "The answer is 56."}],
            "stop_reason": "end_turn"
        });
        record_anthropic(&storage, &req2, &resp2).await.unwrap();

        let runs = storage.list_runs().unwrap();
        // Different heads — the second call extends the first.
        let chains: Vec<_> = runs
            .iter()
            .map(|r| storage.chain_to(&r.head).unwrap())
            .collect();
        let longest = chains.iter().max_by_key(|c| c.len()).unwrap();
        // prompt + assistant message + tool_call + tool_result + final message
        assert_eq!(longest.len(), 5);
        assert!(matches!(longest[2].kind, StepKind::ToolCall { .. }));
        assert!(matches!(longest[3].kind, StepKind::ToolResult { .. }));
    }

    #[tokio::test]
    async fn openai_recording_handles_tool_calls() {
        let dir = tempfile::tempdir().unwrap();
        let storage = SledStorage::open(dir.path()).unwrap();
        let request = json!({
            "model": "gpt-5",
            "messages": [
                {"role": "user", "content": "compute 7*8"}
            ]
        });
        let response = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Using calculator.",
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": "{\"op\":\"mul\",\"a\":7,\"b\":8}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        record_openai(&storage, &request, &response).await.unwrap();
        let runs = storage.list_runs().unwrap();
        assert_eq!(runs.len(), 1);
        let chain = storage.chain_to(&runs[0].head).unwrap();
        // prompt + assistant message + tool_call
        assert_eq!(chain.len(), 3);
        assert!(matches!(chain[2].kind, StepKind::ToolCall { .. }));
    }
}
