//! Embedded HTML viewer over the local sled DB.
//!
//! Single static page (compiled into the binary) that talks to a small JSON
//! API. No build step, no node_modules, no SaaS — just a Rust binary that
//! serves a viewer on localhost.

use std::sync::Arc;

use axum::{
    extract::{ConnectInfo, Path, Query, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        Html, IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use forge::{diff_chains, fork_chain, run_agent_from};
use forge_anthropic::{AnthropicAgent, AnthropicConfig};
use forge_core::agent::Agent;
use forge_core::tool::{Calculator, Tool};
use forge_core::{NodeHash, Step, StepKind};
use forge_gemini::{GeminiAgent, GeminiConfig};
use forge_openai::{OpenAIAgent, OpenAIConfig};
use forge_storage::{RunMeta, Storage};
use futures_util::stream::Stream;
use serde::Deserialize;
use serde_json::{json, Value};
use similar::{ChangeTag, TextDiff};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

const INDEX_HTML: &str = include_str!("./web.html");

#[derive(Clone)]
pub struct WebState {
    pub storage: Arc<dyn Storage>,
}

pub fn router(state: WebState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/api/runs", get(list_runs))
        .route("/api/runs/stream", get(stream_runs))
        .route("/api/runs/{head}", get(run_chain))
        .route("/api/runs/{head}/fork", post(fork_run))
        .route("/api/runs/{head}/continue", post(continue_run))
        .route("/api/diff/{a}/{b}", get(diff))
        .with_state(state)
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

#[derive(Deserialize)]
struct ListRunsQuery {
    tag: Option<String>,
}

async fn list_runs(
    State(s): State<WebState>,
    Query(q): Query<ListRunsQuery>,
) -> Result<Json<Value>, ApiError> {
    let mut runs = s.storage.list_runs().await?;
    if let Some(tag) = q.tag.as_deref() {
        runs.retain(|r| r.tag.as_deref() == Some(tag));
    }
    Ok(Json(json!({ "runs": runs })))
}

async fn run_chain(
    State(s): State<WebState>,
    Path(head): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let head_hash = NodeHash(head);
    let chain = s.storage.chain_to(&head_hash).await?;
    Ok(Json(json!({ "chain": chain })))
}

async fn diff(
    State(s): State<WebState>,
    Path((a, b)): Path<(String, String)>,
) -> Result<Json<Value>, ApiError> {
    let chain_a = s.storage.chain_to(&NodeHash(a)).await?;
    let chain_b = s.storage.chain_to(&NodeHash(b)).await?;
    let result = diff_chains(&chain_a, &chain_b);
    let aligned: Vec<Value> = result
        .aligned
        .iter()
        .map(|entry| match entry {
            forge::AlignedStep::Match(a, _) => json!({ "kind": "match", "step": a }),
            forge::AlignedStep::Modified(a, b) => {
                json!({
                    "kind": "modified",
                    "a": a,
                    "b": b,
                    "token_diff": token_diff_chunks(a, b),
                })
            }
            forge::AlignedStep::OnlyA(s) => json!({ "kind": "only_a", "step": s }),
            forge::AlignedStep::OnlyB(s) => json!({ "kind": "only_b", "step": s }),
        })
        .collect();
    Ok(Json(json!({
        "common_prefix_len": result.common_prefix_len,
        "prefix": chain_a.iter().take(result.common_prefix_len).collect::<Vec<_>>(),
        "aligned": aligned,
    })))
}

/// SSE stream of newly-recorded runs. Powered by `Storage::subscribe_runs`,
/// which pushes from in-process `record_run` calls AND (for Postgres) from
/// `LISTEN forge_runs` notifications. The client appends each event to the
/// runs list without polling.
async fn stream_runs(
    State(s): State<WebState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = s.storage.subscribe_runs();
    let stream = BroadcastStream::new(rx).filter_map(|res| match res {
        Ok(meta) => {
            let payload = serde_json::to_string(&meta).ok()?;
            Some(Ok(Event::default().event("run").data(payload)))
        }
        // Lagged: drop the gap silently — front-end refreshes from /api/runs.
        Err(_) => None,
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Render a step's body the same way the front-end does, so the token diff
/// matches what the user sees.
fn step_body(step: &Step) -> String {
    match &step.kind {
        StepKind::Prompt { content, .. } => content.clone(),
        StepKind::Message { content, .. } => content.clone(),
        StepKind::ToolCall { input, .. } => serde_json::to_string_pretty(input).unwrap_or_default(),
        StepKind::ToolResult { output, .. } => {
            serde_json::to_string_pretty(output).unwrap_or_default()
        }
    }
}

/// Word-level diff over the rendered bodies of two modified steps. Returns a
/// flat list of `{tag: equal|delete|insert, text: ...}` chunks the front-end
/// can colorize inline.
fn token_diff_chunks(a: &Step, b: &Step) -> Vec<Value> {
    let body_a = step_body(a);
    let body_b = step_body(b);
    let diff = TextDiff::from_words(&body_a, &body_b);
    let mut out = Vec::new();
    for change in diff.iter_all_changes() {
        let tag = match change.tag() {
            ChangeTag::Equal => "equal",
            ChangeTag::Delete => "delete",
            ChangeTag::Insert => "insert",
        };
        out.push(json!({ "tag": tag, "text": change.value() }));
    }
    out
}

/// Reject any mutating handler whose peer isn't on the loopback interface.
/// `forge web` has no auth yet (planned), so we defense-in-depth gate the
/// fork/continue endpoints to localhost regardless of `--public`.
fn require_loopback(addr: SocketAddr) -> Result<(), ApiError> {
    if !addr.ip().is_loopback() {
        return Err(ApiError::forbidden(
            "fork/continue are localhost-only until forge web ships auth (use the CLI for remote)",
        ));
    }
    Ok(())
}

#[derive(Deserialize)]
struct ForkBody {
    /// Hash (or unique prefix) of the step to rewrite.
    at: String,
    /// New content. Text for prompt/message; JSON for tool_call/tool_result.
    rewrite: String,
}

async fn fork_run(
    State(s): State<WebState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Path(head): Path<String>,
    Json(body): Json<ForkBody>,
) -> Result<Json<Value>, ApiError> {
    require_loopback(addr)?;
    let head = NodeHash(head);
    let chain = s.storage.chain_to(&head).await?;
    let new_chain = fork_chain(&*s.storage, &chain, &body.at, &body.rewrite).await?;
    let new_head = new_chain
        .last()
        .cloned()
        .ok_or_else(|| ApiError::from(anyhow::anyhow!("fork produced no steps")))?;
    let root = new_chain[0].clone();
    s.storage
        .record_run(&RunMeta {
            head: new_head.clone(),
            root,
            recorded_at_ms: now_ms(),
            tag: None,
        })
        .await?;
    Ok(Json(json!({ "head": new_head })))
}

#[derive(Deserialize)]
struct ContinueBody {
    /// "anthropic" | "openai" | "gemini"
    agent: String,
    model: String,
    #[serde(default)]
    max_turns: Option<usize>,
    /// Tool names to make available. Currently only "calculator" is recognized.
    #[serde(default)]
    tools: Vec<String>,
}

async fn continue_run(
    State(s): State<WebState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Path(head): Path<String>,
    Json(body): Json<ContinueBody>,
) -> Result<Json<Value>, ApiError> {
    require_loopback(addr)?;
    let head = NodeHash(head);
    let prefix: Vec<Step> = s.storage.chain_to(&head).await?;
    let tools = build_tools_by_name(&body.tools)?;
    let mut agent: Box<dyn Agent> = build_continuing_agent_from_strings(
        &body.agent,
        &prefix,
        &body.model,
        tools,
        body.max_turns,
    )?;
    let appended = run_agent_from(Some(head.clone()), agent.as_mut(), &*s.storage).await?;
    if appended.is_empty() {
        return Err(ApiError::from(anyhow::anyhow!(
            "agent emitted no new steps; nothing to record"
        )));
    }
    let new_head = appended.last().cloned().unwrap();
    let root = prefix[0].id.clone();
    s.storage
        .record_run(&RunMeta {
            head: new_head.clone(),
            root,
            recorded_at_ms: now_ms(),
            tag: None,
        })
        .await?;
    Ok(Json(json!({
        "head": new_head,
        "appended": appended.len(),
    })))
}

fn build_tools_by_name(names: &[String]) -> Result<Vec<std::sync::Arc<dyn Tool>>, ApiError> {
    names
        .iter()
        .map(|n| match n.as_str() {
            "calculator" => Ok(std::sync::Arc::new(Calculator) as std::sync::Arc<dyn Tool>),
            other => Err(ApiError::bad_request(format!("unknown tool: {other}"))),
        })
        .collect()
}

fn build_continuing_agent_from_strings(
    agent: &str,
    prefix: &[Step],
    model: &str,
    tools: Vec<std::sync::Arc<dyn Tool>>,
    max_turns: Option<usize>,
) -> Result<Box<dyn Agent>, ApiError> {
    match agent {
        "anthropic" => {
            let cfg = AnthropicConfig::from_env(model.to_string())?;
            let mut a = AnthropicAgent::continuing(cfg, prefix).with_tools(tools);
            if let Some(n) = max_turns {
                a = a.with_max_turns(n);
            }
            Ok(Box::new(a))
        }
        "openai" => {
            let cfg = OpenAIConfig::from_env(model.to_string())?;
            let mut a = OpenAIAgent::continuing(cfg, prefix).with_tools(tools);
            if let Some(n) = max_turns {
                a = a.with_max_turns(n);
            }
            Ok(Box::new(a))
        }
        "gemini" => {
            let cfg = GeminiConfig::from_env(model.to_string())?;
            let mut a = GeminiAgent::continuing(cfg, prefix).with_tools(tools);
            if let Some(n) = max_turns {
                a = a.with_max_turns(n);
            }
            Ok(Box::new(a))
        }
        other => Err(ApiError::bad_request(format!(
            "unknown agent: {other}; expected anthropic / openai / gemini"
        ))),
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn forbidden(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            message: msg.into(),
        }
    }
    fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }
}

impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(value: E) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: value.into().to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "error": self.message }))).into_response()
    }
}
