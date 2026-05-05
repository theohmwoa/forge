//! Embedded HTML viewer over the local sled DB.
//!
//! Single static page (compiled into the binary) that talks to a small JSON
//! API. No build step, no node_modules, no SaaS — just a Rust binary that
//! serves a viewer on localhost.

use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::get,
    Json, Router,
};
use forge::diff_chains;
use forge_core::NodeHash;
use forge_storage::Storage;
use serde::Deserialize;
use serde_json::{json, Value};

const INDEX_HTML: &str = include_str!("./web.html");

#[derive(Clone)]
pub struct WebState {
    pub storage: Arc<dyn Storage>,
}

pub fn router(state: WebState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/api/runs", get(list_runs))
        .route("/api/runs/{head}", get(run_chain))
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
                json!({ "kind": "modified", "a": a, "b": b })
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

struct ApiError(anyhow::Error);

impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(value: E) -> Self {
        Self(value.into())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": self.0.to_string() })),
        )
            .into_response()
    }
}
