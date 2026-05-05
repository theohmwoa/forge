//! An [`Agent`] backed by the Anthropic Messages API.
//!
//! Scope for v0.0.3: single-shot prompt -> assistant response. No tools, no
//! streaming. Each call materializes as two `StepKind`s in the run graph:
//! a `Prompt` and a `Message{role: assistant}`. Tool use and streaming are
//! the next slices.

use std::collections::VecDeque;

use async_trait::async_trait;
use forge_core::agent::Agent;
use forge_core::{NodeHash, StepKind};
use reqwest::header::{HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

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
    /// Steps the agent will emit on subsequent `next_step` calls. Populated
    /// up-front from the user prompt and lazily appended after the API call.
    pending: VecDeque<StepKind>,
    /// Has the API call already been issued? We do it on the first call to
    /// `next_step` and queue both the user prompt and the assistant reply.
    fired: bool,
    user_prompt: String,
}

impl AnthropicAgent {
    pub fn new(config: AnthropicConfig, user_prompt: impl Into<String>) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
            pending: VecDeque::new(),
            fired: false,
            user_prompt: user_prompt.into(),
        }
    }

    async fn fire(&mut self) -> anyhow::Result<()> {
        let req = MessagesRequest {
            model: self.config.model.clone(),
            max_tokens: self.config.max_tokens,
            messages: vec![InputMessage {
                role: "user".into(),
                content: self.user_prompt.clone(),
            }],
        };

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

        tracing::debug!(model = %self.config.model, "calling anthropic messages api");
        let resp = self
            .client
            .post(API_URL)
            .headers(headers)
            .json(&req)
            .send()
            .await?;

        let status = resp.status();
        let body: serde_json::Value = resp.json().await?;
        if !status.is_success() {
            anyhow::bail!("anthropic api error ({status}): {body}");
        }

        let parsed: MessagesResponse = serde_json::from_value(body)?;
        let assistant_text = parsed
            .content
            .into_iter()
            .map(|ContentBlock::Text { text }| text)
            .collect::<Vec<_>>()
            .join("");

        // Queue: prompt step (recorded first), then assistant reply.
        self.pending.push_back(StepKind::Prompt {
            model: self.config.model.clone(),
            content: self.user_prompt.clone(),
        });
        self.pending.push_back(StepKind::Message {
            role: "assistant".into(),
            content: assistant_text,
        });

        Ok(())
    }
}

#[async_trait]
impl Agent for AnthropicAgent {
    async fn next_step(&mut self, _parent: Option<NodeHash>) -> Option<StepKind> {
        if !self.fired {
            self.fired = true;
            if let Err(err) = self.fire().await {
                tracing::error!(?err, "anthropic call failed; emitting nothing");
                return None;
            }
        }
        self.pending.pop_front()
    }
}

#[derive(Serialize)]
struct MessagesRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<InputMessage>,
}

#[derive(Serialize)]
struct InputMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlock {
    Text { text: String },
}
