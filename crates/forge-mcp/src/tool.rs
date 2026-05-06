//! `Tool` trait adapter so MCP-discovered tools drop into any Forge agent's
//! `with_tools(...)` call without changes.
//!
//! Each `McpTool` carries an `Arc<McpClient>` so multiple tools wrapped from
//! the same server share one transport — invariant the agent loop relies on
//! (otherwise each `Tool::run` would spin up a fresh connection).

use std::sync::Arc;

use async_trait::async_trait;
use forge_core::tool::Tool;
use serde_json::Value;

use crate::client::{McpClient, McpToolDescriptor};

/// Adapter wrapping an MCP-discovered tool descriptor + a shared client into
/// the Forge `Tool` trait.
pub struct McpTool {
    descriptor: McpToolDescriptor,
    client: Arc<McpClient>,
}

impl McpTool {
    pub fn new(client: Arc<McpClient>, descriptor: McpToolDescriptor) -> Self {
        Self { descriptor, client }
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.descriptor.name
    }

    fn description(&self) -> &str {
        &self.descriptor.description
    }

    fn schema(&self) -> Value {
        self.descriptor.input_schema.clone()
    }

    async fn run(&self, input: &Value) -> anyhow::Result<Value> {
        self.client
            .call_tool(&self.descriptor.name, input.clone())
            .await
    }
}

/// Convenience: list every tool the client exposes and box each into
/// `Arc<dyn Tool>` so the result is drop-in for `with_tools(...)` on an
/// existing Forge agent.
pub async fn mcp_tools_into_dyn(
    client: Arc<McpClient>,
) -> anyhow::Result<Vec<Arc<dyn Tool>>> {
    let descriptors = client.list_tools().await?;
    Ok(descriptors
        .into_iter()
        .map(|d| Arc::new(McpTool::new(Arc::clone(&client), d)) as Arc<dyn Tool>)
        .collect())
}
