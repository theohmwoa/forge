//! Model Context Protocol (MCP) client + Forge `Tool` adapter.
//!
//! MCP servers expose collections of tools over a JSON-RPC 2.0 wire format.
//! This crate spawns/connects to an MCP server, performs the standard
//! `initialize` / `tools/list` handshake, and wraps each discovered tool as a
//! [`forge_core::tool::Tool`] so existing Forge agents can invoke them with
//! zero changes to the agent code.
//!
//! Wire format: newline-delimited JSON-RPC 2.0 over stdio. One message per
//! line, in either direction. Notifications (no `id`) are accepted but not
//! dispatched to user code.
//!
//! Lifecycle:
//! 1. `McpClient::spawn` (subprocess) or `McpClient::from_streams` (any pair
//!    of `AsyncRead` + `AsyncWrite`).
//! 2. `client.list_tools()` returns descriptors.
//! 3. Wrap each descriptor with [`McpTool::new`]; pass the `Vec<Arc<dyn Tool>>`
//!    to any Forge agent's `with_tools(...)` method.
//! 4. The agent calls `Tool::run(input)` → routes to `McpClient::call_tool` →
//!    JSON-RPC `tools/call` → MCP server's tool handler → result bubbles back.

pub mod client;
pub mod tool;

pub use client::{McpClient, McpToolDescriptor};
pub use tool::{mcp_tools_into_dyn, McpTool};
