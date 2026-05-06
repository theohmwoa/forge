//! JSON-RPC 2.0 client for MCP over newline-delimited stdio.
//!
//! Architecture:
//! - One background reader task drains the server's output stream, parses
//!   each line as a JSON-RPC envelope, and dispatches to the matching pending
//!   oneshot by id.
//! - The public `request()` method allocates an id, registers a oneshot,
//!   writes the framed payload, and awaits the response.
//! - Notifications (no `id`) on the wire are recognized and currently
//!   ignored — MCP supports server-initiated `notifications/*` for things
//!   like `tools/list_changed` which a future caller may wire through.
//!
//! Concurrency: writes are serialized through an async mutex so that
//! interleaved JSON envelopes can never appear on the wire.
//! Multiple in-flight requests are supported because dispatch is keyed by id.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{oneshot, Mutex};

/// Description of a single tool exposed by an MCP server, as returned by
/// `tools/list`. Field shapes mirror the MCP spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDescriptor {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// JSON Schema for the tool's `arguments` object.
    pub input_schema: Value,
}

#[derive(Debug, Serialize)]
struct OutgoingRequest<'a> {
    jsonrpc: &'static str,
    id: u64,
    method: &'a str,
    params: Value,
}

#[derive(Debug, Serialize)]
struct OutgoingNotification<'a> {
    jsonrpc: &'static str,
    method: &'a str,
    params: Value,
}

#[derive(Debug, Deserialize)]
struct IncomingEnvelope {
    #[allow(dead_code)]
    #[serde(default)]
    jsonrpc: Option<String>,
    #[serde(default)]
    id: Option<u64>,
    #[serde(default)]
    result: Option<Value>,
    #[serde(default)]
    error: Option<RpcError>,
    #[serde(default)]
    method: Option<String>,
    #[serde(default)]
    params: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct RpcError {
    code: i64,
    message: String,
    #[allow(dead_code)]
    #[serde(default)]
    data: Option<Value>,
}

type PendingMap = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, RpcError>>>>>;

/// MCP client. Cheap to clone via `Arc<McpClient>`; the reader task and
/// subprocess lifetime are tied to the original instance.
pub struct McpClient {
    next_id: AtomicU64,
    pending: PendingMap,
    writer: Mutex<Box<dyn AsyncWrite + Unpin + Send>>,
    /// Reader task handle. Held so it isn't cancelled before the client is
    /// dropped; aborted on drop.
    reader_task: tokio::task::JoinHandle<()>,
    /// Optional subprocess. Kept alive for the lifetime of the client; killed
    /// on drop. `None` when the client was constructed from in-memory streams.
    #[allow(dead_code)]
    child: Mutex<Option<Child>>,
    /// Server-reported metadata captured during `initialize`. Useful for
    /// debugging multi-server setups.
    server_info: Mutex<Value>,
}

impl McpClient {
    /// Spawn an MCP server as a subprocess and connect to its stdio.
    /// `command` and `args` form the executable invocation; the server's
    /// stderr is inherited so its diagnostics surface in the parent's log.
    pub async fn spawn(command: &str, args: &[&str]) -> anyhow::Result<Arc<Self>> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| anyhow::anyhow!("failed to spawn mcp server `{command}`: {e}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("mcp server has no stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("mcp server has no stdout"))?;
        Self::from_streams(stdin, stdout, Some(child)).await
    }

    /// Build a client over arbitrary async streams. Used by `spawn` and by
    /// in-process tests with `tokio::io::duplex`.
    pub async fn from_streams<W, R>(
        writer: W,
        reader: R,
        child: Option<Child>,
    ) -> anyhow::Result<Arc<Self>>
    where
        W: AsyncWrite + Unpin + Send + 'static,
        R: AsyncRead + Unpin + Send + 'static,
    {
        let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));
        let pending_for_task = Arc::clone(&pending);

        let reader_task = tokio::spawn(reader_loop(reader, pending_for_task));

        let client = Arc::new(Self {
            next_id: AtomicU64::new(1),
            pending,
            writer: Mutex::new(Box::new(writer)),
            reader_task,
            child: Mutex::new(child),
            server_info: Mutex::new(Value::Null),
        });

        // MCP handshake: initialize → wait for response → send `initialized`
        // notification. Spec: https://spec.modelcontextprotocol.io
        let init_result = client
            .request(
                "initialize",
                json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "forge-mcp",
                        "version": env!("CARGO_PKG_VERSION"),
                    }
                }),
            )
            .await?;
        *client.server_info.lock().await = init_result;

        client
            .notify("notifications/initialized", json!({}))
            .await?;

        Ok(client)
    }

    /// Server-side `serverInfo` + capabilities returned during initialize.
    pub async fn server_info(&self) -> Value {
        self.server_info.lock().await.clone()
    }

    /// Issue a JSON-RPC request and await its response.
    async fn request(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);

        let envelope = OutgoingRequest {
            jsonrpc: "2.0",
            id,
            method,
            params,
        };
        if let Err(e) = self.write_envelope(&envelope).await {
            // Write failed — pull our pending entry back out so the reader
            // task doesn't keep an orphaned oneshot around.
            self.pending.lock().await.remove(&id);
            return Err(e);
        }

        match rx.await {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(rpc)) => Err(anyhow::anyhow!(
                "mcp rpc error {} on `{method}`: {}",
                rpc.code,
                rpc.message
            )),
            Err(_) => Err(anyhow::anyhow!(
                "mcp client closed before response to `{method}`"
            )),
        }
    }

    async fn notify(&self, method: &str, params: Value) -> anyhow::Result<()> {
        let envelope = OutgoingNotification {
            jsonrpc: "2.0",
            method,
            params,
        };
        self.write_envelope(&envelope).await
    }

    async fn write_envelope<T: Serialize>(&self, envelope: &T) -> anyhow::Result<()> {
        let mut buf = serde_json::to_vec(envelope)
            .map_err(|e| anyhow::anyhow!("failed to serialize mcp envelope: {e}"))?;
        buf.push(b'\n');
        let mut w = self.writer.lock().await;
        w.write_all(&buf)
            .await
            .map_err(|e| anyhow::anyhow!("mcp write failed: {e}"))?;
        w.flush()
            .await
            .map_err(|e| anyhow::anyhow!("mcp flush failed: {e}"))?;
        Ok(())
    }

    /// List all tools the server exposes via `tools/list`.
    pub async fn list_tools(&self) -> anyhow::Result<Vec<McpToolDescriptor>> {
        let result = self.request("tools/list", json!({})).await?;
        let arr = result["tools"].as_array().ok_or_else(|| {
            anyhow::anyhow!("mcp tools/list response missing `tools` array: {result}")
        })?;
        let tools = arr
            .iter()
            .map(|t| McpToolDescriptor {
                name: t["name"].as_str().unwrap_or("").to_string(),
                description: t["description"].as_str().unwrap_or("").to_string(),
                // Some servers return `inputSchema`, others `input_schema`.
                input_schema: if t["inputSchema"].is_object() {
                    t["inputSchema"].clone()
                } else {
                    t["input_schema"].clone()
                },
            })
            .collect();
        Ok(tools)
    }

    /// Invoke a tool by name with structured arguments.
    ///
    /// MCP tool results carry a `content` array of typed parts. We collapse
    /// text parts into a single string and surface other parts (like
    /// resources, images) verbatim. The shape Forge tools return is a
    /// `serde_json::Value` so the agent's downstream rendering stays generic.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> anyhow::Result<Value> {
        let result = self
            .request(
                "tools/call",
                json!({ "name": name, "arguments": arguments }),
            )
            .await?;
        if result["isError"].as_bool().unwrap_or(false) {
            // Surface as a tool error so the agent's loop sees a normal
            // error string — not a transport-layer panic.
            let detail = collapse_content(&result["content"]);
            anyhow::bail!("mcp tool `{name}` reported error: {detail}");
        }
        let content = result["content"].clone();
        Ok(collapse_content_value(&content))
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        // Best-effort: abort the reader task and let the subprocess receive
        // SIGKILL via tokio's `kill_on_drop`.
        self.reader_task.abort();
    }
}

/// Background loop that parses one JSON-RPC envelope per line and dispatches
/// responses to their pending oneshot channels.
async fn reader_loop<R: AsyncRead + Unpin + Send + 'static>(reader: R, pending: PendingMap) {
    let mut buf = BufReader::new(reader);
    let mut line = String::new();
    loop {
        line.clear();
        match buf.read_line(&mut line).await {
            Ok(0) => {
                tracing::debug!("mcp server closed connection");
                break;
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(error = %e, "mcp read error; closing reader loop");
                break;
            }
        }
        if line.trim().is_empty() {
            continue;
        }
        let env: IncomingEnvelope = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!(error = %err, line = %line.trim(), "mcp: invalid envelope");
                continue;
            }
        };
        match (env.id, env.method.as_deref()) {
            (Some(id), _) => {
                let tx_opt = pending.lock().await.remove(&id);
                if let Some(tx) = tx_opt {
                    let result = if let Some(rpc_err) = env.error {
                        Err(rpc_err)
                    } else {
                        Ok(env.result.unwrap_or(Value::Null))
                    };
                    let _ = tx.send(result);
                } else {
                    tracing::debug!(id, "mcp: unmatched response id (already completed?)");
                }
            }
            (None, Some(method)) => {
                tracing::debug!(method, "mcp: ignoring server notification");
                let _ = env.params; // suppress unused-warning path
            }
            (None, None) => {
                tracing::warn!(line = %line.trim(), "mcp: envelope has neither id nor method");
            }
        }
    }
}

/// Render an MCP `content` array into a JSON value the rest of Forge can
/// store. Pure-text content collapses to a single string; mixed content
/// returns the full structured array so non-text parts aren't lost.
fn collapse_content_value(content: &Value) -> Value {
    let arr = match content.as_array() {
        Some(a) => a,
        None => return content.clone(),
    };
    let all_text = !arr.is_empty()
        && arr
            .iter()
            .all(|p| p["type"].as_str() == Some("text") && p["text"].is_string());
    if all_text {
        let joined: String = arr
            .iter()
            .map(|p| p["text"].as_str().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("");
        Value::String(joined)
    } else {
        Value::Array(arr.clone())
    }
}

fn collapse_content(content: &Value) -> String {
    match collapse_content_value(content) {
        Value::String(s) => s,
        v => v.to_string(),
    }
}
