//! End-to-end protocol tests for the MCP client.
//!
//! These spin up a *fake MCP server* in the same process, wired via
//! `tokio::io::duplex` pipes, so we can exercise the full handshake and
//! request/response loop without an external binary or network. The fake
//! implements just enough of the spec (`initialize`, `tools/list`,
//! `tools/call`, `notifications/initialized`) to behave like a real one
//! from the client's perspective.

use std::sync::Arc;

use forge_core::tool::Tool;
use forge_mcp::{mcp_tools_into_dyn, McpClient};
use serde_json::{json, Value};
use tokio::io::{
    AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader,
};

/// Spawn the fake server task. Returns a join handle that completes when the
/// server's read side hits EOF (i.e. the client dropped). The server treats
/// each line as one JSON-RPC envelope and writes one envelope per response,
/// also newline-delimited.
fn spawn_fake_server<R, W>(reader: R, mut writer: W) -> tokio::task::JoinHandle<()>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut buf = BufReader::new(reader);
        let mut line = String::new();
        loop {
            line.clear();
            match buf.read_line(&mut line).await {
                Ok(0) => break,
                Ok(_) => {}
                Err(_) => break,
            }
            if line.trim().is_empty() {
                continue;
            }
            let env: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let method = env["method"].as_str().unwrap_or("");
            let id = env.get("id").cloned();

            // Notifications: no `id`, no response.
            if id.is_none() {
                continue;
            }

            let response = match method {
                "initialize" => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": { "tools": {} },
                        "serverInfo": {
                            "name": "fake-mcp-test-server",
                            "version": "0.1.0"
                        }
                    }
                }),
                "tools/list" => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "tools": [
                            {
                                "name": "echo",
                                "description": "Returns the input unchanged.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "message": { "type": "string" }
                                    },
                                    "required": ["message"]
                                }
                            },
                            {
                                "name": "sum",
                                "description": "Sums two numbers a + b.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "a": { "type": "number" },
                                        "b": { "type": "number" }
                                    },
                                    "required": ["a", "b"]
                                }
                            },
                            {
                                "name": "fail",
                                "description": "Always returns isError=true.",
                                "inputSchema": { "type": "object", "properties": {} }
                            }
                        ]
                    }
                }),
                "tools/call" => {
                    let name = env["params"]["name"].as_str().unwrap_or("");
                    let args = &env["params"]["arguments"];
                    match name {
                        "echo" => {
                            let msg = args["message"].as_str().unwrap_or("");
                            json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "result": {
                                    "content": [{ "type": "text", "text": msg }],
                                    "isError": false
                                }
                            })
                        }
                        "sum" => {
                            let a = args["a"].as_f64().unwrap_or(0.0);
                            let b = args["b"].as_f64().unwrap_or(0.0);
                            json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "result": {
                                    "content": [
                                        { "type": "text", "text": format!("{}", a + b) }
                                    ],
                                    "isError": false
                                }
                            })
                        }
                        "fail" => json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "content": [
                                    { "type": "text", "text": "tool failed: simulated" }
                                ],
                                "isError": true
                            }
                        }),
                        other => json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": {
                                "code": -32601,
                                "message": format!("unknown tool {other}")
                            }
                        }),
                    }
                }
                _ => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {
                        "code": -32601,
                        "message": format!("unknown method {method}")
                    }
                }),
            };
            let mut payload = serde_json::to_vec(&response).unwrap();
            payload.push(b'\n');
            if writer.write_all(&payload).await.is_err() {
                break;
            }
            if writer.flush().await.is_err() {
                break;
            }
        }
    })
}

/// Build a connected `(McpClient, server_handle)` pair using in-memory pipes.
async fn connect_fake() -> (Arc<McpClient>, tokio::task::JoinHandle<()>) {
    // Two duplex channels: client→server and server→client.
    let (client_writer, server_reader) = tokio::io::duplex(64 * 1024);
    let (server_writer, client_reader) = tokio::io::duplex(64 * 1024);

    let server_handle = spawn_fake_server(server_reader, server_writer);
    let client = McpClient::from_streams(client_writer, client_reader, None)
        .await
        .expect("client handshake");
    (client, server_handle)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn handshake_captures_server_info() {
    let (client, _server) = connect_fake().await;
    let info = client.server_info().await;
    assert_eq!(info["serverInfo"]["name"], "fake-mcp-test-server");
    assert_eq!(info["serverInfo"]["version"], "0.1.0");
    assert_eq!(info["protocolVersion"], "2024-11-05");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_tools_returns_descriptors() {
    let (client, _server) = connect_fake().await;
    let tools = client.list_tools().await.unwrap();
    assert_eq!(tools.len(), 3);
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, vec!["echo", "sum", "fail"]);
    // inputSchema round-trips intact.
    let echo = tools.iter().find(|t| t.name == "echo").unwrap();
    assert_eq!(echo.input_schema["type"], "object");
    assert_eq!(echo.input_schema["properties"]["message"]["type"], "string");
    assert_eq!(echo.input_schema["required"][0], "message");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn call_tool_round_trip_text_content() {
    let (client, _server) = connect_fake().await;
    let result = client
        .call_tool("echo", json!({ "message": "hello mcp" }))
        .await
        .unwrap();
    // All-text content collapses into a plain JSON string.
    assert_eq!(result, Value::String("hello mcp".into()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn call_tool_round_trip_arithmetic() {
    let (client, _server) = connect_fake().await;
    let result = client
        .call_tool("sum", json!({ "a": 2, "b": 3 }))
        .await
        .unwrap();
    assert_eq!(result, Value::String("5".into()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn call_tool_propagates_is_error_as_anyhow() {
    let (client, _server) = connect_fake().await;
    let err = client
        .call_tool("fail", json!({}))
        .await
        .expect_err("fail tool should produce error");
    let s = err.to_string();
    assert!(
        s.contains("simulated"),
        "error must surface server's reason, got: {s}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn call_tool_unknown_name_surfaces_rpc_error() {
    let (client, _server) = connect_fake().await;
    let err = client
        .call_tool("not_a_tool", json!({}))
        .await
        .expect_err("unknown tool should error");
    assert!(err.to_string().contains("unknown tool"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_requests_dispatch_to_correct_pending_oneshot() {
    // Two simultaneous calls — they must NOT interleave responses.
    // Reader-task dispatch is keyed by id, so this is the right test.
    let (client, _server) = connect_fake().await;
    let c1 = Arc::clone(&client);
    let c2 = Arc::clone(&client);

    let f1 = tokio::spawn(async move {
        c1.call_tool("echo", json!({ "message": "one" })).await
    });
    let f2 = tokio::spawn(async move {
        c2.call_tool("sum", json!({ "a": 10, "b": 20 })).await
    });
    let (r1, r2) = tokio::join!(f1, f2);
    assert_eq!(r1.unwrap().unwrap(), Value::String("one".into()));
    assert_eq!(r2.unwrap().unwrap(), Value::String("30".into()));
}

/// Locate the example `echo_server` binary that cargo built next to this
/// test. Returns `None` if it isn't on disk yet — we skip the spawn test in
/// that case rather than fail (e.g. `cargo test -p forge-mcp --lib`).
fn echo_server_binary() -> Option<std::path::PathBuf> {
    // CARGO_BIN_EXE_<example> isn't set for examples; locate by walking up
    // from the test binary's path to `target/<profile>/examples/echo_server`.
    let test_exe = std::env::current_exe().ok()?;
    let mut dir = test_exe.parent()?.to_path_buf(); // .../target/debug/deps
    if dir.ends_with("deps") {
        dir.pop();
    }
    let candidate = dir.join("examples").join("echo_server");
    if candidate.exists() {
        Some(candidate)
    } else {
        None
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawn_real_subprocess_and_call_tool() {
    let bin = match echo_server_binary() {
        Some(p) => p,
        None => {
            eprintln!(
                "skipping: build the example with `cargo build --example echo_server -p forge-mcp` first"
            );
            return;
        }
    };
    let client = McpClient::spawn(bin.to_str().unwrap(), &[])
        .await
        .expect("spawn echo_server");

    let tools = client.list_tools().await.unwrap();
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"echo"));
    assert!(names.contains(&"reverse"));
    assert!(names.contains(&"word_count"));

    // round-trip every tool the example exposes
    let r = client
        .call_tool("reverse", json!({ "message": "abcdef" }))
        .await
        .unwrap();
    assert_eq!(r, Value::String("fedcba".into()));

    let wc = client
        .call_tool("word_count", json!({ "text": "one two three four" }))
        .await
        .unwrap();
    assert_eq!(wc, Value::String("4".into()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mcp_tools_drop_into_forge_tool_trait() {
    // The whole point: an `McpTool` IS a `forge_core::tool::Tool`. Verify
    // every discovered tool can be invoked through the trait without the
    // caller ever knowing it came from MCP.
    let (client, _server) = connect_fake().await;
    let tools: Vec<Arc<dyn Tool>> = mcp_tools_into_dyn(Arc::clone(&client))
        .await
        .unwrap();
    assert_eq!(tools.len(), 3);

    let echo = tools.iter().find(|t| t.name() == "echo").unwrap();
    assert!(echo.description().contains("input"));
    assert_eq!(echo.schema()["properties"]["message"]["type"], "string");

    // And it actually runs through the trait.
    let out = echo
        .run(&json!({ "message": "via Tool trait" }))
        .await
        .unwrap();
    assert_eq!(out, Value::String("via Tool trait".into()));
}
