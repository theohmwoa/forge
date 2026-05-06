//! Standalone MCP server fixture.
//!
//! Exposes three tools: `echo`, `reverse`, and `word_count`. Useful as a
//! zero-dependency target for `forge run --mcp-server` demos and for
//! integration tests that need a real subprocess instead of the in-process
//! duplex-pipe fake.
//!
//! Build with `cargo build --example echo_server` and the binary lands at
//! `target/debug/examples/echo_server`.
//!
//! Wire format: newline-delimited JSON-RPC 2.0 over stdio.

use std::io::{BufRead, Write};

use serde_json::{json, Value};

fn handle_request(method: &str, params: &Value) -> Result<Value, (i64, String)> {
    match method {
        "initialize" => Ok(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": {} },
            "serverInfo": {
                "name": "forge-mcp-echo-server",
                "version": env!("CARGO_PKG_VERSION"),
            }
        })),
        "tools/list" => Ok(json!({
            "tools": [
                {
                    "name": "echo",
                    "description": "Returns the input string unchanged.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "message": { "type": "string" }
                        },
                        "required": ["message"]
                    }
                },
                {
                    "name": "reverse",
                    "description": "Returns the input string reversed.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "message": { "type": "string" }
                        },
                        "required": ["message"]
                    }
                },
                {
                    "name": "word_count",
                    "description": "Returns the whitespace-separated word count of the input.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": { "type": "string" }
                        },
                        "required": ["text"]
                    }
                }
            ]
        })),
        "tools/call" => {
            let name = params["name"].as_str().unwrap_or("");
            let args = &params["arguments"];
            let text = match name {
                "echo" => args["message"]
                    .as_str()
                    .ok_or((-32602, "echo: missing string `message`".into()))?
                    .to_string(),
                "reverse" => {
                    let s = args["message"]
                        .as_str()
                        .ok_or((-32602, "reverse: missing string `message`".into()))?;
                    s.chars().rev().collect::<String>()
                }
                "word_count" => {
                    let s = args["text"]
                        .as_str()
                        .ok_or((-32602, "word_count: missing string `text`".into()))?;
                    let n = s.split_whitespace().count();
                    format!("{n}")
                }
                other => return Err((-32601, format!("unknown tool: {other}"))),
            };
            Ok(json!({
                "content": [{ "type": "text", "text": text }],
                "isError": false
            }))
        }
        _ => Err((-32601, format!("unknown method: {method}"))),
    }
}

fn main() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) if !l.trim().is_empty() => l,
            Ok(_) => continue,
            Err(_) => break,
        };
        let env: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let id = env.get("id").cloned();
        let method = env["method"].as_str().unwrap_or("").to_string();
        let params = env.get("params").cloned().unwrap_or(Value::Null);

        // Notifications (no `id`) get no response.
        if id.is_none() {
            continue;
        }

        let response = match handle_request(&method, &params) {
            Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
            Err((code, message)) => json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": { "code": code, "message": message }
            }),
        };
        let mut payload = serde_json::to_vec(&response).unwrap();
        payload.push(b'\n');
        if out.write_all(&payload).is_err() {
            break;
        }
        if out.flush().is_err() {
            break;
        }
    }
}
