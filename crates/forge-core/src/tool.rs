//! Tools the agent can call. The trait is provider-agnostic; adapter crates
//! map between provider tool schemas (Anthropic / OpenAI / Gemini) and this
//! shape.

use async_trait::async_trait;
use serde_json::{json, Value};

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// JSON Schema describing the tool's input.
    fn schema(&self) -> Value;
    async fn run(&self, input: &Value) -> anyhow::Result<Value>;
}

/// A built-in arithmetic tool. Useful for examples and tests.
pub struct Calculator;

#[async_trait]
impl Tool for Calculator {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Performs arithmetic on two numbers. Input: { op, a, b } where op is one of add, sub, mul, div."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "op": { "type": "string", "enum": ["add", "sub", "mul", "div"] },
                "a": { "type": "number" },
                "b": { "type": "number" }
            },
            "required": ["op", "a", "b"]
        })
    }

    async fn run(&self, input: &Value) -> anyhow::Result<Value> {
        let op = input["op"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("calculator: missing op"))?;
        let a = input["a"]
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("calculator: a must be a number"))?;
        let b = input["b"]
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("calculator: b must be a number"))?;
        let result = match op {
            "add" => a + b,
            "sub" => a - b,
            "mul" => a * b,
            "div" if b == 0.0 => anyhow::bail!("calculator: division by zero"),
            "div" => a / b,
            other => anyhow::bail!("calculator: unknown op {other}"),
        };
        Ok(json!(result))
    }
}

/// Lists files and subdirectories in a directory.
///
/// Output is a JSON array of `{name, is_dir, size_bytes}` objects, sorted by
/// name. This is the canonical "mechanical" tool in the demo set: a model
/// looking at the result is just picking which entry to act on next, which
/// almost never benefits from a heavyweight reasoner.
pub struct ListFiles;

#[async_trait]
impl Tool for ListFiles {
    fn name(&self) -> &str {
        "list_files"
    }

    fn description(&self) -> &str {
        "Lists files and subdirectories in a directory. Input: { path }. \
         Returns array of { name, is_dir, size_bytes }, sorted by name."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "directory path" }
            },
            "required": ["path"]
        })
    }

    async fn run(&self, input: &Value) -> anyhow::Result<Value> {
        let path = input["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("list_files: missing path"))?;
        let mut entries: Vec<Value> = Vec::new();
        for e in std::fs::read_dir(path)
            .map_err(|err| anyhow::anyhow!("list_files: read_dir({path}): {err}"))?
        {
            let e = e?;
            let meta = e.metadata()?;
            entries.push(json!({
                "name": e.file_name().to_string_lossy(),
                "is_dir": meta.is_dir(),
                "size_bytes": meta.len(),
            }));
        }
        entries.sort_by(|a, b| {
            a["name"]
                .as_str()
                .unwrap_or("")
                .cmp(b["name"].as_str().unwrap_or(""))
        });
        Ok(json!(entries))
    }
}

/// Reads file contents as UTF-8 text, with a configurable byte cap to keep
/// tool outputs (and downstream prompt sizes) bounded.
///
/// In the routing demo this is the "synthesis" tool — the next turn after a
/// `read_file` typically wants the strong model, since it has to actually
/// understand and summarize what it just read.
pub struct ReadFile;

#[async_trait]
impl Tool for ReadFile {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Reads a file's contents as UTF-8 text. Input: { path, max_bytes? }. \
         Returns { content, bytes_read, truncated }. Default cap is 65_536 bytes."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "max_bytes": {
                    "type": "integer",
                    "description": "truncate at this many bytes (default 65536)"
                }
            },
            "required": ["path"]
        })
    }

    async fn run(&self, input: &Value) -> anyhow::Result<Value> {
        let path = input["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("read_file: missing path"))?;
        let max = input["max_bytes"].as_u64().unwrap_or(65_536) as usize;
        let bytes =
            std::fs::read(path).map_err(|err| anyhow::anyhow!("read_file: read({path}): {err}"))?;
        let truncated = bytes.len() > max;
        let take = bytes.len().min(max);
        let content = String::from_utf8_lossy(&bytes[..take]).to_string();
        Ok(json!({
            "content": content,
            "bytes_read": take,
            "truncated": truncated,
        }))
    }
}

/// Counts newline-terminated lines and total bytes in a file. Pure mechanical
/// summary — the routing rule for this tool should be the cheapest model.
pub struct CountLines;

#[async_trait]
impl Tool for CountLines {
    fn name(&self) -> &str {
        "count_lines"
    }

    fn description(&self) -> &str {
        "Counts newline-terminated lines in a text file. Input: { path }. \
         Returns { lines, bytes }."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        })
    }

    async fn run(&self, input: &Value) -> anyhow::Result<Value> {
        let path = input["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("count_lines: missing path"))?;
        let bytes = std::fs::read(path)
            .map_err(|err| anyhow::anyhow!("count_lines: read({path}): {err}"))?;
        let lines = bytes.iter().filter(|&&b| b == b'\n').count();
        Ok(json!({ "lines": lines, "bytes": bytes.len() }))
    }
}

/// Searches a text file for a substring (case-sensitive) and returns matching
/// line numbers and contents, capped at `max_matches`.
pub struct SearchText;

#[async_trait]
impl Tool for SearchText {
    fn name(&self) -> &str {
        "search_text"
    }

    fn description(&self) -> &str {
        "Searches a text file for lines containing a substring. \
         Input: { path, query, max_matches? }. Returns { matches: [{line, text}], count }. \
         Default max_matches is 50."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "query": { "type": "string" },
                "max_matches": { "type": "integer" }
            },
            "required": ["path", "query"]
        })
    }

    async fn run(&self, input: &Value) -> anyhow::Result<Value> {
        let path = input["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("search_text: missing path"))?;
        let query = input["query"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("search_text: missing query"))?;
        if query.is_empty() {
            anyhow::bail!("search_text: query must be non-empty");
        }
        let max = input["max_matches"].as_u64().unwrap_or(50) as usize;
        let text = std::fs::read_to_string(path)
            .map_err(|err| anyhow::anyhow!("search_text: read({path}): {err}"))?;
        let mut matches: Vec<Value> = Vec::new();
        for (i, line) in text.lines().enumerate() {
            if line.contains(query) {
                matches.push(json!({ "line": i + 1, "text": line }));
                if matches.len() >= max {
                    break;
                }
            }
        }
        let count = matches.len();
        Ok(json!({ "matches": matches, "count": count }))
    }
}

/// Runs `cargo test` against a Cargo manifest. Returns structured output:
/// exit code, parsed pass/fail counts, plus truncated stdout/stderr.
///
/// This is the centerpiece of the agentic-debugging demo: small models read
/// the truncated stdout, get distracted by the green "ok" line that comes
/// before the red failure block, and propose fixes for the wrong file.
/// Smart models trace the failure backwards to the implementation.
pub struct RunTests;

#[async_trait]
impl Tool for RunTests {
    fn name(&self) -> &str {
        "run_tests"
    }

    fn description(&self) -> &str {
        "Runs `cargo test` on a Rust project. Returns { exit_code, passed, summary, stdout, stderr }. \
         Input: { manifest_path: string, test_filter?: string, timeout_secs?: integer }. \
         Default timeout 90s. Outputs are truncated at ~8KB."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "manifest_path": { "type": "string", "description": "Path to Cargo.toml" },
                "test_filter": { "type": "string", "description": "Optional cargo test name filter" },
                "timeout_secs": { "type": "integer", "description": "Hard timeout in seconds (default 90)" }
            },
            "required": ["manifest_path"]
        })
    }

    async fn run(&self, input: &Value) -> anyhow::Result<Value> {
        let manifest = input["manifest_path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("run_tests: missing manifest_path"))?;
        let filter = input["test_filter"].as_str().unwrap_or("");
        let timeout_secs = input["timeout_secs"].as_u64().unwrap_or(90);

        let mut cmd = tokio::process::Command::new("cargo");
        cmd.args(["test", "--manifest-path", manifest, "--color", "never"]);
        if !filter.is_empty() {
            cmd.arg("--");
            cmd.arg(filter);
        }
        cmd.kill_on_drop(true);

        let fut = cmd.output();
        let result = tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), fut)
            .await
            .map_err(|_| anyhow::anyhow!("run_tests: timed out after {timeout_secs}s"))?
            .map_err(|e| anyhow::anyhow!("run_tests: failed to spawn cargo: {e}"))?;

        let stdout = String::from_utf8_lossy(&result.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&result.stderr).into_owned();
        let summary = parse_test_summary(&stdout);
        let exit_code = result.status.code();
        let passed = result.status.success();

        Ok(json!({
            "exit_code": exit_code,
            "passed": passed,
            "summary": summary,
            "stdout": truncate(&stdout, 8192),
            "stderr": truncate(&stderr, 4096),
        }))
    }
}

/// Pull the canonical "test result: ok. N passed; M failed; K ignored;..."
/// summary line out of cargo test's stdout. Returns a structured object so
/// agents don't have to parse the noise themselves; cheap models that ignore
/// this and re-read stdout often miscount.
fn parse_test_summary(stdout: &str) -> Value {
    let mut total_passed: u64 = 0;
    let mut total_failed: u64 = 0;
    let mut found = false;
    for line in stdout.lines() {
        if line.starts_with("test result:") {
            found = true;
            // Scan for `<number> <keyword>` pairs anywhere in the line:
            // "test result: ok. 3 passed; 0 failed; 0 ignored" reads as
            // "3 passed", "0 failed", "0 ignored" without depending on the
            // chunk's first token (which is "test" / "ok." for the head).
            let words: Vec<&str> = line.split_whitespace().collect();
            for i in 1..words.len() {
                if let Ok(num) = words[i - 1].trim_end_matches(';').parse::<u64>() {
                    let kw = words[i].trim_end_matches(';');
                    match kw {
                        "passed" => total_passed += num,
                        "failed" => total_failed += num,
                        _ => {}
                    }
                }
            }
        }
    }
    if !found {
        return Value::Null;
    }
    json!({ "passed": total_passed, "failed": total_failed })
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let head = &s[..max];
        format!("{head}\n...[truncated {} bytes]", s.len() - max)
    }
}

/// Replaces exactly one occurrence of `old_text` with `new_text` in a file.
/// Refuses if `old_text` is missing or appears more than once — forcing the
/// caller to provide enough context for a unique match. This is the
/// canonical "let the agent autonomously close the loop" tool: agent reads
/// the file, identifies the bug, applies the fix, re-runs tests.
pub struct ApplyPatch;

#[async_trait]
impl Tool for ApplyPatch {
    fn name(&self) -> &str {
        "apply_patch"
    }

    fn description(&self) -> &str {
        "Replace exactly ONE occurrence of old_text with new_text in a file. \
         Errors if old_text is not found OR appears more than once (provide more \
         context to disambiguate). Input: { path, old_text, new_text }. \
         Returns { applied: true, path } on success."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "old_text": { "type": "string", "description": "Verbatim text to replace; must be unique in the file" },
                "new_text": { "type": "string" }
            },
            "required": ["path", "old_text", "new_text"]
        })
    }

    async fn run(&self, input: &Value) -> anyhow::Result<Value> {
        let path = input["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("apply_patch: missing path"))?;
        let old_text = input["old_text"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("apply_patch: missing old_text"))?;
        let new_text = input["new_text"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("apply_patch: missing new_text"))?;
        if old_text.is_empty() {
            anyhow::bail!("apply_patch: old_text must be non-empty");
        }

        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("apply_patch: read({path}): {e}"))?;
        let count = content.matches(old_text).count();
        if count == 0 {
            anyhow::bail!("apply_patch: old_text not found in {path}");
        }
        if count > 1 {
            anyhow::bail!(
                "apply_patch: old_text appears {count} times in {path}; \
                 provide more surrounding context so the match is unique"
            );
        }
        let new_content = content.replacen(old_text, new_text, 1);
        std::fs::write(path, &new_content)
            .map_err(|e| anyhow::anyhow!("apply_patch: write({path}): {e}"))?;
        Ok(json!({
            "applied": true,
            "path": path,
            "occurrences_replaced": 1,
            "new_size_bytes": new_content.len(),
        }))
    }
}

/// Generic shell-command runner with hard timeout and an explicit caller-side
/// allowlist of executables. By default, only `cargo` and `git` are allowed.
/// Anything else returns an error before spawning. The allowlist is a
/// deliberate safety boundary: an LLM-driven agent should NOT be able to
/// shell out to arbitrary binaries.
pub struct RunCommand {
    allowlist: std::collections::HashSet<String>,
}

impl Default for RunCommand {
    fn default() -> Self {
        Self::new()
    }
}

impl RunCommand {
    /// Default allowlist: read-only-ish commands typical of an agent's
    /// debugging loop.
    pub fn new() -> Self {
        let mut allowlist = std::collections::HashSet::new();
        for c in ["cargo", "git", "rustc", "ls", "wc"] {
            allowlist.insert(c.to_string());
        }
        Self { allowlist }
    }

    pub fn with_allowlist(mut self, executables: impl IntoIterator<Item = String>) -> Self {
        self.allowlist = executables.into_iter().collect();
        self
    }
}

#[async_trait]
impl Tool for RunCommand {
    fn name(&self) -> &str {
        "run_command"
    }

    fn description(&self) -> &str {
        "Runs an allowlisted shell command. Input: { command, args?, timeout_secs?, cwd? }. \
         Default allowlist: cargo, git, rustc, ls, wc. Returns { exit_code, stdout, stderr }."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": { "type": "string" },
                "args": { "type": "array", "items": { "type": "string" } },
                "timeout_secs": { "type": "integer" },
                "cwd": { "type": "string" }
            },
            "required": ["command"]
        })
    }

    async fn run(&self, input: &Value) -> anyhow::Result<Value> {
        let command = input["command"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("run_command: missing command"))?;
        if !self.allowlist.contains(command) {
            anyhow::bail!(
                "run_command: `{command}` not in allowlist {:?}",
                self.allowlist
            );
        }
        let args: Vec<String> = input["args"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let timeout_secs = input["timeout_secs"].as_u64().unwrap_or(60);
        let cwd = input["cwd"].as_str();

        let mut cmd = tokio::process::Command::new(command);
        cmd.args(&args).kill_on_drop(true);
        if let Some(d) = cwd {
            cmd.current_dir(d);
        }

        let result =
            tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), cmd.output())
                .await
                .map_err(|_| anyhow::anyhow!("run_command: timed out after {timeout_secs}s"))?
                .map_err(|e| anyhow::anyhow!("run_command: spawn failed: {e}"))?;

        Ok(json!({
            "exit_code": result.status.code(),
            "passed": result.status.success(),
            "stdout": truncate(&String::from_utf8_lossy(&result.stdout), 8192),
            "stderr": truncate(&String::from_utf8_lossy(&result.stderr), 4096),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn calculator_basics() {
        let c = Calculator;
        assert_eq!(
            c.run(&json!({"op": "add", "a": 2, "b": 3})).await.unwrap(),
            json!(5.0)
        );
        assert_eq!(
            c.run(&json!({"op": "mul", "a": 7, "b": 8})).await.unwrap(),
            json!(56.0)
        );
    }

    #[tokio::test]
    async fn calculator_div_by_zero_errors() {
        let err = Calculator
            .run(&json!({"op": "div", "a": 1, "b": 0}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("division"));
    }

    fn schema_is_well_formed(schema: &Value) {
        assert_eq!(schema["type"], "object", "schema not an object: {schema}");
        assert!(
            schema["properties"].is_object(),
            "schema missing properties: {schema}"
        );
        assert!(
            schema["required"].is_array(),
            "schema missing required: {schema}"
        );
        for r in schema["required"].as_array().unwrap() {
            let key = r.as_str().unwrap();
            assert!(
                schema["properties"][key].is_object(),
                "required field {key} not declared in properties: {schema}"
            );
        }
    }

    #[test]
    fn all_tool_schemas_are_well_formed() {
        for s in [
            Calculator.schema(),
            ListFiles.schema(),
            ReadFile.schema(),
            CountLines.schema(),
            SearchText.schema(),
        ] {
            schema_is_well_formed(&s);
        }
    }

    #[tokio::test]
    async fn list_files_returns_sorted_entries() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("b.txt"), "world").unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello").unwrap();
        std::fs::create_dir(dir.path().join("zsub")).unwrap();

        let out = ListFiles
            .run(&json!({ "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap();
        let arr = out.as_array().unwrap();
        let names: Vec<&str> = arr.iter().map(|e| e["name"].as_str().unwrap()).collect();
        assert_eq!(names, vec!["a.txt", "b.txt", "zsub"]);
        assert_eq!(arr[0]["is_dir"], false);
        assert_eq!(arr[2]["is_dir"], true);
        assert_eq!(arr[0]["size_bytes"], 5);
    }

    #[tokio::test]
    async fn list_files_errors_on_missing_path() {
        let err = ListFiles
            .run(&json!({ "path": "/definitely/not/here/forge-tools-test" }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("list_files"));
    }

    #[tokio::test]
    async fn read_file_truncates_at_max_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("big.txt");
        std::fs::write(&p, "abcdefghij").unwrap();

        let full = ReadFile
            .run(&json!({ "path": p.to_str().unwrap() }))
            .await
            .unwrap();
        assert_eq!(full["content"], "abcdefghij");
        assert_eq!(full["bytes_read"], 10);
        assert_eq!(full["truncated"], false);

        let partial = ReadFile
            .run(&json!({ "path": p.to_str().unwrap(), "max_bytes": 4 }))
            .await
            .unwrap();
        assert_eq!(partial["content"], "abcd");
        assert_eq!(partial["bytes_read"], 4);
        assert_eq!(partial["truncated"], true);
    }

    #[tokio::test]
    async fn count_lines_counts_newlines() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.txt");
        std::fs::write(&p, "one\ntwo\nthree\n").unwrap();

        let out = CountLines
            .run(&json!({ "path": p.to_str().unwrap() }))
            .await
            .unwrap();
        assert_eq!(out["lines"], 3);
        assert_eq!(out["bytes"], 14);
    }

    #[tokio::test]
    async fn count_lines_handles_no_trailing_newline() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.txt");
        std::fs::write(&p, "no-newline").unwrap();

        let out = CountLines
            .run(&json!({ "path": p.to_str().unwrap() }))
            .await
            .unwrap();
        // Counts \n only — single line with no newline yields 0 by design.
        assert_eq!(out["lines"], 0);
        assert_eq!(out["bytes"], 10);
    }

    #[tokio::test]
    async fn search_text_returns_matches_with_line_numbers() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("haystack.txt");
        std::fs::write(&p, "alpha\nbeta needle\ngamma\nneedle delta\n").unwrap();

        let out = SearchText
            .run(&json!({ "path": p.to_str().unwrap(), "query": "needle" }))
            .await
            .unwrap();
        assert_eq!(out["count"], 2);
        let matches = out["matches"].as_array().unwrap();
        assert_eq!(matches[0]["line"], 2);
        assert_eq!(matches[0]["text"], "beta needle");
        assert_eq!(matches[1]["line"], 4);
        assert_eq!(matches[1]["text"], "needle delta");
    }

    #[tokio::test]
    async fn search_text_respects_max_matches() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("h.txt");
        std::fs::write(&p, "x\nx\nx\nx\nx\n").unwrap();

        let out = SearchText
            .run(&json!({
                "path": p.to_str().unwrap(),
                "query": "x",
                "max_matches": 2
            }))
            .await
            .unwrap();
        assert_eq!(out["count"], 2);
    }

    #[tokio::test]
    async fn search_text_rejects_empty_query() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("h.txt");
        std::fs::write(&p, "anything").unwrap();
        let err = SearchText
            .run(&json!({ "path": p.to_str().unwrap(), "query": "" }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("query"));
    }

    // -- apply_patch ----------------------------------------------------

    #[tokio::test]
    async fn apply_patch_replaces_unique_match() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f.txt");
        std::fs::write(&p, "fn add(a: i64, b: i64) -> i64 { a + b }").unwrap();
        let res = ApplyPatch
            .run(&json!({
                "path": p.to_str().unwrap(),
                "old_text": "a + b",
                "new_text": "a * b"
            }))
            .await
            .unwrap();
        assert_eq!(res["applied"], true);
        let after = std::fs::read_to_string(&p).unwrap();
        assert_eq!(after, "fn add(a: i64, b: i64) -> i64 { a * b }");
    }

    #[tokio::test]
    async fn apply_patch_rejects_ambiguous_match() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f.txt");
        // "x + y" appears twice — apply_patch must refuse rather than guess.
        std::fs::write(&p, "x + y\nz = (x + y);").unwrap();
        let err = ApplyPatch
            .run(&json!({
                "path": p.to_str().unwrap(),
                "old_text": "x + y",
                "new_text": "x * y"
            }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("appears 2 times"));
        // file is unchanged
        let after = std::fs::read_to_string(&p).unwrap();
        assert_eq!(after, "x + y\nz = (x + y);");
    }

    #[tokio::test]
    async fn apply_patch_rejects_missing_match() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f.txt");
        std::fs::write(&p, "no match here").unwrap();
        let err = ApplyPatch
            .run(&json!({
                "path": p.to_str().unwrap(),
                "old_text": "absent",
                "new_text": "present"
            }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    // -- run_command ----------------------------------------------------

    #[tokio::test]
    async fn run_command_rejects_unallowlisted_executable() {
        let err = RunCommand::new()
            .run(&json!({ "command": "rm", "args": ["-rf", "/"] }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not in allowlist"));
    }

    #[tokio::test]
    async fn run_command_runs_allowed_command_and_returns_structured_output() {
        let out = RunCommand::default()
            .with_allowlist(["echo".to_string()])
            .run(&json!({ "command": "echo", "args": ["hello", "world"] }))
            .await
            .unwrap();
        assert_eq!(out["passed"], true);
        assert!(out["stdout"]
            .as_str()
            .unwrap()
            .trim()
            .ends_with("hello world"));
    }

    // -- run_tests + parse_test_summary --------------------------------

    #[test]
    fn parse_test_summary_extracts_pass_fail_counts() {
        let stdout = "running 2 tests\n\
                      test foo ... ok\n\
                      test bar ... FAILED\n\
                      \n\
                      test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n";
        let s = parse_test_summary(stdout);
        assert_eq!(s["passed"], 1);
        assert_eq!(s["failed"], 1);
    }

    #[test]
    fn parse_test_summary_aggregates_across_multiple_result_lines() {
        // cargo test prints one summary per test target — unit tests, then
        // integration tests, then doc tests. We sum across all of them.
        let stdout = "test result: ok. 3 passed; 0 failed; 0 ignored\n\
                      test result: FAILED. 0 passed; 2 failed; 0 ignored\n\
                      test result: ok. 5 passed; 0 failed; 0 ignored\n";
        let s = parse_test_summary(stdout);
        assert_eq!(s["passed"], 8);
        assert_eq!(s["failed"], 2);
    }

    #[tokio::test]
    async fn run_tests_returns_passed_false_on_compile_error() {
        // Build a one-file fixture that won't compile.
        let dir = tempfile::tempdir().unwrap();
        let manifest = dir.path().join("Cargo.toml");
        std::fs::write(
            &manifest,
            "[package]\nname=\"bad\"\nversion=\"0.0.1\"\nedition=\"2021\"\n[lib]\npath=\"src/lib.rs\"\n",
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(
            dir.path().join("src").join("lib.rs"),
            "pub fn broken() { let x: i64 = \"not an int\"; }",
        )
        .unwrap();

        let result = RunTests
            .run(&json!({
                "manifest_path": manifest.to_str().unwrap(),
                "timeout_secs": 60
            }))
            .await
            .unwrap();
        assert_eq!(result["passed"], false);
        // stderr/stdout will contain compile diagnostics — confirm we
        // captured something useful for the agent.
        let combined = format!(
            "{}{}",
            result["stdout"].as_str().unwrap_or(""),
            result["stderr"].as_str().unwrap_or("")
        );
        assert!(
            combined.contains("error") || combined.contains("expected"),
            "expected compile diagnostics in output, got: {combined}"
        );
    }
}
