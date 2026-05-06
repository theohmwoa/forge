//! Integration tests for tool-keyed model routing — the `--after-tool` feature.
//!
//! These exercise the public `run_routed` orchestrator end-to-end with a
//! synthetic multi-tool conversation, asserting that the model selected for
//! each cycle is determined by the *most recent* `ToolCall` name. They also
//! verify the canonical demo set of tools (list_files, count_lines, read_file,
//! search_text) actually executes correctly against a real filesystem.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use forge::{parse_route_rule, run_routed, RouteTarget};
use forge_core::agent::Agent;
use forge_core::tool::{CountLines, ListFiles, ReadFile, SearchText, Tool};
use forge_core::{NodeHash, Step, StepKind};
use forge_storage::{MemoryStorage, Storage};
use serde_json::{json, Value};

// ---- Test fixtures --------------------------------------------------------

/// Emits a fixed sequence of step kinds and is exhausted after the last one.
/// One `SeqAgent` represents one routed cycle; tests build a `VecDeque` of
/// them and the factory pops the next one each cycle.
struct SeqAgent {
    pending: VecDeque<StepKind>,
}

impl SeqAgent {
    fn new(kinds: Vec<StepKind>) -> Self {
        Self {
            pending: kinds.into(),
        }
    }
}

#[async_trait]
impl Agent for SeqAgent {
    async fn next_step(&mut self, _parent: Option<NodeHash>) -> Option<StepKind> {
        self.pending.pop_front()
    }
}

fn msg(content: &str) -> StepKind {
    StepKind::Message {
        role: "assistant".into(),
        content: content.into(),
    }
}

fn tool_call(name: &str, input: Value) -> StepKind {
    StepKind::ToolCall {
        call_id: format!("call-{name}"),
        name: name.into(),
        input,
    }
}

fn tool_result(name: &str, output: Value) -> StepKind {
    StepKind::ToolResult {
        call_id: format!("call-{name}"),
        output,
    }
}

// ---- Routing tests --------------------------------------------------------

#[tokio::test]
async fn router_swaps_models_per_tool_in_a_multi_tool_conversation() {
    // Scenario: an agent that lists files, counts lines on the chosen file,
    // reads it, then synthesizes a final answer. Three tool boundaries means
    // three model handoffs. The default model is "pro" (synthesis); the cheap
    // model "lite" handles every cycle that comes right after a mechanical
    // tool. read_file has no rule → default applies (smart model synthesizes).
    let storage = MemoryStorage::new();
    let default = RouteTarget::new("fake", "pro");
    let mut rules = HashMap::new();
    rules.insert("list_files".into(), RouteTarget::new("fake", "lite"));
    rules.insert("count_lines".into(), RouteTarget::new("fake", "lite"));
    // No rule for read_file → falls back to default (pro).

    let cycles: VecDeque<Box<dyn Agent>> = VecDeque::from(vec![
        // Cycle 0 — initial planning + first tool call
        Box::new(SeqAgent::new(vec![
            msg("I'll start by listing the files."),
            tool_call("list_files", json!({ "path": "." })),
            tool_result("list_files", json!([{"name": "a.txt"}, {"name": "b.txt"}])),
        ])) as Box<dyn Agent>,
        // Cycle 1 — picks a file (mechanical), should run on lite
        Box::new(SeqAgent::new(vec![
            tool_call("count_lines", json!({ "path": "a.txt" })),
            tool_result("count_lines", json!({ "lines": 42 })),
        ])),
        // Cycle 2 — read content (still mechanical pick), lite
        Box::new(SeqAgent::new(vec![
            tool_call("read_file", json!({ "path": "a.txt" })),
            tool_result("read_file", json!({ "content": "hello world" })),
        ])),
        // Cycle 3 — synthesis (after read_file, no rule → default = pro)
        Box::new(SeqAgent::new(vec![msg(
            "Final answer: a.txt contains a greeting.",
        )])),
    ]);
    let cycles = Arc::new(Mutex::new(cycles));
    let targets_log = Arc::new(Mutex::new(Vec::<RouteTarget>::new()));

    let cycles_for_factory = Arc::clone(&cycles);
    let targets_for_factory = Arc::clone(&targets_log);
    let factory = move |t: &RouteTarget, _prefix: Option<&[Step]>| {
        targets_for_factory.lock().unwrap().push(t.clone());
        Ok(cycles_for_factory
            .lock()
            .unwrap()
            .pop_front()
            .expect("router asked for more cycles than the test scripted"))
    };

    let result = run_routed(&storage, default.clone(), &rules, 16, factory)
        .await
        .unwrap();

    // 4 cycles, 4 model selections.
    let targets = targets_log.lock().unwrap().clone();
    assert_eq!(targets.len(), 4, "expected 4 routed cycles");

    // Cycle 0: default (no tool has run yet).
    assert_eq!(targets[0], default, "cycle 0 must use the default model");

    // Cycle 1: previous tool was list_files → lite.
    assert_eq!(
        targets[1].model, "lite",
        "after list_files, must route to lite"
    );

    // Cycle 2: previous tool was count_lines → lite.
    assert_eq!(
        targets[2].model, "lite",
        "after count_lines, must route to lite"
    );

    // Cycle 3: previous tool was read_file (no rule) → default.
    assert_eq!(
        targets[3], default,
        "after read_file (no rule), must fall back to default"
    );

    // RoutedCycle.last_tool records the tool trailing this cycle's emissions
    // — i.e. what the NEXT cycle's routing decision will key on. So:
    //   cycles[0] ends with list_files.tool_result  → drives cycle 1's pick
    //   cycles[1] ends with count_lines.tool_result → drives cycle 2's pick
    //   cycles[2] ends with read_file.tool_result   → drives cycle 3's pick
    //   cycles[3] emits only a Message; the last tool in the prefix is still read_file
    assert_eq!(result.cycles.len(), 4);
    assert_eq!(result.cycles[0].last_tool.as_deref(), Some("list_files"));
    assert_eq!(result.cycles[1].last_tool.as_deref(), Some("count_lines"));
    assert_eq!(result.cycles[2].last_tool.as_deref(), Some("read_file"));
    assert_eq!(result.cycles[3].last_tool.as_deref(), Some("read_file"));

    // Final step on the chain is the synthesis message — confirms run_routed
    // terminated cleanly on the assistant message and didn't truncate.
    let last_hash = result.chain.last().unwrap();
    let last = storage.get(last_hash).await.unwrap().unwrap();
    match last.kind {
        StepKind::Message { ref content, .. } => {
            assert!(
                content.starts_with("Final answer"),
                "wrong final message: {content}"
            )
        }
        ref other => panic!("expected Message at end, got {other:?}"),
    }
}

#[tokio::test]
async fn router_falls_back_to_default_when_tool_has_no_rule() {
    // One tool, no rule for it → cycle 1 must use the default model.
    let storage = MemoryStorage::new();
    let default = RouteTarget::new("fake", "pro");
    let rules: HashMap<String, RouteTarget> = HashMap::new(); // intentionally empty

    let cycles: VecDeque<Box<dyn Agent>> = VecDeque::from(vec![
        Box::new(SeqAgent::new(vec![
            msg("Calling search_text."),
            tool_call("search_text", json!({ "path": "x", "query": "y" })),
            tool_result("search_text", json!({ "matches": [], "count": 0 })),
        ])) as Box<dyn Agent>,
        Box::new(SeqAgent::new(vec![msg("Nothing matched.")])),
    ]);
    let cycles = Arc::new(Mutex::new(cycles));
    let targets_log = Arc::new(Mutex::new(Vec::<RouteTarget>::new()));

    let c = Arc::clone(&cycles);
    let t = Arc::clone(&targets_log);
    let factory = move |target: &RouteTarget, _: Option<&[Step]>| {
        t.lock().unwrap().push(target.clone());
        Ok(c.lock().unwrap().pop_front().unwrap())
    };

    let _ = run_routed(&storage, default.clone(), &rules, 8, factory)
        .await
        .unwrap();
    let targets = targets_log.lock().unwrap().clone();
    assert_eq!(targets.len(), 2);
    assert_eq!(targets[0], default);
    assert_eq!(
        targets[1], default,
        "no rule for search_text → cycle 1 falls back to default"
    );
}

#[tokio::test]
async fn router_picks_the_most_recent_tool_when_multiple_appear_in_one_cycle() {
    // Edge case: a single cycle emits TWO tool calls back-to-back. The router
    // is documented to key on the *most recent* tool — verify that's what
    // actually happens.
    let storage = MemoryStorage::new();
    let default = RouteTarget::new("fake", "pro");
    let mut rules = HashMap::new();
    rules.insert("list_files".into(), RouteTarget::new("fake", "wrong"));
    rules.insert("count_lines".into(), RouteTarget::new("fake", "right"));

    let cycles: VecDeque<Box<dyn Agent>> = VecDeque::from(vec![
        Box::new(SeqAgent::new(vec![
            msg("Doing two tool calls in one cycle."),
            tool_call("list_files", json!({ "path": "." })),
            tool_result("list_files", json!([])),
            tool_call("count_lines", json!({ "path": "x" })),
            tool_result("count_lines", json!({ "lines": 0 })),
        ])) as Box<dyn Agent>,
        Box::new(SeqAgent::new(vec![msg("done.")])),
    ]);
    let cycles = Arc::new(Mutex::new(cycles));
    let targets_log = Arc::new(Mutex::new(Vec::<RouteTarget>::new()));

    let c = Arc::clone(&cycles);
    let t = Arc::clone(&targets_log);
    let factory = move |target: &RouteTarget, _: Option<&[Step]>| {
        t.lock().unwrap().push(target.clone());
        Ok(c.lock().unwrap().pop_front().unwrap())
    };
    let _ = run_routed(&storage, default, &rules, 8, factory)
        .await
        .unwrap();
    let targets = targets_log.lock().unwrap().clone();
    // Cycle 1 should have routed on the most-recent tool (count_lines).
    assert_eq!(
        targets[1].model, "right",
        "router must pick the most recent tool, not the first"
    );
}

#[test]
fn parse_route_rule_handles_the_demo_rule_set() {
    // The exact rules a real `forge run --after-tool` invocation would parse
    // for this tool set. Confirms the parser supports both `tool:model` and
    // `tool:provider/model` forms used in the README.
    let rules = vec![
        "list_files:fake/lite",
        "count_lines:fake/lite",
        "search_text:lite",
        "read_file:fake/pro",
    ];
    let mut parsed: HashMap<String, RouteTarget> = HashMap::new();
    for raw in rules {
        let (tool, target) = parse_route_rule(raw, "fake").unwrap();
        parsed.insert(tool, target);
    }
    assert_eq!(parsed["list_files"], RouteTarget::new("fake", "lite"));
    assert_eq!(parsed["count_lines"], RouteTarget::new("fake", "lite"));
    // `search_text:lite` — provider falls back to default ("fake").
    assert_eq!(parsed["search_text"], RouteTarget::new("fake", "lite"));
    assert_eq!(parsed["read_file"], RouteTarget::new("fake", "pro"));
}

// ---- Real tool execution against a tempdir --------------------------------

#[tokio::test]
async fn tool_set_runs_against_a_real_filesystem() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), "alpha\nbeta\ngamma\n").unwrap();
    std::fs::write(dir.path().join("b.txt"), "needle in haystack\nbeta\n").unwrap();

    // 1. list_files — both files plus order.
    let listing = ListFiles
        .run(&json!({ "path": dir.path().to_str().unwrap() }))
        .await
        .unwrap();
    let names: Vec<&str> = listing
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["a.txt", "b.txt"]);

    // 2. count_lines on a.txt
    let count = CountLines
        .run(&json!({ "path": dir.path().join("a.txt").to_str().unwrap() }))
        .await
        .unwrap();
    assert_eq!(count["lines"], 3);

    // 3. read_file with truncation
    let content = ReadFile
        .run(&json!({
            "path": dir.path().join("a.txt").to_str().unwrap(),
            "max_bytes": 5
        }))
        .await
        .unwrap();
    assert_eq!(content["content"], "alpha");
    assert_eq!(content["truncated"], true);

    // 4. search_text against b.txt
    let hits = SearchText
        .run(&json!({
            "path": dir.path().join("b.txt").to_str().unwrap(),
            "query": "needle"
        }))
        .await
        .unwrap();
    assert_eq!(hits["count"], 1);
    assert_eq!(hits["matches"][0]["text"], "needle in haystack");
}
