use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Parser, Subcommand, ValueEnum};

mod tui;
mod web;

use forge::{
    audit_runs, auto_run_tool_after_fork, bisect_chains, default_bisect_check, diff_chains,
    fork_chain, parse_route_rule, print_chain, render_audit, render_bisect, render_diff,
    render_routed, run_agent, run_agent_from, run_routed, RouteTarget,
};
use forge_anthropic::{AnthropicAgent, AnthropicConfig};
use forge_core::agent::{Agent, FakeAgent};
use forge_core::tool::{
    ApplyPatch, Calculator, CountLines, ListFiles, ReadFile, RunCommand, RunTests, SearchText, Tool,
};
use forge_core::{NodeHash, Step};
use forge_gemini::{GeminiAgent, GeminiConfig};
use forge_mcp::{mcp_tools_into_dyn, McpClient};
use forge_openai::{OpenAIAgent, OpenAIConfig};
use forge_storage::{PostgresStorage, RunMeta, SledStorage, Storage};

#[derive(Parser)]
#[command(name = "forge", version, about = "Git for agent runs.")]
struct Cli {
    /// Storage URL. A filesystem path opens sled; a `postgres://` URL opens
    /// the Postgres backend.
    #[arg(long, default_value = "./forge.db", global = true)]
    db: String,

    #[command(subcommand)]
    cmd: Cmd,
}

async fn open_storage(url: &str) -> anyhow::Result<std::sync::Arc<dyn Storage>> {
    if url.starts_with("postgres://") || url.starts_with("postgresql://") {
        let store = PostgresStorage::connect(url).await?;
        Ok(std::sync::Arc::new(store))
    } else {
        let store = SledStorage::open(url)?;
        Ok(std::sync::Arc::new(store))
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Run an agent and record every step into the graph.
    Run {
        #[arg(long, value_enum, default_value_t = AgentKind::Fake)]
        agent: AgentKind,
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long, default_value = "claude-sonnet-4-6")]
        model: String,
        /// Tools to expose to the agent.
        #[arg(long, value_enum, value_delimiter = ',')]
        tools: Vec<ToolKind>,
        /// Cap the number of API turns. Useful for handing off to another
        /// model via `forge continue`.
        #[arg(long)]
        max_turns: Option<usize>,
        /// Stream text deltas to stderr as they arrive (Anthropic only).
        #[arg(long, default_value_t = false)]
        stream: bool,
        /// Free-form tag to attach to this run for later filtering.
        #[arg(long)]
        tag: Option<String>,
        /// Tool-keyed routing rule: `tool:[provider/]model`. Repeatable.
        /// When the previous step is a `tool_result` for the named tool,
        /// the next agent turn uses this model. Provider falls back to
        /// `--agent`. Examples:
        ///   --after-tool web_search:claude-haiku-4-5-20251001
        ///   --after-tool execute_sql:openai/ft:gpt-4o-mini:org:sql-v3
        ///   --after-tool screenshot:gemini/gemini-2.5-flash
        #[arg(long, value_name = "TOOL:MODEL")]
        after_tool: Vec<String>,
        /// Spawn one or more MCP servers and register every tool they
        /// expose. Repeatable. Quote the whole command (with args) — Forge
        /// splits on whitespace. Example:
        ///   --mcp-server "uvx mcp-server-fetch"
        ///   --mcp-server "npx -y @modelcontextprotocol/server-filesystem ."
        #[arg(long, value_name = "CMD")]
        mcp_server: Vec<String>,
    },
    /// Walk a recorded run from its head and print the chain.
    Replay { head: String },
    /// List recorded runs (optionally filtered by tag).
    Runs {
        #[arg(long)]
        tag: Option<String>,
    },
    /// Continue an existing run with an agent (potentially a different
    /// model). Records a new run head whose root matches the original.
    Continue {
        run: String,
        #[arg(long, value_enum, default_value_t = AgentKind::Anthropic)]
        agent: AgentKind,
        #[arg(long, default_value = "claude-sonnet-4-6")]
        model: String,
        #[arg(long, value_enum, value_delimiter = ',')]
        tools: Vec<ToolKind>,
        #[arg(long)]
        max_turns: Option<usize>,
    },
    /// Fork a recorded run at a specific step, replacing its content.
    /// The rewrite is interpreted as text for prompt/message steps and as
    /// JSON for tool_call/tool_result steps.
    Fork {
        run: String,
        /// Step in the chain to rewrite (hash or unique prefix).
        #[arg(long)]
        at: String,
        /// New content. Text for prompt/message; JSON for tool_call (input)
        /// and tool_result (output).
        #[arg(long)]
        rewrite: String,
        /// After rewriting, drive a fresh agent forward from the new step.
        /// If the rewritten step is a tool_call, the tool is executed locally
        /// to produce a fresh tool_result before the agent continues.
        #[arg(long, default_value_t = false)]
        r#continue: bool,
        #[arg(long, value_enum, default_value_t = AgentKind::Anthropic)]
        agent: AgentKind,
        #[arg(long, default_value = "claude-sonnet-4-6")]
        model: String,
        #[arg(long, value_enum, value_delimiter = ',')]
        tools: Vec<ToolKind>,
        #[arg(long)]
        max_turns: Option<usize>,
    },
    /// Diff two recorded runs by walking their chains pairwise.
    Diff { a: String, b: String },
    /// `git bisect` for agent runs. Given a `good` run that succeeded and a
    /// `bad` run that failed, walk the divergent tail of `bad` and substitute
    /// each step in turn with the corresponding step from `good`. Report the
    /// first step where the substitution makes the run succeed — that is the
    /// step that caused the failure.
    Bisect {
        /// A run that satisfies the success criterion.
        good: String,
        /// A run that does not.
        bad: String,
        /// Agent to drive the replay forward from each fork.
        #[arg(long, value_enum, default_value_t = AgentKind::Anthropic)]
        agent: AgentKind,
        #[arg(long, default_value = "claude-sonnet-4-6")]
        model: String,
        #[arg(long, value_enum, value_delimiter = ',')]
        tools: Vec<ToolKind>,
        /// Cap on per-trial agent turns. Bisects can be expensive; keep low.
        #[arg(long, default_value_t = 4)]
        max_turns: usize,
        /// Substring the final assistant message must contain for a trial to
        /// be considered "recovered". Defaults to "ends in any non-empty
        /// assistant message" — useful when bad's failure mode is aborting
        /// rather than answering wrong.
        #[arg(long)]
        expect: Option<String>,
    },
    /// Replay each recorded run against a cheaper target model, use a strong
    /// model as judge to compare answers, and report which prompts are safe
    /// to downgrade. Audit-the-audit: candidate trial runs are persisted with
    /// tag `audit-<short>` so you can sanity-check the judge's verdicts in
    /// `forge web`.
    Audit {
        /// Tag filter for which recorded runs to audit. If absent, audits all.
        #[arg(long)]
        tag: Option<String>,
        /// Cap on the number of runs to audit. Auditing makes 2 API calls per
        /// run (target + judge); cap defends your wallet.
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Provider for the cheaper candidate model.
        #[arg(long, value_enum, default_value_t = AgentKind::Anthropic)]
        target_agent: AgentKind,
        /// The cheaper model to test downgrade-readiness against.
        #[arg(long, default_value = "claude-haiku-4-5-20251001")]
        target_model: String,
        /// Provider for the judge.
        #[arg(long, value_enum, default_value_t = AgentKind::Anthropic)]
        judge_agent: AgentKind,
        /// The judge model. Use a strong one — its verdicts decide what gets
        /// downgraded.
        #[arg(long, default_value = "claude-sonnet-4-6")]
        judge_model: String,
        /// Tools to expose to the candidate replay. If empty, the inventory
        /// is auto-detected per run from the recorded ToolCall steps in the
        /// chain — typically what you want.
        #[arg(long, value_enum, value_delimiter = ',')]
        tools: Vec<ToolKind>,
        /// Cap on candidate-replay agent turns. Bump this when auditing
        /// multi-tool agents so the replay can actually finish.
        #[arg(long, default_value_t = 8)]
        target_max_turns: usize,
    },
    /// Open a TUI viewer for a single run, or pass `--diff` to view aligned
    /// diff between two runs.
    View {
        /// Run head hash (or unique prefix). When `--diff` is set, this is run A.
        run: String,
        /// Run B for diff mode.
        #[arg(long)]
        diff: Option<String>,
    },
    /// Run an HTTP recorder proxy. Point your existing agent at this proxy's
    /// URL (e.g. ANTHROPIC_BASE_URL or OPENAI_BASE_URL) and every API call
    /// gets recorded as a Forge run.
    Serve {
        #[arg(long, default_value_t = 7878)]
        port: u16,
        /// Address to bind to. Ignored when `--public` is set.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Bind to 0.0.0.0 (all interfaces). The recorder proxies your API
        /// keys upstream, so do NOT enable this on an untrusted network without
        /// also putting auth in front of it.
        #[arg(long, default_value_t = false)]
        public: bool,
    },
    /// Serve a local web viewer over the run graph. Single embedded HTML page
    /// + JSON API; localhost-only by default.
    Web {
        #[arg(long, default_value_t = 7879)]
        port: u16,
        /// Address to bind to. Ignored when `--public` is set.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Bind to 0.0.0.0 (all interfaces). The web viewer exposes recorded
        /// run contents (prompts, tool calls, model output). Only enable on a
        /// trusted network.
        #[arg(long, default_value_t = false)]
        public: bool,
    },
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum AgentKind {
    Fake,
    Anthropic,
    Openai,
    Gemini,
}

fn build_fresh_agent(
    kind: AgentKind,
    prompt: Option<String>,
    model: String,
    tools: Vec<Arc<dyn Tool>>,
    max_turns: Option<usize>,
    stream: bool,
) -> anyhow::Result<Box<dyn Agent>> {
    match kind {
        AgentKind::Fake => Ok(Box::new(FakeAgent::scripted())),
        AgentKind::Anthropic => {
            let prompt = prompt
                .ok_or_else(|| anyhow::anyhow!("--prompt is required for --agent anthropic"))?;
            let cfg = AnthropicConfig::from_env(model)?;
            let mut a = AnthropicAgent::new(cfg, prompt)
                .with_tools(tools)
                .with_streaming(stream);
            if let Some(n) = max_turns {
                a = a.with_max_turns(n);
            }
            Ok(Box::new(a))
        }
        AgentKind::Openai => {
            let prompt =
                prompt.ok_or_else(|| anyhow::anyhow!("--prompt is required for --agent openai"))?;
            let cfg = OpenAIConfig::from_env(model)?;
            let mut a = OpenAIAgent::new(cfg, prompt)
                .with_tools(tools)
                .with_streaming(stream);
            if let Some(n) = max_turns {
                a = a.with_max_turns(n);
            }
            Ok(Box::new(a))
        }
        AgentKind::Gemini => {
            let prompt =
                prompt.ok_or_else(|| anyhow::anyhow!("--prompt is required for --agent gemini"))?;
            let cfg = GeminiConfig::from_env(model)?;
            let mut a = GeminiAgent::new(cfg, prompt)
                .with_tools(tools)
                .with_streaming(stream);
            if let Some(n) = max_turns {
                a = a.with_max_turns(n);
            }
            Ok(Box::new(a))
        }
    }
}

fn build_continuing_agent(
    kind: AgentKind,
    prefix: &[Step],
    model: String,
    tools: Vec<Arc<dyn Tool>>,
    max_turns: Option<usize>,
) -> anyhow::Result<Box<dyn Agent>> {
    match kind {
        AgentKind::Fake => {
            anyhow::bail!(
                "--continue does not support --agent fake; use anthropic, openai, or gemini"
            )
        }
        AgentKind::Anthropic => {
            let cfg = AnthropicConfig::from_env(model)?;
            let mut a = AnthropicAgent::continuing(cfg, prefix).with_tools(tools);
            if let Some(n) = max_turns {
                a = a.with_max_turns(n);
            }
            Ok(Box::new(a))
        }
        AgentKind::Openai => {
            let cfg = OpenAIConfig::from_env(model)?;
            let mut a = OpenAIAgent::continuing(cfg, prefix).with_tools(tools);
            if let Some(n) = max_turns {
                a = a.with_max_turns(n);
            }
            Ok(Box::new(a))
        }
        AgentKind::Gemini => {
            let cfg = GeminiConfig::from_env(model)?;
            let mut a = GeminiAgent::continuing(cfg, prefix).with_tools(tools);
            if let Some(n) = max_turns {
                a = a.with_max_turns(n);
            }
            Ok(Box::new(a))
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ToolKind {
    Calculator,
    ListFiles,
    ReadFile,
    CountLines,
    SearchText,
    RunTests,
    ApplyPatch,
    RunCommand,
}

fn build_tools(kinds: &[ToolKind]) -> Vec<Arc<dyn Tool>> {
    kinds
        .iter()
        .map(|k| match k {
            ToolKind::Calculator => Arc::new(Calculator) as Arc<dyn Tool>,
            ToolKind::ListFiles => Arc::new(ListFiles) as Arc<dyn Tool>,
            ToolKind::ReadFile => Arc::new(ReadFile) as Arc<dyn Tool>,
            ToolKind::CountLines => Arc::new(CountLines) as Arc<dyn Tool>,
            ToolKind::SearchText => Arc::new(SearchText) as Arc<dyn Tool>,
            ToolKind::RunTests => Arc::new(RunTests) as Arc<dyn Tool>,
            ToolKind::ApplyPatch => Arc::new(ApplyPatch) as Arc<dyn Tool>,
            ToolKind::RunCommand => Arc::new(RunCommand::new()) as Arc<dyn Tool>,
        })
        .collect()
}

/// Resolve tool names from a recorded chain into the matching registered
/// `Tool` implementations. Unknown names are silently dropped (the recorded
/// agent might have used a tool that isn't compiled into this binary).
fn resolve_tools_by_name(names: &[String]) -> Vec<Arc<dyn Tool>> {
    names
        .iter()
        .filter_map(|n| match n.as_str() {
            "calculator" => Some(Arc::new(Calculator) as Arc<dyn Tool>),
            "list_files" => Some(Arc::new(ListFiles) as Arc<dyn Tool>),
            "read_file" => Some(Arc::new(ReadFile) as Arc<dyn Tool>),
            "count_lines" => Some(Arc::new(CountLines) as Arc<dyn Tool>),
            "search_text" => Some(Arc::new(SearchText) as Arc<dyn Tool>),
            "run_tests" => Some(Arc::new(RunTests) as Arc<dyn Tool>),
            "apply_patch" => Some(Arc::new(ApplyPatch) as Arc<dyn Tool>),
            "run_command" => Some(Arc::new(RunCommand::new()) as Arc<dyn Tool>),
            _ => None,
        })
        .collect()
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    if let Err(err) = run().await {
        eprintln!("forge: {err}");
        let mut source = err.source();
        while let Some(s) = source {
            eprintln!("  caused by: {s}");
            source = s.source();
        }
        std::process::exit(1);
    }
}

async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let storage = open_storage(&cli.db).await?;

    match cli.cmd {
        Cmd::Run {
            agent,
            prompt,
            model,
            tools,
            max_turns,
            stream,
            tag,
            after_tool,
            mcp_server,
        } => {
            let mut tool_set = build_tools(&tools);
            // Spawn each MCP server, list its tools, append them to the
            // inventory. Each Arc<McpClient> is held by every tool wrapped
            // from that server, so the connection stays alive until the
            // last tool is dropped.
            let mut _mcp_clients: Vec<std::sync::Arc<McpClient>> = Vec::new();
            for raw in &mcp_server {
                let parts: Vec<&str> = raw.split_whitespace().collect();
                if parts.is_empty() {
                    anyhow::bail!("--mcp-server got an empty command");
                }
                let (cmd, args) = (parts[0], &parts[1..]);
                tracing::info!(cmd = cmd, args = ?args, "spawning mcp server");
                let client = McpClient::spawn(cmd, args)
                    .await
                    .map_err(|e| anyhow::anyhow!("--mcp-server `{raw}` failed to start: {e}"))?;
                let new_tools = mcp_tools_into_dyn(std::sync::Arc::clone(&client))
                    .await
                    .map_err(|e| anyhow::anyhow!("--mcp-server `{raw}` tools/list failed: {e}"))?;
                let names: Vec<String> = new_tools.iter().map(|t| t.name().to_string()).collect();
                println!(
                    "mcp `{cmd}` registered {} tool(s): {}",
                    names.len(),
                    names.join(", ")
                );
                tool_set.extend(new_tools);
                _mcp_clients.push(client);
            }

            let chain = if after_tool.is_empty() {
                // Plain path: single model handles every turn (existing flow).
                let mut agent =
                    build_fresh_agent(agent, prompt, model, tool_set, max_turns, stream)?;
                let chain = run_agent(agent.as_mut(), &*storage).await?;
                if chain.is_empty() {
                    anyhow::bail!("agent emitted no steps; nothing to record");
                }
                chain
            } else {
                // Routed path: per-cycle model swaps based on the rule map.
                let prompt_str = prompt
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("--prompt is required when using --after-tool"))?
                    .clone();
                let default_provider = agent_kind_label(agent);
                let default_target = RouteTarget::new(default_provider, model.clone());

                let mut rules = std::collections::HashMap::new();
                for raw in &after_tool {
                    let (tool, target) = parse_route_rule(raw, default_provider)?;
                    rules.insert(tool, target);
                }

                println!(
                    "routing enabled · default {default_provider}/{model} · {} rule(s)",
                    rules.len()
                );

                let prompt_owned = prompt_str.clone();
                let tools_for_factory = tool_set.clone();
                let factory = move |target: &RouteTarget,
                                    prefix: Option<&[Step]>|
                      -> anyhow::Result<Box<dyn Agent>> {
                    let kind = parse_agent_kind(&target.provider)?;
                    if let Some(prefix) = prefix {
                        build_continuing_agent(
                            kind,
                            prefix,
                            target.model.clone(),
                            tools_for_factory.clone(),
                            Some(1),
                        )
                    } else {
                        build_fresh_agent(
                            kind,
                            Some(prompt_owned.clone()),
                            target.model.clone(),
                            tools_for_factory.clone(),
                            Some(1),
                            false,
                        )
                    }
                };
                let cap = max_turns.unwrap_or(16);
                let result = run_routed(&*storage, default_target, &rules, cap, factory).await?;
                if result.chain.is_empty() {
                    anyhow::bail!("routed agent emitted no steps; nothing to record");
                }
                println!("\n{}", render_routed(&result));
                result.chain
            };

            let head = chain.last().cloned().unwrap();
            let root = chain.first().cloned().unwrap();
            storage
                .record_run(&RunMeta {
                    head: head.clone(),
                    root,
                    recorded_at_ms: now_ms(),
                    tag,
                })
                .await?;

            println!("run complete: {} steps", chain.len());
            println!("--- dag ---");
            print_chain(&*storage, &chain).await?;
            println!("\nhead: {head}");
            println!("replay: forge replay {}", short(&head.0));
        }
        Cmd::Replay { head } => {
            let head = resolve_head(&*storage, &head).await?;
            let steps = storage.chain_to(&head).await?;
            let chain: Vec<NodeHash> = steps.iter().map(|s| s.id.clone()).collect();
            println!("replay: {} steps from head {}", chain.len(), head);
            println!("--- dag ---");
            print_chain(&*storage, &chain).await?;
        }
        Cmd::Runs { tag } => {
            let mut runs = storage.list_runs().await?;
            if let Some(t) = tag.as_deref() {
                runs.retain(|r| r.tag.as_deref() == Some(t));
            }
            if runs.is_empty() {
                println!("no recorded runs in {}", cli.db);
            } else {
                for r in runs {
                    let tag_marker = r
                        .tag
                        .as_ref()
                        .map(|t| format!("  tag={t}"))
                        .unwrap_or_default();
                    println!(
                        "{}  recorded_at={}  root={}{}",
                        short(&r.head.0),
                        r.recorded_at_ms,
                        short(&r.root.0),
                        tag_marker,
                    );
                }
            }
        }
        Cmd::Continue {
            run,
            agent,
            model,
            tools,
            max_turns,
        } => {
            let head = resolve_head(&*storage, &run).await?;
            let prefix_steps: Vec<Step> = storage.chain_to(&head).await?;
            let mut cont = build_continuing_agent(
                agent,
                &prefix_steps,
                model,
                build_tools(&tools),
                max_turns,
            )?;
            let appended = run_agent_from(Some(head.clone()), cont.as_mut(), &*storage).await?;
            if appended.is_empty() {
                anyhow::bail!("agent emitted no new steps; nothing to record");
            }
            let new_head = appended.last().cloned().unwrap();
            let root = prefix_steps[0].id.clone();
            storage
                .record_run(&RunMeta {
                    head: new_head.clone(),
                    root,
                    recorded_at_ms: now_ms(),
                    tag: None,
                })
                .await?;
            let mut full_chain = prefix_steps
                .iter()
                .map(|s| s.id.clone())
                .collect::<Vec<_>>();
            full_chain.extend(appended);
            println!(
                "continued from {}: +{} step(s)",
                short(&head.0),
                full_chain.len() - prefix_steps.len()
            );
            println!("new head: {new_head}");
            println!("--- dag ---");
            print_chain(&*storage, &full_chain).await?;
        }
        Cmd::Fork {
            run,
            at,
            rewrite,
            r#continue,
            agent,
            model,
            tools,
            max_turns,
        } => {
            let head = resolve_head(&*storage, &run).await?;
            let chain = storage.chain_to(&head).await?;
            let mut new_chain = fork_chain(&*storage, &chain, &at, &rewrite).await?;

            // If we rewrote a tool_call AND the user asked us to continue,
            // execute the tool locally to materialize the fresh tool_result
            // before handing off to the API.
            let tool_set = build_tools(&tools);
            if r#continue {
                auto_run_tool_after_fork(&*storage, &mut new_chain, &tool_set).await?;
            }
            let prefix_len_before_continue = new_chain.len();

            if r#continue {
                let prefix_steps: Vec<Step> = {
                    let mut acc = Vec::with_capacity(new_chain.len());
                    for h in &new_chain {
                        let s = storage
                            .get(h)
                            .await?
                            .ok_or_else(|| anyhow::anyhow!("forked step missing: {h}"))?;
                        acc.push(s);
                    }
                    acc
                };
                let last_hash = new_chain.last().cloned().unwrap();
                let mut cont =
                    build_continuing_agent(agent, &prefix_steps, model, tool_set, max_turns)?;
                let appended = run_agent_from(Some(last_hash), cont.as_mut(), &*storage).await?;
                new_chain.extend(appended);
            }

            let new_head = new_chain.last().cloned().unwrap();
            let root = new_chain.first().cloned().unwrap();
            storage
                .record_run(&RunMeta {
                    head: new_head.clone(),
                    root,
                    recorded_at_ms: now_ms(),
                    tag: None,
                })
                .await?;
            println!("forked from {} at step {}", short(&head.0), at);
            if r#continue {
                println!(
                    "continued: +{} step(s)",
                    new_chain.len() - prefix_len_before_continue
                );
            }
            println!("new head: {new_head}");
            println!("--- dag ---");
            print_chain(&*storage, &new_chain).await?;
        }
        Cmd::Bisect {
            good,
            bad,
            agent,
            model,
            tools,
            max_turns,
            expect,
        } => {
            let head_good = resolve_head(&*storage, &good).await?;
            let head_bad = resolve_head(&*storage, &bad).await?;
            let chain_good = storage.chain_to(&head_good).await?;
            let chain_bad = storage.chain_to(&head_bad).await?;

            let tool_set = build_tools(&tools);
            let agent_kind = agent;
            let model_for_agent = model.clone();
            let tools_for_agent = tool_set.clone();
            let max_t = max_turns;
            let build_agent = move |prefix: &[Step]| -> anyhow::Result<Box<dyn Agent>> {
                build_continuing_agent(
                    agent_kind,
                    prefix,
                    model_for_agent.clone(),
                    tools_for_agent.clone(),
                    Some(max_t),
                )
            };

            // Default check vs substring check.
            let check_substring = expect.clone();
            let result = match check_substring {
                Some(needle) => {
                    let needle_owned = needle.clone();
                    let check = move |steps: &[Step]| {
                        steps.last().is_some_and(|s| match &s.kind {
                            forge_core::StepKind::Message { role, content } => {
                                role == "assistant" && content.contains(&needle_owned)
                            }
                            _ => false,
                        })
                    };
                    bisect_chains(&*storage, &chain_good, &chain_bad, build_agent, check).await?
                }
                None => {
                    bisect_chains(
                        &*storage,
                        &chain_good,
                        &chain_bad,
                        build_agent,
                        default_bisect_check,
                    )
                    .await?
                }
            };

            println!(
                "good: {}  ({} steps)",
                short(&head_good.0),
                chain_good.len()
            );
            println!("bad:  {}  ({} steps)", short(&head_bad.0), chain_bad.len());
            print!("{}", render_bisect(&result));
        }
        Cmd::Diff { a, b } => {
            let head_a = resolve_head(&*storage, &a).await?;
            let head_b = resolve_head(&*storage, &b).await?;
            let chain_a = storage.chain_to(&head_a).await?;
            let chain_b = storage.chain_to(&head_b).await?;
            let result = diff_chains(&chain_a, &chain_b);
            println!("A: {}  ({} steps)", short(&head_a.0), chain_a.len());
            println!("B: {}  ({} steps)", short(&head_b.0), chain_b.len());
            print!("{}", render_diff(&result));
        }
        Cmd::Audit {
            tag,
            limit,
            target_agent,
            target_model,
            judge_agent,
            judge_model,
            tools,
            target_max_turns,
        } => {
            // Filter recorded runs by tag (if provided), apply limit.
            let mut runs = storage.list_runs().await?;
            if let Some(t) = tag.as_deref() {
                runs.retain(|r| r.tag.as_deref() == Some(t));
            }
            // Don't audit the trial runs from a previous audit.
            runs.retain(|r| !r.tag.as_deref().is_some_and(|t| t.starts_with("audit-")));
            runs.truncate(limit);
            if runs.is_empty() {
                let filter_msg = tag
                    .as_deref()
                    .map(|t| format!(" matching tag {t:?}"))
                    .unwrap_or_default();
                println!("no recorded runs to audit{filter_msg}.");
                return Ok(());
            }
            let heads: Vec<NodeHash> = runs.iter().map(|r| r.head.clone()).collect();
            let target_label = target_model.replace([':', '/'], "-");

            // Build closures that produce a fresh agent on each call. We
            // capture by clone so each replay is independent.
            //
            // The candidate replay needs the same tool inventory the original
            // run used, otherwise tool-using agents produce empty answers and
            // the judge correctly says "different" — for the wrong reason.
            // We auto-detect from the recorded chain (each call to the factory
            // gets the chain's tool names) but `--tools` overrides the auto.
            let ta = target_agent;
            let tm = target_model.clone();
            let max_turns = target_max_turns;
            let explicit_tools = tools.clone();
            let build_target =
                move |prompt: &str, tool_names: &[String]| -> anyhow::Result<Box<dyn Agent>> {
                    let resolved_tools = if !explicit_tools.is_empty() {
                        build_tools(&explicit_tools)
                    } else {
                        resolve_tools_by_name(tool_names)
                    };
                    build_fresh_agent(
                        ta,
                        Some(prompt.to_string()),
                        tm.clone(),
                        resolved_tools,
                        Some(max_turns),
                        false,
                    )
                };
            let ja = judge_agent;
            let jm = judge_model.clone();
            let build_judge = move |prompt: &str| -> anyhow::Result<Box<dyn Agent>> {
                build_fresh_agent(
                    ja,
                    Some(prompt.to_string()),
                    jm.clone(),
                    Vec::new(),
                    Some(2),
                    false,
                )
            };

            println!(
                "auditing {} run(s) against {} (judge: {})",
                heads.len(),
                target_model,
                judge_model
            );
            println!("each run makes 2 API calls (target + judge); ctrl-c to abort.\n");
            let summary =
                audit_runs(&*storage, &heads, &target_label, build_target, build_judge).await?;
            print!("{}", render_audit(&summary, &target_label));
        }
        Cmd::View { run, diff } => match diff {
            None => {
                let head = resolve_head(&*storage, &run).await?;
                let steps = storage.chain_to(&head).await?;
                tui::view_run(steps, short(&head.0))?;
            }
            Some(b) => {
                let head_a = resolve_head(&*storage, &run).await?;
                let head_b = resolve_head(&*storage, &b).await?;
                let chain_a = storage.chain_to(&head_a).await?;
                let chain_b = storage.chain_to(&head_b).await?;
                tui::view_diff(chain_a, chain_b, short(&head_a.0), short(&head_b.0))?;
            }
        },
        Cmd::Serve { port, host, public } => {
            let state = Arc::new(forge_recorder::RecorderState {
                storage: Arc::clone(&storage),
                client: reqwest::Client::new(),
            });
            let app = forge_recorder::router(state);
            let bind_host = if public { "0.0.0.0" } else { host.as_str() };
            let addr = format!("{bind_host}:{port}");
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            if public {
                println!(
                    "WARNING: forge serve is bound to 0.0.0.0 — anyone reaching this port can\n\
                     proxy through your upstream API keys. Put auth in front of it before exposing."
                );
            }
            println!("forge serve listening on http://{addr}");
            println!();
            println!("for an Anthropic client:");
            println!("  export ANTHROPIC_BASE_URL=http://{addr}");
            println!(
                "  # then call your existing agent normally — every /v1/messages call is recorded"
            );
            println!();
            println!("for an OpenAI client:");
            println!("  export OPENAI_BASE_URL=http://{addr}/v1");
            println!();
            println!("press ctrl-c to stop");
            axum::serve(listener, app).await?;
        }
        Cmd::Web { port, host, public } => {
            let app = web::router(web::WebState {
                storage: Arc::clone(&storage),
            });
            let bind_host = if public { "0.0.0.0" } else { host.as_str() };
            let addr = format!("{bind_host}:{port}");
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            if public {
                println!(
                    "WARNING: forge web is bound to 0.0.0.0 — anyone reaching this port can read\n\
                     your recorded prompts, tool calls, and model outputs. Only do this on a\n\
                     trusted network."
                );
            }
            println!("forge web viewer at http://{addr}");
            println!("(shift-click two runs to diff them)");
            println!();
            println!("press ctrl-c to stop");
            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
            )
            .await?;
        }
    }
    Ok(())
}

async fn resolve_head(storage: &dyn Storage, query: &str) -> anyhow::Result<NodeHash> {
    let runs = storage.list_runs().await?;
    let matches: Vec<_> = runs
        .iter()
        .filter(|r| r.head.0.starts_with(query))
        .collect();
    match matches.len() {
        0 => anyhow::bail!("no recorded run matches {query:?}; try `forge runs`"),
        1 => Ok(matches[0].head.clone()),
        _ => anyhow::bail!(
            "ambiguous head prefix {query:?}: {} runs match",
            matches.len()
        ),
    }
}

fn short(s: &str) -> String {
    s.chars().take(10).collect()
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Stable provider string used in routing rules.
fn agent_kind_label(kind: AgentKind) -> &'static str {
    match kind {
        AgentKind::Fake => "fake",
        AgentKind::Anthropic => "anthropic",
        AgentKind::Openai => "openai",
        AgentKind::Gemini => "gemini",
    }
}

fn parse_agent_kind(label: &str) -> anyhow::Result<AgentKind> {
    match label {
        "fake" => Ok(AgentKind::Fake),
        "anthropic" => Ok(AgentKind::Anthropic),
        "openai" => Ok(AgentKind::Openai),
        "gemini" => Ok(AgentKind::Gemini),
        other => anyhow::bail!(
            "unknown provider in routing rule: {other:?} (expected fake / anthropic / openai / gemini)"
        ),
    }
}
