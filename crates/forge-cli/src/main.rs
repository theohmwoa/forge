use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Parser, Subcommand, ValueEnum};

mod tui;
mod web;

use forge::{
    auto_run_tool_after_fork, diff_chains, fork_chain, print_chain, render_diff, run_agent,
    run_agent_from,
};
use forge_anthropic::{AnthropicAgent, AnthropicConfig};
use forge_core::agent::{Agent, FakeAgent};
use forge_core::tool::{Calculator, Tool};
use forge_core::{NodeHash, Step};
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
    },
    /// Walk a recorded run from its head and print the chain.
    Replay { head: String },
    /// List recorded runs.
    Runs,
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
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },
    /// Serve a local web viewer over the run graph. Single embedded HTML page
    /// + JSON API; localhost-only by default.
    Web {
        #[arg(long, default_value_t = 7879)]
        port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum AgentKind {
    Fake,
    Anthropic,
    Openai,
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
            anyhow::bail!("--continue does not support --agent fake; use anthropic or openai")
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
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ToolKind {
    Calculator,
}

fn build_tools(kinds: &[ToolKind]) -> Vec<Arc<dyn Tool>> {
    kinds
        .iter()
        .map(|k| match k {
            ToolKind::Calculator => Arc::new(Calculator) as Arc<dyn Tool>,
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
        } => {
            let tool_set = build_tools(&tools);
            let mut agent = build_fresh_agent(agent, prompt, model, tool_set, max_turns, stream)?;
            let chain = run_agent(agent.as_mut(), &*storage).await?;
            if chain.is_empty() {
                anyhow::bail!("agent emitted no steps; nothing to record");
            }
            let head = chain.last().cloned().unwrap();
            let root = chain.first().cloned().unwrap();
            storage
                .record_run(&RunMeta {
                    head: head.clone(),
                    root,
                    recorded_at_ms: now_ms(),
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
        Cmd::Runs => {
            let runs = storage.list_runs().await?;
            if runs.is_empty() {
                println!("no recorded runs in {}", cli.db);
            } else {
                for r in runs {
                    println!(
                        "{}  recorded_at={}  root={}",
                        short(&r.head.0),
                        r.recorded_at_ms,
                        short(&r.root.0),
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
        Cmd::Serve { port, host } => {
            let state = Arc::new(forge_recorder::RecorderState {
                storage: Arc::clone(&storage),
                client: reqwest::Client::new(),
            });
            let app = forge_recorder::router(state);
            let addr = format!("{host}:{port}");
            let listener = tokio::net::TcpListener::bind(&addr).await?;
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
        Cmd::Web { port, host } => {
            let app = web::router(web::WebState {
                storage: Arc::clone(&storage),
            });
            let addr = format!("{host}:{port}");
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            println!("forge web viewer at http://{addr}");
            println!("(shift-click two runs to diff them)");
            println!();
            println!("press ctrl-c to stop");
            axum::serve(listener, app).await?;
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
