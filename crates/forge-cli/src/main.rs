use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Parser, Subcommand, ValueEnum};

use forge::{
    auto_run_tool_after_fork, diff_chains, fork_chain, print_chain, render_diff, run_agent,
    run_agent_from,
};
use forge_anthropic::{AnthropicAgent, AnthropicConfig};
use forge_core::agent::{Agent, FakeAgent};
use forge_core::tool::{Calculator, Tool};
use forge_core::{NodeHash, Step};
use forge_storage::{RunMeta, SledStorage, Storage};

#[derive(Parser)]
#[command(name = "forge", version, about = "Git for agent runs.")]
struct Cli {
    /// Path to the on-disk graph database.
    #[arg(long, default_value = "./forge.db", global = true)]
    db: PathBuf,

    #[command(subcommand)]
    cmd: Cmd,
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
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum AgentKind {
    Fake,
    Anthropic,
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
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let storage = SledStorage::open(&cli.db)?;

    match cli.cmd {
        Cmd::Run {
            agent,
            prompt,
            model,
            tools,
            max_turns,
        } => {
            let tool_set = build_tools(&tools);
            let mut agent: Box<dyn Agent> = match agent {
                AgentKind::Fake => Box::new(FakeAgent::scripted()),
                AgentKind::Anthropic => {
                    let prompt = prompt.ok_or_else(|| {
                        anyhow::anyhow!("--prompt is required for --agent anthropic")
                    })?;
                    let cfg = AnthropicConfig::from_env(model)?;
                    let mut a = AnthropicAgent::new(cfg, prompt).with_tools(tool_set);
                    if let Some(n) = max_turns {
                        a = a.with_max_turns(n);
                    }
                    Box::new(a)
                }
            };
            let chain = run_agent(agent.as_mut(), &storage).await?;
            if chain.is_empty() {
                anyhow::bail!("agent emitted no steps; nothing to record");
            }
            let head = chain.last().cloned().unwrap();
            let root = chain.first().cloned().unwrap();
            storage.record_run(&RunMeta {
                head: head.clone(),
                root,
                recorded_at_ms: now_ms(),
            })?;

            println!("run complete: {} steps", chain.len());
            println!("--- dag ---");
            print_chain(&storage, &chain).await?;
            println!("\nhead: {head}");
            println!("replay: forge replay {}", short(&head.0));
        }
        Cmd::Replay { head } => {
            let head = resolve_head(&storage, &head)?;
            let steps = storage.chain_to(&head)?;
            let chain: Vec<NodeHash> = steps.iter().map(|s| s.id.clone()).collect();
            println!("replay: {} steps from head {}", chain.len(), head);
            println!("--- dag ---");
            print_chain(&storage, &chain).await?;
        }
        Cmd::Runs => {
            let runs = storage.list_runs()?;
            if runs.is_empty() {
                println!("no recorded runs in {}", cli.db.display());
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
            if !matches!(agent, AgentKind::Anthropic) {
                anyhow::bail!("forge continue currently requires --agent anthropic");
            }
            let head = resolve_head(&storage, &run)?;
            let prefix_steps: Vec<Step> = storage.chain_to(&head)?;
            let cfg = AnthropicConfig::from_env(model)?;
            let mut cont =
                AnthropicAgent::continuing(cfg, &prefix_steps).with_tools(build_tools(&tools));
            if let Some(n) = max_turns {
                cont = cont.with_max_turns(n);
            }
            let appended = run_agent_from(Some(head.clone()), &mut cont, &storage).await?;
            if appended.is_empty() {
                anyhow::bail!("agent emitted no new steps; nothing to record");
            }
            let new_head = appended.last().cloned().unwrap();
            let root = prefix_steps[0].id.clone();
            storage.record_run(&RunMeta {
                head: new_head.clone(),
                root,
                recorded_at_ms: now_ms(),
            })?;
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
            print_chain(&storage, &full_chain).await?;
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
            let head = resolve_head(&storage, &run)?;
            let chain = storage.chain_to(&head)?;
            let mut new_chain = fork_chain(&storage, &chain, &at, &rewrite).await?;

            // If we rewrote a tool_call AND the user asked us to continue,
            // execute the tool locally to materialize the fresh tool_result
            // before handing off to the API.
            let tool_set = build_tools(&tools);
            if r#continue {
                auto_run_tool_after_fork(&storage, &mut new_chain, &tool_set).await?;
            }
            let prefix_len_before_continue = new_chain.len();

            if r#continue {
                if !matches!(agent, AgentKind::Anthropic) {
                    anyhow::bail!("--continue currently requires --agent anthropic");
                }
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
                let cfg = AnthropicConfig::from_env(model)?;
                let mut cont = AnthropicAgent::continuing(cfg, &prefix_steps).with_tools(tool_set);
                if let Some(n) = max_turns {
                    cont = cont.with_max_turns(n);
                }
                let appended = run_agent_from(Some(last_hash), &mut cont, &storage).await?;
                new_chain.extend(appended);
            }

            let new_head = new_chain.last().cloned().unwrap();
            let root = new_chain.first().cloned().unwrap();
            storage.record_run(&RunMeta {
                head: new_head.clone(),
                root,
                recorded_at_ms: now_ms(),
            })?;
            println!("forked from {} at step {}", short(&head.0), at);
            if r#continue {
                println!(
                    "continued: +{} step(s)",
                    new_chain.len() - prefix_len_before_continue
                );
            }
            println!("new head: {new_head}");
            println!("--- dag ---");
            print_chain(&storage, &new_chain).await?;
        }
        Cmd::Diff { a, b } => {
            let head_a = resolve_head(&storage, &a)?;
            let head_b = resolve_head(&storage, &b)?;
            let chain_a = storage.chain_to(&head_a)?;
            let chain_b = storage.chain_to(&head_b)?;
            let result = diff_chains(&chain_a, &chain_b);
            println!("A: {}  ({} steps)", short(&head_a.0), chain_a.len());
            println!("B: {}  ({} steps)", short(&head_b.0), chain_b.len());
            print!("{}", render_diff(&result));
        }
    }
    Ok(())
}

fn resolve_head(storage: &SledStorage, query: &str) -> anyhow::Result<NodeHash> {
    let runs = storage.list_runs()?;
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
