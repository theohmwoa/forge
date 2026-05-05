use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Parser, Subcommand, ValueEnum};

use forge::{diff_chains, fork_chain, print_chain, render_diff, run_agent};
use forge_anthropic::{AnthropicAgent, AnthropicConfig};
use forge_core::agent::{Agent, FakeAgent};
use forge_core::NodeHash;
use forge_storage::{RunMeta, SledStorage};

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
    },
    /// Walk a recorded run from its head and print the chain.
    Replay { head: String },
    /// List recorded runs.
    Runs,
    /// Fork a recorded run at a specific step, rewriting its text content.
    Fork {
        /// Run head hash (or unique prefix).
        run: String,
        /// Step in the chain to rewrite (hash or unique prefix).
        #[arg(long)]
        at: String,
        /// New text content for the step (Prompt / Message kinds only in v0).
        #[arg(long)]
        rewrite_text: String,
    },
    /// Diff two recorded runs by walking their chains pairwise.
    Diff { a: String, b: String },
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum AgentKind {
    Fake,
    Anthropic,
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
        } => {
            let mut agent: Box<dyn Agent> = match agent {
                AgentKind::Fake => Box::new(FakeAgent::scripted()),
                AgentKind::Anthropic => {
                    let prompt = prompt.ok_or_else(|| {
                        anyhow::anyhow!("--prompt is required for --agent anthropic")
                    })?;
                    let cfg = AnthropicConfig::from_env(model)?;
                    Box::new(AnthropicAgent::new(cfg, prompt))
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
        Cmd::Fork {
            run,
            at,
            rewrite_text,
        } => {
            let head = resolve_head(&storage, &run)?;
            let chain = storage.chain_to(&head)?;
            let new_chain = fork_chain(&storage, &chain, &at, &rewrite_text).await?;
            let new_head = new_chain.last().cloned().unwrap();
            let root = new_chain.first().cloned().unwrap();
            storage.record_run(&RunMeta {
                head: new_head.clone(),
                root,
                recorded_at_ms: now_ms(),
            })?;
            println!("forked from {} at step {}", short(&head.0), at);
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
