use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Parser, Subcommand};

use forge::{print_chain, run_agent};
use forge_core::agent::FakeAgent;
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
        /// Path to a run spec. Ignored in v0.0.2; uses FakeAgent.
        spec: Option<String>,
    },
    /// Walk a recorded run from its head and print the chain.
    Replay {
        /// Run head hash (or any prefix that uniquely identifies one).
        head: String,
    },
    /// List recorded runs.
    Runs,
    /// Fork an existing run from a specific step.
    Fork {
        run: String,
        #[arg(long)]
        at: String,
    },
    /// Diff two runs and surface where they diverged.
    Diff { a: String, b: String },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let storage = SledStorage::open(&cli.db)?;

    match cli.cmd {
        Cmd::Run { spec } => {
            if let Some(spec) = spec {
                tracing::warn!(?spec, "spec parsing not yet implemented; using FakeAgent");
            }
            let mut agent = FakeAgent::scripted();
            let chain = run_agent(&mut agent, &storage).await?;
            let head = chain
                .last()
                .cloned()
                .expect("agent emitted at least one step");
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
        Cmd::Fork { run, at } => {
            tracing::info!(?run, ?at, "fork: not yet implemented");
        }
        Cmd::Diff { a, b } => {
            tracing::info!(?a, ?b, "diff: not yet implemented");
        }
    }
    Ok(())
}

/// Resolve a user-provided head ref: either a full hash or a unique prefix.
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
