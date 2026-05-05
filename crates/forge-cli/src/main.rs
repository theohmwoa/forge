use clap::{Parser, Subcommand};

use forge::{print_chain, run_agent};
use forge_core::agent::FakeAgent;
use forge_storage::MemoryStorage;

#[derive(Parser)]
#[command(name = "forge", version, about = "Git for agent runs.")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run an agent and record every step into the graph.
    Run {
        /// Path to a run spec (TOML/YAML). Ignored in v0.0.1; uses FakeAgent.
        spec: Option<String>,
    },
    /// Fork an existing run from a specific step.
    Fork {
        run: String,
        #[arg(long)]
        at: String,
    },
    /// Diff two runs and surface where they diverged.
    Diff { a: String, b: String },
    /// Replay a recorded run deterministically.
    Replay { run: String },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Run { spec } => {
            if let Some(spec) = spec {
                tracing::warn!(?spec, "spec parsing not yet implemented; using FakeAgent");
            }
            let storage = MemoryStorage::new();
            let mut agent = FakeAgent::scripted();
            let chain = run_agent(&mut agent, &storage).await?;
            println!("run complete: {} steps\n--- dag ---", chain.len());
            print_chain(&storage, &chain).await?;
        }
        Cmd::Fork { run, at } => {
            tracing::info!(?run, ?at, "fork: not yet implemented");
        }
        Cmd::Diff { a, b } => {
            tracing::info!(?a, ?b, "diff: not yet implemented");
        }
        Cmd::Replay { run } => {
            tracing::info!(?run, "replay: not yet implemented");
        }
    }
    Ok(())
}
