use clap::{Parser, Subcommand};

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
        /// Path to a run spec (TOML/YAML).
        spec: String,
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
            tracing::info!(?spec, "run: not yet implemented");
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
