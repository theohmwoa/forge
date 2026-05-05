# Contributing to forge

## Dev setup

```bash
git clone https://github.com/theohmwoa/forge && cd forge
cargo build
cargo test --all
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
```

Rust 1.75+ (workspace pins via `rust-toolchain.toml`). No nightly features.

## Install from source

```bash
cargo install --path crates/forge-cli --force
forge --help
```

The binary lands in `~/.cargo/bin/forge`. Make sure that's in your `PATH`.

## Layout

```
crates/
├── forge-anthropic   # Anthropic Messages API adapter
├── forge-cli         # the `forge` binary (this is what cargo install builds)
├── forge-core        # step types + Agent / Tool / Matcher traits
├── forge-gemini      # Google Gemini generateContent adapter
├── forge-openai      # OpenAI Chat Completions adapter
├── forge-recorder    # axum proxy: records LLM calls into Forge
└── forge-storage     # Storage trait + Memory / Sled / Postgres backends
```

## Add a new agent backend

1. New crate `forge-<provider>` mirroring `forge-anthropic`.
2. Implement the `Agent` trait (in `forge-core`):
   ```rust
   #[async_trait]
   impl Agent for MyProviderAgent {
       async fn next_step(&mut self, parent: Option<NodeHash>) -> Option<StepKind>;
   }
   ```
3. Provide both `MyProviderAgent::new(config, prompt)` for fresh runs and
   `MyProviderAgent::continuing(config, prefix)` that translates a
   `&[Step]` prefix into the provider's wire format.
4. Wire into `forge-cli/src/main.rs`: add an `AgentKind` variant and
   match arm in `build_fresh_agent` / `build_continuing_agent`.

## Add a new tool

In `forge-core::tool` or your own crate:

```rust
pub struct MyTool;

#[async_trait]
impl Tool for MyTool {
    fn name(&self) -> &str { "mytool" }
    fn description(&self) -> &str { "..." }
    fn schema(&self) -> serde_json::Value { /* JSON Schema */ }
    async fn run(&self, input: &serde_json::Value) -> anyhow::Result<serde_json::Value> {
        // ...
    }
}
```

Add a `ToolKind` variant in `forge-cli/src/main.rs`.

## Add a new storage backend

Implement the `Storage` trait in `forge-storage`:

```rust
#[async_trait]
impl Storage for MyStorage {
    async fn put(&self, step: Step) -> anyhow::Result<NodeHash>;
    async fn get(&self, hash: &NodeHash) -> anyhow::Result<Option<Step>>;
    async fn children(&self, hash: &NodeHash) -> anyhow::Result<Vec<NodeHash>>;
    async fn record_run(&self, meta: &RunMeta) -> anyhow::Result<()>;
    async fn run_meta(&self, head: &NodeHash) -> anyhow::Result<Option<RunMeta>>;
    async fn list_runs(&self) -> anyhow::Result<Vec<RunMeta>>;
    // chain_to has a default impl walking parents via get; override if
    // your backend can do it in one query (Postgres uses a recursive CTE).
}
```

Wire into `forge-cli/src/main.rs::open_storage` so the CLI auto-detects
your backend from the `--db` URL prefix.

## Code style

- `cargo fmt --all` before committing (CI checks this).
- `cargo clippy --all-targets -- -D warnings` clean — `#[allow(...)]` only
  with a comment explaining why.
- No new `unwrap` in non-test code; use `anyhow::bail!` or `?`.
- Tests live in `#[cfg(test)] mod tests` next to the code.
- Postgres tests gated on `TEST_POSTGRES_URL` so CI without a DB still
  passes.

## Commit messages

Imperative subject, `<60` chars. Body explains the *why*, references
issues only when relevant. No emoji.

## Releasing

Not yet — pre-1.0. Bump `version` in `Cargo.toml`'s `[workspace.package]`
when we get there.
