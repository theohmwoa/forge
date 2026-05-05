# forge

**Git for agent runs.** Content-addressed, branchable, diffable, replayable.

> Status: very early. Public design exploration. APIs will break.

## What it is

Every agent execution is a DAG of content-addressed steps. Each step (prompt, tool call, response, sandbox run) is a hash-keyed node. The graph is the source of truth.

That gives you four things you can't get from append-only logs:

- **Fork.** Branch any run from any step. Try a different model, a different prompt, a different tool — without re-running the prefix.
- **Diff.** Compare two runs and see exactly where they diverged. Tool calls, outputs, semantic drift.
- **Replay.** Re-execute a run deterministically against a different backend. Catch regressions before they ship.
- **Resume.** Crash mid-run, resume from the last hashed node. No lost work.

## Why

LLM agents are stateful, expensive, and non-deterministic. The default observability story is "log everything, scroll through JSONL." That doesn't help you answer the question every team actually has: *what changed, and why?*

Forge treats agent runs the way `git` treats source code: a commit graph you can inspect, branch, and diff.

## Architecture

Cargo workspace, three crates:

| crate | what it owns |
|---|---|
| `forge-core` | step types, hashes, graph invariants, `Agent` and `Matcher` traits |
| `forge-storage` | `Storage` trait + in-memory / sled / Postgres backends |
| `forge-anthropic` | Anthropic Messages API adapter (single-shot for now) |
| `forge-cli` | the `forge` binary (`run`, `replay`, `runs`, `fork`, `diff`) |

Storage backends planned: in-memory (done), sled, Postgres (single source of truth, durable resume).

## Roadmap

- [x] Workspace skeleton, core types, in-memory storage
- [x] `forge run` against a stub agent harness (FakeAgent)
- [x] sled backend; `forge runs` and `forge replay` read from disk
- [x] Anthropic Messages API adapter (single-shot, no tools yet)
- [x] `Matcher` trait sketch (interception / tripwires)
- [ ] Anthropic tool-use loop (multi-turn, tool calls round-tripped)
- [ ] `forge fork <run>@<step>` — load state, swap one variable, drive forward
- [ ] `forge diff` v0: structural alignment of tool calls
- [ ] Tripwires v0: regex / JSON-path matchers, abort + auto-fork on hit
- [ ] HTTP recorder middleware (drop-in for any agent)
- [ ] Adapter for [Rig](https://rig.rs/) (thin shim once tool-use lands)
- [ ] `forge diff` v1: LLM-judge for "why did these diverge?"
- [ ] Deterministic replay (seeded where APIs allow, full request/response capture)
- [ ] Postgres backend (durable resume across machines, large blob dedup)
- [ ] TUI viewer (`ratatui`)

## Quick start

```bash
# scripted demo agent (no API key required)
cargo run -- run --agent fake

# real Anthropic call
export ANTHROPIC_API_KEY=...
cargo run -- run --agent anthropic --prompt "what is 2 + 2"

cargo run -- runs                   # list recorded runs
cargo run -- replay <prefix>        # walk the chain back from disk
```

## License

Dual-licensed under [MIT](LICENSE-MIT) and [Apache-2.0](LICENSE-APACHE), at your option.
