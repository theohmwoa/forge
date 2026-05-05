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
| `forge-core` | step types, hashes, graph invariants |
| `forge-storage` | `Storage` trait + in-memory / sled / Postgres backends |
| `forge-cli` | the `forge` binary (`run`, `fork`, `diff`, `replay`) |

Storage backends planned: in-memory (done), sled, Postgres (single source of truth, durable resume).

## Roadmap

- [x] Workspace skeleton, core types, in-memory storage
- [ ] `forge run` against a stub agent harness
- [ ] sled backend
- [ ] `forge diff` with semantic step alignment
- [ ] `forge replay` with seeded model nondeterminism
- [ ] `forge fork` mid-run
- [ ] Postgres backend (durable resume across machines)
- [ ] TUI viewer (`ratatui`)
- [ ] Adapter for [Rig](https://rig.rs/)

## License

Dual-licensed under [MIT](LICENSE-MIT) and [Apache-2.0](LICENSE-APACHE), at your option.
