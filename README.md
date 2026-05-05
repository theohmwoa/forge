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
- [x] `forge fork <run> --at <step> --rewrite-text` (Prompt/Message kinds)
- [x] `forge diff` v0: pairwise chain walk, finds first divergence
- [ ] Anthropic tool-use loop (multi-turn, tool calls round-tripped)
- [ ] `forge fork ... --continue` — drive a fresh agent forward from the fork
- [ ] HTTP recorder middleware (drop-in for any agent)
- [ ] Adapter for [Rig](https://rig.rs/) (thin shim once tool-use lands)
- [ ] `forge diff` v1: LLM-judge for "why did these diverge?"
- [ ] Tripwires v0 (only useful once a real agent loop is intercepting; deferred)
- [ ] Deterministic replay (seeded where APIs allow, full request/response capture)
- [ ] Postgres backend (durable resume across machines, large blob dedup)
- [ ] TUI viewer (`ratatui`)

## Quick start

```bash
# record a 5-step scripted run (no API key required)
forge run --agent fake

# real Anthropic call (single-shot, no tools yet)
export ANTHROPIC_API_KEY=...
forge run --agent anthropic --prompt "what is 2 + 2"

# list recorded runs
forge runs

# walk a run back from disk
forge replay <head-prefix>

# fork a run at a step, rewriting its content
forge fork <run-prefix> --at <step-prefix> --rewrite-text "..."

# diff two runs (shared prefix is content-addressed equal, so cheap)
forge diff <head-a> <head-b>
```

End-to-end example (FakeAgent emits a 5-step conversation; we fork at the
assistant's first message and diff):

```
$ forge run --agent fake
run complete: 5 steps
  0  6952335a6f  prompt[fake-model-v1]
  1  fadbc99433  message[assistant]   "I'll use the calculator tool."
  2  ff2e474b78  tool_call[calculator]
  3  2cec969b11  tool_result[call-1]
  4  715ac59405  message[assistant]   "The sum is 5."

$ forge fork 715ac594 --at fadbc994 --rewrite-text "Let me solve this without tools."
new head: cb7329bca5...

$ forge diff 715ac594 cb7329bc
shared prefix: 1 step(s)
--- only in A ---  (4 steps: original assistant -> tool flow -> answer)
--- only in B ---  (1 step: rewritten assistant)
```

## License

Dual-licensed under [MIT](LICENSE-MIT) and [Apache-2.0](LICENSE-APACHE), at your option.
