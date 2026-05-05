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
| `forge-core` | step types, hashes, graph invariants, `Agent` / `Matcher` / `Tool` traits |
| `forge-storage` | `Storage` trait, `MemoryStorage`, `SledStorage` |
| `forge-anthropic` | Anthropic Messages API adapter (multi-turn tool use, prompt caching) |
| `forge-openai` | OpenAI Chat Completions adapter (multi-turn tool use) |
| `forge-gemini` | Google Gemini `generateContent` adapter (multi-turn function calling) |
| `forge-cli` | the `forge` binary (`run`, `runs`, `replay`, `continue`, `fork`, `diff`) |

Storage backends planned: in-memory (done), sled, Postgres (single source of truth, durable resume).

## Roadmap

- [x] Workspace skeleton, core types, in-memory storage
- [x] `forge run` against a stub agent harness (FakeAgent)
- [x] sled backend; `forge runs` and `forge replay` read from disk
- [x] Anthropic Messages API adapter (single-shot, no tools yet)
- [x] `Matcher` trait sketch (interception / tripwires)
- [x] `forge fork <run> --at <step> --rewrite-text` (Prompt/Message kinds)
- [x] `forge diff` v0: pairwise chain walk, finds first divergence
- [x] Anthropic tool-use loop (multi-turn, tool calls executed locally)
- [x] `Tool` trait + built-in `Calculator` tool
- [x] `forge fork ... --continue` — drive a fresh agent forward from the fork
- [x] `forge diff` v1: structural alignment via signature-based LCS, modified/only markers
- [x] Fork supports every step kind (text for prompt/message, JSON for tool_call/tool_result)
- [x] Auto-execute tool after fork-on-tool-call when `--continue` is set
- [x] `forge continue <run>` + `--max-turns` on run/continue/fork — mid-run model handoffs
- [x] OpenAI Chat Completions adapter (`--agent openai`)
- [x] Cross-provider continuations: run with Claude, continue with GPT-5
- [x] Friendly error rendering (no Rust backtrace on missing API key)
- [x] `forge view` TUI: timeline + content panes, single-run and aligned-diff modes
- [x] Anthropic SSE streaming with text deltas to stderr
- [x] `forge serve` HTTP recorder: drop-in proxy for any agent (Anthropic + OpenAI)
- [x] Streaming pass-through in the recorder (SSE tee)
- [x] Auto-threading via content-addressed prefix detection
- [x] `forge web` embedded HTML viewer (single binary, no build pipeline)
- [x] `examples/` with runnable scripts
- [x] Postgres backend (`--db postgres://...`) for multi-process / multi-machine
- [x] OpenAI SSE streaming (matches Anthropic; `--stream` works on both)
- [x] Gemini adapter (`--agent gemini`, multi-turn function calling, streaming)
- [x] Anthropic prompt caching (`cache_control: ephemeral` on the latest input block)
- [x] `forge web` token-level inline diff for modified steps
- [x] Friendly Postgres connect errors (one-line messages for auth / DNS / refused)
- [x] `--public` flag on `forge serve` and `forge web` to opt-in to non-localhost bind
- [ ] HTTP recorder middleware (drop-in for any agent)
- [ ] Adapter for [Rig](https://rig.rs/) (thin shim once tool-use lands)
- [ ] `forge diff` v1: LLM-judge for "why did these diverge?"
- [ ] Tripwires v0 (only useful once a real agent loop is intercepting; deferred)
- [ ] Deterministic replay (seeded where APIs allow, full request/response capture)
- [ ] Postgres backend (durable resume across machines, large blob dedup)
- [ ] TUI viewer (`ratatui`)

## Install

```bash
git clone https://github.com/theohmwoa/forge && cd forge
cargo install --path crates/forge-cli --force
forge --help
```

The binary lands in `~/.cargo/bin/forge`. Make sure that's in your `PATH`.
Single binary, no runtime deps.

## Quick start

```bash
# record a 5-step scripted run (no API key required)
forge run --agent fake

# real Anthropic call with tool use
export ANTHROPIC_API_KEY=...
forge run --agent anthropic --tools calculator --prompt "what is 47 * 53?"

# OpenAI works the same way
export OPENAI_API_KEY=...
forge run --agent openai --model gpt-5 --tools calculator --prompt "what is 47 * 53?"

# Gemini, ditto
export GEMINI_API_KEY=...
forge run --agent gemini --model gemini-2.5-flash --tools calculator --prompt "what is 47 * 53?"

# list recorded runs
forge runs

# walk a run back from disk
forge replay <head-prefix>

# fork at any step. Text for prompt/message, JSON for tool_call/tool_result.
forge fork <run> --at <prompt-prefix>     --rewrite "ask differently"
forge fork <run> --at <tool-call-prefix>  --rewrite '{"op":"mul","a":7,"b":8}'
forge fork <run> --at <tool-result-prefix> --rewrite '99'

# fork + continue: drive a fresh agent forward from the new step.
# For tool_call rewrites, the tool runs locally first to produce a real result.
forge fork <run> --at <step> --rewrite "..." --continue --tools calculator

# stop a run early so another model can pick up
forge run --agent anthropic --model claude-haiku-4-5-20251001 --max-turns 1 \
    --prompt "what is 47 * 53?" --tools calculator
forge continue <head> --model claude-sonnet-4-6 --tools calculator

# diff two runs (shared prefix is content-addressed equal, so cheap)
forge diff <head-a> <head-b>

# interactive TUI: timeline on the left, content on the right
forge view <head>
forge view <head-a> --diff <head-b>     # aligned diff with j/k navigation

# local web viewer (single embedded HTML page, localhost only by default)
forge web --port 7879

# HTTP recorder — drop-in proxy for any existing agent
forge serve --port 7878
export ANTHROPIC_BASE_URL=http://127.0.0.1:7878
./my-existing-agent.py    # every API call now records into Forge
```

## How it works with your existing agent

Forge has three integration paths, in increasing order of friction:

1. **HTTP recorder (`forge serve`)** — point your existing agent at the proxy
   via `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL`. Works with the Anthropic
   and OpenAI SDKs in Python, TypeScript, Go, etc., and anything built on
   top (LangChain, LlamaIndex, Rig, raw HTTP). Streaming responses are
   teed; multi-turn calls are auto-threaded via content-addressed prefix
   detection — no session id needed.
2. **Native Rust integration** — use `forge-anthropic` / `forge-openai`
   crates directly for the full agent + recording pipeline.
3. **Direct graph access** — the run graph is just a sled DB on disk. Walk
   it from any language without a Forge client.

See the [`examples/`](examples/) directory for runnable scripts covering
each path.

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

legend: =  same   ~  modified   -  only in A   +  only in B

~  message[assistant]
-    I'll use the calculator tool.
+    Let me solve this without tools.
-  tool_call[calculator]
-    {"a":2,"b":3,"op":"add"}
-  tool_result[call-1]
-    5
-  message[assistant]
-    The sum is 5.
```

## License

Dual-licensed under [MIT](LICENSE-MIT) and [Apache-2.0](LICENSE-APACHE), at your option.
