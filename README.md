# forge

**A tool-aware model router for agent loops** — plus a content-addressed run graph that makes every part of the loop inspectable, branchable, and replayable.

> Status: very early. Public design exploration. APIs will break.

## TL;DR — route different tools to different models in the same agent loop

```bash
$ forge run --tools web_search,execute_sql,screenshot \
    --model claude-sonnet-4-6 \
    --after-tool web_search:claude-haiku-4-5-20251001 \
    --after-tool execute_sql:openai/ft:gpt-4o-mini:org:sql-v3 \
    --after-tool screenshot:gemini/gemini-2.5-flash \
    --prompt "..."
```

Every tool boundary is a swap point. After a specific tool's result lands, the next turn uses a model picked for that tool:

- **Cost** — cheap models digest mechanical tool output; smart models plan and synthesize
- **Specialization** — SQL-fine-tuned model after `execute_sql`, code-fine-tuned after `run_tests`, vision after `screenshot`
- **Privacy** — local model after a tool that returned PII so the cloud model never sees the sensitive payload
- **Modality** — vision-only model only when an image was just produced

The router is keyed on the most recent `ToolCall` in the chain, evaluated fresh each cycle. Rules are `tool:[provider/]model`. No rule for a tool means the default (`--model`) handles that turn. Cross-provider rules work because the run graph is provider-neutral; each adapter translates the prefix into its own wire format.

## Plus: a graph, not a log

```bash
$ forge audit --target-model claude-haiku-4-5 --judge-model claude-sonnet-4-6
auditing 52 run(s) against claude-haiku-4-5

  ✓  47 EQUIVALENT  — safe to downgrade
  ⚠   5 DIFFERENT   — keep on the original model
```

`forge audit` measures per-prompt downgrade safety against your real recorded runs, with an LLM judge. The audit and the routing close the loop: measure → derive rules → apply.

## Cheap by default, smart only where it matters

Routing isn't *only* cost arbitrage. Sometimes the cheap model genuinely fails the task — and routing rescues it with a single flag.

A planted-bug fixture lives at `tests/fixtures/buggy/`: `multiply(a, b) -> a + b` instead of `a * b`. Three unit tests pin it down. The agent gets three tools: `run_tests`, `read_file`, `apply_patch`.

**Pure `gemini-2.5-flash-lite`, no routing.** The agent calls `run_tests`, sees cargo's `test result: FAILED. 1 failed; 2 passed`, and replies:

> *"The crate already passes all tests. There is no bug to fix."*

The cheap model hallucinates success in direct contradiction of the structured tool output. No patch applied. Tests still failing.

**Same agent, same prompt, same tools — one flag added:**

```bash
$ forge run --agent gemini --model gemini-2.5-flash-lite \
    --tools run-tests,read-file,apply-patch \
    --after-tool run_tests:gemini-2.5-flash \
    --after-tool read_file:gemini-2.5-flash \
    --prompt "..."

routed run: 5 cycle(s)
  cycle 0  [after run_tests  ]  flash-lite  +3 step(s)   ← cheap: ran the failing tests
  cycle 1  [after read_file  ]  flash       +2 step(s)   ← smart: read the source
  cycle 2  [after apply_patch]  flash       +2 step(s)   ← smart: applied a + b → a * b
  cycle 3  [after run_tests  ]  flash-lite  +2 step(s)   ← cheap: re-ran tests
  cycle 4  [after run_tests  ]  flash       +1 step      ← smart: synthesized FIXED
```

Final assistant message: `FIXED`. `cargo test` independently confirms `3 passed; 0 failed`. The fix the agent applied is in the chain — `apply_patch(old="a + b", new="a * b")`. Reproducible end-to-end on a real codebase, not synthetic.

The smart model only handled the two cycles where *structured interpretation matters* (read_file → understand source, run_tests → diagnose failure). Mechanical iteration stayed on the cheap model. Same agent loop. One config flag.

## What it is

Every agent execution is a DAG of content-addressed steps. Each step (prompt, tool call, response, sandbox run) is a hash-keyed node. The graph is the source of truth.

That gives you five things you can't get from append-only logs:

- **Fork.** Branch any run from any step. Try a different model, a different prompt, a different tool — without re-running the prefix.
- **Diff.** Compare two runs and see exactly where they diverged. Tool calls, outputs, semantic drift.
- **Bisect.** `forge bisect <good> <bad>` finds the first step that caused a failure by forking with `good`'s value at each divergent step and asking "did this recover?" Only possible because the graph is content-addressed and forks are free.
- **Replay.** Re-execute a run deterministically against a different backend. Catch regressions before they ship.
- **Resume.** Crash mid-run, resume from the last hashed node. No lost work.

## Why

LLM agents are stateful, expensive, and non-deterministic. The default observability story is "log everything, scroll through JSONL." That doesn't help you answer the question every team actually has: *what changed, and why?*

Forge treats agent runs the way `git` treats source code: a commit graph you can inspect, branch, and diff.

## Architecture

Cargo workspace:

| crate | what it owns |
|---|---|
| `forge-core` | step types, hashes, graph invariants, `Agent` / `Matcher` / `Tool` traits + the built-in tool library (calculator, fs ops, run_tests, apply_patch, run_command) |
| `forge-storage` | `Storage` trait — `MemoryStorage`, `SledStorage`, `PostgresStorage` |
| `forge-anthropic` | Anthropic Messages API adapter (multi-turn tool use, prompt caching, SSE streaming) |
| `forge-openai` | OpenAI Chat Completions adapter (multi-turn tool use, SSE streaming) |
| `forge-gemini` | Google Gemini adapter — public API + Vertex AI bearer auth + Gemini 3.x thinking-mode tool continuations |
| `forge-mcp` | Model Context Protocol JSON-RPC client + `Tool` adapter — drop in any MCP server's tools as native Forge tools |
| `forge-rig` | Record a [Rig](https://rig.rs) conversation as a Forge step chain |
| `forge-recorder` | HTTP recorder proxy (Anthropic + OpenAI, auto-threaded via content-addressed prefix detection) |
| `forge-cli` | the `forge` binary (`run`, `runs`, `replay`, `continue`, `fork`, `diff`, `bisect`, `audit`, `view`, `serve`, `web`) |

Storage backends: in-memory (tests), sled (default, single-process), Postgres (multi-process / multi-machine, durable resume, `LISTEN/NOTIFY` for live web updates).

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
- [x] Live `forge web` updates via Postgres `LISTEN/NOTIFY` (broadcast for sled / memory)
- [x] Web UI fork/continue actions (localhost-only until auth ships)
- [x] Anthropic cache-hit telemetry (`cache_creation_input_tokens` / `cache_read_input_tokens`)
- [x] `forge-rig` crate to record a [Rig](https://rig.rs) conversation
- [x] `forge bisect` — git-bisect for agent runs (find the step that caused a failure)
- [x] `forge audit` — replay recorded runs against cheaper models, judge equivalence, surface safe downgrades
- [x] `forge run --after-tool` — tool-aware per-turn model routing (cost / specialization / privacy)
- [ ] `forge audit --by-tool` — derive routing rules empirically from per-tool downgrade safety
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

# Vertex AI (bearer auth via gcloud ADC) — same flags, different env
export FORGE_VERTEX=1 FORGE_VERTEX_PROJECT=my-gcp-project
export GOOGLE_VERTEX_ACCESS_TOKEN=$(gcloud auth application-default print-access-token)
forge run --agent gemini --model gemini-3.1-pro-preview --tools calculator --prompt "..."

# point forge at any MCP server; its tools register automatically
# (alongside built-in tools — no Rust changes needed)
forge run --agent anthropic \
    --mcp-server "npx -y @modelcontextprotocol/server-filesystem ." \
    --prompt "..."

# autonomous bug-hunting on the planted-bug fixture — see top of README
forge run --agent gemini --model gemini-2.5-flash-lite \
    --tools run-tests,read-file,apply-patch \
    --after-tool run_tests:gemini-2.5-flash \
    --after-tool read_file:gemini-2.5-flash \
    --prompt "Run tests on tests/fixtures/buggy. If any fail, find the bug, apply_patch it, re-run. Output FIXED or BROKEN."

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

# git-bisect for agent runs: which step caused the failure?
# each trial forks `bad` at a divergent step with `good`'s value, drives
# the agent forward, and checks whether the run recovered.
forge bisect <good-head> <bad-head> --expect "sum is 5"

# audit recorded runs for cost: which can be safely downgraded to haiku?
# forge replays each prompt against the cheaper model and uses sonnet to
# judge equivalence; trial runs persist with tag audit-<model> for review.
forge audit --tag prod --target-model claude-haiku-4-5-20251001

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

[MIT](LICENSE).
