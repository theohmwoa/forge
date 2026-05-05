# forge — status

Snapshot of what's built, what's verified, and what's worth doing next.

## What ships today

### Storage
- **`Storage` trait** — async `put`, `get`, `children`, `record_run`,
  `run_meta`, `list_runs`, `chain_to`. `chain_to` has a default impl that
  walks parents via `get`; backends override with single-query versions
  where it pays off (Postgres uses a recursive CTE).
- **`MemoryStorage`** — in-process, ephemeral. Tests + smoke runs.
- **`SledStorage`** — single-process embedded DB. Default backend.
- **`PostgresStorage`** — multi-process, multi-machine. Uses `sqlx` with
  rustls + a connection pool. Schema is created idempotently on connect:

  ```
  forge_steps(id PK, parent, kind JSONB, timestamp_ms)
  forge_runs(head PK, root, recorded_at_ms)
  ```

  CLI selects a backend by URL scheme: filesystem path → sled,
  `postgres://...` → Postgres.

### Agents
- **`forge-anthropic`** — Anthropic Messages API, multi-turn tool use,
  SSE streaming with text deltas to stderr (`--stream`).
- **`forge-openai`** — OpenAI Chat Completions API, multi-turn tool use,
  SSE streaming (delta accumulation for both content and tool_calls
  by index).
- **Cross-provider continuations** — `forge continue <head> --agent X`
  re-translates the chain into the new provider's wire format. Works
  even mid-tool-use because the prefix is content-addressed.
- **`AnthropicAgent::with_max_turns(n)` / `OpenAIAgent::with_max_turns(n)`**
  — graceful early stop for handoffs.

### Tools
- **`Tool` trait** — async `name`, `description`, `schema`, `run`.
- **`Calculator`** — built-in arithmetic tool for examples and tests.

### Run graph
- Content-addressed steps via blake3.
- Step kinds: `Prompt`, `Message{role,content}`, `ToolCall{call_id,name,input}`,
  `ToolResult{call_id,output}`.
- Hash includes `(parent, kind)` only — timestamp is excluded so
  identical continuations of the same prefix collapse to the same node.

### CLI commands
- `forge run --agent <fake|anthropic|openai> --prompt ... --tools ...
   --max-turns N --stream` — drive an agent, record everything.
- `forge runs` — list recorded runs.
- `forge replay <head>` — walk a run from disk.
- `forge fork <run> --at <step> --rewrite ... [--continue]` — branch at
  any step. `--rewrite` is text for prompt/message, JSON for
  tool_call/tool_result. `--continue` drives a fresh agent from the
  rewrite point; for tool_call rewrites, the tool is locally executed
  to materialize the new tool_result before the agent picks up.
- `forge continue <run>` — extend an existing run with a (potentially
  different) agent. Pairs with `--max-turns` for model handoffs.
- `forge diff <a> <b>` — content-addressed prefix detection +
  signature-based LCS alignment of the divergent tail.
  Markers: `=` (same), `~` (modified), `-` (only A), `+` (only B).
- `forge view <head> [--diff <head-b>]` — `ratatui` TUI viewer.
- `forge serve --port N` — drop-in HTTP recorder. Proxies
  `/v1/messages` (Anthropic) and `/v1/chat/completions` (OpenAI). Tees
  SSE streams. Multi-call conversations are auto-threaded by hashing
  the prefix and reusing existing nodes.
- `forge web --port N` — embedded HTML viewer over the run graph.
  Single static page (no build pipeline), localhost-only by default,
  shift-click to diff two runs.

### Examples
- `examples/01-basic-run.sh` … `examples/06-cross-provider-diff.sh` —
  six runnable shell scripts. Scripts that need API keys exit cleanly
  if the relevant env var isn't set.

### Tests
- 36 tests across 8 crates. All `cargo clippy --all-targets -- -D warnings`
  and `cargo test --all` clean.
- `Postgres` tests are gated on `TEST_POSTGRES_URL`.

## What was verified end-to-end

- `forge run --agent fake` produces a 5-step chain.
- `forge fork <head> --at <assistant> --rewrite "..."` produces a
  shorter chain sharing the prompt as a content-addressed prefix.
- `forge diff <orig> <fork>` renders a clean unified diff with one
  modified entry and three only-A entries.
- `forge serve` proxies Anthropic and OpenAI, threads multi-call
  conversations into one tree, tees SSE streams to clients.
- `forge web` serves the HTML page + JSON endpoints
  (`/api/runs`, `/api/runs/{head}`, `/api/diff/{a}/{b}`) — verified
  via `curl` (Chrome localhost was blocked by the extension's
  permission policy, but the JSON shapes are correct).

## What's worth adding next

Ranked by likely value × cost.

### High value, small cost
1. **Postgres `children` query optimization** — the current default-impl
   `chain_to` is overridden with a CTE; do the same for `children`
   (currently OK but a single SELECT is cheaper than the trait default).
2. **Friendly error rendering for Postgres connect failures** — currently
   surfaces a long sqlx error chain. Map to a one-liner.
3. **Conversation tags / labels API** — let the recorder accept a
   `x-forge-tag: my-experiment` header and store it in `RunMeta`.
   30-line change. Big UX win for "browse my runs by experiment."
4. **Anthropic prompt caching headers** — passthrough is fine today,
   but native adapters should set `cache_control` on the system prompt
   when present.

### High value, medium cost
5. **Streaming pass-through for OpenAI in the recorder** — Anthropic
   streaming is teed; OpenAI is currently stripped to non-streaming.
   Mirror what we did for Anthropic.
6. **`forge web` shift-click diff polish** — works but the diff side
   panel could show a token-level diff (currently shows full A and B
   stacked). Use `similar` crate or a tiny LCS over characters.
7. **Auth for `forge serve` / `forge web`** — bearer-token gate behind
   `--token <value>`. Required before binding to anything but localhost.
8. **Rig adapter** — wrap an existing Rig agent and record its calls.
   Smaller surface than direct API integration, but unlocks the
   "every Rust LLM project" audience.

### High value, big cost
9. **Tripwires** (the matcher trait stub already exists) — declarative
   interception rules over streaming output. Auto-fork on hit.
   The hard part isn't the runtime, it's the rule language.
10. **Postgres `LISTEN/NOTIFY` for live `forge web`** — push new runs
    to the browser instead of polling every 5s.
11. **First-class Gemini adapter** — Google's API has a different
    enough shape that it's a real port, not a tweak.
12. **Web UI fork/continue actions** — wire the existing CLI verbs into
    HTML buttons so non-CLI users can branch a run from the browser.
    Requires careful auth (#7) before exposing.

### Production-readiness gaps
- Multi-tenancy in `forge serve` (per-user runs, isolation).
- Postgres migration story beyond `CREATE TABLE IF NOT EXISTS`.
- Telemetry / OpenTelemetry exporter.
- Rate limiting on the proxy.
- Backpressure handling in the SSE tee for very long streams.

### Definitely-not-yet
- A SaaS/web app SPA. The whole project leans into "single binary,
  cargo-installable, runs locally" — adding a real web app would
  contradict that. The embedded `forge web` is sufficient.

## Known limits

- `forge serve` doesn't preserve cross-request *Run* identity beyond
  content-addressed prefix matching. Two clients sending the same
  conversation end up sharing nodes, which is correct but means
  per-tenant labelling needs `--tag` (gap #3 above).
- Sled's children-index update isn't safe under multi-process writers.
  The Postgres backend is the answer for that case.
- Anthropic's streaming SSE protocol may add new event types over
  time; unknown event types are currently silently skipped.
- `prefix_to_history` (in the OpenAI adapter) handles all current
  step kinds but assumes the chain was originally produced by an
  OpenAI-shaped agent. Cross-provider continuations from Anthropic to
  OpenAI work but aren't formally tested.

## Repo health snapshot

```
crates/
├── forge-anthropic   # Anthropic Messages API
├── forge-cli         # forge binary
├── forge-core        # types, agent + tool + matcher traits
├── forge-openai      # OpenAI Chat Completions API
├── forge-recorder    # axum proxy with auto-threading + SSE tee
└── forge-storage     # Memory / Sled / Postgres
```

- 36 tests passing
- clippy `-D warnings` clean across all targets
- ratatui TUI viewer
- examples/ directory with 6 runnable scripts
- README documents all integration paths
