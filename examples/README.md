# Forge examples

Each script demonstrates one feature. Run them from the repo root after
`cargo build --release` (binary lives at `target/release/forge`).

| Script | What it shows |
|---|---|
| [`01-basic-run.sh`](01-basic-run.sh) | Record a scripted run, list it, replay it from disk |
| [`02-fork-and-diff.sh`](02-fork-and-diff.sh) | Fork at a step, rewrite content, diff the branches |
| [`03-tool-call-rewrite.sh`](03-tool-call-rewrite.sh) | Fork at a tool_call, rewrite its JSON args, auto-execute the tool |
| [`04-recorder-curl.sh`](04-recorder-curl.sh) | Use the HTTP recorder with raw `curl` (no agent code required) |
| [`05-multi-model-handoff.sh`](05-multi-model-handoff.sh) | Haiku does turn 1, Sonnet picks up |
| [`06-cross-provider-diff.sh`](06-cross-provider-diff.sh) | Same prompt → Claude vs GPT-5 → structural diff |

Scripts that need real API access exit cleanly with a message if the relevant
environment variable isn't set, so the no-key examples (`01`, `02`, `03`,
`04`) always run end-to-end.
