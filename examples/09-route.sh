#!/usr/bin/env bash
set -euo pipefail
# `forge run --after-tool` — tool-aware model routing within an agent loop.
#
# Every tool boundary is a chance to swap models. Use cases:
#
#   • Cost: cheap models digest mechanical tool output (search results,
#     file listings); smart models handle the planning and final synthesis.
#
#   • Specialization: SQL-tuned model after `execute_sql`, code-tuned
#     after `run_tests`, vision after `screenshot`.
#
#   • Privacy: local model after a tool that returned PII so the cloud
#     model never sees the sensitive payload.
#
# Each rule is `tool:[provider/]model`. When the previous step is a
# tool_result for that tool, the next agent cycle uses the rule's model.
# Otherwise the default (`--model`) handles the cycle. Rules are
# evaluated per-cycle, looking at the most recent ToolCall in the chain.
#
# Demo: a calculator-using agent where the planner runs on Sonnet and
# the post-tool digestion runs on Haiku (≈10x cheaper).

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ANTHROPIC_API_KEY not set; this demo needs real API calls."
    exit 0
fi

DB="$(mktemp -d)/forge.db"
FORGE=${FORGE_BIN:-./target/release/forge}

echo "==> default Sonnet for the plan, Haiku for digesting calculator results"
"$FORGE" --db "$DB" run \
    --agent anthropic \
    --model claude-sonnet-4-6 \
    --tools calculator \
    --after-tool calculator:claude-haiku-4-5-20251001 \
    --max-turns 6 \
    --prompt "What's 47 * 53, and is the result divisible by 3? Use the calculator tool for both."

echo
echo "==> the routed run is recorded in the graph; replay or open in forge web."
echo "    $FORGE --db $DB runs"
"$FORGE" --db "$DB" runs | head -3
