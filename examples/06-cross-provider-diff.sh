#!/usr/bin/env bash
set -euo pipefail
# Same prompt, two providers, structural diff of how each model approached it.

if [[ -z "${ANTHROPIC_API_KEY:-}" || -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Both ANTHROPIC_API_KEY and OPENAI_API_KEY must be set; skipping."
    exit 0
fi

DB="$(mktemp -d)/forge.db"
FORGE=${FORGE_BIN:-./target/release/forge}
PROMPT="What is 47 * 53? You may use the calculator tool."

echo "==> claude"
"$FORGE" --db "$DB" run --agent anthropic \
    --model claude-sonnet-4-6 --tools calculator --prompt "$PROMPT"
echo
A=$("$FORGE" --db "$DB" runs | awk '{print $1}' | head -1)

echo "==> gpt-5"
"$FORGE" --db "$DB" run --agent openai \
    --model gpt-5 --tools calculator --prompt "$PROMPT"
echo
B=$("$FORGE" --db "$DB" runs | awk '{print $1}' | head -1)

echo "==> forge diff (structural alignment)"
"$FORGE" --db "$DB" diff "$A" "$B"
