#!/usr/bin/env bash
set -euo pipefail
# Haiku does turn 1, Sonnet picks up — recorded as one threaded run.

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ANTHROPIC_API_KEY is not set; skipping live handoff example."
    exit 0
fi

DB="$(mktemp -d)/forge.db"
FORGE=${FORGE_BIN:-./target/release/forge}

echo "==> haiku does 1 turn"
"$FORGE" --db "$DB" run --agent anthropic \
    --model claude-haiku-4-5-20251001 \
    --max-turns 1 \
    --tools calculator \
    --prompt "What is 47 * 53? You can use the calculator tool."
echo

HEAD=$("$FORGE" --db "$DB" runs | awk '{print $1}' | head -1)
echo "==> sonnet picks up from $HEAD"
"$FORGE" --db "$DB" continue "$HEAD" \
    --agent anthropic \
    --model claude-sonnet-4-6 \
    --tools calculator
