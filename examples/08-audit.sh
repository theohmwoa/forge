#!/usr/bin/env bash
set -euo pipefail
# `forge audit`: find the prompts you're overpaying for.
#
# Replay each recorded run through a cheaper target model. Use a strong
# model as judge to compare answers. Report which prompts are safe to
# downgrade — and which need to stay on the expensive model.
#
# 80% of agent API calls don't need a frontier model (industry consensus,
# 2026). The hard part is figuring out *which* 80%. This audits a
# representative sample for you in one command.

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ANTHROPIC_API_KEY not set; the audit needs real API calls."
    exit 0
fi

DB="$(mktemp -d)/forge.db"
FORGE=${FORGE_BIN:-./target/release/forge}

echo "==> 1. record some 'production' runs (Sonnet, simple Q&A)"
for q in \
    "What is the capital of France?" \
    "What is 13 * 17?" \
    "Name the largest ocean." \
    "Who wrote Hamlet?" \
    "What year did WW2 end?"; do
    "$FORGE" --db "$DB" run --agent anthropic --model claude-sonnet-4-6 \
        --prompt "$q" --tag prod > /dev/null
    echo "  recorded: $q"
done

echo
echo "==> 2. audit: which of these can run on Haiku for ~10x cheaper?"
"$FORGE" --db "$DB" audit \
    --tag prod \
    --target-model claude-haiku-4-5-20251001 \
    --judge-model claude-sonnet-4-6 \
    --limit 5

echo
echo "==> 3. browse the candidate trial runs in forge web (tag: audit-claude-haiku-...)"
echo "    $FORGE --db $DB web"
