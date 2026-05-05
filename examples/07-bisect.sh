#!/usr/bin/env bash
set -euo pipefail
# `forge bisect`: git-bisect for agent runs.
#
# Given a `good` run and a `bad` run that share a prefix, walk the divergent
# tail of `bad` and substitute each step in turn with the corresponding step
# from `good`. The first substitution that recovers the run identifies the
# step that caused the failure.
#
# This script demonstrates the workflow end-to-end with a synthetic
# good-vs-bad pair built from the FakeAgent run + a single fork. The actual
# bisect step requires an LLM API key (the agent must drive the rewritten
# fork forward); we use Anthropic here.

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ANTHROPIC_API_KEY not set; demo will stop before the live bisect step."
    echo "(The fork + diff portions still run.)"
fi

DB="$(mktemp -d)/forge.db"
FORGE=${FORGE_BIN:-./target/release/forge}

echo "==> 1. record a 'good' run with the FakeAgent (5-step canonical chain)"
"$FORGE" --db "$DB" run --agent fake | head -10
echo

GOOD=$("$FORGE" --db "$DB" runs | awk '{print $1}' | head -1)
echo "good head: $GOOD"
echo

echo "==> 2. fork the good run at the final assistant message and rewrite to a 'wrong' answer"
"$FORGE" --db "$DB" fork "$GOOD" --at "715ac594" --rewrite "the answer is wrong" | head -3
BAD=$("$FORGE" --db "$DB" runs | awk '{print $1}' | head -1)
echo "bad head:  $BAD"
echo

echo "==> 3. diff to confirm the chains share 4 steps and diverge at step 4"
"$FORGE" --db "$DB" diff "$GOOD" "$BAD"
echo

if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "==> 4. forge bisect — find the step that caused the failure"
    echo "    (each trial forks bad at a divergent step with good's value, then"
    echo "     drives a real Anthropic call forward to see if the run recovers)"
    "$FORGE" --db "$DB" bisect "$GOOD" "$BAD" \
        --agent anthropic --model claude-haiku-4-5-20251001 \
        --expect "sum is 5" --max-turns 2
    echo
    echo "==> 5. the trial runs are tagged so they're easy to find or clean up"
    "$FORGE" --db "$DB" runs --tag "bisect-step-4"
fi
