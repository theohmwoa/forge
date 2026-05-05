#!/usr/bin/env bash
set -euo pipefail
# Fork at a step, rewrite its content, diff the branches.

DB="$(mktemp -d)/forge.db"
FORGE=${FORGE_BIN:-./target/release/forge}

"$FORGE" --db "$DB" run --agent fake > /dev/null
ORIG=$("$FORGE" --db "$DB" runs | awk '{print $1}' | head -1)

# Find the first assistant message step in the chain.
ASSISTANT=$(
    "$FORGE" --db "$DB" replay "$ORIG" \
        | awk '/message\[assistant\]/{print $2; exit}'
)

echo "==> forking $ORIG at assistant step $ASSISTANT"
"$FORGE" --db "$DB" fork "$ORIG" --at "$ASSISTANT" \
    --rewrite "Let me solve this without tools." > /dev/null

FORK=$("$FORGE" --db "$DB" runs | awk '{print $1}' | head -1)
echo
echo "==> forge diff $ORIG $FORK"
"$FORGE" --db "$DB" diff "$ORIG" "$FORK"
