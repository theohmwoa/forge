#!/usr/bin/env bash
set -euo pipefail
# Fork at a tool_call, rewrite its JSON input, observe the new chain.
# (No --continue here because FakeAgent doesn't support continuation; the
# rewrite stops at the new tool_call. Slice 04 shows continuation.)

DB="$(mktemp -d)/forge.db"
FORGE=${FORGE_BIN:-./target/release/forge}

"$FORGE" --db "$DB" run --agent fake > /dev/null
ORIG=$("$FORGE" --db "$DB" runs | awk '{print $1}' | head -1)

TOOL_CALL=$(
    "$FORGE" --db "$DB" replay "$ORIG" \
        | awk '/tool_call/{print $2; exit}'
)

echo "==> forking $ORIG at tool_call $TOOL_CALL"
echo "    rewriting input from add(2,3) to mul(7,8)"
"$FORGE" --db "$DB" fork "$ORIG" --at "$TOOL_CALL" \
    --rewrite '{"op":"mul","a":7,"b":8}'

FORK=$("$FORGE" --db "$DB" runs | awk '{print $1}' | head -1)
echo
echo "==> forge diff $ORIG $FORK"
"$FORGE" --db "$DB" diff "$ORIG" "$FORK"
