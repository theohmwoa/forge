#!/usr/bin/env bash
set -euo pipefail
# Record a scripted run, list it, replay it from disk.

DB="$(mktemp -d)/forge.db"
FORGE=${FORGE_BIN:-./target/release/forge}

echo "==> forge run --agent fake"
"$FORGE" --db "$DB" run --agent fake
echo

echo "==> forge runs"
"$FORGE" --db "$DB" runs
echo

HEAD=$("$FORGE" --db "$DB" runs | awk '{print $1}' | head -1)
echo "==> forge replay $HEAD"
"$FORGE" --db "$DB" replay "$HEAD"
