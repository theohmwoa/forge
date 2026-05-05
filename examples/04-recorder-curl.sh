#!/usr/bin/env bash
set -euo pipefail
# Run the recorder, hit it with raw curl as if it were Anthropic's API,
# and verify the run was recorded into Forge.

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ANTHROPIC_API_KEY is not set; skipping live recorder example."
    echo "(Set it to make this script call the real Anthropic API through the proxy.)"
    exit 0
fi

DB="$(mktemp -d)/forge.db"
FORGE=${FORGE_BIN:-./target/release/forge}

# Pick a free port.
PORT=${FORGE_PORT:-7878}

echo "==> starting forge serve in background"
"$FORGE" --db "$DB" serve --port "$PORT" > /tmp/forge-serve.log 2>&1 &
PID=$!
trap "kill $PID 2>/dev/null || true" EXIT
sleep 0.5

echo "==> POST to http://127.0.0.1:$PORT/v1/messages (proxying to Anthropic)"
echo "    tagging the run with x-forge-tag: my-experiment"
curl -sS -X POST "http://127.0.0.1:$PORT/v1/messages" \
    -H "x-api-key: $ANTHROPIC_API_KEY" \
    -H "anthropic-version: 2023-06-01" \
    -H "x-forge-tag: my-experiment" \
    -H "content-type: application/json" \
    -d '{
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "say only the digit 5"}]
    }' | head -c 200
echo
echo

echo "==> forge runs --tag my-experiment"
"$FORGE" --db "$DB" runs --tag my-experiment

HEAD=$("$FORGE" --db "$DB" runs | awk '{print $1}' | head -1)
echo
echo "==> forge replay $HEAD"
"$FORGE" --db "$DB" replay "$HEAD"
