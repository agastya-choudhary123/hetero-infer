#!/usr/bin/env bash
# Two stages as two processes on this machine, each with a hard memory budget.
# Useful for development and for the memory claim; the two stages share one GPU,
# so it understates what two real machines do with concurrent streams.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}"
BUDGET="${BUDGET:-2.5}"
PROMPT="${PROMPT:-Explain pipeline parallelism in two sentences.}"

python3 -m hetero.cli worker --port 29501 --name node-b --mem-gib "$BUDGET" &
WORKER=$!
trap 'kill $WORKER 2>/dev/null || true' EXIT
sleep 3
python3 -m hetero.cli run --model "$MODEL" --peer 127.0.0.1:29501:node-b:"$BUDGET" \
  --name node-a --mem-gib "$BUDGET" --objective latency --max-tokens 64 --prompt "$PROMPT"
