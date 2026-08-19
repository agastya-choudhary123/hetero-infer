#!/usr/bin/env bash
# Two real machines over the LAN.
#
# On the second box (the one that will hold the tail of the model):
#   python3 -m hetero.cli worker --port 29501 --name laptop --mem-gib 6
#
# Then here, with PEER set to that machine's address:
#   PEER=192.168.1.42:29501:laptop:6 ./scripts/run_lan.sh
#
# The coordinator probes both machines and the wire between them before it
# decides where to cut, so the two boxes do not need to be alike.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PEER:?set PEER=host:port:name:mem_gib}"
MODEL="${MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}"
BUDGET="${BUDGET:-6}"
OBJECTIVE="${OBJECTIVE:-latency}"
python3 -m hetero.cli run --model "$MODEL" --peer "$PEER" --name "$(hostname -s)" \
  --mem-gib "$BUDGET" --objective "$OBJECTIVE" --max-tokens 64 \
  --prompt "${PROMPT:-Explain pipeline parallelism in two sentences.}"
