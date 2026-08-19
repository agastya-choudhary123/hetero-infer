#!/usr/bin/env bash
# Run the model across this machine and one other, with one command.
#
#   ./scripts/pool.sh agastya@desktop.local
#   ./scripts/pool.sh agastya@desktop.local bench
#
# Mirrors the project to the other machine over SSH, installs what it needs,
# starts it, runs, and shuts it down again. The only thing you have to do by
# hand is turn on Remote Login over there:
#   System Settings -> General -> Sharing -> Remote Login
#
# Budgets are chosen so that neither machine could hold the model on its own —
# that is the thing being demonstrated.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${1:-}"
MODE="${2:-run}"
if [[ -z "$HOST" ]]; then
  echo "usage: $0 user@host [run|bench]" >&2
  echo "   eg: $0 $(whoami)@desktop.local" >&2
  exit 64
fi

SSH="${SSH:-ssh}"
RSYNC="${RSYNC:-rsync}"
MODEL="${MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}"
PORT="${PORT:-29501}"
REMOTE_DIR="${REMOTE_DIR:-hetero-infer}"
LOG=/tmp/hetero-worker.log
PIDFILE=/tmp/hetero-worker.pid

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }

# --- 1. can we reach it -----------------------------------------------------
say "checking $HOST"
if ! $SSH -o BatchMode=yes -o ConnectTimeout=8 "$HOST" true 2>/dev/null; then
  cat >&2 <<MSG
cannot ssh to $HOST.

On that machine: System Settings -> General -> Sharing -> Remote Login (on).
Then, once, from here:   ssh-copy-id $HOST
MSG
  exit 1
fi

ARCH=$($SSH "$HOST" 'uname -m' | tr -d '\r')
if [[ "$ARCH" != "arm64" ]]; then
  echo "that machine is $ARCH; this runs on Apple Silicon (MLX) only." >&2
  exit 1
fi

# --- 2. pick budgets so neither box can hold the model alone ----------------
read -r MODEL_GIB BUDGET <<<"$(python3 - "$MODEL" <<'PY'
import sys
from hetero.shard import WeightIndex, snapshot_dir
gib = WeightIndex(snapshot_dir(sys.argv[1])).total_bytes() / 2**30
print(f"{gib:.2f} {gib * 0.65:.2f}")
PY
)"
BUDGET="${BUDGET_OVERRIDE:-$BUDGET}"
say "model is ${MODEL_GIB} GiB; giving each machine a ${BUDGET} GiB budget"
say "neither one can hold it, so if this generates text, the pool did it"

# --- 3. mirror the project and its deps -------------------------------------
say "syncing project to $HOST:$REMOTE_DIR"
$RSYNC -az --delete \
  --exclude .git --exclude __pycache__ --exclude 'bench/results' \
  ./ "$HOST:$REMOTE_DIR/"

say "installing dependencies over there (quiet unless something is missing)"
$SSH "$HOST" "cd $REMOTE_DIR && python3 -m pip install -q -r requirements.txt"

# --- 4. make sure the weights are already over there ------------------------
# Fetching 4 GB while the coordinator waits on the handshake is a good way to
# hit a timeout, so do it up front where the progress bar is visible.
say "fetching model on $HOST if it is not cached yet"
$SSH "$HOST" "python3 -c \"from huggingface_hub import snapshot_download as d; d('$MODEL')\" >/dev/null"

# --- 5. start the worker ----------------------------------------------------
cleanup() {
  say "stopping worker on $HOST"
  $SSH "$HOST" "kill \$(cat $PIDFILE) 2>/dev/null; rm -f $PIDFILE" 2>/dev/null || true
}
trap cleanup EXIT

say "starting worker on $HOST (first run downloads the model, be patient)"
$SSH "$HOST" "cd $REMOTE_DIR && rm -f $LOG && \
  nohup python3 -m hetero.cli worker --port $PORT --name '$HOST' --mem-gib $BUDGET \
  > $LOG 2>&1 & echo \$! > $PIDFILE"

for _ in $(seq 1 60); do
  if $SSH "$HOST" "grep -q listening $LOG" 2>/dev/null; then break; fi
  sleep 1
done
if ! $SSH "$HOST" "grep -q listening $LOG" 2>/dev/null; then
  echo "worker did not come up; its log says:" >&2
  $SSH "$HOST" "cat $LOG" >&2 || true
  exit 1
fi

# --- 6. drive it from here --------------------------------------------------
PEER_HOST="${HOST#*@}"
PEER="$PEER_HOST:$PORT:${HOST}:$BUDGET"
say "running"
echo
if [[ "$MODE" == "bench" ]]; then
  python3 scripts/bench_all.py --model "$MODEL" --peer "$PEER" --mem-gib "$BUDGET"
else
  python3 -m hetero.cli run --model "$MODEL" --peer "$PEER" \
    --name "$(hostname -s)" --mem-gib "$BUDGET" --objective latency \
    --max-tokens "${MAX_TOKENS:-64}" \
    --prompt "${PROMPT:-Explain, in two sentences, what pipeline parallelism is.}"
fi
