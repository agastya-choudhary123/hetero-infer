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
# Apple silicon runs the MLX backend; anything else runs the NumPy one.
if [[ "$ARCH" == "arm64" ]]; then
  BACKEND=auto; REQS=requirements.txt
else
  BACKEND=numpy; REQS=requirements-cpu.txt
  say "$HOST is $ARCH — no MLX there, so it will run the NumPy backend"
fi

# That machine needs a Python new enough for the runtime; /usr/bin/python3 on
# an older macOS is often too old, so find the best one available.
# A non-interactive ssh gets a bare PATH, so look in the usual install
# locations as well as whatever PATH happens to hold.
RPY=$($SSH "$HOST" '
for v in 3.13 3.12 3.11 3.10 3.9; do
  for q in /opt/homebrew/bin/python$v /usr/local/bin/python$v \
           /Library/Frameworks/Python.framework/Versions/$v/bin/python3 \
           $(command -v python$v 2>/dev/null); do
    [ -x "$q" ] || continue
    "$q" -c "import sys; raise SystemExit(0 if sys.version_info>=(3,9) else 1)" 2>/dev/null \
      && { echo "$q"; exit 0; }
  done
done
q=$(command -v python3 2>/dev/null) || exit 1
"$q" -c "import sys; raise SystemExit(0 if sys.version_info>=(3,9) else 1)" 2>/dev/null && echo "$q"
' | tr -d '\r')
if [[ -z "$RPY" ]]; then
  echo "no python >= 3.9 on $HOST; install one (python.org or brew) and retry." >&2
  exit 1
fi
say "using $RPY on $HOST"

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
if awk "BEGIN{exit !($BUDGET < $MODEL_GIB)}"; then
  say "neither one can hold it, so if this generates text, the pool did it"
else
  say "note: ${BUDGET} GiB is more than the model needs, so this run is not "\
      "demonstrating the memory split — lower BUDGET_OVERRIDE to force it"
fi

# --- 3. mirror the project and its deps -------------------------------------
say "syncing project to $HOST:$REMOTE_DIR"
$RSYNC -az --delete \
  --exclude .git --exclude __pycache__ --exclude 'bench/results' --exclude '*.so' \
  ./ "$HOST:$REMOTE_DIR/"

say "installing dependencies over there (quiet unless something is missing)"
$SSH "$HOST" "cd $REMOTE_DIR && $RPY -m pip install -q --user -r $REQS"

# The fused int4 kernel is what makes a CPU node worth having: weights stay
# packed, so it holds 8x more layers and decodes several times faster.
if [[ "$BACKEND" != "auto" ]]; then
  say "building the int4 kernel on $HOST"
  $SSH "$HOST" "cd $REMOTE_DIR && bash kernel/build.sh"
fi

# --- 4. make sure the weights are already over there ------------------------
# Fetching 4 GB while the coordinator waits on the handshake is a good way to
# hit a timeout, so do it up front where the progress bar is visible.
say "fetching model on $HOST if it is not cached yet"
$SSH "$HOST" "$RPY -c \"from huggingface_hub import snapshot_download as d; d('$MODEL')\" >/dev/null"

# --- 5. start the worker ----------------------------------------------------
# Hold the worker with a background ssh from this side rather than nohup'ing it
# over there: a remote background process keeps the ssh channel open and the
# command never returns. This way the worker also dies with the connection,
# which is exactly the cleanup we want.
WORKER_LOG=/tmp/hetero-remote-worker.log
SSH_PID=""
cleanup() {
  say "stopping worker on $HOST"
  [[ -n "$SSH_PID" ]] && kill "$SSH_PID" 2>/dev/null || true
  $SSH -n "$HOST" "pkill -f 'hetero.cli worker'" 2>/dev/null || true
}
trap cleanup EXIT

say "starting worker on $HOST"
$SSH -n "$HOST" "pkill -f 'hetero.cli worker' 2>/dev/null; true" || true
: > "$WORKER_LOG"
$SSH "$HOST" "cd $REMOTE_DIR && exec $RPY -u -m hetero.cli worker \
  --port $PORT --name '${HOST#*@}' --mem-gib $BUDGET --backend $BACKEND" \
  > "$WORKER_LOG" 2>&1 < /dev/null &
SSH_PID=$!

for _ in $(seq 1 120); do
  grep -q listening "$WORKER_LOG" && break
  kill -0 "$SSH_PID" 2>/dev/null || { echo "worker exited:"; cat "$WORKER_LOG"; exit 1; }
  sleep 1
done
if ! grep -q listening "$WORKER_LOG"; then
  echo "worker did not come up; its log says:" >&2
  cat "$WORKER_LOG" >&2
  exit 1
fi
sed -n '1,20p' "$WORKER_LOG"

# --- 6. drive it from here --------------------------------------------------
PEER_HOST="${HOST#*@}"
PEER="$PEER_HOST:$PORT:${HOST}:$BUDGET"
say "running"
echo
if [[ "$MODE" == "bench" ]]; then
  python3 -u scripts/bench_all.py --model "$MODEL" --peer "$PEER" --mem-gib "$BUDGET"
else
  python3 -u -m hetero.cli run --model "$MODEL" --peer "$PEER" \
    --name "$(hostname -s)" --mem-gib "$BUDGET" --objective latency \
    --max-tokens "${MAX_TOKENS:-64}" \
    --prompt "${PROMPT:-Explain, in two sentences, what pipeline parallelism is.}"
fi
