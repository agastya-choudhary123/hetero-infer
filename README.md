# hetero-infer

Pipeline-parallel LLM inference across machines that are each too small to
hold the model. The model is cut into contiguous slices of layers, each slice
runs on a different machine, and hidden states are passed between them over
TCP. The cut points come from measuring each machine's memory and speed and
the network between them, so the machines don't have to match. Apple silicon
nodes run MLX. Other machines (Intel Macs, Linux) use a NumPy backend with a
fused int4 C kernel.

On one M4, a 4-bit Qwen2.5-7B (3.99 GiB of weights) split across two nodes
capped at 2.5 GiB each runs at 24.1 tok/s. The whole model in one process
runs at 24.3 tok/s. Greedy output is token-for-token identical to the
single-process run, and each stage's forward pass exactly matches `mlx_lm`.

[NOTES.md](NOTES.md) has the longer results: a real two-machine run (an M4
plus a 2013 Intel MacBook Pro over Wi-Fi), the int4 kernel, and the
concurrency measurements.

## Setup

```sh
pip install -r requirements.txt        # Apple silicon (MLX)
pip install -r requirements-cpu.txt    # machines without MLX
```

Only dense `qwen2` quantized checkpoints are supported. Anything else raises
`NotImplementedError`.

Two nodes on one machine, each with a 2.5 GiB budget:

```sh
./scripts/run_local_2node.sh
```

Two real machines. Turn on Remote Login on the other machine, then run:

```sh
./scripts/pool.sh you@desktop.local          # generate text
./scripts/pool.sh you@desktop.local bench    # full benchmark suite
```

That script copies the project over, installs dependencies, downloads the
weights, starts the remote worker, runs, and shuts it down again. Each node's
budget is set to 65% of the model size, so neither machine could run the
model alone.

To run the two sides by hand:

```sh
python3 -m hetero.cli worker --port 29501 --name laptop --mem-gib 6   # remote
PEER=192.168.1.42:29501:laptop:6 ./scripts/run_lan.sh                 # local
```

## Usage

```sh
python3 -m hetero.cli profile                              # measure this machine
python3 -m hetero.cli plan --node a:2.5:1 --node b:2.5:2   # try a split without a pool
python3 -m hetero.cli run --peer 127.0.0.1:29501:node-b:2.5 --mem-gib 2.5
python3 scripts/bench_all.py --peer 127.0.0.1:29501:node-b:2.5 --mem-gib 2.5
```

If the pool doesn't have enough memory, it says so up front:

```
error: no feasible split: model needs 3.99 GiB of weights plus KV cache,
       pool offers 3.00 GiB across 2 nodes
```

Three nodes work the same way. Three 1.7 GiB budgets run the 7B at
23.7 tok/s.

## Benchmarks

Qwen2.5-7B 4-bit on an Apple M4 (16 GB). The "emulated" rows are two
processes on one machine with a simulated link (see Caveats). Raw results
are in `bench/results/results.json`.

| | tok/s | peak memory |
|---|---|---|
| One process, whole model | 24.31 | 3.99 GiB (over budget) |
| Two nodes, loopback | 24.08 | 2.36 + 1.63 GiB |
| Two nodes, 1 GbE (emulated) | 23.66 | same |
| Two nodes, Wi-Fi 5 (emulated) | 22.07 | same |
| Two nodes, weak Wi-Fi (emulated) | 17.22 | same |

Here's where the network overhead comes from:

| link | ms/token | wire time | GPU slowdown | % lost vs. free link |
|---|---|---|---|---|
| loopback | 41.54 | 0.21 | −0.08 | −0.21% |
| 10 GbE (0.1 ms RTT) | 41.78 | 0.33 | 0.05 | 0.40% |
| 1 GbE (0.3 ms) | 42.27 | 0.59 | 0.30 | 1.59% |
| 100 Mbit (1 ms) | 43.83 | 1.95 | −1.17 | 1.15% |
| Wi-Fi 5 (300 Mbps, 2 ms) | 45.32 | 2.72 | 1.22 | 8.22% |

"GPU slowdown" means the stage's own forward pass got slower, because the GPU
lowered its clocks while it waited on the network. On slow links that's more
than a third of the total cost. With four requests in flight, weak Wi-Fi goes
from 16.18 to 23.24 tok/s, about the same as loopback (details in NOTES.md).

The planner predicted 41.31 ms/token and the measured result was
41.54 ms/token.

## How it works

```
   stage 0                      stage 1                    stage N-1
 ┌──────────┐  hidden states  ┌──────────┐              ┌───────────┐
 │ embed    │ ───7.5 KB/tok─► │ layers   │ ──► ... ───► │ layers    │
 │ layers   │                 │          │              │ norm+head │
 └──────────┘ ◄───────────────────── token id (4 B) ────└───────────┘
```

**Token ids, not logits.** The last stage samples the next token and sends
back just its id. Sending logits for a 152k vocabulary would be about 300 KB
per token. On very slow nodes the planner can put the output head on stage 0
instead, and then hidden states come back around the ring.

**Loading shards (`shard.py`).** Shapes and offsets come from the
safetensors header, and each stage reads only the tensors it owns. Modules
are built directly from the loaded arrays with no random initialization, so
peak memory during loading is the shard size. A node refuses any shard that's
bigger than its budget, and the tests check this.

**Planning (`planner.py`).** A dynamic program over cut points. There are two
objectives. `latency` is for a single stream and minimizes the sum of stage
times plus network crossings. `throughput` is for microbatching and minimizes
the slowest stage. Memory is a hard constraint, ties go to the plan with more
headroom, and tied embeddings are counted twice when the head and the
embedding table end up on different stages.

**Transport (`transport.py`).** Framed TCP (`[u32 header_len][json
header][payload]`) with `TCP_NODELAY`. Each stage has separate sender and
receiver threads, so it can keep computing while data moves.

**Profiling.** When a node connects, it runs a probe shard and reports its
per-layer time and memory budget, and the coordinator measures the link.
Nodes are profiled one at a time. When they ran concurrently on one machine,
each node looked 3–7x slower than it really was, and the planner cut in the
wrong places.

## Caveats

- **Where the numbers come from.** The link, codec, concurrency, and prefill
  results are two processes on one M4 with an emulated link. The two-machine
  results in NOTES.md use real hardware and real Wi-Fi, but with Qwen2.5-1.5B,
  because the 7B is far too slow on a 2013 CPU.
- **A slow node can't make generation faster.** Pipeline parallelism splits
  memory, not work. Every token still goes through every layer in order. If
  a much slower node isn't needed for memory, the planner gives it zero
  layers.
- **One shared GPU understates concurrency.** On one machine both stages
  share the same GPU, so loopback throughput barely improves past 2
  streams. The weak Wi-Fi results, where the gain comes from hiding network
  time, are the more meaningful ones.
- **Slow links are emulated.** The sender delays each frame by
  `rtt/2 + bytes/bandwidth`. macOS timer coalescing makes `time.sleep(8ms)`
  about 3 ms late, so the shaper sleeps in shrinking steps and spins at the
  end, which keeps the error under 0.6 ms. It doesn't simulate jitter, packet
  loss, or contention.
- **Only pipeline parallelism**, and no failure recovery. If a node dies, the
  request fails.
- **Only Qwen2.** The planner, transport, and runtime don't depend on the
  architecture. Adding another means writing its `DecoderLayer` in
  `model.py`.

## Tests

```sh
python3 tests/test_correctness.py    # each stage vs. mlx_lm, exact match
python3 tests/test_pipeline.py       # two processes over real sockets vs. one process
```

`test_correctness.py` compares the split model against `mlx_lm` (max abs
diff = 0) at several cut points, on both tied and untied checkpoints. It
checks chunked prefill against the reference chunk by chunk, because fp16
batched and cached attention use different kernels, and both implementations
drift from one-shot prefill in the same way.

`test_pipeline.py` starts a real worker subprocess. It checks that no stage
ever holds the whole model and that the output tokens exactly match the
single-process run.

## Layout

```
hetero/
  shard.py        safetensors header parsing and selective loading
  model.py        Qwen2 as a stage (MLX): quantized linears, KV cache
  cpu_model.py    the same stage in NumPy, for machines without MLX
  backends.py     picks MLX or NumPy
  planner.py      DP over cut points
  transport.py    framed TCP, background I/O, link shaper, fp16/int8 codec
  profile.py      compute, memory, and link probes
  engine.py       stage runner and sampling
  coordinator.py  stage 0: probe, plan, connect the ring, generate
  worker.py       pool member
  bench.py        benchmarks
  cli.py          profile | plan | worker | run
kernel/           fused int4 GEMV in C (AVX2 and NEON), build and autotune scripts
scripts/          launchers and the benchmark suite
tests/
bench/results/    results.json for the numbers above
```
