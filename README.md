hetero-infer
------------

hetero-infer runs a transformer across several machines when no single one can
hold it. It splits the model into contiguous slices of layers, puts each slice
on a different box, and streams hidden states between them over TCP. The
partitioner picks the cut points from measured per-machine memory and speed and
the measured link between them, so the machines do not have to be alike.

A 4-bit Qwen2.5-7B, 3.99 GiB of weights, runs across two nodes capped at 2.50
GiB each. Neither node can hold it. The pool runs it at 24.1 tok/s, against
24.3 tok/s for a single machine that could hold it.

Greedy output is token-for-token identical to the single-process run, and each
stage's forward pass is bit-exact against `mlx_lm`.

### Documentation quick links

* [Setup](#setup)
* [Usage](#usage)
* [Benchmarks](#benchmarks)
* [Caveats](#caveats)
* [NOTES.md](NOTES.md) — two real machines, the int4 kernel, and the overlap results

### Requirements

Apple M4, 16 GB unified memory, macOS 15.3.1, MLX 0.32, Python 3.12. Dense
`qwen2` quantised checkpoints only.

```
$ pip install -r requirements.txt
```

### Setup

Two nodes on one machine, 2.5 GiB budget each:

```
$ ./scripts/run_local_2node.sh
```

Two real machines, one command. Turn on Remote Login on the other box, then:

```
$ ./scripts/pool.sh you@desktop.local          # generate
$ ./scripts/pool.sh you@desktop.local bench    # full measurement suite
```

That mirrors the project over, installs what it needs, fetches the weights,
starts the far side, runs, and shuts it down. Budgets are picked at 65% of the
model size so neither machine could hold it alone. If text comes out, the pool
produced it.

To drive the two sides by hand, start the far one and point this one at it:

```
$ python3 -m hetero.cli worker --port 29501 --name laptop --mem-gib 6
$ PEER=192.168.1.42:29501:laptop:6 ./scripts/run_lan.sh
```

### Usage

```
$ python3 -m hetero.cli profile                             # measure this box
$ python3 -m hetero.cli plan --node a:2.5:1 --node b:2.5:2  # what-if, no pool needed
$ python3 scripts/bench_all.py --peer 127.0.0.1:29501:node-b:2.5 --mem-gib 2.5
```

A pool that genuinely lacks the memory says so instead of thrashing:

```
$ python3 -m hetero.cli run --peer 127.0.0.1:29501:node-b:1.5 --mem-gib 1.5
error: no feasible split: model needs 3.99 GiB of weights plus KV cache,
       pool offers 3.00 GiB across 2 nodes
```

Three nodes work the same way: 3 x 1.7 GiB budgets run the same 7B at 23.7
tok/s.

### Benchmarks

| | tok/s | peak memory held |
|---|---|---|
| One machine, whole model | 24.31 | 3.99 GiB, over budget |
| Two nodes, loopback | 24.08 | 2.36 + 1.63 GiB |
| Two nodes, 1 GbE (emulated) | 23.66 | same |
| Two nodes, Wi-Fi 5 (emulated) | 22.07 | same |
| Two nodes, weak Wi-Fi (emulated) | 17.22 | same |

What the network actually costs, as a share of what the same partition achieves
with a free one:

| link | ms/token | on the wire | lost to GPU clock drop | % of peak lost |
|---|---|---|---|---|
| loopback (control) | 41.54 | 0.21 | −0.08 | −0.21% |
| 10 GbE (10 Gbps, 0.1 ms) | 41.78 | 0.33 | 0.05 | 0.40% |
| 1 GbE (1 Gbps, 0.3 ms) | 42.27 | 0.59 | 0.30 | 1.59% |
| 100 Mbit (100 Mbps, 1 ms) | 43.83 | 1.95 | −1.17 | 1.15% |
| Wi-Fi 5 (300 Mbps, 2 ms) | 45.32 | 2.72 | 1.22 | 8.22% |

The clock-drop column is not transfer time. It is the stage's own forward pass
getting slower because the GPU dropped clocks while waiting on the wire, and it
is worth more than a third of the bill on a slow link. [NOTES.md](NOTES.md) has
that measured in isolation, along with the concurrency results: four streams in
flight take weak Wi-Fi from 16.18 to 23.24 tok/s, which is the loopback rate.

The planner's predictions hold up: 41.31 ms/token predicted, 41.54 measured.

### How it works

```
   stage 0                      stage 1                    stage N-1
 ┌──────────┐  hidden states  ┌──────────┐              ┌───────────┐
 │ embed    │ ───7.5 KB/tok─► │ layers   │ ──► ... ───► │ layers    │
 │ layers   │                 │          │              │ norm+head │
 └──────────┘ ◄───────────────────── token id (4 B) ────└───────────┘
```

The ring returns a token id, not logits. The last stage samples and sends back
the chosen token. The vocabulary is 152k wide, so returning logits would put
~300 KB on the wire per step instead of a handful of bytes, and that 40x would
make the network dominate on any home link.

`shard.py` means nothing ever holds the whole model. The safetensors header is
parsed for shapes and offsets without touching tensor data, and a stage pulls
only the tensors it owns. Modules take their loaded arrays in `__init__` and
never random-initialise, so peak memory during load is the shard size rather
than twice it. A node refuses a shard exceeding its declared budget, and the
peaks are asserted in the tests.

`planner.py` is a DP over cut points. Stages are contiguous slices assigned in
ring order, so a plan is a list of cuts. Two objectives: `latency` is single
stream with no overlap, so cost is the sum of stage times plus every crossing
and a faster node should get more layers; `throughput` is microbatched with
stages overlapping, so cost is the max over stages and the aim is to balance.
Memory is a hard constraint, ties break toward headroom, and tied embeddings
are accounted for, since a tied model makes the head stage carry its own copy
of the embedding table.

`transport.py` is framed TCP, `[u32 header_len][json header][payload]`, with
`TCP_NODELAY` and separate sender and receiver threads behind queues so a stage
never blocks on the wire when it could be computing. That is what makes the
overlap results possible.

Nodes measure themselves. On connect each node times a probe shard and reports
its own per-layer cost and memory budget, and the coordinator times the real
wire to it. Profiling is strictly serialised: two nodes profiling at once,
which is exactly what happens when you emulate a pool on one box, makes every
node look 3-7x slower than it is and the planner then cuts in the wrong place.

### Caveats

Which numbers come from where. The link, codec, concurrency and prefill tables
are two processes on one M4 with an emulated link. The two-machine section in
NOTES.md is real hardware over real Wi-Fi. The 7B numbers are single-machine;
the cross-machine run used the 1.5B, because 28 layers of a 7B on a 2013 CPU is
a slideshow regardless of how good the kernel is.

A slow node cannot make generation faster. Pipeline parallelism splits memory,
not work, and every token still crosses every layer in order. Given a free
choice the partitioner assigns a much slower node zero layers, which is the
correct answer. That node earns its place only when the model does not fit
without it.

Sharing one GPU understates concurrency. Both stages contend for the same GPU
here, so they cannot truly compute at once, which is why loopback throughput
barely improves past 2 streams while the busiest stage sits at 99%. The ceiling
is one GPU, not the pipeline. The weak-Wi-Fi column, where the gain comes from
hiding wire time rather than overlapping compute, is the trustworthy one.

The slow links are emulated. A shaper delays each frame by `rtt/2 +
bytes/bandwidth` at the sender. macOS coalesces timers, so a naive
`time.sleep(8ms)` lands ~3.2 ms late and would have silently inflated every
shaped row; the shaper halves the remaining time repeatedly and spins the last
stretch, holding error under 0.6 ms at an 8 ms target. Real Wi-Fi also has
jitter, loss and contention that this does not reproduce.

One architecture. Anything but dense `qwen2` raises `NotImplementedError`
rather than failing obscurely. The partitioner, transport and runtime are
architecture-agnostic, so adding one means writing its `DecoderLayer` in
`model.py`.

Pipeline parallelism only, no tensor parallelism. Contiguous layer slices, one
ring, no failure recovery: if a node dies mid-request, the request dies with it.

### Tests

```
$ python3 tests/test_correctness.py    # stages vs mlx_lm reference, bit-exact
$ python3 tests/test_pipeline.py       # two processes, real sockets, vs single process
```

`test_correctness.py` checks the split model against `mlx_lm`, max|diff| = 0, at
several cut points and on both tied and untied checkpoints. It checks chunked
prefill against the reference the same way: both implementations drift from
one-shot prefill identically, since fp16 batched and cached attention take
different kernel paths, so tracking the reference chunk-for-chunk is the
meaningful check.

`test_pipeline.py` spawns a real worker subprocess, asserts no stage ever holds
the whole model, and asserts the tokens match the single-process run exactly.

### Layout

```
hetero/
  shard.py        safetensors header parsing, ownership, selective loading
  model.py        Qwen2 as an addressable stage; quantised linears, KV cache
  planner.py      DP over cut points; memory as a hard constraint
  transport.py    framed TCP, background I/O, link shaper, fp16/int8 codec
  profile.py      per-node compute/memory probes, link RTT and bandwidth
  engine.py       stage runner, ring message handling, sampling
  coordinator.py  stage 0 and conductor: probe, plan, wire, drive
  bench.py        decode / throughput / prefill / single-node measurements
  cli.py          profile | plan | worker | run
kernel/q4gemv.c   fused int4 GEMV, AVX2 and NEON
scripts/          local and LAN launchers, full benchmark suite
tests/            reference-equivalence and end-to-end pipeline tests
bench/results/    results.json from the run reported above
```
