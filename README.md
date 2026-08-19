# hetero-infer

Run a model that no single machine in the pool can hold.

Your laptop can't fit the weights and your desktop can't either, but together
they can. This splits a transformer into contiguous slices of layers, puts each
slice on a different box, and streams hidden states between them over TCP. The
partitioner decides where to cut from measured per-machine memory and speed and
the measured link between them, so the two boxes don't have to be alike.

Everything below was measured on the hardware described in **Setup**. Nothing is
extrapolated or modelled.

---

## The headline

A 4-bit Qwen2.5-7B (**3.99 GiB** of weights) run across two nodes with a hard
**2.50 GiB budget each**. Neither node can hold the model; the pool runs it at
**24.1 tok/s**, which is the same speed as a single machine that *could* hold it
(24.3 tok/s) — the split itself costs nothing when the wire is free.

| | tok/s | peak memory held |
|---|---|---|
| One machine, whole model | 24.31 | 3.99 GiB — over budget, impossible under the 2.5 GiB cap |
| Two nodes, loopback | 24.08 | 2.36 GiB + 1.63 GiB |
| Two nodes, 1 GbE (emulated) | 23.66 | same |
| Two nodes, Wi-Fi 5 (emulated) | 22.07 | same |
| Two nodes, weak Wi-Fi (emulated) | 17.22 | same |

The pipeline's greedy output is **token-for-token identical** to the
single-process run, and the per-stage forward pass is **bit-exact** against
`mlx_lm`'s reference implementation.

And what the network costs, as a share of what the same partition achieves with
a free network:

| link | ms/token | on the wire | lost to GPU clock drop | **% of peak lost** |
|---|---|---|---|---|
| loopback (control) | 41.54 | 0.21 | −0.08 | −0.21% |
| 10 GbE (10 Gbps, 0.1 ms) | 41.78 | 0.33 | 0.05 | 0.40% |
| 1 GbE (1 Gbps, 0.3 ms) | 42.27 | 0.59 | 0.30 | 1.59% |
| Wi-Fi 5 (300 Mbps, 2 ms) | 45.32 | 2.72 | 1.22 | 8.22% |
| 100 Mbit (100 Mbps, 1 ms) | 43.83 | 1.95 | −1.17 | 1.15% |
| weak Wi-Fi (50 Mbps, 8 ms) | 58.07 | 10.42 | 6.16 | 28.17% |

**On a wired home network the network is nearly free.** Single-stream decode
moves one hidden state per stage crossing — 7593 bytes/token measured, hidden
size 3584 in fp16 — so a gigabit link costs about 1.6% and 10 GbE costs almost
nothing. It only starts to hurt when round-trip latency does: the 8 ms link
costs 28%.

The control row and the two anomalies are worth reading honestly. Repeating the
unshaped config 8 times gives a spread of 0.20 ms (0.5%), but the machine drifts
by around ±1 ms over a sweep this long, which is why each row is measured
against a free-network reference taken immediately before it. The 100 Mbit row's
1.15% is below that drift — treat everything under ~2% as "in the noise, the
link is free." The weak Wi-Fi row is far above it and is real.

---

## Two machines, for real

An M4 MacBook Pro and a 2013 Intel MacBook Pro, over home Wi-Fi. The Intel
machine cannot run MLX at all — it runs the NumPy backend with a fused int4
kernel — and it is 19x slower per layer. The pool still runs a model neither
machine is allowed to hold:

```
link measured: 47 Mbps, 8.4 ms RTT
[M4 laptop]       394 us/layer  [mlx]
[2013 Intel]     7725 us/layer  [numpy+q4]

plan[latency, head on stage 0]   predicted 125.13 ms/token
  stage 0  M4 laptop   layers  0-13 (14)  0.52 GiB / 0.53 GiB  embed head
  stage 1  2013 Intel  layers 14-27 (14)  0.40 GiB / 0.53 GiB

prompt 8 tok | TTFT 994 ms | 32 tokens | 8.07 tok/s
```

Model is 0.81 GiB; each node was capped at 0.53 GiB, so neither could hold it.
Measured 8.07 tok/s against 8.0 predicted.

Two things make this work at all.

**The output projection is a placement decision, not a fixed position.** The
vocabulary matmul is 152k wide: 1.7 ms on the M4's GPU, 853 ms on the CPU
backend. Pinning it to the last stage would have made the old Mac's head
dominate everything. Instead the planner chooses, and when it picks stage 0 the
ring returns hidden states rather than a token id.

**Prefill needs a different kernel than decode.** With one row the operation is
bandwidth-bound and the fused GEMV wins. With a whole prompt it is a real GEMM,
which BLAS does far better — except BLAS cannot read 4-bit weights, and
expanding them in NumPy costs ~150 ms per projection because every step
allocates a temporary. `q4_dequant` does that expansion in one C pass into a
reused 512-row buffer that stays in cache. Measured on the 2013 machine, one
8960x1536 projection with a 57-row prompt:

```
NumPy dequant + BLAS   156.6 ms
fused GEMV, row by row 100.7 ms
C dequant + BLAS        51.8 ms
```

The crossover between the two paths is ~16 rows. End to end this took
time-to-first-token on the pair from 13.0 s to 3.3 s.

**A fused int4 GEMV, because decode is bandwidth-bound.** Dequantising 4-bit
weights to float32 at load moves eight times the bytes per token. The kernel in
`kernel/q4gemv.c` unpacks into registers instead, with AVX2 and NEON paths:

```
                          M4 (NEON, 8 thr)     2013 Intel (AVX2, 4 thr)
NumPy float32 GEMV            0.81 ms                3.75 ms
fused int4 GEMV               0.28 ms  (2.9x)        1.69 ms  (2.2x)
```

It matches the NumPy path to 3e-7 and generates identical tokens. The larger
win is memory: weights stay packed, so a CPU node holds **8x more layers**.

Short prefills go through the same kernel row by row rather than dequantising
for BLAS, which is what a chat-length prompt actually wants — that alone took
TTFT on this pair from 9.8 s to 0.99 s.

## Three findings worth the trouble

**1. A slow link makes your compute slower too.** The `clock drop` column is not
transfer time — it is the stage's own forward pass getting *slower*. Idling the
GPU while waiting on the wire lets it drop clocks, so the next token's compute
costs more. Measured directly, outside the pipeline:

```
idle gap  0.0 ms before each step -> compute 23.46 ms
idle gap  2.0 ms                  -> compute 23.90 ms
idle gap  8.0 ms                  -> compute 27.55 ms   (+17%)
```

At 8 ms RTT this is 6.2 ms/token — 37% of that link's total cost is compute the
GPU never gets back. Accounting for the network only as "bytes ÷ bandwidth"
misses more than a third of the bill.

**2. Shrinking activations is the wrong lever for decode.** int8 activations
halve what goes on the wire (8608 → 5121 bytes/token) and buy essentially
nothing: 45.35 → 43.62 ms/token on 100 Mbit, and nothing at all on the 8 ms
link. Single-token decode is latency-bound, not bandwidth-bound — you are paying
for the round trip, not the payload.

Quality is checkpoint-dependent and worth checking before you reach for it: on
the 7B, int8 activations reproduce the fp16 output exactly for 40 greedy tokens,
but on Qwen2.5-3B they diverge after 6 of 24. Halving the payload is not free,
and on decode it buys nothing anyway.

Prefill is the opposite, because it puts whole chunks on the wire. With a
512-token prompt in one exposed chunk on the weak link, int8 is a genuine win:
**2677 → 2417 ms**.

**3. Overlap recovers almost all of a bad link.** Two ways, both measured:

*Concurrent streams.* One request leaves each stage idle while the other works.
Several in flight fill the gaps — on the 8 ms link, aggregate throughput goes
**16.18 → 23.24 tok/s** at 4 streams, which is the loopback rate. The link is
completely hidden, and the busiest stage goes from 46% to 99% utilisation.

| streams | loopback tok/s | weak Wi-Fi tok/s | busiest stage util |
|---|---|---|---|
| 1 | 21.19 | 16.18 | 46–56% |
| 2 | 22.69 | 21.93 | 92–98% |
| 4 | 23.08 | 23.24 | 99% |
| 8 | 23.07 | 23.18 | 99% |

*Chunked prefill.* Splitting the prompt lets stage 0 work on chunk k+1 while
stage 1 is still on chunk k. On the weak link a 512-token prompt goes **2748 ms
in one chunk → 2173 ms in 32-token chunks (−21%)**. On loopback chunking is a
small loss (2112 → 2147 ms), which is the honest shape of it: chunking buys you
nothing you weren't already losing to the wire.

---

## How it works

```
   stage 0                      stage 1                    stage N-1
 ┌──────────┐  hidden states  ┌──────────┐              ┌───────────┐
 │ embed    │ ───7.5 KB/tok─► │ layers   │ ──► ... ───► │ layers    │
 │ layers   │                 │          │              │ norm+head │
 └──────────┘ ◄───────────────────── token id (4 B) ────└───────────┘
```

**The ring returns a token id, not logits.** The last stage samples and sends
back the chosen token. The vocabulary is 152k wide, so returning logits would
put ~300 KB on the wire per step instead of a handful of bytes — a 40x
difference that would make the network dominate on any home link.

**`shard.py` — nothing ever holds the whole model.** The safetensors header is
parsed for shapes and offsets without touching tensor data, and a stage pulls
only the tensors it owns. Modules take their loaded arrays in `__init__` and
never random-initialise, so peak memory during load is the shard size rather
than twice it. A node refuses a shard that exceeds its declared budget, and the
peaks are asserted in the test suite.

**`planner.py` — a DP over cut points.** Stages are contiguous slices assigned in
ring order, so a plan is just a list of cuts. Two objectives:

- `latency` — single stream, no overlap, so the cost is the *sum* of stage times
  plus every crossing. A faster node should get more layers.
- `throughput` — microbatched, stages overlap, so the cost is the *max* over
  stages. The aim is to balance, not to hoard.

Memory is a hard constraint (weights + KV cache at the target context), ties
break toward leaving headroom, and tied embeddings are accounted for — a tied
model makes the head stage carry its own copy of the embedding table.

The predictions are good: **41.31 ms/token predicted, 41.54 measured (−0.5%)**.
Asked to plan a pool it hasn't got, it moves layers the way you'd hope:

```
$ python3 -m hetero.cli plan --node laptop:1.6:1.0 --node desktop:3.5:2.0
plan[latency]     stage 0 laptop  layers 0-3   (4)   stage 1 desktop layers 4-27  (24)
plan[throughput]  stage 0 laptop  layers 0-9  (10)   stage 1 desktop layers 10-27 (18)
```

Latency mode dumps 24 of 28 layers on the faster box; throughput mode balances
by *time*, not by layer count.

**`transport.py` — framed TCP with background I/O.** `[u32 header_len][json
header][payload]`, `TCP_NODELAY`, and separate sender/receiver threads behind
queues so a stage never blocks on the wire when it could be computing. That is
what makes the overlap results above possible. It also carries an optional link
shaper for controlled sweeps (see **Caveats**).

**Nodes measure themselves.** On connect, each node times a probe shard and
reports its own per-layer cost and memory budget, and the coordinator times the
real wire to it. Nothing assumes the pool is symmetric. Profiling is strictly
serialised — two nodes profiling at once (exactly what happens when you emulate
a pool on one box) makes every node look 3-7x slower than it is and the planner
then cuts in the wrong place.

---

## Setup and running

Apple M4, 16 GB unified memory, macOS 15.3.1, MLX 0.32, Python 3.12.
Model: `mlx-community/Qwen2.5-7B-Instruct-4bit` (28 layers, hidden 3584, 4-bit
group 64).

```bash
pip install -r requirements.txt
```

Two nodes on this machine, 2.5 GiB budget each:

```bash
./scripts/run_local_2node.sh
```

Two real machines, one command. Turn on Remote Login on the other box
(System Settings -> General -> Sharing), then:

```bash
./scripts/pool.sh you@desktop.local          # generate
./scripts/pool.sh you@desktop.local bench    # full measurement suite
```

It mirrors the project over, installs what it needs, fetches the weights,
starts the far side, runs, and shuts it down again. Budgets are picked at 65%
of the model size so that neither machine could hold it alone — if text comes
out, the pool produced it.

To drive the two sides by hand instead, start the far one:

```bash
python3 -m hetero.cli worker --port 29501 --name laptop --mem-gib 6
```

and point this one at it:

```bash
PEER=192.168.1.42:29501:laptop:6 ./scripts/run_lan.sh
```

Other commands:

```bash
python3 -m hetero.cli profile                      # measure this box
python3 -m hetero.cli plan --node a:2.5:1 --node b:2.5:2   # what-if, no pool needed
python3 scripts/bench_all.py --peer 127.0.0.1:29501:node-b:2.5 --mem-gib 2.5
```

A pool that genuinely lacks the memory says so instead of thrashing:

```
$ python3 -m hetero.cli run --peer 127.0.0.1:29501:node-b:1.5 --mem-gib 1.5
error: no feasible split: model needs 3.99 GiB of weights plus KV cache,
       pool offers 3.00 GiB across 2 nodes
```

Three nodes work the same way — 3 x 1.7 GiB budgets run the same 7B at 23.7 tok/s.

### Tests

```bash
python3 tests/test_correctness.py    # stages vs mlx_lm reference, bit-exact
python3 tests/test_pipeline.py       # two processes, real sockets, vs single process
```

`test_correctness.py` checks the split model against `mlx_lm` (max|diff| = 0,
at several cut points and on both tied and untied checkpoints) and checks
chunked prefill against the reference run the same way — both implementations
drift from one-shot prefill identically, since fp16 batched and cached attention
take different kernel paths, so tracking the reference chunk-for-chunk is the
meaningful check.

`test_pipeline.py` spawns a real worker subprocess, asserts no stage ever holds
the whole model, and asserts the generated tokens match the single-process run
exactly.

---

## Caveats

**Which numbers come from where.** The link sweep, codec, concurrency and
prefill tables are two processes on one M4 with an emulated link. The
two-machine section is real hardware over real Wi-Fi. The 7B numbers are
single-machine; the cross-machine run used the 1.5B, because 28 layers of a 7B
on a 2013 CPU is a slideshow regardless of how good the kernel is.

**A slow node cannot make generation faster.** Pipeline parallelism splits
memory, not work — every token still crosses every layer in order. Given a free
choice, the partitioner assigns a much slower node zero layers, which is the
correct answer. That node earns its place only when the model does not fit
without it.

**Sharing one GPU understates concurrency.** Both stages contend for the same
GPU here, so they cannot truly compute at once. That is why loopback throughput
barely improves past 2 streams (21.2 → 23.1 tok/s) while the busiest stage sits
at 99% — the ceiling is one GPU, not the pipeline. On two real machines this
should scale further; the weak-Wi-Fi column, where the gain comes from hiding
*wire* time rather than overlapping compute, is the trustworthy one.

**The slow links are emulated, not real.** A shaper delays each frame by
`rtt/2 + bytes/bandwidth` at the sender. macOS coalesces timers, so a naive
`time.sleep(8ms)` lands ~3.2 ms late and would have silently inflated every
shaped row; the shaper halves the remaining time repeatedly and spins the last
stretch, which holds error under 0.6 ms at an 8 ms target. Real Wi-Fi also has
jitter, loss and contention that this does not reproduce. The measured loopback
link between the two processes is 25 Gbps at 0.048 ms RTT.

**One architecture.** Dense `qwen2` quantised checkpoints. Anything else raises
`NotImplementedError` rather than failing obscurely. The partitioner, transport
and runtime are architecture-agnostic — adding one means writing its
`DecoderLayer` in `model.py`.

**Scope.** Pipeline parallelism only, no tensor parallelism. Contiguous layer
slices, one ring, no failure recovery: if a node dies mid-request the request
dies with it.

---

## Layout

```
hetero/
  shard.py        safetensors header parsing, ownership, selective loading
  model.py        Qwen2 as an addressable stage; quantised linears, KV cache
  planner.py      DP over cut points; memory as a hard constraint
  transport.py    framed TCP, background I/O threads, link shaper, fp16/int8 codec
  profile.py      per-node compute/memory probes, link RTT and bandwidth
  engine.py       stage runner, ring message handling, sampling
  coordinator.py  stage 0 + conductor: probe, plan, wire, drive
  bench.py        decode / throughput / prefill / single-node measurements
  cli.py          profile | plan | worker | run
scripts/          local and LAN launchers, full benchmark suite
tests/            reference-equivalence and end-to-end pipeline tests
bench/results/    results.json from the run reported above
```
