# hetero-infer — measurement notes

Longer-form results that used to live in the README.

## Two machines, for real

An M4 MacBook Pro and a 2013 Intel MacBook Pro over home Wi-Fi. The Intel
machine cannot run MLX at all; it runs the NumPy backend with a fused int4
kernel, and it is 19x slower per layer. The pool still runs a model neither
machine is allowed to hold:

```
link measured: 47 Mbps, 8.4 ms RTT
[M4 laptop]       394 us/layer  [mlx]
[2013 Intel]     7725 us/layer  [numpy+q4]

plan[latency, head on stage 0]   predicted 125.13 ms/token
  stage 0  M4 laptop   layers  0-13 (14)  0.52 GiB / 0.53 GiB  embed head
  stage 1  2013 Intel  layers 14-27 (14)  0.40 GiB / 0.53 GiB

prompt 57 tok | TTFT 3.0 s | 102 tokens | 9.27 tok/s
```

Model is 0.81 GiB, each node capped at 0.53 GiB. Measured 8.07 tok/s against
8.0 predicted.

Two things make it work. The output projection is a placement decision rather
than a fixed position: the vocabulary matmul is 152k wide, 1.7 ms on the M4's
GPU and 853 ms on the CPU backend, so pinning it to the last stage would let
the old Mac's head dominate everything. The planner chooses instead, and when
it picks stage 0 the ring returns hidden states rather than a token id.

And prefill needs a different kernel than decode. With one row the operation is
bandwidth-bound and the fused GEMV wins; with a whole prompt it is a real GEMM,
which BLAS does far better, except BLAS cannot read 4-bit weights and expanding
them in NumPy costs ~150 ms per projection because every step allocates a
temporary. `q4_dequant` does the expansion in one C pass into a reused 512-row
buffer that stays in cache. One 8960x1536 projection, 57-row prompt, on the
2013 machine:

```
NumPy dequant + BLAS   156.6 ms
fused GEMV, row by row 100.7 ms
C dequant + BLAS        51.8 ms
```

Crossover is ~16 rows. End to end this took TTFT on the pair from 13.0 s to
3.3 s.

## The fused int4 GEMV

Dequantising 4-bit weights to float32 at load moves eight times the bytes per
token. `kernel/q4gemv.c` unpacks into registers instead, with AVX2 and NEON
paths:

```
one 8960x1536 projection      M4 (NEON)          2013 Intel (AVX2)
NumPy float32 GEMV              0.81 ms              3.75 ms
fused int4 GEMV, float          0.24 ms              1.41 ms
fused int4 GEMV, int16 dot         -                 1.06 ms
```

On AVX2 the dot product runs in 16-bit integers through `vpmaddwd`. Bisecting
the loop stage by stage on the 2013 machine showed where the time went: a
load-only pass sustains 10-11 GB/s, and adding just the nibble extract dropped
it to 3.7, so the cost was the per-word variable shift rather than the
arithmetic. Splitting a whole 32-byte block with one shift and staying in 8-
then 16-bit integers avoids both that and the integer-to-float conversion. `x`
is quantised to int16 once per call, costing relative error ~4e-5, which did
not change the generated tokens.

It matches the NumPy path to 3e-7 and generates identical tokens. The larger
win is memory: weights stay packed, so a CPU node holds 8x more layers. Short
prefills go through the same kernel row by row rather than dequantising for
BLAS, which took TTFT on this pair from 9.8 s to 0.99 s.

## A slow link makes your compute slower too

The `clock drop` column in the README's link table is not transfer time. It is
the stage's own forward pass getting slower, because idling the GPU while
waiting on the wire lets it drop clocks and the next token's compute costs
more. Measured directly, outside the pipeline:

```
idle gap  0.0 ms before each step -> compute 23.46 ms
idle gap  2.0 ms                  -> compute 23.90 ms
idle gap  8.0 ms                  -> compute 27.55 ms   (+17%)
```

At 8 ms RTT that is 6.2 ms/token, so 37% of that link's total cost is compute
the GPU never gets back. Accounting for the network as bytes ÷ bandwidth misses
more than a third of the bill.

## Shrinking activations is the wrong lever for decode

int8 activations halve what goes on the wire, 8608 → 5121 bytes/token, and buy
essentially nothing: 45.35 → 43.62 ms/token on 100 Mbit, and nothing at all on
the 8 ms link. Single-token decode is latency-bound, not bandwidth-bound. You
pay for the round trip, not the payload.

Quality is checkpoint-dependent and worth checking first. On the 7B, int8
activations reproduce the fp16 output exactly for 40 greedy tokens; on
Qwen2.5-3B they diverge after 6 of 24.

Prefill is the opposite, because it puts whole chunks on the wire. With a
512-token prompt in one exposed chunk on the weak link, int8 is a genuine win:
2677 → 2417 ms.

## Overlap recovers almost all of a bad link

Concurrent streams. One request leaves each stage idle while the other works,
and several in flight fill the gaps. On the 8 ms link aggregate throughput goes
16.18 → 23.24 tok/s at 4 streams, which is the loopback rate, and the busiest
stage goes from 46% to 99% utilisation:

| streams | loopback tok/s | weak Wi-Fi tok/s | busiest stage util |
|---|---|---|---|
| 1 | 21.19 | 16.18 | 46–56% |
| 2 | 22.69 | 21.93 | 92–98% |
| 4 | 23.08 | 23.24 | 99% |
| 8 | 23.07 | 23.18 | 99% |

Chunked prefill. Splitting the prompt lets stage 0 work on chunk k+1 while
stage 1 is still on chunk k. On the weak link a 512-token prompt goes 2748 ms
in one chunk to 2173 ms in 32-token chunks, a 21% gain. On loopback chunking is
a small loss, 2112 → 2147 ms, which is the honest shape of it: chunking buys
back what you were losing to the wire and nothing more.
