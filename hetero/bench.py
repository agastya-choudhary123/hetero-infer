"""Measurement harness.

Everything here is measured, never modelled. The two headline numbers are:

  tok/s          steady-state decode rate for a model no single node can hold
  network loss   the share of each token's wall time that is not stage compute

The second is computed as (wall - sum of per-stage compute) / wall, where the
per-stage compute is timed on the nodes themselves during the same run. That
makes the ceiling a measured quantity — the same partition on the same silicon
with a free network — rather than an estimate.
"""
from __future__ import annotations

import json
import statistics
import time
from typing import Dict, List, Optional

from .coordinator import Coordinator


def decode_bench(c: Coordinator, prompt: str, warmup: int = 8, measure: int = 48,
                 req: str = "bench") -> Dict:
    """Time steady-state single-stream decode, with per-token latencies."""
    tok = c.load_tokenizer()
    ids = tok(prompt)["input_ids"]
    c._send_prefill(req, ids, 128)
    _, last = c.recv_token()
    for i in range(warmup):
        c._send_decode(req, last, i + 1)
        _, last = c.recv_token()

    c.reset_stats()
    lat = []
    t_start = time.perf_counter()
    for i in range(measure):
        t0 = time.perf_counter()
        c._send_decode(req, last, warmup + i + 1)
        _, last = c.recv_token()
        lat.append(time.perf_counter() - t0)
    wall = time.perf_counter() - t_start
    stats = c.query_stats()
    c.free(req)

    stage_compute = [c.runner.stats.compute_s] + [s["compute_s"] for s in stats]
    names = [c.nodes[0].name] + [s["name"] for s in stats]
    bytes_out = c.data_out.stats()["bytes_sent"] - getattr(c, "data_out_reset_bytes", 0)
    for s in stats:
        bytes_out += s["link"]["bytes_sent"]

    per_tok = wall / measure
    compute_tok = sum(stage_compute) / measure
    lat_sorted = sorted(lat)
    return {
        "tokens": measure,
        "wall_s": wall,
        "tok_per_s": measure / wall,
        "ms_per_token": per_tok * 1e3,
        "p50_ms": lat_sorted[len(lat) // 2] * 1e3,
        "p99_ms": lat_sorted[min(int(len(lat) * 0.99), len(lat) - 1)] * 1e3,
        "compute_ms_per_token": compute_tok * 1e3,
        "offstage_ms_per_token": (per_tok - compute_tok) * 1e3,
        "network_loss_pct": 100.0 * (per_tok - compute_tok) / per_tok,
        "stage_compute_ms": {n: s / measure * 1e3 for n, s in zip(names, stage_compute)},
        "bytes_per_token": bytes_out / measure,
    }


def throughput_bench(c: Coordinator, prompt: str, streams: int, max_tokens: int = 24) -> Dict:
    """Several sequences in flight so stages overlap instead of idling."""
    c.reset_stats()
    r = c.generate_many([prompt] * streams, max_tokens=max_tokens)
    stats = c.query_stats()
    stage_compute = [c.runner.stats.compute_s] + [s["compute_s"] for s in stats]
    return {
        "streams": streams,
        "tokens": r["tokens"],
        "wall_s": r["wall_s"],
        "tok_per_s": r["tok_per_s"],
        "ttft_ms": r["ttft_s"] * 1e3,
        "busiest_stage_util_pct": 100.0 * max(stage_compute) / r["wall_s"],
        "stage_compute_s": stage_compute,
    }


def prefill_codec_bench(c: Coordinator, n_tokens: int = 512, chunk: int = 256,
                        repeats: int = 3) -> Dict:
    """Prefill puts whole chunks on the wire, so unlike decode it is genuinely
    bandwidth-bound — the place where a smaller activation format should pay."""
    out = {}
    for codec in ("fp16", "int8"):
        c.set_codec(codec)
        r = prefill_bench(c, n_tokens=n_tokens, chunks=[chunk], repeats=repeats)
        out[codec] = r["by_chunk"][chunk]
    c.set_codec("fp16")
    return out


def prefill_bench(c: Coordinator, n_tokens: int = 512, chunks: Optional[List[int]] = None,
                  repeats: int = 3) -> Dict:
    """Chunked prefill: smaller chunks let stage 0 work while stage 1 does too."""
    tok = c.load_tokenizer()
    ids = tok("the " * n_tokens)["input_ids"][:n_tokens]
    out = {}
    for chunk in (chunks or [n_tokens, 256, 128, 64, 32]):
        if chunk > n_tokens:
            continue
        best = None
        for r in range(repeats):
            req = f"pf{chunk}_{r}"
            t0 = time.perf_counter()
            c._send_prefill(req, ids, chunk)
            c.recv_token(timeout=900)
            dt = time.perf_counter() - t0
            c.free(req)
            best = dt if best is None else min(best, dt)
        out[chunk] = {"ttft_ms": best * 1e3, "prefill_tok_per_s": n_tokens / best}
    return {"prompt_tokens": n_tokens, "by_chunk": out}


def single_node_bench(model: str, warmup: int = 8, measure: int = 48,
                      prompt: str = "Explain pipeline parallelism.",
                      max_tokens: int = 0) -> Dict:
    """The ceiling: every layer in one process, no sockets at all.

    This is what the model would run at on a single machine that *could* hold
    it — the reference the pipeline is measured against. It is only runnable
    when one box has room for the whole model; the point of the project is the
    case where it does not.
    """
    import mlx.core as mx
    from transformers import AutoTokenizer
    from .engine import Sampler
    from .model import KVCache, load_stage
    from .shard import ShardSpec, WeightIndex, snapshot_dir

    d = snapshot_dir(model)
    index = WeightIndex(d)
    L = index.config["num_hidden_layers"]
    mx.reset_peak_memory()
    stage, nbytes = load_stage(d, ShardSpec(0, L, True, True), index)
    peak = mx.get_peak_memory()
    tok = AutoTokenizer.from_pretrained(d)
    ids = tok(prompt)["input_ids"]
    caches = [KVCache() for _ in stage.layers]
    sampler = Sampler(0.0)

    y = stage(mx.array([ids]), caches, last_only=True)
    last = int(sampler(y[:, -1, :]).item())
    text = [last]
    for _ in range(warmup):
        y = stage(mx.array([[last]]), caches, last_only=True)
        last = int(sampler(y[:, -1, :]).item())
        text.append(last)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(measure):
        y = stage(mx.array([[last]]), caches, last_only=True)
        last = int(sampler(y[:, -1, :]).item())
        text.append(last)
    mx.synchronize()
    wall = time.perf_counter() - t0

    out = {"tokens": measure, "wall_s": wall, "tok_per_s": measure / wall,
           "ms_per_token": wall / measure * 1e3, "weight_gib": nbytes / 2**30,
           "peak_gib": peak / 2**30}
    if max_tokens:
        # The timed loop may have produced fewer tokens than the caller wants to
        # compare against; keep going (untimed) until the sequence is long enough.
        while len(text) < max_tokens:
            y = stage(mx.array([[last]]), caches, last_only=True)
            last = int(sampler(y[:, -1, :]).item())
            text.append(last)
        seq = text[:max_tokens]
        out["text"] = tok.decode(seq)
        out["token_ids"] = seq
    del stage, caches
    mx.clear_cache()
    return out
