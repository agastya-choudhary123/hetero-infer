"""Measure what each machine can actually do, rather than guessing.

Compute is probed by timing a small shard and differencing against a shard with
zero layers, which cancels the fixed embed/head cost and leaves a clean
per-layer number. Link bandwidth and RTT are probed over the real socket.
"""
from __future__ import annotations

import subprocess
import time
from typing import Tuple

import mlx.core as mx

from .model import KVCache, load_stage
from .planner import NodeProfile
from .shard import ShardSpec, WeightIndex


def system_memory() -> int:
    out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
    return int(out.stdout.strip())


def auto_mem_budget(fraction: float = 0.65) -> int:
    """A conservative slice of unified memory to hand the runtime."""
    return int(system_memory() * fraction)


def _time_decode(stage, caches, x, iters: int = 24, warmup: int = 6) -> float:
    for _ in range(warmup):
        y = stage(x, caches)
        mx.eval(y)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = stage(x, caches)
        mx.eval(y)
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def _time_bare(fn, iters: int = 24, warmup: int = 6) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def measure_node(model_dir: str, name: str = "local", probe_layers: int = 4,
                 mem_bytes: int = 0, ctx_warm: int = 256) -> NodeProfile:
    """Time a probe shard on this machine. Frees each shard before the next."""
    index = WeightIndex(model_dir)
    L = index.config["num_hidden_layers"]
    probe_layers = min(probe_layers, L)
    ids = mx.array([[151643]])

    # embed + head only: the fixed per-step cost, and the embed half of it
    s0, _ = load_stage(model_dir, ShardSpec(0, 0, True, True), index)
    t_fixed = _time_decode(s0, [], ids)
    t_embed = _time_bare(lambda: s0.embed(ids))
    chunk = mx.array([[151643] * 256])
    t_fixed_prefill = _time_bare(lambda: s0(chunk, []), iters=6, warmup=2)
    del s0
    mx.clear_cache()

    # embed + k layers + head: adds k layers of work
    spec = ShardSpec(0, probe_layers, True, True)
    s1, _ = load_stage(model_dir, spec, index)
    caches = [KVCache() for _ in range(probe_layers)]
    warm = mx.array([[151643] * ctx_warm])
    mx.eval(s1(warm, caches))          # populate cache so decode sees real context
    t_k = _time_decode(s1, caches, ids)

    # prefill: difference the same 256-token chunk against the 0-layer stage,
    # then divide out both tokens and layers
    def prefill_once():
        return s1(chunk, [KVCache() for _ in range(probe_layers)])
    t_prefill = _time_bare(prefill_once, iters=6, warmup=2)
    del s1, caches
    mx.clear_cache()

    per_layer = max((t_k - t_fixed) / probe_layers, 1e-9)
    per_tok_layer = max((t_prefill - t_fixed_prefill) / (256 * probe_layers), 1e-12)
    return NodeProfile(
        name=name,
        mem_bytes=mem_bytes or auto_mem_budget(),
        decode_s_per_layer=per_layer,
        prefill_s_per_layer_token=per_tok_layer,
        embed_s=t_embed,
        head_s=max(t_fixed - t_embed, 0.0),
    )


def probe_link(chan, rounds: int = 40, payload_mb: float = 8.0) -> Tuple[float, float]:
    """Initiator side: returns (rtt_ms, bandwidth_mbps). Peer must run echo_link."""
    rtts = []
    for i in range(rounds):
        t0 = time.perf_counter()
        chan.send({"t": "ping", "i": i})
        chan.recv(timeout=30)
        rtts.append(time.perf_counter() - t0)
    rtts.sort()
    rtt_ms = rtts[len(rtts) // 2] * 1e3

    blob = b"\0" * int(payload_mb * 1e6)
    t0 = time.perf_counter()
    chan.send({"t": "bw"}, blob)
    chan.recv(timeout=120)
    dt = time.perf_counter() - t0 - rtt_ms / 1e3
    bw = (len(blob) * 8.0) / max(dt, 1e-9) / 1e6
    return rtt_ms, bw


def echo_link(chan) -> None:
    """Responder side of probe_link. Returns when the peer says done."""
    while True:
        h, _ = chan.recv(timeout=300)
        if h.get("t") == "done":
            return
        chan.send({"t": "ack"})
