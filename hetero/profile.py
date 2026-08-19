"""Measure what each machine can actually do, rather than guessing.

Compute is probed by timing a small shard and differencing against a shard with
zero layers, which cancels the fixed embed/head cost and leaves a clean
per-layer number. Link bandwidth and RTT are probed over the real socket.
"""
from __future__ import annotations

import subprocess
import time
from typing import Tuple

from . import backends
from .planner import NodeProfile
from .shard import ShardSpec, WeightIndex


def system_memory() -> int:
    out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
    return int(out.stdout.strip())


def auto_mem_budget(fraction: float = 0.65) -> int:
    """A conservative slice of unified memory to hand the runtime."""
    return int(system_memory() * fraction)


def _time_decode(stage, caches, x, backend, iters: int = 24, warmup: int = 6) -> float:
    for _ in range(warmup):
        backend.eval(stage(x, caches))
    t0 = time.perf_counter()
    for _ in range(iters):
        backend.eval(stage(x, caches))
    return (time.perf_counter() - t0) / iters


def _time_bare(fn, backend, iters: int = 24, warmup: int = 6) -> float:
    for _ in range(warmup):
        backend.eval(fn())
    t0 = time.perf_counter()
    for _ in range(iters):
        backend.eval(fn())
    return (time.perf_counter() - t0) / iters


def measure_node(model_dir: str, name: str = "local", probe_layers: int = 4,
                 mem_bytes: int = 0, ctx_warm: int = 256, backend=None) -> NodeProfile:
    """Time this machine's three costs directly: layers, embedding, output head.

    Each is measured on a shard that contains only that part, rather than by
    differencing two larger measurements. On a slow CPU node the output
    projection costs hundreds of milliseconds while a layer costs tens, and
    subtracting one from the other loses the layer cost entirely in the noise —
    which makes the partitioner believe that node's layers are free.
    """
    backend = backend or backends.get("auto")
    slow = backend.name != "mlx"
    if slow:
        probe_layers = min(probe_layers, 2)
    index = WeightIndex(model_dir)
    cfg = index.config
    L = cfg["num_hidden_layers"]
    probe_layers = max(min(probe_layers, L), 1)
    D = cfg["hidden_size"]
    iters, warm = (4, 2) if slow else (24, 6)
    pf_iters, pf_warm = (2, 1) if slow else (6, 2)
    pf_tokens = 32 if slow else 256

    # decoder layers only: no embedding, no output projection
    s_layers, _ = backend.load_stage(model_dir, ShardSpec(0, probe_layers, False, False), index)
    caches = backend.new_caches(probe_layers)
    backend.eval(s_layers(backend.hidden((1, min(ctx_warm, 64) if slow else ctx_warm, D)), caches))
    one = backend.hidden((1, 1, D))
    t_layer = _time_bare(lambda: s_layers(one, caches), backend, iters, warm) / probe_layers

    chunk = backend.hidden((1, pf_tokens, D))
    t_pref = _time_bare(lambda: s_layers(chunk, backend.new_caches(probe_layers)),
                        backend, pf_iters, pf_warm) / (pf_tokens * probe_layers)
    del s_layers, caches
    backend.clear_cache()

    # embedding lookup only
    s_embed, _ = backend.load_stage(model_dir, ShardSpec(0, 0, True, False), index)
    ids = backend.array([[151643]])
    t_embed = _time_bare(lambda: s_embed.embed(ids), backend, iters, warm)
    del s_embed
    backend.clear_cache()

    # final norm + vocabulary projection only
    s_head, _ = backend.load_stage(model_dir, ShardSpec(0, 0, False, True), index)
    t_head = _time_bare(lambda: s_head(one, [], last_only=True), backend, iters, warm)
    del s_head
    backend.clear_cache()

    return NodeProfile(
        name=name,
        mem_bytes=mem_bytes or auto_mem_budget(),
        decode_s_per_layer=max(t_layer, 1e-9),
        prefill_s_per_layer_token=max(t_pref, 1e-12),
        embed_s=t_embed,
        head_s=t_head,
        backend=backend.name,
        weight_expansion=backend.weight_expansion,
    )


def probe_link(chan, rounds: int = 12, payload_mb: float = 1.0) -> Tuple[float, float]:
    """Initiator side: returns (rtt_ms, bandwidth_mbps). Peer must run echo_link."""
    rtts = []
    for i in range(rounds):
        t0 = time.perf_counter()
        chan.send({"t": "ping", "i": i})
        chan.recv(timeout=30)
        rtts.append(time.perf_counter() - t0)
    rtts.sort()
    rtt_ms = rtts[len(rtts) // 2] * 1e3

    # Keep this small. A slow link and a slow peer turn a large probe into
    # minutes of startup, and one megabyte already resolves a home network.
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
