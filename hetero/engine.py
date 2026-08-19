"""The pipeline runtime.

Topology is a ring: stage 0 embeds, each stage forwards hidden states to its
successor, and the last stage samples a token and sends the *token id* back to
stage 0. Sending the id rather than the logits matters — the vocab is 152k
wide, so returning logits would put 300 KB on the wire per step instead of a
handful of bytes.

Every stage keeps its own KV cache, keyed by request. A request's steps arrive
in order because a step cannot start until the previous one's token completes
the ring, so caches stay consistent without any explicit sequencing.

Overlap comes from the transport's background threads plus FIFO frame
processing: stage 0 can be computing request B (or prompt chunk k+1) while
stage 1 is still on request A (chunk k).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from typing import Dict, List, Optional

from . import backends
from .shard import ShardSpec, WeightIndex
from .transport import Channel, decode, encode


@dataclass
class StageStats:
    compute_s: float = 0.0
    frames: int = 0
    tokens: int = 0
    wait_s: float = 0.0
    bytes_out: int = 0


class StageRunner:
    """Owns one shard, its caches, and the loop that services frames."""

    def __init__(self, stage, spec: ShardSpec, n_layers_total: int,
                 codec: str = "fp16", sampler: Optional["Sampler"] = None,
                 name: str = "stage", backend=None, body_only: bool = False):
        self.name = name
        # Stage 0 owns the head in a mixed pool, but must not apply it on the
        # way out — only when the hidden state comes back around the ring.
        self.body_only = body_only
        self.backend = backend or backends.get("auto")
        self.stage = stage
        self.spec = spec
        self.n_layers_total = n_layers_total
        self.codec = codec
        self.sampler = sampler
        self.caches: Dict[str, List[KVCache]] = {}
        self.stats = StageStats()

    def cache_for(self, req: str) -> List:
        c = self.caches.get(req)
        if c is None:
            c = self.backend.new_caches(len(self.stage.layers))
            self.caches[req] = c
        return c

    def free(self, req: str) -> None:
        self.caches.pop(req, None)

    def run_frame(self, header: dict, payload: bytes):
        """Process one forward frame. Returns (out_header, out_payload)."""
        req = header["req"]
        caches = self.cache_for(req)
        t0 = time.perf_counter()
        if self.spec.embed:
            x = self.backend.array(header["ids"])
        else:
            x = self.backend.from_wire(decode(header, payload))
        want_tok = bool(header.get("logits", True)) and not self.body_only
        if self.body_only:
            y = self.stage.body(x, caches)
        else:
            y = self.stage(x, caches, apply_head=want_tok, last_only=True)
        if self.spec.head and not want_tok and not self.body_only:
            self.backend.eval(y)
            self.stats.compute_s += time.perf_counter() - t0
            self.stats.frames += 1
            return None, None          # intermediate prefill chunk: nothing to return
        if self.spec.head and not self.body_only:
            tok = self.sampler(y[:, -1, :], self.backend)
            self.stats.compute_s += time.perf_counter() - t0
            self.stats.frames += 1
            self.stats.tokens += 1
            return ({"t": "tok", "req": req, "tok": tok,
                     "step": header.get("step", 0), "last": header.get("last", True)}, b"")
        self.backend.eval(y)
        self.stats.compute_s += time.perf_counter() - t0
        self.stats.frames += 1
        h, p = encode(self.backend.to_wire(y), self.codec)
        h.update({"t": "fwd", "req": req, "step": header.get("step", 0),
                  "last": header.get("last", True),
                  "logits": header.get("logits", True)})
        self.stats.bytes_out += len(p)
        return h, p

    def serve(self, inbound: Channel, outbound: Channel, stop_check=None) -> None:
        """Middle/last-stage loop: pull frames, compute, push onward."""
        while True:
            t0 = time.perf_counter()
            header, payload = inbound.recv(timeout=None)
            self.stats.wait_s += time.perf_counter() - t0
            t = header.get("t")
            if t == "fwd":
                h, p = self.run_frame(header, payload)
                if h is not None:
                    outbound.send(h, p)
            elif t == "free":
                self.free(header["req"])
                if not self.spec.head:
                    outbound.send(header)
            elif t == "codec":
                self.codec = header["codec"]
                outbound.send(header)
            elif t == "shape":
                from .transport import LinkProfile
                outbound.link = LinkProfile(header.get("bandwidth_mbps"),
                                            header.get("rtt_ms"))
                outbound.send(header)
            elif t == "statq":
                # Stats ride around the ring, each stage appending its own, and
                # arrive back at stage 0 without interrupting service.
                header.setdefault("stats", []).append(
                    {"name": self.name, "compute_s": self.stats.compute_s,
                     "frames": self.stats.frames, "wait_s": self.stats.wait_s,
                     "bytes_out": self.stats.bytes_out,
                     "link": outbound.stats()})
                outbound.send(header)
            elif t == "reset":
                self.stats = StageStats()
                if not self.spec.head:
                    outbound.send(header)
            elif t == "ping":
                outbound.send({"t": "ack"})
            elif t == "bw":
                outbound.send({"t": "ack"})
            elif t == "stop":
                if not self.spec.head:
                    outbound.send(header)
                return


class Sampler:
    """Greedy or temperature/top-p sampling, delegated to the backend."""

    def __init__(self, temperature: float = 0.0, top_p: float = 1.0, seed: int = 0):
        self.temperature = temperature
        self.top_p = top_p
        self.rng = np.random.default_rng(seed)

    def __call__(self, logits, backend) -> int:
        return backend.sample(logits, self.temperature, self.top_p, self.rng)


def build_stage(model_dir: str, spec: ShardSpec, index: Optional[WeightIndex] = None,
                budget_bytes: int = 0, backend=None):
    """Load a shard, refusing to exceed this node's declared memory budget."""
    backend = backend or backends.get("auto")
    index = index or WeightIndex(model_dir)
    need = resident_bytes(index, spec, backend.weight_expansion)
    if budget_bytes and need > budget_bytes:
        raise MemoryError(
            f"shard needs {need / 2**30:.2f} GiB in memory on the {backend.name} "
            f"backend but this node's budget is {budget_bytes / 2**30:.2f} GiB")
    stage, nbytes = backend.load_stage(model_dir, spec, index)
    return stage, nbytes, index


def resident_bytes(index: WeightIndex, spec: ShardSpec, expansion: float = 1.0) -> int:
    """How much memory a shard actually occupies once loaded.

    The NumPy backend dequantises decoder layers to float32, so they cost eight
    times their on-disk size; embedding and output projection stay quantised.
    """
    lb = index.layer_bytes()
    layer = sum(lb[spec.start:spec.end])
    other = index.shard_bytes(spec) - layer
    return int(layer * expansion + other)
