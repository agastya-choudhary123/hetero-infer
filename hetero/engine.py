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
from typing import Dict, List, Optional

import mlx.core as mx

from .model import KVCache, Stage, load_stage
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

    def __init__(self, stage: Stage, spec: ShardSpec, n_layers_total: int,
                 codec: str = "fp16", sampler: Optional["Sampler"] = None,
                 name: str = "stage"):
        self.name = name
        self.stage = stage
        self.spec = spec
        self.n_layers_total = n_layers_total
        self.codec = codec
        self.sampler = sampler
        self.caches: Dict[str, List[KVCache]] = {}
        self.stats = StageStats()

    def cache_for(self, req: str) -> List[KVCache]:
        c = self.caches.get(req)
        if c is None:
            c = [KVCache() for _ in self.stage.layers]
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
            x = mx.array(header["ids"])
        else:
            x = decode(header, payload)
        want_tok = bool(header.get("logits", True))
        y = self.stage(x, caches, apply_head=want_tok, last_only=True)
        if self.spec.head and not want_tok:
            mx.eval(y)
            self.stats.compute_s += time.perf_counter() - t0
            self.stats.frames += 1
            return None, None          # intermediate prefill chunk: nothing to return
        if self.spec.head:
            tok = self.sampler(y[:, -1, :], req)
            mx.eval(tok)
            self.stats.compute_s += time.perf_counter() - t0
            self.stats.frames += 1
            self.stats.tokens += 1
            return ({"t": "tok", "req": req, "tok": int(tok.item()),
                     "step": header.get("step", 0), "last": header.get("last", True)}, b"")
        mx.eval(y)
        self.stats.compute_s += time.perf_counter() - t0
        self.stats.frames += 1
        h, p = encode(y, self.codec)
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
    """Greedy or temperature/top-p sampling, run on the head stage."""

    def __init__(self, temperature: float = 0.0, top_p: float = 1.0, seed: int = 0):
        self.temperature = temperature
        self.top_p = top_p
        self.key = mx.random.key(seed)

    def __call__(self, logits: mx.array, req: str = "") -> mx.array:
        if self.temperature <= 0:
            return mx.argmax(logits, axis=-1)
        logits = logits.astype(mx.float32) / self.temperature
        if self.top_p < 1.0:
            probs = mx.softmax(logits, axis=-1)
            idx = mx.argsort(-probs, axis=-1)
            sp = mx.take_along_axis(probs, idx, axis=-1)
            cum = mx.cumsum(sp, axis=-1)
            keep = cum - sp < self.top_p
            sp = mx.where(keep, sp, 0.0)
            self.key, sub = mx.random.split(self.key)
            pick = mx.random.categorical(mx.log(sp + 1e-20), key=sub)
            return mx.take_along_axis(idx, pick[..., None], axis=-1).squeeze(-1)
        self.key, sub = mx.random.split(self.key)
        return mx.random.categorical(logits, key=sub)


def build_stage(model_dir: str, spec: ShardSpec, index: Optional[WeightIndex] = None,
                budget_bytes: int = 0):
    """Load a shard, refusing to exceed this node's declared memory budget."""
    index = index or WeightIndex(model_dir)
    need = index.shard_bytes(spec)
    if budget_bytes and need > budget_bytes:
        raise MemoryError(
            f"shard needs {need / 2**30:.2f} GiB but this node's budget is "
            f"{budget_bytes / 2**30:.2f} GiB")
    stage, nbytes = load_stage(model_dir, spec, index)
    return stage, nbytes, index
