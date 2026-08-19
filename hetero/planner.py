"""Where to cut the model.

Stages are contiguous slices of layers assigned to nodes in a fixed ring order,
so a plan is just a list of cut points. We pick them with a DP over
(layer, node) that minimises one of two objectives:

  latency     single stream, no overlap: cost is the SUM of stage times plus
              every link crossing. A fast node should get more layers.
  throughput  microbatched, stages overlap: cost is the MAX over stages of
              (compute + outgoing comms). The pipeline runs at its slowest
              stage, so the aim is to balance, not to hoard.

Memory is a hard constraint: weights + KV cache for the assigned layers must
fit the node's declared budget, or the assignment is infeasible.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from typing import List, Optional

from .shard import ShardSpec, WeightIndex


@dataclass
class NodeProfile:
    """What one machine brings to the pool."""
    name: str
    host: str = "127.0.0.1"
    port: int = 0
    mem_bytes: int = 0                 # hard budget for weights + KV
    decode_s_per_layer: float = 0.0    # seconds per layer per decode step
    prefill_s_per_layer_token: float = 0.0
    embed_s: float = 0.0               # embedding lookup per decode step
    head_s: float = 0.0                # final norm + lm_head per decode step

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "NodeProfile":
        return NodeProfile(**{k: v for k, v in d.items() if k in NodeProfile.__annotations__})


@dataclass
class LinkSpec:
    """The wire between consecutive stages."""
    bandwidth_mbps: float = 1000.0
    rtt_ms: float = 0.3

    def seconds_for(self, nbytes: int) -> float:
        return self.rtt_ms / 2000.0 + (nbytes * 8.0) / (self.bandwidth_mbps * 1e6)


@dataclass
class Plan:
    cuts: List[int]                    # length N+1, cuts[0]=0, cuts[-1]=L
    nodes: List[NodeProfile]
    specs: List[ShardSpec] = field(default_factory=list)
    objective: str = "throughput"
    predicted_decode_s: float = 0.0
    stage_compute_s: List[float] = field(default_factory=list)
    stage_comm_s: List[float] = field(default_factory=list)
    stage_bytes: List[int] = field(default_factory=list)
    stage_mem_bytes: List[int] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        d["nodes"] = [n.to_json() for n in self.nodes]
        d["specs"] = [s.to_json() for s in self.specs]
        return d

    def describe(self) -> str:
        lines = [f"plan[{self.objective}]  predicted {self.predicted_decode_s * 1e3:.2f} ms/token "
                 f"({1.0 / self.predicted_decode_s:.1f} tok/s)"]
        for i, (n, s) in enumerate(zip(self.nodes, self.specs)):
            tag = ("embed " if s.embed else "") + ("head" if s.head else "")
            lines.append(
                f"  stage {i} {n.name:<10} layers {s.start:>3}-{s.end - 1:<3} "
                f"({s.n_layers:>2})  mem {self.stage_mem_bytes[i] / 2**30:5.2f} GiB / "
                f"{n.mem_bytes / 2**30:5.2f} GiB  compute {self.stage_compute_s[i] * 1e3:6.2f} ms  "
                f"comm {self.stage_comm_s[i] * 1e3:5.2f} ms {tag}")
        return "\n".join(lines)


def kv_bytes_per_layer(cfg: dict, max_ctx: int, batch: int = 1, dtype_bytes: int = 2) -> int:
    n_kv = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    return 2 * batch * n_kv * head_dim * max_ctx * dtype_bytes


def hidden_bytes(cfg: dict, tokens: int = 1, batch: int = 1, codec: str = "fp16") -> int:
    d = cfg["hidden_size"]
    if codec == "int8":
        return batch * tokens * (d + 4)
    return batch * tokens * d * 2


def _stage_mem(index: WeightIndex, spec: ShardSpec, kv_per_layer: int) -> int:
    return index.shard_bytes(spec) + spec.n_layers * kv_per_layer


def _stage_time(node: NodeProfile, spec: ShardSpec) -> float:
    t = spec.n_layers * node.decode_s_per_layer
    if spec.embed:
        t += node.embed_s
    if spec.head:
        t += node.head_s
    return t


def plan_pipeline(index: WeightIndex, nodes: List[NodeProfile], link: LinkSpec,
                  objective: str = "throughput", max_ctx: int = 4096,
                  batch: int = 1, codec: str = "fp16") -> Plan:
    """DP over cut points. Returns the best feasible plan or raises."""
    cfg = index.config
    L = int(cfg["num_hidden_layers"])
    N = len(nodes)
    if N < 2:
        raise ValueError("need at least two nodes to split a model")

    kv = kv_bytes_per_layer(cfg, max_ctx, batch)
    # Cost of shipping one activation payload to the next stage.
    act = hidden_bytes(cfg, tokens=1, batch=batch, codec=codec)
    comm = link.seconds_for(act + 96)   # +header

    INF = float("inf")
    # best[i][c] = cost of assigning layers [c:] to nodes i.. ; choose cut c2
    best = [[INF] * (L + 1) for _ in range(N + 1)]
    choice = [[-1] * (L + 1) for _ in range(N + 1)]
    best[N][L] = 0.0

    def combine(a: float, b: float) -> float:
        return max(a, b) if objective == "throughput" else a + b

    for i in range(N - 1, -1, -1):
        node = nodes[i]
        for c in range(L + 1):
            for c2 in range(c, L + 1):
                if best[i + 1][c2] == INF:
                    continue
                spec = ShardSpec(c, c2, i == 0, i == N - 1)
                if _stage_mem(index, spec, kv) > node.mem_bytes:
                    continue
                # last stage sends only a sampled token id back to the head
                out = 0.0 if i == N - 1 else comm
                cost = combine(_stage_time(node, spec) + out, best[i + 1][c2])
                # Ties are common (equal-speed nodes make the latency objective
                # indifferent to where the cut falls). Break them toward the
                # split that leaves the most memory headroom, since KV cache
                # grows into it as context does. The high power makes a stage
                # that is nearly full much worse than two half-full ones.
                cost += 1e-12 * (_stage_mem(index, spec, kv) / node.mem_bytes) ** 8
                if cost < best[i][c]:
                    best[i][c] = cost
                    choice[i][c] = c2

    if best[0][0] == INF:
        total = index.total_bytes() / 2**30
        cap = sum(n.mem_bytes for n in nodes) / 2**30
        raise MemoryError(
            f"no feasible split: model needs {total:.2f} GiB of weights plus KV cache, "
            f"pool offers {cap:.2f} GiB across {N} nodes")

    cuts, c = [0], 0
    for i in range(N):
        c = choice[i][c]
        cuts.append(c)

    specs = [ShardSpec(cuts[i], cuts[i + 1], i == 0, i == N - 1) for i in range(N)]
    plan = Plan(cuts=cuts, nodes=nodes, specs=specs, objective=objective)
    plan.stage_compute_s = [_stage_time(n, s) for n, s in zip(nodes, specs)]
    plan.stage_comm_s = [0.0 if i == N - 1 else comm for i in range(N)]
    plan.stage_bytes = [index.shard_bytes(s) for s in specs]
    plan.stage_mem_bytes = [_stage_mem(index, s, kv) for s in specs]
    # Round trip of the token id back to stage 0 is paid once per step either way.
    ret = link.seconds_for(96)
    if objective == "throughput":
        plan.predicted_decode_s = max(c + m for c, m in
                                      zip(plan.stage_compute_s, plan.stage_comm_s)) + ret
    else:
        plan.predicted_decode_s = sum(plan.stage_compute_s) + sum(plan.stage_comm_s) + ret
    return plan


def ideal_decode_s(nodes: List[NodeProfile], plan: Plan) -> float:
    """Same partition, zero-cost network — the ceiling we measure loss against."""
    return sum(plan.stage_compute_s)
