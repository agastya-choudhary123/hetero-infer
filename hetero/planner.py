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
    backend: str = "mlx"
    weight_expansion: float = 1.0      # layer bytes in memory / bytes on disk

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
    head_node: int = -1                # which stage owns the vocabulary projection
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
        lines = [f"plan[{self.objective}, head on stage {self.head_node}]  predicted {self.predicted_decode_s * 1e3:.2f} ms/token "
                 f"({1.0 / self.predicted_decode_s:.1f} tok/s)"]
        for i, (n, s) in enumerate(zip(self.nodes, self.specs)):
            tag = ("embed " if s.embed else "") + ("head" if s.head else "")
            lines.append(
                f"  stage {i} {n.name:<10} [{n.backend:>5}] layers {s.start:>3}-{s.end - 1:<3} "
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


def _stage_mem(index: WeightIndex, spec: ShardSpec, kv_per_layer: int,
               expansion: float = 1.0) -> int:
    from .engine import resident_bytes
    return resident_bytes(index, spec, expansion) + spec.n_layers * kv_per_layer


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
    """Choose where the head lives, then where to cut.

    The ring returns to stage 0 either way, so the output projection can sit on
    the last stage (which then returns a token id) or on stage 0 (which then
    gets hidden states back and projects them itself). On a mixed pool that
    choice dominates: the vocabulary matmul costs milliseconds on a GPU and
    most of a second on a CPU, so it belongs on the fastest node.
    """
    best_plan, best_cost = None, float("inf")
    for head_at in ({len(nodes) - 1, 0} if len(nodes) > 1 else {len(nodes) - 1}):
        try:
            p = _plan_with_head(index, nodes, link, head_at, objective, max_ctx, batch, codec)
        except MemoryError as e:
            last_err = e
            continue
        if p.predicted_decode_s < best_cost:
            best_plan, best_cost = p, p.predicted_decode_s
    if best_plan is None:
        raise last_err
    return best_plan


def _plan_with_head(index: WeightIndex, nodes: List[NodeProfile], link: LinkSpec,
                    head_at: int, objective: str = "throughput", max_ctx: int = 4096,
                    batch: int = 1, codec: str = "fp16") -> Plan:
    """DP over cut points for one head placement."""
    cfg = index.config
    L = int(cfg["num_hidden_layers"])
    N = len(nodes)
    if N < 2:
        raise ValueError("need at least two nodes to split a model")

    kv = kv_bytes_per_layer(cfg, max_ctx, batch)
    # Cost of shipping one activation payload to the next stage.
    act = hidden_bytes(cfg, tokens=1, batch=batch, codec=codec)
    comm = link.seconds_for(act + 96)   # +header
    ret_comm = link.seconds_for(96)     # a token id is a handful of bytes

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
                spec = ShardSpec(c, c2, i == 0, i == head_at)
                if _stage_mem(index, spec, kv, node.weight_expansion) > node.mem_bytes:
                    continue
                # The last stage returns a token id if it owns the head, and a
                # full hidden state if stage 0 owns it instead.
                out = (comm if head_at == 0 else ret_comm) if i == N - 1 else comm
                cost = combine(_stage_time(node, spec) + out, best[i + 1][c2])
                # Ties are common (equal-speed nodes make the latency objective
                # indifferent to where the cut falls). Break them toward the
                # split that leaves the most memory headroom, since KV cache
                # grows into it as context does. The high power makes a stage
                # that is nearly full much worse than two half-full ones.
                cost += 1e-12 * (_stage_mem(index, spec, kv, node.weight_expansion)
                                 / node.mem_bytes) ** 8
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

    specs = [ShardSpec(cuts[i], cuts[i + 1], i == 0, i == head_at) for i in range(N)]
    plan = Plan(cuts=cuts, nodes=nodes, specs=specs, objective=objective, head_node=head_at)
    plan.stage_compute_s = [_stage_time(n, s) for n, s in zip(nodes, specs)]
    plan.stage_comm_s = [(comm if head_at == 0 else ret_comm) if i == N - 1 else comm
                         for i in range(N)]
    plan.stage_bytes = [index.shard_bytes(s) for s in specs]
    plan.stage_mem_bytes = [_stage_mem(index, s, kv, n.weight_expansion)
                            for n, s in zip(nodes, specs)]
    if objective == "throughput":
        plan.predicted_decode_s = max(c + m for c, m in
                                      zip(plan.stage_compute_s, plan.stage_comm_s))
    else:
        plan.predicted_decode_s = sum(plan.stage_compute_s) + sum(plan.stage_comm_s)
    return plan


def ideal_decode_s(nodes: List[NodeProfile], plan: Plan) -> float:
    """Same partition, zero-cost network — the ceiling we measure loss against."""
    return sum(plan.stage_compute_s)
