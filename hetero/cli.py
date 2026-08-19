"""Command line: profile a box, plan a split, run a worker, or run a prompt."""
from __future__ import annotations

import argparse
import json
import sys

from .planner import LinkSpec, NodeProfile


def parse_peer(s: str) -> NodeProfile:
    """host:port[:name[:mem_gib]]"""
    parts = s.split(":")
    host, port = parts[0], int(parts[1])
    name = parts[2] if len(parts) > 2 and parts[2] else f"{host}:{port}"
    mem = int(float(parts[3]) * 2**30) if len(parts) > 3 else 0
    return NodeProfile(name=name, host=host, port=port, mem_bytes=mem)


def add_pool_args(ap):
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--peer", action="append", default=[],
                    help="host:port[:name[:mem_gib]] for each non-head node, in ring order")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=29500)
    ap.add_argument("--mem-gib", type=float, default=0.0, help="this node's memory budget")
    ap.add_argument("--codec", choices=["fp16", "int8"], default="fp16")
    ap.add_argument("--objective", choices=["latency", "throughput"], default="latency")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--backend", default="auto", choices=["auto", "mlx", "numpy", "cpu"],
                    help="backend for this node (stage 0)")
    ap.add_argument("--shape-link", action="store_true",
                    help="apply the emulated link profile to real sends")
    ap.add_argument("--bandwidth-mbps", type=float, default=None)
    ap.add_argument("--rtt-ms", type=float, default=None)


def build_coordinator(a):
    from .coordinator import Coordinator
    nodes = [NodeProfile(name=a.name if hasattr(a, "name") and a.name else "head",
                         host=a.host, port=a.port, mem_bytes=int(a.mem_gib * 2**30))]
    nodes += [parse_peer(p) for p in a.peer]
    link = LinkSpec(bandwidth_mbps=a.bandwidth_mbps or 1000.0, rtt_ms=a.rtt_ms or 0.3)
    return Coordinator(a.model, nodes, link, host=a.host, port=a.port, codec=a.codec,
                       objective=a.objective, max_ctx=a.max_ctx,
                       shape_link=a.shape_link, self_budget_gib=a.mem_gib,
                       temperature=getattr(a, "temperature", 0.0),
                       backend=getattr(a, "backend", "auto"))


def cmd_profile(a) -> int:
    from .profile import measure_node
    from .shard import snapshot_dir
    p = measure_node(snapshot_dir(a.model), name=a.name or "local",
                     mem_bytes=int(a.mem_gib * 2**30))
    print(json.dumps(p.to_json(), indent=2))
    return 0


def cmd_worker(a) -> int:
    from .worker import main
    argv = ["--port", str(a.port), "--host", a.host]
    if a.name:
        argv += ["--name", a.name]
    if a.mem_gib:
        argv += ["--mem-gib", str(a.mem_gib)]
    argv += ["--backend", a.backend]
    return main(argv) or 0


def cmd_run(a) -> int:
    c = build_coordinator(a)
    try:
        c.start()
    except MemoryError as e:
        print(f"error: {e}", file=sys.stderr)
        print("       add a node, raise --mem-gib, or pick a smaller model.",
              file=sys.stderr)
        return 2
    print(c.plan.describe())
    r = c.generate(a.prompt, max_tokens=a.max_tokens, chunk=a.chunk,
                   chat=getattr(a, "chat", False))
    print("\n--- output ---")
    print(r.text)
    print("---")
    print(f"prompt {r.prompt_tokens} tok | TTFT {r.ttft_s * 1e3:.0f} ms | "
          f"{len(r.tokens)} generated | {r.tok_per_s:.2f} tok/s | "
          f"{r.total_s:.1f} s wall")
    stats = c.stop()
    for s in stats:
        print(f"  [{s['name']}] compute {s['compute_s'] * 1e3:8.0f} ms over {s['frames']:3d} frames"
              f" | {s['link']['bytes_sent'] / 1e6:5.2f} MB on the wire"
              f" | peak {s['peak'] / 2**30:.2f} GiB")
    return 0


def cmd_plan(a) -> int:
    """Model a pool on paper: what split would we choose, and what would it cost?"""
    from .profile import measure_node
    from .planner import plan_pipeline
    from .shard import WeightIndex, snapshot_dir
    d = snapshot_dir(a.model)
    idx = WeightIndex(d)
    base = measure_node(d, name="probe", mem_bytes=0)
    nodes = []
    for spec in a.node:
        name, mem, speed = spec.split(":")
        p = NodeProfile(name=name, mem_bytes=int(float(mem) * 2**30),
                        decode_s_per_layer=base.decode_s_per_layer / float(speed),
                        prefill_s_per_layer_token=base.prefill_s_per_layer_token / float(speed),
                        embed_s=base.embed_s / float(speed), head_s=base.head_s / float(speed))
        nodes.append(p)
    link = LinkSpec(bandwidth_mbps=a.bandwidth_mbps or 1000.0, rtt_ms=a.rtt_ms or 0.3)
    print(f"model {a.model}: {idx.total_bytes() / 2**30:.2f} GiB of weights, "
          f"{idx.config['num_hidden_layers']} layers")
    print(f"this box measured at {base.decode_s_per_layer * 1e6:.0f} us/layer; "
          f"node speeds are relative to it")
    for obj in ("latency", "throughput"):
        p = plan_pipeline(idx, nodes, link, objective=obj, max_ctx=a.max_ctx, codec=a.codec)
        print(p.describe())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("hetero", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("profile", help="measure this machine")
    p.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    p.add_argument("--name", default=None)
    p.add_argument("--mem-gib", type=float, default=0.0)
    p.set_defaults(fn=cmd_profile)

    p = sub.add_parser("worker", help="join a pool and wait for a plan")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--name", default=None)
    p.add_argument("--mem-gib", type=float, default=0.0)
    p.add_argument("--backend", default="auto", choices=["auto", "mlx", "numpy", "cpu"])
    p.set_defaults(fn=cmd_worker)

    p = sub.add_parser("run", help="plan, wire the ring, and generate")
    add_pool_args(p)
    p.add_argument("--name", default="head")
    p.add_argument("--prompt", default="Explain what a pipeline-parallel LLM does, briefly.")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--chat", action="store_true",
                   help="wrap the prompt in the model's chat template")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("plan", help="what-if a pool without launching it")
    p.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    p.add_argument("--node", action="append", required=True,
                   help="name:mem_gib:relative_speed")
    p.add_argument("--bandwidth-mbps", type=float, default=None)
    p.add_argument("--rtt-ms", type=float, default=None)
    p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--codec", default="fp16")
    p.set_defaults(fn=cmd_plan)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
