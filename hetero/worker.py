"""A pool member. Waits for a plan, loads its shard, then serves the ring.

A worker owns one listening socket and sorts incoming connections by their
hello frame: the coordinator opens a control link, the predecessor stage opens
a data link. The successor link is opened outbound.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time

import mlx.core as mx

from .engine import Sampler, StageRunner, build_stage
from .planner import LinkSpec
from .shard import ShardSpec, snapshot_dir
from .transport import Channel, LinkProfile, connect, listen


def accept_roles(srv: socket.socket, roles, link: LinkProfile = None):
    """Accept connections until every named role has arrived."""
    out = {}
    while set(roles) - set(out):
        conn, _ = srv.accept()
        ch = Channel(conn, link)
        hello, _ = ch.recv(timeout=60)
        role = hello.get("role")
        if role not in roles:
            ch.close()
            continue
        out[role] = ch
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("hetero-worker")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--mem-gib", type=float, default=0.0,
                    help="hard memory budget for this node; refuses shards that exceed it")
    args = ap.parse_args(argv)
    name = args.name or f"worker:{args.port}"
    budget = int(args.mem_gib * 2**30)

    srv = listen(args.host, args.port)
    print(f"[{name}] listening on {args.host}:{args.port}"
          + (f" budget {args.mem_gib:.2f} GiB" if budget else ""), flush=True)

    while True:
        chans = accept_roles(srv, {"control"})
        ctrl = chans["control"]
        try:
            serve_session(ctrl, srv, name, budget)
        except (ConnectionError, EOFError) as e:
            print(f"[{name}] session ended: {e}", flush=True)
        finally:
            try:
                ctrl.close()
            except Exception:
                pass
            mx.clear_cache()


def serve_session(ctrl: Channel, srv: socket.socket, name: str, budget: int) -> None:
    # Phase 1: the coordinator probes this node. We measure ourselves rather
    # than let it assume the pool is symmetric, and we answer link pings so it
    # can time the wire between us.
    # Setup phase: answer link probes and self-measure until a plan arrives.
    while True:
        header, _ = ctrl.recv(timeout=900)
        t = header.get("t")
        if t in ("ping", "bw"):
            ctrl.send({"t": "ack"})
        elif t == "probe":
            from .profile import measure_node
            prof = measure_node(snapshot_dir(header["model"]), name=name, mem_bytes=budget)
            mx.clear_cache()
            print(f"[{name}] profiled: {prof.decode_s_per_layer * 1e6:.0f} us/layer, "
                  f"budget {prof.mem_bytes / 2**30:.2f} GiB", flush=True)
            ctrl.send({"t": "profile", **prof.to_json()})
        elif t == "plan":
            break
        else:
            raise RuntimeError(f"unexpected setup message {header}")
    spec = ShardSpec.from_json(header["spec"])
    model_dir = snapshot_dir(header["model"])
    link = LinkProfile(header.get("bandwidth_mbps"), header.get("rtt_ms"))
    codec = header.get("codec", "fp16")

    t0 = time.perf_counter()
    mx.reset_peak_memory()
    stage, nbytes, index = build_stage(model_dir, spec, budget_bytes=budget)
    load_s = time.perf_counter() - t0
    peak = mx.get_peak_memory()
    print(f"[{name}] loaded layers {spec.start}-{spec.end - 1} "
          f"({nbytes / 2**30:.2f} GiB) in {load_s:.1f}s, peak {peak / 2**30:.2f} GiB", flush=True)

    runner = StageRunner(stage, spec, index.config["num_hidden_layers"], codec,
                         Sampler(header.get("temperature", 0.0), header.get("top_p", 1.0),
                                 header.get("seed", 0)) if spec.head else None,
                         name=name)
    ctrl.send({"t": "ready", "name": name, "bytes": nbytes, "peak": peak,
               "load_s": load_s, "layers": [spec.start, spec.end]})

    data_in = accept_roles(srv, {"data"}, link)["data"]
    nxt = header["next"]
    out_sock = connect(nxt["host"], nxt["port"], retries=1200)
    data_out = Channel(out_sock, link)
    data_out.send({"role": "data"})
    print(f"[{name}] ring wired -> {nxt['host']}:{nxt['port']}", flush=True)
    ctrl.send({"t": "wired", "name": name})

    try:
        runner.serve(data_in, data_out)
    finally:
        ctrl.send({"t": "stats", "name": name,
                   "compute_s": runner.stats.compute_s, "frames": runner.stats.frames,
                   "wait_s": runner.stats.wait_s, "bytes_out": runner.stats.bytes_out,
                   "link": data_out.stats(), "peak": mx.get_peak_memory()})
        time.sleep(0.2)
        data_in.close()
        data_out.close()


if __name__ == "__main__":
    sys.exit(main())
