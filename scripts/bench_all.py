#!/usr/bin/env python3
"""Run the full measurement suite against a live pool and write JSON + tables.

Assumes the peer workers are already up. Everything printed here is measured in
this process against real sockets; nothing is extrapolated.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hetero.bench import decode_bench, prefill_bench, throughput_bench
from hetero.cli import parse_peer
from hetero.coordinator import Coordinator
from hetero.planner import LinkSpec, NodeProfile

NOISE_FLOOR_PCT = 0.5   # measured: 8 repeats of the unshaped config, spread 0.5%

LINKS = [
    (None, None, "loopback (unshaped)"),
    (10000, 0.1, "10 GbE"),
    (1000, 0.3, "1 GbE"),
    (300, 2.0, "Wi-Fi 5, typical"),
    (100, 1.0, "100 Mbit switched"),
    (50, 8.0, "weak Wi-Fi"),
]

PROMPT = "Explain, in detail, how pipeline parallelism splits a transformer across machines."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--peer", action="append", required=True)
    ap.add_argument("--mem-gib", type=float, default=2.5)
    ap.add_argument("--port", type=int, default=29500)
    ap.add_argument("--objective", default="latency")
    ap.add_argument("--out", default="bench/results/results.json")
    ap.add_argument("--measure", type=int, default=48)
    ap.add_argument("--no-single-baseline", action="store_true",
                    help="skip the whole-model reference (use when it cannot fit)")
    a = ap.parse_args()

    # Measure the ceiling first and free it, so the pool never coexists with
    # a full copy of the model in memory.
    single = None
    if not a.no_single_baseline:
        from hetero.bench import single_node_bench
        print("== single-process ceiling (whole model, one box, no network) ==", flush=True)
        single = single_node_bench(a.model, measure=a.measure, prompt=PROMPT, max_tokens=24)
        print(f"  {single['tok_per_s']:.2f} tok/s, {single['ms_per_token']:.2f} ms/token, "
              f"peak {single['peak_gib']:.2f} GiB\n", flush=True)

    nodes = [NodeProfile(name="node-a", host="0.0.0.0", port=a.port,
                         mem_bytes=int(a.mem_gib * 2**30))]
    nodes += [parse_peer(p) for p in a.peer]
    c = Coordinator(a.model, nodes, LinkSpec(), port=a.port, objective=a.objective,
                    self_budget_gib=a.mem_gib)
    c.start()
    print(c.plan.describe(), flush=True)

    R = {
        "single_node": single,
        "meta": {
            "model": a.model,
            "model_bytes": c.index.total_bytes(),
            "layers": c.index.config["num_hidden_layers"],
            "hidden": c.index.config["hidden_size"],
            "nodes": [n.to_json() for n in c.nodes],
            "budget_gib_per_node": a.mem_gib,
            "objective": a.objective,
            "measured_link": {"bandwidth_mbps": c.link.bandwidth_mbps,
                              "rtt_ms": c.link.rtt_ms},
            "host": platform.platform(),
            "note": "both stages run as separate processes on one machine; "
                    "they share a GPU, so stage compute cannot truly overlap",
        },
        "plan": c.plan.to_json(),
        "memory": {
            "model_gib": c.index.total_bytes() / 2**30,
            "per_node_budget_gib": a.mem_gib,
            "stage_weight_gib": [b / 2**30 for b in c.plan.stage_bytes],
            "stage_peak_gib": [c.own_peak / 2**30] + [w["peak"] / 2**30 for w in c.worker_info],
        },
    }

    # Verify the pipeline says the same thing the single process did.
    if single and "token_ids" in single:
        g = c.generate(PROMPT, max_tokens=24, req="verify")
        R["agrees_with_single_node"] = g.tokens[:24] == single["token_ids"][:24]
        print(f"  greedy output identical to single-process run: "
              f"{R['agrees_with_single_node']}\n", flush=True)

    # The first measurement of a session reads slow; discard one.
    c.set_link(None, None)
    decode_bench(c, PROMPT, warmup=6, measure=a.measure)

    print("\n== link sweep (single stream) ==", flush=True)
    print("  (each row is measured against a free-network reference taken "
          "immediately before it;\n   the first row is unshaped-vs-unshaped, "
          "so its loss figure is the method's own error)")
    print(f"{'link':>22} {'ms/tok':>8} {'tok/s':>7} {'compute':>8} {'offstage':>9} "
          f"{'clockdrop':>9} {'loss%':>7}")
    R["link_sweep"] = []
    for bw, rtt, label in LINKS:
        # Re-measure the free-network reference immediately before each shaped
        # config. Within a config the measurement is stable to ~0.2%, but the
        # machine drifts over a sweep this long, and an adjacent baseline is
        # the only way to attribute a difference to the link rather than drift.
        c.set_link(None, None)
        ref = decode_bench(c, PROMPT, warmup=6, measure=a.measure)
        c.set_link(bw, rtt)
        r = decode_bench(c, PROMPT, warmup=6, measure=a.measure)
        r.update(link=label, bandwidth_mbps=bw, rtt_ms=rtt,
                 reference_ms_per_token=ref["ms_per_token"],
                 reference_compute_ms_per_token=ref["compute_ms_per_token"])
        # Total cost of the link = time on the wire, plus the compute the GPU
        # loses to dropping clocks while it waits. Both are measured.
        r["clock_drop_ms_per_token"] = r["compute_ms_per_token"] - ref["compute_ms_per_token"]
        r["loss_vs_free_network_pct"] = 100.0 * (1 - r["tok_per_s"] / ref["tok_per_s"])
        R["link_sweep"].append(r)
        print(f"{label:>22} {r['ms_per_token']:8.2f} {r['tok_per_s']:7.2f} "
              f"{r['compute_ms_per_token']:8.2f} {r['offstage_ms_per_token']:9.3f} "
              f"{r['clock_drop_ms_per_token']:9.3f} {r['loss_vs_free_network_pct']:7.2f}",
              flush=True)

    print("\n== activation codec (bytes on the wire) ==", flush=True)
    print(f"{'link':>22} {'codec':>6} {'B/tok':>7} {'ms/tok':>8} {'loss%':>7}")
    R["codec"] = []
    for bw, rtt, label in [(100, 1.0, "100 Mbit switched"), (50, 8.0, "weak Wi-Fi")]:
        for codec in ("fp16", "int8"):
            c.set_link(bw, rtt)
            c.set_codec(codec)
            r = decode_bench(c, PROMPT, warmup=6, measure=a.measure)
            r.update(link=label, codec=codec)
            R["codec"].append(r)
            print(f"{label:>22} {codec:>6} {r['bytes_per_token']:7.0f} "
                  f"{r['ms_per_token']:8.2f} {r['network_loss_pct']:7.2f}", flush=True)
    c.set_codec("fp16")

    print("\n== concurrent streams (overlap) ==", flush=True)
    print(f"{'link':>22} {'streams':>7} {'tok/s':>8} {'ttft ms':>8} {'busiest util%':>13}")
    R["throughput"] = []
    for bw, rtt, label in [(None, None, "loopback (unshaped)"), (50, 8.0, "weak Wi-Fi")]:
        c.set_link(bw, rtt)
        for s in (1, 2, 4, 8):
            r = throughput_bench(c, PROMPT, streams=s, max_tokens=24)
            r.update(link=label)
            R["throughput"].append(r)
            print(f"{label:>22} {s:7d} {r['tok_per_s']:8.2f} {r['ttft_ms']:8.0f} "
                  f"{r['busiest_stage_util_pct']:13.1f}", flush=True)

    print("\n== prefill: activation format under a slow link ==", flush=True)
    from hetero.bench import prefill_codec_bench
    c.set_link(50, 8.0)
    # One chunk for the whole prompt, so the transfer is exposed rather than
    # overlapped — otherwise chunking hides the very thing being measured.
    pc = prefill_codec_bench(c, n_tokens=512, chunk=512)
    R["prefill_codec"] = pc
    for codec, v in pc.items():
        print(f"  weak Wi-Fi, 512-token prompt, {codec}: {v['ttft_ms']:.0f} ms "
              f"({v['prefill_tok_per_s']:.0f} tok/s prefill)", flush=True)

    print("\n== chunked prefill (hiding the link) ==", flush=True)
    R["prefill"] = []
    for bw, rtt, label in [(None, None, "loopback (unshaped)"), (50, 8.0, "weak Wi-Fi")]:
        c.set_link(bw, rtt)
        r = prefill_bench(c, n_tokens=512)
        r["link"] = label
        R["prefill"].append(r)
        print(f"  {label}: " + "  ".join(
            f"chunk {k}: {v['ttft_ms']:.0f} ms" for k, v in r["by_chunk"].items()), flush=True)

    # How close was the partitioner's own prediction?
    base = R["link_sweep"][0]
    R["prediction"] = {
        "predicted_ms_per_token": c.plan.predicted_decode_s * 1e3,
        "measured_ms_per_token": base["ms_per_token"],
        "error_pct": 100.0 * (c.plan.predicted_decode_s * 1e3 - base["ms_per_token"])
        / base["ms_per_token"],
    }
    print(f"\n== partitioner accuracy ==\n  predicted "
          f"{R['prediction']['predicted_ms_per_token']:.2f} ms/token, measured "
          f"{R['prediction']['measured_ms_per_token']:.2f} ms/token "
          f"({R['prediction']['error_pct']:+.1f}%)", flush=True)

    c.set_link(None, None)
    R["final_stats"] = c.stop()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
