"""End-to-end: a two-process pipeline must generate exactly what one box does.

Spawns a real worker subprocess, wires a real ring over real sockets, and
compares greedy output token-for-token against the whole model run in a single
process. Also asserts the memory claim: neither stage ever holds enough of the
model to run it alone.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hetero.bench import single_node_bench
from hetero.cli import parse_peer
from hetero.coordinator import Coordinator
from hetero.planner import LinkSpec, NodeProfile

MODEL = os.environ.get("HETERO_TEST_MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit")
BUDGET = float(os.environ.get("HETERO_TEST_BUDGET", "1.4"))
PORT = int(os.environ.get("HETERO_TEST_PORT", "29700"))
PROMPT = "List three uses for a screwdriver."
NTOK = 24


def main() -> int:
    ref = single_node_bench(MODEL, warmup=2, measure=4, prompt=PROMPT, max_tokens=NTOK)
    print(f"single process: {ref['weight_gib']:.2f} GiB of weights, "
          f"{ref['tok_per_s']:.1f} tok/s")
    assert ref["weight_gib"] > BUDGET, (
        f"test is meaningless unless the model ({ref['weight_gib']:.2f} GiB) "
        f"exceeds one node's budget ({BUDGET} GiB)")

    w = subprocess.Popen(
        [sys.executable, "-m", "hetero.cli", "worker", "--port", str(PORT + 1),
         "--name", "t-b", "--mem-gib", str(BUDGET)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        time.sleep(3)
        nodes = [NodeProfile(name="t-a", host="127.0.0.1", port=PORT,
                             mem_bytes=int(BUDGET * 2**30)),
                 parse_peer(f"127.0.0.1:{PORT + 1}:t-b:{BUDGET}")]
        c = Coordinator(MODEL, nodes, LinkSpec(), host="127.0.0.1", port=PORT,
                        objective="latency", self_budget_gib=BUDGET)
        c.start()
        print(c.plan.describe())

        peaks = [c.own_peak] + [i["peak"] for i in c.worker_info]
        for p in peaks:
            assert p < ref["weight_gib"] * 2**30, "a stage held the whole model"
        print(f"peak per stage: {[round(p / 2**30, 2) for p in peaks]} GiB "
              f"— none holds the {ref['weight_gib']:.2f} GiB model")

        g = c.generate(PROMPT, max_tokens=NTOK)
        print(f"pipeline: {g.tok_per_s:.1f} tok/s, TTFT {g.ttft_s * 1e3:.0f} ms")
        assert g.tokens[:NTOK] == ref["token_ids"][:NTOK], (
            f"output differs:\n  pipeline: {g.tokens[:NTOK]}\n  single:   "
            f"{ref['token_ids'][:NTOK]}")
        print("greedy output identical to single-process run")

        # int8 activations should still produce sensible text, though not
        # necessarily the same tokens — that is the point of measuring it.
        c.set_codec("int8")
        g8 = c.generate(PROMPT, max_tokens=NTOK, req="int8")
        same = sum(a == b for a, b in zip(g8.tokens, ref["token_ids"][:NTOK]))
        print(f"int8 activations: {same}/{NTOK} tokens match fp16")
        c.stop()
        print("PASS")
        return 0
    finally:
        w.terminate()
        try:
            w.wait(timeout=10)
        except subprocess.TimeoutExpired:
            w.kill()


if __name__ == "__main__":
    sys.exit(main())
