"""Stage 0 and the conductor: plans the split, wires the ring, drives requests."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import backends
from .engine import Sampler, StageRunner, build_stage
from .planner import LinkSpec, NodeProfile, Plan, plan_pipeline
from .shard import ShardSpec, WeightIndex, snapshot_dir
from .transport import Channel, LinkProfile, connect, decode, listen
from .worker import accept_roles


@dataclass
class GenResult:
    text: str = ""
    tokens: List[int] = field(default_factory=list)
    prompt_tokens: int = 0
    ttft_s: float = 0.0
    decode_s: float = 0.0
    total_s: float = 0.0

    @property
    def tok_per_s(self) -> float:
        n = max(len(self.tokens) - 1, 1)
        return n / self.decode_s if self.decode_s > 0 else 0.0


class Coordinator:
    def __init__(self, model: str, nodes: List[NodeProfile], link: LinkSpec,
                 host: str = "0.0.0.0", port: int = 29500, codec: str = "fp16",
                 objective: str = "throughput", max_ctx: int = 4096,
                 shape_link: bool = False, temperature: float = 0.0,
                 top_p: float = 1.0, seed: int = 0, self_budget_gib: float = 0.0,
                 backend: str = "auto"):
        self.backend = backends.get(backend)
        self.model = model
        self.model_dir = snapshot_dir(model)
        self.index = WeightIndex(self.model_dir)
        self.nodes = nodes
        self.link = link
        self.host, self.port = host, port
        self.codec = codec
        self.objective = objective
        self.max_ctx = max_ctx
        self.shape_link = shape_link
        self.sampling = dict(temperature=temperature, top_p=top_p, seed=seed)
        self.self_budget = int(self_budget_gib * 2**30)
        self.plan: Optional[Plan] = None
        self.ctrl: List[Channel] = []
        self.worker_info: List[dict] = []
        self.tokenizer = None

    # -- setup -----------------------------------------------------------
    def make_plan(self) -> Plan:
        self.plan = plan_pipeline(self.index, self.nodes, self.link,
                                  objective=self.objective, max_ctx=self.max_ctx,
                                  codec=self.codec)
        return self.plan

    def probe(self, measure_link: bool = True) -> None:
        """Open control links, measure every node and the wire, then plan."""
        from .profile import measure_node, probe_link
        N = len(self.nodes)
        for i in range(1, N):
            node = self.nodes[i]
            ch = Channel(connect(node.host, node.port, retries=1200))
            ch.send({"role": "control"})
            self.ctrl.append(ch)

        if measure_link and self.ctrl:
            rtts, bws = [], []
            for ch in self.ctrl:
                rtt, bw = probe_link(ch)
                rtts.append(rtt)
                bws.append(bw)
            self.link = LinkSpec(bandwidth_mbps=min(bws), rtt_ms=max(rtts))
            print(f"  link measured: {self.link.bandwidth_mbps:.0f} Mbps, "
                  f"{self.link.rtt_ms:.3f} ms RTT", flush=True)

        # Strictly one node at a time. Two nodes profiling at once — which is
        # exactly what happens when a pool is emulated on one box — makes every
        # node look several times slower than it is, and the planner then cuts
        # in the wrong place.
        own = measure_node(self.model_dir, name=self.nodes[0].name,
                           mem_bytes=self.self_budget or self.nodes[0].mem_bytes,
                           backend=self.backend)
        own.host, own.port = self.nodes[0].host, self.nodes[0].port
        measured = [own]
        for i, ch in enumerate(self.ctrl, start=1):
            ch.send({"t": "probe", "model": self.model})
            h, _ = ch.recv(timeout=900)
            assert h.get("t") == "profile", h
            p = NodeProfile.from_json(h)
            p.host, p.port = self.nodes[i].host, self.nodes[i].port
            if self.nodes[i].mem_bytes:
                p.mem_bytes = self.nodes[i].mem_bytes
            measured.append(p)
        for p in measured:
            print(f"  [{p.name}] {p.decode_s_per_layer * 1e6:.0f} us/layer, "
                  f"budget {p.mem_bytes / 2**30:.2f} GiB", flush=True)
        self.nodes = measured
        self.backend.clear_cache()
        self.make_plan()

    def start(self, probe_first: bool = True) -> None:
        if probe_first and not self.ctrl:
            self.probe()
        plan = self.plan or self.make_plan()
        N = len(self.nodes)
        srv = listen(self.host, self.port)
        shaping = LinkProfile(self.link.bandwidth_mbps, self.link.rtt_ms) if self.shape_link \
            else LinkProfile()

        for i in range(1, N):
            node = self.nodes[i]
            ch = self.ctrl[i - 1]
            nxt = self.nodes[(i + 1) % N]
            ch.send({"t": "plan", "model": self.model,
                     "spec": plan.specs[i].to_json(),
                     "next": {"host": nxt.host if i + 1 < N else _peer_host(self.host),
                              "port": nxt.port if i + 1 < N else self.port},
                     "codec": self.codec,
                     "bandwidth_mbps": self.link.bandwidth_mbps if self.shape_link else None,
                     "rtt_ms": self.link.rtt_ms if self.shape_link else None,
                     **self.sampling})

        self.worker_info = []
        for ch in self.ctrl:
            h, _ = ch.recv(timeout=900)
            if h.get("t") != "ready":
                raise RuntimeError(f"worker failed: {h}")
            self.worker_info.append(h)
            print(f"  [{h['name']}] layers {h['layers'][0]}-{h['layers'][1] - 1} "
                  f"{h['bytes'] / 2**30:.2f} GiB loaded in {h['load_s']:.1f}s "
                  f"(peak {h['peak'] / 2**30:.2f} GiB)", flush=True)

        # stage 0 lives in this process
        t0 = time.perf_counter()
        self.backend.reset_peak()
        stage, nbytes, _ = build_stage(self.model_dir, plan.specs[0], self.index,
                                       budget_bytes=self.self_budget,
                                       backend=self.backend)
        self.own_bytes, self.own_peak = nbytes, self.backend.peak_bytes()
        print(f"  [{self.nodes[0].name}] layers {plan.specs[0].start}-{plan.specs[0].end - 1} "
              f"{nbytes / 2**30:.2f} GiB loaded in {time.perf_counter() - t0:.1f}s "
              f"(peak {self.own_peak / 2**30:.2f} GiB)", flush=True)
        self.runner = StageRunner(stage, plan.specs[0], self.index.config["num_hidden_layers"],
                                  self.codec,
                                  Sampler(**self.sampling) if plan.specs[0].head else None,
                                  name=self.nodes[0].name, backend=self.backend,
                                  body_only=(plan.head_node == 0))
        self.head_local = plan.head_node == 0

        nxt = self.nodes[1]
        self.data_out = Channel(connect(nxt.host, nxt.port, retries=1200), shaping)
        self.data_out.send({"role": "data"})
        self.data_in = accept_roles(srv, {"data"})["data"]
        for ch in self.ctrl:
            h, _ = ch.recv(timeout=300)
            assert h.get("t") == "wired", h
        self.srv = srv
        print("  ring wired", flush=True)

    def load_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        return self.tokenizer

    # -- request driving --------------------------------------------------
    def _send_prefill(self, req: str, ids: List[int], chunk: int) -> None:
        """Push prompt chunks into the ring. Only the last one asks for logits."""
        for lo in range(0, len(ids), chunk):
            hi = min(lo + chunk, len(ids))
            last = hi >= len(ids)
            h, p = self.runner.run_frame(
                {"req": req, "ids": [ids[lo:hi]], "step": 0, "logits": last}, b"")
            self.data_out.send(h, p)

    def _send_decode(self, req: str, tok: int, step: int) -> None:
        h, p = self.runner.run_frame(
            {"req": req, "ids": [[tok]], "step": step, "logits": True}, b"")
        self.data_out.send(h, p)

    def set_link(self, bandwidth_mbps=None, rtt_ms=None) -> None:
        """Retune the emulated link on every hop, mid-session."""
        self.data_out.link = LinkProfile(bandwidth_mbps, rtt_ms)
        self.data_out.send({"t": "shape", "bandwidth_mbps": bandwidth_mbps, "rtt_ms": rtt_ms})
        while True:
            h, _ = self.data_in.recv(timeout=60)
            if h.get("t") == "shape":
                return

    def set_codec(self, codec: str) -> None:
        """Switch the on-wire activation format for every hop, mid-session."""
        self.codec = codec
        self.runner.codec = codec
        self.data_out.send({"t": "codec", "codec": codec})
        while True:
            h, _ = self.data_in.recv(timeout=60)
            if h.get("t") == "codec":
                return

    def query_stats(self) -> List[dict]:
        """Ask every stage for its counters without tearing the ring down."""
        self.data_out.send({"t": "statq", "stats": []})
        while True:
            h, _ = self.data_in.recv(timeout=120)
            if h.get("t") == "statq":
                return h["stats"]

    def reset_stats(self) -> None:
        """Zero every stage's counters so a measurement covers steady state only."""
        from .engine import StageStats
        self.runner.stats = StageStats()
        self.data_out.send({"t": "reset"})
        self.data_out_reset_bytes = self.data_out.stats()["bytes_sent"]

    def recv_token(self, timeout: float = 600):
        """Next completed step as (req, token).

        When stage 0 owns the head, what comes back around the ring is a hidden
        state rather than a token, and the projection and sampling happen here.
        """
        while True:
            h, p = self.data_in.recv(timeout=timeout)
            if h.get("t") == "tok":
                return h["req"], h["tok"]
            if h.get("t") == "fwd":
                if not self.head_local:
                    raise RuntimeError("hidden state returned but no head on stage 0")
                if not h.get("logits", True):
                    continue          # intermediate prefill chunk, nothing to sample
                hid = self.backend.from_wire(decode(h, p))
                logits = self.runner.stage.head_forward(hid, last_only=True)
                return h["req"], self.runner.sampler(logits[:, -1, :], self.backend)
            raise RuntimeError(f"unexpected frame {h}")

    def free(self, req: str) -> None:
        self.runner.free(req)
        self.data_out.send({"t": "free", "req": req})

    def generate(self, prompt: str, max_tokens: int = 64, chunk: int = 128,
                 stop_on_eos: bool = True, req: str = "r0") -> GenResult:
        tok = self.load_tokenizer()
        ids = tok(prompt)["input_ids"]
        eos = _eos_ids(tok)
        res = GenResult(prompt_tokens=len(ids))
        t_start = time.perf_counter()
        self._send_prefill(req, ids, chunk)
        _, tok0 = self.recv_token()
        res.ttft_s = time.perf_counter() - t_start
        res.tokens.append(tok0)
        t_dec = time.perf_counter()
        step = 1
        while len(res.tokens) < max_tokens:
            if stop_on_eos and res.tokens[-1] in eos:
                break
            self._send_decode(req, res.tokens[-1], step)
            _, nxt = self.recv_token()
            res.tokens.append(nxt)
            step += 1
        res.decode_s = time.perf_counter() - t_dec
        res.total_s = time.perf_counter() - t_start
        keep = [t for t in res.tokens if t not in eos] if stop_on_eos else res.tokens
        res.text = tok.decode(keep)
        self.free(req)
        return res

    def generate_many(self, prompts: List[str], max_tokens: int = 32,
                      chunk: int = 128) -> Dict:
        """Run several sequences at once so stages overlap instead of idling."""
        tk = self.load_tokenizer()
        eos = _eos_ids(tk)
        reqs = {}
        for i, p in enumerate(prompts):
            reqs[f"m{i}"] = {"ids": tk(p)["input_ids"], "out": [], "step": 1, "done": False}
        t0 = time.perf_counter()
        for r, st in reqs.items():
            self._send_prefill(r, st["ids"], chunk)
        live = len(reqs)
        first = None
        while live:
            r, tok = self.recv_token(timeout=900)
            st = reqs[r]
            if first is None:
                first = time.perf_counter() - t0
            st["out"].append(tok)
            if len(st["out"]) >= max_tokens or tok in eos:
                st["done"] = True
                live -= 1
                self.free(r)
                continue
            self._send_decode(r, tok, st["step"])
            st["step"] += 1
        total = time.perf_counter() - t0
        gen = sum(len(s["out"]) for s in reqs.values())
        return {"streams": len(reqs), "tokens": gen, "wall_s": total,
                "ttft_s": first, "tok_per_s": gen / total,
                "texts": {r: tk.decode([t for t in s["out"] if t not in eos])
                          for r, s in reqs.items()}}

    # -- teardown ---------------------------------------------------------
    def collect_stats(self) -> List[dict]:
        self.data_out.send({"t": "stop"})
        out = []
        for ch in self.ctrl:
            deadline = time.time() + 30
            while time.time() < deadline:
                h, _ = ch.recv(timeout=30)
                if h.get("t") == "stats":
                    out.append(h)
                    break
        return out

    def stop(self) -> List[dict]:
        stats = self.collect_stats()
        stats.insert(0, {"name": self.nodes[0].name, "compute_s": self.runner.stats.compute_s,
                         "frames": self.runner.stats.frames,
                         "wait_s": self.runner.stats.wait_s,
                         "bytes_out": self.runner.stats.bytes_out,
                         "link": self.data_out.stats(), "peak": self.own_peak})
        for ch in self.ctrl:
            ch.close()
        self.data_in.close()
        self.data_out.close()
        self.srv.close()
        return stats


def _eos_ids(tok) -> set:
    ids = set()
    for attr in ("eos_token_id", "eos_token_ids"):
        v = getattr(tok, attr, None)
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, (list, tuple, set)):
            ids.update(int(x) for x in v)
    return ids


def _peer_host(bind_host: str) -> str:
    """Address the last stage should dial back on."""
    if bind_host in ("0.0.0.0", ""):
        import socket as _s
        try:
            s = _s.socket(_s.AF_INET, _s.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except OSError:
            return "127.0.0.1"
    return bind_host
