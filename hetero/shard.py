"""Shard specification and selective weight loading.

The whole point of this project is that no single machine ever holds the whole
model, so nothing here is allowed to materialise more than one stage's weights.
We read the safetensors header to learn tensor shapes/offsets (cheap, no data
touched), pick only the tensors this stage owns, and pull those alone.
"""
from __future__ import annotations

import glob
import json
import os
import struct
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}


@dataclass(frozen=True)
class ShardSpec:
    """Which slice of the network a stage owns.

    Layers are the half-open interval [start, end). ``embed`` and ``head`` mark
    the stage that owns the token embedding and the output norm + lm_head.
    """
    start: int
    end: int
    embed: bool
    head: bool

    @property
    def n_layers(self) -> int:
        return self.end - self.start

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "ShardSpec":
        return ShardSpec(int(d["start"]), int(d["end"]), bool(d["embed"]), bool(d["head"]))



def snapshot_dir(model_id: str) -> str:
    """Resolve a local path or an HF repo id to a snapshot directory."""
    if os.path.isdir(model_id):
        return model_id
    cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    pat = os.path.join(cache, "hub", "models--" + model_id.replace("/", "--"), "snapshots", "*")
    hits = sorted(glob.glob(pat))
    if not hits:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_id)
    return hits[-1]


def read_header(path: str) -> Dict[str, dict]:
    """Parse a safetensors header without reading tensor data."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        head = json.loads(f.read(n))
    head.pop("__metadata__", None)
    return head


class WeightIndex:
    """Maps tensor name -> (file, dtype, shape, nbytes) across a snapshot."""

    def __init__(self, model_dir: str):
        self.dir = model_dir
        files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no safetensors in {model_dir}")
        self.entries: Dict[str, Tuple[str, str, List[int], int]] = {}
        for path in files:
            for name, meta in read_header(path).items():
                shape = list(meta["shape"])
                nb = _DTYPE_BYTES[meta["dtype"]]
                for s in shape:
                    nb *= s
                self.entries[name] = (path, meta["dtype"], shape, nb)
        self.config = json.load(open(os.path.join(model_dir, "config.json")))

    @property
    def tied(self) -> bool:
        """Tied embeddings mean the head stage reuses the embedding matrix."""
        return bool(self.config.get("tie_word_embeddings", False))

    def owns(self, name: str, spec: ShardSpec) -> bool:
        if name.startswith("model.layers."):
            return spec.start <= int(name.split(".")[2]) < spec.end
        if name.startswith("model.embed_tokens"):
            # With tied weights the head stage needs its own copy of the table.
            return spec.embed or (self.tied and spec.head)
        if name.startswith("lm_head") or name == "model.norm.weight":
            return spec.head
        raise KeyError(f"unclassified tensor {name}")

    def names(self) -> List[str]:
        return list(self.entries)

    def nbytes(self, name: str) -> int:
        return self.entries[name][3]

    def layer_bytes(self) -> List[int]:
        """Weight bytes for each decoder layer, indexed by layer number."""
        n = int(self.config["num_hidden_layers"])
        out = [0] * n
        for name, (_, _, _, nb) in self.entries.items():
            if name.startswith("model.layers."):
                out[int(name.split(".")[2])] += nb
        return out

    def embed_bytes(self) -> int:
        return sum(nb for n, (_, _, _, nb) in self.entries.items()
                   if n.startswith("model.embed_tokens"))

    def head_bytes(self) -> int:
        """Bytes a head stage must hold, including a tied embedding copy."""
        b = sum(nb for n, (_, _, _, nb) in self.entries.items()
                if n.startswith("lm_head") or n == "model.norm.weight")
        return b + (self.embed_bytes() if self.tied else 0)

    def total_bytes(self) -> int:
        return sum(nb for _, _, _, nb in self.entries.values())

    def shard_bytes(self, spec: ShardSpec) -> int:
        return sum(nb for n, (_, _, _, nb) in self.entries.items() if self.owns(n, spec))

    def load_shard(self, spec: ShardSpec):
        """Read only this stage's tensors off disk. Returns {name: mx.array}."""
        import mlx.core as mx
        from safetensors import safe_open

        wanted: Dict[str, List[str]] = {}
        for name, (path, _, _, _) in self.entries.items():
            if self.owns(name, spec):
                wanted.setdefault(path, []).append(name)

        out = {}
        for path, names in wanted.items():
            with safe_open(path, framework="numpy") as f:
                for name in names:
                    out[name] = mx.array(f.get_tensor(name))
        mx.eval(list(out.values()))
        return out
