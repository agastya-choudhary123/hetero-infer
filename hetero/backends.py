"""Backend selection: MLX on Apple silicon, NumPy anywhere else.

The runtime holds a backend rather than importing a framework directly, so a
pool can mix an M-series Mac with a machine that has no MLX at all. Everything
above this file is written in terms of the small surface defined here.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class MLXBackend:
    name = "mlx"
    weight_expansion = 1.0     # weights stay in their quantised form

    def __init__(self):
        import mlx.core as mx
        self.mx = mx

    def load_stage(self, model_dir, spec, index=None):
        from .model import load_stage
        return load_stage(model_dir, spec, index)

    def new_caches(self, n: int) -> List:
        from .model import KVCache
        return [KVCache() for _ in range(n)]

    def array(self, data):
        return self.mx.array(data)

    def eval(self, *xs):
        self.mx.eval(*xs)

    def hidden(self, shape) -> "object":
        return self.mx.zeros(shape, self.mx.float16)

    def to_wire(self, x) -> np.ndarray:
        return np.array(x.astype(self.mx.float16), copy=False)

    def from_wire(self, a: np.ndarray):
        return self.mx.array(a)

    def sample(self, logits, temperature: float, top_p: float, rng) -> int:
        mx = self.mx
        if temperature <= 0:
            return int(mx.argmax(logits, axis=-1).item())
        logits = logits.astype(mx.float32) / temperature
        if top_p < 1.0:
            probs = mx.softmax(logits, axis=-1)
            idx = mx.argsort(-probs, axis=-1)
            sp = mx.take_along_axis(probs, idx, axis=-1)
            keep = mx.cumsum(sp, axis=-1) - sp < top_p
            sp = mx.where(keep, sp, 0.0)
            pick = mx.random.categorical(mx.log(sp + 1e-20))
            return int(mx.take_along_axis(idx, pick[..., None], axis=-1).squeeze(-1).item())
        return int(mx.random.categorical(logits).item())

    def peak_bytes(self) -> int:
        return self.mx.get_peak_memory()

    def reset_peak(self):
        self.mx.reset_peak_memory()

    def clear_cache(self):
        self.mx.clear_cache()


class NumpyBackend:
    """CPU backend. With the fused int4 kernel built, weights stay packed;
    without it they are expanded to float32 and cost eight times the memory."""

    name = "numpy"

    def __init__(self):
        from .cpu_model import KERNEL
        self.fused = KERNEL is not None
        self.weight_expansion = 1.0 if self.fused else 8.0
        if self.fused:
            self.name = "numpy+q4"

    def load_stage(self, model_dir, spec, index=None):
        from .cpu_model import load_stage
        return load_stage(model_dir, spec, index)

    def new_caches(self, n: int) -> List:
        from .cpu_model import KVCache
        return [KVCache() for _ in range(n)]

    def array(self, data):
        return np.asarray(data)

    def eval(self, *xs):
        pass                       # numpy is eager

    def hidden(self, shape) -> np.ndarray:
        return np.zeros(shape, np.float32)

    def to_wire(self, x) -> np.ndarray:
        return np.asarray(x, dtype=np.float16)

    def from_wire(self, a: np.ndarray):
        return np.asarray(a, dtype=np.float32)

    def sample(self, logits, temperature: float, top_p: float, rng) -> int:
        z = np.asarray(logits, dtype=np.float32).reshape(-1)
        if temperature <= 0:
            return int(z.argmax())
        z = z / temperature
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        if top_p < 1.0:
            order = np.argsort(-p)
            keep = np.cumsum(p[order]) - p[order] < top_p
            masked = np.zeros_like(p)
            masked[order[keep]] = p[order[keep]]
            p = masked / masked.sum()
        return int(rng.choice(len(p), p=p))

    def peak_bytes(self) -> int:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    def reset_peak(self):
        pass

    def clear_cache(self):
        pass


def get(name: str = "auto"):
    """`auto` prefers MLX and falls back to NumPy when it is unavailable."""
    if name in ("auto", "mlx"):
        try:
            return MLXBackend()
        except ImportError:
            if name == "mlx":
                raise RuntimeError("mlx requested but not installed on this machine")
    if name in ("auto", "numpy", "cpu"):
        return NumpyBackend()
    raise ValueError(f"unknown backend {name}")
