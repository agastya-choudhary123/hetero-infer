"""A NumPy implementation of the same stage, for machines without MLX.

An Intel Mac cannot run MLX, but it can still hold layers and multiply
matrices — Accelerate gives real BLAS throughput. This is a plain-NumPy port
of `model.py`, validated op-by-op against the MLX version to float32 rounding.

Two memory choices matter here:

* Decoder layer weights are dequantised to float32 once at load. That is 8x the
  4-bit size, but it lets BLAS do the matmuls at full speed, and a node this
  slow will only be given a couple of layers anyway.
* The embedding and the output projection are left quantised. At 152k x 3584
  they would be 2 GiB each in float32; instead embeddings dequantise only the
  rows looked up, and the output projection dequantises in blocks as it
  multiplies. That trades ~260 MB of reads for 2 GB.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from .shard import ShardSpec, WeightIndex

HEAD_BLOCK = 512           # rows dequantised at a time on the GEMM path, sized
                           # so the expanded block stays in cache
# Below this many rows the fused GEMV loop wins; above it, expanding a block
# and handing it to BLAS does. Measured crossover on both machines is ~16.
GEMV_ROWS_MAX = int(os.environ.get("HETERO_GEMV_ROWS", "16"))


def _load_kernel():
    """The fused int4 GEMV, if it has been built for this machine."""
    import ctypes
    so = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "kernel", "libq4gemv.so")
    if not os.path.exists(so):
        return None
    lib = ctypes.CDLL(so)
    lib.q4_gemv.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4
    lib.q4_gemv.restype = ctypes.c_int
    lib.q4_dequant.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5
    lib.q4_dequant.restype = ctypes.c_int
    return lib


def _tuning():
    """Thread count chosen by kernel/autotune.py on this machine."""
    import json
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "kernel", "tuning.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


KERNEL = _load_kernel()
TUNING = _tuning()
NTHREADS = (int(os.environ.get("HETERO_THREADS", "0"))
            or TUNING.get("threads")
            or (os.cpu_count() or 2))


def dequantize(w_u32: np.ndarray, scales: np.ndarray, biases: np.ndarray,
               group_size: int, bits: int, out: Optional[np.ndarray] = None) -> np.ndarray:
    """Unpack MLX's quantised layout: 32/bits values per uint32, low bits first."""
    per = 32 // bits
    mask = (1 << bits) - 1
    shifts = (np.arange(per, dtype=np.uint32) * bits)
    q = (w_u32[..., None] >> shifts) & mask
    q = q.reshape(w_u32.shape[0], -1).astype(np.float32)
    s = scales.astype(np.float32).repeat(group_size, axis=1)
    b = biases.astype(np.float32).repeat(group_size, axis=1)
    if out is not None:
        np.multiply(q, s, out=out)
        np.add(out, b, out=out)
        return out
    return q * s + b


class QLinear:
    """Weights stay 4-bit in memory; how they are multiplied depends on shape.

    Decoding one token is a GEMV that reads every weight once, so it is bound by
    memory traffic and goes through the fused kernel, which unpacks into
    registers. Prefill multiplies many rows at once, is compute-bound, and is
    better served by dequantising a block and handing it to BLAS.

    Without the kernel built, weights are expanded to float32 once at load
    instead — correct, but eight times the memory and slower per token.
    """

    def __init__(self, w, scales, biases, bias, group_size, bits):
        self.bias = None if bias is None else bias.astype(np.float32)
        self.group_size, self.bits = group_size, bits
        self.n_out, self.n_in = w.shape[0], w.shape[1] * (32 // bits)
        if KERNEL is not None and bits == 4:
            self.w = np.ascontiguousarray(w)
            self.scales = np.ascontiguousarray(scales, dtype=np.float16)
            self.biases = np.ascontiguousarray(biases, dtype=np.float16)
            self.W = None
        else:
            self.W = dequantize(w, scales, biases, group_size, bits)
            self.w = None

    def _gemv(self, x: np.ndarray) -> np.ndarray:
        xc = np.ascontiguousarray(x, dtype=np.float32)
        out = np.empty(self.n_out, dtype=np.float32)
        rc = KERNEL.q4_gemv(self.w.ctypes.data, self.scales.ctypes.data,
                            self.biases.ctypes.data, xc.ctypes.data, out.ctypes.data,
                            self.n_out, self.n_in, self.group_size, NTHREADS)
        if rc != 0:
            raise RuntimeError(f"q4_gemv failed with {rc}")
        return out

    def _gemm(self, flat: np.ndarray) -> np.ndarray:
        """Expand a block of rows in C, then let BLAS multiply.

        NumPy's own dequantisation allocates a temporary per step and costs more
        than the matmul it feeds; doing it in one C pass into a reused,
        cache-sized buffer is what makes prefill affordable on a CPU node.
        """
        out = np.empty((flat.shape[0], self.n_out), dtype=np.float32)
        buf = self._buf(HEAD_BLOCK)
        for lo in range(0, self.n_out, HEAD_BLOCK):
            hi = min(lo + HEAD_BLOCK, self.n_out)
            rc = KERNEL.q4_dequant(self.w.ctypes.data, self.scales.ctypes.data,
                                   self.biases.ctypes.data, buf.ctypes.data,
                                   lo, hi, self.n_in, self.group_size, NTHREADS)
            if rc != 0:
                raise RuntimeError(f"q4_dequant failed with {rc}")
            out[:, lo:hi] = flat @ buf[:hi - lo].T
        return out

    def _buf(self, rows: int) -> np.ndarray:
        b = getattr(self, "_scratch", None)
        if b is None or b.shape != (rows, self.n_in):
            b = np.empty((rows, self.n_in), dtype=np.float32)
            self._scratch = b
        return b

    def __call__(self, x: np.ndarray) -> np.ndarray:
        flat = np.asarray(x, dtype=np.float32).reshape(-1, x.shape[-1])
        if self.W is not None:
            out = flat @ self.W.T
        elif flat.shape[0] <= GEMV_ROWS_MAX:
            # Short prompts go row-by-row through the fused kernel. Dequantising
            # the matrix in NumPy to use BLAS only pays off once there are
            # enough rows to amortise it, and a chat prompt rarely has that many.
            out = np.stack([self._gemv(flat[r]) for r in range(flat.shape[0])])
        else:
            out = self._gemm(flat)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*x.shape[:-1], self.n_out)


QLinearBlocked = QLinear      # the vocabulary projection takes the same path


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    x = x.astype(np.float32)
    return x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * w


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x, dtype=np.float32))


class RoPE:
    """Rotary embedding with a cached cos/sin table.

    Decode calls this twice per layer with a single position, so recomputing
    the transcendentals every time is pure waste; the table grows as context
    does and is shared by every layer.
    """

    def __init__(self, dims: int, base: float):
        self.half = dims // 2
        self.inv = base ** (-np.arange(0, self.half, dtype=np.float32) * 2.0 / dims)
        self._n = 0
        self._cos = self._sin = None

    def _table(self, upto: int):
        if self._n < upto:
            n = max(upto, 256, self._n * 2)
            ang = np.arange(n, dtype=np.float32)[:, None] * self.inv[None, :]
            self._cos, self._sin = np.cos(ang), np.sin(ang)
            self._n = n
        return self._cos, self._sin

    def __call__(self, x: np.ndarray, offset: int = 0) -> np.ndarray:
        L = x.shape[2]
        cos, sin = self._table(offset + L)
        cos = cos[offset:offset + L]
        sin = sin[offset:offset + L]
        x1, x2 = x[..., :self.half], x[..., self.half:]
        out = np.empty_like(x)
        np.subtract(x1 * cos, x2 * sin, out=out[..., :self.half])
        np.add(x1 * sin, x2 * cos, out=out[..., self.half:])
        return out


class KVCache:
    """Same growth policy as the MLX cache, so the two behave alike."""

    STEP = 256

    def __init__(self):
        self.keys = self.values = None
        self.offset = 0

    def update_and_fetch(self, k, v):
        prev, L = self.offset, k.shape[2]
        if self.keys is None or prev + L > self.keys.shape[2]:
            B, H, _, D = k.shape
            grow = ((L + self.STEP - 1) // self.STEP) * self.STEP
            nk = np.zeros((B, H, prev + grow, D), np.float32)
            nv = np.zeros((B, H, prev + grow, D), np.float32)
            if self.keys is not None:
                nk[..., :prev, :] = self.keys[..., :prev, :]
                nv[..., :prev, :] = self.values[..., :prev, :]
            self.keys, self.values = nk, nv
        self.keys[..., prev:prev + L, :] = k
        self.values[..., prev:prev + L, :] = v
        self.offset = prev + L
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]


def sdpa(q, k, v, scale, causal):
    """Grouped-query attention without expanding K and V.

    np.repeat on the key and value heads allocates a full copy of the cache on
    every call, which at decode dwarfs the arithmetic. Reshaping the query into
    (kv-head, group) instead lets the same K broadcast across its group.
    """
    B, H, L, D = q.shape
    KH, S = k.shape[1], k.shape[2]
    G = H // KH
    q5 = q.reshape(B, KH, G, L, D)
    s = np.einsum("bkgld,bksd->bkgls", q5, k, optimize=True) * scale
    if causal and L > 1:
        i = np.arange(L)[:, None] + (S - L)
        s = np.where(np.arange(S)[None, :] <= i, s, -np.inf)
    s -= s.max(-1, keepdims=True)
    e = np.exp(s)
    e /= e.sum(-1, keepdims=True)
    o = np.einsum("bkgls,bksd->bkgld", e, v, optimize=True)
    return o.reshape(B, H, L, D)


class Attention:
    def __init__(self, cfg, w, p):
        dim = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.head_dim = dim // self.n_heads
        self.scale = self.head_dim ** -0.5
        gs, bits = cfg["quantization"]["group_size"], cfg["quantization"]["bits"]
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            q = f"{p}.self_attn.{proj}"
            setattr(self, proj, QLinear(w[f"{q}.weight"], w[f"{q}.scales"],
                                        w[f"{q}.biases"], w.get(f"{q}.bias"), gs, bits))
        self.rope = RoPE(self.head_dim, cfg["rope_theta"])

    def __call__(self, x, cache, causal):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv, -1).transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset)
        k = self.rope(k, offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        o = sdpa(q, k, v, self.scale, causal)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP:
    def __init__(self, cfg, w, p):
        gs, bits = cfg["quantization"]["group_size"], cfg["quantization"]["bits"]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            q = f"{p}.mlp.{proj}"
            setattr(self, proj, QLinear(w[f"{q}.weight"], w[f"{q}.scales"],
                                        w[f"{q}.biases"], w.get(f"{q}.bias"), gs, bits))

    def __call__(self, x):
        return self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer:
    def __init__(self, cfg, w, idx):
        p = f"model.layers.{idx}"
        self.eps = cfg["rms_norm_eps"]
        self.self_attn = Attention(cfg, w, p)
        self.mlp = MLP(cfg, w, p)
        self.in_w = w[f"{p}.input_layernorm.weight"].astype(np.float32)
        self.post_w = w[f"{p}.post_attention_layernorm.weight"].astype(np.float32)

    def __call__(self, x, cache, causal):
        x = x + self.self_attn(rms_norm(x, self.in_w, self.eps), cache, causal)
        return x + self.mlp(rms_norm(x, self.post_w, self.eps))


class Stage:
    """Mirrors model.Stage so the runtime cannot tell which backend it holds."""

    def __init__(self, cfg: dict, spec: ShardSpec, w: Dict[str, np.ndarray]):
        if cfg.get("model_type") != "qwen2" or "quantization" not in cfg:
            raise NotImplementedError("cpu backend handles quantised qwen2 only")
        self.cfg, self.spec = cfg, spec
        self.quant = cfg["quantization"]
        gs, bits = self.quant["group_size"], self.quant["bits"]
        tied = bool(cfg.get("tie_word_embeddings", False))
        if spec.embed or (tied and spec.head):
            self.embed_w = w["model.embed_tokens.weight"]
            self.embed_s = w["model.embed_tokens.scales"]
            self.embed_b = w["model.embed_tokens.biases"]
        self.layers = [DecoderLayer(cfg, w, i) for i in range(spec.start, spec.end)]
        if spec.head:
            self.norm_w = w["model.norm.weight"].astype(np.float32)
            if tied:
                self.lm_head = QLinearBlocked(self.embed_w, self.embed_s, self.embed_b,
                                              None, gs, bits)
            else:
                self.lm_head = QLinearBlocked(w["lm_head.weight"], w["lm_head.scales"],
                                              w["lm_head.biases"], w.get("lm_head.bias"),
                                              gs, bits)

    def embed(self, ids: np.ndarray) -> np.ndarray:
        flat = ids.reshape(-1)
        rows = dequantize(self.embed_w[flat], self.embed_s[flat], self.embed_b[flat],
                          self.quant["group_size"], self.quant["bits"])
        return rows.reshape(*ids.shape, -1)

    def body(self, x, caches=None, mask=None):
        """Embedding (if owned) plus this stage's layers. No output head."""
        if self.spec.embed:
            x = self.embed(np.asarray(x))
        x = np.asarray(x, dtype=np.float32)
        causal = x.shape[1] > 1
        for i, layer in enumerate(self.layers):
            x = layer(x, None if caches is None else caches[i], causal)
        return x

    def head_forward(self, x, last_only: bool = True):
        """Final norm and vocabulary projection, applied wherever the head lives."""
        x = np.asarray(x, dtype=np.float32)
        if last_only:
            x = x[:, -1:, :]
        return self.lm_head(rms_norm(x, self.norm_w, self.cfg["rms_norm_eps"]))

    def __call__(self, x, caches=None, mask=None, apply_head=True, last_only=False):
        if self.spec.embed:
            x = self.embed(np.asarray(x))
        x = np.asarray(x, dtype=np.float32)
        causal = x.shape[1] > 1
        for i, layer in enumerate(self.layers):
            x = layer(x, None if caches is None else caches[i], causal)
        if self.spec.head and apply_head:
            if last_only:
                x = x[:, -1:, :]
            x = self.lm_head(rms_norm(x, self.norm_w, self.cfg["rms_norm_eps"]))
        return x


def load_stage(model_dir: str, spec: ShardSpec, index: Optional[WeightIndex] = None):
    index = index or WeightIndex(model_dir)
    weights = index.load_shard(spec, framework="numpy")
    stage = Stage(index.config, spec, weights)
    return stage, index.shard_bytes(spec)
