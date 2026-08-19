"""A Qwen2 decoder written as an addressable *stage*.

Two rules shape this file:

1. A stage builds only the layers it owns. Modules take their loaded arrays in
   ``__init__`` and never random-initialise, so peak memory during load is the
   shard size, not twice it.
2. The forward pass takes and returns hidden states, so a stage is a pure
   function of (hidden, cache) and can sit anywhere in a pipeline.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .shard import ShardSpec


class QLinear(nn.Module):
    """Quantised matmul over pre-loaded weights."""

    def __init__(self, w: mx.array, scales: mx.array, biases: mx.array,
                 bias: Optional[mx.array], group_size: int, bits: int):
        super().__init__()
        self.weight, self.scales, self.biases = w, scales, biases
        if bias is not None:
            self.bias = bias
        self.group_size, self.bits = group_size, bits

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.quantized_matmul(x, self.weight, self.scales, self.biases,
                                transpose=True, group_size=self.group_size, bits=self.bits)
        if "bias" in self:
            y = y + self.bias
        return y


class RMSNorm(nn.Module):
    def __init__(self, weight: mx.array, eps: float):
        super().__init__()
        self.weight, self.eps = weight, eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, cfg: dict, w: Dict[str, mx.array], p: str):
        super().__init__()
        dim = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.head_dim = dim // self.n_heads
        self.scale = self.head_dim ** -0.5
        gs, bits = cfg["quantization"]["group_size"], cfg["quantization"]["bits"]
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            q = f"{p}.self_attn.{proj}"
            setattr(self, proj, QLinear(w[f"{q}.weight"], w[f"{q}.scales"], w[f"{q}.biases"],
                                        w.get(f"{q}.bias"), gs, bits))
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=cfg["rope_theta"])

    def __call__(self, x: mx.array, cache, mask) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv, -1).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, cfg: dict, w: Dict[str, mx.array], p: str):
        super().__init__()
        gs, bits = cfg["quantization"]["group_size"], cfg["quantization"]["bits"]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            q = f"{p}.mlp.{proj}"
            setattr(self, proj, QLinear(w[f"{q}.weight"], w[f"{q}.scales"], w[f"{q}.biases"],
                                        w.get(f"{q}.bias"), gs, bits))

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: dict, w: Dict[str, mx.array], idx: int):
        super().__init__()
        p = f"model.layers.{idx}"
        eps = cfg["rms_norm_eps"]
        self.self_attn = Attention(cfg, w, p)
        self.mlp = MLP(cfg, w, p)
        self.input_layernorm = RMSNorm(w[f"{p}.input_layernorm.weight"], eps)
        self.post_attention_layernorm = RMSNorm(w[f"{p}.post_attention_layernorm.weight"], eps)

    def __call__(self, x: mx.array, cache, mask) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), cache, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class Stage(nn.Module):
    """One contiguous slice of the model, optionally with embedding and/or head."""

    SUPPORTED = {"qwen2"}

    def __init__(self, cfg: dict, spec: ShardSpec, w: Dict[str, mx.array]):
        super().__init__()
        mt = cfg.get("model_type")
        if mt not in self.SUPPORTED:
            raise NotImplementedError(
                f"model_type {mt!r} is not implemented; this runtime handles "
                f"dense {sorted(self.SUPPORTED)} checkpoints. Adding an "
                f"architecture means writing its DecoderLayer here — the "
                f"partitioner, transport and runtime are architecture-agnostic.")
        if "quantization" not in cfg:
            raise NotImplementedError(
                "only quantised checkpoints are implemented (mlx-community "
                "*-4bit and friends)")
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
            self.norm = RMSNorm(w["model.norm.weight"], cfg["rms_norm_eps"])
            if tied:
                self.lm_head = QLinear(self.embed_w, self.embed_s, self.embed_b,
                                       None, gs, bits)
            else:
                self.lm_head = QLinear(w["lm_head.weight"], w["lm_head.scales"],
                                       w["lm_head.biases"], w.get("lm_head.bias"), gs, bits)

    def embed(self, ids: mx.array) -> mx.array:
        rows = self.embed_w[ids]
        return mx.dequantize(rows, self.embed_s[ids], self.embed_b[ids],
                             group_size=self.quant["group_size"], bits=self.quant["bits"])

    def body(self, x, caches=None, mask=None):
        """Embedding (if owned) plus this stage's layers. No output head."""
        if self.spec.embed:
            x = self.embed(x)
        if mask is None and x.shape[1] > 1:
            mask = "causal"
        for i, layer in enumerate(self.layers):
            x = layer(x, None if caches is None else caches[i], mask)
        return x

    def head_forward(self, x, last_only: bool = True):
        """Final norm and vocabulary projection, applied wherever the head lives."""
        if last_only:
            x = x[:, -1:, :]
        return self.lm_head(self.norm(x))

    def __call__(self, x: mx.array, caches: Optional[List] = None, mask=None,
                 apply_head: bool = True, last_only: bool = False) -> mx.array:
        """x: token ids if this stage embeds, else hidden states [B, L, D].

        ``apply_head`` lets intermediate prefill chunks skip the output
        projection entirely, and ``last_only`` runs it on the final position
        alone — the vocab matmul is 152k wide, so doing it per prompt token
        would cost more than the layers themselves.
        """
        if self.spec.embed:
            x = self.embed(x)
        if mask is None and x.shape[1] > 1:
            mask = "causal"
        for i, layer in enumerate(self.layers):
            x = layer(x, None if caches is None else caches[i], mask)
        if self.spec.head and apply_head:
            if last_only:
                x = x[:, -1:, :]
            x = self.lm_head(self.norm(x))
        return x


class KVCache:
    """Growing KV cache for one layer, allocated in steps to limit reallocs."""

    STEP = 256

    def __init__(self):
        self.keys = self.values = None
        self.offset = 0

    def update_and_fetch(self, k: mx.array, v: mx.array):
        prev, L = self.offset, k.shape[2]
        if self.keys is None or prev + L > self.keys.shape[2]:
            B, H, _, D = k.shape
            grow = ((L + self.STEP - 1) // self.STEP) * self.STEP
            new_k = mx.zeros((B, H, prev + grow, D), k.dtype)
            new_v = mx.zeros((B, H, prev + grow, D), v.dtype)
            if self.keys is not None:
                new_k[..., :prev, :] = self.keys[..., :prev, :]
                new_v[..., :prev, :] = self.values[..., :prev, :]
            self.keys, self.values = new_k, new_v
        self.keys[..., prev:prev + L, :] = k
        self.values[..., prev:prev + L, :] = v
        self.offset = prev + L
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]


def load_stage(model_dir: str, spec: ShardSpec, index=None):
    """Load exactly one stage. Returns (Stage, bytes_loaded)."""
    from .shard import WeightIndex
    index = index or WeightIndex(model_dir)
    nbytes = index.shard_bytes(spec)
    weights = index.load_shard(spec)
    stage = Stage(index.config, spec, weights)
    mx.eval(stage.parameters())
    return stage, nbytes
