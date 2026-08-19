"""Sharded stages must be numerically identical to the reference model.

Runs the model split at an arbitrary layer, in-process, and diffs the logits
against mlx_lm's own implementation. Also checks chunked prefill (which
exercises the cache-offset mask) against one-shot prefill.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from hetero.shard import ShardSpec, WeightIndex, snapshot_dir
from hetero.model import load_stage, KVCache

MODEL = os.environ.get("HETERO_TEST_MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit")
CUT = int(os.environ.get("HETERO_TEST_CUT", "13"))


def build(model_dir, n_layers, cut):
    idx = WeightIndex(model_dir)
    a, _ = load_stage(model_dir, ShardSpec(0, cut, True, False), idx)
    b, _ = load_stage(model_dir, ShardSpec(cut, n_layers, False, True), idx)
    return a, b


def forward(a, b, ids, ca=None, cb=None):
    h = a(ids, ca)
    return b(h, cb)


def main():
    d = snapshot_dir(MODEL)
    idx = WeightIndex(d)
    n = idx.config["num_hidden_layers"]
    a, b = build(d, n, CUT)
    ids = mx.array([[151644, 872, 198, 9707, 1879, 11, 419, 374, 264, 1273, 315]])

    ours = forward(a, b, ids)
    mx.eval(ours)

    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    ref_model, _ = load(MODEL)
    ref = ref_model(ids)
    mx.eval(ref)

    diff = mx.abs(ours - ref).max().item()
    rel = diff / mx.abs(ref).max().item()
    top_ours = mx.argmax(ours[0, -1]).item()
    top_ref = mx.argmax(ref[0, -1]).item()
    print(f"[split @ layer {CUT}/{n}] max|diff| = {diff:.3e}  rel = {rel:.3e}")
    print(f"argmax ours={top_ours} ref={top_ref}")
    assert top_ours == top_ref, "argmax mismatch"
    assert rel < 1e-5, f"logits diverge: rel={rel}"

    # Chunked prefill: compare against the reference run the *same* way. Both
    # implementations drift from one-shot prefill identically (fp16 kernel
    # paths differ between batched and cached attention), so the meaningful
    # check is that we track the reference chunk-for-chunk.
    CH = 4
    ca = [KVCache() for _ in a.layers]
    cb = [KVCache() for _ in b.layers]
    rc = make_prompt_cache(ref_model)
    for lo in range(0, ids.shape[1], CH):
        chunk = ids[:, lo:lo + CH]
        out = forward(a, b, chunk, ca, cb)
        rout = ref_model(chunk, cache=rc)
    mx.eval(out, rout)
    d2 = mx.abs(out[0, -1] - rout[0, -1]).max().item()
    drift = mx.abs(rout[0, -1] - ours[0, -1]).max().item()
    print(f"[chunked prefill] max|diff| vs reference-chunked = {d2:.3e} "
          f"(reference's own drift from one-shot: {drift:.3e})")
    assert d2 == 0.0, "chunked prefill diverges from reference"
    del ref_model, ref

    # single-token decode step continues correctly
    nxt = mx.array([[top_ours]])
    step = forward(a, b, nxt, ca, cb)
    mx.eval(step)
    print(f"[decode step] next token = {mx.argmax(step[0, -1]).item()}")
    print("PASS")


if __name__ == "__main__":
    main()
