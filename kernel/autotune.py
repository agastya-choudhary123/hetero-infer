#!/usr/bin/env python3
"""Pick the fastest kernel build for THIS machine and record the choice.

The best row-blocking factor and thread count differ by CPU — an M4 and a 2013
Haswell disagree — so rather than guess, build the variants, time them on
shapes the model actually uses, and keep the winner.
"""
import ctypes, json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHAPES = [(8960, 1536), (1536, 8960), (1536, 1536)]   # MLP up/down and attention
ROWS = [1, 2, 4, 8]


def build(rows, out):
    env = dict(os.environ, ROWS=str(rows), OUT=out)
    subprocess.run(["bash", os.path.join(HERE, "build.sh")], env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.join(HERE, out)


def time_lib(path, threads):
    lib = ctypes.CDLL(path)
    lib.q4_gemv.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4
    lib.q4_gemv.restype = ctypes.c_int
    total = 0.0
    rng = np.random.default_rng(0)
    for n_out, n_in in SHAPES:
        gs = 64
        w = rng.integers(0, 2**32, size=(n_out, n_in // 8), dtype=np.uint64).astype(np.uint32)
        sc = (rng.standard_normal((n_out, n_in // gs)) * 0.01).astype(np.float16)
        bi = (rng.standard_normal((n_out, n_in // gs)) * 0.01).astype(np.float16)
        x = rng.standard_normal(n_in).astype(np.float32)
        out = np.empty(n_out, np.float32)

        def call():
            lib.q4_gemv(w.ctypes.data, sc.ctypes.data, bi.ctypes.data,
                        x.ctypes.data, out.ctypes.data, n_out, n_in, gs, threads)
        for _ in range(4):
            call()
        t0 = time.perf_counter()
        for _ in range(25):
            call()
        total += (time.perf_counter() - t0) / 25
    return total


def main():
    cpus = os.cpu_count() or 4
    thread_opts = sorted({1, 2, 4, cpus, cpus * 2} & set(range(1, 17)))
    best, best_t = None, float("inf")
    print(f"autotuning on {cpus} logical cpus", flush=True)
    for rows in ROWS:
        path = build(rows, f"lib_tune_r{rows}.so")
        for nt in thread_opts:
            os.environ["HETERO_KERNEL_THREADS"] = str(max(thread_opts))
            t = time_lib(path, nt)
            if t < best_t:
                best, best_t = (rows, nt), t
        print(f"  ROWS={rows}: best so far {best} at {best_t*1e3:.3f} ms", flush=True)

    rows, nt = best
    build(rows, "libq4gemv.so")
    cfg = {"rows": rows, "threads": nt, "seconds": best_t,
           "cpus": cpus, "arch": os.uname().machine}
    with open(os.path.join(HERE, "tuning.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for f in os.listdir(HERE):
        if f.startswith("lib_tune_r"):
            os.remove(os.path.join(HERE, f))
    print(f"chose ROWS={rows}, {nt} threads ({best_t*1e3:.3f} ms for the three shapes)")


if __name__ == "__main__":
    main()
