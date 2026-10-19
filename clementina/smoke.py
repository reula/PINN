#!/usr/bin/env python
"""Diagnose JAX on the Intel GPUs of Clementina XXI.

Run it through 02_smoke.sbatch (it must be inside a Slurm GPU allocation), or
directly on a compute node obtained with srun.

    python smoke.py                 # full check, exits non-zero on failure
    python smoke.py --allow-cpu     # same, but a CPU backend is a warning
    python smoke.py --json out.json # also write the results as JSON

The list of checks is deliberately the set of primitives wave_pinn's DSGNAR path
uses (jit, grad, vmap, batched JVP, SVD, DCT, while_loop, donation), so a green
run here says the port has a chance, and a red one names the missing kernel
instead of a 400-line traceback from inside the optimiser.

Exit codes: 0 all good; 1 a check failed; 2 no accelerator (CPU fallback).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback

# Third-party imports are deliberately not at module level: if numpy or jax hangs
# (as the first oneAPI attempt did), the "starting" line below is the only proof
# that the process got far enough to matter.


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


def main(argv=None) -> int:
    print(f"smoke.py: start, python {sys.version.split()[0]}, exe {sys.executable}", flush=True)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="do not fail just because the backend is the CPU")
    ap.add_argument("--reps", type=int, default=0,
                    help="timed repetitions (default: 5 on an accelerator, 2 on the CPU)")
    ap.add_argument("--skip-bench", action="store_true", help="skip the matmul timings")
    ap.add_argument("--json", default=None, help="write results to this file")
    args = ap.parse_args(argv)

    print("smoke.py: importing numpy", flush=True)
    import numpy as np

    hr("environment")
    import jax
    import jaxlib

    print(f"python      {sys.version.split()[0]}  ({sys.executable})")
    print(f"platform    {platform.platform()}")
    print(f"jax         {jax.__version__}")
    print(f"jaxlib      {jaxlib.__version__}")
    try:
        print(f"x64 enabled {jax.config.x64_enabled}")
    except Exception:  # noqa: BLE001
        pass
    try:
        import jax_oneapi_plugin
        print(f"plugin      {getattr(jax_oneapi_plugin, '__version__', '?')} ({jax_oneapi_plugin.__file__})")
    except Exception as exc:  # noqa: BLE001
        print(f"plugin      not importable: {exc}")
    try:
        backend = jax.default_backend()
        devs = jax.devices()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("\nFATAL: jax.devices() raised; the oneAPI plugin did not start.", flush=True)
        return 2
    print(f"backend     {backend}")
    print(f"devices     {devs}")
    for d in devs:
        kind = getattr(d, "device_kind", "?")
        stats = None
        try:
            stats = d.memory_stats()
        except Exception:  # noqa: BLE001
            pass
        print(f"            {d}  kind={kind}  memory_stats={stats}")
    try:
        jax.print_environment_info()
    except Exception:  # noqa: BLE001
        pass

    is_cpu = backend == "cpu"
    nreps = args.reps or (2 if is_cpu else 5)
    results: list[dict] = []

    def check(name, fn):
        t0 = time.perf_counter()
        try:
            detail = fn()
            ok = True
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"
            ok = False
            traceback.print_exc()
        dt = time.perf_counter() - t0
        results.append({"check": name, "ok": ok, "seconds": dt, "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name:36s} {dt * 1e3:9.1f} ms   {detail}", flush=True)
        return ok

    def bench_matmul(n: int, dtype):
        import jax.numpy as jnp
        x = jnp.ones((n, n), dtype)
        f = jax.jit(lambda a: a @ a)

        def run():
            y = f(x)
            jax.block_until_ready(y)          # compile + first run
            t0 = time.perf_counter()
            for _ in range(nreps):
                y = f(x)
                jax.block_until_ready(y)
            dt = (time.perf_counter() - t0) / nreps
            gflops = 2.0 * n ** 3 / dt / 1e9
            return f"{dt * 1e3:.1f} ms/iter  {gflops:.1f} GFLOP/s ({dtype.__name__.replace('jnp.', '')}, n={n})"

        return run

    hr(f"core primitives ({nreps} timed reps)")
    import jax.numpy as jnp
    import jax.scipy.fft  # noqa: F401  (jax.scipy.fft is not imported by default)

    def _elementwise():
        f = jax.jit(lambda a: jnp.sum(jnp.tanh(a) * a))
        return f"{float(f(jnp.arange(8.0, dtype=jnp.float32))):.4f}"

    check("elementwise + jit", _elementwise)

    def _random():
        k = jax.random.split(jax.random.PRNGKey(0))[0]
        return f"{jnp.asarray(jax.random.normal(k, (4, 4))).shape}"

    check("random key/split/normal", _random)

    def _grad():
        g = jax.grad(lambda x: jnp.sum(x ** 2))(jnp.ones(4, jnp.float32))
        return f"grad sum {float(jnp.sum(g)):.4f}, shape {g.shape}"

    check("grad (reverse mode)", _grad)

    check("vmap",
          lambda: f"{jax.vmap(lambda x: jnp.sum(x ** 2))(jnp.ones((3, 4), jnp.float32)).shape}")

    def _batched_jvp():
        # the tangent batch DSGNAR differentiates: vmap over jax.jvp
        f = lambda p: jnp.sum(p ** 2)                                  # noqa: E731
        tangents = jnp.ones((3, 4), jnp.float32)
        out = jax.vmap(lambda v: jax.jvp(f, (jnp.ones(4, jnp.float32),), (v,))[1])(tangents)
        return str(out.shape)

    check("vmap over jvp (DSGNAR tangent batch)", _batched_jvp)

    def _dct_roundtrip():
        x = jnp.asarray(np.random.default_rng(0).normal(size=(4, 16)), jnp.float32)
        y = jax.scipy.fft.idct(jax.scipy.fft.dct(x, type=2, norm="ortho"), type=2, norm="ortho")
        return f"roundtrip err {float(jnp.max(jnp.abs(y - x))):.2e}"

    check("jax.scipy.fft.dct/idct (DSGNAR sketching)", _dct_roundtrip)

    check("jnp.linalg.svd (tangent truncation)", lambda: str(
        jnp.linalg.svd(jnp.asarray(np.random.default_rng(0).normal(size=(64, 24)), jnp.float32),
                       full_matrices=False)[1].shape))

    check("lax.scan", lambda: f"{jax.lax.scan(lambda c, x: (c + x, c), 0.0, jnp.arange(5.0, dtype=jnp.float32))[1]}")

    check("lax.while_loop", lambda: f"{jax.jit(lambda x: jax.lax.while_loop(lambda c: c < 10, lambda c: c + 1, x))(jnp.int32(0))}")

    check("jit donates its argument", lambda: str(
        jax.jit(lambda a: a * 2.0, donate_argnums=0)(jnp.ones(8, jnp.float32)).shape))

    def _pytree():
        import jax.flatten_util  # not imported by default; wave_pinn/losses.py does this too
        t = {"w": jnp.ones((3, 2), jnp.float32), "b": jnp.ones(2, jnp.float32)}
        flat, unflatten = jax.flatten_util.ravel_pytree(t)
        return f"{flat.shape} -> {unflatten(flat).keys()}"

    check("flatten/unflatten pytree", _pytree)

    def _transfer():
        host = np.ones(256 * 1024 * 1024 // 8, dtype=np.float64)   # 256 MiB
        return f"sum>0: {bool(float(jax.device_put(jnp.asarray(host)).sum()) > 0)}"

    check("host->device->host 256 MiB", _transfer)

    if not args.skip_bench:
        hr("timings (shape of the work these runs do)")
        check("matmul f32 2048", bench_matmul(2048, jnp.float32))

    hr("float64 (the repo's default: Config.precision = 'float64')")
    try:
        jax.config.update("jax_enable_x64", True)
        print("x64 now enabled\n", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"could not enable x64: {exc}")

    check("x64 dtypes", lambda: f"{jnp.ones(1, jnp.float64).dtype}")

    check("f64 matmul 512", bench_matmul(512, jnp.float64))

    check("f64 svd 256x64", lambda: str(
        jnp.linalg.svd(jnp.asarray(np.random.default_rng(1).normal(size=(256, 64)), jnp.float64),
                       full_matrices=False)[1].shape))

    def _dct_roundtrip_f64():
        x = jnp.asarray(np.random.default_rng(2).normal(size=(8, 32)), jnp.float64)
        y = jax.scipy.fft.idct(jax.scipy.fft.dct(x, type=2, norm="ortho"), type=2, norm="ortho")
        return f"err {float(jnp.max(jnp.abs(y - x))):.2e}"

    check("f64 dct/idct roundtrip", _dct_roundtrip_f64)

    def mlp_bench():
        """The PINN shape: 6x20 MLP, 2201 collocation points, loss + backward pass."""
        n_in, width, n_layers, batch = 2, 20, 6, 2201
        keys = jax.random.split(jax.random.PRNGKey(0), n_layers)
        shapes = [(n_in, width)] + [(width, width)] * (n_layers - 2) + [(width, 1)]
        params = [(jax.random.normal(k, s, jnp.float64) * 0.1,
                   jnp.zeros(s[1], jnp.float64)) for k, s in zip(keys, shapes)]
        x = jnp.asarray(np.random.default_rng(3).uniform(-1, 1, size=(batch, n_in)), jnp.float64)

        def loss(params, x):
            h = x
            for i, (w, b) in enumerate(params):
                h = h @ w + b
                if i < len(params) - 1:
                    h = jnp.tanh(h)
            return jnp.mean(h ** 2)

        f = jax.jit(jax.value_and_grad(loss))
        jax.block_until_ready(f(params, x)[0])
        t0 = time.perf_counter()
        for _ in range(nreps):
            v, g = f(params, x)
            jax.block_until_ready(v)
        dt = (time.perf_counter() - t0) / nreps
        return f"{dt * 1e3:.1f} ms/iter  loss {float(v):.6e}"

    check("f64 6x20 MLP value+grad, batch 2201", mlp_bench)

    hr("summary")
    bad = [r for r in results if not r["ok"]]
    print(f"{len(results) - len(bad)}/{len(results)} checks passed")
    if bad:
        print("\nfailed:")
        for r in bad:
            print(f"  - {r['check']}: {r['detail']}")
    if is_cpu:
        print("\nWARNING: backend is CPU.  In a GPU allocation this means the oneAPI")
        print("plugin did not see a device (see README.md, 'Troubleshooting').")
    elif not bad:
        print("\nThe Intel GPU answered every primitive these runs need.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"backend": backend, "devices": [str(d) for d in devs],
                       "jax": jax.__version__, "jaxlib": jaxlib.__version__,
                       "results": results}, fh, indent=2)
        print(f"\nwrote {args.json}")

    if bad:
        return 1
    if is_cpu and not args.allow_cpu:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
