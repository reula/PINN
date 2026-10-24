#!/usr/bin/env python
"""Size the chunk for the parallel reference-grid evaluation on the GPU backend.

`_eval_model_jit` compiled for the 16641-point LDC reference grid needs ~27 GiB on the GPU
(it is 15 MiB on CPU): XLA materialises per-lane weight products for the vmap.  This probe
prints XLA's buffer requirement vs the number of points in the compiled batch, then times
the chunked/device-parallel wrapper from trainer64 with one and two devices.

    python ldc_eval_probe.py
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import pathlib
import sys

# This script lives in <repo>/clementina; the vendored package is next to it.
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Evolution_try" / "vendor" / "dsgnar_pinn"))

import numpy as np                      # noqa: E402
import jax                              # noqa: E402
import jax.numpy as jnp                 # noqa: E402
import pinn                             # noqa: E402
from pinn import trainer64 as T         # noqa: E402

jax.config.update("jax_enable_x64", True)
print(f"[probe] backend {jax.default_backend()} | devices {[str(d) for d in jax.devices()]}",
      flush=True)


class P:
    x_min, x_max = 0.0, 1.0
    Re = 100.0
    problem_name = "LDC"


model = pinn.SIREN(jax.random.PRNGKey(0), P, time_dependent=False,
                   hidden_dims=(60, 60, 60, 60), periodic_bc=False, n_inputs=2, n_outputs=3)
coords = jnp.asarray(np.load(REPO / "ldc" / "reference_gen" / "ldc_reference.npz")["coords"],
                     dtype=jnp.float64)
n_total = int(coords.shape[0])
print(f"[probe] {n_total} reference points, {sum(x.size for x in jax.tree_util.tree_leaves(model))} "
      f"parameters", flush=True)

f = jax.jit(lambda m, c: jax.vmap(lambda x: m(x))(c))
for n in (n_total, 4096, 2048, 1024, 512, 256):
    c = jax.ShapeDtypeStruct((n, 2), jnp.float64)
    comp = f.lower(model, c).compile()
    ma = comp.memory_analysis()
    print(f"[probe] batch {n:6d}: temp {ma.temp_size_in_bytes/2**30:7.3f} GiB | "
          f"args {ma.argument_size_in_bytes/2**20:8.2f} MiB | "
          f"out {ma.output_size_in_bytes/2**20:8.2f} MiB", flush=True)

for chunk in (512, 1024, 2048):
    for parallel in (False, True):
        T._EVAL_CHUNK, T._EVAL_PARALLEL = chunk, parallel
        t0 = time.perf_counter()
        out = T._eval_model_jit(model, coords)
        np.asarray(out[:, 0])                      # force execution + host transfer
        dt = time.perf_counter() - t0
        print(f"[probe] chunk {chunk:5d} parallel={parallel}: {dt:6.2f}s  out {out.shape}",
              flush=True)

# correctness: chunked vs the largest single shot that fits
T._EVAL_CHUNK, T._EVAL_PARALLEL = 512, False
a = np.asarray(T._eval_model_jit(model, coords[:2048]))
b = np.asarray(T._eval_model_batch_jit(model, coords[:2048]))
print(f"[probe] chunked-vs-single max|diff| {np.abs(a-b).max():.3e}", flush=True)
print("[probe] done", flush=True)
