#!/usr/bin/env python
"""Which float64 DCT sizes does the Intel backend actually plan?

The paper-preset exact-residual step dies with
    RET_CHECK failure (fft_thunk.cc:201) fft_plan != nullptr
    Failed to create cuFFT batched plan with scratch allocator
on the SRCT/lift DCT of J: (3*batch, n_params) -> (3*batch, padded n_params) with
n_params = 11347 (padded to 16384) at batch 1024/4096.  This probes the transform
alone across row counts and dtypes, so we can pick a chunk size that works instead of
guessing at run scale.

    python ldc_fft_probe.py
"""

from __future__ import annotations

import os
import traceback

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import numpy as np              # noqa: E402
import jax                     # noqa: E402
import jax.numpy as jnp        # noqa: E402

print(f"[fft ] backend {jax.default_backend()} | devices "
      f"{[str(d) for d in jax.devices()]} | XLA_FLAGS={os.environ.get('XLA_FLAGS')}",
      flush=True)

N_PARAMS = 11347                     # paper preset
for dtype_name, dtype in (("f64", jnp.float64), ("f32", jnp.float32)):
    jax.config.update("jax_enable_x64", dtype_name == "f64")
    for rows in (3072, 2048, 1536, 1024, 768, 512):
        x = jnp.asarray(np.random.default_rng(0).standard_normal((rows, N_PARAMS)),
                        dtype=dtype)
        try:
            y = jax.scipy.fft.dct(x, type=2, norm="ortho", axis=1)
            jax.block_until_ready(y)
            print(f"[fft ] {dtype_name} rows={rows:5d}  OK   out {y.shape} "
                  f"absmax {float(jnp.max(jnp.abs(y))):.3e}", flush=True)
        except Exception as exc:                                    # noqa: BLE001
            msg = str(exc).splitlines()[0][:160]
            print(f"[fft ] {dtype_name} rows={rows:5d}  FAIL {type(exc).__name__}: {msg}",
                  flush=True)
print("[fft ] done", flush=True)
