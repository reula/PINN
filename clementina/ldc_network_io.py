#!/usr/bin/env python
"""Save / load the trained LDC network: parameters *and* what is needed to rebuild it.

A flat parameter vector is useless without the architecture that produced it.  This
module defines the one file format both LDC scripts write, and the loader that turns
it back into a callable model:

    <run>/params_flat.npy   the trainable arrays, flattened, float64, C order
    <run>/network.json      hidden_dims, seed, SIREN options, driver/config,
                            jax/jaxlib versions, backend, per-channel rel L2, ...
    <run>/params.eqx        optional exact equinox leaves (same arrays, structure)

Reconstruction is: build SIREN with the recorded options and the recorded seed,
partition it, flatten, check the length against the saved vector, and
``eqx.combine(unflatten(saved), static)``.  The seed only matters for the parameter
initialisation, which is then overwritten by the saved arrays; the static leaves
(x_mins/x_maxs, flags) come from the freshly built model, so any key would do.

    python ldc_params.py --run <run dir>            # score + (re)make field figures
    python ldc_params.py --params P.npy --network N.json
"""

from __future__ import annotations

import pathlib

import numpy as np
import jax
import equinox as eqx
import pinn

PARAMS_FILE = "params_flat.npy"
NETWORK_FILE = "network.json"
LEAVES_FILE = "params.eqx"


class LidDrivenCavityProblem(pinn.Problem):
    """Attributes SIREN needs when it is rebuilt (the equations are not used here)."""

    x_min, x_max = 0.0, 1.0
    Re = 100.0
    problem_name = "LidDrivenCavity"
    ref_path = ""                     # filled by the caller / CLI


def model_meta(*, hidden_dims, seed=0, driver="", config=None, jax_version=None,
               jaxlib_version=None, backend=None, devices=None, extra=None) -> dict:
    """The `network.json` payload: everything needed to rebuild and to cite the run."""
    meta = {
        "model": "SIREN",
        "hidden_dims": list(hidden_dims),
        "seed": int(seed),
        "time_dependent": False,
        "periodic_bc": False,
        "n_inputs": 2,
        "n_outputs": 3,
        "omega_0": 1.0,
        "omega_hidden": 1.0,
        "n_fourier": 0,
        "learnable_omega": True,
        "skip_connection": False,
        "driver": driver,
        "config": config or {},
        "jax_version": jax_version or jax.__version__,
        "jaxlib_version": jaxlib_version,
        "backend": backend,
        "devices": devices,
    }
    if extra:
        meta.update(extra)
    return meta


def save_network(outdir, model, meta: dict, *, name: str = PARAMS_FILE,
                 write_leaves: bool = True) -> int:
    """Write the flat parameters (+ optional equinox leaves) and the metadata JSON."""
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    params, _static = eqx.partition(model, eqx.is_array)
    flat, _unflatten = jax.flatten_util.ravel_pytree(params)
    flat = np.asarray(flat, dtype=np.float64)
    np.save(outdir / name, flat)
    meta = dict(meta)
    meta["n_parameters"] = int(flat.size)
    meta["params_file"] = name
    meta["saved_utc"] = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if write_leaves:
        try:
            eqx.tree_serialise_leaves(str(outdir / LEAVES_FILE), params)
            meta["leaves_file"] = LEAVES_FILE
        except Exception as exc:                                  # noqa: BLE001
            meta["leaves_file"] = f"(not written: {type(exc).__name__}: {exc})"
    (outdir / NETWORK_FILE).write_text(__import__("json").dumps(meta, indent=2))
    return int(flat.size)


def build_model(meta: dict):
    """Rebuild the SIREN from `network.json` (parameters are not loaded yet)."""
    return pinn.SIREN(
        jax.random.PRNGKey(int(meta.get("seed", 0))),
        LidDrivenCavityProblem,
        time_dependent=bool(meta.get("time_dependent", False)),
        hidden_dims=tuple(meta["hidden_dims"]),
        periodic_bc=bool(meta.get("periodic_bc", False)),
        n_inputs=int(meta.get("n_inputs", 2)),
        n_outputs=int(meta.get("n_outputs", 3)),
        omega_0=float(meta.get("omega_0", 1.0)),
        omega_hidden=float(meta.get("omega_hidden", 1.0)),
        n_fourier=int(meta.get("n_fourier", 0)),
        learnable_omega=bool(meta.get("learnable_omega", True)),
        skip_connection=bool(meta.get("skip_connection", False)),
    )


def load_network(run_dir, params_name: str | None = None, network_name: str = NETWORK_FILE):
    """(model, meta) from a run directory; raises if the vector does not fit the net."""
    import json
    run_dir = pathlib.Path(run_dir)
    meta = json.loads((run_dir / network_name).read_text())
    params_name = params_name or meta.get("params_file", PARAMS_FILE)
    flat = np.load(run_dir / params_name)
    model = build_model(meta)
    params, static = eqx.partition(model, eqx.is_array)
    ref_flat, unflatten = jax.flatten_util.ravel_pytree(params)
    if int(flat.size) != int(ref_flat.size):
        raise ValueError(f"{run_dir / params_name}: {flat.size} parameters, but "
                         f"network {meta['hidden_dims']} needs {ref_flat.size}")
    return eqx.combine(unflatten(jnp_array(flat)), static), meta


def jnp_array(x):
    """float64 array (x64 must already be enabled by the caller)."""
    import jax.numpy as jnp
    return jnp.asarray(x, dtype=jnp.float64)


def rel_l2(model, ref, *, shift_p: bool = True) -> dict:
    """Relative L2 per channel against a pinn.Reference, plus the mean."""
    import jax.numpy as jnp

    @jax.jit
    def predict(m, coords):
        return jax.vmap(lambda c: m(c))(coords)

    coords = jnp.asarray(ref.coords, dtype=jnp.float64)
    pred = np.asarray(predict(model, coords))
    sol = np.asarray(ref.solution)
    channels = [str(c) for c in ref.channels]
    out = {}
    for i, ch in enumerate(channels):
        a, b = pred[:, i], sol[:, i]
        if ch == "p" and shift_p:
            a, b = a - a.mean(), b - b.mean()
        out[ch] = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    out["mean"] = float(np.mean([out[c] for c in channels]))
    return out
