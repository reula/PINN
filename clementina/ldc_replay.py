#!/usr/bin/env python
"""Replay the first DSGNAR steps of ldc_run.py exactly, dumping each step's metrics.

The isolated one-step probe is clean on the Intel tile (rho 0.956 accept, no NaN),
yet the real run's step 1 gets rho = NaN.  The difference is the random state: the real
run's sketch comes from `init_state(k2)` where k2 is the second half of the key after
the model and the initial conditions were drawn.  This script reproduces that chain
key for key, runs a few steps, and reports rho / accepted / radius plus a NaN census of
the metrics arrays at every step.

    python ldc_replay.py --outdir /tmp/replay_cpu                 # reference
    python ldc_replay.py --outdir /data/.../replay_intel          # GPU node
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--outdir", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--float32-steps", type=int, default=4)
ap.add_argument("--float64-steps", type=int, default=2)
ARGS = ap.parse_args()
OUT = pathlib.Path(ARGS.outdir).resolve()
OUT.mkdir(parents=True, exist_ok=True)


def _repo_root() -> pathlib.Path:
    marker = pathlib.Path("Evolution_try") / "vendor" / "dsgnar_pinn" / "pinn" / "__init__.py"
    for start in (os.environ.get("CXXI_REPO"), str(pathlib.Path(__file__).resolve().parent),
                  str(pathlib.Path.cwd())):
        if not start:
            continue
        base = pathlib.Path(start).resolve()
        for p in (base, *base.parents):
            if (p / marker).exists():
                return p
    raise RuntimeError("repo root not found: set CXXI_REPO")


REPO = _repo_root()
sys.path.insert(0, str(REPO / "Evolution_try" / "vendor" / "dsgnar_pinn"))

import numpy as np                                             # noqa: E402
import jax                                                     # noqa: E402
import jax.numpy as jnp                                        # noqa: E402
import equinox as eqx                                          # noqa: E402
import pinn                                                    # noqa: E402

op = pinn.operators
BACKEND = jax.default_backend()
print(f"[repl] backend {BACKEND} | devices {[str(d) for d in jax.devices()]}", flush=True)


class LidDrivenCavityProblem(pinn.Problem):
    x_min, x_max = 0.0, 1.0
    Re = 100.0
    problem_name = "LidDrivenCavity"

    def __init__(self, *, n_pde, n_bc, n_ic=0):
        self.n_pde, self.n_bc, self.n_ic = n_pde, n_bc, n_ic

    def residual_fns(self):
        return {"pde": self.pde_residual, "bc": self.bc_residual}

    def pde_residual(self, model, coords):
        u_ = lambda c: model(c)[0]
        v_ = lambda c: model(c)[1]
        p_ = lambda c: model(c)[2]
        u, v = u_(coords), v_(coords)
        inv_Re = 1.0 / self.Re
        res_u = (u * op.grad(u_, coords, 0) + v * op.grad(u_, coords, 1)
                 + op.grad(p_, coords, 0)
                 - inv_Re * (op.grad2(u_, coords, 0) + op.grad2(u_, coords, 1)))
        res_v = (u * op.grad(v_, coords, 0) + v * op.grad(v_, coords, 1)
                 + op.grad(p_, coords, 1)
                 - inv_Re * (op.grad2(v_, coords, 0) + op.grad2(v_, coords, 1)))
        return jnp.array([res_u, res_v, 2.0 * (op.grad(u_, coords, 0)
                                               + op.grad(v_, coords, 1))])

    def bc_residual(self, model, coords):
        u, v = model(coords)[0], model(coords)[1]
        u_t = jnp.where(jnp.isclose(coords[1], 1.0), 4.0 * coords[0] * (1.0 - coords[0]), 0.0)
        return jnp.array([u - u_t, v, 0.0])

    def samplers(self):
        return {"pde": self._sample_pde, "bc": self._sample_bc}

    def _sample_pde(self, key, n, *_):
        n_lid, n_corner = n // 16, n // 16
        n_bulk = n - n_lid - n_corner
        k_bulk, k_lx, k_ly, k_l, k_r = jax.random.split(key, 5)
        bulk = jax.random.uniform(k_bulk, (n_bulk, 2), minval=0.0, maxval=1.0)
        lid = jnp.stack([jax.random.uniform(k_lx, (n_lid,)),
                         jax.random.uniform(k_ly, (n_lid,), minval=0.9, maxval=1.0)], axis=1)
        half = n_corner // 2
        s1 = jax.random.uniform(k_l, (half, 2))
        left = jnp.stack([0.1 * s1[:, 0] ** 2, 1.0 - 0.1 * s1[:, 1] ** 2], axis=1)
        s2 = jax.random.uniform(k_r, (half, 2))
        right = jnp.stack([1.0 - 0.1 * s2[:, 0] ** 2, 1.0 - 0.1 * s2[:, 1] ** 2], axis=1)
        return jnp.concatenate([bulk, lid, left, right], axis=0)

    def _sample_bc(self, key, n, *_):
        n_corner = n // 16
        n_uniform = n - n_corner
        half = n_corner // 2
        k_pt, k_dim, k_face, k_l, k_r = jax.random.split(key, 5)
        pts = jax.random.uniform(k_pt, (n_uniform, 2), minval=0.0, maxval=1.0)
        face_dim = jax.random.randint(k_dim, (n_uniform,), 0, 2)
        face_val = jax.random.randint(k_face, (n_uniform,), 0, 2).astype(pts.dtype)
        pts = pts.at[jnp.arange(n_uniform), face_dim].set(face_val)
        left = jnp.stack([0.1 * jax.random.uniform(k_l, (half,)), jnp.ones((half,))], axis=1)
        right = jnp.stack([1.0 - 0.1 * jax.random.uniform(k_r, (half,)), jnp.ones((half,))],
                          axis=1)
        return jnp.concatenate([pts, left, right], axis=0)


HIDDEN, N_PDE, N_BC = (48, 48, 48), 2 ** 13, 2 ** 12
SKETCH, BATCH, PROBE_BATCH, SUB_BATCH = 1665, 1024, 128, 4
MIN_RADIUS, TARGET_RHO, PDE_WEIGHT = 1e-14, 0.075, 1e-2

# ── exactly ldc_run.py's initialisation order ──────────────────────────────────
# ldc_run.py enables x64 *before* building the model; that decides the dtype of the
# initial parameters (built in f64, cast to f32 for the first pass), so reproduce it.
jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(ARGS.seed)
problem = LidDrivenCavityProblem(n_pde=N_PDE, n_bc=N_BC)
model = pinn.SIREN(key, LidDrivenCavityProblem, time_dependent=False, hidden_dims=HIDDEN,
                   periodic_bc=False, n_inputs=2, n_outputs=3)
params, static = eqx.partition(model, eqx.is_array)
flat0, unflatten = jax.flatten_util.ravel_pytree(params)
fns, samplers = problem.residual_fns(), problem.samplers()
key, sk = jax.random.split(key)                     # initial CONDITIONS
CONDITIONS = ((fns["pde"], samplers["pde"](sk, N_PDE, 0.0, 0.0)),
              (fns["bc"], samplers["bc"](sk, N_BC, 0.0, 0.0)))
WEIGHTS = (PDE_WEIGHT, 1.0)

solver = pinn.optimiser.Optimiser(
    residual_sketch=SKETCH, parameter_sketch=SKETCH, n_hashes=4, sub_batch_size=SUB_BATCH,
    batch_size=BATCH, probe_batch_size=PROBE_BATCH, n_probes=24, window_scale=3.0,
    min_radius=MIN_RADIUS, max_radius=1e3, pchip_grid=512, newton_iters=80)
key, k2 = jax.random.split(key)
opt_state = solver.init_state(k2, initial_radius=1.0)

report = {"backend": BACKEND, "devices": [str(d) for d in jax.devices()],
          "jax": jax.__version__, "steps": []}
_ALL_METRICS: list = []

step_total = 0
for bits, n_steps in ((32, ARGS.float32_steps), (64, ARGS.float64_steps)):
    jax.config.update("jax_enable_x64", bits == 64)
    dtype = jnp.float64 if bits == 64 else jnp.float32
    params = jax.tree_util.tree_map(
        lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, params)
    key, sk = jax.random.split(key)
    conds = ((fns["pde"], samplers["pde"](sk, N_PDE, 0.0, 0.0)),
             (fns["bc"], samplers["bc"](sk, N_BC, 0.0, 0.0)))
    for _ in range(n_steps):
        loss, params, opt_state, metrics = solver.step(
            params=params, static=static, state=opt_state, conditions=conds,
            target_rho=TARGET_RHO, weights=WEIGHTS)
        jax.block_until_ready(loss)
        step_total += 1
        entry = {"step": step_total, "bits": bits, "loss": float(loss),
                 "rho": float(metrics["rho"]), "accepted": bool(metrics["accepted"]),
                 "radius": float(metrics["radius"]), "lam": float(metrics["lam"]),
                 "arrays": {}}
        for k in ("S", "probe_radii", "probe_rhos", "probe_losses", "pchip_y"):
            a = np.asarray(metrics[k])
            finite = a[np.isfinite(a)]
            entry["arrays"][k] = {
                "nan": int(np.isnan(a).sum()), "inf": int(np.isinf(a).sum()),
                "min": float(finite.min()) if finite.size else None,
                "max": float(finite.max()) if finite.size else None}
        _ALL_METRICS.append({k: np.asarray(metrics[k]) for k in
                             ("S", "probe_radii", "probe_rhos", "probe_losses",
                              "pchip_y", "lam")})
        report["steps"].append(entry)
        print(f"[repl] {step_total:3d} float{bits}  loss {entry['loss']:.4e}  "
              f"rho {entry['rho']:+.3f}  {'acc' if entry['accepted'] else 'rej'}  "
              f"radius {entry['radius']:.2e}", flush=True)
        bad = {k: v for k, v in entry["arrays"].items() if v["nan"] or v["inf"]}
        if bad:
            print(f"[repl]      NaN/inf stages: {bad}", flush=True)

# keep the raw arrays of every step for cross-platform diffing
np.savez_compressed(
    OUT / "replay_arrays.npz",
    **{f"step{e['step']}_{k}": np.asarray(a)
       for e, m in zip(report["steps"], _ALL_METRICS)
       for k, a in m.items()})
(OUT / "replay.json").write_text(json.dumps(report, indent=2))
print(f"[repl] written {OUT}/replay.json", flush=True)
