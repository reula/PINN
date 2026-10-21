#!/usr/bin/env python
"""Where does the NaN come from on the Intel backend?

At the *initial* state of the a30 preset the A30 and the CPU both take step 1 with
rho = 0.82 (accept); the Intel tile gets rho = NaN and rejects.  rho is built from

    rho = actual_reduction / predicted_reduction

and the predicted reduction comes from the sketched Jacobian.  This script rebuilds
that chain at the initial parameters, in both dtypes, and reports NaNs/amplitude at
each stage:

    dense Jacobian (jacrev, one macro-batch)  ->  sketched B, r_sketch
      ->  svd(B) -> g  ->  secular-equation solve (probe lambdas) -> predicted reduction

Whatever is already NaN at a stage is where the backend misbehaves; the stage before
it is the last good one.  Run it on CPU (reference) and on the Intel tile and compare.

    python ldc_nan_probe.py --outdir /tmp/nan_cpu
    python ldc_nan_probe.py --outdir /data/.../nan_intel      # GPU node
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
import pinn.optimiser as O                                     # noqa: E402

op = pinn.operators
BACKEND = jax.default_backend()
print(f"[nan ] backend {BACKEND} | devices {[str(d) for d in jax.devices()]}", flush=True)


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
SKETCH, BATCH, PROBE_BATCH, SUB_BATCH, N_HASHES = 1665, 1024, 128, 4, 4
N_PROBES, WINDOW, NEWTON_ITERS, PCHIP_GRID = 24, 3.0, 80, 512


def _stats(name: str, x, report: dict) -> None:
    a = np.asarray(x)
    n_nan = int(np.isnan(a).sum())
    finite = a[np.isfinite(a)]
    report[name] = {
        "shape": list(a.shape), "dtype": str(a.dtype), "nan": n_nan,
        "size": int(a.size),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "absmax": float(np.abs(finite).max()) if finite.size else None,
    }
    print(f"[nan ] {name:28s} dtype={report[name]['dtype']:8s} nan={n_nan:6d}/{a.size:7d} "
          f"absmax={report[name]['absmax']}", flush=True)


problem = LidDrivenCavityProblem(n_pde=N_PDE, n_bc=N_BC)
fns = problem.residual_fns()
samplers = problem.samplers()
report = {"backend": BACKEND, "devices": [str(d) for d in jax.devices()],
          "jax": jax.__version__, "stages": {}}

for bits in (32, 64):
    jax.config.update("jax_enable_x64", bits == 64)
    dtype = jnp.float64 if bits == 64 else jnp.float32
    # same RNG chain as ldc_run.py: seed -> model; split -> initial conditions; split -> pass
    key = jax.random.PRNGKey(ARGS.seed)
    model = pinn.SIREN(key, LidDrivenCavityProblem, time_dependent=False, hidden_dims=HIDDEN,
                       periodic_bc=False, n_inputs=2, n_outputs=3)
    params, static = eqx.partition(model, eqx.is_array)
    params = jax.tree_util.tree_map(
        lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, params)
    flat, unflatten = jax.flatten_util.ravel_pytree(params)
    key, _sk0 = jax.random.split(key)          # initial CONDITIONS (unused here)
    key, sk = jax.random.split(key)            # this pass's points
    conds = (("pde", fns["pde"], samplers["pde"](sk, N_PDE, 0.0, 0.0)),
             ("bc",  fns["bc"],  samplers["bc"](sk,  N_BC,  0.0, 0.0)))
    weights = (1e-2, 1.0)

    tag = f"float{bits}"
    print(f"\n[nan ] ===== {tag} =====", flush=True)
    st = report["stages"].setdefault(tag, {})

    # 1. one macro-batch of the dense Jacobian
    name, fn, pts = conds[0]
    J, r = O._jacobian_block(flat, unflatten, static, fn, pts[:BATCH], SUB_BATCH)
    jax.block_until_ready(J)
    _stats(f"{tag} J_macrobatch[{name}]", J, st)
    _stats(f"{tag} r_macrobatch[{name}]", r, st)

    # 2. sketched system per condition, accumulated like _step_impl
    srct = O._make_srct(jax.random.PRNGKey(12345), flat.size, SKETCH, dtype)
    B = jnp.zeros((SKETCH, SKETCH), dtype)
    r_sk = jnp.zeros((SKETCH,), dtype)
    for name, fn, pts in conds:
        Bi, rsi, rfi = O._kernel_sketch(
            flat, unflatten, static, fn, pts, jax.random.PRNGKey(777 + int(weights[0] * 0)),
            srct, residual_sketch=SKETCH, parameter_sketch=SKETCH, n_hashes=N_HASHES,
            sub_batch_size=SUB_BATCH, batch_size=BATCH)
        scale = jnp.sqrt(weights[0] if name == "pde" else weights[1]) / jnp.sqrt(rfi.shape[0])
        B = B + scale * Bi
        r_sk = r_sk + scale * rsi
        _stats(f"{tag} B_cond[{name}]", Bi, st)
        _stats(f"{tag} r_sketch[{name}]", rsi, st)
    jax.block_until_ready(B)
    _stats(f"{tag} B", B, st)
    _stats(f"{tag} r_sketch", r_sk, st)

    # 3. SVD of the sketched core
    U, S, Vt = jnp.linalg.svd(B, full_matrices=False)
    jax.block_until_ready(S)
    _stats(f"{tag} S_singular", S, st)
    g = S * (U.T @ r_sk)
    _stats(f"{tag} g", g, st)
    st["cond_S"] = float(jnp.linalg.cond(B)) if bits == 64 else None

    # 4. secular-equation solve over the probe window (state radius = 1.0)
    cur_r = jnp.clip(jnp.array(1.0, dtype), 1e-14, 1e3)
    r_lo, r_hi = jnp.maximum(cur_r / WINDOW, 1e-14), jnp.minimum(cur_r * WINDOW, 1e3)
    probe_radii = jnp.geomspace(r_lo, r_hi, N_PROBES)
    lambdas = O._solve_subproblems(S, g, probe_radii, NEWTON_ITERS)
    jax.block_until_ready(lambdas)
    _stats(f"{tag} probe_lambdas", lambdas, st)

    # predicted reductions exactly as _step_and_pred does
    def _pred(lam):
        denom = S ** 2 + lam
        pred = jnp.sum(g ** 2 * (S ** 2 + 2 * lam) / denom ** 2)
        return pred

    preds = jax.vmap(_pred)(lambdas)
    _stats(f"{tag} probe_preds", preds, st)

    # 5. pchip of rho (the radius selector); probe_rhos need actual reductions, skip here
    fine_log_r = jnp.linspace(jnp.log(r_lo), jnp.log(r_hi), PCHIP_GRID)
    _stats(f"{tag} probe_radii", probe_radii, st)
    st["fine_log_r_range"] = [float(fine_log_r[0]), float(fine_log_r[-1])]

(OUT / "nan_probe.json").write_text(json.dumps(report, indent=2))
print(f"\n[nan ] written {OUT}/nan_probe.json", flush=True)
