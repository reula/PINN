#!/usr/bin/env python
"""Is the DSGNAR model machinery numerically sound on this backend?

The paper optimiser steers its trust region with `rho = actual/predicted reduction`,
where the prediction comes from the sketched Jacobian.  On the A30 that rho sits at
its 0.075 target and the loss falls geometrically; on the Intel tile it is erratic from
the first float64 step and the radius collapses.  This script measures, at the *initial
parameters and sampled points of the a30 preset*, the ingredients of that prediction:

  1. auto-diff consistency: jacrev vs jacfwd of the residual (relative error), and both
     against central finite differences along random directions (model-free check);
  2. DCT round-trip at the sizes the sketch uses (1665 and 512 columns);
  3. SVD reconstruction error for a 1665x1665 float64 matrix (the sketch solve).

Run it on each platform and compare the JSON/npz.  On CPU it is the reference; on the
Intel tile the same script answers whether the Jacobian itself is wrong (reverse-mode
bug) or whether the discrepancy lives in the probe/pchip radius control, which this
script deliberately does not touch.

    python ldc_jac_check.py --outdir /tmp/jac_cpu          # local, free
    python ldc_jac_check.py --outdir /data/.../jac_intel   # on a GPU node
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
ap.add_argument("--points", type=int, default=4, help="collocation points in the Jacobian")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--allow-cpu", action="store_true",
                help="no-op here: this check is CPU-friendly by design; the flag keeps "
                     "the cluster scripts' convention")
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
import jax.scipy.fft                                           # noqa: E402,F401

jax.config.update("jax_enable_x64", True)
BACKEND = jax.default_backend()
DEVICES = [str(d) for d in jax.devices()]
op = pinn.operators
print(f"[jac ] backend {BACKEND} | devices {DEVICES} | outdir {OUT}", flush=True)


class LidDrivenCavityProblem(pinn.Problem):
    """a30-preset problem (same class as ldc_run.py / ldc_paper_run.py)."""

    x_min, x_max = 0.0, 1.0
    Re = 100.0
    problem_name = "LidDrivenCavity"
    ref_path = str(REPO / "ldc" / "reference_gen" / "ldc_reference.npz")

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
        res_div = op.grad(u_, coords, 0) + op.grad(v_, coords, 1)
        return jnp.array([res_u, res_v, 2.0 * res_div])

    def bc_residual(self, model, coords):
        u, v = model(coords)[0], model(coords)[1]
        u_target = jnp.where(jnp.isclose(coords[1], 1.0), 4.0 * coords[0] * (1.0 - coords[0]),
                             0.0)
        return jnp.array([u - u_target, v, 0.0])

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


# ── the exact a30 initial state: seed 0, 48-48-48, the same sampled points ─────────
HIDDEN, N_PDE, N_BC = (48, 48, 48), 2 ** 13, 2 ** 12
problem = LidDrivenCavityProblem(n_pde=N_PDE, n_bc=N_BC)
key = jax.random.PRNGKey(ARGS.seed)
model = pinn.SIREN(key, LidDrivenCavityProblem, time_dependent=False, hidden_dims=HIDDEN,
                   periodic_bc=False, n_inputs=2, n_outputs=3)
params, static = eqx.partition(model, eqx.is_array)
flat0, unflatten = jax.flatten_util.ravel_pytree(params)
fns, samplers = problem.residual_fns(), problem.samplers()
key, sk = jax.random.split(key)
pde_pts = samplers["pde"](sk, N_PDE, 0.0, 0.0)[: ARGS.points]
bc_pts = samplers["bc"](sk, N_BC, 0.0, 0.0)[: ARGS.points]
print(f"[jac ] d={flat0.size} | pde points {pde_pts.shape} | bc points {bc_pts.shape}",
      flush=True)


def _model_of(flat):
    return eqx.combine(unflatten(flat), static)


def _residual_vector(fn, pts):
    def r(flat):
        m = _model_of(flat)
        out = jax.vmap(lambda c: fn(m, c))(pts)
        return out.ravel()
    return r


results = {"backend": BACKEND, "devices": DEVICES, "jax": jax.__version__,
           "n_parameters": int(flat0.size), "points": ARGS.points}

# 1) reverse vs forward mode, and central finite differences along random directions
r_pde = _residual_vector(fns["pde"], pde_pts)
r_bc = _residual_vector(fns["bc"], bc_pts)
arrays = {}
for tag, r in (("pde", r_pde), ("bc", r_bc)):
    J_rev = jax.jacrev(r)(flat0)
    J_fwd = jax.jacfwd(r)(flat0)
    jax.block_until_ready(J_rev)
    scale = float(jnp.max(jnp.abs(J_fwd))) + 1e-300
    print(f"[jac ] {tag}: jacrev vs jacfwd  max|diff|/max|J_fwd| = "
          f"{float(jnp.max(jnp.abs(J_rev - J_fwd))) / scale:.3e}", flush=True)
    fd_rel = []
    for i in range(3):
        v = jax.random.normal(jax.random.PRNGKey(1000 + i), (flat0.size,), dtype=jnp.float64)
        v = v / jnp.linalg.norm(v)
        eps = 1e-6 * (1.0 + float(jnp.linalg.norm(flat0)) / np.sqrt(flat0.size))
        fd = (r(flat0 + eps * v) - r(flat0 - eps * v)) / (2 * eps)
        ad = J_fwd @ v
        fd_rel.append(float(jnp.linalg.norm(fd - ad) / (jnp.linalg.norm(ad) + 1e-300)))
    results[f"jacrev_vs_jacfwd_rel_{tag}"] = float(
        jnp.max(jnp.abs(J_rev - J_fwd)) / scale)
    results[f"finite_difference_rel_{tag}"] = fd_rel
    print(f"[jac ] {tag}: central FD vs jacfwd rel errors {['%.2e' % x for x in fd_rel]}",
          flush=True)
    arrays[f"J_rev_{tag}"] = np.asarray(J_rev[:4])
    arrays[f"J_fwd_{tag}"] = np.asarray(J_fwd[:4])

# 2) DCT round-trip at sketch sizes
for n in (512, 1665):
    x = jnp.asarray(np.random.default_rng(7).normal(size=(4, n)))
    y = jax.scipy.fft.idct(jax.scipy.fft.dct(x, type=2, norm="ortho"), type=2, norm="ortho")
    err = float(jnp.max(jnp.abs(y - x)))
    results[f"dct_roundtrip_maxerr_{n}"] = err
    print(f"[jac ] dct/idct n={n}: max roundtrip err {err:.3e}", flush=True)

# 3) SVD of the sketch-sized matrix
n = 1665
A = jnp.asarray(np.random.default_rng(11).normal(size=(n, n)))
U, s, Vt = jnp.linalg.svd(A, full_matrices=False)
recon = float(jnp.linalg.norm((U * s) @ Vt - A) / jnp.linalg.norm(A))
results["svd_recon_rel_1665"] = recon
results["svd_top_singular"] = [float(x) for x in np.asarray(s[:5])]
results["svd_sum_singular"] = float(jnp.sum(s))
print(f"[jac ] svd 1665: relative reconstruction error {recon:.3e}", flush=True)
arrays["svd_top_singular"] = np.asarray(s[:5])

np.savez_compressed(OUT / "jac_arrays.npz", **arrays)
(OUT / "jac_check.json").write_text(json.dumps(results, indent=2))
print(f"[jac ] written {OUT}/jac_check.json and jac_arrays.npz", flush=True)
