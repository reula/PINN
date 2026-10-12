#!/usr/bin/env python
"""Audit the DSGNAR sketch against the exact Jacobian.

    <python> scripts/dsgnar_sketch_audit.py [--k 4 8] [--keys 8]

Why this exists: the sibling project measured the sketched model as over-optimistic
by a factor 12 at `s = 128`, which walked its trust region to `delta_min` in ~28
iterations.  Porting the cure here did not help (see the measurement table beside
the DSGNAR knobs in `config.py`), and the reason is that `s = d/3` is a far milder
compression, so the over-optimism is small.  That argument was inference.  This
script measures it instead: it forms the EXACT Jacobian -- cheap here, `n = 2201` so
`J` is 38.7 MB, and `jvp_columns` produces all 2201 columns in under a second --
and compares, at the same point in parameter space:

  * the model decrease the sketch predicts against the exact one, which isolates
    the sketch from the factor-of-two convention in the paper's `pred` (that factor
    cancels in the ratio, so what is left is the sketch's own error);
  * the Gauss-Newton step against the step DSGNAR actually lifts, by angle;
  * the singular spectra of `J` and of the sketch, and the numerical rank -- the
    paper only asks that `s` exceed it, which we can now check rather than assume.

It is run at two points: a random initialisation, and a converged solution with a
fresh batch, which is the situation the rounds mechanism creates and where the one
observed stall happened.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

from wave_pinn.config import Config
from wave_pinn.losses import Objective
from wave_pinn.model import init_params
from wave_pinn.optim.dsgnar import (N_NEWTON, count_sketch, jvp_columns, lift_srct,
                                    make_srct, solve_subproblems, srct_columns,
                                    step_and_pred, _dct_matrix)
from wave_pinn.sampling import make_batch
from wave_pinn.train import configure_jax

N = 2201


def data_term(r, J, p):
    """``½(‖r‖² − ‖r + Jp‖²)`` -- the exact quadratic model decrease, λ = 0."""
    rp = r + J @ p
    return 0.5 * (float(jnp.dot(r, r)) - float(jnp.dot(rp, rp)))


def dsgnar_step(res_fn, r, flat, s, n, dtype, key, K, lam, chunk):
    """Reproduce one DSGNAR model evaluation for a single λ."""
    srct = make_srct(key, n, s)
    dct = _dct_matrix(n, dtype)
    B = srct_columns(srct, n, s, dtype, dct_matrix=dct)
    j_cols = jvp_columns(res_fn, flat, B, chunk).T          # (M, s)
    j_sketch, r_tilde = count_sketch(j_cols, r, key, s, k_hashes=K)
    u, sing, vt = jnp.linalg.svd(j_sketch, full_matrices=False)
    g = sing * (u.T @ r_tilde)
    p_sk, pred = step_and_pred(sing, g, vt, lam)
    return lift_srct(p_sk, srct, n), float(pred), jnp.asarray(sing)


def audit(label, flat, cfg, res_fn, J, r, s, n, dtype, M, ks, n_keys, chunk, lams,
          scan_s=None):
    print(f"\n=== {label} ===")
    print(f"    loss {cfg.__class__.__name__}  |r| {float(jnp.linalg.norm(r)):.3e}   "
          f"loss {float(jnp.dot(r, r)) / M:.3e}")
    sv_true = np.asarray(jnp.linalg.svd(J, compute_uv=False))
    print(f"    exact J: {J.shape}, singular values max {sv_true[0]:.3e} "
          f"min {sv_true[-1]:.3e}, ratio {sv_true[0] / max(sv_true[-1], 1e-300):.3e}")
    for thr in (1e-8, 1e-10, 1e-12):
        rank = int((sv_true > thr * sv_true[0]).sum())
        print(f"      numerical rank at {thr:.0e} of the largest: {rank} / {n}  "
              f"(s = {s})")
    p_true_cache = {}
    for lam in lams:
        # the exact Gauss-Newton step from the full Jacobian
        JtJ = J.T @ J
        p_true = jnp.linalg.solve(JtJ + lam * jnp.eye(n, dtype=dtype), -(J.T @ r))
        p_true_cache[lam] = p_true
        true_dec = data_term(r, J, p_true) * 2.0 / M         # in the objective's units
        print(f"\n  lambda = {lam:.3e}   |p_true| {float(jnp.linalg.norm(p_true)):.3e}   "
              f"exact model decrease {true_dec:.3e}")
        print(f"    {'K':>3} {'cos(p,p_true)':>14} {'rel err':>10} {'pred/exact':>11} "
              f"{'pred/exact(med)':>16}")
        for K in ks:
            coss, errs, ratios = [], [], []
            for i in range(n_keys):
                key = jax.random.PRNGKey(1000 * K + i)
                p_sk, pred, _ = dsgnar_step(res_fn, r, flat, s, n, dtype, key, K,
                                            lam, chunk)
                np_sk = np.asarray(p_sk)
                np_tr = np.asarray(p_true)
                nn = np.linalg.norm(np_tr)
                coss.append(float(np_sk @ np_tr / max(np.linalg.norm(np_sk) * nn, 1e-300)))
                errs.append(float(np.linalg.norm(np_sk - np_tr) / max(nn, 1e-300)))
                dec = data_term(r, J, p_sk) * 2.0 / M
                ratios.append(pred * (1.0 / M) / dec if dec != 0 else np.nan)
            print(f"    {K:>3} {np.median(coss):>14.4f} {np.median(errs):>10.3f} "
                  f"{np.median(ratios):>11.3f} {np.median(ratios):>16.3f}")
            print(f"        (cos range {min(coss):+.4f} to {max(coss):+.4f} over "
                  f"{n_keys} sketch draws)")
    if scan_s:
        print(f"\n  sketch-size scan at lambda = 1e-3 and 1.0 (K = {ks[0]})")
        print(f"    {'s':>6} {'s/d':>6} {'cos (lam=1e-3)':>15} {'cos (lam=1)':>12}")
        for s_i in scan_s:
            row = []
            for lam in (1e-3, 1.0):
                coss = []
                for i in range(n_keys):
                    key = jax.random.PRNGKey(7 * s_i + i)
                    p_sk, _, _ = dsgnar_step(res_fn, r, flat, s_i, n, dtype, key, ks[0],
                                             lam, chunk)
                    np_sk, np_tr = np.asarray(p_sk), np.asarray(p_true_cache[lam])
                    coss.append(float(np_sk @ np_tr
                                      / max(np.linalg.norm(np_sk) * np.linalg.norm(np_tr), 1e-300)))
                row.append(np.median(coss))
            print(f"    {s_i:>6} {s_i / n:>6.2f} {row[0]:>15.4f} {row[1]:>12.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--keys", type=int, default=6)
    ap.add_argument("--s", type=int, nargs="+", default=None,
                    help="sketch sizes to scan; default is cfg.dsgnar_sketch alone.  "
                         "A scan reuses the exact J, so it is cheap, and it tests the "
                         "prediction that the COLUMN sketch is the limit here: the "
                         "paper wants s to comfortably exceed the numerical rank of J, "
                         "and at s = d/3 it does not.")
    ap.add_argument("--skip-cold", action="store_true",
                    help="audit only the rounds situation; the cold-start point costs "
                         "a full 2201^2 SVD on its own")
    ap.add_argument("--converged", default="/tmp/conv_old/theta.npy",
                    help="a converged theta; skipped if absent")
    a = ap.parse_args()

    cfg = Config(T=2.0, n_coll=N, sampler="random", dsgnar_sketch=733)
    configure_jax(cfg)
    n = N
    s = int(cfg.dsgnar_sketch)
    dtype = jnp.float64
    chunk = int(cfg.dsgnar_chunk)
    M = n

    lams = [1e-8, 1e-3, 1.0, 1e2]

    # ---------------------------------------------------------------- cold start
    if a.skip_cold:
        print("(cold-start point skipped)")
    key0 = jax.random.PRNGKey(cfg.seed)
    params = init_params(cfg, key0)
    obj = Objective(cfg, make_batch(cfg, key0), params)
    flat = jax.flatten_util.ravel_pytree(params)[0]
    res_fn = obj._residual
    r = res_fn(flat)
    if not a.skip_cold:
        Jt = jvp_columns(res_fn, flat, jnp.eye(n, dtype=dtype), chunk)   # (n, M) = J^T
        audit("cold start: random parameters", flat, obj.cfg, res_fn, Jt.T, r, s, n,
              dtype, M, a.k, a.keys, chunk, lams, scan_s=a.s)

    # ------------------------------------------------- the rounds situation
    if os.path.exists(a.converged):
        warm = jnp.asarray(np.load(a.converged), dtype=dtype)
        # a FRESH batch, which is what a round begins with
        obj_w = Objective(cfg, make_batch(cfg, jax.random.PRNGKey(cfg.seed + 7919)), params)
        res_w = obj_w._residual
        r_w = res_w(warm)
        Jt_w = jvp_columns(res_w, warm, jnp.eye(n, dtype=dtype), chunk)
        audit("rounds situation: converged theta, fresh batch", warm, obj_w.cfg,
              res_w, Jt_w.T, r_w, s, n, dtype, M, a.k, a.keys, chunk, lams, scan_s=a.s)
    else:
        print(f"\n(no converged theta at {a.converged}; run a DSGNAR phase to make one)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
