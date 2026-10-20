#!/usr/bin/env python
"""Can the network REPRESENT the (rotated) reference, and what does the PINN loss say about it?
    .venv/bin/python scripts/fit_reference.py runs/<run> --ref <specref .npz> [--iters 5000]

Two measurements for a run whose error against the spectral reference has stopped improving:

(a) The reference under the run's own objective: its total loss (the training loss, same weights,
    pins, centre-of-mass term) on fresh training samples, rotated onto the run's data axis
    (`cfg.lam_bc_axis`).  The reference is a spectral fit on its own collocation points; this is
    its residual on the points the network trains on.

(b) The representation ceiling: the run's network, warm-started from its PINN parameters (or from
    scratch with --scratch), fitted by plain regression to the rotated reference's h and lambda
    on a training-distribution sample -- no PDE involved -- with SSBroyden on the merged Crunch.
    Then graded as grade_rotated.py grades a run, and its PINN loss evaluated as in (a).

Reading: if the fit cannot get h much below the PINN run's error, the floor is the network's
capacity; if it can, but its PINN loss is ABOVE the run's, the PINN optimum trades the exact
solution for a lower residual -- the objective, not the network, sets the floor.
Writes <run>/fit_reference.json (fit_reference_scratch.json with --scratch).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grade_rotated import rotation_to                      # noqa: E402
from stationary.evaluate import load_run                  # noqa: E402
from stationary.geometry import Fields, christoffel       # noqa: E402
from stationary.losses import total_loss                  # noqa: E402
from stationary.model import point_fields                 # noqa: E402
from stationary.problem import sample_shell               # noqa: E402
from stationary.specref import SpectralAxisym, compare    # noqa: E402
from stationary.train import _crunch_minimize, exact_asset, make_batch  # noqa: E402


class RotatedReference:
    """The axisymmetric spectral reference rotated by R (R z = data axis), with the model interface
    the losses use: apply(c, X) -> batched Fields, Gamma = Christoffel(h) (metric-only)."""
    derives_gamma = True

    def __init__(self, spec: SpectralAxisym, R):
        self.spec, self.R = spec, jnp.asarray(R)

    def h_lam(self, c):
        R, s = self.R, self.spec

        def h_of(x):
            y = R.T @ x
            return R @ (s.exact(y).h + s.delta(c, y)[0]) @ R.T

        def lam_of(x):
            y = R.T @ x
            return s.exact(y).lam + s.delta(c, y)[1]
        return h_of, lam_of

    def fields_of(self, c):
        h_of, lam_of = self.h_lam(c)
        return lambda x: Fields(h_of(x), christoffel(h_of, x), lam_of(x))

    def apply(self, c, X):
        return jax.vmap(self.fields_of(c))(X)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--params", default="params.pkl")
    ap.add_argument("--iters", type=int, default=5000, help="regression iterations (SSBroyden)")
    ap.add_argument("--block", type=int, default=250)
    ap.add_argument("--n", type=int, default=16384, help="regression points (training distribution)")
    ap.add_argument("--samples", type=int, default=3, help="fresh samples for the PINN losses")
    ap.add_argument("--scratch", action="store_true", help="fit from a fresh initialisation")
    a = ap.parse_args(argv)

    cfg, model, state = load_run(a.run, a.params)[:3]
    ex = exact_asset(cfg)
    lam_inf = jnp.asarray(cfg.lam_inf if cfg.lam_inf is not None else cfg.lam_inf_init)
    R = rotation_to(cfg.lam_bc_axis if cfg.lam_bc_axis is not None else (0, 0, 1))
    z = np.load(a.ref)
    spec = SpectralAxisym(cfg, ex, ls=tuple(int(v) for v in z["ls"]), n_radial=int(z["n_radial"]))
    rref = RotatedReference(spec, R)
    c_ref = jnp.asarray(z["c"])
    ref_fields = rref.fields_of(c_ref)

    def pinn_losses(st, mdl):
        f = jax.jit(lambda b: total_loss(st, b, cfg, mdl, ex, lam_inf=lam_inf))
        out = []
        for i in range(a.samples):
            v, parts = f(make_batch(jax.random.PRNGKey(424242 + 31 * i), cfg))
            out.append((float(v), {k: float(p) for k, p in parts.items()}))
        return out

    def grade(params):
        cm = compare(point_fields(model, params), ref_fields, ex, cfg, modes=False)
        return {k: cm[k] for k in ("rms_dh", "max_dh", "rms_dlam", "max_dlam",
                                   "perturbation_rms_h", "perturbation_rms_lam")} | {
            "bins": [{k: b[k] for k in ("rho_lo", "rho_hi", "rms_dh", "rms_dlam")} for b in cm["bins"]]}

    res = {"run": a.run, "ref": a.ref, "axis": list(map(float, R @ np.array([0, 0, 1.0])))}

    # (a) the reference under the run's objective
    ra = pinn_losses({"net": c_ref}, rref)
    res["reference_pinn_loss"] = [v for v, _ in ra]
    res["reference_pinn_parts"] = ra[0][1]
    print(f"[fit] (a) reference, rotated, under the run's loss: "
          + " ".join(f"{v:.3e}" for v, _ in ra), flush=True)
    print("[fit]     parts: " + "  ".join(f"{k}={v:.2e}" for k, v in ra[0][1].items() if v), flush=True)

    # the PINN run itself, for comparison
    rp = pinn_losses(state, model)
    res["run_pinn_loss"] = [v for v, _ in rp]
    res["run_grade"] = grade(state["net"])
    print(f"[fit]     the run: loss {rp[0][0]:.3e}, rms dh {res['run_grade']['rms_dh']:.3e}, "
          f"rms dlambda {res['run_grade']['rms_dlam']:.3e}", flush=True)

    # (b) regression of the network onto the reference
    xs = sample_shell(jax.random.PRNGKey(777), a.n, cfg)
    h_ref_of, lam_ref_of = rref.h_lam(c_ref)
    th, tl = jax.vmap(h_ref_of)(xs), jax.vmap(lam_ref_of)(xs)
    net0 = state["net"]
    if a.scratch:
        net0 = model.init(jax.random.PRNGKey(cfg.seed + 5), xs[:64])
    flat0, unflat = jax.flatten_util.ravel_pytree(net0)

    def fit_loss(flat):
        f = model.apply(unflat(flat), xs)
        return jnp.mean(jnp.sum((f.h - th) ** 2, axis=(-2, -1))) + jnp.mean((f.lam - tl) ** 2)

    fv = jax.jit(fit_loss)
    fvg = jax.jit(jax.value_and_grad(fit_loss))
    minimize, where = _crunch_minimize(linesearch="armijo", impl="merged")
    if minimize is None:
        raise SystemExit(f"no merged Crunch: {where}")
    x, H, n = flat0, jnp.eye(flat0.size, dtype=flat0.dtype), flat0.size
    t0 = time.time()
    f_prev = float(fv(x))
    print(f"[fit] (b) regression: {n} parameters, {a.n} points, start {f_prev:.3e} "
          f"({'scratch' if a.scratch else 'warm start from the run'})", flush=True)
    hist = [f_prev]
    for b in range(math.ceil(a.iters / a.block)):
        r = minimize(fv, fvg, x, {"maxiter": a.block, "gtol": 0.0, "initial_H": H,
                                  "update_method": "ssbroyden2", "initial_scale": b == 0,
                                  "line_search": "armijo", "alphamin": 1e-15, "max_iter": 100,
                                  "scale_invariant": True})
        f_new = float(r.fun)
        H = r.hess_inv if f_new < f_prev else jnp.eye(n, dtype=x.dtype)
        x, f_prev = r.x, f_new
        hist.append(f_new)
        if (b + 1) % 4 == 0 or b == 0:
            print(f"[fit]   block {b + 1:3d}: regression loss {f_new:.3e}  ({time.time() - t0:.0f}s)",
                  flush=True)
    fitted = unflat(x)
    res["fit_history"] = hist
    res["fit_grade"] = grade(fitted)
    st_fit = dict(state)
    st_fit["net"] = fitted
    rf = pinn_losses(st_fit, model)
    res["fit_pinn_loss"] = [v for v, _ in rf]
    res["fit_pinn_parts"] = rf[0][1]
    g = res["fit_grade"]
    print(f"[fit]     fitted network vs reference: rms dh {g['rms_dh']:.3e} (max {g['max_dh']:.2e}), "
          f"rms dlambda {g['rms_dlam']:.3e} (max {g['max_dlam']:.2e})", flush=True)
    for bb in g["bins"]:
        print(f"[fit]       rho {bb['rho_lo']:5.1f}-{bb['rho_hi']:5.1f}  dlambda {bb['rms_dlam']:.2e}  "
              f"dh {bb['rms_dh']:.2e}", flush=True)
    print(f"[fit]     its PINN loss: " + " ".join(f"{v:.3e}" for v in res["fit_pinn_loss"])
          + "   parts: " + "  ".join(f"{k}={v:.2e}" for k, v in rf[0][1].items() if v), flush=True)
    tag = "_scratch" if a.scratch else ""
    with open(os.path.join(a.run, f"fit_reference{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
