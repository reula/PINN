"""Side-by-side comparison of several runs, with the numbers that decide between them.

    python -m stationary.compare runs/control_ord1b runs/control_ord2 runs/control_ord4_x64
    python -m stationary.compare runs/*/ --json logs/compare.json

Every quantity is recomputed here with the same code for every run (and in float64), so
the columns are comparable even when the runs were made with different code versions,
different Robin orders or different precision:

  * what the run was: precision (read off the stored parameters), size, Adam/L-BFGS steps,
    wall time, Robin orders;
  * lambda on the inner and outer spheres against the exact reference;
  * the boundary-condition residuals (rms) -- the honest measure of "did it converge";
  * the four PDE residuals (rms, raw units);
  * the CHART-INDEPENDENT content: (R0, k) read off the solution, the lambda-family
    residual and the l=1,2,3 amplitudes at rho_out (0 for a spherically symmetric run).
    These are the numbers to compare across runs, because the metric components depend on
    the harmonic chart (see README section 5 / HUB.md section 6).

The last block prints a one-line verdict per run: PASS/FAIL against the run's own exact
reference, using tolerances that can be given on the command line.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from dataclasses import fields

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from . import exact
from .evaluate import load_run
from .geometry import residuals_batch
from .invariants import family_params_from_solution
from .losses import inner_bc_terms, outer_bc_terms
from .model import point_fields
from .multipoles import lambda_multipoles
from .problem import Config, sample_sphere
from .train import exact_asset


def measure(run_dir: str, params_file: str = "params.pkl") -> dict:
    """Everything we want to compare, for one run."""
    cfg, model, state = load_run(run_dir, params_file)
    pf = point_fields(model, state["net"])
    key = jax.random.PRNGKey(0)
    dts = {str(v.dtype) for v in jax.tree.leaves(state["net"]) if hasattr(v, "dtype")}
    rep_path = os.path.join(run_dir, "report.json")
    rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
    # A run made before a config field existed has that field's CURRENT default applied
    # when we load it, so its recomputed residuals can differ from what its own log said.
    # Count them, so an old run cannot be read as if it had been made by today's code.
    stored = json.load(open(os.path.join(run_dir, "config.json")))
    missing = [f.name for f in fields(Config) if f.name not in stored]

    m = {"run": os.path.basename(os.path.normpath(run_dir)),
         "precision": "/".join(sorted(dts)),
         "missing_keys": len(missing),
         "missing_which": ",".join(missing[:4]) + ("..." if len(missing) > 4 else ""),
         "arch": f"{cfg.arch} {cfg.width}x{cfg.depth} f{cfg.fourier}",
         "orders": (cfg.robin_orders or {k: cfg.robin_order for k in ("h", "G", "lam")}),
         "outer_bc": cfg.outer_bc,
         "steps": rep.get("steps"), "lbfgs": rep.get("lbfgs_steps"),
         "wall_min": rep.get("wall_seconds", float("nan")) / 60.0,
         "final_loss": rep.get("final_loss", float("nan")),
         "lam0": cfg.lam0, "rho_in": cfg.rho_in, "rho_out": cfg.rho_out,
         "inner_radius": cfg.inner_radius,
         "k_implied": exact.k_from_lambda0(cfg.R0, cfg.lam0, r_areal=cfg.inner_radius)}

    # ---------------------------------------------------------------- the two spheres
    xi = sample_sphere(key, 512, cfg.rho_in)
    xo = sample_sphere(key, 512, cfg.rho_out)
    m["lam_in"] = float(jnp.mean(jax.vmap(lambda x: pf(x).lam)(xi)))
    m["lam_out"] = float(jnp.mean(jax.vmap(lambda x: pf(x).lam)(xo)))
    inn = inner_bc_terms(pf, xi, cfg)
    m["bc_in_lam"] = math.sqrt(float(inn["lam"]))
    m["bc_in_h_tan"] = math.sqrt(float(inn["h_tan"]))
    # areal radius of the inner sphere, from the induced metric (chart independent)
    hs = jax.vmap(lambda x: pf(x).h)(xi)
    nn = xi / jnp.linalg.norm(xi, axis=-1, keepdims=True)
    hrr = jnp.einsum("nij,ni,nj->n", hs, nn, nn)
    m["ra_in"] = cfg.rho_in * math.sqrt(float(jnp.mean((jnp.trace(hs, axis1=-2, axis2=-1)
                                                        - hrr) / 2.0)))
    out = outer_bc_terms(pf, xo, cfg, exact_asset(cfg),
                         lam_inf=cfg.lam_inf if cfg.lam_inf is not None else cfg.lam_inf_init)
    for k, v in out.items():
        m[f"bc_out_{k}"] = math.sqrt(float(v))

    # ------------------------------------------------------------------ PDE residuals
    res = residuals_batch(pf, sample_sphere(key, 2048, cfg.rho_out))
    for k, v in res.items():
        m[f"pde_{k}"] = math.sqrt(float(jnp.mean(v ** 2)))

    # ---------------------------------------------------- chart-independent content
    inv = family_params_from_solution(pf, cfg)
    m["fit_R0"] = inv["R0"]
    m["fit_k"] = inv["k"]
    m["fit_dlam"] = inv["max_resid_lambda"]
    m["fit_n_R"] = inv["n_R_points"]
    coef, power, _ = lambda_multipoles(pf, cfg.rho_out, lmax=3)
    m["amp_l1"] = math.sqrt(float(power[1]))
    m["amp_l2"] = math.sqrt(float(power[2]))
    m["amp_l3"] = math.sqrt(float(power[3]))
    m["mean_out"] = float(coef[(0, 0)]) / math.sqrt(4.0 * math.pi)

    # ------------------------------------------------------- against its own exact data
    ref = None
    if cfg.outer_bc == "dirichlet_exact":
        ref = exact_asset(cfg)
    elif getattr(cfg, "ref_asymptotic", None) is not None:
        ref, _ = exact.reference_fields_asymptotic(cfg.R0, cfg.ref_asymptotic, cfg.rho_in,
                                                   r_areal=cfg.inner_radius)
    elif cfg.ref_solution:
        ref = exact_asset(cfg)
    if ref is not None:
        m["ref_lam_in"] = float(ref(cfg.rho_in * jnp.array([1.0, 0.0, 0.0])).lam)
        m["ref_lam_out"] = float(ref(cfg.rho_out * jnp.array([1.0, 0.0, 0.0])).lam)
        d = jax.vmap(lambda x: jnp.abs(pf(x).lam - ref(x).lam))(xo)
        m["dlam_out"] = float(jnp.max(d))
        chk = inner_bc_terms(ref, xi, cfg)
        m["ref_bc_in"] = max(math.sqrt(float(v)) for v in chk.values())
    return m


# ------------------------------------------------------------------ presentation
ROWS = [
    ("what it is", None),
    ("run", "run"), ("precision", "precision"), ("arch", "arch"),
    ("config fields it predates",
     lambda m: f"{m['missing_keys']}" + (f" ({m['missing_which']})" if m["missing_keys"] else "")),
    ("outer BC", "outer_bc"), ("robin orders", "orders"),
    ("steps (Adam+LBFGS)", lambda m: f"{m['steps']}+{m['lbfgs']}"),
    ("wall (min)", lambda m: f"{m['wall_min']:.1f}"),
    ("final loss", lambda m: f"{m['final_loss']:.3e}"),
    ("lambda on the spheres", None),
    ("lambda_0 (imposed)", lambda m: f"{m['lam0']:.7f}"),
    ("lambda(rho_in) network", lambda m: f"{m['lam_in']:.7f}"),
    ("lambda(rho_out) network", lambda m: f"{m['lam_out']:.7f}"),
    ("lambda(rho_out) exact", lambda m: f"{m.get('ref_lam_out', float('nan')):.7f}"),
    ("difference at rho_out", lambda m: f"{m.get('dlam_out', float('nan')):.2e}"),
    ("k implied by the data", lambda m: f"{m['k_implied']:.6f}"),
    ("inner sphere areal radius", lambda m: f"{m['ra_in']:.7f}"),
    ("boundary conditions (rms)", None),
    ("inner: lambda", lambda m: f"{m['bc_in_lam']:.2e}"),
    ("inner: h_tan", lambda m: f"{m['bc_in_h_tan']:.2e}"),
    ("outer: h", lambda m: f"{m.get('bc_out_h', float('nan')):.2e}"),
    ("outer: lambda", lambda m: f"{m.get('bc_out_lam', float('nan')):.2e}"),
    ("PDE residuals (rms, raw)", None),
    ("compat", lambda m: f"{m['pde_compat']:.2e}"),
    ("gauge", lambda m: f"{m['pde_gauge']:.2e}"),
    ("lam_eq", lambda m: f"{m['pde_lam_eq']:.2e}"),
    ("ricci", lambda m: f"{m['pde_ricci']:.2e}"),
    ("chart-independent content", None),
    ("(R0, k) read off", lambda m: f"({m['fit_R0']:.5f}, {m['fit_k']:.6f})"),
    ("max |lambda - lambda_fam|", lambda m: f"{m['fit_dlam']:.2e}"),
    ("l=1,2,3 amplitudes at rho_out",
     lambda m: f"{m['amp_l1']:.1e} {m['amp_l2']:.1e} {m['amp_l3']:.1e}"),
]


def table(ms: list[dict]):
    names = [m["run"] for m in ms]
    w = max(24, max(len(n) for n in names) + 1)
    print(f"{'quantity':32s}" + "".join(f"{n:>{w}s}" for n in names))
    print("-" * (32 + w * len(names)))
    for label, get in ROWS:
        if get is None:
            print(f"{label}")
            continue
        if isinstance(get, str):                      # a plain field name
            key = get
            get = lambda m, key=key: str(m[key])
        print(f"{label:32s}" + "".join(f"{get(m):>{w}s}" for m in ms))


def verdicts(ms: list[dict], lam_tol: float, bc_tol: float):
    print()
    print(f"verdict: |lambda(rho_out) - exact| < {lam_tol:g}  and  outer BC rms < {bc_tol:g}")
    for m in ms:
        d = m.get("dlam_out", float("nan"))
        b = max(m.get("bc_out_h", float("nan")), m.get("bc_out_lam", float("nan")))
        ok = (d == d and b == b and d < lam_tol and b < bc_tol)
        why = "" if ok else f"   (dlam={d:.2e}, bc={b:.2e})"
        print(f"  {m['run']:28s} {'PASS' if ok else 'FAIL'}{why}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+", help="run directories (globs are expanded by the shell)")
    p.add_argument("--params-file", default="params.pkl")
    p.add_argument("--json", default=None, help="also write the raw numbers here")
    p.add_argument("--lam-tol", type=float, default=1e-4,
                   help="tolerance on |lambda(rho_out) - exact| for PASS")
    p.add_argument("--bc-tol", type=float, default=1e-4,
                   help="tolerance on the outer BC rms for PASS")
    a = p.parse_args()

    dirs = []
    for r in a.runs:
        dirs.extend(sorted(glob.glob(r)) or [r])
    ms = []
    for d in dirs:
        if not os.path.exists(os.path.join(d, "config.json")):
            print(f"skip {d}: no config.json", file=sys.stderr)
            continue
        try:
            ms.append(measure(d, a.params_file))
        except Exception as exc:                      # one bad run must not lose the table
            print(f"skip {d}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if not ms:
        p.error("nothing to compare")
    table(ms)
    verdicts(ms, a.lam_tol, a.bc_tol)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(ms, fh, indent=2, default=float)
        print(f"\nraw numbers -> {a.json}")


if __name__ == "__main__":
    main()
