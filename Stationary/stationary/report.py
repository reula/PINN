"""One-screen text report of a run, meant to be pasted into a discussion.

    python -m stationary.report --outdir runs/<name>
    python -m stationary.report --outdir runs/<name> --params-file ckpt.pkl
    python -m stationary.report --outdir runs/<name> --out runs/<name>/report.txt

Prints, in order: which code produced the run, how it was configured, the loss and its
groups, the inner and outer boundary data (imposed vs achieved), lambda against rho, the
multipole content at the outer sphere, the PDE residuals, the comparison with the exact
reference when the run has one, and -- if the run's log is found -- the reweighting
history, which is where a silently de-weighted boundary condition shows up.

Written into the run directory as report.txt by postprocess.sh (which run_hub.sh calls at
the end of every run), so a finished run always has one. No plotting, so it works
headless: importing matplotlib is deliberately avoided.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from . import exact
from .diagnostics import inner_boundary_report
from .evaluate import load_run
from .geometry import residuals_batch
from .losses import inner_bc_terms, outer_bc_terms
from .model import point_fields
from .multipoles import lambda_multipoles, multipole_radial_profile
from .problem import lam_inner_bc, sample_sphere

THETAS = (0.0, 0.7, 1.5707963)


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              timeout=10).stdout.strip()
    except Exception:
        return ""


def _line(c="-"):
    print(c * 78)


class _Tee:
    """Print to the terminal and to a file at the same time (for --out)."""

    def __init__(self, path):
        self.fh = open(path, "w")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.fh.write(s)
        return len(s)

    def flush(self):
        self.stdout.flush()
        self.fh.flush()


def _section(title):
    print()
    _line()
    print(title)
    _line()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--params-file", default="params.pkl")
    p.add_argument("--log", default=None, help="run log (default <repo>/logs/<name>.log)")
    p.add_argument("--out", default=None,
                   help="also write the report to this file (default: none)")
    a = p.parse_args()
    run_dir = a.outdir or a.run_dir
    if run_dir is None:
        p.error("give the run directory, as `--outdir RUN` or as the first argument")
    if a.out:
        sys.stdout = _Tee(a.out)

    cfg, model, state = load_run(run_dir, a.params_file)
    pf = point_fields(model, state["net"])
    key = jax.random.PRNGKey(0)

    # ---------------------------------------------------------------- provenance
    _section("RUN / CODE")
    report_path = os.path.join(run_dir, "report.json")
    rep = json.load(open(report_path)) if os.path.exists(report_path) else {}
    head = _git("rev-parse", "--short", "HEAD")
    dirty = _git("status", "--porcelain")
    train_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    with open(train_py) as fh:
        fix = "pde_keys" in fh.read()
    print(f"run dir        : {run_dir}")
    print(f"params file    : {a.params_file}")
    print(f"git HEAD       : {head or '?'}{'  (+%d uncommitted files)' % len(dirty.splitlines()) if dirty else '  (clean)'}")
    print(f"BC-weight fix  : {'PRESENT (reweighting touches interior groups only)' if fix else 'ABSENT -> boundary weights can be driven to zero'}")
    if rep:
        print(f"steps          : {rep.get('steps')} Adam + {rep.get('lbfgs_steps')} L-BFGS"
              f"   wall {rep.get('wall_seconds', 0) / 60:.1f} min")
        print(f"final loss     : {rep.get('final_loss', float('nan')):.4e}")

    # ------------------------------------------------------------------- config
    _section("CONFIG")
    print(f"arch           : {cfg.arch}   width {cfg.width} x depth {cfg.depth}, fourier {cfg.fourier}")
    print(f"domain         : rho in [{cfg.rho_in:g}, {cfg.rho_out:g}]   "
          f"inner sphere areal radius {cfg.inner_radius:g}   radial sampling {cfg.radial}")
    print(f"inner data     : lambda_0 = {cfg.lam0:g}   S1 = {cfg.lam_bc_S1:g}   S2 = {cfg.lam_bc_S2:g}"
          f"   (h_rr constrained: {cfg.inner_h_rr})")
    lam_inf = cfg.lam_inf if cfg.lam_inf is not None else cfg.lam_inf_init
    orders = cfg.robin_orders or {k: cfg.robin_order for k in ("h", "G", "lam")}
    print(f"outer BC       : {cfg.outer_bc}, lambda_inf = {lam_inf:g}, Robin orders {orders},"
          f" Gamma condition included: {cfg.robin_include_G}")
    print(f"loss weights   : w_inner = {cfg.w_inner:g}, w_outer = {cfg.w_outer:g}, "
          f"reweight_every = {cfg.reweight_every}")
    print(f"sampling       : n_coll {cfg.n_coll}, n_bnd {cfg.n_bnd}, scale_ref {cfg.scale_ref}")

    # ------------------------------------------- the exact asset the BCs themselves use
    # Two objects that are easy to confuse: the reference that enters the outer boundary
    # condition (milestone-1 Dirichlet data, or the manufactured Robin source) and the
    # reference used only for comparison. This mirrors train.build() so the boundary
    # residuals below are the ones the loss actually saw. outer_bc_terms requires it.
    bc_exact = None
    if cfg.outer_bc == "dirichlet_exact":
        bc_exact = exact.exact_fields(cfg.R0, exact.k_from_lambda0(cfg.R0, cfg.lam0))
    elif cfg.ref_solution or cfg.robin_source:
        if getattr(cfg, "ref_asymptotic", None) is not None:
            bc_exact, _ = exact.reference_fields_asymptotic(cfg.R0, cfg.ref_asymptotic,
                                                            cfg.rho_in,
                                                            r_areal=cfg.inner_radius)
        else:
            bc_exact, _ = exact.reference_fields(cfg.R0, cfg.lam0, cfg.rho_in,
                                                 cfg.inner_h_rr or 1.0,
                                                 r_areal=cfg.inner_radius)

    # -------------------------------------------------------------------- loss
    hist_path = os.path.join(run_dir, "history.json")
    if os.path.exists(hist_path):
        hist = json.load(open(hist_path))
        _section("LOSS TRAJECTORY (first, middle, last logged)")
        keys = [k for k in ("loss", "pde_compat", "pde_ricci", "pde_gauge", "pde_lam_eq",
                            "inner_lam", "inner_h_tan", "outer_h", "outer_lam") if k in hist[0]]
        print("    step  " + "".join(f"{k:>13}" for k in keys))
        for i in (0, len(hist) // 2, len(hist) - 1):
            row = hist[i]
            print(f"{row['step']:>8}  " + "".join(
                f"{row.get(k, float('nan')):>13.3e}" for k in keys))

    # ------------------------------------------------------------ inner boundary
    _section("INNER BOUNDARY  (imposed vs network)")
    xs = jnp.stack([cfg.rho_in * jnp.sin(jnp.array(THETAS)),
                    jnp.zeros(len(THETAS)),
                    cfg.rho_in * jnp.cos(jnp.array(THETAS))], axis=-1)
    lam_net = jax.vmap(lambda x: pf(x).lam)(xs)
    lam_bc = jax.vmap(lambda x: lam_inner_bc(x, cfg))(xs)
    print(f"    theta     imposed        network        diff")
    for th, b, n in zip(THETAS, lam_bc, lam_net):
        print(f"{th:>9.2f} {float(b):>13.7f} {float(n):>14.7f} {float(n - b):>11.2e}")
    inn = inner_bc_terms(pf, sample_sphere(key, 512, cfg.rho_in), cfg)
    # areal radius of the inner sphere, from the same tested route as evaluate.py, plus the
    # radial-radial component of h there, which is the one piece of the metric the BCs do
    # not fix (--inner-h-rr aside).
    xs_in = sample_sphere(key, 512, cfg.rho_in)
    n_in = xs_in / jnp.linalg.norm(xs_in, axis=-1, keepdims=True)
    h_in = jax.vmap(lambda x: pf(x).h)(xs_in)
    # NOTE the operand order: jnp.einsum in jax 0.11 rejects "ni,nij,nj->n" (opt_einsum
    # reorders the operands and then miscounts the indices) while the same contraction
    # written in the order the operands are passed works.  Keep h first.
    h_rr_in = jnp.einsum("nij,ni,nj->n", h_in, n_in, n_in)
    geom = inner_boundary_report(pf, cfg)          # reports the areal radius SQUARED
    r2 = geom["areal_radius2_mean"]
    print(f"inner sphere areal radius: {math.sqrt(r2):.8f}"
          f"   (r^2 = {r2:.6f}, imposed {cfg.inner_radius:g})"
          f"   h_rr there {float(jnp.mean(h_rr_in)):.6f}")
    print("inner BC residuals (rms): " + "  ".join(f"{k}={jnp.sqrt(v):.2e}" for k, v in inn.items()))

    # ------------------------------------------------------------ outer boundary
    _section("OUTER BOUNDARY")
    xo = jnp.stack([cfg.rho_out * jnp.sin(jnp.array(THETAS)),
                    jnp.zeros(len(THETAS)),
                    cfg.rho_out * jnp.cos(jnp.array(THETAS))], axis=-1)
    lam_out = jax.vmap(lambda x: pf(x).lam)(xo)
    print(f"    theta      lambda(rho_out)     distance from lambda_inf = {lam_inf:g}")
    for th, v in zip(THETAS, lam_out):
        print(f"{th:>9.2f} {float(v):>18.7f} {float(v - lam_inf):>20.3e}")
    out = outer_bc_terms(pf, sample_sphere(key, 512, cfg.rho_out), cfg, bc_exact, lam_inf=lam_inf)
    print(f"outer BC residuals ({cfg.outer_bc}, rms): "
          + "  ".join(f"{k}={jnp.sqrt(v):.2e}" for k, v in out.items()))

    # ------------------------------------------------------------ lambda vs rho
    # The shape of lambda(rho), which is what says whether the run sits on the
    # non-trivial branch: it must leave lambda_0 at the inner sphere and flatten onto
    # lambda_inf. Eight intervals, geometric in rho.
    _section("LAMBDA vs RHO")
    print("       rho      lam(th=0)   lam(th=90)")
    for i in range(9):
        rho = cfg.rho_in * (cfg.rho_out / cfg.rho_in) ** (i / 8.0)
        vals = [float(pf(rho * jnp.array([math.sin(math.radians(t)), 0.0,
                                          math.cos(math.radians(t))])).lam)
                for t in (0.0, 90.0)]
        print(f"{rho:>10.4f} {vals[0]:>14.7f} {vals[1]:>12.7f}")

    # ---------------------------------------------------------------- multipoles
    _section(f"MULTIPOLES OF lambda AT rho = {cfg.rho_out:g}")
    coef, power, _ = lambda_multipoles(pf, cfg.rho_out, lmax=3)
    # coefficients are in the real-SH basis with Y_00 = 1/sqrt(4 pi), so a_00 is the mean
    # of lambda only up to that factor; print the mean itself, it is the readable number.
    mean_out = float(coef[(0, 0)]) / math.sqrt(4.0 * math.pi)
    print(f"    l=0:  mean lambda {mean_out:.7f}   = lambda_inf {lam_inf:g} "
          f"{mean_out - lam_inf:+.3e}")
    for l in range(1, 4):
        print(f"    l={l}:  amplitude {float(jnp.sqrt(power[l])):.4e}")
    # sample the decay inside the domain (not beyond rho_out, where the net extrapolates)
    try:
        rhos = [float(cfg.rho_in * (cfg.rho_out / cfg.rho_in) ** (i / 6.0)) for i in range(1, 7)]
        prof = multipole_radial_profile(pf, rhos, lmax=3, lam_inf=lam_inf)
        print(f"    decay with rho (fitted power vs expected -(l+1)), fitted over "
              f"rho = {rhos[0]:.3g} .. {rhos[-1]:.3g}:")
        for l in range(4):
            f = prof[l]["fitted_power"]
            if f != f:
                print(f"      l={l}: no amplitude above the noise floor at any of these radii"
                      f"   (expected {prof[l]['expected_power']})")
            else:
                print(f"      l={l}: fitted {f: .3f}   expected {prof[l]['expected_power']}")
    except Exception as exc:
        print(f"    decay fit skipped: {exc}")

    # ---------------------------------------------------------------- residuals
    _section("PDE RESIDUALS (raw units)")
    res = residuals_batch(pf, sample_sphere(key, 2048, cfg.rho_out))
    for k, v in res.items():
        a_ = jnp.abs(v)
        print(f"    {k:10s} rms {float(jnp.sqrt(jnp.mean(v ** 2))):.3e}   max {float(jnp.max(a_)):.3e}")

    # ---------------------------------------------------------------- reference
    ref = None
    if getattr(cfg, "ref_asymptotic", None) is not None:
        ref, _ = exact.reference_fields_asymptotic(cfg.R0, cfg.ref_asymptotic, cfg.rho_in,
                                                   r_areal=cfg.inner_radius)
    else:
        ref = bc_exact          # dirichlet_exact, or the --ref-solution asset
    if ref is not None:
        _section("VS EXACT REFERENCE")
        print(f"    reference lambda(rho_in) = {float(ref(cfg.rho_in * jnp.array([1.0, 0, 0])).lam):.7f}"
              f"   (run imposed {cfg.lam0:g})")
        xr = sample_sphere(jax.random.PRNGKey(1), 2048, cfg.rho_out)

        def diff(x):
            f, e = pf(x), ref(x)
            return (jnp.max(jnp.abs(f.h - e.h)), jnp.max(jnp.abs(f.G - e.G)),
                    jnp.abs(f.lam - e.lam))

        dh, dG, dl = jax.vmap(diff)(xr)
        print(f"    max |dh| = {float(jnp.max(dh)):.3e}   max |dGamma| = {float(jnp.max(dG)):.3e}"
              f"   max |dlambda| = {float(jnp.max(dl)):.3e}")
    else:
        _section("VS EXACT REFERENCE")
        print("    none configured for this run (no --ref-solution / --dirichlet-exact)")

    # --------------------------------------------------------------------- log
    log = a.log or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "logs", os.path.basename(os.path.normpath(run_dir)) + ".log")
    _section("REWEIGHTING HISTORY (from the log, if found)")
    if os.path.exists(log):
        lines = [ln for ln in open(log) if ln.startswith("[reweight")]
        print(f"    log: {log}")
        if lines:
            for ln in lines[:2] + (["    ...\n"] if len(lines) > 3 else []) + lines[-2:]:
                print("    " + ln.rstrip())
            print("    -> outer/inner weights falling over time means boundary conditions")
            print("       were being de-weighted (the pre-fix behaviour).")
        else:
            print("    no [reweight] lines (reweighting off, or log rotated)")
    else:
        print(f"    log not found at {log} (pass --log PATH)")


if __name__ == "__main__":
    main()
