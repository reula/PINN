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
from .invariants import family_params_from_solution
from .losses import (inner_bc_terms, outer_bc_terms, outer_pin_terms,
                     reference_consistency)
from .model import point_fields
from .multipoles import lambda_multipoles, multipole_radial_profile
from .problem import lam_inner_bc, sample_shell, sample_sphere

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

    # ------------------------------------------------- what the optimiser really was
    # A run made before `qn_method` existed must not be relabelled by the field's default:
    # check the stored config (and report.json) for the key itself, once, and use it in both
    # the RUN/CODE line and the CONFIG block.
    raw_cfg = {}
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        raw_cfg = json.load(open(cfg_path))
    qn_method = raw_cfg.get("qn_method")
    qn_known = qn_method is not None
    if qn_method is None:
        qn_method = "lbfgs"
    qn_label = "L-BFGS" if qn_method == "lbfgs" else "SSBroyden"

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
    # how the run was actually made, read off the saved parameters (not from any flag)
    dts = {str(v.dtype) for v in jax.tree.leaves(state["net"]) if hasattr(v, "dtype")}
    print(f"precision      : {', '.join(sorted(dts))}"
          + ("   <- trained in float64 (x64): ~1e-16 round-off headroom"
             if "float64" in dts else
             "   <- trained in float32 (the default): round-off floor ~1e-7"))
    if rep:
        # the quasi-Newton phase is SSBroyden unless it declined (no Crunch checkout, or the
        # dense inverse Hessian too large) and fell back to optax.lbfgs.  Runs made before
        # those fields existed carry neither, so fall back to config.json's qn_method.
        qn_used = rep.get("qn_stopped_at")
        qn_txt = qn_label
        if not qn_known:
            qn_txt += " (this run predates --qn-method)"
        elif qn_used is None:
            qn_txt += f" (cap {rep.get('lbfgs_steps')})"
        else:
            qn_txt += f" ({qn_used} of a {rep.get('lbfgs_steps')} cap)"
        adam = rep.get("adam_stopped_at", rep.get("steps"))
        print(f"steps          : {adam} Adam + {qn_txt}"
              f"   wall {rep.get('wall_seconds', 0) / 60:.1f} min")
        print(f"final loss     : {rep.get('final_loss', float('nan')):.4e}")

    # ------------------------------------------------------------------- config
    _section("CONFIG")
    print(f"arch           : {cfg.arch}   width {cfg.width} x depth {cfg.depth}, fourier {cfg.fourier}")
    print(f"domain         : rho in [{cfg.rho_in:g}, {cfg.rho_out:g}]   "
          f"inner sphere areal radius {cfg.inner_radius:g}   radial sampling {cfg.radial}")
    print(f"exact solution : R0 = {cfg.R0:g}   ref_solution {cfg.ref_solution}   "
          f"ref_asymptotic {getattr(cfg, 'ref_asymptotic', None)}   "
          f"robin_source {cfg.robin_source}")
    print(f"inner data     : lambda_0 = {cfg.lam0:g}"
          f"{' (derived from k=1)' if getattr(cfg, 'lam0_auto', False) else ''}"
          f"   ->  lambda -> k = {exact.k_from_lambda0(cfg.R0, cfg.lam0, r_areal=cfg.inner_radius):.6f}"
          f"   S1 = {cfg.lam_bc_S1:g}   S2 = {cfg.lam_bc_S2:g}"
          f"   (h_rr: {cfg.inner_h_rr if cfg.inner_h_rr is not None else 'free'})")
    lam_inf = cfg.lam_inf if cfg.lam_inf is not None else cfg.lam_inf_init
    orders = cfg.robin_orders or {k: cfg.robin_order for k in ("h", "G", "lam")}
    print(f"outer BC       : {cfg.outer_bc}, lambda_inf = {lam_inf:g}, Robin orders {orders},"
          f" Gamma condition included: {cfg.robin_include_G}")
    _pins = ", ".join(nm for nm, on in (("lam mean", getattr(cfg, "pin_lam", False)),
                                        ("h_tan", getattr(cfg, "pin_h_tan", False)),
                                        ("h_rr", getattr(cfg, "pin_h_rr", False))) if on)
    if _pins:
        print(f"far-field pins : {_pins} to the reference's values at rho_out,"
              f" w_pin = {getattr(cfg, 'w_pin', float('nan')):g}")
    print(f"loss weights   : w_inner = {cfg.w_inner:g}, w_outer = {cfg.w_outer:g}, "
          f"reweight_every = {cfg.reweight_every}")
    if getattr(cfg, "w_lam_eq_radial", 0.0):
        print(f"radial eq term : w_lam_eq_radial = {cfg.w_lam_eq_radial:g}"
              f"   (rho^3 d/drho of the lambda equation; the run's report.json carries its"
              f" final value as pde_lam_eq_radial)")
    print(f"sampling       : n_coll {cfg.n_coll}, n_bnd {cfg.n_bnd}, scale_ref {cfg.scale_ref}")
    if qn_known:
        print(f"optimiser      : Adam then {qn_method}"
              f"   blocks of {raw_cfg.get('qn_block', '-')}"
              f", plateau {raw_cfg.get('plateau_tol', '-')} over"
              f" {raw_cfg.get('plateau_patience', '-')} blocks (loss AND outer Robin)")
    else:
        print(f"optimiser      : Adam then optax.lbfgs"
              f"   (this run predates the quasi-Newton settings, so the plateau rule did"
              f" not apply)")

    # ------------------------------------------- the exact asset the BCs themselves use
    # Two objects that are easy to confuse: the reference that enters the outer boundary
    # condition (milestone-1 Dirichlet data, or the manufactured Robin source) and the
    # reference used only for comparison. This mirrors train.build() so the boundary
    # residuals below are the ones the loss actually saw. outer_bc_terms requires it.
    # One asset, built in ONE place.  This block used to re-derive the reference here, which
    # silently gave every new kind of reference (the Weyl two-black-hole solution, for one) a
    # report that compared the run against a DIFFERENT solution than the one its boundary
    # data came from.  train.exact_asset is the single definition; use it.
    from .train import exact_asset
    bc_exact = exact_asset(cfg)

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
    if cfg.inner_bc == "reference" and bc_exact is not None:
        # The imposed inner data is the reference's own lambda; comparing against the
        # round-sphere polynomial here would report the difference between two different
        # solutions (0.2 for the Weyl run) as if it were the network's error.
        lam_bc = jax.vmap(lambda x: bc_exact(x).lam)(xs)
        _imposed = "reference"
    else:
        lam_bc = jax.vmap(lambda x: lam_inner_bc(x, cfg))(xs)
        _imposed = "imposed"
    print(f"    theta     {_imposed:<8}     network        diff")
    for th, b, n in zip(THETAS, lam_bc, lam_net):
        print(f"{th:>9.2f} {float(b):>13.7f} {float(n):>14.7f} {float(n - b):>11.2e}")
    inn = inner_bc_terms(pf, sample_sphere(key, 512, cfg.rho_in), cfg,
                         exact_fields=bc_exact)
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
    # In reference mode nothing imposes an areal radius -- the inner data is whatever the
    # reference carries -- so quoting cfg.inner_radius next to the measured value would read
    # as a mismatch (and for the Weyl run it is a factor 3.7 out, the rod geometry).
    _imp = ("reference mode: the inner data is the reference's own, no areal radius is "
            "imposed" if cfg.inner_bc == "reference" else f"imposed {cfg.inner_radius:g}")
    print(f"inner sphere areal radius: {math.sqrt(r2):.8f}"
          f"   (r^2 = {r2:.6f}, {_imp})"
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
    # The far-field pins, if this run had them.  They are the only boundary terms that look
    # at the VALUES, and a run whose Robin residual is at its floor while lambda(rho_out) is
    # far from the reference is exactly the failure they exist to prevent, so print both
    # the reference's value and the difference the run achieved.
    pin = outer_pin_terms(pf, sample_sphere(key, 512, cfg.rho_out), cfg, bc_exact)
    if pin:
        x0 = cfg.rho_out * jnp.array([0.0, 0.0, 1.0])
        e0 = bc_exact(x0)
        n0 = x0 / jnp.linalg.norm(x0)
        h_rr0 = n0 @ e0.h @ n0
        g20 = (jnp.trace(e0.h) - h_rr0) / 2.0
        ref_vals = {"lam": float(e0.lam), "h_tan": float(g20), "h_rr": float(h_rr0)}
        print("far-field VALUES (rms of candidate - reference at rho_out): "
              + "  ".join(f"{k}={jnp.sqrt(v):.2e} (ref {ref_vals[k]:.7f})"
                          for k, v in pin.items()))
        print(f"    weighted by w_pin = {getattr(cfg, 'w_pin', float('nan')):g};"
              f" lam is pinned through its spherical MEAN, so an l >= 1 tail at rho_out"
              f" is not penalised (see the multipoles below).")

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
    if cfg.gauge_source != "none":
        # "gauge" below is the condition the run IMPOSED.  The harmonic residual is a
        # property of the chart (large for a chart adapted to a two-black-hole solution,
        # which is why the source was imposed at all) and would be read as an error here.
        from .losses import gauge_source_of
        print("    (gauge is the imposed inhomogeneous condition, NOT Gamma = 0)")
        res = residuals_batch(pf, sample_sphere(key, 2048, cfg.rho_out),
                              gauge_source_of(cfg, pf))
    else:
        res = residuals_batch(pf, sample_sphere(key, 2048, cfg.rho_out))
    for k, v in res.items():
        a_ = jnp.abs(v)
        print(f"    {k:10s} rms {float(jnp.sqrt(jnp.mean(v ** 2))):.3e}   max {float(jnp.max(a_)):.3e}")

    # ---------------------------------------------------------------- reference
    ref = bc_exact
    if ref is not None:
        _section("VS EXACT REFERENCE")
        print(f"    reference lambda(rho_in) = {float(ref(cfg.rho_in * jnp.array([1.0, 0, 0])).lam):.7f}"
              f"   (run imposed {cfg.lam0:g})")
        # Over the WHOLE shell, not just the outer sphere: a run can match at rho_out and
        # still be far off near the inner boundary, which is exactly what a wrong inner
        # areal radius does (the discrepancy decays like 1/rho^2).
        xr = sample_shell(jax.random.PRNGKey(1), 4096, cfg)

        def diff(x):
            f, e = pf(x), ref(x)
            return (jnp.max(jnp.abs(f.h - e.h)), jnp.max(jnp.abs(f.G - e.G)),
                    jnp.abs(f.lam - e.lam))

        dh, dG, dl = jax.vmap(diff)(xr)
        print(f"    over the shell   max |dh| = {float(jnp.max(dh)):.3e}"
              f"   max |dGamma| = {float(jnp.max(dG)):.3e}"
              f"   max |dlambda| = {float(jnp.max(dl)):.3e}")
        xo2 = sample_sphere(jax.random.PRNGKey(2), 2048, cfg.rho_out)
        dho, _, dlo = jax.vmap(diff)(xo2)
        print(f"    at rho_out       max |dh| = {float(jnp.max(dho)):.3e}"
              f"   max |dlambda| = {float(jnp.max(dlo)):.3e}")
        # Is the reference itself compatible with the inner data?  If not, the run had no
        # consistent solution to find; say so instead of leaving the reader to wonder.
        chk = reference_consistency(ref, cfg, exact_fields=ref)
        worst = max(chk.values())
        print("    reference vs the imposed inner data: "
              + "  ".join(f"{k}={v:.2e}" for k, v in chk.items())
              + ("   <- CONSISTENT" if worst < 1e-10 else
                 "   <- INCONSISTENT: no metric can satisfy both sets of data"))
    else:
        _section("VS EXACT REFERENCE")
        print("    none configured for this run (no --ref-solution / --dirichlet-exact)")

    # ------------------------------------------------ chart-independent comparison
    # The harmonic chart is not unique: the residual family F = c1 rho + c2 F2 of harmonic
    # diffeomorphisms changes h_ij without changing the geometry, and what it changes is
    # precisely h_rr on the inner sphere (h_rr = 1/F'^2 there).  Comparing h componentwise
    # against a reference written in another chart therefore overstates the error -- in
    # runs/control_ord1 the shell-wide max|dh| is 2.3e-2 while every geometric quantity
    # agrees much better.  These are the numbers that mean something.
    _section("CHART-INDEPENDENT (geometric invariants)")
    ang = float(sum(jnp.sqrt(power[l]) for l in (1, 2, 3))) if power else 0.0
    if ang > 1e-6:
        print(f"    CAUTION: lambda has angular content (l=1,2,3 amplitudes sum to {ang:.2e}):")
        print(f"             this solution is NOT spherically symmetric, and the relations")
        print(f"             below assume it is.  Indicative only (meaningless for the dipole).")
    try:
        inv = family_params_from_solution(pf, cfg)
        k_target = exact.k_from_lambda0(cfg.R0, cfg.lam0, r_areal=cfg.inner_radius)
        print(f"    family read off the solution: R0 = {inv['R0']:.6f}   k = {inv['k']:.6f}"
              f"     (the run's data imply R0 = {cfg.R0:.6f}, k = {k_target:.6f})")
        print(f"    max |R - 2 R0^2/r_a^4| = {inv['max_resid_R']:.3e}"
              f"   max |lambda - lambda_family| = {inv['max_resid_lambda']:.3e}")
        print("         r_a       lambda             R     2 R0^2/r_a^4   lambda_family")
        n = len(inv["ra"])
        for i in range(0, n, max(1, n // 7)):
            ra = float(inv["ra"][i])
            g = (math.sqrt(ra**2 + inv["R0"]**2) - inv["R0"]) / \
                (math.sqrt(ra**2 + inv["R0"]**2) + inv["R0"])
            print(f"    {ra:10.4f} {inv['lam'][i]:12.7f} {inv['R'][i]:12.4e}"
                  f" {2 * inv['R0']**2 / ra**4:12.4e}   {inv['k'] * g:12.7f}")
    except Exception as exc:                                    # never lose the report
        print(f"    skipped: {exc}")

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
