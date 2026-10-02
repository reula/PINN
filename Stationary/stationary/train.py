"""Training driver: Adam warm-up followed by an L-BFGS polish.

Usage:
    .venv/bin/python -m stationary.train --steps 2000 --outdir runs/m1_smoke

Long runs (a 20k-step Adam phase takes ~2 h) should write resumable checkpoints,
so that an interrupted session costs minutes rather than the whole run:

    python -m stationary.train --steps 20000 --ckpt-every 500 --outdir runs/m2
    python -m stationary.train --steps 20000 --ckpt-every 500 --outdir runs/m2 --resume auto

`--resume auto` looks for <outdir>/ckpt.pkl and continues the Adam phase from the
step recorded there. Pass the same `--steps` as the original run: the LR schedule
and the resampling cadence are functions of the step index, so changing them would
alter the trajectory instead of continuing it (a mismatch is reported).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import asdict

import jax
import jax.numpy as jnp
import optax

from . import diagnostics, exact
from .losses import (FIRST_ORDER_PDE_KEYS, GROUP_KEYS, default_weights, equation_keys,
                     group_terms, outer_pin_terms, pde_radial_terms, reference_consistency,
                     total_loss)
from .model import (AxisymHybridNet, FieldNet, HybridNet, SymFieldNet, SymHybridNet,
                    point_fields)
from .problem import Config, sample_shell, sample_sphere


def make_model(cfg: Config):
    cls = {"sym": SymFieldNet, "sym_hybrid": SymHybridNet, "hybrid": HybridNet,
           "axisym_hybrid": AxisymHybridNet, "mlp": FieldNet}.get(cfg.arch, FieldNet)
    kw = dict(width=cfg.width, depth=cfg.depth, fourier=cfg.fourier,
              rho_in=cfg.rho_in, rho_out=cfg.rho_out)
    if cfg.arch in ("sym", "sym_hybrid", "axisym_hybrid"):
        kw["decay"] = cfg.decay_feature
    return cls(**kw)


def exact_asset(cfg: Config, n_quad: int | None = None):
    """The exact solution this run uses for its data, or None.

    Three distinct roles, one object each: the outer boundary data of `dirichlet_exact`,
    the manufactured source of a Robin run (`robin_source`), and the reference kept for
    diagnostics (`ref_solution`).  This is the single place that builds it, so that
    training, the report and the comparison tool all see the same reference.

    `n_quad` overrides the Weyl quadrature node count and is for DIAGNOSTICS ONLY: `h_cart`
    runs the quadrature inside the metric, so a twice-differentiated metric costs n_quad times
    a cheap one, and the geometry block of `report.py` -- a few dozen scalars -- took minutes
    at the default 400.  The Gauss-Legendre rule converges exponentially on this integrand, so
    64 nodes is accurate to ~1e-12 away from the rods.  None (the default) means
    `cfg.weyl_n_quad`, so training and every boundary condition are untouched.
    """
    if getattr(cfg, "weyl", False):
        from .weyl import Rods, rotated_fields, rotation_matrix
        rot = (rotation_matrix(cfg.weyl_rotate_deg)
               if getattr(cfg, "weyl_rotate_deg", 0.0) else None)
        return rotated_fields(Rods.pair(cfg.weyl_half_length, cfg.weyl_half_length_b,
                                        cfg.weyl_half_gap),
                              cfg.weyl_n_quad if n_quad is None else int(n_quad), rot)
    if cfg.outer_bc == "dirichlet_exact":
        return exact.exact_fields(cfg.R0, exact.k_from_lambda0(cfg.R0, cfg.lam0))
    if cfg.ref_solution or cfg.robin_source:
        if getattr(cfg, "ref_asymptotic", None) is not None:
            fields, _ = exact.reference_fields_asymptotic(
                cfg.R0, cfg.ref_asymptotic, cfg.rho_in, r_areal=cfg.inner_radius)
        else:
            fields, _ = exact.reference_fields(cfg.R0, cfg.lam0, cfg.rho_in,
                                               cfg.inner_h_rr or 1.0,
                                               r_areal=cfg.inner_radius)
        return fields
    return None


def build(cfg: Config, init_from: str | None = None):
    model = make_model(cfg)
    key = jax.random.PRNGKey(cfg.seed)
    k1, k2 = jax.random.split(key)
    x0 = sample_shell(k1, min(cfg.n_coll, 64), cfg)
    if init_from is None:
        params = model.init(k2, x0)
    else:
        with open(init_from, "rb") as fh:
            saved = pickle.load(fh)
        params = saved["net"] if "net" in saved else saved
        print(f"[build] initialised network parameters from {init_from}")

    # exact solution (used for the milestone-1 outer BC and for diagnostics)
    exact_fields = exact_asset(cfg)

    state = {"net": params}
    if cfg.outer_bc == "robin" and cfg.lam_inf is None:
        # learnable asymptotic value (initialised away from lambda_0 on purpose)
        state["lam_inf"] = jnp.array(cfg.lam_inf_init)

    # ---------------------------------------------------------- sanity of the data
    # The exact reference is the outer boundary data (dirichlet_exact) or the source of
    # the manufactured Robin condition.  If it does not also satisfy the INNER data, the
    # two boundary conditions come from different solutions, no metric satisfies both,
    # and the run converges to a compromise instead of the exact solution.  Two seconds
    # here, or a wasted run later.
    # With the inner data fixed, the asymptotic value of the exact solution is k =
    # lambda_0 (rho_g+R0)/(rho_g-R0); the outer Robin condition drives lambda towards
    # lam_inf, so the two must agree or the exact solution is not a solution of the
    # problem being solved.
    k = exact.k_from_lambda0(cfg.R0, cfg.lam0, r_areal=cfg.inner_radius)
    if (cfg.outer_bc == "robin" and cfg.lam_inf is not None
            and abs(cfg.lam_inf - k) > 1e-6 * max(1.0, abs(k))):
        print(f"[build] WARNING: the inner data select the exact solution with "
              f"lambda -> k = {k:.6f} (lambda_0 = {cfg.lam0:.7g} on the areal-radius-"
              f"{cfg.inner_radius:g} sphere, R0 = {cfg.R0:g}), but the outer Robin "
              f"condition drives lambda to lam_inf = {cfg.lam_inf:g}.")
        print(f"[build]   the exact solution does not satisfy that outer condition.  Use "
              f"--lam-inf {k:.6f}, or --lam0 "
              f"{exact.lambda0_from_k(cfg.R0, cfg.lam_inf, cfg.inner_radius):.7f} to keep "
              f"lam_inf (and re-check the inner data).")
    if exact_fields is not None:
        chk = reference_consistency(exact_fields, cfg, exact_fields=exact_fields)
        # The reference enters the loss as boundary data only in these two cases; without
        # them it is a comparison asset and a mismatch is not a convergence problem.
        matters = cfg.outer_bc == "dirichlet_exact" or cfg.robin_source
        bad = {k: v for k, v in chk.items() if v > 1e-10}
        if bad:
            print(f"[build] {'WARNING: the exact reference violates the imposed INNER data' if matters else 'note: the comparison reference does not satisfy the imposed INNER data'}"
                  f": " + "  ".join(f"{k}={v:.3e}" for k, v in bad.items()))
            if matters:
                print("[build]   the inner data and the outer data come from different "
                      "solutions: no metric satisfies both, so this run cannot converge "
                      "to the exact one.")
            if "h_tan" in bad:
                print(f"[build]   h_tan: inner_radius = {cfg.inner_radius:g}, but the "
                      f"canonical-chart reference has areal radius "
                      f"{float(jnp.sqrt(max(cfg.rho_in**2 - cfg.R0**2, 0.0))):.6f} at "
                      f"rho = {cfg.rho_in:.6f}"
                      + ("  -> --inner-radius that value (or --rho-in "
                         f"{exact.rho_in(cfg.R0):.6f})"
                         if cfg.outer_bc == "dirichlet_exact" else
                         "  -> --ref-solution builds a reference for this radius"))
            if "h_rr" in bad:
                print(f"[build]   h_rr: you imposed {cfg.inner_h_rr:g}; the reference has "
                      f"rms {float(jnp.sqrt(chk['h_rr'])):.3e} there.  h_rr over-determines "
                      f"the radial gauge -- drop --inner-h-rr unless you are using "
                      f"--ref-solution, which builds the reference with it.")
            if "lam" in bad:
                if cfg.lam_bc_S1 or cfg.lam_bc_S2:
                    print("[build]   lam: expected -- a spherically symmetric reference "
                          "cannot carry S1/S2; it serves for comparison only.")
                else:
                    print(f"[build]   lam: you imposed lam0 = {cfg.lam0:g} at "
                          f"rho = {cfg.rho_in:g}; the reference has "
                          f"{float(exact_fields(cfg.rho_in * jnp.array([1.0, 0.0, 0.0])).lam):.6f}"
                          f"  -> check --lam0 / --R0 / --rho-in")
    if cfg.pin_lam_robin:
        # No reference involved: say what the condition IS and what it implies, so the log
        # records the pin the run was actually given.
        lam_inf = cfg.lam_inf if cfg.lam_inf is not None else cfg.lam_inf_init
        print(f"[pin] pinning the ORDER-1 ROBIN COMBINATION of the spherical mean of lambda "
              f"at rho_out = {cfg.rho_out:g}:")
        print(f"[pin]   rho d_rho <lambda> + (<lambda> - lambda_inf) = 0   with "
              f"lambda_inf = {lam_inf:g}   w_pin = {cfg.w_pin:g}")
        print(f"[pin]   the mean of the pointwise order-1 condition, so it needs NO reference; "
              f"its own residual on the exact")
        print(f"[pin]   solution is the order-1 floor (6.6e-05 in lambda at shell ratio 100, "
              f"~4e-06 at ratio 400).")
    if cfg.pin_h_robin:
        print(f"[pin] pinning the AVERAGED Robin condition for the METRIC at rho_out = "
              f"{cfg.rho_out:g}   w_pin = {cfg.w_pin:g}")
        b = cfg.h_robin_bases or {}
        b_rr, b_g2 = b.get("h_rr", 3.0), b.get("g2", 2.0)
        print(f"[pin]   rho d_rho <h_rr> + {b_rr:g} (<h_rr> - 1) = 0"
              f"        <h_rr> is the chart scale")
        print(f"[pin]   rho d_rho <g2>   + {b_g2:g} (<g2>   - 1) = 0"
              f"        <g2> is the outer sphere's areal radius")
        print(f"[pin]   bases in use: <h_rr> {b_rr:g}, <g2> {b_g2:g}"
              + ("" if cfg.h_robin_bases else
                 "   (DEFAULTS, measured on the spherical reference: rho^-3.000 and"
                 " rho^-1.998."))
        print(f"[pin]   That measurement is an ASSUMPTION about this problem, whose inner data"
              f" are angular, and its multipole expansion")
        print(f"[pin]   need not match the spherical one.  --h-robin-bases overrides it.")
        print(f"[pin]   NO reference needed.  The MEANS are pinned; the pointwise order-3 "
              f"condition is untouched.")
    if (cfg.pin_lam or cfg.pin_h_tan or cfg.pin_h_rr) and exact_fields is not None:
        # Say what is being pinned, and to what.  A value pin is only meaningful if the
        # number is the one the exact solution has there, so print it and let the report
        # show both it and what the run achieved (`outer_pin_terms` measures the difference).
        xo = cfg.rho_out * jnp.array([0.0, 0.0, 1.0])
        e = exact_fields(xo)
        n = xo / jnp.linalg.norm(xo)
        h_rr = n @ e.h @ n
        g2 = (jnp.trace(e.h) - h_rr) / 2.0
        which = ", ".join(nm for nm, on in (("lam (spherical mean)", cfg.pin_lam),
                                            ("h_tan", cfg.pin_h_tan),
                                            ("h_rr", cfg.pin_h_rr)) if on)
        print(f"[pin] far-field values at rho_out = {cfg.rho_out:g} from the reference: "
              f"lambda = {float(e.lam):.7f}   h_tan = {float(g2):.7f}   "
              f"h_rr = {float(h_rr):.7f}")
        print(f"[pin] pinning {which}   w_pin = {cfg.w_pin:g}   "
              f"(the true solution differs from these by ~|S2|(rho_in/rho_out)^3 = "
              f"{abs(cfg.lam_bc_S2) * (cfg.rho_in / cfg.rho_out) ** 3:.1e}, i.e. far below "
              f"the Robin floors)")
    return model, state, exact_fields


def make_batch(key, cfg: Config):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "coll": sample_shell(k1, cfg.n_coll, cfg),
        "inner": sample_sphere(k2, cfg.n_bnd, cfg.rho_in),
        "outer": sample_sphere(k3, cfg.n_bnd_outer or cfg.n_bnd, cfg.rho_out),
    }


CKPT_NAME = "ckpt.pkl"


def batch_for_step(cfg: Config, step: int):
    """The collocation batch in effect at 1-based Adam step `step`.

    Rebuilt from the seed instead of being stored, so that a resumed run sees
    exactly the same sampling sequence as an uninterrupted one.
    """
    last = 1
    if cfg.resample_every and step > 1:
        last = 1 + ((step - 1) // cfg.resample_every) * cfg.resample_every
    return make_batch(jax.random.PRNGKey(cfg.seed + last), cfg)


def save_checkpoint(path, state, opt_state, weights, history, step, cfg, extra=None):
    """Dump everything needed to continue from where this was called.

    `weights` is included because the gradient-norm reweighting is path
    dependent: it multiplies the previous weights, so it cannot be recomputed
    from the step index alone.

    `extra` carries what the phase-specific caller knows and this function cannot: the
    quasi-Newton phase records `phase="qn"` and its inverse Hessian, which is what lets
    `--resume auto` re-enter that phase instead of rewinding to the end of Adam.  Both
    phases use the same file and the same key, so there is exactly one resume path.
    """
    payload = {
        "state": jax.tree.map(jax.device_get, state),
        "opt_state": jax.tree.map(jax.device_get, opt_state) if opt_state is not None else None,
        "weights": jax.tree.map(jax.device_get, weights),
        "history": history,
        "step": step,
        "config": asdict(cfg),
    }
    if extra:
        payload.update(extra)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh)
    os.replace(tmp, path)   # atomic: a kill mid-write cannot corrupt the previous checkpoint
    return path


def write_params(outdir, state, name: str = "params.pkl"):
    """Write the parameter state itself (`params.pkl`, or `params_adam.pkl`), atomically.

    This is the bare tree that `--init-from` reads and that `postprocess.sh` falls back to
    when there is no ckpt.pkl -- so it is the one copy of the trained weights, and it goes
    through a temp file and a rename for the same reason save_checkpoint does: a kill during
    the write must not be able to destroy the previous one.
    """
    path = os.path.join(outdir, name)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(jax.tree.map(jax.device_get, state), fh)
    os.replace(tmp, path)
    return path


def print_config_summary(cfg: Config):
    """The effective physics of this run, first thing in the log.

    Every one of these values is a flag; if a flag did not reach the process (an empty
    shell array in `./run_hub.sh "${COMMON[@]}" ...` is the classic way), the run silently
    becomes the default milestone-1 configuration -- R0 = 1, lambda_0 = 1, rho_out = 20,
    dirichlet_exact -- which is a perfectly valid run of something you did not ask for.
    Printing them makes that visible in the first screen of the log.
    """
    orders = cfg.robin_orders or {k: cfg.robin_order for k in ("h", "G", "lam")}
    k = exact.k_from_lambda0(cfg.R0, cfg.lam0, r_areal=cfg.inner_radius)
    print("=" * 72)
    print("effective configuration (every value below is a flag; check them)")
    print(f"  model       {cfg.arch}   {cfg.width} x {cfg.depth}, fourier {cfg.fourier}"
          f"   (Gamma derived from h)")
    print(f"  exact data  R0 = {cfg.R0:g}   lambda_0 = {cfg.lam0:.7g}"
          f"   S1 = {cfg.lam_bc_S1:g}   S2 = {cfg.lam_bc_S2:g}"
          f"   ->  lambda -> k = {k:.7g}" + ("" if abs(k - 1.0) < 1e-9 else "   (k != 1!)"))
    print(f"  domain      rho in [{cfg.rho_in:g}, {cfg.rho_out:g}]"
          f"   inner sphere areal radius {cfg.inner_radius:g}"
          f"   h_rr {cfg.inner_h_rr if cfg.inner_h_rr is not None else 'free'}")
    if cfg.outer_bc == "robin":
        # `lam_inf` fixed vs learnable is a real difference -- an optimisable lambda_inf
        # lets the trivial (flat, lambda = const) branch re-select itself -- so the banner
        # must not call a fixed value "learnable".
        exps = {k: (int(v) if float(v).is_integer() else float(v))
                for k, v in (cfg.robin_exps or {}).items()}
        print(f"  outer BC    robin   lambda_inf = "
              f"{cfg.lam_inf if cfg.lam_inf is not None else cfg.lam_inf_init}"
              f" ({'fixed' if cfg.lam_inf is not None else 'learnable'})"
              f"   orders {orders}   base exponents {exps}"
              f"   Gamma condition {cfg.robin_include_G}"
              f"   source {cfg.robin_source}")
        pins = ", ".join(nm for nm, on in (
            ("lam mean (order-1 Robin combination, no reference needed)", cfg.pin_lam_robin),
            ("lam mean (value, from the reference)", cfg.pin_lam),
            ("averaged h Robin: <h_rr> base 3, <g2> base 2 (no reference needed)",
             cfg.pin_h_robin),
            ("h_tan", cfg.pin_h_tan), ("h_rr", cfg.pin_h_rr)) if on)
        if pins:
            print(f"  far-field   PINNED at rho_out: {pins}"
                  f"   w_pin {cfg.w_pin:g}   (values above, '[pin]' line)")
    else:
        print(f"  outer BC    {cfg.outer_bc} (h and lambda from the exact solution)")
    print(f"  weights     w_inner {cfg.w_inner:g}   w_outer {cfg.w_outer:g}"
          f"   reweight every {cfg.reweight_every}   pde ramp {cfg.pde_ramp_steps}"
          + (f"   w_lam_eq_radial {cfg.w_lam_eq_radial:g}"
             if getattr(cfg, 'w_lam_eq_radial', 0.0) else ""))
    print(f"  sampling    n_coll {cfg.n_coll}   n_bnd {cfg.n_bnd} (outer {cfg.n_bnd_outer or cfg.n_bnd})   radial {cfg.radial}"
          f"   decay feature {cfg.decay_feature}")
    prec = ("float64 (x64: ~2x slower, and NOT comparable with the float32 runs)"
            if jax.config.jax_enable_x64 else "float32 (default)")
    print(f"  precision   {prec}")
    print(f"  plan        {cfg.steps} Adam + {cfg.lbfgs_steps} "
          f"{'SSBroyden' if cfg.qn_method == 'ssbroyden' else 'L-BFGS'}   outdir {cfg.outdir}")
    print("=" * 72)


def print_reference_summary(cfg: Config, exact_fields):
    """The numbers the exact reference takes on the two spheres.

    This is the quickest way to see whether the run is on the solution you intended:
    the M2 control reads lambda(rho_in) = 0.333333, lambda(100) = 0.988519 and
    lambda -> k = 1, while the M1 default configuration reads lambda -> phi^2 = 2.618034.
    `k` is what lambda tends to at infinity, and for the exact family it is fixed by the
    inner data: k = lambda_0 (rho_g+R0)/(rho_g-R0) with rho_g = sqrt(inner_radius^2+R0^2).
    """
    lam_in = float(exact_fields(cfg.rho_in * jnp.array([1.0, 0.0, 0.0])).lam)
    lam_out = float(exact_fields(cfg.rho_out * jnp.array([1.0, 0.0, 0.0])).lam)
    k = exact.k_from_lambda0(cfg.R0, cfg.lam0, r_areal=cfg.inner_radius)
    print(f"[ref] exact reference: lambda({cfg.rho_in:g}) = {lam_in:.6f} "
          f"(imposed {cfg.lam0:g})   lambda({cfg.rho_out:g}) = {lam_out:.6f}   "
          f"lambda -> k = {k:.6f}")
    print(f"[ref]   every run uses the k = 1 branch: this one has k = {k:.6f}"
          + ("" if abs(k - 1.0) < 1e-9 else "  <-- NOT 1: --lam0 was given explicitly"))


# ------------------------------------------------------------------- SSBroyden
def _crunch_candidates(here: str):
    """Everywhere a Crunch checkout is looked for, in order of preference.

    Crunch -- the JAX fork of `jax.scipy.optimize` that adds the self-scaling Broyden, a
    translation of Optim.jl's `SSBroyden` -- is TRACKED in this repository as `Jax/Crunch`
    (it entered in 231e705), so `<repo>/Jax` comes first and an existing checkout behaves
    exactly as before.  A copy inside `Stationary/` itself, in the repo root, or ABOVE the
    repo is found too, and that is not cosmetic: the hub is synced with
    `rsync ... Stationary/ hub:.../Stationary/`, so a machine carrying only `Stationary/` has
    no `<repo>/Jax` at all -- with `<Stationary>/Crunch`, `<Stationary>/Jax/Crunch` or
    `../Crunch` the quasi-Newton phase still gets SSBroyden instead of quietly falling back
    to optax.lbfgs.

    `here` is `<repo>/Stationary`.  The returned paths are the ones to put on `sys.path`: the
    directories that may contain `Crunch/`, and with it the `line_search_backtracking` module
    the fork imports at its top level (`bfgs_backtracking.py` does `from
    line_search_backtracking import backtracking`, so a root without that file cannot work).
    """
    repo = os.path.dirname(os.path.abspath(here))                    # <repo>
    up = os.path.dirname(repo)                                       # above <repo>
    ordered = [os.path.join(repo, "Jax"),                            # tracked here
               os.path.join(here, "Jax"), here,                      # Stationary, and in it
               repo,                                                # the repo root
               os.path.join(up, "Jax"), up]                          # and above the repo
    seen, out = set(), []
    for cand in ordered:
        cand = os.path.normpath(cand)
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def _crunch_minimize(root: str | None = None):
    """Crunch's SciPy-style `minimize`, or (None, reason) when it is not available.

    Looks in every location `_crunch_candidates` lists and returns the first that imports;
    the phase prints which one it was.  `CRUNCH_ROOT`, or the `root` argument, overrides the
    search with a SINGLE location -- deliberately, so that a typo is reported rather than
    silently satisfied by some other copy.  The import is lazy and soft in every case: when
    nothing imports, the quasi-Newton phase falls back to optax.lbfgs with a printed reason.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # <repo>/Stationary
    explicit = root or os.environ.get("CRUNCH_ROOT")
    candidates = [os.path.normpath(explicit)] if explicit else _crunch_candidates(here)
    reasons = []
    for cand in candidates:
        if not os.path.isdir(os.path.join(cand, "Crunch", "Optimizers")):
            reasons.append(f"no Crunch/Optimizers under {cand}")
            continue
        if cand not in sys.path:
            sys.path.append(cand)
        try:
            from Crunch.Optimizers.minimize_backtracking import minimize
        except Exception as exc:                                         # pragma: no cover
            # A root can hold a Crunch that does not import: an incomplete copy, or one whose
            # line_search_backtracking is missing.  Undo what the attempt left behind -- in
            # particular sys.modules, where a half-built `Crunch` would otherwise be returned
            # to the NEXT candidate's `import Crunch` and fail there for the wrong reason.
            reasons.append(f"{cand}: {exc}")
            for mod in [m for m in list(sys.modules)
                        if m in ("Crunch", "line_search_backtracking")
                        or m.startswith("Crunch.")]:
                sys.modules.pop(mod, None)
            try:
                sys.path.remove(cand)
            except ValueError:
                pass
            continue
        return minimize, cand
    return None, ("; ".join(reasons) if reasons else "no candidate root")


# ------------------------------------------------------------------ reweighting
# Gradient-norm adaptive weighting of the four interior equation groups.  It lives out here,
# outside the optimiser loops, because every phase uses it: the rule used to sit inside the
# Adam loop, so a run that skipped or shortened Adam never reweighted at all.
# `runs/production_quad_quarter_pin_r400` advertised `reweight_every = 1500`, ran 500 Adam +
# 8000 quasi-Newton iterations, and logged no `[reweight]` line -- the period was longer
# than its whole Adam phase.
def _reweight_crosses(cfg: Config, lo: int, hi: int) -> bool:
    """Does the cumulative iteration interval (lo, hi] contain a reweighting point?

    For the two phases that advance one iteration at a time -- Adam and the `optax.lbfgs`
    fallback -- where `(it - 1, it]` crossing a multiple is exactly Adam's old trigger.  The
    SSBroyden phase advances in blocks instead and carries an explicit counter
    (`_first_reweight_after`), so that a block whose line search stops early cannot re-trigger
    on a point it has already passed.  `reweight_every` counts OPTIMISER ITERATIONS on one
    counter that every phase advances, so the schedule is continuous from Adam into the
    quasi-Newton phase.
    """
    e = int(cfg.reweight_every or 0)
    return e > 0 and hi > 1 and hi >= e and hi // e > lo // e


def _first_period_after(every: int, iters: int):
    """First multiple of `every` strictly after `iters`, or None if the period is off."""
    e = int(every or 0)
    return None if e <= 0 else (iters // e + 1) * e


def _first_reweight_after(cfg: Config, iters: int):
    """First reweighting point strictly after `iters` cumulative optimiser iterations.

    This is where the quasi-Newton phase picks the schedule up from the Adam phase: passing
    `cfg.steps` skips any point Adam already consumed, so the two phases never both reweight
    at the same iteration.
    """
    return _first_period_after(cfg.reweight_every, iters)


def _device_memory():
    """(bytes in use, bytes limit) for the default device, or (None, None).

    The companion to `_map_budget` and NOT the same thing: mappings count address-space regions
    (what the CPU backend exhausts), while these are the allocator's own bytes, which is what
    the GPU backend exhausts.  A run that died with every request from 2.00 GiB down to 342 MiB
    refused for `jit_while` needed this number and there was no way to get it after the fact --
    the process is gone and `memory_stats` is not recoverable.
    """
    try:
        st = jax.devices()[0].memory_stats()
        if not st:
            return None, None
        return st.get("bytes_in_use"), st.get("bytes_limit")
    except Exception:
        return None, None


def _dev_str() -> str:
    """`dev=in/limit GiB` for a log line, or "" where the backend does not report it."""
    u, lim = _device_memory()
    if u is None or lim is None:
        return ""
    return f"  dev={u / 2**30:.2f}/{lim / 2**30:.2f}GiB"


def _map_budget():
    """(mapped regions this process holds, the kernel's limit for it), or (None, None).

    Linux only; everywhere else the answer is "not measurable" and every caller must cope.
    This is what the quasi-Newton phase runs out of: `minimize_bfgs` is not jitted and builds
    its line search -- and the whole objective inside it -- afresh on every call, so each
    block compiles a new XLA program whose LLVM section memory is never returned.  Those are
    tiny code mappings, so RSS does not move and no memory limit is ever reached; the count
    climbs by a fixed amount per block until `vm.max_map_count` (65530 by default) refuses the
    next small one, which surfaces as `LLVM ERROR: Unable to allocate section memory!` after
    `allocateMappedMemory failed with error: Cannot allocate memory`.
    """
    try:
        with open("/proc/self/maps") as fh:
            cur = sum(1 for _ in fh)
        with open("/proc/sys/vm/max_map_count") as fh:
            return cur, int(fh.read().split()[0])
    except (OSError, ValueError):
        return None, None


def boundary_number(parts) -> float:
    """The unweighted boundary content the plateau test watches, from BOTH spheres.

    Every `inner_*`, `outer_*` and `pin_*` term, added.  The inner sphere used to be left out,
    which let a run stop on a flat loss and a flat outer boundary while the inner data were
    still being approached: `production_quad_quarter_pin_r400_long` stopped at step 1300 with
    its outer Robin at 1.1e-09 and an inner lambda residual of 1.6e-04, the worst number in
    its own report.  A boundary still being approached is not convergence, wherever it is.
    """
    return sum(v for k, v in parts.items() if k.startswith(("inner_", "outer_", "pin_")))


def _pde_str(parts, pde_keys) -> str:
    """The `pde(...)` fragment of the per-step log, shared by the Adam and quasi-Newton lines.

    Lists only the groups the formulation imposes (`losses.equation_keys`), so a metric-only
    run does not print a `compat` that is not in its loss.  The residual itself is still
    computed and still lands in `report.json` as `pde_compat`: it is the structural check that
    Gamma really is the Christoffel symbol of h, and it would be the first thing to move if the
    derivation broke.
    """
    names = {"lam_eq": "lam"}
    bits = [f"{names.get(k, k)}={float(parts.get(f'pde_{k}', 0.0)):.2e}" for k in pde_keys]
    lam_r = float(parts.get("pde_lam_eq_radial", 0.0))
    if lam_r:
        bits.append(f"lam_rad={lam_r:.2e}")
    return "pde(" + " ".join(bits) + ")"


def _reweighted(cfg: Config, weights, g, when: int, verbose: bool = True, pde_keys=None):
    """One gradient-norm reweighting update; returns the new weight dict.

    Reweights ONLY the interior equation groups the loss actually carries -- `pde_keys`, which
    the caller takes from `losses.equation_keys(model)`.  For a metric-only model that is
    `ricci, gauge, lam_eq`: `compat` holds identically there, so it is not in the loss and must
    not be balanced against terms that are.  (The `reweight_floor` rule below would drop it
    anyway, its gradient sitting at the round-off floor, but a group absent from the loss being
    weighted by a rule that cannot see the loss is exactly the kind of accident this avoids.)

    The boundary terms are data, and w ~ 1/||grad term|| is backwards for them: the condition
    that is most violated has the largest gradient and so receives the SMALLEST weight -- which
    is how an earlier run silenced its outer Robin condition (w_outer fell to 6.25 while
    w_inner rose to 229), let lambda stay at its inner value and drift onto the trivial flat
    branch.
    """
    pde_keys = tuple(pde_keys or FIRST_ORDER_PDE_KEYS)
    # ... and not even all of those: a group that is satisfied identically (compat is, in
    # the metric-only schemes) has a gradient at the round-off floor, and w ~ 1/||grad||
    # then grows it without bound while pushing the others down.  Leave such groups alone
    # and keep them out of the target.
    gmax = max(float(g[i]) for i, k in enumerate(GROUP_KEYS) if k in pde_keys)
    live = [k for k in pde_keys if float(g[GROUP_KEYS.index(k)]) > cfg.reweight_floor * gmax]
    if not live:
        # Every interior group is at the round-off floor (a converged solution, or a state
        # where all four gradients underflow to zero).  There is nothing to balance between,
        # and `jnp.stack` of an empty list raises -- so leave the weights alone.  This was
        # reachable before the rule was extracted from the Adam loop; the guard is new.
        if verbose:
            print(f"[reweight {when}] every interior group is at the gradient floor "
                  f"(max {gmax:.2e}); weights unchanged", flush=True)
        return dict(weights)
    target = jnp.mean(jnp.stack([g[GROUP_KEYS.index(k)] for k in live]))
    w0 = {k: v for k, v in default_weights(cfg).items()}
    new = {}
    for i, k in enumerate(GROUP_KEYS):
        if k not in live:
            new[k] = weights[k]        # boundary data, or nothing to balance
            continue
        ideal = target / (g[i] + 1e-300)
        ratio = jnp.clip(ideal / weights[k], cfg.reweight_max_ratio_inv,
                         1.0 / cfg.reweight_max_ratio_inv)
        lo, hi = w0[k] / cfg.reweight_band, w0[k] * cfg.reweight_band
        new[k] = jnp.clip(weights[k] * jnp.sqrt(ratio), lo, hi)
    if verbose:
        print(f"[reweight {when}] " + " ".join(f"{k}={float(new[k]):.3g}" for k in GROUP_KEYS),
              flush=True)
    return new


def ssbroyden_phase(state, batch, weights, loss_fn, cfg: Config, verbose: bool = True,
                    gradnorms=None, history=None, pde_keys=None, start_H=None, start_total=0):
    """Quasi-Newton phase with Crunch's self-scaling Broyden; None means "use optax.lbfgs".

    Two things make it decline, both reported rather than raised: no Crunch checkout is found
    (see `_crunch_candidates` for where it is looked for), or the dense inverse-Hessian estimate
    would not fit.  That estimate is n_params^2, so the production network (13 828
    parameters) needs 1.53 GB in float64 and 0.76 GB in float32, against `qn_max_H_gb`.

    The optimiser works on a flat vector (`jax.flatten_util`), is given the identity as its
    initial inverse Hessian, and selects the self-scaling Broyden recurrence with
    `update_method="ssbroyden2"` -- `initial_H` and that switch travel inside `options`, which
    is where the SciPy-style wrapper forwards them.  One batch, held fixed: the line search
    needs a single objective.

    `gradnorms(state, batch)` is the jitted per-group gradient-norm function used by the
    Adam phase's reweighting.  Given it, this phase reweights the interior equation groups
    on the same cumulative-iteration schedule (`cfg.reweight_every`), which is what makes
    `--steps 0 --reweight-every 1500` mean something.  A reweight changes the objective, so
    the carried inverse Hessian is dropped and `initial_scale` re-engaged for that block:
    H describes the old function, and the first step of a new objective needs the same
    treatment the first block gets.
    """
    minimize, where = _crunch_minimize()
    if minimize is None:
        if verbose:
            print(f"[qn] SSBroyden unavailable ({where}); using optax.lbfgs", flush=True)
        return None

    flat0, unflatten = jax.flatten_util.ravel_pytree(state)
    n = int(flat0.size)
    import math as _math
    gb = n * n * jnp.dtype(flat0.dtype).itemsize / 2**30
    if gb > cfg.qn_max_H_gb:
        if verbose:
            print(f"[qn] SSBroyden wants a dense {n}x{n} inverse Hessian = {gb:.2f} GB "
                  f"> --qn-max-H-gb {cfg.qn_max_H_gb:g}; using optax.lbfgs.  Shrink the "
                  f"network, or raise the cap if the memory is really there.", flush=True)
        return None

    _traces = [0]        # see the resource line: a retrace is a leaked LLVM module

    @jax.jit
    def fun(flat, batch, weights):
        """The objective handed to Crunch -- JITTED, and it has to be.

        Crunch evaluates it as `jax.value_and_grad(fun)(x0)` with no jit of its own
        (`Optimizers/bfgs_backtracking.py:108`), once at the start of every block.  Unjitted,
        JAX traces the function and then executes its primitives ONE AT A TIME: measured
        62.4 s per call against 1.1 s with the jit cache warm, a factor of 56, and it is the
        same pathology as eager dispatch rather than merely a slow one -- a thousand separate
        compilations, each taking its own allocation out of the BFC pool.

        That is what killed `production_quad_quarter_pin_r400_long` at its FIRST block:
        `bfc_allocator.cc:598` refusing 1.27 MiB with the "this may mean fragmentation"
        notice, then the autotuner unable to find 27.39 MiB, the traceback pointing straight
        at `bfgs_backtracking.py:108` -> `fun`.  The pool had grown and fragmented under the
        per-primitive allocations and could not serve the next graph.
        """
        _traces[0] += 1
        value, _ = loss_fn(unflatten(flat), batch, 1.0, weights)
        return value

    @jax.jit
    def loss_value_and_parts(flat, batch, weights):
        """(loss, parts) at these parameters -- COMPILED, and it has to be.

        This has to be jitted.  Called eagerly it runs the whole loss at the collocation
        points op by op under Python: measured at 39.9 s against 0.052 s compiled for
        n_coll 2048 in float64, a factor of 764.  The phase needs it once before the first
        block and once per block (for the plateau test's outer number and the log line), so
        at `--qn-block 100` an eager full-loss evaluation used to land in every hundredth
        iteration -- the right order to account for the 0.69-0.85 s per iteration these runs
        take (`production_quad_quarter_pin_r400` spent 119.8 min on 8001 iterations,
        `production_quad_pin` 185.1 min on 16000) against the 0.048 s per value+gradient the
        tooling reports.

        On a 12 GiB slice it is also what BREAKS them.  Eager dispatch materialises every
        intermediate, including the ones compilation fuses away, and those grow with the
        collocation count -- so `production_quad_quarter_pin_r400_long` (n_coll 32768) died at
        its first quasi-Newton block with the BFC allocator refusing 3.80 MiB, then 1.27 MiB,
        then 432 KiB, and finally failing inside the autotuner on 27.39 MiB.  Those are not
        the sizes a working set fails on; that is a pool already filled and fragmented by the
        two eager evaluations that ran immediately before the compiled gradient.
        """
        return loss_fn(unflatten(flat), batch, 1.0, weights)

    def outer_res(flat, b=None):
        """(unweighted boundary contribution from BOTH spheres, all loss parts) here.

        The plateau test looks at the boundary terms as well as at the total loss: one group
        can dominate the total (the order-2 control's final loss was 97% its lambda-equation
        while its boundary was already at the floor), so "the loss stopped moving" can hide an
        abandoned boundary condition.  The pins are included: a run whose far-field value is
        still moving is not converged, however flat its loss looks.

        BOTH spheres, and the inner terms were missing until this run made the cost visible:
        `production_quad_quarter_pin_r400_long` stopped at step 1300 with its loss and its
        outer Robin flat to 1e-4 relative, while the inner residual was the worst number in
        its own report -- 1.6e-04 in lambda against the 16384-point run's 1.7e-05.  A boundary
        that is still being approached is not convergence, whichever sphere it is on.

        The parts come back too because the per-block checkpoint needs them: they are what
        makes a crashed run's `history.json` a real trajectory rather than a single number.
        """
        parts = {k: float(v) for k, v in loss_value_and_parts(flat, b if b is not None else batch, weights)[1].items()}
        return boundary_number(parts), parts

    def write_progress(row):
        """Append a trajectory row and put it, plus the weights, on disk -- atomically.

        This is what makes a crash cost at most one block instead of the whole phase.  The
        phase runs for hours and has no checkpoint of its own, and with `--steps 0` there is
        no ckpt.pkl either: a run killed inside its first block used to leave ONLY
        `params_adam.pkl` -- the Adam warm-up, none of the work -- and `history.json`, which
        is what makes `report.txt` say how far the run got, was not written until the end.
        A report with no trajectory cannot tell a converged 8000 from an abandoned 800.

        The PDE weights and the curvature estimate are not carried: a continuation
        re-adapts the weights and rebuilds H over its first blocks.
        """
        rows_so_far.append(row)
        tmp = os.path.join(cfg.outdir, "history.json.tmp")
        with open(tmp, "w") as fh:
            json.dump(history_prefix + rows_so_far, fh, indent=2)
        os.replace(tmp, os.path.join(cfg.outdir, "history.json"))
        write_params(cfg.outdir, unflatten(x))
        # AND a resume checkpoint, at the same place and under the same key the Adam phase
        # uses, so there is one resume path rather than two.  Before this, `ckpt.pkl` was
        # written only inside the Adam loop, so `--resume auto` on a run that died here --
        # which is where hours are spent -- rewound to the END OF ADAM and threw the whole
        # quasi-Newton phase away.  The step is the cumulative counter, so the Adam loop runs
        # zero iterations on resume and `qn_stopped_at` recovers the iteration count.
        #
        # `carry_H` is the inverse Hessian, n^2 float64.  Worth carrying at 2285 parameters
        # (42 MB, once per block); not at 13828 (1.5 GB per block, 300 blocks).  Above the
        # gate the field and the step are still checkpointed and the phase rebuilds H, which
        # is what `write_progress` re-adapting the weights already assumed.
        save_checkpoint(os.path.join(cfg.outdir, CKPT_NAME), unflatten(x), None, weights,
                        history_prefix + rows_so_far, cfg.steps + total, cfg,
                        extra={"phase": "qn",
                               "qn_H": (jax.device_get(H) if carry_H else None)})

    # Blocks, not one long call: the inverse Hessian is carried from block to block (that is
    # what makes the quasi-Newton phase work), and the loss is inspected between blocks so
    # the run can stop when it PLATEAUS instead of at a fixed iteration count.
    #
    # On a RESUME this is the Hessian the interrupted phase had reached, when the checkpoint
    # could carry it (it is n^2 floats: free at 2285 parameters, 1.5 GB at 13828, which is why
    # `carry_H` gates it).  Without it the phase restarts from the identity, which costs the
    # first few blocks rather than the whole run -- the field, not the curvature, is what
    # cannot be re-derived cheaply.
    H = (jnp.asarray(start_H, dtype=flat0.dtype) if start_H is not None
         else jnp.eye(n, dtype=flat0.dtype))
    maps_start, maps_limit = _map_budget()
    if verbose and maps_start is not None:
        # BEFORE the first block, and that is the point: the phase's first block compiles the
        # objective (jit_fun) and the line search, and a run that dies IN that compile used to
        # leave no resource line at all -- only the post-block one existed.  This is the
        # baseline those compilations start from, and with `maps_limit` it says how much room
        # there was.
        print(f"[qn]   before any block: maps={maps_start} of {maps_limit} "
              f"({100 * maps_start // maps_limit}% of the kernel's limit already in use)",
              flush=True)
    x = flat0
    # ONE compiled call for the opening loss, its parts and the outer number.  This used to
    # be two eager ones (`fun(x)` and `outer_res(x)`), and at n_coll 32768 they are what
    # filled and fragmented the allocator immediately before Crunch compiled its first
    # gradient -- see loss_value_and_parts.  parts_start is evaluated HERE, before any
    # reweighting, so the opening row of the history is measured under the same weights as
    # `losses[0]`; it used to be recomputed at the end, after the loop had reweighted.
    _v0, _p0 = loss_value_and_parts(x, batch, weights)
    f = float(_v0)
    parts_start = {k: float(v) for k, v in _p0.items()}
    o = boundary_number(parts_start)
    block = max(1, int(cfg.qn_block))
    # On a resume, only the iterations still owed: `--lbfgs-steps` is the cap on the CUMULATIVE
    # counter (that is what the per-block line prints, and what the plateau test compares), so
    # resuming a 200-iteration run with --lbfgs-steps 400 must run four more blocks, not eight.
    # It ran eight before this: the cap was recomputed from zero, so every resume bought
    # another full budget and a resumed run overran by exactly what it had already done.
    n_blocks = int(_math.ceil(max(0, int(cfg.lbfgs_steps) - start_total) / block))
    # The opening row MEASURES the state the phase starts from.  On a fresh start that is the
    # state after Adam, i.e. step cfg.steps + 1; on a resume it is the state at the iteration
    # the checkpoint carries, so it takes that step rather than one past it -- otherwise a
    # resume that finds nothing left to do reports one iteration more than it has done.
    step0 = cfg.steps + (start_total if start_total > 0 else 1)
    history_prefix = list(history or [])
    rows_so_far = [{"step": step0, "loss": f, **parts_start}]
    losses = [f]
    outers = [o]
    # The plateau DECISION uses a batch that NEVER changes, so consecutive blocks are
    # comparable.  With --resample-every the training loss is measured on a different sample
    # every cadence, so its block-to-block difference is sampling noise -- and the test asks
    # whether the value stopped IMPROVING, so any upward jitter reads as a plateau.  Measured
    # on pq_u100_s2_12: it stopped at 13000 of 30000 with the loss going 3.881211e-06 ->
    # 4.608388e-06 and the outer Robin 7.36e-10 -> 1.47e-09, both RISES produced by the
    # resample and not by the optimiser.  Seeded like `losses`/`outers` above, so that the
    # `[-1 - pat]` indices refer to the same blocks.
    plateau_batch = make_batch(jax.random.PRNGKey(cfg.seed + 31337), cfg)
    _fp0, _ = loss_value_and_parts(x, plateau_batch, weights)
    _op0, _ = outer_res(x, plateau_batch)
    plosses, pouters = [float(_fp0)], [float(_op0)]
    best_f, best_o, best_at, best_x = _fp0, _op0, 0, None
    stopped_on_plateau = False
    total = start_total
    status = -1
    t0 = time.time()
    if verbose:
        print(f"[qn] SSBroyden (ssbroyden2) from {where}: {n} parameters, {gb:.2f} GB "
              f"inverse Hessian, blocks of {block} up to {cfg.lbfgs_steps} iterations "
              f"({n_blocks} blocks, plateau_tol {cfg.plateau_tol:g} over "
              f"{cfg.plateau_patience} blocks), start loss {f:.6e}", flush=True)
        if cfg.reweight_every and gradnorms is None:
            print(f"[qn] reweight_every = {cfg.reweight_every} was asked for but no "
                  f"gradient-norm function was supplied: the PDE weights stay as configured",
                  flush=True)
    # The reweighting points still to come, on the cumulative iteration counter Adam was
    # advancing.  The first one is the first multiple of `reweight_every` after the Adam
    # phase (so a point Adam already consumed is not used twice here).  A monotone counter
    # rather than a window test: a block may stop early in its line search, and a window
    # would then re-trigger on a point it had already passed.
    e_rs = int(cfg.resample_every or 0)
    next_rs = _first_period_after(e_rs, cfg.steps + total)
    e_rw = int(cfg.reweight_every or 0)
    # The inverse Hessian, in bytes: n^2 float64.  Carried in the per-block checkpoint only
    # while it is small enough that writing it every block is cheaper than rebuilding it.
    carry_H = (n * n * 8 <= 128 * 1024 * 1024)
    next_rw = (_first_reweight_after(cfg, cfg.steps + total)
               if gradnorms is not None else None)
    for b in range(n_blocks):
        # Reweight BEFORE the block, and throw away the curvature estimate: the objective has
        # just changed, so H no longer describes it.  `initial_scale` then re-derives the step
        # length for H = I on this block, exactly as it does for the first one.  The points are
        # consumed against the block's PLANNED span, which is the most a block can cover.
        reweighted = False
        while next_rw is not None and cfg.steps + total + block >= next_rw:
            weights = _reweighted(cfg, weights, gradnorms(unflatten(x), batch), next_rw,
                                  verbose, pde_keys)
            next_rw += e_rw
            reweighted = True
        # REDRAW the collocation sample, on the same cumulative counter the reweight uses.
        # The inverse Hessian is KEPT: a reweight changes the objective function and H is then
        # describing the old one, but a resample changes only the sample -- the function is the
        # same one, estimated on fresh points -- so H remains a valid approximation to it.
        while next_rs is not None and cfg.steps + total + block >= next_rs:
            batch = make_batch(jax.random.PRNGKey(cfg.seed + 777 + next_rs), cfg)
            next_rs += e_rs
            if verbose:
                print(f"[resample {cfg.steps + total + block}] new collocation sample "
                      f"of {cfg.n_coll} points", flush=True)
        if reweighted:
            H = jnp.eye(n, dtype=flat0.dtype)
        if b == 0 and maps_start is not None:
            maps_pre, _ = _map_budget()
            print(f"[qn]   entering block 1 (this is where jit_fun compiles): "
                  f"maps={maps_pre}" + _dev_str(), flush=True)
        # the data-before-shape rule: same shapes, new values, no recompilation
        res = minimize(lambda flat: fun(flat, batch, weights), x, args=(), method="BFGS",
                       options={"maxiter": block, "gtol": cfg.qn_gtol,
                                "initial_H": H,
                                # initial_scale engages SSBroyden's tau_k^A: without it the
                                # first step is -grad with H = I, which the Wolfe line search
                                # cannot bracket when the Adam warm-up has left the gradient
                                # large (measured: zero iterations, status 3 "zoom failed").
                                # Later blocks carry a real H, so it is not needed there --
                                # nor after a reweight, where H has been reset to I.
                                "update_method": "ssbroyden2",
                                "initial_scale": (b == 0 or reweighted)})
        if b == 0 and maps_start is not None:
            m, _ = _map_budget()
            # the delta ACROSS the first minimize call: that one call compiles jit_fun and the
            # line search, and on this problem it is the single largest mapping cost there is
            print(f"[qn]   block 1 compiled and ran: maps={m}  "
                  f"(+{(m or 0) - maps_pre} across the first minimize call, which in total "
                  f"took maps {maps_start} -> {m})" + _dev_str(), flush=True)
        x = res.x
        f = float(res.fun)
        if res.hess_inv is not None:
            H = res.hess_inv
        total += int(res.nit)
        status = int(res.status)
        o, parts_b = outer_res(x)
        losses.append(f)
        outers.append(o)
        fp, _ = loss_value_and_parts(x, plateau_batch, weights)
        op, _ = outer_res(x, plateau_batch)
        plosses.append(float(fp))
        pouters.append(float(op))
        write_progress({"step": cfg.steps + total, "loss": f, **parts_b})
        if b == 0 and maps_start is not None:
            # Self-calibrating: the cost depends on the size of the compiled line search, so
            # it is MEASURED on this run's first block rather than assumed.  Said once, before
            # the run wastes an hour dying at block 23 for the fourth time.
            maps_now, _ = _map_budget()
            cost = (maps_now or 0) - maps_start
            if cost > 0:
                room = (maps_limit - maps_now) // cost
                if n_blocks - 1 > room:
                    safe = max(block, -(-int(cfg.lbfgs_steps) // max(1, int(room))))
                    print(f"[qn] WARNING: {cost} address-space regions leaked per block, and "
                          f"{maps_limit - maps_now} of the kernel's {maps_limit} remain: this "
                          f"process has room for ~{room} more blocks of the {n_blocks - 1} "
                          f"planned.", flush=True)
                    print(f"[qn]   It will die of 'LLVM ERROR: Unable to allocate section "
                          f"memory' near block {1 + int(room)}, after ~"
                          f"{int(room) * block} more iterations.  Use --qn-block {safe} or "
                          f"larger: the leak is per BLOCK, not per iteration, so bigger"
                          f" blocks cost nothing.", flush=True)
        if cfg.log_resources:
            # `jit` is the number of compiled executables this objective holds.  It must stay
            # at 1: it is one jitted function called with one shape, so ANY growth means a
            # retrace, and a retrace is a new LLVM module whose section memory is never freed.
            # `maps` is the kernel's mapping count for this process, against vm.max_map_count.
            res = [f"traces={_traces[0]}"]
            try:
                with open("/proc/self/maps") as fh:
                    res.append(f"maps={sum(1 for _ in fh)}")
            except OSError:
                pass
            try:
                with open("/proc/self/status") as fh:
                    for ln in fh:
                        if ln.startswith("VmRSS:"):
                            res.append(f"rss={int(ln.split()[1]) // 1024}MB")
                            break
            except OSError:
                pass
            if (d := _dev_str()):
                res.append(d.strip())
            print(f"[qn]   resources: {'  '.join(res)}", flush=True)
        if verbose:
            # The same breakdown the Adam loop prints, for the same reason: a run that stalls
            # has to be attributed to an equation group WHILE it is running.  With `--steps 0`
            # there is no Adam line to carry it, and the only other number here is the total
            # loss, which cannot tell a stuck lambda-equation from a stuck Ricci group -- a run
            # at 6.2e-03 falling 1% per block looks the same whether the cause is the equation,
            # the gauge, or the radial term.  (`outer=` is gone as a separate item because
            # `bc_out + pin` is the same number, and each part is more use than their sum.)
            pins = sum(v for k, v in parts_b.items() if k.startswith("pin_"))
            print(f"[qn] block {b + 1:3d}: {total:5d} iterations, loss {f:.6e}  "
                  f"{_pde_str(parts_b, pde_keys)}  "
                  f"bc_in={sum(v for k, v in parts_b.items() if k.startswith('inner_')):.2e} "
                  f"bc_out={sum(parts_b[k] for k in parts_b if k.startswith('outer_')):.2e}"
                  + (f"  pin={pins:.2e}" if pins else "")
                  + f"  (status {status})", flush=True)
        pat = int(cfg.plateau_patience)
        if status == 0:
            break
        # BEST SO FAR, not "better than `pat` blocks ago".  On a FIXED sample a transient rise
        # is not a plateau, and the previous form stopped on the first one: measured on
        # pq_u100_s2_12 the fixed-sample value went 8.38e-06 -> 1.58e-04 while the training
        # loss fell to 4.21e-06, and the run stopped at 14000 of 30000 on that first rise.  A
        # new best resets the counter; only `pat` blocks without one is a plateau.
        # 1% improvement is a new best.  This USED to be cfg.plateau_tol -- the same value that
        # decides the stop -- and the two pull opposite ways: a tolerance large enough to force
        # a plateau is large enough that no block ever beats the best, so best_x stayed None and
        # the restoration could never fire at all.  Verified by a plateau run printing the
        # restoration line for the first time.
        if fp < best_f - 0.01 * abs(best_f):
            best_f, best_at, best_x = fp, len(plosses), x
        if op < best_o - 0.01 * abs(best_o):
            best_o = op
        if total >= cfg.plateau_min_iters and len(plosses) - 1 - best_at > pat:
                stopped_on_plateau = True
                if verbose:
                    print(f"[qn] plateaued: no new best for {pat} blocks"
                          f" (best loss {best_f:.6e} at block {best_at}, "
                          f"best outer Robin {best_o:.6e}); "
                          f"tol {cfg.plateau_tol:g} of the best); stopping after {total} "
                          f"iterations", flush=True)
                break

    # RESTORE THE BEST FIELD.  A plateau stop otherwise discards the best parameters the run
    # ever found: measured on pq_u100_s2_12, it stopped with a training loss of 4.419711e-06
    # when block 1 had reached 3.986826e-06, and with pin_h_rr 6.53e-06 against 9.64e-07 at the
    # previous stop.  `best_x` is the field that set the fixed-sample best whose absence the
    # plateau rule detected, so it is the field that rule is about.
    # ONLY on a plateau stop.  At the iteration cap the restoration is actively wrong: at
    # ratio 200 it kept block 10 (training loss 1.49e-05) over the cap's 7.45e-06.
    if best_x is not None and stopped_on_plateau:
        if verbose:
            print(f"[qn] restored the best field seen (fixed-sample loss {best_f:.6e}); "
                  f"the last block's was {f:.6e}", flush=True)
    state = unflatten(x)
    # The per-block rows already are the trajectory, and the last of them is the final
    # state; `write_progress` has been putting them on disk as they happened.  One row per
    # block replaces the old two-row summary, which is what lets a crashed run's report show
    # where it stopped -- and costs nothing, since `outer_res` was already evaluating the
    # parts the plateau test needs.
    history = list(rows_so_far)
    if verbose:
        print(f"[qn] SSBroyden: loss {losses[0]:.6e} -> {f:.6e} in {total} iterations "
              f"({time.time() - t0:.1f}s, status {status}, "
              f"converged={status == 0}, stopped={total < cfg.lbfgs_steps})", flush=True)
    return state, history, weights


def train(cfg: Config, verbose: bool = True, init_from: str | None = None,
          resume: str | None = None):
    os.makedirs(cfg.outdir, exist_ok=True)
    # The configuration is known before anything is computed, so record it now rather than
    # at the end: `postprocess.sh` refuses a directory without config.json ("not a run
    # directory"), so writing it last meant a run that died anywhere in the quasi-Newton
    # phase -- which has no checkpoint of its own, and with `--steps 0` no ckpt.pkl either --
    # could not be post-processed AT ALL, whatever had survived in it.  Nothing mutates cfg
    # after this point.
    # NEVER clobber the configuration of a run we are resuming into.  This write happens
    # before `build()`, so a launch whose flags are wrong dies AFTER replacing the run's own
    # config with its own -- and every later post-process then rebuilds the model from the
    # wrong architecture and dies too.  Measured: an attempt with an empty $COMMON left
    # `config.json` saying `arch sym, rho in [2.236, 20], R0 = 1, dirichlet_exact, n_coll 4096`
    # while the checkpoint was `axisym_hybrid, rho in [1, 100], R0 = 0.5773, robin, 32768`, and
    # the run's own report, figures and report.json were all produced against the wrong model.
    # The incoming flags still get recorded, under a name post-processing does not read.
    _res = resume if resume is not None else cfg.resume
    _cfg_path = os.path.join(cfg.outdir, "config.json")
    _keep = _res is not None and os.path.exists(_cfg_path)
    with open(os.path.join(cfg.outdir, "config.resume.json" if _keep else "config.json"),
              "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)
    if _keep:
        print(f"[config] resuming: keeping {_cfg_path} and writing these launch flags to "
              f"config.resume.json instead", flush=True)
    if verbose:
        print_config_summary(cfg)
    model, state, exact_fields = build(cfg, init_from)
    # NOT wired here yet.  Turning compat off in the training path broke the Adam phase's
    # gradient-norm loop, which indexes group_terms(...)[k] for every k in GROUP_KEYS and
    # raises KeyError: 'compat' at train.py:1053.  This was missed because the tests used
    # --steps 0, which skips Adam altogether.  The proper fix is for that loop to iterate the
    # keys the formulation imposes (losses.equation_keys(model)) rather than GROUP_KEYS; until
    # then the residual is formed everywhere, as it was before.
    # from .geometry import set_want_compat
    # set_want_compat(not model.derives_gamma)
    if verbose and exact_fields is not None:
        print_reference_summary(cfg, exact_fields)
    loss_fn = lambda st, batch, sc: total_loss(st, batch, cfg, model, exact_fields, pde_scale=sc)

    batch = make_batch(jax.random.PRNGKey(cfg.seed + 1), cfg)
    schedule = optax.cosine_decay_schedule(cfg.lr, max(cfg.steps, 1), alpha=0.01)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state = opt.init(state)

    weights = default_weights(cfg)
    # The equation groups this formulation actually imposes: three for a metric-only model,
    # where compatibility holds identically, four for the first-order ones.  The loss, the
    # reweighting and the per-block log all read it from here so they cannot disagree.
    pde_keys = equation_keys(model)
    lam_inf_fixed = None if cfg.lam_inf is None else jnp.asarray(cfg.lam_inf)
    loss_fn = lambda st, b, sc, w: total_loss(st, b, cfg, model, exact_fields,
                                              pde_scale=sc, weights=w, lam_inf=lam_inf_fixed)

    @jax.jit
    def step(state, opt_state, batch, sc, w):
        (loss, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(state, batch, sc, w)
        updates, opt_state = opt.update(grads, opt_state, state)
        state = optax.apply_updates(state, updates)
        return state, opt_state, loss, parts

    @jax.jit
    def group_gradnorms(state, batch):
        def one(k):
            g = jax.grad(lambda st: group_terms(st, batch, cfg, model, exact_fields)[k])(state)
            return optax.global_norm(g)
        return jnp.stack([one(k) for k in GROUP_KEYS])

    history = []
    resume_step = 0
    if resume is None:
        resume = cfg.resume
    # Defaults for a fresh run, and for any checkpoint written before `phase` existed.
    resume_phase, resume_H, resume_loss = "adam", None, None
    if resume is not None:
        rpath = os.path.join(cfg.outdir, CKPT_NAME) if resume == "auto" else resume
        if not os.path.exists(rpath):
            raise FileNotFoundError(
                f"--resume {resume!r}: no checkpoint found at {rpath}. If the run was "
                f"interrupted before step --ckpt-every, or --ckpt-every was larger than "
                f"--steps, there is nothing to resume from.")
        with open(rpath, "rb") as fh:
            ck = pickle.load(fh)
        state = ck["state"]
        opt_state = ck["opt_state"]
        weights = ck["weights"]
        history = list(ck.get("history", []))
        resume_step = int(ck["step"])
        # Which phase wrote this.  "adam" (or a checkpoint from before this key existed) means
        # the Adam optimiser state is live and the Adam loop continues; "qn" means the
        # quasi-Newton phase had started, so the Adam loop must run zero iterations and the
        # phase is re-entered with its iteration count -- and its Hessian, when it was small
        # enough to carry.
        resume_phase = ck.get("phase", "adam")
        resume_H = ck.get("qn_H")
        # Warn rather than silently change the trajectory: the Adam LR schedule and
        # the resampling/reweighting cadences are all functions of the step index.
        prev = ck.get("config", {})
        for key in ("steps", "lr", "resample_every", "reweight_every", "pde_ramp_steps"):
            if key in prev and prev[key] != getattr(cfg, key):
                print(f"[resume] WARNING: {key}={getattr(cfg, key)!r} differs from the "
                      f"checkpoint's {prev[key]!r}; this will not reproduce the original "
                      f"run", flush=True)
        # The SHAPE keys are a refusal, not a warning: arch/width/depth/fourier decide the
        # parameter pytree, so `model.apply` cannot consume the checkpoint at all -- and
        # without this the failure is a Flax ScopeParamShapeError raised from inside the
        # network ("initializer expected (1, 20), existing parameter has shape (3, 20)"),
        # which names neither the flag nor either architecture, and it arrives only after the
        # run has loaded everything.  Measured: an attempt launched with an empty $COMMON fell
        # back to the default `arch = sym` (one input feature) against a checkpoint written by
        # `axisym_hybrid` (three), and died exactly that way.
        _shape_keys = ("arch", "width", "depth", "fourier")
        _bad = {k: (prev.get(k), getattr(cfg, k)) for k in _shape_keys
                if k in prev and prev[k] != getattr(cfg, k)}
        if _bad:
            raise ValueError(
                f"--resume {resume!r}: the checkpoint was written by a different NETWORK. "
                + "; ".join(f"{k}: checkpoint {a!r}, this run {b!r}"
                            for k, (a, b) in _bad.items())
                + "\n  These decide the parameter shapes, so the checkpoint cannot be loaded "
                  "at all.  Pass the flags that built it -- at least"
                  " --arch/--width/--depth/--fourier -- or start a fresh run in a new"
                  " --outdir.  An empty shell variable in the launch command is the usual"
                  " reason this happens.")
        if resume_step >= cfg.steps:
            print(f"[resume] {rpath} is already at step {resume_step} "
                  f"(steps={cfg.steps}); Adam phase complete", flush=True)
        else:
            print(f"[resume] continuing from {rpath} at step {resume_step}", flush=True)
        if resume_phase == "qn":
            # The field IS the result of the quasi-Newton work, so the report needs a loss for
            # it before the phase gets a chance to print a new one -- and `step` cannot supply
            # it, because that needs an Adam optimiser state this checkpoint does not have.
            resume_loss = float(history[-1]["loss"]) if history else None
            carried = ("carrying its inverse Hessian" if resume_H is not None else
                       "without the inverse Hessian, which that phase rebuilds "
                       "(too large to write every block)")
            print(f"[resume] checkpoint is from the QUASI-NEWTON phase at iteration "
                  f"{max(0, resume_step - cfg.steps)}, {carried}", flush=True)
        batch = batch_for_step(cfg, max(resume_step + 1, 1))

    # `resume_loss` is None for a fresh run and for an Adam checkpoint, so this is the same
    # `loss = None` as before in both of those cases.
    loss = resume_loss
    t0 = time.time()
    adam_losses = []
    adam_outers = []
    adam_stopped = None
    for it in range(resume_step + 1, cfg.steps + 1):
        sc = 1.0 if cfg.pde_ramp_steps <= 0 else min(1.0, it / cfg.pde_ramp_steps)
        if cfg.resample_every and it % cfg.resample_every == 1 and it > 1:
            batch = make_batch(jax.random.PRNGKey(cfg.seed + it), cfg)
        if _reweight_crosses(cfg, it - 1, it):
            weights = _reweighted(cfg, weights, group_gradnorms(state, batch), it, verbose,
                                  pde_keys)
        state, opt_state, loss, parts = step(state, opt_state, batch, jnp.asarray(sc), weights)
        # the Adam phase stops on the same plateau rule (it is a warm-up, not the workhorse)
        if it % cfg.log_every == 0:
            # both spheres, exactly as the quasi-Newton phase's plateau test does
            cur_outer = boundary_number(parts)
            adam_losses.append(float(loss))
            adam_outers.append(cur_outer)
            pat = int(cfg.plateau_patience)
            if it >= cfg.plateau_min_iters and len(adam_losses) > pat:
                old_f, old_o = adam_losses[-1 - pat], adam_outers[-1 - pat]
                if ((old_f - float(loss)) < cfg.plateau_tol * max(abs(old_f), 1e-300)
                        and (old_o - cur_outer) < cfg.plateau_tol * max(abs(old_o), 1e-300)):
                    if verbose:
                        print(f"[adam {it:6d}] plateaued: loss {old_f:.6e} -> "
                              f"{float(loss):.6e}, outer Robin {old_o:.6e} -> "
                              f"{cur_outer:.6e}; going to the quasi-Newton phase",
                              flush=True)
                    adam_stopped = it
                    break
        if it % cfg.log_every == 0 or it == 1:
            rec = {"step": it, "loss": float(loss),
                   **{k: float(v) for k, v in parts.items()}}
            history.append(rec)
            if verbose:
                # `pin` is its own field rather than folded into bc_out: the pins are values
                # (data), the Robin terms are decay conditions, and when a run goes to the
                # wrong branch it is precisely bc_out that stays at its floor.
                pins = sum((float(v) for k, v in parts.items() if k.startswith("pin_")), 0.0)
                print(f"[adam {it:6d}] loss={float(loss):.4e}  "
                      f"{_pde_str(parts, pde_keys)}  "
                      f"bc_in={float(sum(v for k, v in parts.items() if k.startswith('inner_'))):.2e} "
                      f"bc_out={float(sum(parts[k] for k in parts if k.startswith('outer_'))):.2e}"
                      + (f"  pin={pins:.2e}" if pins else ""),
                      flush=True)
        if cfg.ckpt_every and it % cfg.ckpt_every == 0:
            save_checkpoint(os.path.join(cfg.outdir, CKPT_NAME), state, opt_state,
                            weights, history, it, cfg)
            if verbose:
                print(f"[ckpt  {it:6d}] wrote {os.path.join(cfg.outdir, CKPT_NAME)}",
                      flush=True)

    if loss is None and opt_state is not None:
        # The Adam phase ran zero iterations (--steps 0, or a resume already at the
        # end of the schedule): evaluate once so the report still carries a loss.
        # `opt_state is None` means a `phase == "qn"` checkpoint, whose loss came with it.
        _, _, loss, parts = step(state, opt_state, batch, jnp.asarray(1.0), weights)

    # ------------------------------------------------------------- quasi-Newton
    if resume_phase != "qn":
        # NOT on a quasi-Newton resume: this would overwrite the Adam warm-up's parameters
        # with the field the quasi-Newton phase has since reached, and `params_adam.pkl` is
        # the record of where that warm-up ended.
        write_params(cfg.outdir, state, "params_adam.pkl")

    qn_stopped_at = 0
    if cfg.lbfgs_steps > 0:
        batch = make_batch(jax.random.PRNGKey(cfg.seed + 777), cfg)
        qn = None
        if cfg.qn_method == "ssbroyden":
            qn = ssbroyden_phase(state, batch, weights, loss_fn, cfg, verbose,
                                 gradnorms=group_gradnorms, history=history,
                                 pde_keys=pde_keys, start_H=resume_H,
                                 start_total=max(0, resume_step - cfg.steps))
        if qn is not None:
            state, qn_history, weights = qn
            history.extend(qn_history)
            loss = qn_history[-1]["loss"]
            qn_stopped_at = qn_history[-1]["step"] - cfg.steps
        else:
            solver = optax.lbfgs(
                learning_rate=1.0, memory_size=20,
                linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=30, verbose=False))
            lst = solver.init(state)

            @jax.jit
            def lstep(state, lst, batch, w):
                (value, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(state, batch, 1.0, w)
                updates, lst = solver.update(
                    grads, lst, state, value=value, grad=grads,
                    value_fn=lambda p: loss_fn(p, batch, 1.0, w)[0])
                return optax.apply_updates(state, updates), lst, value, parts

            prev = None
            for it in range(1, cfg.lbfgs_steps + 1):
                # Same cumulative schedule as Adam and SSBroyden.  A reweight changes the
                # objective, so the L-BFGS memory (the last 20 curvature pairs) describes the
                # old one and is dropped; the step length is recomputed from the new gradient.
                if _reweight_crosses(cfg, cfg.steps + it - 1, cfg.steps + it):
                    weights = _reweighted(cfg, weights, group_gradnorms(state, batch),
                                          cfg.steps + it, verbose, pde_keys)
                    lst = solver.init(state)
                state, lst, loss, parts = lstep(state, lst, batch, weights)
                if it % max(cfg.lbfgs_steps // 20, 1) == 0 or it == 1:
                    history.append({"step": cfg.steps + it, "loss": float(loss),
                                    **{k: float(v) for k, v in parts.items()}})
                    if verbose:
                        print(f"[lbfgs {it:5d}] loss={float(loss):.6e}", flush=True)
                cur = float(loss)
                if prev is not None and abs(prev - cur) < 1e-14 * max(1.0, abs(prev)):
                    break
                prev = cur

    wall = time.time() - t0

    # ------------------------------------------------------------- diagnostics
    pf = point_fields(model, state["net"])
    report = {"wall_seconds": wall, "final_loss": float(loss),
              "steps": cfg.steps, "lbfgs_steps": cfg.lbfgs_steps,
              "adam_stopped_at": adam_stopped if adam_stopped is not None else cfg.steps,
              # what the quasi-Newton phase actually was: the stored run info has to say so,
              # otherwise every report reads "Adam + L-BFGS" whatever was used
              "qn_method": cfg.qn_method, "qn_stopped_at": qn_stopped_at,
              "plateau_tol": cfg.plateau_tol, "plateau_patience": cfg.plateau_patience,
              "qn_block": cfg.qn_block}
    report.update(diagnostics.residual_report(pf, cfg))
    report.update(diagnostics.inner_boundary_report(pf, cfg))
    if exact_fields is not None:
        report.update(diagnostics.exact_comparison(pf, exact_fields, cfg))
    if "lam_inf" in state:
        report["lam_inf"] = float(state["lam_inf"])
    # The far-field pins are the one boundary check that looks at VALUES, so the stored run
    # has to say what they came to: a zero outer Robin residual with an unmet pin is exactly
    # the failure mode the pins exist to expose.
    for k, v in outer_pin_terms(pf, batch["outer"], cfg, exact_fields).items():
        report[f"pin_{k}"] = float(v)
    # ... and the radial-derivative equation term, which is otherwise visible only inside the
    # total loss: an interior term, so it belongs next to the residual_report numbers.
    for k, v in pde_radial_terms(pf, batch["coll"], cfg).items():
        report[f"pde_{k}"] = float(v)
    if cfg.make_figures:
        try:
            from .multipoles import make_figures
            report["figures"] = make_figures(pf, cfg, cfg.outdir, exact_fields=exact_fields)
        except Exception as exc:                      # plotting must never kill a run
            print(f"[figures] failed: {exc}", flush=True)

    # config.json was already written at the top of this function: it is the record the run
    # is identified by, and postprocess.sh refuses a directory without it, so it must not
    # depend on getting this far.  Rewritten here only so that it cannot go stale if some
    # later change starts mutating cfg mid-run.
    with open(os.path.join(cfg.outdir, "config.json"), "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)
    with open(os.path.join(cfg.outdir, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    with open(os.path.join(cfg.outdir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    write_params(cfg.outdir, state)

    if verbose:
        print("\n=== final diagnostics ===")
        for k in sorted(report):
            v = report[k]
            print(f"  {k:28s} {v:.6e}" if isinstance(v, float) else f"  {k:28s} {v}")
    return state, report, history


# ---------------------------------------------------------------------- CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--lbfgs-steps", type=int, default=None)
    p.add_argument("--qn-method", choices=("ssbroyden", "lbfgs"), default=None,
                   help="quasi-Newton phase: Crunch's SSBroyden (default), or optax.lbfgs")
    p.add_argument("--vtk", action="store_true",
                   help="write VTK files for VisIt when this run is post-processed")
    p.add_argument("--vtk-n-half", type=int, default=None,
                   help="points per half axis of the VTK grid (geometric grading)")
    p.add_argument("--vtk-physical-inner", type=float, default=None,
                   help="inner radius in the coordinates the VTK files are written in")
    p.add_argument("--h-robin-bases", type=str, default=None, dest="h_robin_bases",
                   help="decay powers for --pin-h-robin, e.g. h_rr=3,g2=2.  The defaults were "
                        "measured on the SPHERICAL reference and are an assumption about this "
                        "problem's own multipole expansion; set them to test it")
    p.add_argument("--qn-block", type=int, default=None,
                   help="iterations per quasi-Newton block (the plateau is checked between blocks)")
    p.add_argument("--log-resources", action="store_true", dest="log_resources",
                   help="append jit-cache size and mapped-region count to every "
                        "per-block line, for diagnosing host allocation failures")
    p.add_argument("--plateau-tol", type=float, default=None,
                   help="relative loss improvement below which the run is said to have plateaued")
    p.add_argument("--plateau-min-iters", type=int, default=None,
                   help="never stop before this many iterations, however flat the loss looks")
    p.add_argument("--qn-gtol", type=float, default=None,
                   help="quasi-Newton stop on ||grad||_inf")
    p.add_argument("--plateau-patience", type=int, default=None,
                   help="consecutive blocks (Adam: log_every checks) without improvement")
    p.add_argument("--qn-max-H-gb", type=float, default=None, dest="qn_max_H_gb",
                   help="memory cap for SSBroyden's dense inverse Hessian (n_params^2)")
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--R0", type=float, default=None)
    p.add_argument("--lam0", type=float, default=None)
    p.add_argument("--rho-out", type=float, default=None)
    p.add_argument("--n-coll", type=int, default=None)
    p.add_argument("--n-bnd", type=int, default=None)
    p.add_argument("--n-bnd-outer", type=int, default=None, dest="n_bnd_outer",
                   help="points on the OUTER sphere; defaults to --n-bnd.  The outer sphere "
                        "carries the Robin conditions and every pin, the inner only the "
                        "imposed data, so the two do not want the same resolution")
    p.add_argument("--resample-every", type=int, default=None)
    p.add_argument("--lam-inference", type=float, default=None, dest="lam_inf")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--fourier", type=int, default=None)
    p.add_argument("--outer-bc", type=str, default=None)
    p.add_argument("--rho-in", type=float, default=None)
    p.add_argument("--lam-inf", type=float, default=None)
    p.add_argument("--lam-inf-init", type=float, default=None)
    p.add_argument("--robin-exps", type=str, default=None)
    p.add_argument("--robin-order", type=int, default=None)
    p.add_argument("--robin-orders", type=str, default=None,
                   help="per-field orders, e.g. h=4,G=1,lam=4")
    p.add_argument("--no-robin-G", action="store_true")
    p.add_argument("--pin-lam", action="store_true",
                   help="pin the spherical MEAN of lambda at rho_out to the exact "
                        "reference's value there (the monopole, i.e. the branch); the "
                        "l >= 1 content stays free for the Robin conditions")
    p.add_argument("--pin-lam-robin", action="store_true", dest="pin_lam_robin",
                   help="pin the same monopole through the ORDER-1 ROBIN COMBINATION "
                        "instead of its value: rho d_rho <lam> + (<lam> - lam_inf) = 0, the "
                        "mean of the pointwise --robin-orders lam=1 condition.  Needs lam_inf "
                        "and NO reference, so unlike --pin-lam it works on a run whose inner "
                        "data are angular (S1/S2), where no spherical solution exists to pin "
                        "against")
    p.add_argument("--pin-h-tan", action="store_true", dest="pin_h_tan",
                   help="pin the tangential metric at rho_out to the reference's, angle "
                        "by angle (the areal-radius content)")
    p.add_argument("--pin-h-rr", action="store_true", dest="pin_h_rr",
                   help="pin h_rr at rho_out as well (the radial gauge component)")
    p.add_argument("--pin-h-robin", action="store_true", dest="pin_h_robin",
                   help="pin the AVERAGED Robin condition for the metric, instead of the "
                        "value pins above: rho d_rho <h_rr> + 3(<h_rr>-1) = 0 and "
                        "rho d_rho <g2> + 2(<g2>-1) = 0, order 1 at each quantity's own "
                        "leading decay power (measured 3 and 2; NOT the pointwise condition's "
                        "base 2).  Needs lam_inf and NO reference.  The means are what is "
                        "pinned -- the pointwise order-3 condition is untouched and is what "
                        "constrains the angular content")
    p.add_argument("--pin-far", action="store_true",
                   help="all three far-field pins (lam mean, h_tan, h_rr)")
    p.add_argument("--w-pin", type=float, default=None,
                   help="weight of the far-field pin group (independent of --w-outer)")
    p.add_argument("--w-lam-eq-radial", type=float, default=None, dest="w_lam_eq_radial",
                   help="weight of the rho-scaled RADIAL DERIVATIVE of the lambda equation "
                        "(0, the default, drops the term and its cost entirely).  The "
                        "lambda-equation is second order, so without this the loss never "
                        "sees lambda''', which is where the order-3 Robin condition can hide "
                        "a wrong far-field level")
    p.add_argument("--inner-radius", type=float, default=None)
    p.add_argument("--lam-bc-S1", type=float, default=None, dest="lam_bc_S1")
    p.add_argument("--lam-bc-S2", type=float, default=None, dest="lam_bc_S2")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-inner-h-rr", action="store_true")
    p.add_argument("--ref-solution", action="store_true")
    p.add_argument("--decay-feature", action="store_true")
    p.add_argument("--robin-source", action="store_true")
    p.add_argument("--no-robin-source", action="store_true",
                   help="drop the manufactured Robin source, so the outer condition imposes "
                        "DECAY instead of matching the exact operator.  The loss then has no "
                        "zero whenever the exact solution violates that condition at rho_out "
                        "(it does by 6.3e-02 in lambda for the closest Weyl shell), so this is "
                        "the honest no-exact-solution-available case, not a converged one")
    p.add_argument("--ref-asymptotic", type=float, default=None,
                   help="build the diagnostic reference with this asymptotic lambda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--init-from", type=str, default=None)
    p.add_argument("--ckpt-every", type=int, default=None, dest="ckpt_every",
                   help="write a resumable checkpoint every N Adam steps (0 = off)")
    p.add_argument("--resume", type=str, default=None,
                   help="checkpoint to continue from, or 'auto' for <outdir>/ckpt.pkl")
    p.add_argument("--pde-ramp-steps", type=int, default=None)
    p.add_argument("--arch", type=str, default=None)
    p.add_argument("--radial", type=str, default=None)
    p.add_argument("--scale-ref", type=float, default=None)
    p.add_argument("--scale-ref-rho-in", action="store_true")
    p.add_argument("--scale-exps", type=str, default=None)
    p.add_argument("--reweight-every", type=int, default=None)
    p.add_argument("--reweight-band", type=float, default=None,
                   help="how far a PDE weight may drift from its configured value")
    p.add_argument("--w-outer", type=float, default=None)
    p.add_argument("--w-inner", type=float, default=None)
    p.add_argument("--inner-bc", type=str, default=None, choices=("spherical", "reference"),
                   help="inner data: the round sphere + polynomial lambda, or the exact "
                        "reference (for a manufactured run whose solution is neither)")
    p.add_argument("--gauge-source", type=str, default=None,
                   choices=("none", "cylindrical"),
                   help="harmonic gauge, or the cylindrical source a chart adapted to an "
                        "axisymmetric solution carries (built from the candidate's metric)")
    p.add_argument("--weyl", action="store_true",
                   help="manufactured run on the Weyl two-black-hole reference; implies "
                        "--inner-bc reference --gauge-source cylindrical --outer-bc robin "
                        "--robin-source --ref-solution, and a rho_in that clears the rods")
    p.add_argument("--weyl-half-length", type=float, default=None, dest="weyl_half_length",
                   help="mass of the upper black hole (the rod length is twice it)")
    p.add_argument("--weyl-half-length-b", type=float, default=None,
                   dest="weyl_half_length_b",
                   help="mass of the lower black hole; omit for equal masses.  Unequal "
                        "masses break z -> -z: the fields gain a dipole, the equations do "
                        "not change")
    p.add_argument("--weyl-half-gap", type=float, default=None, dest="weyl_half_gap")
    p.add_argument("--weyl-n-quad", type=int, default=None, dest="weyl_n_quad")
    p.add_argument("--weyl-rotate-deg", type=float, default=None, dest="weyl_rotate_deg",
                   help="rotate the configuration by this angle in the z-x plane (about +y). "
                        "The inner data and the gauge source rotate with it, and the solution "
                        "loses every symmetry, so this needs the 3-D ansatz (--weyl picks it)")
    a = p.parse_args(argv)
    cfg = Config()
    if a.steps is not None:
        cfg.steps = a.steps
    if a.lbfgs_steps is not None:
        cfg.lbfgs_steps = a.lbfgs_steps
    if a.qn_method is not None:
        cfg.qn_method = a.qn_method
    if a.qn_max_H_gb is not None:
        cfg.qn_max_H_gb = a.qn_max_H_gb
    if a.qn_block is not None:
        cfg.qn_block = a.qn_block
    if a.vtk:
        cfg.make_vtk = True
    if a.vtk_n_half is not None:
        cfg.vtk_n_half = a.vtk_n_half
    if a.vtk_physical_inner is not None:
        cfg.vtk_physical_inner = a.vtk_physical_inner
    if a.plateau_tol is not None:
        cfg.plateau_tol = a.plateau_tol
    if a.plateau_patience is not None:
        cfg.plateau_patience = a.plateau_patience
    if a.plateau_min_iters is not None:
        cfg.plateau_min_iters = a.plateau_min_iters
    if a.qn_gtol is not None:
        cfg.qn_gtol = a.qn_gtol
    if a.outdir is not None:
        cfg.outdir = a.outdir
    if a.R0 is not None:
        cfg.R0 = a.R0
    if a.lam0 is not None:
        # an explicit lambda_0 wins over the derived k = 1 value: say so, or the final
        # __post_init__() would recompute it
        cfg.lam0 = a.lam0
        cfg.lam0_auto = False
    if a.rho_out is not None:
        cfg.rho_out = a.rho_out
    if a.n_coll is not None:
        cfg.n_coll = a.n_coll
    if a.n_bnd is not None:
        cfg.n_bnd = a.n_bnd
    if a.n_bnd_outer is not None:
        cfg.n_bnd_outer = int(a.n_bnd_outer)
    if a.width is not None:
        cfg.width = a.width
    if a.depth is not None:
        cfg.depth = a.depth
    if a.fourier is not None:
        cfg.fourier = a.fourier
    if a.outer_bc is not None:
        cfg.outer_bc = a.outer_bc
    if a.rho_in is not None:
        cfg.rho_in = float(a.rho_in)
    if a.lam_inf is not None:
        cfg.lam_inf = float(a.lam_inf)
    if a.lam_inf_init is not None:
        cfg.lam_inf_init = float(a.lam_inf_init)
    if a.robin_order is not None:
        cfg.robin_order = a.robin_order
    if a.robin_orders is not None:
        cfg.robin_orders = {k: int(v) for k, v in
                            (kv.split("=") for kv in a.robin_orders.split(","))}
    if a.no_robin_G:
        cfg.robin_include_G = False
    if a.inner_radius is not None:
        cfg.inner_radius = a.inner_radius
    if a.lam_bc_S1 is not None:
        cfg.lam_bc_S1 = a.lam_bc_S1
    if a.lam_bc_S2 is not None:
        cfg.lam_bc_S2 = a.lam_bc_S2
    if a.no_figures:
        cfg.make_figures = False
    if a.robin_exps is not None:
        vals = [float(v) for v in a.robin_exps.split(",")]
        cfg.robin_exps = dict(zip(["h", "G", "lam"], vals))
    if a.pin_far:
        cfg.pin_lam = cfg.pin_h_tan = cfg.pin_h_rr = True
    if a.pin_lam:
        cfg.pin_lam = True
    if a.pin_lam_robin:
        cfg.pin_lam_robin = True
    if a.pin_h_tan:
        cfg.pin_h_tan = True
    if a.pin_h_rr:
        cfg.pin_h_rr = True
    if a.pin_h_robin:
        cfg.pin_h_robin = True
    if a.h_robin_bases is not None:
        cfg.h_robin_bases = {k: float(v) for k, v in
                             (kv.split("=") for kv in a.h_robin_bases.split(","))}
    if a.log_resources:
        cfg.log_resources = True
    if a.w_pin is not None:
        cfg.w_pin = float(a.w_pin)
    if a.w_lam_eq_radial is not None:
        cfg.w_lam_eq_radial = float(a.w_lam_eq_radial)
    if a.seed is not None:
        cfg.seed = a.seed
    if getattr(a, "init_from", None) is not None:
        cfg.init_from = a.init_from
    if getattr(a, "ckpt_every", None) is not None:
        cfg.ckpt_every = a.ckpt_every
    if getattr(a, "resume", None) is not None:
        cfg.resume = a.resume
    if a.pde_ramp_steps is not None:
        cfg.pde_ramp_steps = a.pde_ramp_steps
    if a.arch is not None:
        cfg.arch = a.arch
    if a.reweight_every is not None:
        cfg.reweight_every = a.reweight_every
    if a.resample_every is not None:
        # This was MISSING: the flag existed, the Config field existed, and nothing ever set
        # one from the other -- so --resample-every was inert in BOTH phases, and the Adam
        # phase's resampling never fired either.
        cfg.resample_every = a.resample_every
    if a.reweight_band is not None:
        cfg.reweight_band = a.reweight_band
    if a.scale_ref is not None:
        cfg.scale_ref = a.scale_ref
    if a.scale_exps is not None:
        vals = [float(v) for v in a.scale_exps.split(",")]
        cfg.scale_exps = dict(zip(["compat", "ricci", "gauge", "lam_eq"], vals))
    if a.radial is not None:
        cfg.radial = a.radial
    if a.w_outer is not None:
        cfg.w_outer = a.w_outer
    if a.w_inner is not None:
        cfg.w_inner = a.w_inner
    if a.no_inner_h_rr:
        cfg.inner_h_rr = None
    if a.ref_solution:
        cfg.ref_solution = True
    if a.decay_feature:
        cfg.decay_feature = True
    if a.robin_source:
        cfg.robin_source = True
    if a.ref_asymptotic is not None:
        cfg.ref_asymptotic = a.ref_asymptotic
    if a.inner_bc is not None:
        cfg.inner_bc = a.inner_bc
    if a.gauge_source is not None:
        cfg.gauge_source = a.gauge_source
    if a.weyl_half_length is not None:
        cfg.weyl_half_length = float(a.weyl_half_length)
    if a.weyl_half_length_b is not None:
        cfg.weyl_half_length_b = float(a.weyl_half_length_b)
    if a.weyl_half_gap is not None:
        cfg.weyl_half_gap = float(a.weyl_half_gap)
    if a.weyl_n_quad is not None:
        cfg.weyl_n_quad = a.weyl_n_quad
    if a.weyl_rotate_deg is not None:
        cfg.weyl_rotate_deg = float(a.weyl_rotate_deg)
    if a.weyl:
        # The reference IS the solution here, so the inner data, the Robin source and the
        # comparison asset all come from it; the only things left to choose are the shell
        # (which must clear the rods -- they reach rho = half_gap + 2*half_length) and how
        # far out to put the outer sphere.  Nothing about the network is special-cased.
        cfg.weyl = True
        cfg.ref_solution = True
        cfg.robin_source = True
        cfg.inner_bc = "reference"
        cfg.gauge_source = "cylindrical"
        if a.arch is None and cfg.weyl_rotate_deg:
            # rotated: no symmetry left, so the general 3-D ansatz is the only correct choice
            cfg.arch = "mlp"
        if a.arch is None:
            # A spherically symmetric ansatz has no angular freedom at all, so it cannot
            # represent a two-black-hole field: it would converge to a compromise and its
            # numbers would mean nothing.  The Weyl field is axisymmetric (rods on the z
            # axis), and AxisymHybridNet's h_ij = (1+a) d_ij + b n_i n_j + c (n_i z_j +
            # z_i n_j) + d z_i z_j with a..d functions of (rho, mu) represents every
            # axisymmetric field content -- with five functions of two variables instead
            # of twenty-five of three, which the optimiser notices.
            cfg.arch = "axisym_hybrid"
        if a.outer_bc is None:
            cfg.outer_bc = "robin"
        if a.rho_in is None:
            # clear the LARGER of the two rods
            biggest = max(cfg.weyl_half_length,
                          cfg.weyl_half_length_b or cfg.weyl_half_length)
            cfg.rho_in = 3.0 * (cfg.weyl_half_gap + 2.0 * biggest)
        if a.rho_out is None:
            cfg.rho_out = 10.0 * cfg.rho_in
        if a.lam_inf is None:
            cfg.lam_inf = 1.0            # lambda -> 1 at infinity for Weyl
    if a.no_robin_source:
        # after the --weyl block, which sets it on: this is the way to ask for the run without
        cfg.robin_source = False
    cfg.__post_init__()
    if a.scale_ref_rho_in:
        cfg.scale_ref = cfg.rho_in
    return cfg


if __name__ == "__main__":
    a = parse_args()
    train(a, init_from=a.init_from, resume=a.resume)
