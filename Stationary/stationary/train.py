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
import time
from dataclasses import asdict

import jax
import jax.numpy as jnp
import optax

from . import diagnostics, exact
from .losses import (GROUP_KEYS, default_weights, group_terms, reference_consistency,
                     total_loss)
from .model import (AxisymHybridNet, FieldNet, HybridNet, SymFieldNet, SymHybridNet,
                    point_fields)
from .problem import Config, sample_shell, sample_sphere


def make_model(cfg: Config):
    cls = {"sym": SymFieldNet, "sym_hybrid": SymHybridNet, "hybrid": HybridNet,
           "axisym_hybrid": AxisymHybridNet}.get(cfg.arch, FieldNet)
    kw = dict(width=cfg.width, depth=cfg.depth, fourier=cfg.fourier,
              rho_in=cfg.rho_in, rho_out=cfg.rho_out)
    if cfg.arch in ("sym", "sym_hybrid", "axisym_hybrid"):
        kw["decay"] = cfg.decay_feature
    return cls(**kw)


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
    exact_fields = None
    if cfg.outer_bc == "dirichlet_exact":
        k = exact.k_from_lambda0(cfg.R0, cfg.lam0)
        exact_fields = exact.exact_fields(cfg.R0, k)
    elif cfg.ref_solution or cfg.robin_source:
        if getattr(cfg, "ref_asymptotic", None) is not None:
            exact_fields, _ = exact.reference_fields_asymptotic(
                cfg.R0, cfg.ref_asymptotic, cfg.rho_in, r_areal=cfg.inner_radius)
        else:
            exact_fields, _ = exact.reference_fields(cfg.R0, cfg.lam0, cfg.rho_in,
                                                     cfg.inner_h_rr or 1.0,
                                                     r_areal=cfg.inner_radius)

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
    if exact_fields is not None:
        chk = reference_consistency(exact_fields, cfg)
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
    return model, state, exact_fields


def make_batch(key, cfg: Config):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "coll": sample_shell(k1, cfg.n_coll, cfg),
        "inner": sample_sphere(k2, cfg.n_bnd, cfg.rho_in),
        "outer": sample_sphere(k3, cfg.n_bnd, cfg.rho_out),
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


def save_checkpoint(path, state, opt_state, weights, history, step, cfg):
    """Dump everything needed to continue an interrupted Adam phase.

    `weights` is included because the gradient-norm reweighting is path
    dependent: it multiplies the previous weights, so it cannot be recomputed
    from the step index alone.
    """
    payload = {
        "state": jax.tree.map(jax.device_get, state),
        "opt_state": jax.tree.map(jax.device_get, opt_state),
        "weights": jax.tree.map(jax.device_get, weights),
        "history": history,
        "step": step,
        "config": asdict(cfg),
    }
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh)
    os.replace(tmp, path)   # atomic: a kill mid-write cannot corrupt the previous checkpoint
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
    print("=" * 72)
    print("effective configuration (every value below is a flag; check them)")
    print(f"  model       {cfg.arch}   {cfg.width} x {cfg.depth}, fourier {cfg.fourier}"
          f"   (Gamma derived from h)")
    print(f"  exact data  R0 = {cfg.R0:g}   lambda_0 = {cfg.lam0:g}"
          f"   S1 = {cfg.lam_bc_S1:g}   S2 = {cfg.lam_bc_S2:g}")
    print(f"  domain      rho in [{cfg.rho_in:g}, {cfg.rho_out:g}]"
          f"   inner sphere areal radius {cfg.inner_radius:g}"
          f"   h_rr {cfg.inner_h_rr if cfg.inner_h_rr is not None else 'free'}")
    if cfg.outer_bc == "robin":
        print(f"  outer BC    robin   lambda_inf = "
              f"{cfg.lam_inf if cfg.lam_inf is not None else cfg.lam_inf_init} (learnable)"
              f"   orders {orders}   Gamma condition {cfg.robin_include_G}"
              f"   source {cfg.robin_source}")
    else:
        print(f"  outer BC    {cfg.outer_bc} (h and lambda from the exact solution)")
    print(f"  weights     w_inner {cfg.w_inner:g}   w_outer {cfg.w_outer:g}"
          f"   reweight every {cfg.reweight_every}   pde ramp {cfg.pde_ramp_steps}")
    print(f"  sampling    n_coll {cfg.n_coll}   n_bnd {cfg.n_bnd}   radial {cfg.radial}"
          f"   decay feature {cfg.decay_feature}")
    print(f"  plan        {cfg.steps} Adam + {cfg.lbfgs_steps} L-BFGS   outdir {cfg.outdir}")
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
    print(f"[ref]   reference values: M1 default (R0=1, areal radius 2, lam0=1) -> k = 2.618034;"
          f"   M2 control (R0=1/sqrt3, areal radius 1, lam0=1/3) -> k = 1")


def train(cfg: Config, verbose: bool = True, init_from: str | None = None,
          resume: str | None = None):
    os.makedirs(cfg.outdir, exist_ok=True)
    if verbose:
        print_config_summary(cfg)
    model, state, exact_fields = build(cfg, init_from)
    if verbose and exact_fields is not None:
        print_reference_summary(cfg, exact_fields)
    loss_fn = lambda st, batch, sc: total_loss(st, batch, cfg, model, exact_fields, pde_scale=sc)

    batch = make_batch(jax.random.PRNGKey(cfg.seed + 1), cfg)
    schedule = optax.cosine_decay_schedule(cfg.lr, max(cfg.steps, 1), alpha=0.01)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state = opt.init(state)

    weights = default_weights(cfg)
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
        # Warn rather than silently change the trajectory: the Adam LR schedule and
        # the resampling/reweighting cadences are all functions of the step index.
        prev = ck.get("config", {})
        for key in ("steps", "lr", "resample_every", "reweight_every", "pde_ramp_steps"):
            if key in prev and prev[key] != getattr(cfg, key):
                print(f"[resume] WARNING: {key}={getattr(cfg, key)!r} differs from the "
                      f"checkpoint's {prev[key]!r}; this will not reproduce the original "
                      f"run", flush=True)
        if resume_step >= cfg.steps:
            print(f"[resume] {rpath} is already at step {resume_step} "
                  f"(steps={cfg.steps}); Adam phase complete", flush=True)
        else:
            print(f"[resume] continuing from {rpath} at step {resume_step}", flush=True)
        batch = batch_for_step(cfg, max(resume_step + 1, 1))

    loss = None
    t0 = time.time()
    for it in range(resume_step + 1, cfg.steps + 1):
        sc = 1.0 if cfg.pde_ramp_steps <= 0 else min(1.0, it / cfg.pde_ramp_steps)
        if cfg.resample_every and it % cfg.resample_every == 1 and it > 1:
            batch = make_batch(jax.random.PRNGKey(cfg.seed + it), cfg)
        if (cfg.reweight_every and cfg.reweight_every > 0 and it > 1
                and it % cfg.reweight_every == 0):
            g = group_gradnorms(state, batch)
            # Reweight ONLY the interior equation groups.  The boundary terms are data,
            # and w ~ 1/||grad term|| is backwards for them: the condition that is most
            # violated has the largest gradient and so receives the SMALLEST weight --
            # which is how an earlier run silenced its outer Robin condition (w_outer
            # fell to 6.25 while w_inner rose to 229), let lambda stay at its inner
            # value and drift onto the trivial flat branch.
            pde_keys = ("compat", "ricci", "gauge", "lam_eq")
            target = jnp.mean(jnp.stack([g[i] for i, k in enumerate(GROUP_KEYS)
                                         if k in pde_keys]))
            w_new = {}
            for i, k in enumerate(GROUP_KEYS):
                if k not in pde_keys:
                    w_new[k] = weights[k]           # fixed: cfg.w_inner / cfg.w_outer
                    continue
                ideal = target / (g[i] + 1e-300)
                ratio = jnp.clip(ideal / weights[k], cfg.reweight_max_ratio_inv,
                                 1.0 / cfg.reweight_max_ratio_inv)
                w_new[k] = weights[k] * jnp.sqrt(ratio)
            weights = w_new
            if verbose:
                print(f"[reweight {it}] " + " ".join(
                    f"{k}={float(weights[k]):.3g}" for k in GROUP_KEYS), flush=True)
        state, opt_state, loss, parts = step(state, opt_state, batch, jnp.asarray(sc), weights)
        if it % cfg.log_every == 0 or it == 1:
            rec = {"step": it, "loss": float(loss),
                   **{k: float(v) for k, v in parts.items()}}
            history.append(rec)
            if verbose:
                print(f"[adam {it:6d}] loss={float(loss):.4e}  "
                      f"pde(compat={float(parts['pde_compat']):.2e} "
                      f"ricci={float(parts['pde_ricci']):.2e} "
                      f"gauge={float(parts['pde_gauge']):.2e} "
                      f"lam={float(parts['pde_lam_eq']):.2e})  "
                      f"bc_in={float(sum(v for k, v in parts.items() if k.startswith('inner_'))):.2e} "
                      f"bc_out={float(sum(parts[k] for k in parts if k.startswith('outer_'))):.2e}",
                      flush=True)
        if cfg.ckpt_every and it % cfg.ckpt_every == 0:
            save_checkpoint(os.path.join(cfg.outdir, CKPT_NAME), state, opt_state,
                            weights, history, it, cfg)
            if verbose:
                print(f"[ckpt  {it:6d}] wrote {os.path.join(cfg.outdir, CKPT_NAME)}",
                      flush=True)

    if loss is None:
        # The Adam phase ran zero iterations (--steps 0, or a resume already at the
        # end of the schedule): evaluate once so the report still carries a loss.
        _, _, loss, parts = step(state, opt_state, batch, jnp.asarray(1.0), weights)

    # ------------------------------------------------------------------ L-BFGS
    with open(os.path.join(cfg.outdir, "params_adam.pkl"), "wb") as fh:
        pickle.dump(jax.tree.map(lambda a: jax.device_get(a), state), fh)

    if cfg.lbfgs_steps > 0:
        batch = make_batch(jax.random.PRNGKey(cfg.seed + 777), cfg)
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
              "steps": cfg.steps, "lbfgs_steps": cfg.lbfgs_steps}
    report.update(diagnostics.residual_report(pf, cfg))
    report.update(diagnostics.inner_boundary_report(pf, cfg))
    if exact_fields is not None:
        report.update(diagnostics.exact_comparison(pf, exact_fields, cfg))
    if "lam_inf" in state:
        report["lam_inf"] = float(state["lam_inf"])
    if cfg.make_figures:
        try:
            from .multipoles import make_figures
            report["figures"] = make_figures(pf, cfg, cfg.outdir)
        except Exception as exc:                      # plotting must never kill a run
            print(f"[figures] failed: {exc}", flush=True)

    with open(os.path.join(cfg.outdir, "config.json"), "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)
    with open(os.path.join(cfg.outdir, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    with open(os.path.join(cfg.outdir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(cfg.outdir, "params.pkl"), "wb") as fh:
        pickle.dump(jax.tree.map(lambda a: jax.device_get(a), state), fh)

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
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--R0", type=float, default=None)
    p.add_argument("--lam0", type=float, default=None)
    p.add_argument("--rho-out", type=float, default=None)
    p.add_argument("--n-coll", type=int, default=None)
    p.add_argument("--n-bnd", type=int, default=None)
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
    p.add_argument("--inner-radius", type=float, default=None)
    p.add_argument("--lam-bc-S1", type=float, default=None, dest="lam_bc_S1")
    p.add_argument("--lam-bc-S2", type=float, default=None, dest="lam_bc_S2")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-inner-h-rr", action="store_true")
    p.add_argument("--ref-solution", action="store_true")
    p.add_argument("--decay-feature", action="store_true")
    p.add_argument("--robin-source", action="store_true")
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
    p.add_argument("--w-outer", type=float, default=None)
    p.add_argument("--w-inner", type=float, default=None)
    a = p.parse_args(argv)
    cfg = Config()
    if a.steps is not None:
        cfg.steps = a.steps
    if a.lbfgs_steps is not None:
        cfg.lbfgs_steps = a.lbfgs_steps
    if a.outdir is not None:
        cfg.outdir = a.outdir
    if a.R0 is not None:
        cfg.R0 = a.R0
    if a.lam0 is not None:
        cfg.lam0 = a.lam0
    if a.rho_out is not None:
        cfg.rho_out = a.rho_out
    if a.n_coll is not None:
        cfg.n_coll = a.n_coll
    if a.n_bnd is not None:
        cfg.n_bnd = a.n_bnd
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
    cfg.__post_init__()
    if a.scale_ref_rho_in:
        cfg.scale_ref = cfg.rho_in
    return cfg


if __name__ == "__main__":
    a = parse_args()
    train(a, init_from=a.init_from, resume=a.resume)
