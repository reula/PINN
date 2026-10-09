"""The driver: build the problem, run the optimiser, measure, save artefacts."""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np

from .config import Config, prepare_outdir, resolve_outdir
from .evaluate import (error_metrics, plot_error_map, plot_history, plot_solution,
                       solution_on_grid)
from .features import feature_dim
from .losses import Objective
from .model import ansatz_u, init_params, n_parameters
from .optim import run_optimizer
from .sampling import make_batch


def configure_jax(cfg: Config) -> None:
    """Must run before any JAX computation (x64 changes every dtype)."""
    jax.config.update("jax_enable_x64", cfg.precision == "float64")


def build(cfg: Config):
    """``(params, objective, cfg)`` -- a fresh run, or a warm start when ``init_from`` is set.

    ``cfg`` is returned because ``n_coll = 0`` means "one collocation point per
    trainable parameter", which can only be resolved once the network exists.
    """
    key = jax.random.PRNGKey(cfg.seed)
    k_init, k_batch = jax.random.split(key)
    params = init_params(cfg, k_init)
    n_par = n_parameters(params)
    if cfg.n_coll <= 0:
        # One residual per parameter: below that the residual is under-determined
        # and the optimiser can drive the sampled loss to zero while the solution
        # drifts; far above it, each iteration costs more for little extra signal.
        cfg = cfg.replace(n_coll=n_par)
    batch = make_batch(cfg, k_batch)

    if cfg.init_from:
        path = cfg.init_from
        if os.path.isdir(path):
            path = os.path.join(path, "theta.npy")
        flat = jnp.asarray(np.load(path), dtype=jnp.float64 if cfg.precision == "float64"
                           else jnp.float32)
        template = Objective(cfg, batch, params)
        if int(flat.size) != template.n:
            raise ValueError(
                f"init_from {path} holds {flat.size} parameters but this configuration gives "
                f"{template.n}: a warm start needs the same architecture *and* feature map")
        params = template.unflatten(flat)
    return params, Objective(cfg, batch, params), cfg


def _other_objective(cfg: Config, like_params, seed_offset: int) -> Objective:
    key = jax.random.PRNGKey(cfg.seed + seed_offset)
    return Objective(cfg, make_batch(cfg, key), like_params)


def run(cfg: Config, verbose: bool = True, save: bool = True) -> Dict:
    """Run one experiment end to end and return a result dictionary."""
    cfg.validate()
    configure_jax(cfg)
    outdir = resolve_outdir(cfg)
    if save:
        archived = prepare_outdir(cfg, outdir)
        if archived and verbose:
            print(f"[cfg] {outdir} held an earlier run; moved it to {archived}", flush=True)

    params, objective, cfg = build(cfg)
    flat0, _ = jax.flatten_util.ravel_pytree(params)
    n_par = n_parameters(params)

    if verbose:
        print(f"[cfg] {cfg.equation}:  u_tt - c^2 u_xx = 0" if cfg.equation == "wave2"
              else f"[cfg] {cfg.equation}:  u_t + c u_x = 0")
        print(f"[cfg] L={cfg.L} c={cfg.c} T={cfg.T}  u0={cfg.u0}(sigma={cfg.u0_sigma})  "
              f"periodic={cfg.periodic}")
        print(f"[cfg] features={cfg.features} modes={cfg.n_modes} -> d_in={feature_dim(cfg)}; "
              f"MLP {cfg.n_layers}x{cfg.n_neurons} {cfg.activation} = {n_par} parameters")
        print(f"[cfg] optimiser={cfg.optimizer} steps={cfg.steps} qn_steps={cfg.qn_steps} "
              f"qn_block={cfg.qn_block} n_coll={cfg.n_coll} sampler={cfg.sampler} "
              f"precision={cfg.precision}")
        if cfg.resample_every:
            from .config import resample_schedule
            per = {"adam": cfg.steps, "ssbroyden": cfg.qn_steps, "dsgnar": cfg.dsgnar_steps,
                   "trustregion": cfg.tr_maxiter}
            plan = ", ".join(f"{k} from {resample_schedule(cfg, v)[0]:.0f} x{growth:g}"
                             for k, v in per.items() if k in cfg.optimizer and v
                             for growth in [resample_schedule(cfg, v)[1]])
            print(f"[cfg] redrawing the {cfg.n_coll} collocation points ({plan}; "
                  f"geometric, at least {cfg.min_resamples} redraws per phase)")
        else:
            print("[cfg] collocation points are FROZEN (resample_every=0)")

    # ---- collocation redraws ----------------------------------------------------
    # The residual is estimated on a finite sample.  Left frozen, the optimiser
    # eventually fits *that* sample: measured on this problem, continuing a
    # converged solution on a 512-point batch drove the sampled loss from 1.8e-9 to
    # 5.6e-10 while the error against the exact solution rose from 1.0e-4 to 2.3e-4.
    # Redrawing keeps the sampled residual an unbiased estimate of the true one.
    resample_counter = {"i": 0}

    def resample() -> Objective:
        resample_counter["i"] += 1
        k = jax.random.PRNGKey((int(cfg.seed) + 7919 * resample_counter["i"]) % (2 ** 31 - 1))
        return objective.with_batch(make_batch(cfg, k))

    resample_fn = resample if cfg.resample_every else None

    # ---- optional probe: residual on an independent sample, and the true error ---
    # Both matter and they answer different questions.  The independent residual
    # says whether the optimisation is still finding real structure or has started
    # fitting the collocation sample; the error against the exact solution says
    # whether any of that reaches the answer.  A frozen collocation batch makes the
    # first capable of falling while the second stands still, so the two are
    # recorded together.
    test_obj = None
    test_history: List[Dict] = []
    probe_every = int(cfg.eval_every) if cfg.eval_every and cfg.eval_every > 0 else 0
    probe_state = {"last": -10 ** 18}

    if probe_every:
        test_obj = _other_objective(cfg, params, seed_offset=10_000)
        if verbose:
            print(f"[cfg] probing every {probe_every} steps: loss on an independent "
                  f"sample, and the relative L2 error against the exact solution")

    def callback(step: int, loss: float, flat) -> None:
        if test_obj is None or step - probe_state["last"] < probe_every:
            return
        probe_state["last"] = step
        test_history.append({
            "step": int(step),
            "train_loss": float(loss),
            "test_loss": test_obj.loss(flat),
            "rel_l2": error_metrics(solution_on_grid(
                lambda tt, xx: ansatz_u(objective.unflatten(flat), cfg, tt, xx), cfg))
                ["rel_l2_space_time"],
        })

    t_start = time.time()
    flat, history, infos, phase_histories = run_optimizer(
        objective, flat0, cfg, verbose=verbose, callback=callback, resample=resample_fn)
    wall = time.time() - t_start

    params = objective.unflatten(flat)

    # ---- evaluation ------------------------------------------------------------
    grid = solution_on_grid(lambda tt, xx: ansatz_u(params, cfg, tt, xx), cfg)
    metrics = error_metrics(grid)
    if verbose:
        print(f"[eval] space-time relative L2 = {metrics['rel_l2_space_time']:.6e}   "
              f"final-time rel L2 = {metrics['final_time_rel_l2']:.6e}")
        for t in grid["times"]:
            m = metrics["per_time"][float(t)]
            print(f"[eval]   t={float(t):4.2f}   rel L2 = {m['rel_l2']:.6e}   "
                  f"max|err| = {m['linf']:.6e}")

    result = {
        "config": cfg.to_dict(),
        "outdir": outdir,
        "n_parameters": n_par,
        "final_loss": float(history[-1]["loss"]) if history else float("nan"),
        "history": history,
        "phase_histories": phase_histories,
        "phase_infos": infos,
        "metrics": metrics,
        "wall": wall,
        "test_history": test_history,
    }

    if save:
        _save(cfg, outdir, result, grid, flat, metrics, history, phase_histories, infos, wall)
    return result


def _save(cfg, outdir, result, grid, flat, metrics, history, phase_histories, infos, wall) -> None:
    cfg.to_json(os.path.join(outdir, "config.json"))
    np.save(os.path.join(outdir, "theta.npy"), np.asarray(flat))
    with open(os.path.join(outdir, "history.json"), "w") as fh:
        json.dump({"history": history,
                   "phase_histories": phase_histories,
                   "phase_infos": infos,
                   "test_history": result["test_history"],
                   "metrics": metrics,
                   "n_parameters": result["n_parameters"],
                   "final_loss": result["final_loss"],
                   "wall": wall}, fh, indent=2)
    np.savez_compressed(
        os.path.join(outdir, "fields.npz"),
        x=grid["x"],
        times=np.asarray(grid["times"]),
        u=np.stack([grid["u"][float(t)] for t in grid["times"]]),
        exact=np.stack([grid["exact"][float(t)] for t in grid["times"]]),
        abs_err=np.stack([grid["abs_err"][float(t)] for t in grid["times"]]),
    )
    eq = "u_tt - c^2 u_xx = 0" if cfg.equation == "wave2" else "u_t + c u_x = 0"
    plot_solution(grid, os.path.join(outdir, "solution.png"),
                  title=f"{cfg.label}: {eq},  {cfg.optimizer}")
    plot_error_map(grid, os.path.join(outdir, "error_map.png"))
    plot_history(history, os.path.join(outdir, "loss.png"), extra=phase_histories)

    with open(os.path.join(outdir, "report.md"), "w") as fh:
        fh.write(_report(cfg, result, metrics, infos, wall))


def _report(cfg, result, metrics, infos, wall) -> str:
    lines: List[str] = []
    A = lines.append
    A(f"# {cfg.label}\n")
    A(f"* equation: `{cfg.equation}` with `c = {cfg.c}`, domain `x in [{-cfg.L}, {cfg.L}]`, "
      f"`t in [0, {cfg.T}]`, periodic in x")
    A(f"* initial data: `u0 = {cfg.u0}(sigma={cfg.u0_sigma})`, `v0 = -c u0'`, hard-coded as "
      f"`u = u0 + t v0 + t^2 N`")
    A(f"* features: `{cfg.features}` with {cfg.n_modes} modes")
    A(f"* network: {cfg.n_layers} layers x {cfg.n_neurons} neurons, {cfg.activation}, "
      f"{result['n_parameters']} parameters")
    A(f"* collocation: {cfg.n_coll} points, sampler `{cfg.sampler}`")
    A(f"* optimiser: `{cfg.optimizer}`")
    A("")
    A(f"final training loss: **{result['final_loss']:.6e}**")
    A("")
    A(f"space-time relative L2 error: **{metrics['rel_l2_space_time']:.6e}**")
    A("")
    A("| t | rel L2 | max abs err |")
    A("|---|--------|-------------|")
    for t in metrics["per_time"]:
        m = metrics["per_time"][t]
        A(f"| {t:g} | {m['rel_l2']:.6e} | {m['linf']:.6e} |")
    A("")
    A(f"wall time: {wall:.1f} s")
    A("")
    for i, info in enumerate(infos):
        A(f"* phase {i+1}: `{info.get('optimizer')}` in {info.get('iterations')} iterations, "
          f"{info.get('wall', float('nan')):.1f} s, stopped: {info.get('stopped')}")
        rounds = info.get("rounds") or []
        if len(rounds) > 1:
            A("")
            A("  Converge, redraw, converge again.  `param change` is how far the round had to")
            A("  move the solution to fit the sample it was just handed, and `on next sample`")
            A("  is the loss of this round's solution on the *following* round's points --")
            A("  neither should grow.  A param change that does not fall towards the")
            A("  numerical floor means the solution is fitting the collocation points.")
            A("")
            A("  | round | iterations | loss (own sample) | loss on next sample | param change |")
            A("  |---|---|---|---|---|")
            for rec in rounds:
                nxt = rec.get("loss_on_next_sample")
                A(f"  | {rec['round']+1} | {rec['iterations']} | {rec['final_loss']:.3e} | "
                  f"{'--' if nxt is None else f'{nxt:.3e}'} | {rec['param_rel_change']:.2e} |")
            A("")
    return "\n".join(lines) + "\n"
