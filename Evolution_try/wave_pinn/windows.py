"""Slab marching in time: solve ``[0, T]`` window by window.

Why this exists
---------------
A single global solve on ``[0, T]`` with the hard-constrained ansatz fails as soon
as ``T`` grows, and §6.6 of the README measures why: the hard-coded part is the
first two terms of a Taylor expansion in ``t``, ``u0 + t v0``, whose size grows
linearly -- 61 at ``T = 20`` -- so the network's ``t^2 N`` has to cancel a large
number with a small one.  Worse, with ``n_coll = n_parameters`` the sampled
residual system is square and therefore exactly solvable whether or not the PDE
has been solved, so the loss falls while the error grows.

Marching removes the first problem at its root.  The chain is

    [t_0, t_1], [t_1, t_2], ..., [t_{W-1}, t_W],

and on window ``k`` the ansatz is built from the *previous window's own solution*
rather than from the initial data:

    u(t, x) = U_k(x) + (t - t_k) V_k(x) + factor(t - t_k) * N_theta(t - t_k, x),

    U_k(x) = u_{k-1}(t_k, x),   V_k(x) = d_t u_{k-1}(t_k, x),

with ``U_k`` and ``V_k`` **frozen** -- they enter the residual as data.  On the
first window they are the exact ``u0`` and ``v0``.  Each window is then a short
problem in which the hard part only has to cancel ``dt * V_k``, and every window
reuses the same optimiser code, unchanged: the driver below is a loop around
:func:`wave_pinn.optim.run_optimizer`.

The network sees the *local* time ``t - t_k``, so the function it has to learn does
not change character along the chain.

What is and is not continuous
-----------------------------
``u`` and ``u_t`` are continuous across window edges by construction, because the
next window starts from the previous one's value and slope.  The second derivative
is not: each window has its own network, so ``N`` jumps at the edges, and the
residual is only minimised window by window.  That is the price of marching, and
it is why the windows in the report are evaluated on the interior of each slab as
well as at the edges.

Usage
-----
    python -m wave_pinn.cli --set windows=10 --set T=20 --set optimizer=dsgnar ...
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np

from .config import Config, resolve_outdir
from .evaluate import (error_metrics, plot_error_map, plot_history, plot_solution,
                       solution_on_grid)
from .features import feature_dim
from .losses import Objective
from .model import ansatz_u, init_params, n_parameters
from .optim import run_optimizer
from .problem import initial_data
from .sampling import make_batch
from .train import configure_jax


@dataclass
class Slab:
    """One time window and the network that covers it."""

    cfg: Config
    index: int
    t0: float
    t1: float
    prev: Optional["Slab"] = None
    params: Any = None
    metrics: Dict = field(default_factory=dict)

    # ---------------------------------------------------------------- the IC
    def ic(self, x):
        """``(U(x), V(x))``: the solution and its time derivative at ``self.t0``."""
        if self.prev is None:
            return initial_data(self.cfg, x)
        return self.prev.solution(self.t0, x), self.prev.time_derivative(self.t0, x)

    # ------------------------------------------------------------ evaluation
    def solution(self, t, x):
        """``u(t, x)`` on this window (``t`` may be a batch sharing one value)."""
        return ansatz_u(self.params, self.cfg, t, x, ic=self.ic, t0=self.t0)

    def time_derivative(self, t, x):
        """``d_t u(t, x)`` -- what seeds the next window."""
        def one(tt, xx):
            return jax.grad(
                lambda s: ansatz_u(self.params, self.cfg, s, xx, ic=self.ic, t0=self.t0))(tt)

        if jnp.ndim(x) == 0:
            return one(t, x)
        return jax.vmap(one)(jnp.full_like(x, t) if jnp.ndim(t) == 0 else t, x)


class Chain:
    """The piecewise solution assembled from a list of trained slabs."""

    def __init__(self, cfg: Config, slabs: List[Slab]):
        self.cfg = cfg
        self.slabs = slabs

    def slab_for(self, t: float) -> Slab:
        for s in self.slabs:
            if s.t0 <= t <= s.t1:
                return s
        return self.slabs[-1] if t > self.slabs[-1].t1 else self.slabs[0]

    def u(self, t, x):
        """``u(t, x)`` for a scalar time (the evaluation grid is one slice at a time)."""
        tv = float(np.asarray(t).reshape(-1)[0]) if jnp.ndim(t) else float(t)
        return self.slab_for(tv).solution(t, x)

    def __call__(self, t, x):
        return self.u(t, x)


# --------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------
def run_windows(cfg: Config, verbose: bool = True, save: bool = True) -> Dict:
    """Train one network per time window, chaining the initial data forward."""
    cfg.validate()
    configure_jax(cfg)
    if cfg.windows < 1:
        raise ValueError("windows must be >= 1")
    outdir = resolve_outdir(cfg)
    if save:
        os.makedirs(outdir, exist_ok=True)

    edges = np.linspace(0.0, cfg.T, cfg.windows + 1)
    slabs: List[Slab] = []
    window_records: List[Dict] = []
    t_start = time.time()

    for k in range(cfg.windows):
        slab = Slab(cfg, k, float(edges[k]), float(edges[k + 1]),
                    prev=(slabs[-1] if slabs else None))
        key = jax.random.PRNGKey(int(cfg.seed) + 101 * k)
        params = init_params(cfg, key)
        if cfg.init_from and k == 0:
            path = cfg.init_from
            if os.path.isdir(path):
                path = os.path.join(path, "theta.npy")
            params = Objective(cfg, make_batch(cfg, key, t0=slab.t0, t1=slab.t1, ic=slab.ic),
                               params, ic=slab.ic, t0=slab.t0).unflatten(
                jnp.asarray(np.load(path), dtype=jnp.float64))
        batch = make_batch(cfg, jax.random.PRNGKey(int(cfg.seed) + 7919 * (k + 1)),
                           t0=slab.t0, t1=slab.t1, ic=slab.ic)
        objective = Objective(cfg, batch, params, ic=slab.ic, t0=slab.t0)
        flat0, _ = jax.flatten_util.ravel_pytree(params)
        n_par = n_parameters(params)

        if verbose:
            print(f"\n[win {k+1}/{cfg.windows}] t in [{slab.t0:g}, {slab.t1:g}]  "
                  f"{n_par} parameters  dt = {slab.t1 - slab.t0:g}", flush=True)

        counter = {"i": 0}

        def resample(_k=k):
            counter["i"] += 1
            return objective.with_batch(make_batch(
                cfg, jax.random.PRNGKey((int(cfg.seed) + 7919 * (100 + counter["i"])) % (2 ** 31 - 1)),
                t0=slab.t0, t1=slab.t1, ic=slab.ic))

        flat, history, infos, phase_histories = run_optimizer(
            objective, flat0, cfg, verbose=verbose,
            resample=(resample if cfg.resample_every else None))
        slab.params = objective.unflatten(flat)
        slabs.append(slab)

        # error of this window alone, on its own interior
        n_in = max(3, min(11, cfg.windows * 3))
        times = np.linspace(slab.t0, slab.t1, n_in)[1:]
        grid = solution_on_grid(lambda tt, xx, s=slab: s.solution(tt, xx), cfg,
                                times=[float(x) for x in times], n_x=cfg.n_test_x)
        m = error_metrics(grid)
        slab.metrics = m
        window_records.append({
            "index": k, "t0": slab.t0, "t1": slab.t1,
            "final_loss": float(history[-1]["loss"]) if history else float("nan"),
            "rel_l2": m["rel_l2_space_time"],
            "wall": sum(float(i.get("wall") or 0.0) for i in infos),
            "phase_infos": infos,
        })
        if verbose:
            print(f"[win {k+1}/{cfg.windows}] loss {window_records[-1]['final_loss']:.3e}   "
                  f"window rel L2 {m['rel_l2_space_time']:.3e}", flush=True)

    chain = Chain(cfg, slabs)
    wall = time.time() - t_start
    grid = solution_on_grid(chain, cfg)
    metrics = error_metrics(grid)
    if verbose:
        print(f"\n[chain] space-time relative L2 = {metrics['rel_l2_space_time']:.6e}")
        for t in grid["times"]:
            mm = metrics["per_time"][float(t)]
            print(f"[chain]   t={float(t):5.1f}   rel L2 = {mm['rel_l2']:.6e}   "
                  f"max|err| = {mm['linf']:.6e}")

    result = {
        "config": cfg.to_dict(),
        "outdir": outdir,
        "n_parameters": n_parameters(slabs[0].params),
        "windows": window_records,
        "final_loss": window_records[-1]["final_loss"],
        "metrics": metrics,
        "wall": wall,
    }
    if save:
        _save(cfg, outdir, result, grid, slabs, window_records, metrics, wall)
    return result


def _save(cfg, outdir, result, grid, slabs, window_records, metrics, wall) -> None:
    cfg.to_json(os.path.join(outdir, "config.json"))
    np.save(os.path.join(outdir, "theta.npy"),
            np.asarray(jax.flatten_util.ravel_pytree(slabs[0].params)[0]))
    with open(os.path.join(outdir, "windows.json"), "w") as fh:
        json.dump({"windows": window_records, "metrics": metrics,
                   "n_parameters": result["n_parameters"], "wall": wall}, fh, indent=2)
    # per-window parameters, so the chain can be rebuilt without retraining
    for s in slabs:
        flat, _ = jax.flatten_util.ravel_pytree(s.params)
        np.save(os.path.join(outdir, f"theta_window{s.index}.npy"), np.asarray(flat))
    np.savez_compressed(
        os.path.join(outdir, "fields.npz"),
        x=grid["x"], times=np.asarray(grid["times"]),
        u=np.stack([grid["u"][float(t)] for t in grid["times"]]),
        exact=np.stack([grid["exact"][float(t)] for t in grid["times"]]),
        abs_err=np.stack([grid["abs_err"][float(t)] for t in grid["times"]]),
    )
    eq = "u_tt - c^2 u_xx = 0" if cfg.equation == "wave2" else "u_t + c u_x = 0"
    plot_solution(grid, os.path.join(outdir, "solution.png"),
                  title=f"{cfg.label}: {eq}, {cfg.optimizer}, {cfg.windows} windows")
    plot_error_map(grid, os.path.join(outdir, "error_map.png"))
    plot_history([{"step": r["index"], "loss": r["final_loss"]} for r in window_records],
                 os.path.join(outdir, "loss.png"), extra=None)
    with open(os.path.join(outdir, "report.md"), "w") as fh:
        fh.write(_report(cfg, result, metrics, window_records, wall))


def _report(cfg, result, metrics, window_records, wall) -> str:
    L: List[str] = []
    A = L.append
    A(f"# {cfg.label} (windowed)\n")
    A(f"* equation: `{cfg.equation}`, `c = {cfg.c}`, `x in [{-cfg.L}, {cfg.L}]`, "
      f"`t in [0, {cfg.T}]`, periodic in x")
    A(f"* ansatz: `{cfg.ansatz}` with the initial data of each window taken from the "
      f"previous window (`U = u_k-1(t_k)`, `V = d_t u_k-1(t_k)`, both frozen)")
    A(f"* windows: {cfg.windows} of `dt = {cfg.T / cfg.windows:g}`")
    A(f"* features: `{cfg.features}`; network {cfg.n_layers} x {cfg.n_neurons} per window")
    A(f"* optimiser: `{cfg.optimizer}`\n")
    A(f"space-time relative L2 error: **{metrics['rel_l2_space_time']:.6e}**\n")
    A("| t | rel L2 | max abs err |")
    A("|---|--------|-------------|")
    for t in metrics["per_time"]:
        m = metrics["per_time"][t]
        A(f"| {t:g} | {m['rel_l2']:.6e} | {m['linf']:.6e} |")
    A("")
    A("| window | t range | final loss | window rel L2 | wall (s) |")
    A("|---|---|---|---|---|")
    for r in window_records:
        A(f"| {r['index']+1} | [{r['t0']:g}, {r['t1']:g}] | {r['final_loss']:.3e} | "
          f"{r['rel_l2']:.3e} | {r['wall']:.0f} |")
    A("")
    A(f"wall time: {wall:.1f} s")
    return "\n".join(L) + "\n"
