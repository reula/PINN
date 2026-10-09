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

from .config import Config, prepare_outdir, resolve_outdir
from .evaluate import (error_metrics, plot_error_map, plot_history, plot_solution,
                       solution_on_grid)
from .features import feature_dim
from .losses import Objective
from .model import ansatz_u, init_params, n_parameters
from .optim import run_optimizer
from .problem import exact_solution, initial_data
from .sampling import make_batch
from .train import configure_jax


# --------------------------------------------------------------------------
# the edge representation: data, not a callable chain
# --------------------------------------------------------------------------
# Window k's ansatz is written relative to the solution at its start edge,
#     u(t, x) = U(x) + (t - t_k) V(x) + f(t - t_k) N_theta(t - t_k, x).
# If U and V were themselves network expressions -- U = u_{k-1}(t_k), and that
# window's own U, V came from window k-2 -- then evaluating window k would walk
# the whole chain: its residual graph would be k deep, cost O(k) per evaluation,
# and every derivative would be taken through k nested networks.
#
# They do not have to be.  The solution at an edge is a smooth 2L-periodic
# function of x, so it is stored as its Fourier coefficients and evaluated by
# spectral interpolation.  Each window then carries its own two coefficient
# arrays and nothing else: depth one, constant cost, and the only error the
# hand-over introduces is the truncation of a series that converges
# geometrically for these fields (the Gaussian's coefficients fall like
# exp(-0.2 m^2), so sixty modes are already at machine precision).
#
# It also makes a window independently evaluable, which is what the per-window
# theta_*.npy files were supposed to give and did not: reconstructing window 8
# used to require replaying windows 1 to 7.


def edge_grid(cfg: Config, n: Optional[int] = None):
    """Uniform periodic grid for sampling an edge (endpoint excluded)."""
    n = int(n or getattr(cfg, "ic_grid", 256))
    return jnp.linspace(-cfg.L, cfg.L, n, endpoint=False)


def _spectral_basis(cfg: Config, n: int, n_modes: int):
    """Integer wavenumbers to keep, and the angular frequencies e^{i omega x}."""
    k_int = jnp.fft.fftfreq(n, d=1.0 / n)
    keep = jnp.abs(k_int) <= int(n_modes)
    omega = jnp.where(keep, k_int * (jnp.pi / cfg.L), 0.0)
    return keep, omega


def to_spectral(cfg: Config, values, n_modes: Optional[int] = None):
    """Fourier coefficients of a periodic field sampled on :func:`edge_grid`."""
    n = int(values.shape[0])
    keep, _ = _spectral_basis(cfg, n, n_modes or getattr(cfg, "ic_modes", 64))
    return jnp.where(keep, jnp.fft.fft(values) / n, 0.0)


def from_spectral(cfg: Config, coeffs, x, n_modes: Optional[int] = None):
    """Evaluate the (truncated) Fourier series at arbitrary ``x``.

    The ``+ L`` is not cosmetic: the DFT takes its origin at the first sample, and
    :func:`edge_grid` starts at ``-L``.  Without the shift every retained mode is
    rotated by ``e^{i pi k}``, which is a sign flip per mode -- the interpolant
    then bears no resemblance to the field it came from.
    """
    n = int(coeffs.shape[0])
    _, omega = _spectral_basis(cfg, n, n_modes or getattr(cfg, "ic_modes", 64))
    xs = jnp.atleast_1d(x)
    phase = jnp.exp(1j * omega[:, None] * (xs + cfg.L)[None, :])
    out = jnp.real(coeffs @ phase)
    return out if jnp.ndim(x) else out[0]


@dataclass
class Slab:
    """One time window, its network, and the edge data it was handed."""

    cfg: Config
    index: int
    t0: float
    t1: float
    prev: Optional["Slab"] = None
    params: Any = None
    metrics: Dict = field(default_factory=dict)
    # the IC this window was given, as Fourier coefficients (None on window 1,
    # which uses the exact initial data)
    u_edge: Any = None
    v_edge: Any = None
    # its own solution at self.t1, materialised for the next window
    u_out: Any = None          # Fourier coefficients  (hard hand-over)
    v_out: Any = None
    u_out_vals: Any = None     # values on the edge grid (soft hand-over)
    v_out_vals: Any = None
    u_edge_vals: Any = None    # what this window was handed, as values
    v_edge_vals: Any = None

    @property
    def ansatz_mode(self) -> str:
        """``"net"`` for a soft hand-over, otherwise the configured ansatz.

        Window 1 always keeps the hard-coded *physical* initial condition; it is
        the artificial hand-overs that may be softened.
        """
        mode = getattr(self.cfg, "window_ic", "hard")
        if mode == "hard":
            return self.cfg.ansatz
        if self.index == 0 and mode != "soft_all":
            return self.cfg.ansatz          # window 1 keeps the exact physical IC
        return "net"

    @property
    def soft_ic(self):
        """``(x_edge, U, V, w)`` for the penalty, or None on the hard path.

        The targets are the *values* the previous window left on the edge grid.
        If only the Fourier coefficients are present -- a Slab built by hand, or
        one whose values were not kept -- they are recovered from them, so the
        soft path does not depend on the order in which a slab was assembled.
        """
        if self.ansatz_mode != "net":
            return None
        x = edge_grid(self.cfg)
        if self.index == 0:
            # "soft_all": even the first window enforces the physical initial data
            # as a penalty rather than building it into the ansatz.
            u, v = initial_data(self.cfg, x)
            return (x, u, v, float(getattr(self.cfg, "w_ic", 1.0e2)))
        u, v = self.u_edge_vals, self.v_edge_vals
        if u is None and self.u_edge is not None:
            u = from_spectral(self.cfg, self.u_edge, x)
        if v is None and self.v_edge is not None:
            v = from_spectral(self.cfg, self.v_edge, x)
        if u is None or v is None:
            raise ValueError(
                "window_ic='soft' needs the previous window's edge data; "
                "materialise_edge() has not run or the chain was not wired")
        return (x, u, v, float(getattr(self.cfg, "w_ic", 1.0e2)))

    @property
    def width(self) -> float:
        """The time interval this network operates over -- its natural t scale."""
        return float(self.t1 - self.t0)

    # ---------------------------------------------------------------- the IC
    def ic(self, x):
        """``(U(x), V(x))`` at ``self.t0``, from this window's stored edge data.

        Depth one: the two coefficient arrays are all this needs.  Window 1 has
        none and uses the exact initial data.
        """
        if self.u_edge is None or self.v_edge is None:
            return initial_data(self.cfg, x)
        return (from_spectral(self.cfg, self.u_edge, x),
                from_spectral(self.cfg, self.v_edge, x))

    def materialise_edge(self, x=None):
        """Sample this window's own solution at ``self.t1`` and store its coefficients.

        Called once, after the window is trained: this is the hand-over, and after
        it the next window depends on two arrays rather than on a network.
        """
        x = edge_grid(self.cfg) if x is None else x
        u = self.solution(jnp.full_like(x, self.t1), x)
        v = self.time_derivative(self.t1, x)
        self.u_out_vals = u
        self.v_out_vals = v
        self.u_out = to_spectral(self.cfg, u)
        self.v_out = to_spectral(self.cfg, v)
        return self.u_out, self.v_out

    # ------------------------------------------------------------ evaluation
    def solution(self, t, x):
        """``u(t, x)`` on this window (``t`` may be a batch sharing one value)."""
        return ansatz_u(self.params, self.cfg, t, x, ic=self.ic, t0=self.t0,
                        t_scale=self.width, ansatz=self.ansatz_mode)

    def time_derivative(self, t, x):
        """``d_t u(t, x)`` -- what seeds the next window."""
        def one(tt, xx):
            return jax.grad(
                lambda s: ansatz_u(self.params, self.cfg, s, xx, ic=self.ic,
                                    t0=self.t0, t_scale=self.width,
                                    ansatz=self.ansatz_mode))(tt)

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
        archived = prepare_outdir(cfg, outdir)
        if archived and verbose:
            print(f"[cfg] {outdir} held an earlier run; moved it to {archived}", flush=True)

    edges = np.linspace(0.0, cfg.T, cfg.windows + 1)
    slabs: List[Slab] = []
    window_records: List[Dict] = []
    t_start = time.time()

    for k in range(cfg.windows):
        prev = slabs[-1] if slabs else None
        slab = Slab(cfg, k, float(edges[k]), float(edges[k + 1]), prev=prev,
                    u_edge=(prev.u_out if prev is not None else None),
                    v_edge=(prev.v_out if prev is not None else None),
                    u_edge_vals=(prev.u_out_vals if prev is not None else None),
                    v_edge_vals=(prev.v_out_vals if prev is not None else None))
        key = jax.random.PRNGKey(int(cfg.seed) + 101 * k)
        params = init_params(cfg, key)
        if cfg.init_from and k == 0:
            path = cfg.init_from
            if os.path.isdir(path):
                path = os.path.join(path, "theta.npy")
            params = Objective(cfg, make_batch(cfg, key, t0=slab.t0, t1=slab.t1, ic=slab.ic),
                               params, ic=slab.ic, t0=slab.t0,
                               t_scale=slab.width, ansatz=slab.ansatz_mode,
                               soft_ic=slab.soft_ic).unflatten(
                jnp.asarray(np.load(path), dtype=jnp.float64))
        batch = make_batch(cfg, jax.random.PRNGKey(int(cfg.seed) + 7919 * (k + 1)),
                           t0=slab.t0, t1=slab.t1, ic=slab.ic)
        objective = Objective(cfg, batch, params, ic=slab.ic, t0=slab.t0,
                              t_scale=slab.width, ansatz=slab.ansatz_mode,
                              soft_ic=slab.soft_ic)
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
        slab.materialise_edge()          # hand-over as data, not as a network
        slabs.append(slab)

        # What the window inherited, against what it achieved.  The hand-over is an
        # initial-value problem, so a window's error is (at best) the error of the
        # solution it was handed at t0: measuring BOTH separates "this window solved
        # badly" from "this window solved a badly posed problem", which is otherwise
        # indistinguishable from the outside -- and it is the number that decides
        # whether to spend more iterations per window or fewer windows.
        xs = jnp.linspace(-cfg.L, cfg.L, cfg.n_test_x)
        u_in, _v_in = slab.ic(xs)
        ex_in = exact_solution(cfg, jnp.full_like(xs, slab.t0), xs)
        den_in = float(jnp.sqrt(jnp.mean(ex_in ** 2)))
        ic_rel = (float(jnp.sqrt(jnp.mean((u_in - ex_in) ** 2))) / den_in) if den_in > 0 else 0.0

        # error of this window alone, on its own interior
        n_in = max(3, min(11, cfg.windows * 3))
        times = np.linspace(slab.t0, slab.t1, n_in)[1:]
        grid = solution_on_grid(lambda tt, xx, s=slab: s.solution(tt, xx), cfg,
                                times=[float(x) for x in times], n_x=cfg.n_test_x)
        m = error_metrics(grid)
        slab.metrics = m
        # Same measure at both ends of the hand-over: inherited error at t0, error at
        # t1.  The space-time rel L2 above mixes times and is not comparable with it.
        end_rel = m["per_time"][float(times[-1])]["rel_l2"]
        window_records.append({
            "index": k, "t0": slab.t0, "t1": slab.t1,
            "final_loss": float(history[-1]["loss"]) if history else float("nan"),
            "ic_rel_l2": ic_rel,
            "rel_l2": m["rel_l2_space_time"],
            "rel_l2_end": end_rel,
            "amplification": (end_rel / ic_rel) if ic_rel > 0 else float("nan"),
            "wall": sum(float(i.get("wall") or 0.0) for i in infos),
            "resamples": sum(int(i.get("resamples") or 0) for i in infos),
            "phase_infos": infos,
        })
        if verbose:
            print(f"[win {k+1}/{cfg.windows}] loss {window_records[-1]['final_loss']:.3e}   "
                  f"inherited {ic_rel:.3e} -> end {end_rel:.3e} "
                  f"(x{window_records[-1]['amplification']:.2f})", flush=True)

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
    # Per-window parameters AND edge data.  With the edge coefficients saved,
    # window k can be evaluated from (params_k, u_edge_k, v_edge_k) alone -- no
    # replay of the chain -- which is the point of materialising it.
    for s in slabs:
        flat, _ = jax.flatten_util.ravel_pytree(s.params)
        np.save(os.path.join(outdir, f"theta_window{s.index}.npy"), np.asarray(flat))
        if s.u_edge is not None:
            np.savez(os.path.join(outdir, f"edge_window{s.index}.npz"),
                     u_edge=np.asarray(s.u_edge), v_edge=np.asarray(s.v_edge),
                     u_out=np.asarray(s.u_out), v_out=np.asarray(s.v_out))
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
    if getattr(cfg, "window_ic", "hard") == "soft":
        A(f"* ansatz: window 1 is `{cfg.ansatz}` on the exact initial data; later windows are a "
          f"plain network with a penalty (`w_ic = {cfg.w_ic:g}`) pulling `u` and `u_t` onto the "
          f"previous window's solution at the shared edge, so an inherited error can be corrected")
    else:
        A(f"* ansatz: `{cfg.ansatz}` with each window's initial data frozen into the ansatz "
          f"(`U = u_k-1(t_k)`, `V = d_t u_k-1(t_k)`, stored as Fourier coefficients)")
    A(f"* windows: {cfg.windows} of `dt = {cfg.T / cfg.windows:g}`, hand-over `{cfg.window_ic}`"
      + (f" (w_ic = {cfg.w_ic:g})" if cfg.window_ic == "soft" else ""))
    A(f"* features: `{cfg.features}`; network {cfg.n_layers} x {cfg.n_neurons} per window")
    A(f"* optimiser: `{cfg.optimizer}`\n")
    A(f"space-time relative L2 error: **{metrics['rel_l2_space_time']:.6e}**\n")
    A("| t | rel L2 | max abs err |")
    A("|---|--------|-------------|")
    for t in metrics["per_time"]:
        m = metrics["per_time"][t]
        A(f"| {t:g} | {m['rel_l2']:.6e} | {m['linf']:.6e} |")
    A("")
    A("| window | t range | final loss | redraws | inherited rel L2 | window rel L2 | amplification | wall (s) |")
    A("|---|---|---|---|---|---|---|---|")
    for r in window_records:
        A(f"| {r['index']+1} | [{r['t0']:g}, {r['t1']:g}] | {r['final_loss']:.3e} | "
          f"{r.get('resamples', 0)} | "
          f"{r.get('ic_rel_l2', float('nan')):.3e} | {r['rel_l2']:.3e} | "
          f"x{r.get('amplification', float('nan')):.2f} | {r['wall']:.0f} |")
    A("")
    A(f"wall time: {wall:.1f} s")
    return "\n".join(L) + "\n"
