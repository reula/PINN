"""Error measurement against the exact solution, and the plots that summarise a run."""

from __future__ import annotations

import json
import os
from typing import Dict, List

import jax.numpy as jnp
import numpy as np

from .config import Config
from .problem import exact_solution
from .sampling import test_grid


def solution_on_grid(u_fn, cfg: Config, times=None, n_x: int | None = None) -> Dict:
    """Sample a solution ``u_fn(t, x)`` and the exact solution on ``times x [-L, L]``.

    ``u_fn`` rather than a parameter vector, because in the windowed mode the
    solution is piecewise: a different network covers each time slab.
    """
    n_x = n_x or cfg.n_test_x
    x = jnp.linspace(-cfg.L, cfg.L, n_x)
    times = list(times if times is not None else cfg.snapshot_times)
    out = {"x": np.asarray(x), "times": times, "u": {}, "exact": {}, "abs_err": {}}
    for t in times:
        tt = jnp.full_like(x, float(t))
        u = np.asarray(u_fn(tt, x))
        ex = np.asarray(exact_solution(cfg, tt, x))
        out["u"][float(t)] = u
        out["exact"][float(t)] = ex
        out["abs_err"][float(t)] = u - ex
    return out


def error_metrics(grid: Dict) -> Dict:
    """Per-time relative L2 and L-infinity errors, plus the space-time relative L2."""
    per_time = {}
    num = 0.0
    den = 0.0
    for t in grid["times"]:
        t = float(t)
        u, ex = grid["u"][t], grid["exact"][t]
        nrm = float(np.sqrt(np.mean(ex ** 2)))
        err = float(np.sqrt(np.mean((u - ex) ** 2)))
        per_time[t] = {"rel_l2": err / nrm if nrm > 0 else err,
                       "linf": float(np.max(np.abs(u - ex))),
                       "rel_linf": float(np.max(np.abs(u - ex)) / np.max(np.abs(ex)))}
        num += float(np.mean((u - ex) ** 2))
        den += float(np.mean(ex ** 2))
    return {"per_time": per_time,
            "rel_l2_space_time": float(np.sqrt(num / den)) if den > 0 else float("nan"),
            "final_time_rel_l2": per_time[float(grid["times"][-1])]["rel_l2"]}


def evaluate(u_fn, cfg: Config, times=None, n_x: int | None = None) -> Dict:
    grid = solution_on_grid(u_fn, cfg, times=times, n_x=n_x)
    return {"grid": grid, "metrics": error_metrics(grid)}


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def _mpl():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-wazepinn")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_solution(grid: Dict, path: str, title: str = "") -> None:
    plt = _mpl()
    times = [float(t) for t in grid["times"]]
    n = len(times)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.0), sharey=True)
    if n == 1:
        axes = [axes]
    x = grid["x"]
    for ax, t in zip(axes, times):
        ax.plot(x, grid["exact"][t], "k-", lw=1.6, label="exact")
        ax.plot(x, grid["u"][t], "r--", lw=1.4, label="network")
        ax.set_title(f"u(t, x),  t = {t:g}")
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("u")
    axes[0].legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_error_map(grid: Dict, path: str) -> None:
    """Pointwise error and the network/exact fields over the whole space-time slab."""
    plt = _mpl()
    x = grid["x"]
    times = np.asarray([float(t) for t in grid["times"]])
    U = np.stack([grid["u"][float(t)] for t in times])
    E = np.stack([grid["abs_err"][float(t)] for t in times])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
    im0 = axes[0].pcolormesh(x, times, U, shading="auto", cmap="RdBu_r")
    axes[0].set_title("network  u")
    im1 = axes[1].pcolormesh(x, times, E, shading="auto", cmap="magma")
    axes[1].set_title("|error|")
    axes[2].semilogy(times, [np.sqrt(np.mean(E[i] ** 2)) for i in range(len(times))], "o-")
    axes[2].set_title("L2 error vs t")
    axes[2].set_xlabel("t")
    for ax in axes[:2]:
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im0 if ax is axes[0] else im1, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_history(history: List[Dict], path: str, extra: List[Dict] | None = None) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if history:
        steps = [h["step"] for h in history]
        losses = [max(h["loss"], 1e-300) for h in history]
        ax.semilogy(steps, losses, "-", lw=1.4, label="training loss")
    for i, h in enumerate(extra or []):
        if not h:
            continue
        st = [d["step"] for d in h]
        lo = [max(d["loss"], 1e-300) for d in h]
        ax.semilogy(st, lo, "--", lw=1.1, label=f"phase {i+1}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
