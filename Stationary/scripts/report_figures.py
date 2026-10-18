#!/usr/bin/env python
"""Figures of REPORT_PIN_COM.md and REPORT_OPTIMISERS.md, from scripts/report_metrics.py's JSON and
the runs' history.json files.

    .venv/bin/python scripts/report_figures.py --metrics plots/report_metrics.json --runs-dir runs

Colours: a validated categorical palette (dataviz skill) in fixed order; in the optimiser figures
the colour is the OPTIMISER (SSBroyden blue, DSGNAR orange) in every panel.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"]
INK, MUTED, GRID = "#1f1f1e", "#898781", "#e1e0d9"
SSB, DSG = C[0], C[1]

# the four steps of the pin/COM story, all 2285 parameters, same data (S2 = -1/12)
STEPS = [("ib16k_ssbroyden", "rho^2 weights, lambda pin"),
         ("m4_ssb_rho4", "rho^4, lambda pin"),
         ("m5_ssb_norsmp", "rho^4, pin on P"),
         ("m7_dsg_com", "rho^4, pin on P, COM gauge")]


def style(ax):
    ax.grid(True, which="major", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def fig_pin_com(M, out):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    for k, (name, lab) in enumerate(STEPS):
        if name not in M:
            continue
        p = M[name]["profile"]
        rho = np.asarray(p["rho"])
        ax[0].loglog(rho, p["dlam_rms"], color=C[k], lw=2, label=lab)
        ax[1].loglog(rho, p["dh_rms"], color=C[k], lw=2, label=lab)
        o = M[name]["odd"]
        ax[2].loglog(o["rho"], np.maximum(np.abs(o["l1"]), 1e-14), color=C[k], lw=2, label=lab)
    ref = M[STEPS[0][0]]["profile"]
    for a, key in ((ax[0], "pert_lam_rms"), (ax[1], "pert_h_rms")):
        a.loglog(ref["rho"], ref[key], "--", color=INK, lw=1.4,
                 label="perturbation size (reference $-$ spherical)")
    ax[0].set_title(r"$\lambda$ error: angular rms of $|\lambda_{net}-\lambda_{ref}|$", fontsize=11)
    ax[1].set_title(r"$h$ error: angular rms of $\max_{ij}|h_{net}-h_{ref}|$", fontsize=11)
    ax[2].set_title(r"the odd part: $|l=1$ coefficient of $\lambda_{net}-\lambda_{ref}|$", fontsize=11)
    for a in ax:
        a.set_xlabel(r"$\rho$")
        style(a)
    hs, ls = ax[0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle("Error against the fine nonlinear reference (l <= 10, 36 radial, its loss 2.8e-11), "
                 "one step at a time", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _ssb_curve(run_dir):
    h = json.load(open(os.path.join(run_dir, "history.json")))
    rep = json.load(open(os.path.join(run_dir, "report.json")))
    h = [e for e in h if isinstance(e, dict) and "loss" in e and "step" in e]
    n = max(e["step"] for e in h) or 1
    wall = float(rep.get("wall_seconds") or 0.0)
    # SSBroyden's history has no clock: time is spread uniformly over the iterations (stated)
    return (np.array([e["step"] / n * wall / 60 for e in h]), np.array([e["loss"] for e in h]))


def _dsg_curve(run_dir):
    h = json.load(open(os.path.join(run_dir, "history.json")))
    t, L, off, last = [], [], 0.0, 0.0
    for e in h:
        if not isinstance(e, dict) or "wall" not in e:
            continue
        w = float(e["wall"])
        if w < last:              # a new restart cycle restarts the clock
            off += last
        last = w
        t.append((off + w) / 60)
        L.append(e["loss"])
    return np.array(t), np.array(L)


PAIRS = [("rho^2 weights, lambda pin, from Adam (6.3e-2)",
          [("ib16k_ssbroyden", "SSBroyden (hub GPU)", "ssb"), ("m2_dsg_s2000_t075", "DSGNAR s=2000 (A30)", "dsg")]),
         ("rho^4, pin on P, from the rho^4 floor run (9.4e-9)",
          [("m5_ssb_norsmp", "SSBroyden, no resampling", "ssb"), ("m5_dsg_cyc", "DSGNAR, 5 restart cycles", "dsg")]),
         ("rho^4, pin on P, COM gauge, from the run above (0.395)",
          [("m7_dsg_com", "DSGNAR, 5 restart cycles", "dsg")])]


def fig_optimisers(runs_dir, out):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    for a, (title, members) in zip(ax, PAIRS):
        for name, lab, kind in members:
            d = os.path.join(runs_dir, name)
            if not os.path.exists(os.path.join(d, "history.json")):
                continue
            t, L = (_ssb_curve if kind == "ssb" else _dsg_curve)(d)
            a.semilogy(t, L, color=SSB if kind == "ssb" else DSG, lw=2, label=lab)
        a.set_title(title, fontsize=10.5)
        a.set_xlabel("wall time (min)")
        style(a)
        a.legend(fontsize=8.5, frameon=False)
    ax[2].annotate("SSBroyden from the same start:\nstalls at iteration 0\n(line search fails)",
                   xy=(0.55, 0.75), xycoords="axes fraction", fontsize=9, color=SSB)
    ax[0].set_ylabel("loss (each run's own objective)")
    fig.suptitle("SSBroyden (blue) against DSGNAR (orange): loss against wall time, same start in "
                 "each panel", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--metrics", default="plots/report_metrics.json")
    ap.add_argument("--runs-dir", default="runs")
    a = ap.parse_args(argv)
    M = json.load(open(a.metrics))
    fig_pin_com(M, "plots/report_pin_com.png")
    fig_optimisers(a.runs_dir, "plots/report_optimisers.png")
    print("wrote plots/report_pin_com.png, plots/report_optimisers.png")


if __name__ == "__main__":
    main()
