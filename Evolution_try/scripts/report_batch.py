#!/usr/bin/env python
"""Collect every finished windowed run into one report and one figure.

    <python> scripts/report_batch.py [run ...]

With no arguments it discovers every ``runs/*/`` that holds a ``windows.json``,
so a partial batch reports on what exists rather than failing, and the two
hand-run T=20 windows are included alongside the batch without being named.

The figure answers, in order: how the error grows along the integration, how the
runs rank overall, how much each hand-over amplifies the error, and whether each
window actually converged on its own sample.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
OUT_PNG = os.path.join(ROOT, "plots", "batch_report.png")
OUT_MD = os.path.join(ROOT, "BATCH_REPORT.md")

IC_COLOR = {"hard": "#c0392b", "soft": "#1f6feb", "soft_all": "#2e8b57"}
IC_LABEL = {"hard": "hard hand-over", "soft": "soft (window 1 hard)",
            "soft_all": "soft, first window too"}
OPT_STYLE = {"dsgnar": "-", "ssbroyden": "--", "jaxopt_broyden": ":"}
OPT_MARK = {"dsgnar": "o", "ssbroyden": "s", "jaxopt_broyden": "^"}


def load_run(path: str) -> Optional[Dict]:
    cfg_p = os.path.join(path, "config.json")
    win_p = os.path.join(path, "windows.json")
    if not (os.path.exists(cfg_p) and os.path.exists(win_p)):
        return None
    cfg = json.load(open(cfg_p))
    win = json.load(open(win_p))
    rec: Dict = {
        "name": os.path.basename(path.rstrip("/")),
        "optimizer": cfg.get("optimizer"),
        "window_ic": cfg.get("window_ic", "hard"),
        "n_coll": cfg.get("n_coll"),
        "w_ic": cfg.get("w_ic"),
        "ansatz": cfg.get("ansatz"),
        "t_scale": cfg.get("t_scale"),
        "precision": cfg.get("precision"),
        "rounds": cfg.get("resample_rounds"),
        "wall": win.get("wall"),
        "overall": win.get("metrics", {}).get("rel_l2_space_time"),
        "windows": win.get("windows", []),
    }
    f = os.path.join(path, "fields.npz")
    if os.path.exists(f):
        d = np.load(f)
        rec["t"] = np.asarray(d["times"], dtype=float)
        rec["rel"] = (np.linalg.norm(d["u"] - d["exact"], axis=1)
                      / np.linalg.norm(d["exact"], axis=1))
    else:
        rec["t"] = rec["rel"] = None
    return rec


def main(argv: List[str]) -> int:
    names = argv[1:]
    if names:
        paths = [os.path.join(RUNS, n) for n in names]
    else:
        paths = [os.path.join(RUNS, n) for n in sorted(os.listdir(RUNS))]
    runs = [r for r in (load_run(p) for p in paths) if r]
    runs = [r for r in runs if r["windows"]]
    if not runs:
        print("no windowed runs found under runs/")
        return 1
    runs.sort(key=lambda r: (r["optimizer"] or "", r["window_ic"] or "", r["n_coll"] or 0))

    print(f"{len(runs)} windowed runs\n")
    hdr = f"{'run':30s} {'optimiser':16s} {'hand-over':10s} {'n_coll':>7s} {'rel L2':>10s} {'amp':>6s} {'wall':>7s} {'F64':>5s}"
    print(hdr)
    print("-" * len(hdr))
    for r in runs:
        amps = [w["amplification"] for w in r["windows"]
                if w.get("amplification") and np.isfinite(w["amplification"])]
        amp = float(np.mean(amps)) if amps else float("nan")
        print(f"{r['name']:30s} {r['optimizer'] or '?':16s} {r['window_ic']:10s} "
              f"{r['n_coll'] or 0:7d} {r['overall']:10.3e} {amp:6.2f} "
              f"{r['wall'] or 0:6.0f}s {str(r['precision'] == 'float64'):>5s}")

    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
    fig.suptitle(f"Windowed T = 20 — {len(runs)} runs, float64, 10 windows of dt = 2\n"
                 f"colour = how a window inherits the previous one, "
                 f"line = optimiser, marker = n_coll", fontsize=12)

    # (a) error along the integration
    a = ax[0, 0]
    for r in runs:
        if r["rel"] is None:
            continue
        a.semilogy(r["t"], np.maximum(r["rel"], 1e-16),
                   marker=OPT_MARK.get(r["optimizer"], "o"), ls=OPT_STYLE.get(r["optimizer"], "-"),
                   color=IC_COLOR.get(r["window_ic"], "grey"), lw=1.5, ms=4, alpha=.9,
                   label=f"{r['optimizer']}/{r['window_ic']}/{r['n_coll']}")
    a.set_xlabel("t"); a.set_ylabel(r"relative $L_2$ error")
    a.set_title("(a) error along the integration"); a.grid(alpha=.3, which="both")
    a.legend(fontsize=6.5, ncol=2)

    # (b) overall ranking
    a = ax[0, 1]
    xs = np.arange(len(runs))
    a.bar(xs, [max(r["overall"], 1e-18) for r in runs],
          color=[IC_COLOR.get(r["window_ic"], "grey") for r in runs], alpha=.9)
    for i, r in enumerate(runs):
        a.text(i, max(r["overall"], 1e-18) * 1.6, f"{r['overall']:.0e}", ha="center", fontsize=6.5)
    a.set_yscale("log"); a.set_xticks(xs)
    a.set_xticklabels([f"{r['optimizer'][:4]}\n{r['window_ic']}\n{r['n_coll']}" for r in runs],
                      fontsize=6.5)
    a.set_ylabel(r"space-time relative $L_2$")
    a.set_title("(b) whole-integration accuracy"); a.grid(alpha=.3, axis="y")

    # (c) amplification per hand-over
    a = ax[1, 0]
    for r in runs:
        amps = [w.get("amplification") for w in r["windows"]]
        amps = [x if (x and np.isfinite(x)) else np.nan for x in amps]
        a.plot(np.arange(1, len(amps) + 1), amps,
               marker=OPT_MARK.get(r["optimizer"], "o"), ls=OPT_STYLE.get(r["optimizer"], "-"),
               color=IC_COLOR.get(r["window_ic"], "grey"), lw=1.4, ms=4, alpha=.9)
    a.axhline(1.0, color="k", ls=":", lw=1.2)
    a.set_xlabel("hand-over number"); a.set_ylabel("error out / error in")
    a.set_title("(c) amplification at each hand-over  (1 = the chain preserves the error)")
    a.grid(alpha=.3)

    # (d) per-window convergence
    a = ax[1, 1]
    for r in runs:
        a.semilogy([w["t1"] for w in r["windows"]],
                   np.maximum([w["final_loss"] for w in r["windows"]], 1e-16),
                   marker=OPT_MARK.get(r["optimizer"], "o"), ls=OPT_STYLE.get(r["optimizer"], "-"),
                   color=IC_COLOR.get(r["window_ic"], "grey"), lw=1.4, ms=4, alpha=.9)
    a.set_xlabel("t at the end of the window"); a.set_ylabel("final loss in the window")
    a.set_title("(d) did each window converge on its own sample?")
    a.grid(alpha=.3, which="both")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nwrote {os.path.relpath(OUT_PNG, ROOT)}")

    # ---- markdown -----------------------------------------------------------
    L = ["# Batch report: windowed T = 20", "",
         f"{len(runs)} runs, all float64, 10 windows of dt = 2.  Generated by "
         "`scripts/report_batch.py`.", "",
         "| run | optimiser | hand-over | n_coll | space-time rel L2 | mean amplification | wall |",
         "|---|---|---|---|---|---|---|"]
    for r in runs:
        amps = [w["amplification"] for w in r["windows"]
                if w.get("amplification") and np.isfinite(w["amplification"])]
        amp = f"{np.mean(amps):.2f}" if amps else "--"
        L.append(f"| `{r['name']}` | {r['optimizer']} | {r['window_ic']} | {r['n_coll']} | "
                 f"{r['overall']:.3e} | {amp} | {r['wall']:.0f} s |")
    L += ["", "## Per window", "",
          "| run | window | t | final loss | inherited rel L2 | window rel L2 | amplification | redraws |",
          "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        for w in r["windows"]:
            L.append(f"| `{r['name']}` | {w['index']+1} | [{w['t0']:g}, {w['t1']:g}] | "
                     f"{w['final_loss']:.2e} | {w.get('ic_rel_l2', float('nan')):.2e} | "
                     f"{w['rel_l2']:.2e} | x{w.get('amplification', float('nan')):.2f} | "
                     f"{w.get('resamples', 0)} |")
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {os.path.relpath(OUT_MD, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
