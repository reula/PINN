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
# marker encodes the SKETCH SIZE, which is the axis these batches exist to
# compare.  With two collocation sizes AND three sketch sizes the same (optimiser,
# hand-over) pair occurs six times; encoding any two of them in one channel draws
# distinguishable runs as a single line.
def sketch_label(cfg):
    s = cfg.get("dsgnar_sketch") or 0
    if s == 0:
        return "d/3"
    return "d/2" if s == 1100 else ("0.68d" if s == 1500 else f"s={s}")


SKETCH_MARK = {"d/3": "o", "d/2": "s", "0.68d": "^"}


def mark(run):
    return SKETCH_MARK.get(run["sketch"], "D")


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
        "sketch": sketch_label(cfg),
        "retries": sum(x.get("attempts", 0) for x in win.get("windows", [])),
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

    fig, ax = plt.subplots(2, 3, figsize=(19, 9.5))
    fig.suptitle(f"Windowed T = 20 — {len(runs)} runs, float64, 10 windows of dt = 2\n"
                 f"colour = hand-over, line = optimiser, "
                 f"marker = sketch size (o d/3, s d/2, ^ 0.68 d)", fontsize=12)

    # (a) error along the integration
    a = ax[0, 0]
    for r in runs:
        if r["rel"] is None:
            continue
        a.semilogy(r["t"], np.maximum(r["rel"], 1e-16),
                   marker=mark(r), ls=OPT_STYLE.get(r["optimizer"], "-"),
                   color=IC_COLOR.get(r["window_ic"], "grey"), lw=1.5, ms=4, alpha=.9,
                   label=f"{r['optimizer']}/{r['window_ic']}/{r['n_coll']}")
    a.set_xlabel("t"); a.set_ylabel(r"relative $L_2$ error")
    a.set_title("(a) error along the integration"); a.grid(alpha=.3, which="both")
    a.legend(fontsize=6.5, ncol=2)

    # (b) overall ranking
    a = ax[0, 2]
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

    # (f) stalls: windows that needed a redraw-and-retry
    a = ax[0, 1]
    xs = np.arange(len(runs))
    a.bar(xs, [r["retries"] for r in runs], color=[IC_COLOR.get(r["window_ic"], "grey") for r in runs])
    a.set_xticks(xs)
    a.set_xticklabels([f"{r['optimizer'][:4]}\n{r['window_ic'][:6]}\n{r['sketch']}" for r in runs],
                      fontsize=5.5)
    a.set_ylabel("windows that stalled and were retried")
    a.set_title("(f) the stall is not rare -- and at n_coll=2201 it is not always rescued")
    a.grid(alpha=.3, axis="y")

    # (c) amplification per hand-over
    a = ax[1, 0]
    for r in runs:
        amps = [w.get("amplification") for w in r["windows"]]
        amps = [x if (x and np.isfinite(x)) else np.nan for x in amps]
        a.plot(np.arange(1, len(amps) + 1), amps,
               marker=mark(r), ls=OPT_STYLE.get(r["optimizer"], "-"),
               color=IC_COLOR.get(r["window_ic"], "grey"), lw=1.4, ms=4, alpha=.9)
    a.axhline(1.0, color="k", ls=":", lw=1.2)
    # log scale: one stalled window reached x1709 and would flatten every other
    # curve into the axis, hiding the question this panel exists to answer
    a.set_yscale("log")
    a.set_xlabel("hand-over number"); a.set_ylabel("error out / error in  (log)")
    a.set_title("(c) amplification at each hand-over  (1 = the chain preserves the error)")
    a.grid(alpha=.3)

    # (d) per-window convergence
    a = ax[1, 1]
    for r in runs:
        a.semilogy([w["t1"] for w in r["windows"]],
                   np.maximum([w["final_loss"] for w in r["windows"]], 1e-16),
                   marker=mark(r), ls=OPT_STYLE.get(r["optimizer"], "-"),
                   color=IC_COLOR.get(r["window_ic"], "grey"), lw=1.4, ms=4, alpha=.9)
    a.set_xlabel("t at the end of the window"); a.set_ylabel("final loss in the window")
    a.set_title("(d) did each window converge on its own sample?")
    a.grid(alpha=.3, which="both")

    # (e) the sketch-size effect: the axis these two batches exist to measure
    a = ax[1, 2]
    for ic in ("hard", "soft", "soft_all"):
        pts = [r for r in runs if r["window_ic"] == ic and r["n_coll"] == 8192
               and r["optimizer"] == "dsgnar"]
        order = {"d/3": 0, "d/2": 1, "0.68d": 2}
        pts = sorted([p for p in pts if p["sketch"] in order], key=lambda r: order[r["sketch"]])
        if not pts:
            continue
        a.semilogy([order[p["sketch"]] for p in pts], [p["overall"] for p in pts],
                   marker="o", color=IC_COLOR.get(ic, "grey"), lw=1.8, ms=7,
                   label=IC_LABEL.get(ic, ic))
        for p in pts:
            a.annotate(f"{p['overall']:.1e}", (order[p["sketch"]], p["overall"]),
                       textcoords="offset points", xytext=(4, 4), fontsize=7)
    a.set_xticks([0, 1, 2]); a.set_xticklabels(["d/3\n(733)", "d/2\n(1100)", "0.68 d\n(1500)"])
    a.set_ylabel(r"space-time relative $L_2$")
    a.set_title("(e) the sketch size is the dominant knob\n(dsgnar, n_coll = 8192)")
    a.grid(alpha=.3, which="both"); a.legend(fontsize=8)

    # (f) how often a window had to be rescued by a redraw-and-retry.  In these
    # runs it is not rare, and at n_coll = 2201 it does not always rescue.
    if False:
        pass
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
