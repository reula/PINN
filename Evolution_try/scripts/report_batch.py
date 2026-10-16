#!/usr/bin/env python
"""Collect every finished windowed run into one report and one figure.

    <python> scripts/report_batch.py [run ...]

With no arguments it discovers every ``runs/*/`` that holds a ``windows.json``,
so a partial batch reports on what exists rather than failing, and the two
hand-run T=20 windows are included alongside the batch without being named.

The figure has one column per hand-over, with three rows: how the error grows
along the integration, how much each hand-over amplifies it, and how every run of
that hand-over ranks.  The best three runs are coloured, the rest are grey.
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

IC_LABEL = {"hard": "hard hand-over", "soft": "soft (window 1 hard)",
            "soft_all": "soft, first window too"}
HANDOVER_ORDER = {"soft_all": 0, "soft": 1, "hard": 2}
# the best three runs of each hand-over; validated categorical palette (dataviz skill)
TOP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
GREY = "#b4b2ac"
AMP_CAP = 100.0


def sketch_label(cfg):
    s = cfg.get("dsgnar_sketch") or 0
    if s in (0, 733):
        return "d/3"
    return "d/2" if s == 1100 else ("0.68d" if s == 1500 else f"s={s}")


def short(run) -> str:
    """A label that tells the runs of one hand-over apart: optimiser, n_coll, sketch, switches."""
    if run["optimizer"] != "dsgnar":
        return f"{run['optimizer']} {run['n_coll']}"
    lab = f"dsgnar {run['n_coll']} s={run['sketch']}"
    ch = run["changes"]
    if ch.count(",") >= 4:
        ch = "all switches"
    ch = ch.replace("target 0.075, Delta_min 1e-08", "paper's targets")
    if ch != "--":
        lab += f" ({ch})"
    if run["name"].endswith("_audit"):
        lab += " re-run"
    if run["name"].startswith("T20win"):
        lab += " hand-run"
    return lab


def changes(cfg) -> str:
    """What this run changes relative to the batch's DSGNAR defaults (README 6.13)."""
    out = []
    if cfg.get("optimizer") == "dsgnar":
        if float(cfg.get("dsgnar_stage1_ratio", 0.15)) != 0.15:
            out.append(f"target {cfg['dsgnar_stage1_ratio']:g}")
        if float(cfg.get("dsgnar_delta_min", 1e-14)) != 1e-14:
            out.append(f"Delta_min {cfg['dsgnar_delta_min']:g}")
        if float(cfg.get("window_warm_start", 0.0) or 0.0) > 0:
            out.append("warm start")
        if cfg.get("dsgnar_reject_overshoot"):
            out.append("overshoot rejection")
        if float(cfg.get("dsgnar_reweight_alpha", 0.0) or 0.0) > 0:
            out.append("reweighting")
        if cfg.get("dsgnar_fixed_sketch"):
            out.append("fixed sketch")
    return ", ".join(out) if out else "--"


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
        "changes": changes(cfg),
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

    # One column per hand-over; within each, the best three runs are coloured and
    # named, the rest are grey context.  With 30 runs, giving every run its own
    # style made the legend and the bar labels unreadable.
    ics = sorted({r["window_ic"] for r in runs}, key=lambda k: HANDOVER_ORDER.get(k, 9))
    fig, ax = plt.subplots(3, len(ics), figsize=(6.4 * len(ics), 15),
                           squeeze=False, gridspec_kw={"height_ratios": [1, 1, 1.5]})
    # the first two rows share their y axis; the bar row cannot, it lists different runs
    for row in (0, 1):
        for col in range(1, len(ics)):
            ax[row, col].sharey(ax[row, 0])
            ax[row, col].tick_params(axis="y", labelleft=False)
    fig.suptitle(f"Windowed T = 20: {len(runs)} runs, float64, 10 windows of dt = 2.  "
                 f"Coloured: the best three of each hand-over; grey: the rest.  "
                 f"Run-to-run variation is about x4.", fontsize=12)
    for col, ic in enumerate(ics):
        group = sorted((r for r in runs if r["window_ic"] == ic), key=lambda r: r["overall"])
        top = {r["name"]: TOP_COLORS[k] for k, r in enumerate(group[:len(TOP_COLORS)])}
        # grey first, so the coloured runs are drawn on top
        drawn = [r for r in group if r["name"] not in top][::-1] + group[:len(top)][::-1]

        a = ax[0, col]
        for r in drawn:
            if r["rel"] is None:
                continue
            c = top.get(r["name"])
            a.semilogy(r["t"], np.maximum(r["rel"], 1e-16), color=c or GREY,
                       lw=2.0 if c else 1.0, alpha=1 if c else .6, zorder=3 if c else 1,
                       label=short(r) if c else None)
        a.plot([], [], color=GREY, lw=1.0, label=f"the other {len(group) - len(top)}")
        a.set_title(f"{IC_LABEL.get(ic, ic)}: error along the integration", fontsize=11)
        a.set_xlabel("t")
        a.grid(alpha=.3, which="both"); a.legend(fontsize=8, loc="lower right")

        a = ax[1, col]
        for r in drawn:
            amps = [w.get("amplification") for w in r["windows"]]
            amps = np.array([x if (x and np.isfinite(x)) else np.nan for x in amps])
            c = top.get(r["name"])
            k = np.arange(1, len(amps) + 1)
            a.plot(k, np.minimum(amps, AMP_CAP), color=c or GREY, marker="o" if c else None,
                   ms=3.5, lw=2.0 if c else 1.0, alpha=1 if c else .6, zorder=3 if c else 1)
            for kk, x in zip(k, amps):
                if x > AMP_CAP:   # a stalled window: off the scale, value written in
                    a.annotate(f"x{x:.0e}", (kk, AMP_CAP), xytext=(-4, -12),
                               textcoords="offset points", ha="right", fontsize=7.5,
                               color="#52514e")
        a.axhline(1.0, color="k", ls=":", lw=1.2)
        # log scale, capped: a stalled window can reach x1e5 and flatten every other curve
        a.set_yscale("log"); a.set_ylim(0.2, AMP_CAP * 1.3)
        a.set_title("amplification at each hand-over (1 = error preserved)", fontsize=11)
        a.set_xlabel("hand-over number"); a.grid(alpha=.3)

        a = ax[2, col]
        ys = np.arange(len(group))[::-1]
        a.barh(ys, [r["overall"] for r in group], left=1e-10, height=.7,
               color=[top.get(r["name"], GREY) for r in group],
               hatch=None, edgecolor="none")
        for y, r in zip(ys, group):
            tag = f"{r['overall']:.1e}" + (f"   {r['retries']} retried" if r["retries"] else "")
            a.text(r["overall"] * 1.4, y, tag, va="center", fontsize=7.5, color="#52514e")
        a.set_xscale("log"); a.set_xlim(1e-10, 3)
        a.set_yticks(ys); a.set_yticklabels([short(r) for r in group], fontsize=7.5)
        a.tick_params(axis="y", labelleft=True)
        a.set_xlabel(r"space-time relative $L_2$ over T = 20")
        a.set_title("all runs of this hand-over, best at the top", fontsize=11)
        a.grid(alpha=.3, axis="x")
    ax[0, 0].set_ylabel(r"relative $L_2$ error")
    ax[1, 0].set_ylabel("error out / error in")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nwrote {os.path.relpath(OUT_PNG, ROOT)}")

    # ---- markdown -----------------------------------------------------------
    L = ["# Windowed T = 20: every run", "",
         f"{len(runs)} runs, all float64, 10 windows of dt = 2, ansatz t2.  Generated by "
         "`scripts/report_batch.py` from each `runs/<name>/windows.json`; rerun it after new runs.",
         "",
         "**How to read it.**  The error is the space-time relative L2 of the whole T = 20 chain "
         "against the exact solution.  Run-to-run variation of one configuration is about **x4** "
         "(`dsgnar_soft_all_nc2201` 7.8e-06 against its re-run `dsgnar_soft_all_nc2201_audit` "
         "3.0e-05), so only larger differences mean anything.  `s` is the column sketch "
         "(d = 2201 parameters; d/3 = 733).  *changes* are the switches of README 6.13 that differ "
         "from the batch defaults (target 0.15, Delta_min 1e-14, fresh init per window, no "
         "overshoot rejection, no reweighting).  *retries* counts windows redrawn after a stall; "
         "*worst window* is the largest final window loss.  Within each hand-over, best first.",
         "",
         "![one column per hand-over: error along the integration, amplification at each "
         "hand-over, and every run ranked; the best three coloured](plots/batch_report.png)",
         ""]
    order = HANDOVER_ORDER
    best = min(r["overall"] for r in runs if r["overall"] is not None)
    for ic in sorted({r["window_ic"] for r in runs}, key=lambda k: order.get(k, 9)):
        group = sorted((r for r in runs if r["window_ic"] == ic), key=lambda r: r["overall"])
        L += [f"## Hand-over: {ic}", "",
              "| run | optimiser | n_coll | s | changes | rel L2 (T = 20) | worst window | "
              "retries | mean amplification | wall |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in group:
            amps = [w["amplification"] for w in r["windows"]
                    if w.get("amplification") and np.isfinite(w["amplification"])]
            amp = f"{np.mean(amps):.2f}" if amps else "--"
            worst = max(w["final_loss"] for w in r["windows"])
            err = f"{r['overall']:.2e}"
            if r["overall"] == best:
                err = f"**{err}**"
            s_lab = r["sketch"] if r["optimizer"] == "dsgnar" else "--"
            L.append(f"| `{r['name']}` | {r['optimizer']} | {r['n_coll']} | {s_lab} | "
                     f"{r['changes']} | {err} | {worst:.0e} | {r['retries']} | {amp} | "
                     f"{r['wall'] / 60:.0f} min |")
        L.append("")
    L += ["## Per window", "",
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
