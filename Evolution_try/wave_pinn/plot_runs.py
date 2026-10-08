"""Side-by-side comparison of finished runs: loss curves and final-time profiles.

    cd Evolution_try
    /Users/reula/jax_env/bin/python -m wave_pinn.plot_runs \
        --runs ssbroyden_ref dsgnar_ref --out runs/comparison.png

Reads only what :mod:`wave_pinn.train` already wrote (``history.json`` and
``fields.npz``), so it never re-trains anything.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np

from .evaluate import _mpl


def load_run(path: str) -> Dict:
    with open(os.path.join(path, "history.json")) as fh:
        hist = json.load(fh)
    with open(os.path.join(path, "config.json")) as fh:
        cfg = json.load(fh)
    fields = np.load(os.path.join(path, "fields.npz"))
    return {"dir": path, "config": cfg, "history": hist, "fields": fields,
            "label": cfg.get("label", os.path.basename(path))}


def resolve(name: str, runs_dir: str) -> str:
    if os.path.isdir(name):
        return name
    cand = os.path.join(runs_dir, name)
    if os.path.isdir(cand):
        return cand
    hits = sorted(glob.glob(os.path.join(runs_dir, f"*{name}*")))
    if len(hits) == 1:
        return hits[0]
    raise SystemExit(f"cannot resolve run {name!r} under {runs_dir} ({len(hits)} matches)")


def main(argv=None) -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--runs-dir", default=os.path.join(here, "runs"))
    p.add_argument("--out", default=os.path.join(here, "runs", "comparison.png"))
    p.add_argument("--time", type=float, default=None,
                   help="time slice for the profile panel (default: the last one)")
    p.add_argument("--t-min", type=float, default=None, help="clip the loss panel's y-axis below this")
    args = p.parse_args(argv)

    runs: List[Dict] = [load_run(resolve(name, args.runs_dir)) for name in args.runs]
    plt = _mpl()
    fig, axes = plt.subplots(1, 4, figsize=(20.5, 4.2))

    # ---- loss histories -----------------------------------------------------
    for r in runs:
        h = r["history"]["history"]
        ax = axes[0]
        ax.semilogy([d["step"] for d in h], [max(d["loss"], 1e-300) for d in h],
                    "-", lw=1.5, label=r["label"])
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("training loss (normalised residual)")
    axes[0].grid(alpha=0.3, which="both")
    if args.t_min:
        axes[0].set_ylim(bottom=args.t_min)
    axes[0].legend(fontsize=8)

    # ---- error against the exact solution -----------------------------------
    for r in runs:
        f = r["fields"]
        times = [float(t) for t in f["times"]]
        err = [np.sqrt(np.mean(f["abs_err"][i] ** 2)) / np.sqrt(np.mean(f["exact"][i] ** 2))
               for i in range(len(times))]
        axes[1].semilogy(times, err, "o-", label=r["label"])
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("relative L2 error")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8)

    # ---- the true error along the run ---------------------------------------
    # This is the panel that answers "is running further buying anything?".  The
    # loss can keep falling on a frozen sample while this stands still, or even
    # rises; only this curve says whether the solution is improving.
    for r in runs:
        th = [d for d in r["history"].get("test_history", []) if "rel_l2" in d]
        if not th:
            continue
        axes[2].semilogy([d["step"] for d in th], [max(d["rel_l2"], 1e-300) for d in th],
                         "o-", ms=4, label=r["label"])
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("relative L2 error (exact solution)")
    axes[2].grid(alpha=0.3, which="both")
    axes[2].legend(fontsize=8)

    # ---- final-time profiles ------------------------------------------------
    r0 = runs[0]
    times = [float(t) for t in r0["fields"]["times"]]
    t_show = float(args.time) if args.time is not None else times[-1]
    idx = int(np.argmin([abs(t - t_show) for t in times]))
    x = r0["fields"]["x"]
    axes[3].plot(x, r0["fields"]["exact"][idx], "k-", lw=2.0, label="exact")
    for r in runs:
        f = r["fields"]
        times_r = [float(t) for t in f["times"]]
        j = int(np.argmin([abs(t - t_show) for t in times_r]))
        axes[3].plot(f["x"], f["u"][j], "--", lw=1.3, label=r["label"])
    axes[3].set_title(f"u at t = {times[idx]:g}")
    axes[3].set_xlabel("x")
    axes[3].set_ylabel("u")
    axes[3].grid(alpha=0.3)
    axes[3].legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"written {args.out}")

    for r in runs:
        m = r["history"].get("metrics", {})
        print(f"{r['label']:>18}  loss {r['history'].get('final_loss'):.3e}  "
              f"rel L2 {m.get('rel_l2_space_time'):.3e}  "
              f"wall {r['history'].get('wall', float('nan')):.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
