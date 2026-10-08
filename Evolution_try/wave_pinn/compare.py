"""Collect every run under ``runs/`` into one comparison table.

    cd Evolution_try
    /Users/reula/jax_env/bin/python -m wave_pinn.compare
    /Users/reula/jax_env/bin/python -m wave_pinn.compare --runs runs --out runs/COMPARISON.md

The table is built from each run's ``history.json`` (which also carries the
metrics) and ``config.json``, so it needs no re-evaluation.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

COLUMNS = [
    ("label", "run"),
    ("optimizer", "optimiser"),
    ("features", "features"),
    ("n_modes", "modes"),
    ("n_coll", "n_coll"),
    ("final_loss", "final loss"),
    ("rel_l2_space_time", "rel L2 (space-time)"),
    ("rel_l2_T", "rel L2 at T"),
    ("wall", "wall (s)"),
    ("n_iter", "iterations"),
]


def collect(runs_dir: str) -> List[Dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "history.json"))):
        d = os.path.dirname(path)
        try:
            with open(path) as fh:
                hist = json.load(fh)
            with open(os.path.join(d, "config.json")) as fh:
                cfg = json.load(fh)
        except (OSError, ValueError):
            continue
        metrics = hist.get("metrics") or {}
        per_time = metrics.get("per_time") or {}
        t_last = max((float(k) for k in per_time), default=None)
        rows.append({
            "label": cfg.get("label", os.path.basename(d)),
            "optimizer": cfg.get("optimizer", "?"),
            "features": cfg.get("features", "?"),
            "n_modes": cfg.get("n_modes", ""),
            "n_coll": cfg.get("n_coll", ""),
            "final_loss": hist.get("final_loss"),
            "rel_l2_space_time": metrics.get("rel_l2_space_time"),
            "rel_l2_T": per_time.get(str(t_last), {}).get("rel_l2") if t_last is not None else None,
            "wall": hist.get("wall"),
            "n_iter": sum(i.get("iterations") or 0 for i in hist.get("phase_infos", [])),
            "dir": d,
        })
    rows.sort(key=lambda r: (r["rel_l2_space_time"] if r["rel_l2_space_time"] is not None else 1e9))
    return rows


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3e}" if (v and abs(v) < 1e-3) or abs(v) >= 1e4 else f"{v:.4g}"
    return str(v)


def render(rows: List[Dict]) -> str:
    head = "| " + " | ".join(name for _, name in COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    lines = [head, sep]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(key)) for key, _ in COLUMNS) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default=os.path.join(here, "runs"))
    p.add_argument("--out", default=None, help="write the markdown table here as well")
    args = p.parse_args(argv)
    rows = collect(args.runs)
    table = render(rows)
    print(table)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write("# Run comparison\n\n")
            fh.write("Sorted by space-time relative L2 error.\n\n")
            fh.write(table + "\n")
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
