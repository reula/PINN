"""Parse the `laplace_probe.py` logs into one JSON, so the figures are reproducible.

The probe prints, per batch, the iteration count, the stop detail, the max error and the per-sector
decomposition.  This turns those logs (plus the production DSGNAR histories, downsampled) into
`plots/flat_results.json`, which `make_flat_figures.py` reads.  Kept separate from the figures so
the numbers can be inspected without matplotlib.

    ./.venv/bin/python collect_flat_results.py [--log PATH ...] [--out plots/flat_results.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

SETTINGS = re.compile(r"settings (.+)")
HEADER = re.compile(r"optimiser (\w+), (\d+) collocation \+ (\d+) boundary points, radial hybrid "
                    r"with log fraction ([\d.]+), (\d+) independent batches")
CASE = re.compile(r"flat Laplacian, ratio-4 shell, order (\d+), S1 = ([\d.eE+-]+), S2 = ([\d.eE+-]+)")
BATCH = re.compile(r"batch (\d+):\s+(\d+) iterations \(([^)]*)\), max\|err\| ([\d.eE+-]+)")
SECTOR = re.compile(r"l=(\d):\s+growing rho\^(\d) ([+-][\d.eE+-]+)\s+decaying rho\^-(\d) "
                    r"([+-][\d.eE+-]+)\s+\(sector fit ([\d.eE+-]+)\)")


def parse(path):
    out, cur = [], None
    for line in open(path, errors="replace"):
        m = CASE.search(line)
        if m:
            cur = {"order": int(m.group(1)), "s1": float(m.group(2)), "s2": float(m.group(3)),
                   "batches": [], "source": os.path.basename(path)}
            out.append(cur)
            continue
        m = SETTINGS.search(line)
        if m and cur is not None:
            cur["settings"] = m.group(1).strip()
            continue
        m = HEADER.search(line)
        if m and cur is not None:
            cur["optimizer"] = m.group(1)
            cur["n_coll"], cur["n_bnd"] = int(m.group(2)), int(m.group(3))
            cur["log_frac"], cur["n_batches"] = float(m.group(4)), int(m.group(5))
            continue
        m = BATCH.search(line)
        if m and cur is not None:
            cur["batches"].append({"batch": int(m.group(1)), "iterations": int(m.group(2)),
                                   "detail": m.group(3), "max_err": float(m.group(4)), "sectors": {}})
            continue
        m = SECTOR.search(line)
        if m and cur is not None and cur["batches"]:
            l = int(m.group(1))
            cur["batches"][-1]["sectors"][l] = {"grow": float(m.group(3)),
                                                "decay": float(m.group(5)),
                                                "fit": float(m.group(6))}
    return [c for c in out if c["batches"]]


def production_histories(glob_pat, every=10):
    """Loss vs iteration of the finished production runs, for the convergence figure."""
    out = {}
    for path in sorted(glob.glob(glob_pat)):
        name = os.path.basename(path).replace("hist_", "").replace(".json", "")
        hist = json.load(open(path))
        hist = [e for e in hist if e.get("accepted") is not None]
        out[name] = {"step": [int(e["step"]) for e in hist[::every]],
                     "loss": [float(e["loss"]) for e in hist[::every]],
                     "radius": [float(e["radius"]) for e in hist[::every]]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", default=[])
    ap.add_argument("--out", default=os.path.join(HERE, "plots", "flat_results.json"))
    ap.add_argument("--production-hist", default="/tmp/hist_*.json")
    args = ap.parse_args()

    logs = args.log or sorted(glob.glob(os.path.join(HERE, "plots", "flat_probe_*.log")))
    cases = []
    for path in logs:
        if os.path.exists(path):
            got = parse(path)
            print(f"{os.path.basename(path)}: {len(got)} case(s)")
            cases += got
    payload = {"flat": cases, "production": production_histories(args.production_hist)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(payload, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}: {len(cases)} flat cases, "
          f"{len(payload['production'])} production histories")
    for c in cases:
        errs = [b["max_err"] for b in c["batches"]]
        print(f"  {c['optimizer']:9s} order {c['order']} S1={c['s1']:g}: "
              f"max|err| {min(errs):.3e}..{max(errs):.3e} over {len(errs)} batches")


if __name__ == "__main__":
    main()
