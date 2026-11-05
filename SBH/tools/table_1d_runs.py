"""Band errors and Misner-Sharp mass of several sing_1d.py runs at chosen times, as a text table. (2026-11-05)

    python tools/table_1d_runs.py "T1,T2,..." NAME=RUN_DIR [NAME=RUN_DIR ...]

RUN_DIR/report.npz from sing_1d.py (t, r, err (n_t, n_r, 10) signed f - f_ref against the 1D reference
reference_1d/ref_single_m0.5_dr0.005_t20.npz, mass (n_t, n_r) m/M - 1). Per run and time (the nearest saved
window end; runs that have not reached a time are skipped): rms of the alpha | chi | Khat errors in the bands
r = 0.05-0.2 / 0.2-1 / 1-3 / 3-10, m/M - 1 at r = 0.3 / 1 / 3 / 10, and max |m/M - 1| over r >= 0.2 and all saved
times up to it. The companion plot is tools/plot_1d_runs_t.py.
"""
import os
import sys

import numpy as np

BANDS = ((0.05, 0.2), (0.2, 1.0), (1.0, 3.0), (3.0, 10.0))
FIELDS = ((8, "alpha"), (4, "chi"), (5, "Khat"))

times = [float(s) for s in sys.argv[1].split(",")]
runs = [s.rsplit("=", 1) for s in sys.argv[2:]]
Z = {name: np.load(os.path.join(d, "report.npz")) for name, d in runs}
w = max(len(n) for n, _ in runs)
for name, z in Z.items():
    print(f"{name:{w}s}  saved t up to {z['t'].max():g} ({len(z['t'])} windows)")
for t0 in times:
    print(f"\nt = {t0:g}: " + " | ".join(f for _, f in FIELDS) + " rms in r = 0.05-0.2 / 0.2-1 / 1-3 / 3-10"
          " | m/M - 1 at r = 0.3 / 1 / 3 / 10 | max |m/M - 1| (r >= 0.2, t' <= t)")
    for name, z in Z.items():
        if z["t"].max() < t0 - 1e-9:
            continue
        k, r = int(np.argmin(abs(z["t"] - t0))), z["r"]
        e = z["err"][k]
        cols = [" ".join(f"{np.sqrt((e[(r >= a) & (r < b), i] ** 2).mean()):.0e}" for a, b in BANDS) for i, _ in FIELDS]
        m = " ".join(f"{np.interp(r0, r, z['mass'][k]):+.1e}" for r0 in (0.3, 1.0, 3.0, 10.0))
        mx = np.abs(z["mass"][:k + 1][:, r >= 0.2]).max()
        print(f"{name:{w}s}  " + " | ".join(cols) + f" | {m} | {mx:.1e}")
