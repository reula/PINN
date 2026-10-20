"""Three-level self-convergence of the 1D reference (validation of the reference itself).
    python3 conv_check.py out/f5_dr0.02.npz out/f5_dr0.01.npz out/f5_dr0.005.npz
Prints, per radial band and time, the convergence factor
Q = |u_h - u_h/2| / |u_h/2 - u_h/4| (16 for 4th order) and the Richardson
error estimate of the finest run, |u_h/2 - u_h/4| / (Q_th - 1) with Q_th = 16."""
import sys
import numpy as np
from scipy.interpolate import CubicSpline

A, B, C = (np.load(f) for f in sys.argv[1:4])
fields = list(A["fields"])
bands = [(0.03, 0.1), (0.1, 0.5), (0.5, 5.0), (5.0, 20.0)]
rq = np.geomspace(0.03, 20.0, 800)
def at(D, t, i):
    k = int(np.argmin(np.abs(D["t"] - t)))
    return CubicSpline(D["r"], D["U"][k, i])(rq)
for t in (1.0, 2.0, 5.0, 10.0):
    print(f"t = {t}")
    for i, f in enumerate(fields):
        a, b, c = at(A, t, i), at(B, t, i), at(C, t, i)
        line = []
        for lo, hi in bands:
            m = (rq >= lo) & (rq < hi)
            d1 = np.sqrt(np.mean((a[m] - b[m])**2)); d2 = np.sqrt(np.mean((b[m] - c[m])**2))
            line.append(f"[{lo:g},{hi:g}) Q={d1 / max(d2, 1e-300):5.1f} err~{d2 / 15:.1e}")
        print(f"  {f:6s} " + "  ".join(line))
