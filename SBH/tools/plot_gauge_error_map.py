"""Where and when do the network's gauge errors appear, and do they travel as a gauge mode? (2026-11-13)

    python tools/plot_gauge_error_map.py OUT.png REF.npz LABEL=RUN_DIR [LABEL=RUN_DIR ...]

The user (13 Nov): a gauge mode may grow from a small error near the origin, where the solution is not controlled.
One row per run, columns = signed error (network - finite-difference reference, report.npz) of B (beta^r),
g (Ghat^r) and alpha over (r, t), r logarithmic, symmetric-log colour scale (linear below 1e-5).
Overlaid, from the reference REF.npz (the 1D reference of the run: vacuum or scalar), the outgoing characteristics
    shift mode  dr/dt = -B + sqrt(gt^rr)           (black; principal part d_t B = 3/4 g, d_t g ~ 4/3 gt^rr B'')
    lapse mode  dr/dt = -B + sqrt(2 alpha g^rr)    (green dashed; 1+log)
    light       dr/dt = -B + alpha sqrt(g^rr)      (grey dotted)
started at t = 0, 2, 4 from r = 0.06 (shift) and from r = 0.25, 0.6 (lapse, light), and the horizon (magenta).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
from scipy.interpolate import RegularGridInterpolator as RGI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference_1d"))
from sph_reference_scalar import diagnostics                       # noqa: E402

out, fref = sys.argv[1:3]
runs = [s.split("=", 1) for s in sys.argv[3:]]
Z = np.load(fref)
rr, tt, U = Z["r"], Z["t"], Z["U"]
sel = rr < 26
rr, U = rr[sel], U[:, :, sel]
grr = U[:, 4] / (U[:, 0] + U[:, 1])
SPEED = {"shift": -U[:, 9] + 1 / np.sqrt(U[:, 0] + U[:, 1]),
         "lapse": -U[:, 9] + np.sqrt(2 * np.clip(U[:, 8] * grr, 0, None)),
         "light": -U[:, 9] + U[:, 8] * np.sqrt(np.clip(grr, 0, None))}
SPEED = {k: RGI((tt, rr), v, bounds_error=False, fill_value=None) for k, v in SPEED.items()}


def curve(kind, t0, r0, dt=0.005):
    t, r, T, R = t0, r0, [t0], [r0]
    while t < tt[-1] and 0.03 < r < 25:
        k1 = SPEED[kind]((t, r)); k2 = SPEED[kind]((min(t + dt, tt[-1]), r + dt * k1))
        r += 0.5 * dt * (k1 + k2); t += dt
        T.append(t); R.append(r)
    return np.array(R), np.array(T)


pad = lambda u: np.concatenate([u, np.zeros((12 - u.shape[0], u.shape[1]))]) if u.shape[0] < 12 else u
AH = np.array([diagnostics(Z["r"], pad(u))[0] for u in Z["U"]])
CUR = ([("shift", t0, 0.06) for t0 in (0, 2, 4)] + [(k, t0, r0) for k in ("lapse", "light") for t0 in (0, 2, 4)
                                                     for r0 in (0.25, 0.6)])
CUR = [(k, curve(k, t0, r0)) for k, t0, r0 in CUR]
STY = {"shift": dict(c="k", lw=1.1), "lapse": dict(c="#1b7837", lw=1.0, ls="--"), "light": dict(c="0.45", lw=0.9, ls=":")}

plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(len(runs), 3, figsize=(16, 3.9 * len(runs)), squeeze=False)
for row, (lab, d) in zip(ax, runs):
    R = np.load(os.path.join(d, "report.npz"))
    r, t, err = R["r"], R["t"], R["err"]
    for A, (i, nm) in zip(row, ((9, "B (beta^r)"), (7, "g (Ghat^r)"), (8, "alpha"))):
        e = err[:, :, i]
        vmax = 10.0 ** np.ceil(np.log10(np.abs(e[:, r >= 0.05]).max()))
        pc = A.pcolormesh(r, t, e, cmap="RdBu_r", norm=SymLogNorm(1e-5, vmin=-vmax, vmax=vmax), shading="nearest")
        fig.colorbar(pc, ax=A, pad=0.01)
        for k, (cr, ct) in CUR:
            A.plot(cr, ct, **STY[k])
        A.plot(AH, tt, c="m", lw=1.3)
        A.set_xscale("log"); A.set_xlim(0.05, 24); A.set_ylim(0, t.max())
        A.set_title(f"{lab}: error of {nm}"); A.set_xlabel("r"); A.set_ylabel("t")
h = [plt.Line2D([], [], **STY[k], label=k + " mode, outgoing") for k in STY] + [plt.Line2D([], [], c="m", label="horizon")]
ax[0, 0].legend(handles=h, fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(out, dpi=105)
print("wrote", out)
