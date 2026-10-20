"""Overview figure of a 1D reference run: alpha vs t at fixed radii, and
alpha profiles at a few times.
    python3 plot_reference.py ref_single_m0.5_dr0.005.npz [--check other.npz] [--out fig.png]
--check overlays alpha(r = 0.03, t) of a second run (e.g. a coarser dr) as a
resolution check."""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical slots 1-4
SEQ = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # one-hue ramp, light->dark = early->late

p = argparse.ArgumentParser()
p.add_argument("ref")
p.add_argument("--check", default=None)
p.add_argument("--out", default=None)
a = p.parse_args()
d = np.load(a.ref)
r, t, U = d["r"], d["t"], d["U"]
AL = list(d["fields"]).index("alpha")
def alpha_at(D, R):
    return np.array([CubicSpline(D["r"], D["U"][k, AL])(R) for k in range(len(D["t"]))])

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
for c, R in zip(CAT, (0.03, 0.1, 0.5, 2.0)):
    ax1.plot(t, alpha_at(d, R), color=c, lw=2, label=f"r = {R:g}")
if a.check:
    c2 = np.load(a.check)
    ax1.plot(c2["t"], alpha_at(c2, 0.03), color=CAT[0], lw=1, ls=":", label=f"r = 0.03, dr = {float(c2['dr']):g}")
ax1.set_xlabel("t"); ax1.set_ylabel("alpha"); ax1.set_ylim(0, 1)
ax1.set_title(f"lapse at fixed radius (m = {float(d['mass']):g}, dr = {float(d['dr']):g})", fontsize=9)
ax1.grid(alpha=0.25, lw=0.5); ax1.legend(frameon=False, fontsize=8, loc="center right")
times = [x for x in (0, 1, 2, 3, 5, 10, 20, 25) if x <= t[-1] + 1e-9][:6]
for c, T in zip(SEQ[-len(times):], times):
    k = int(np.argmin(np.abs(t - T)))
    ax2.plot(r, U[k, AL], color=c, lw=2, label=f"t = {T:g}")
ax2.set_xscale("log"); ax2.set_xlim(0.01, 24); ax2.set_ylim(0, 1)
ax2.set_xlabel("r"); ax2.set_ylabel("alpha"); ax2.set_title("lapse profiles", fontsize=9)
ax2.grid(alpha=0.25, lw=0.5); ax2.legend(frameon=False, fontsize=8)
fig.tight_layout()
out = a.out or a.ref.replace(".npz", "_alpha.png")
fig.savefig(out, dpi=140)
print("wrote", out)
