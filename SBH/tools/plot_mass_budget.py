"""Where does the Misner-Sharp mass error come from? (2026-10-21)

    python tools/plot_mass_budget.py RUN.npz OUT.png [--title T]

RUN.npz from `tools/mass_function.py --both_ends` (every window at t_start and t_end, along +x and -x).
With m_k(t) the mass of window k's network at time t, the error at the end of window K is
    m_K(t_K) - M = [m_0(0) - M]                                   (initial-data fit)
                 + sum_k [m_k(t_{k+1}) - m_k(t_k)]                (built up INSIDE the windows)
                 + sum_k [m_{k+1}(t_{k+1}) - m_k(t_{k+1})]        (JUMPS at the hand-offs)
exactly (telescoping), at every radius. Prints that budget at a few radii and plots
(a) m/M - 1 vs t at those radii (each window its own segment, so the jumps show),
(b) the in-window and hand-off sums vs r, (c) the radial density d(m - M)/d ln r of the final error.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("npz"); ap.add_argument("out"); ap.add_argument("--title", default="")
a = ap.parse_args()
z = np.load(a.npz)
M, r = float(z["M"]), z["r"]
m = 0.5 * (z["m_plus"] + z["m_minus"]) / M - 1.0          # (n_eval, n_r), ray average
win, side, t = z["window"], z["side"], z["t"]
W = int(win.max()) + 1
start = np.array([m[(win == k) & (side == 0)][0] for k in range(W)])
end = np.array([m[(win == k) & (side == 1)][0] for k in range(W)])
t_end = np.array([t[(win == k) & (side == 1)][0] for k in range(W)])
initial = start[0]
inside = (end - start).sum(0)
handoff = (start[1:] - end[:-1]).sum(0) if W > 1 else np.zeros_like(initial)
final = end[-1]
assert np.allclose(initial + inside + handoff, final)

radii = [0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
print(f"{a.title}: {W} windows to t = {t_end[-1]:g}; mass error budget m/M - 1 (ray average)")
print(f"{'r':>6s} {'initial':>10s} {'in-window':>10s} {'hand-offs':>10s} {'= final':>10s}")
for rr in radii:
    i = int(np.argmin(np.abs(r - rr)))
    print(f"{r[i]:6.2f} {initial[i]:+10.2e} {inside[i]:+10.2e} {handoff[i]:+10.2e} {final[i]:+10.2e}")

fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
cols = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
ax = axs[0]
for rr, c in zip(radii, cols):
    i = int(np.argmin(np.abs(r - rr)))
    for k in range(W):
        ts = [t[(win == k) & (side == 0)][0], t_end[k]]
        ax.plot(ts, [start[k, i], end[k, i]], color=c, lw=1.6, label=f"r = {rr:g}" if k == 0 else None)
ax.axhline(0, color="#6b6a65", lw=0.8); ax.set_xlabel("t"); ax.set_ylabel("m/M − 1")
ax.set_title("(a) mass error, each window its own segment", loc="left", fontsize=10); ax.legend(fontsize=8)
ax = axs[1]
ax.plot(r, inside, color="#2a78d6", label="built up inside windows")
ax.plot(r, handoff, color="#eb6834", ls="--", label="jumps at hand-offs")
ax.plot(r, initial, color="#1baf7a", ls=":", label="initial-data fit (t = 0)")
ax.plot(r, final, color="#1f1f1e", lw=1.0, label="total at the last window end")
ax.axhline(0, color="#6b6a65", lw=0.8); ax.set_xscale("log"); ax.set_xlim(0.1, 20)
lim = 1.3 * np.max(np.abs(final[(r > 0.1) & (r < 20)])); ax.set_ylim(-lim, lim)
ax.set_xlabel("r"); ax.set_ylabel("contribution to m/M − 1")
ax.set_title("(b) budget of the final error, by radius", loc="left", fontsize=10); ax.legend(fontsize=8)
ax = axs[2]
lr = np.log(r)
for k, c in zip(np.linspace(0, W - 1, 4).astype(int), cols[1::1]):
    dens = np.gradient(end[k] - initial, lr)
    ax.plot(r, dens, color=c, label=f"t = {t_end[k]:g}")
ax.axhline(0, color="#6b6a65", lw=0.8); ax.set_xscale("log"); ax.set_xlim(0.1, 20)
sel = (r > 0.15) & (r < 20)
lim = 1.3 * np.max(np.abs(np.gradient(end[-1] - initial, lr)[sel])); ax.set_ylim(-lim, lim)
ax.set_xlabel("r"); ax.set_ylabel("d(m − m(t=0))/M / d ln r")
ax.set_title("(c) where the error is produced (per unit ln r)", loc="left", fontsize=10); ax.legend(fontsize=8)
for x in axs:
    x.grid(True, color="#e4e3de", lw=0.6)
    for sp in ("top", "right"):
        x.spines[sp].set_visible(False)
fig.suptitle(a.title, x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(a.out, dpi=130)
