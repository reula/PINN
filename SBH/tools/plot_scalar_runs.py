"""Scalar-field 1D runs (sing_1d.py --scalar) against the scalar reference. (2026-11-11)

    python tools/plot_scalar_runs.py OUT.png REF.npz LABEL=RUN_DIR [LABEL=RUN_DIR ...]

RUN_DIR holds report.npz (err (n_t, n_r, 12) = network - reference on a log grid of r; mass = (m - m_ref)/M_ADM)
and, for --tpl_mu runs, mu.csv (window, t, mu, d mu/dt, tau). Panels:
  1. horizon mass M_AH(t): the reference (black), each network (from its fields: reference + err on the report
     grid, same horizon finder), and the template mass mu(t) of the --tpl_mu runs (dashed);
  2. coordinate radius of the horizon r_AH(t), and the template's core radius 0.225 mu/0.5 (dashed);
  3. worst |m - m_ref| / M_ADM over r >= 0.2 against t;
  4-6. rms error of alpha, chi and phi against t in r = 0.2-1 (solid) and 3-10 (dashed).
Prints a table at t = 2, 4, 6, 8.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference_1d"))
from sph_reference_scalar import diagnostics                       # noqa: E402

out, fref = sys.argv[1:3]
runs = [s.split("=", 1) for s in sys.argv[3:]]
Z = np.load(fref)
COLS = ["#d1495b", "#2a6fdb", "#1b9e77", "#e69f00", "#7b3294"]
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
dref = [diagnostics(Z["r"], u) for u in Z["U"]]
ax[0, 0].plot(Z["t"], [d[1] for d in dref], c="k", lw=2, label="finite-difference reference")
ax[0, 1].plot(Z["t"], [d[0] for d in dref], c="k", lw=2, label="finite-difference reference")
rms = lambda e, r, lo, hi: np.sqrt((e[:, (r >= lo) & (r < hi)] ** 2).mean(axis=1))
for (lab, d), c in zip(runs, COLS):
    R = np.load(os.path.join(d, "report.npz"))
    r, t, err, mass = R["r"], R["t"], R["err"], R["mass"]
    ah = []
    for j, T in enumerate(t):
        jr = int(np.argmin(abs(Z["t"] - T)))
        f = err[j].T + np.stack([np.interp(r, Z["r"], Z["U"][jr, i]) for i in range(err.shape[2])])
        ah.append(diagnostics(r, f)[:2])
    ah = np.array(ah)
    ax[0, 0].plot(t, ah[:, 1], c=c, label=lab)
    ax[0, 1].plot(t, ah[:, 0], c=c, label=lab)
    fmu = os.path.join(d, "mu.csv")
    mu = np.loadtxt(fmu, delimiter=",", ndmin=2) if os.path.exists(fmu) else None
    if mu is not None:
        ax[0, 0].plot(mu[:, 1], mu[:, 2], c=c, ls="--", label=lab + ": template mass mu")
        ax[0, 1].plot(mu[:, 1], 0.225 * mu[:, 2] / 0.5, c=c, ls="--", label=lab + ": core radius")
    worst = np.abs(mass[:, r >= 0.2]).max(axis=1)
    ax[0, 2].semilogy(t, worst, c=c, label=lab)
    for A, i in zip(ax[1], (8, 4, 10)):
        A.semilogy(t, rms(err[:, :, i], r, 0.2, 1.0), c=c, label=lab)
        A.semilogy(t, rms(err[:, :, i], r, 3.0, 10.0), c=c, ls="--")
    print(lab)
    for T in (2, 4, 6, 8):
        if t.max() < T - 1e-9:
            continue
        j = int(np.argmin(abs(t - T)))
        jr = int(np.argmin(abs(Z["t"] - T)))
        bands = ((0.2, 1.0), (1.0, 3.0), (3.0, 10.0))
        txt = " | ".join(nm + " " + " ".join(f"{rms(err[j:j + 1, :, i], r, lo, hi)[0]:.0e}" for lo, hi in bands)
                         for nm, i in (("alpha", 8), ("chi", 4), ("Khat", 5), ("phi", 10)))
        print(f"  t = {T}: {txt} | worst mass {worst[:j + 1].max():.1e} | M_AH {ah[j, 1]:.5f} (ref {dref[jr][1]:.5f})"
              + (f" mu {mu[int(np.argmin(abs(mu[:, 1] - T))), 2]:.5f}" if mu is not None else ""))
for A, ttl in zip(ax.ravel(), ("horizon mass M_AH(t)", "coordinate radius of the horizon r_AH(t)",
                               "worst |m - m_ref| / M_ADM, r >= 0.2", "rms error of alpha: r = 0.2-1 solid, 3-10 dashed",
                               "rms error of chi", "rms error of phi")):
    A.set_title(ttl); A.set_xlabel("t"); A.grid(alpha=0.3)
ax[0, 0].legend(fontsize=7); ax[1, 0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(out, dpi=110)
print("wrote", out)
