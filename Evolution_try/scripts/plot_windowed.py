import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Loss and error of the two windowed T = 20 runs.
#
#   <python> scripts/plot_windowed.py [runA runB ...]
#
# Reads runs/<name>/windows.json (per-window summary) and fields.npz (the solution
# on the snapshot grid), so it needs the run directory present locally -- copy it
# from the login node with
#   scp -r serafin.ccad.unc.edu.ar:/home/reula/Julia/PINN/Evolution_try/runs/<name> runs/
RUNS = [("T20win_hard", "hard hand-over (frozen into the ansatz)", "#c0392b", "o"),
        ("T20win_soft", "soft hand-over (penalty, w_ic=100)",      "#1f6feb", "s")]

def load(name):
    w = json.load(open(f"runs/{name}/windows.json"))
    f = np.load(f"runs/{name}/fields.npz")
    # fields.npz["times"] already holds the snapshot TIMES, in days of this
    # project's unit time -- not indices.  Scaling them again put the x axis at
    # 2T and made panel (b) look like a 40-long run.
    t = np.asarray(f["times"], dtype=float)
    rel = np.linalg.norm(f["u"] - f["exact"], axis=1) / np.linalg.norm(f["exact"], axis=1)
    wins = w["windows"]
    return dict(
        rel_t=t, rel=rel,
        t_end=np.array([r["t1"] for r in wins]),
        loss=np.array([r["final_loss"] for r in wins]),
        inherited=np.array([r["ic_rel_l2"] for r in wins]),
        achieved=np.array([r["rel_l2"] for r in wins]),
        amp=np.array([r["amplification"] for r in wins]),
        overall=w["metrics"]["rel_l2_space_time"], wall=w["wall"])

d = {n: load(n) for n, _, _, _ in RUNS}

fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))
fig.suptitle("Windowed T = 20, 10 windows of dt = 2 — DSGNAR, 2201 parameters, n_coll = 2201\n"
             "same everything except how a window inherits the previous one",
             fontsize=13, y=0.985)

# 1 -- loss per window
a = ax[0, 0]
for n, lab, c, mk in RUNS:
    a.semilogy(d[n]["t_end"], np.maximum(d[n]["loss"], 1e-16), mk + "-", color=c, label=lab, lw=1.6, ms=5)
a.set_xlabel("t at the end of the window"); a.set_ylabel("final training loss in the window")
a.set_title("(a) each window's own converged loss")
a.grid(alpha=.3, which="both"); a.legend(fontsize=9)

# 2 -- error vs t
a = ax[0, 1]
for n, lab, c, mk in RUNS:
    a.semilogy(d[n]["rel_t"], np.maximum(d[n]["rel"], 1e-16), mk + "-", color=c, label=lab, lw=1.6, ms=5)
    a.annotate(f"{d[n]['overall']:.1e}", (d[n]["rel_t"][-1], d[n]["rel"][-1]),
               textcoords="offset points", xytext=(6, -2), color=c, fontsize=9)
a.set_xlabel("t"); a.set_ylabel(r"relative $L_2$ error vs the exact solution")
a.set_title("(b) error along the integration  (label = space-time rel $L_2$)")
a.grid(alpha=.3, which="both"); a.legend(fontsize=9)

# 3 -- inherited vs achieved
a = ax[1, 0]
for n, lab, c, mk in RUNS:
    inh = np.where(d[n]["inherited"] > 0, d[n]["inherited"], np.nan)
    a.semilogy(d[n]["t_end"], inh, mk + "--", color=c, alpha=.45,
               lw=1.3, ms=4, label=f"{lab} — inherited")   # window 1 inherits the exact IC
    a.semilogy(d[n]["t_end"], np.maximum(d[n]["achieved"], 1e-16), mk + "-", color=c,
               lw=1.7, ms=5, label=f"{lab} — achieved")
a.set_xlabel("t at the end of the window"); a.set_ylabel(r"relative $L_2$")
a.set_title("(c) what a window was handed, and what it left behind")
a.grid(alpha=.3, which="both"); a.legend(fontsize=8)

# 4 -- amplification
a = ax[1, 1]
w = 0.38
idx = np.arange(len(d["T20win_hard"]["amp"]))
for i, (n, lab, c, mk) in enumerate(RUNS):
    a.bar(idx + (i - 0.5) * w, d[n]["amp"], width=w, color=c, alpha=.85, label=lab)
a.axhline(1.0, color="k", lw=1.2, ls=":")
a.text(len(idx) - 0.5, 1.15, "×1 — neither grows nor shrinks", ha="right", fontsize=8)
a.set_xticks(idx); a.set_xticklabels([f"{int(v)}" for v in d["T20win_hard"]["t_end"]], fontsize=8)
a.set_xlabel("t at the end of the window")
a.set_ylabel("error out / error in, per hand-over")
a.set_title("(d) amplification at each hand-over — the compounding rate")
a.grid(alpha=.3, axis="y"); a.legend(fontsize=9)

fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig("plots/windowed_T20_hard_vs_soft.png", dpi=150)
print("wrote plots/windowed_T20_hard_vs_soft.png")
for n, lab, _, _ in RUNS:
    print(f"{n:14s} space-time rel L2 {d[n]['overall']:.3e}   wall {d[n]['wall']:.0f} s   "
          f"mean amplification {np.nanmean(d[n]['amp'][1:]):.2f}")
