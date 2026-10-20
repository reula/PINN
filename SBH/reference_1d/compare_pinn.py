"""Compare a single-puncture PINN run with the 1D reference (validation only).

    JAX_PLATFORMS=cpu python3 compare_pinn.py RUN_DIR REF.npz [--out PREFIX]

RUN_DIR holds pinn_checkpoint.pkl and the pinn_checkpoint_w####.pkl files of
a run made with --n_stars 1 --BL_masses m --BL_momenta 0 --BL_momenta_y 0.
REF.npz comes from sph_reference.py with the same mass, gauge parameters and
damping. The puncture must be at the origin (the n_stars=1 default).

For every window boundary t_k it evaluates the PINN of the window that ENDS
at t_k (and, as a handoff check, the window that STARTS there) along four rays
(+x, +y, +z and the (1,1,1) diagonal) and projects each output on the
spherically symmetric form used by the reference:
    gt_ij = a delta_ij + b n_i n_j,  At_ij = c delta_ij + d n_i n_j,
    Ghat^i = g n^i,  beta^i = B n^i,  scalars as they are.
Outputs (PREFIX defaults to RUN_DIR/compare_ref):
    PREFIX_errors.csv    per time and field: relative L2 error on
                         [--r_min, --r_max] (ray average), max |error|, and
                         the non-spherical part of the PINN (ray spread and
                         off-form components) as an anisotropy measure
    PREFIX_puncture.csv  alpha and chi at r = --r_punc (0.03, the radius at
                         which puncture_track.txt samples alpha), PINN vs
                         reference
    PREFIX_summary.txt   scoreboard at a few times (alpha near the puncture,
                         banded L2 errors, Theta, anisotropy, puncture drift)
    PREFIX_alpha_puncture.png, PREFIX_profiles.png, PREFIX_errors.png
Differences close to r = R_domain partly reflect the PINN's Sommerfeld
boundary (the reference has none there); keep --r_max well inside.
"""
import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))

FIELDS = ("a", "b", "c", "d", "chi", "Khat", "Theta", "g", "alpha", "B")
# radial bands for the per-band L2 errors (log-spaced radii, so each band is
# weighted by its number of decades, not its volume)
BANDS = {"core": (0.05, 0.5), "mid": (0.5, 5.0), "far": (5.0, 20.0)}
RAYS = {"+x": (1, 0, 0), "+y": (0, 1, 0), "+z": (0, 0, 1), "diag": (1, 1, 1)}


def parse():
    p = argparse.ArgumentParser(description="PINN vs 1D reference, single puncture")
    p.add_argument("run_dir")
    p.add_argument("ref")
    p.add_argument("--out", default=None)
    p.add_argument("--r_min", type=float, default=0.05)
    p.add_argument("--r_max", type=float, default=20.0)
    p.add_argument("--n_r", type=int, default=600, help="radii per ray (log-spaced)")
    p.add_argument("--r_punc", type=float, default=0.03)
    p.add_argument("--profile_times", default="0.25,1.5,3,5,10",
                   help="times (window boundaries) shown in the profile plot")
    return p.parse_args()


def load_run(run_dir):
    with open(os.path.join(run_dir, "pinn_checkpoint.pkl"), "rb") as f:
        man = pickle.load(f)
    wins = []
    for wf in man["window_files"]:
        with open(os.path.join(run_dir, wf), "rb") as f:
            wins.append(pickle.load(f))
    return man, wins


def module_for(cfg):
    from load_module import load
    dom = cfg.get("domain", {})
    argv = ["--N_LAYERS", str(cfg["N_LAYERS"]), "--HIDDEN", str(cfg["HIDDEN"]),
            "--ic_mode", cfg["ic_mode"], "--N_window", str(cfg["N_window"]),
            "--tfinal", repr(float(dom["tfinal"]))]
    if "R" in dom:
        argv += ["--R_domain", repr(float(dom["R"]))]
    if cfg.get("n_stars", 1) != 1:
        raise SystemExit("compare_pinn.py needs a --n_stars 1 run")
    argv += ["--n_stars", "1", "--BL_masses", repr(float(np.ravel(cfg["BL_masses"])[0])),
             "--BL_momenta", "0", "--BL_momenta_y", "0"]
    if "chi_bg_mode" in cfg:
        argv += ["--chi_bg_mode", cfg["chi_bg_mode"]]
    if "bh_center_mode" in cfg:
        argv += ["--bh_center_mode", cfg["bh_center_mode"]]
    return load(argv)


def main():
    a = parse()
    out = a.out or os.path.join(a.run_dir, "compare_ref")
    man, wins = load_run(a.run_dir)
    cfg = man["model_config"]
    for key in ("BL_momenta", "BL_momenta_y"):
        if key in cfg and np.any(np.abs(np.ravel(cfg[key])) > 0):
            raise SystemExit(f"{key} = {cfg[key]}: the reference is for a non-boosted puncture")
    ns = module_for(cfg)
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = np.load(a.ref)
    m_run = float(np.ravel(cfg["BL_masses"])[0])
    if not np.isclose(float(ref["mass"]), m_run):
        raise SystemExit(f"mass mismatch: run {m_run}, reference {float(ref['mass'])}")
    r_ref, t_ref, U_ref = ref["r"], ref["t"], ref["U"]

    radii = np.geomspace(a.r_min, a.r_max, a.n_r)
    full_output = ns["full_output"]
    unpack = ns["unpack_sym3"]

    def eval_window(w, t, xyz):
        # Same as the BH code's _get_eval_fn (defined after the training
        # loop, so not available through load_module): hard-IC wrapper if
        # needed, then the moving chi background from this window's track.
        if w.get("ic_mode", cfg["ic_mode"]) == "hard":
            fn = ns["make_hard_ic_apply_fn"](ns["model"].apply, w["t_start"], w["t_end"],
                                             ic_history=w.get("ic_history"))
        else:
            fn = ns["apply_fn"]
        fn, p = ns["_wrap_moving_chi_bg_if_needed"](fn, w["params"], w.get("traj_tt"), w.get("traj_xx"))
        ws = jnp.array(w["t_start"], dtype=jnp.float64)
        we = jnp.array(w["t_end"], dtype=jnp.float64)
        X = jnp.asarray(xyz)
        T = jnp.full((X.shape[0], 1), float(t))
        return np.asarray(full_output(fn, p, T, X[:, :1], X[:, 1:2], X[:, 2:3], ws, we))

    def radial(outp, n):
        """Project (N,22) outputs at points r*n on the spherical form;
        returns (10,N) radial functions and (N,) off-form magnitude."""
        gt = np.asarray(unpack(jnp.asarray(outp[:, 0:6])))
        At = np.asarray(unpack(jnp.asarray(outp[:, 6:12])))
        e = np.cross(n, [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0])
        e /= np.linalg.norm(e)
        nn = np.outer(n, n)
        def ab(T):
            a_ = np.einsum("i,nij,j->n", e, T, e)
            b_ = np.einsum("i,nij,j->n", n, T, n) - a_
            off = T - (a_[:, None, None] * np.eye(3) + b_[:, None, None] * nn)
            return a_, b_, np.sqrt(np.sum(off**2, axis=(1, 2)))
        A, Bb, o1 = ab(gt)
        C, D, o2 = ab(At)
        G = outp[:, 15:18] @ n
        Bt = outp[:, 19:22] @ n
        o3 = np.linalg.norm(outp[:, 15:18] - G[:, None] * n, axis=1)
        o4 = np.linalg.norm(outp[:, 19:22] - Bt[:, None] * n, axis=1)
        U = np.stack([A, Bb, C, D, outp[:, 12], outp[:, 13], outp[:, 14], G, outp[:, 18], Bt])
        return U, np.stack([o1, o2, o3, o4])

    def ref_at(t, r):
        k = int(np.argmin(np.abs(t_ref - t)))
        if abs(t_ref[k] - t) > 1e-9:
            return None
        return np.stack([np.interp(r, r_ref, U_ref[k, i]) for i in range(10)])

    rows, prow = [], []
    profiles = {}
    t_edges = sorted({float(w["t_end"]) for w in wins})
    by_end = {float(w["t_end"]): w for w in wins}
    by_start = {float(w["t_start"]): w for w in wins}
    mask = (radii >= a.r_min) & (radii <= a.r_max)
    for t in [0.0] + t_edges:
        R = ref_at(t, radii)
        if R is None:
            print(f"t = {t}: no reference snapshot, skipped")
            continue
        for side, w in (("end", by_end.get(t)), ("start", by_start.get(t))):
            if w is None:
                continue
            per_ray = []
            off = []
            for name, d in RAYS.items():
                n = np.asarray(d, float) / np.linalg.norm(d)
                U, o = radial(eval_window(w, t, radii[:, None] * n[None, :]), n)
                per_ray.append(U)
                off.append(o)
            P = np.stack(per_ray)                       # (rays, 10, n_r)
            Pm = P.mean(axis=0)
            spread = np.abs(P - Pm[None]).max(axis=0)   # ray-to-ray spread
            offm = np.stack(off).max(axis=0)            # (4, n_r)
            if side == "end":
                profiles[t] = Pm
            for i, f in enumerate(FIELDS):
                err = P[:, i, mask] - R[i, mask][None]
                l2 = np.sqrt(np.mean(err**2))
                scale = np.sqrt(np.mean(R[i, mask]**2))
                band = {}
                for bname, (lo, hi) in BANDS.items():
                    bm = (radii >= lo) & (radii < hi)
                    band["l2_" + bname] = np.sqrt(np.mean((P[:, i, bm] - R[i, bm][None])**2)) if bm.any() else np.nan
                rows.append(dict(t=t, side=side, field=f, l2=l2,
                                 rel_l2=l2 / scale if scale > 1e-12 else np.nan,
                                 max_abs=np.abs(err).max(), ray_spread=spread[i, mask].max(), **band))
            rows.append(dict(t=t, side=side, field="off_form",
                             l2=np.nan, rel_l2=np.nan, max_abs=offm[:, mask].max(), ray_spread=np.nan,
                             **{"l2_" + b: np.nan for b in BANDS}))
            Up, _ = radial(eval_window(w, t, np.array([[0.0, 0.0, a.r_punc]])), np.array([0.0, 0.0, 1.0]))
            Rp = ref_at(t, np.array([a.r_punc]))
            prow.append(dict(t=t, side=side, alpha_pinn=Up[8, 0], alpha_ref=Rp[8, 0],
                             chi_pinn=Up[4, 0], chi_ref=Rp[4, 0]))

    import csv
    with open(out + "_errors.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    with open(out + "_puncture.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(prow[0].keys()))
        wr.writeheader(); wr.writerows(prow)

    # alpha near the puncture vs time
    pe = [p for p in prow if p["side"] == "end"]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    k_r = int(np.argmin(np.abs(r_ref - a.r_punc)))
    ax.plot(t_ref, [np.interp(a.r_punc, r_ref, U_ref[k, 8]) for k in range(len(t_ref))], "k-", label="1D reference")
    ax.plot([p["t"] for p in pe], [p["alpha_pinn"] for p in pe], "o-", ms=3, label="PINN (window end)")
    ps = [p for p in prow if p["side"] == "start"]
    ax.plot([p["t"] for p in ps], [p["alpha_pinn"] for p in ps], "x", ms=4, label="PINN (next window start)")
    ax.set_xlabel("t"); ax.set_ylabel(f"alpha at r = {a.r_punc}"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out + "_alpha_puncture.png", dpi=130); plt.close(fig)

    # profiles
    want = [float(s) for s in a.profile_times.split(",")]
    tsel = [t for t in want if t in profiles]
    fig, axs = plt.subplots(2, 5, figsize=(17, 6.5))
    for i, (f, ax) in enumerate(zip(FIELDS, axs.ravel())):
        for j, t in enumerate(tsel):
            R = ref_at(t, radii)
            ax.plot(radii, R[i], "-", color=f"C{j}", lw=1, label=f"ref t={t:g}")
            ax.plot(radii, profiles[t][i], "--", color=f"C{j}", lw=1, label=f"PINN t={t:g}")
        ax.set_xscale("log"); ax.set_title(f); ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out + "_profiles.png", dpi=110); plt.close(fig)

    # error vs time
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, f in enumerate(FIELDS):
        rr = [x for x in rows if x["field"] == f and x["side"] == "end"]
        ax.semilogy([x["t"] for x in rr], [x["l2"] for x in rr], label=f, color=f"C{i}")
    ax.set_xlabel("t"); ax.set_ylabel(f"L2 error on r in [{a.r_min:g}, {a.r_max:g}]")
    ax.legend(ncol=2, fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out + "_errors.png", dpi=130); plt.close(fig)
    # scoreboard: a few numbers per time, for comparing runs at a glance
    track = None
    tp = os.path.join(a.run_dir, "puncture_track.txt")
    if os.path.exists(tp):
        track = np.loadtxt(tp)
    lines = [f"PINN vs 1D reference: {os.path.abspath(a.run_dir)} vs {a.ref}",
             f"L2 errors over log-spaced radii per band core [0.05,0.5), mid [0.5,5), far [5,20); window ends",
             f"{'t':>6} {'a(r=.03) pinn/ref':>18} {'alpha core':>10} {'alpha mid':>9} {'chi core':>9} "
             f"{'Khat core':>9} {'Theta max':>9} {'gt(a) core':>10} {'off_form':>8} {'|x_punc|':>8}"]
    for t in sorted({r["t"] for r in rows}):
        if t not in (0.0, 0.25, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0) and t != max(r["t"] for r in rows):
            continue
        side = "start" if t == 0.0 else "end"
        g = {r["field"]: r for r in rows if r["t"] == t and r["side"] == side}
        pp = [p for p in prow if p["t"] == t and p["side"] == side]
        dist = np.nan
        if track is not None:
            k = np.argmin(np.abs(track[:, 0] - t))
            dist = float(np.hypot(track[k, 1], track[k, 2]))
        lines.append(f"{t:6.2f} {pp[0]['alpha_pinn']:8.4f}/{pp[0]['alpha_ref']:.4f} "
                     f"{g['alpha']['l2_core']:10.2e} {g['alpha']['l2_mid']:9.2e} {g['chi']['l2_core']:9.2e} "
                     f"{g['Khat']['l2_core']:9.2e} {g['Theta']['max_abs']:9.2e} {g['a']['l2_core']:10.2e} "
                     f"{g['off_form']['max_abs']:8.2e} {dist:8.1e}")
    with open(out + "_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out + "_errors.csv", out + "_puncture.csv", out + "_summary.txt", "and 3 PNGs")
    for p in pe:
        print(f"t={p['t']:6.2f}  alpha(r={a.r_punc}) PINN {p['alpha_pinn']:.4f}  ref {p['alpha_ref']:.4f}")


if __name__ == "__main__":
    main()
