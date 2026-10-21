"""Misner-Sharp mass function and constraint monitors along a line, for saved windows.

    JAX_ENABLE_X64=1 python tools/mass_function.py RUN_DIR --out RUN.npz [--nr 200] -- <physics flags>

For a single non-spinning puncture of mass M the spacetime is Schwarzschild, so in any spherically
symmetric slicing the Misner-Sharp mass
    m = (R/2) [1 - (d_x R)^2 / gamma_xx + R^2 (K_yy / gamma_yy)^2],   R = x sqrt(gamma_yy)
evaluated on the x-axis through the puncture must equal M at every radius and time; m/M - 1 is a
coordinate-independent error. (gamma_ij = gt_ij/chi, K_ij = (At_ij + gt_ij K/3)/chi, K = Khat + 2 Theta.)
For isotropic Schwarzschild data (t = 0) the formula gives m = M exactly.

At each window end (and at t = 0 from window 0) it evaluates, along +x and -x from the tracked
puncture position (the origin when the run has no track), at radii log-spaced in [0.05, 20]:
m(r), Theta(r) and |Z^i|(r). Saved to an npz: t (n_t,), r (n_r,), m_plus/m_minus, theta_plus/minus,
z_plus/minus (n_t, n_r), center (n_t, 2).
"""
import argparse
import os
import pickle
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_defs import load

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--out", required=True)
ap.add_argument("--nr", type=int, default=200)
ap.add_argument("--rmin", type=float, default=0.05)
ap.add_argument("--rmax", type=float, default=20.0)
ap.add_argument("--both_ends", action="store_true", help="evaluate every window at t_start and t_end")
argv = sys.argv[1:]
rest = argv[argv.index("--") + 1:] if "--" in argv else []
a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)

txt = open(os.path.join(a.run, "code_md5.txt")).read() if os.path.exists(os.path.join(a.run, "code_md5.txt")) else ""
chi_mode = "physical" if "--chi_bg_mode physical" in txt else "moving"
extra = ["--chi_bg_mode", chi_mode] + (["--bh_center_mode", "fixed"] if chi_mode == "physical" else [])
ns = load(rest + extra, outdir=tempfile.mkdtemp(prefix="mf_"))
jax, jnp, np = ns["jax"], ns["jnp"], ns["np"]
GT, AT = ns["GT_SLICE"], ns["AT_SLICE"]
CHI, KH, TH = ns["CHI_IDX"], ns["KHAT_IDX"], ns["THETA_IDX"]
M = float(ns["BL_masses"][0]) if hasattr(ns["BL_masses"], "__len__") else float(ns["BL_masses"])
comp = list(ns["CCZ4_METRIC3_COMPONENTS"])
iXX, iYY = comp.index((0, 0)), comp.index((1, 1))

r = np.geomspace(a.rmin, a.rmax, a.nr)
man = pickle.load(open(os.path.join(a.run, "pinn_checkpoint.pkl"), "rb"))
states = []
for f in man["window_files"]:
    st = pickle.load(open(os.path.join(a.run, f), "rb"))
    st["params"] = jax.tree_util.tree_map(jnp.asarray, st["params"])
    states.append(st)


def profile(st, t, center, sgn):
    tt = jnp.asarray(st["traj_tt"]) if "traj_tt" in st else None
    xx = jnp.asarray(st["traj_xx"]) if "traj_xx" in st else None
    fn, p, ws, we = ns["_network_apply_for_state"](st, tt, xx)
    cx, cy = center
    rr = jnp.asarray(r).reshape(-1, 1)

    def fields(s):                       # s = signed distance along x from the centre, shape (N,1)
        X = cx + s
        Y = jnp.full_like(s, cy)
        Z = jnp.zeros_like(s)
        T = jnp.full_like(s, t)
        return ns["full_output"](fn, p, T, X, Y, Z, ws, we)

    def R_of(s):
        o = fields(s)
        g_yy = o[:, GT.start + iYY] / o[:, CHI]
        return jnp.abs(s[:, 0]) * jnp.sqrt(g_yy)

    s = sgn * rr
    o = fields(s)
    R, dR = jax.jvp(R_of, (s,), (jnp.ones_like(s),))           # dR/ds, s along +-x
    chi = o[:, CHI]
    g_xx = o[:, GT.start + iXX] / chi
    g_yy = o[:, GT.start + iYY] / chi
    K = o[:, KH] + 2.0 * o[:, TH]
    K_yy = (o[:, AT.start + iYY] + o[:, GT.start + iYY] * K / 3.0) / chi
    m = 0.5 * R * (1.0 - dR ** 2 / g_xx + R ** 2 * (K_yy / g_yy) ** 2)
    P = p
    theta = ns["hamiltonian_constraint"](fn, P, jnp.full_like(s, t), cx + s, jnp.full_like(s, cy), jnp.zeros_like(s), ws, we)
    zv = ns["gauge_source_vector"](fn, P, jnp.full_like(s, t), cx + s, jnp.full_like(s, cy), jnp.zeros_like(s), ws, we)
    zn = jnp.sqrt(jnp.sum(jnp.reshape(zv, (zv.shape[0], -1)) ** 2, axis=1))
    return np.asarray(m), np.asarray(theta).reshape(-1), np.asarray(zn)


out = {k: [] for k in ("t", "m_plus", "m_minus", "theta_plus", "theta_minus", "z_plus", "z_minus", "center", "window", "side")}
if a.both_ends:   # every window at its start AND its end: separates in-window growth from hand-off jumps
    evals = [(st, float(st[key]), i, side) for i, st in enumerate(states)
             for key, side in (("t_start", 0), ("t_end", 1))]
else:
    evals = [(states[0], float(states[0]["t_start"]), 0, 0)] + [(st, float(st["t_end"]), i, 1) for i, st in enumerate(states)]
for st, t, wi, side in evals:
    if "traj_xx" in st and "traj_tt" in st:
        tt = np.asarray(st["traj_tt"]); xx = np.asarray(st["traj_xx"])
        center = (float(np.interp(t, tt, xx[0, 0])), float(np.interp(t, tt, xx[0, 1])))
    else:
        center = (0.0, 0.0)
    mp, thp, zp = profile(st, t, center, +1.0)
    mm, thm, zm = profile(st, t, center, -1.0)
    for k, v in (("t", t), ("m_plus", mp), ("m_minus", mm), ("theta_plus", thp), ("theta_minus", thm),
                 ("z_plus", zp), ("z_minus", zm), ("center", center), ("window", wi), ("side", side)):
        out[k].append(v)
    sel = (r >= 0.5) & (r <= 10.0)
    err = max(np.max(np.abs(mp[sel] / M - 1)), np.max(np.abs(mm[sel] / M - 1)))
    print(f"t={t:6.3f}  centre ({center[0]:+.4f},{center[1]:+.4f})  max|m/M-1| on r in [0.5,10]: {err:.3e}", flush=True)
np.savez(a.out, r=r, M=M, **{k: np.asarray(v) for k, v in out.items()})
print("wrote", a.out)
