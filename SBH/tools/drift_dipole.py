"""What pushes the puncture: the dipole of the shift's source at the tracked puncture (2026-10-26).

    JAX_ENABLE_X64=1 python tools/drift_dipole.py RUN_DIR [--rho 0.05 0.1 0.2 0.4] [--ndir 48] -- <physics flags>

The tracker moves the puncture with dx_p/dt = -beta(x_p), and the shift obeys the Gamma-driver
    d_t beta^i = beta^j d_j beta^i + (3/4) Ghat^i - eta beta^i,
so beta(x_p), and with it the drift, is fed by the l = 1 part of Ghat^i at the puncture (for the exact,
spherical solution every vector is radial and averages to zero over a sphere). CCZ4 writes
    Ghat^i = Gt^i[gt] + 2 Z^i / chi
(Gt^i the conformal metric's own contracted Christoffels, Z^i the momentum-constraint variable), so the
sphere average <Ghat> splits exactly into a METRIC part <Gt> (an asymmetric conformal metric) and a
CONSTRAINT part <2 Z/chi>. At t = 0 and every window end, on spheres of radius rho about the tracked
puncture, prints |<beta>|, |<Ghat>|, |<Gt>|, |<2Z/chi>| and the drift so far.
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
ap.add_argument("--rho", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4])
ap.add_argument("--ndir", type=int, default=48)
ap.add_argument("--every", type=int, default=4, help="every this many windows")
argv = sys.argv[1:]
rest = argv[argv.index("--") + 1:] if "--" in argv else []
a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)

txt = open(os.path.join(a.run, "code_md5.txt")).read() if os.path.exists(os.path.join(a.run, "code_md5.txt")) else ""
chi_mode = "physical" if "--chi_bg_mode physical" in txt else "moving"
extra = ["--chi_bg_mode", chi_mode] + (["--bh_center_mode", "fixed"] if chi_mode == "physical" else [])
ns = load(rest + extra, outdir=tempfile.mkdtemp(prefix="dd_"))
jax, jnp, np = ns["jax"], ns["jnp"], ns["np"]
CHI, GH = ns["CHI_IDX"], ns["GHAT_SLICE"]
BETA = slice(19, 22)                          # shift columns of full_output (as reference_1d/compare_pinn.py)

k = np.arange(a.ndir) + 0.5
cz = 1.0 - 2.0 * k / a.ndir
ph = np.pi * (1.0 + 5 ** 0.5) * k
n = np.stack([np.sqrt(1 - cz ** 2) * np.cos(ph), np.sqrt(1 - cz ** 2) * np.sin(ph), cz], axis=1)

man = pickle.load(open(os.path.join(a.run, "pinn_checkpoint.pkl"), "rb"))
states = [pickle.load(open(os.path.join(a.run, f), "rb")) for f in man["window_files"]]
for st in states:
    st["params"] = jax.tree_util.tree_map(jnp.asarray, st["params"])

print(f"{'t':>6s} {'rho':>5s} {'|<beta>|':>9s} {'|<Ghat>|':>9s} {'|<Gt>|':>9s} {'|<2Z/chi>|':>10s} "
      f"{'cos(Ghat,2Z/chi)':>16s} {'|x_p|':>8s}")
evals = [(states[0], float(states[0]["t_start"]))] + [(st, float(st["t_end"])) for i, st in enumerate(states)
                                                      if (i + 1) % a.every == 0]
for st, t in evals:
    tt = jnp.asarray(st["traj_tt"]) if "traj_tt" in st else None
    xx = jnp.asarray(st["traj_xx"]) if "traj_xx" in st else None
    fn, p, ws, we = ns["_network_apply_for_state"](st, tt, xx)
    if "traj_xx" in st and "traj_tt" in st:
        c = np.array([np.interp(t, np.asarray(st["traj_tt"]), np.asarray(st["traj_xx"])[0, d]) for d in (0, 1)] + [0.0])
    else:
        c = np.zeros(3)
    for rho in a.rho:
        X = jnp.asarray(c[None, :] + rho * n)
        T = jnp.full((a.ndir, 1), t)
        args = (T, X[:, :1], X[:, 1:2], X[:, 2:3], ws, we)
        o = ns["full_output"](fn, p, *args)
        Zv = ns["gauge_source_vector"](fn, p, *args)            # chi (Ghat - Gt) / 2
        beta = np.asarray(o[:, BETA]).mean(0)
        ghat = np.asarray(o[:, GH]).mean(0)
        zc = np.asarray(2.0 * Zv / o[:, CHI][:, None]).mean(0)   # Ghat - Gt
        gt = ghat - zc
        cos = float(ghat @ zc / (np.linalg.norm(ghat) * np.linalg.norm(zc) + 1e-300))
        print(f"{t:6.2f} {rho:5.2f} {np.linalg.norm(beta):9.2e} {np.linalg.norm(ghat):9.2e} {np.linalg.norm(gt):9.2e} "
              f"{np.linalg.norm(zc):10.2e} {cos:16.2f} {np.linalg.norm(c):8.1e}", flush=True)
