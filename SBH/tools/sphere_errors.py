"""Errors averaged over spheres, as functions of radius and time (2026-10-26).

    JAX_ENABLE_X64=1 python tools/sphere_errors.py RUN_DIR REF.npz --out RUN.npz [--ndir 48 --nr 100] -- <physics flags>

At t = 0 and every window end, evaluates the window's network on spheres of `--nr` log-spaced radii in
[--rmin, --rmax] with `--ndir` Fibonacci directions each, and averages over each sphere:
  * the Misner-Sharp mass error m/M - 1, about the tracked puncture (as tools/mass_function.py, with
    x -> the radial direction n and y -> a tangential direction e):
        m = (R/2) [1 - (d_s R)^2 / gamma_nn + R^2 (K_ee / gamma_ee)^2],   R = s sqrt(gamma_ee);
  * PINN - 1D reference for alpha, chi, Khat and Theta, about the origin (where the reference is).
With --stage2 PKL the run's output plus its stage-2 correction (stage2.py) is evaluated instead.
Saved: t (n_t,), r (n_r,), and for q in mass, alpha, chi, Khat, Theta: q_mean (signed sphere average)
and q_rms (root mean square over the sphere), each (n_t, n_r). Plot with tools/plot_sphere_errors.py.
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
ap.add_argument("ref")
ap.add_argument("--out", required=True)
ap.add_argument("--nr", type=int, default=100)
ap.add_argument("--ndir", type=int, default=48)
ap.add_argument("--rmin", type=float, default=0.1)
ap.add_argument("--rmax", type=float, default=20.0)
ap.add_argument("--stage2", default=None, help="stage2_params.pkl: evaluate the run PLUS its stage-2 correction")
argv = sys.argv[1:]
rest = argv[argv.index("--") + 1:] if "--" in argv else []
a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)

txt = open(os.path.join(a.run, "code_md5.txt")).read() if os.path.exists(os.path.join(a.run, "code_md5.txt")) else ""
chi_mode = "physical" if "--chi_bg_mode physical" in txt else "moving"
extra = ["--chi_bg_mode", chi_mode] + (["--bh_center_mode", "fixed"] if chi_mode == "physical" else [])
ns = load(rest + extra, outdir=tempfile.mkdtemp(prefix="se_"))
jax, jnp, np = ns["jax"], ns["jnp"], ns["np"]
GT, AT = ns["GT_SLICE"], ns["AT_SLICE"]
CHI, KH, TH = ns["CHI_IDX"], ns["KHAT_IDX"], ns["THETA_IDX"]
unpack = ns["unpack_sym3"]
M = float(np.ravel(ns["BL_masses"])[0])
ALPHA = 18                                   # lapse column of full_output (as reference_1d/compare_pinn.py)
corr = None
if a.stage2:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from stage2_net import load_correction
    corr, CFG = load_correction(ns, a.stage2)
    print(f"adding the stage-2 correction {a.stage2} (iteration {CFG['iteration']}, {CFG['windows']} windows)")
REF = {"chi": (CHI, 4), "Khat": (KH, 5), "Theta": (TH, 6), "alpha": (ALPHA, 8)}   # (output column, reference row)

ref = np.load(a.ref)
r_ref, t_ref, U_ref = ref["r"], ref["t"], ref["U"]
r = np.geomspace(a.rmin, a.rmax, a.nr)
k = np.arange(a.ndir) + 0.5                  # Fibonacci sphere
cz = 1.0 - 2.0 * k / a.ndir
ph = np.pi * (1.0 + 5 ** 0.5) * k
n = np.stack([np.sqrt(1 - cz ** 2) * np.cos(ph), np.sqrt(1 - cz ** 2) * np.sin(ph), cz], axis=1)
e = np.cross(n, np.where(np.abs(n[:, 2:3]) < 0.9, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]))
e /= np.linalg.norm(e, axis=1, keepdims=True)
N = np.repeat(n, a.nr, axis=0)               # point j = (direction j // nr, radius j % nr)
E = np.repeat(e, a.nr, axis=0)
S = np.tile(r, a.ndir)[:, None]

man = pickle.load(open(os.path.join(a.run, "pinn_checkpoint.pkl"), "rb"))
states = [pickle.load(open(os.path.join(a.run, f), "rb")) for f in man["window_files"]]
if corr is not None:
    states = states[:CFG["windows"]]
for st in states:
    st["params"] = jax.tree_util.tree_map(jnp.asarray, st["params"])


def evaluate(st, t):
    tt = jnp.asarray(st["traj_tt"]) if "traj_tt" in st else None
    xx = jnp.asarray(st["traj_xx"]) if "traj_xx" in st else None
    fn, p, ws, we = ns["_network_apply_for_state"](st, tt, xx)
    if "traj_xx" in st and "traj_tt" in st:
        c = np.array([np.interp(t, np.asarray(st["traj_tt"]), np.asarray(st["traj_xx"])[0, d]) for d in (0, 1)] + [0.0])
    else:
        c = np.zeros(3)

    def out(s, centre):                      # s: (P,1) distance along N from centre
        X = jnp.asarray(centre)[None, :] + s * jnp.asarray(N)
        T = jnp.full_like(s, t)
        o = ns["full_output"](fn, p, T, X[:, :1], X[:, 1:2], X[:, 2:3], ws, we)
        return o if corr is None else o + corr(T, X[:, :1], X[:, 1:2], X[:, 2:3])

    def metric(o):
        return unpack(o[:, GT]) / o[:, CHI][:, None, None]

    def R_of(s):
        g = metric(out(s, c))
        return s[:, 0] * jnp.sqrt(jnp.einsum("pi,pij,pj->p", E, g, E))

    s = jnp.asarray(S)
    o = out(s, c)
    R, dR = jax.jvp(R_of, (s,), (jnp.ones_like(s),))
    g = metric(o)
    K = o[:, KH] + 2.0 * o[:, TH]
    Kij = (unpack(o[:, AT]) + unpack(o[:, GT]) * K[:, None, None] / 3.0) / o[:, CHI][:, None, None]
    g_nn = jnp.einsum("pi,pij,pj->p", N, g, N)
    g_ee = jnp.einsum("pi,pij,pj->p", E, g, E)
    K_ee = jnp.einsum("pi,pij,pj->p", E, Kij, E)
    m = 0.5 * R * (1.0 - dR ** 2 / g_nn + R ** 2 * (K_ee / g_ee) ** 2)
    res = {"mass": np.asarray(m).reshape(a.ndir, a.nr) / M - 1.0}
    kt = int(np.argmin(np.abs(t_ref - t)))
    assert abs(t_ref[kt] - t) < 1e-9, f"no reference snapshot at t = {t}"
    o0 = np.asarray(out(s, np.zeros(3)))
    for q, (col, row) in REF.items():
        res[q] = o0[:, col].reshape(a.ndir, a.nr) - np.interp(r, r_ref, U_ref[kt, row])[None, :]
    return res


acc = {"t": []}
for st, t in [(states[0], float(states[0]["t_start"]))] + [(st, float(st["t_end"])) for st in states]:
    res = evaluate(st, t)
    acc["t"].append(t)
    for q, v in res.items():
        acc.setdefault(q + "_mean", []).append(v.mean(axis=0))
        acc.setdefault(q + "_rms", []).append(np.sqrt((v ** 2).mean(axis=0)))
    i1 = int(np.argmin(np.abs(r - 1.0)))
    print(f"t = {t:6.3f}  <m/M-1>(r=1) {res['mass'].mean(0)[i1]:+.2e}  min over r {res['mass'].mean(0).min():+.2e}"
          f"  alpha rms(r=1) {np.sqrt((res['alpha'][:, i1] ** 2).mean()):.2e}", flush=True)
np.savez(a.out, r=r, M=M, **{q: np.asarray(v) for q, v in acc.items()})
print("wrote", a.out)
