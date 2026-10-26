"""l-decomposition of the per-ray Misner-Sharp mass error and of the lapse error vs the 1D reference on spheres
# Usage: python3 reference_1d/l_decomp.py  (from the collaborators, 2026-10-26) RUN_DIR WINDOWS(e.g. 7,15,27,39) RADII(e.g. 2,4,6,8,10,12,16)   (single puncture, about the origin)
(PINN_LESSONS 25.24 (a)). Per ray: radial functions a (transverse gt), b, c, chi, K = Khat + 2 Theta along the ray,
m = (R/2)(1 - R'^2 chi/(a+b) + R^2 (c/a + K/3)^2), R = r sqrt(a/chi) (BH_INVARIANTS.md), FD in r. Real spherical harmonics
up to LMAX fitted by least squares on Fibonacci directions; energy share per l."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
os.environ.setdefault("BH_SCRIPT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "EEccz4_BH_puncture.py"))
import compare_pinn as cp, BH_invariants as bi
from scipy.special import sph_harm_y
import jax, jax.numpy as jnp
run = sys.argv[1]; wins_wanted = [int(w) for w in sys.argv[2].split(",")]
radii = [float(r) for r in sys.argv[3].split(",")]
man, wins = cp.load_run(run); ns = bi.module_for(man["model_config"])
ref = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_single_m0.5_dr0.005.npz")); rr, tr, U = ref["r"], ref["t"], ref["U"]
ND, LMAX, hh = 300, 4, 2e-3
i = np.arange(ND) + 0.5; th = np.arccos(1 - 2 * i / ND); ph = np.pi * (1 + 5 ** 0.5) * i
n = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
cols, ls = [], []
for l in range(LMAX + 1):
    for m in range(-l, l + 1):
        Y = sph_harm_y(l, abs(m), th, ph)
        cols.append(np.real(Y) if m >= 0 else np.imag(Y)); ls.append(l)
A = np.stack(cols, 1); ls = np.array(ls)
e = np.cross(n, np.where(np.abs(n[:, 2:3]) < 0.9, [[0, 0, 1.0]], [[1.0, 0, 0]])); e /= np.linalg.norm(e, axis=1, keepdims=True)
def parts(o):
    gt = np.asarray(ns["unpack_sym3"](jnp.asarray(o[:, 0:6]))); At = np.asarray(ns["unpack_sym3"](jnp.asarray(o[:, 6:12])))
    a = np.einsum("ni,nij,nj->n", e, gt, e); b = np.einsum("ni,nij,nj->n", n, gt, n) - a
    c = np.einsum("ni,nij,nj->n", e, At, e)
    return a, b, c, o[:, 12], o[:, 13] + 2 * o[:, 14], o[:, 18]
def lshare(f):
    co, *_ = np.linalg.lstsq(A, f, rcond=None)
    E = np.array([np.sum(co[ls == l] ** 2) for l in range(LMAX + 1)]); res = f - A @ co
    return E / (E.sum() + np.sum(res ** 2) * 4 * np.pi / ND + 1e-300), co[0] / np.sqrt(4 * np.pi)
for wi in wins_wanted:
    w = wins[wi]; t = float(w["t_end"]); k = int(np.argmin(abs(tr - t)))
    fn, p = ns["_wrap_moving_chi_bg_if_needed"](ns["apply_fn"], w["params"], w.get("traj_tt"), w.get("traj_xx"))
    print(f"t = {t:.2f} (window {wi}): per sphere: mean m/M-1, rms over the sphere, energy share l = 0..{LMAX} (+rest) | lapse error: mean, rms, shares")
    for r in radii:
        R = {}
        for s in (-1, 0, 1):
            X = jnp.asarray(n * (r + s * hh)); T = jnp.full((ND, 1), t)
            o = np.asarray(ns["full_output"](fn, p, T, X[:, :1], X[:, 1:2], X[:, 2:3], jnp.array(w["t_start"]), jnp.array(w["t_end"])))
            a, b, c, chi, K, al = parts(o); R[s] = (r + s * hh) * np.sqrt(a / chi)
            if s == 0: A0, B0, C0, CHI, KK, AL = a, b, c, chi, K, al
        Rp = (R[1] - R[-1]) / (2 * hh)
        m = 0.5 * R[0] * (1 - Rp ** 2 * CHI / (A0 + B0) + R[0] ** 2 * (C0 / A0 + KK / 3) ** 2)
        dm = m / 0.5 - 1; da = AL - np.interp(r, rr, U[k, 8])
        sm, m0 = lshare(dm); sa, a0 = lshare(da)
        print(f"  r = {r:5.1f}: m/M-1 mean {dm.mean():+.3f} rms {np.sqrt(np.mean(dm**2)):.3f} min {dm.min():+.3f} | l-shares " + " ".join(f"{q:.2f}" for q in sm)
              + f" | dalpha mean {da.mean():+.1e} rms {np.sqrt(np.mean(da**2)):.1e} | l-shares " + " ".join(f"{q:.2f}" for q in sa), flush=True)
