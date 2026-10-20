"""Where does the single puncture lose spherical symmetry? (2026-10-17)

For window-end times t and radii r, evaluate the PINN on a sphere of N directions
centred (a) at the origin and (b) at the tracked puncture position x_p(t), and
decompose the angular variation of each scalar into real spherical-harmonic
degrees l = 1..4 (least squares). Reports, per field, t, r and centre:
  rel   = rms over directions of (f - mean) / max(|mean|, floor)
  P_l   = fraction of the angular variance in degree l (l = 1 dipole: an offset
          centre; l = 4 with cubic pattern: Cartesian-grid / axis bias)
  cub   = fraction of variance in the cubic invariant x^4+y^4+z^4 (l = 4, axis-aligned)
Also the off-form tensor part of gt (deviation from a delta + b nn) as 'gt_off'.

Usage: JAX_PLATFORMS=cpu python3 asym_diag.py run_dir --out prefix
"""
import argparse, os, sys
import numpy as np
from scipy.special import sph_harm_y

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SCALARS = {"chi": 12, "Khat": 13, "Theta": 14, "alpha": 18}


def fib_dirs(n):
    i = np.arange(n) + 0.5
    ph = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)


def real_sh_basis(dirs, lmax=4):
    x, y, z = dirs.T
    theta = np.arccos(np.clip(z, -1, 1)); phi = np.arctan2(y, x)
    cols, deg = [], []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, abs(m), theta, phi)
            v = np.sqrt(2) * (Y.imag if m < 0 else Y.real) if m != 0 else Y.real
            cols.append(v); deg.append(l)
    return np.array(cols).T, np.array(deg)


def decompose(f, B, deg, cub):
    coef, *_ = np.linalg.lstsq(B, f, rcond=None)
    var = {l: float(np.sum(coef[deg == l] ** 2)) for l in range(1, deg.max() + 1)}
    tot = sum(var.values()) or 1e-300
    fc = f - f.mean()
    c = cub - cub.mean()
    cub_frac = float((fc @ c) ** 2 / (c @ c) / max(fc @ fc, 1e-300))
    return {l: v / tot for l, v in var.items()}, cub_frac


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--out", default="asym")
    p.add_argument("--times", default="2.5,5,7.5,10")
    p.add_argument("--radii", default="0.05,0.1,0.2,0.3,0.5,1,2,3,5,10")
    p.add_argument("--n_dir", type=int, default=400)
    a = p.parse_args()
    import compare_pinn as cp
    import jax.numpy as jnp
    man, wins = cp.load_run(a.run_dir)
    cfg = man["model_config"]
    ns = cp.module_for(cfg)
    full_output, unpack = ns["full_output"], ns["unpack_sym3"]
    track = np.loadtxt(os.path.join(a.run_dir, "puncture_track.txt"))

    def eval_window(w, t, xyz):
        fn = ns["apply_fn"]
        fn, pr = ns["_wrap_moving_chi_bg_if_needed"](fn, w["params"], w.get("traj_tt"), w.get("traj_xx"))
        X = jnp.asarray(xyz)
        T = jnp.full((X.shape[0], 1), float(t))
        return np.asarray(full_output(fn, pr, T, X[:, :1], X[:, 1:2], X[:, 2:3],
                                      jnp.array(w["t_start"]), jnp.array(w["t_end"])))

    dirs = fib_dirs(a.n_dir)
    B, deg = real_sh_basis(dirs)
    cub = np.sum(dirs ** 4, axis=1)
    nn = np.einsum("ni,nj->nij", dirs, dirs)
    e = np.cross(dirs, np.where(np.abs(dirs[:, 2:3]) < 0.9, [[0, 0, 1.0]], [[1.0, 0, 0]]))
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    rows = []
    for t in [float(s) for s in a.times.split(",")]:
        w = min(wins, key=lambda w_: abs(w_["t_end"] - t))
        k = int(np.argmin(np.abs(track[:, 0] - t)))
        xp = np.array([track[k, 1], track[k, 2], 0.0])
        for cname, c0 in (("origin", np.zeros(3)), ("puncture", xp)):
            for r in [float(s) for s in a.radii.split(",")]:
                out = eval_window(w, t, c0 + r * dirs)
                fields = {k_: out[:, i] for k_, i in SCALARS.items()}
                gt = np.asarray(unpack(jnp.asarray(out[:, 0:6])))
                aa = np.einsum("ni,nij,nj->n", e, gt, e)
                bb = np.einsum("ni,nij,nj->n", dirs, gt, dirs) - aa
                off = gt - (aa[:, None, None] * np.eye(3) + bb[:, None, None] * nn)
                fields["gt_off"] = np.sqrt(np.sum(off ** 2, axis=(1, 2)))
                for name, f in fields.items():
                    mean = float(f.mean())
                    if name == "gt_off":
                        rel = mean          # already a deviation, absolute
                        P, cf = {l: np.nan for l in range(1, 5)}, np.nan
                    else:
                        rel = float(f.std() / max(abs(mean), 1e-3))
                        P, cf = decompose(f, B, deg, cub)
                    rows.append((t, cname, r, name, mean, rel, P[1], P[2], P[3], P[4], cf))
        print(f"t={t} done (x_p = {xp[0]:.4f}, {xp[1]:.4f})", flush=True)
    with open(a.out + ".csv", "w") as fh:
        fh.write("t,centre,r,field,mean,rel,P1,P2,P3,P4,cubic\n")
        for row in rows:
            fh.write("%g,%s,%g,%s,%.6e,%.4e,%.3f,%.3f,%.3f,%.3f,%.3f\n" % row)


if __name__ == "__main__":
    main()
