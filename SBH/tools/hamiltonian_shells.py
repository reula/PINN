"""The true Hamiltonian constraint averaged on spheres, and the mass change it implies (2026-10-26).

    JAX_ENABLE_X64=1 python tools/hamiltonian_shells.py RUN_DIR SPEC [SPEC ...] [--ndir 48] -- <physics flags>

SPEC = window index, optionally followed by s (evaluate at the window's start) or e (end, default):
e.g. `0s 0e 15 39`. For each, on spheres about the origin (single puncture) at radii 0.3 ... 15:
    H = R + K^2 - K_ij K^ij
(gamma_ij = gt_ij / chi, its Ricci scalar R from `z4_bianchi.geometry`, K_ij = (At_ij + gt_ij K / 3) / chi,
K = Khat + 2 Theta) -- NOT the script's `hamiltonian_constraint`, which returns Theta. Prints the sphere
mean <H> (the l = 0 part) and rms of H, and the mass change the mean implies in spherical symmetry,
    dm/dR = R^2 H / 4   (R ~ r far from the hole; integrated from r = 0.5),
to compare with the Misner-Sharp mass error (tools/sphere_errors.py).
"""
import argparse
import os
import pickle
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from load_defs import load

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("specs", nargs="+")
ap.add_argument("--ndir", type=int, default=48)
argv = sys.argv[1:]
rest = argv[argv.index("--") + 1:] if "--" in argv else []
a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)

ns = load(rest, outdir=tempfile.mkdtemp(prefix="hs_"))
jax, jnp, np = ns["jax"], ns["jnp"], ns["np"]
import z4_bianchi as zb

GT, AT, CHI, KH, TH = ns["GT_SLICE"], ns["AT_SLICE"], ns["CHI_IDX"], ns["KHAT_IDX"], ns["THETA_IDX"]
unpack = ns["unpack_sym3"]
M = float(np.ravel(ns["BL_masses"])[0])
man = pickle.load(open(os.path.join(a.run, "pinn_checkpoint.pkl"), "rb"))
k = np.arange(a.ndir) + 0.5
cz = 1 - 2 * k / a.ndir
ph = np.pi * (1 + 5 ** 0.5) * k
n = np.stack([np.sqrt(1 - cz ** 2) * np.cos(ph), np.sqrt(1 - cz ** 2) * np.sin(ph), cz], 1)
rs = np.array([0.3, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15])

for spec in a.specs:
    wi, side = (int(spec[:-1]), spec[-1]) if spec[-1] in "se" else (int(spec), "e")
    st = pickle.load(open(os.path.join(a.run, man["window_files"][wi]), "rb"))
    st["params"] = jax.tree_util.tree_map(jnp.asarray, st["params"])
    tt = jnp.asarray(st["traj_tt"]) if "traj_tt" in st else None
    xx = jnp.asarray(st["traj_xx"]) if "traj_xx" in st else None
    fn, p, ws, we = ns["_network_apply_for_state"](st, tt, xx)
    t = float(st["t_end"] if side == "e" else st["t_start"])

    def out(x3, t=t, fn=fn, p=p, ws=ws, we=we):
        return ns["full_output"](fn, p, jnp.full((1, 1), t), x3[0:1].reshape(1, 1), x3[1:2].reshape(1, 1),
                                 x3[2:3].reshape(1, 1), ws, we)[0]

    def H_at(x3, out=out):
        g, gi, _, _, Ric = zb.geometry(lambda y: unpack(out(y)[GT]) / out(y)[CHI], x3)
        o = out(x3)
        K = o[KH] + 2.0 * o[TH]
        Kij = (unpack(o[AT]) + unpack(o[GT]) * K / 3.0) / o[CHI]
        return jnp.einsum("ij,ij->", gi, Ric) + K ** 2 - jnp.einsum("ij,ij->", Kij, gi @ Kij @ gi)

    Hv = jax.jit(jax.vmap(H_at))
    Hs = [np.asarray(Hv(jnp.asarray(r * n))) for r in rs]
    Hm = np.array([h.mean() for h in Hs])
    Hr = np.array([np.sqrt((h ** 2).mean()) for h in Hs])
    i0 = int(np.argmin(np.abs(rs - 0.5)))
    f = rs ** 2 * Hm / 4
    dm = np.zeros_like(rs)
    dm[i0 + 1:] = np.cumsum(0.5 * (f[i0 + 1:] + f[i0:-1]) * np.diff(rs[i0:]))
    print(f"window {wi} {'start' if side == 's' else 'end'}, t = {t:g}")
    print("   r      " + " ".join(f"{r:>8g}" for r in rs))
    print("   <H>    " + " ".join(f"{v:+8.1e}" for v in Hm))
    print("   rms H  " + " ".join(f"{v:8.1e}" for v in Hr))
    print("   dm/M   " + " ".join(f"{v / M:+8.3f}" if r >= 0.5 else f"{'':>8s}" for r, v in zip(rs, dm)), flush=True)
