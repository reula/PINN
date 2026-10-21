"""Why DSGNAR stops: is its step still a descent direction near the floor? (2026-10-21)

    JAX_ENABLE_X64=1 python tools/sketch_noise.py [--qn_its 300] [--draws 20]

For a small BH problem, at the initial network and after `qn_its` SSBroyden iterations, draws
`draws` independent sketches (CountSketch C on the residual rows, SRCT Omega S on the parameters,
s = n/3) and compares, in the small-radius limit that decides the trust region's fate, the
directions
    d_ds  = Omega S Omega S^T J^T C^T C r     (DSGNAR: residual AND parameters sketched)
    d_ex  = Omega S Omega S^T J^T r           (parameters sketched, exact residual)
with the true gradient g = J^T r: cos(d, g) and how often g.d < 0, i.e. the step goes UPHILL
(d is the descent direction's negative: the step is -d). Also prints the "relative gradient"
|J^T r| / (|J|_F |r|), which the residual-sketch noise ~ 1/sqrt(s) is compared with.
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_defs import load
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--qn_its", type=int, default=300)
ap.add_argument("--draws", type=int, default=20)
a, _ = ap.parse_known_args()

ns = load(["--Nint", "1500", "--NI", "300", "--Nb", "600", "--HIDDEN", "16", "--N_LAYERS", "3",
           "--deriv_mode", "taylor", "--chi_bg_mode", "physical", "--bh_center_mode", "fixed",
           "--n_stars", "1", "--BL_masses", "0.5", "--BL_momenta", "0", "--BL_momenta_y", "0"],
          outdir=tempfile.mkdtemp(prefix="sn_"))
jax, jnp = ns["jax"], ns["jnp"]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dsgnar

model = ns["model"]
key = jax.random.PRNGKey(4)
ws, we = jnp.array(0.0), jnp.array(0.25)
params = jax.tree_util.tree_map(lambda v: v.astype(jnp.float64), model.init(key, *(jnp.ones((8, 1)),) * 4, ws, we))
td = ns["build_train_data"](ws, we, key, star_centers=[(0.0, 0.0, 0.0)])
td = td[0] if isinstance(td[0], tuple) else td
fn = ns["apply_fn"]
args = (ns["lam_bc"], ns["lamg0"], ns["w_pos"], None, None, ns["lam_einstein"], ns["lam_gauge"], ns["r_eps"], *td)
w0, unf = ns["flatten_util"].ravel_pytree(params)
obj = ns["_DsgnarObjective"](fn, unf, args, w0)
n = int(w0.size)
s = n // 3
M = int(obj.residual(w0).shape[0])
loss = jax.jit(lambda w: ns["compute_bfgs_loss"](fn, unf(w), *args))
vg = jax.jit(jax.value_and_grad(lambda w: ns["compute_bfgs_loss"](fn, unf(w), *args)))


def qn(w, its):
    mb = ns["_minimize_mod"]
    r = mb.minimize(loss, vg, w, {"maxiter": its, "gtol": 0.0, "update_method": "ssbroyden2",
                                   "initial_scale": True, "line_search": "wolfe", "c2": 0.9,
                                   "scale_invariant": True, "max_iter": 100, "alphamin": 1e-15})
    return r.x


def analyse(w, tag):
    r = obj.residual(w)                                    # scaled so 0.5 mean(r^2) = loss
    _, vjp = jax.vjp(obj._residual, w)
    g = vjp(r)[0]                                          # J^T r (true gradient direction)
    # |J|_F estimated with 32 random probes
    _, jvp = jax.linearize(obj._residual, w)
    Z = jax.random.normal(key, (32, n))
    JF = float(jnp.sqrt(jnp.mean(jnp.sum(jax.vmap(jvp)(Z) ** 2, axis=1))))
    rel = float(jnp.linalg.norm(g) / (JF * jnp.linalg.norm(r)))
    cds, cex, up = [], [], 0
    for d in range(a.draws):
        k1, k2 = jax.random.split(jax.random.PRNGKey(100 + d))
        srct = dsgnar.make_srct(k1, n, s)
        signs, buckets = dsgnar.count_sketch_draws(k2, M, s, 4, w.dtype)
        CJ = obj.sketch_rows(w, signs, buckets, s, jnp.ones((M,), w.dtype))          # (s, n) = C J
        Cr = dsgnar.apply_count_sketch(r[:, None], signs, buckets, s)[:, 0]          # (s,)
        JtCtCr = CJ.T @ Cr                                                            # J^T C^T C r
        P = lambda v: dsgnar.lift_srct(dsgnar.apply_srct(v[None, :], srct)[0], srct, n)  # Omega S Omega S^T v
        d_ds, d_ex = P(JtCtCr), P(g)
        c1 = float(d_ds @ g / (jnp.linalg.norm(d_ds) * jnp.linalg.norm(g)))
        c2 = float(d_ex @ g / (jnp.linalg.norm(d_ex) * jnp.linalg.norm(g)))
        cds.append(c1); cex.append(c2); up += c1 < 0
    cds, cex = np.array(cds), np.array(cex)
    print(f"{tag:22s} loss {float(loss(w)):.3e}  relative gradient {rel:.2e} (1/sqrt(s) = {1 / np.sqrt(s):.2e})\n"
          f"   DSGNAR step  cos(d, g): mean {cds.mean():+.3f}, min {cds.min():+.3f}, max {cds.max():+.3f}; "
          f"uphill in {up}/{a.draws} draws\n"
          f"   exact-residual step cos(d, g): mean {cex.mean():+.3f}, min {cex.min():+.3f} (never uphill: g.d = |S^T g|^2)", flush=True)


print(f"n = {n} parameters, s = {s}, M = {M} residual rows")
analyse(w0, "initial network")
w1 = qn(w0, a.qn_its)
analyse(w1, f"after {a.qn_its} QN its")
w2 = qn(w1, 4 * a.qn_its)
analyse(w2, f"after {5 * a.qn_its} QN its")
