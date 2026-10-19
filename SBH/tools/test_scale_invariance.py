"""Is the QN optimizer independent of the loss scale? (the Crunch authors' test, 2026-10-19)

    JAX_ENABLE_X64=1 python tools/test_scale_invariance.py [--its 40] [--restarts 3]

Runs the production QN settings (SSBroyden2, strong Wolfe, h0_scaling, H carried
between restarts, one fixed sample) on the BH loss multiplied by K = 1, 1e8 and
1e-8, with Crunch's scale_invariant off and on, and on + --qn_loss_norm, gtol = 0 (the absolute gradient
threshold is a separate issue). An invariant optimizer gives the same parameters
and the same loss/K at every restart; a difference means some absolute epsilon
(on y.s, s.Hs, y.Hy, ...) is acting, i.e. a "floor" that may be a threshold, not
the network.
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_defs import load

ap = argparse.ArgumentParser()
ap.add_argument("--its", type=int, default=40, help="QN iterations per restart")
ap.add_argument("--restarts", type=int, default=3)
a, rest = ap.parse_known_args()

ns = load(["--Nint", "1500", "--NI", "300", "--Nb", "600", "--HIDDEN", "16", "--N_LAYERS", "3"] + rest,
          outdir=tempfile.mkdtemp(prefix="tsi_"))
jax, jnp, np = ns["jax"], ns["jnp"], ns["np"]
mb = ns["_minimize_mod"]
model = ns["model"]
key = jax.random.PRNGKey(5)
ws, we = jnp.array(0.0), jnp.array(0.25)
params = jax.tree_util.tree_map(lambda v: v.astype(jnp.float64),
                                model.init(key, *(jnp.ones((8, 1)),) * 4, ws, we))
centers = ns["find_bh_centers_shift_driven"]()
td = ns["build_train_data"](ws, we, key, star_centers=centers)
td = td[0] if isinstance(td[0], tuple) else td
fn = ns["_wrap_with_moving_chi_bg"](ns["apply_fn"])
tt = jnp.linspace(0.0, 0.25, 9)
xx = jnp.repeat(jnp.array([[c[0], c[1]] for c in centers])[:, :, None], 9, axis=2)
args = (ns["lam_bc"], ns["lamg0"], ns["w_pos"], tt, xx, ns["lam_einstein"], ns["lam_gauge"], ns["r_eps"], *td)
w0, unf = ns["flatten_util"].ravel_pytree(params)
loss = jax.jit(lambda w: ns["compute_bfgs_loss"](fn, unf(w), *args))
vg = jax.jit(jax.value_and_grad(lambda w: ns["compute_bfgs_loss"](fn, unf(w), *args)))


def run(K, invariant, norm=False):
    # norm: --qn_loss_norm, the loss divided by its value at the start (here K * loss(w0))
    c = K / (K * float(loss(w0))) if norm else K
    fv = jax.jit(lambda w: c * loss(w))
    fvg = jax.jit(lambda w: tuple(c * v for v in vg(w)))
    x, H, hist = w0, jnp.eye(w0.size), []
    for r in range(a.restarts):
        opts = {"maxiter": a.its, "gtol": 0.0, "update_method": "ssbroyden2", "initial_scale": True,
                "alphamin": 1e-15, "max_iter": 100, "line_search": "wolfe", "c2": 0.9, "initial_H": H}
        if invariant:
            opts["scale_invariant"] = True
        res = mb.minimize(fv, fvg, x, opts)
        x, H = res.x, (res.hess_inv + res.hess_inv.T) / 2
        hist.append((float(res.fun) / c, int(res.nit), int(res.nskip)))
    return x, hist


for invariant, norm in ((False, False), (True, False), (True, True)):
    print(f"scale_invariant = {invariant}, loss_norm = {norm}")
    x1, h1 = run(1.0, invariant, norm)
    for K in (1.0, 1e8, 1e-8):
        xk, hk = run(K, invariant, norm) if K != 1.0 else (x1, h1)
        dx = float(jnp.max(jnp.abs(xk - x1)) / jnp.max(jnp.abs(x1)))
        line = "  ".join(f"r{i + 1}: {l:.6e} ({n} its, {sk} skipped)" for i, (l, n, sk) in enumerate(hk))
        print(f"  K={K:<6g} loss/K per restart: {line}   max|x-x(K=1)|/max|x| = {dx:.1e}")
