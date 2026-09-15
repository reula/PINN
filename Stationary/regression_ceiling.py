"""Diagnostic: can the network REPRESENT the exact solution at all?

Pure supervised regression of the network outputs onto the exact solution
(no PDE residual involved).  This separates "architecture/optimiser cannot fit the
function" from "the PINN loss is hard to minimise".
"""
import argparse
import json
import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax

from stationary import exact
from stationary.geometry import pack_gamma, pack_sym
from stationary.model import FieldNet, batch_fields
from stationary.problem import Config, sample_shell


def target(x, R0, k):
    ef = exact.exact_fields(R0, k)
    f = jax.vmap(ef)(x)
    return f.h, f.G, f.lam


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--fourier", type=int, default=8)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--outdir", default="runs/regression")
    p.add_argument("--lbfgs-steps", type=int, default=0)
    p.add_argument("--radial", default="uniform")
    a = p.parse_args()

    cfg = Config(R0=1.0, lam0=1.0, radial=a.radial)
    cfg.__post_init__()
    k = exact.k_from_lambda0(cfg.R0, cfg.lam0)
    model = FieldNet(width=a.width, depth=a.depth, fourier=a.fourier,
                     rho_in=cfg.rho_in, rho_out=cfg.rho_out)
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    params = model.init(k1, sample_shell(k2, 8, cfg))
    xs = sample_shell(k2, a.n, cfg)
    th, tG, tlam = target(xs, cfg.R0, k)
    th = th - jnp.eye(3)                     # network predicts the deviation

    def loss_fn(params, xs, th, tG, tlam):
        f = model.apply(params, xs)
        return (jnp.mean((f.h - jnp.eye(3) - th) ** 2)
                + jnp.mean((f.G - tG) ** 2)
                + jnp.mean(jnp.log(f.lam / tlam) ** 2))

    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adam(optax.cosine_decay_schedule(a.lr, a.steps, alpha=0.01)))
    st = opt.init(params)

    @jax.jit
    def step(params, st, xs, th, tG, tlam):
        loss, g = jax.value_and_grad(loss_fn)(params, xs, th, tG, tlam)
        u, st = opt.update(g, st, params)
        return optax.apply_updates(params, u), st, loss

    for it in range(1, a.steps + 1):
        params, st, loss = step(params, st, xs, th, tG, tlam)
        if it % max(a.steps // 10, 1) == 0 or it == 1:
            print(f"[regress {it:6d}] loss={float(loss):.4e}", flush=True)

    if a.lbfgs_steps > 0:
        lf = lambda p_, xx, aa, bb, cc: loss_fn(p_, xx, aa, bb, cc)
        solver = optax.lbfgs(learning_rate=1.0, memory_size=20,
                             linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=30))
        lst = solver.init(params)

        @jax.jit
        def lstep(params, lst, xs, th, tG, tlam):
            value, g = jax.value_and_grad(loss_fn)(params, xs, th, tG, tlam)
            u, lst = solver.update(g, lst, params, value=value, grad=g,
                                   value_fn=lambda q: loss_fn(q, xs, th, tG, tlam))
            return optax.apply_updates(params, u), lst, value

        for it in range(1, a.lbfgs_steps + 1):
            params, lst, loss = lstep(params, lst, xs, th, tG, tlam)
            if it % max(a.lbfgs_steps // 10, 1) == 0 or it == 1:
                print(f"[regress-lbfgs {it:5d}] loss={float(loss):.4e}", flush=True)

    # final errors on a fresh sample
    xv = sample_shell(jax.random.PRNGKey(99), 8192, cfg)
    f = model.apply(params, xv)
    hv, Gv, lamv = target(xv, cfg.R0, k)
    out = {
        "steps": a.steps, "lbfgs_steps": a.lbfgs_steps, "width": a.width, "depth": a.depth, "fourier": a.fourier,
        "max_dh": float(jnp.max(jnp.abs(f.h - hv))),
        "max_dG": float(jnp.max(jnp.abs(f.G - Gv))),
        "max_dlam_rel": float(jnp.max(jnp.abs(f.lam / lamv - 1.0))),
        "final_loss": float(loss),
    }
    print(json.dumps(out, indent=2))
    import pickle
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "params.pkl"), "wb") as fh:
        pickle.dump(jax.tree.map(lambda q: jax.device_get(q), params), fh)
    # error profile vs rho (where is the error concentrated?)
    rr = jnp.linalg.norm(xv, axis=-1)
    err = jnp.max(jnp.abs(f.h - hv), axis=(1, 2))
    bins = jnp.linspace(cfg.rho_in, cfg.rho_out, 13)
    idx = jnp.clip(jnp.digitize(rr, bins) - 1, 0, 11)
    prof = [float(jnp.max(jnp.where(idx == b, err, 0.0))) for b in range(12)]
    print("max|dh| by rho bin:", [round(v, 4) for v in prof])
    print("rho bins:", [round(float(v), 2) for v in bins])
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "regression.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
