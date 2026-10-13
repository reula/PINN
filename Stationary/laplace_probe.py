"""Flat-Laplacian probe: solve the reduced problem with either optimiser, on fresh batches.

The problem is the one in `Laplace_Robin.md` and `tests/test_laplace_robin.py`, which is imported
here as the single definition of it (the note documents that choice):

    Delta lam = 0                                 rho_in < |x| < rho_out
    lam       = lam0 + S1 n_z + S2 (3 n_z^2 - 1)/2                 |x| = rho_in
    B_n[lam - lam0] = 0                                            |x| = rho_out

with a shell ratio of 4, where the order of the condition decides the field.  This driver adds the
three things the shipped tests do not have:

* **either optimiser** -- the test's SSBroyden (`train._crunch_minimize`) or the doubly-sketched
  Gauss-Newton phase (`dsgnar.dsgnar_phase`), the latter with an in-phase **resample** so the
  batch is redrawn every `--resample-every` iterations;
* **several independent batches** (`--batches`), each with its own fresh collocation points and a
  FRESH set of test points, because a single draw says nothing about the spread;
* a **hybrid radial law** (`--log-frac`), `sample_shell`'s mixture of log-uniform and
  volume-uniform points, which is what production uses.

It then decomposes the error field into angular sectors, projecting each `l` onto the two
homogeneous radial powers of the flat Laplacian (`rho^l` growing, `rho^-(l+1)` decaying).  That is
the quantity that matters here: `Laplace_Robin.md` Sec. 6.3 measures the order-1 error as a
spurious GROWING DIPOLE with `A = S1/(rho_in + 2 rho_out^3/rho_in^2)`, i.e. proportional to the
data's dipole amplitude -- so running with `--s1 0` asks whether a dipole appears when there is
none in the data.

    ./.venv/bin/python laplace_probe.py --order 1 --s1 0 --s2 0.05 --batches 3
    ./.venv/bin/python laplace_probe.py --order 1 --s1 0 --s2 0.05 --optimizer dsgnar
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))


def load_problem():
    """`tests/test_laplace_robin.py` as a module: the one definition of the reduced problem."""
    path = os.path.join(HERE, "tests", "test_laplace_robin.py")
    spec = importlib.util.spec_from_file_location("laplace_robin_problem", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["laplace_robin_problem"] = mod   # dataclasses look the module up in sys.modules
    spec.loader.exec_module(mod)
    return mod


class FlatObjective:
    """The residual-vector view of the flat loss that `dsgnar_phase` needs.

    Residual vector = [ sqrt(w/n) * rho^2 Delta lam ] ++ [ sqrt(w_bc/n_bc) * (lam - datum) ] ++
    [ sqrt(w_bc/n_bc) * B_n[lam - lam0] ], so that 0.5 * mean(r^2) is exactly the loss the
    SSBroyden path minimises (up to the constant 1/2 that the two conventions differ by).
    """

    def __init__(self, problem, cfg, batch, model, params):
        self.p = problem
        self.cfg = cfg
        self.model = model
        self.coll, self.inner, self.outer = batch
        self.order = cfg._probe_order
        self.w_bc = cfg._probe_w_bc
        self.flat0, self.unflatten = jax.flatten_util.ravel_pytree(params)
        self.n = int(self.flat0.size)
        self.dtype = self.flat0.dtype
        self._res_fn = jax.jit(self._raw_residual)
        self._loss_fn = jax.jit(lambda flat: 0.5 * jnp.mean(self._raw_residual(flat) ** 2))

    def _raw_residual(self, flat):
        p, cfg, order, w_bc = self.p, self.cfg, self.order, self.w_bc
        model = self.model

        def lam(y):
            return model.apply(self.unflatten(flat), y[None, :])[0]

        lap = jax.vmap(lambda y: jnp.trace(jax.hessian(lam)(y)))(self.coll)
        r_in = jax.vmap(lambda y: lam(y) - p.lam_inner_bc(y, cfg))(self.inner)
        r_out = jax.vmap(lambda y: p.robin_operator(lam, y, p.BASE, order, p.LAM0))(self.outer)
        rc = jnp.sum(self.coll * self.coll, axis=1) * lap
        return jnp.concatenate([
            rc * jnp.sqrt(1.0 / self.coll.shape[0]),
            r_in * jnp.sqrt(w_bc / self.inner.shape[0]),
            r_out * jnp.sqrt(w_bc / self.outer.shape[0]),
        ])

    # the names `dsgnar_phase` reads
    _residual = property(lambda self: self._res_fn)
    _loss = property(lambda self: self._loss_fn)

    def residual(self, flat):
        return self._res_fn(jnp.asarray(flat, dtype=self.dtype))

    def loss(self, flat):
        return self._loss_fn(jnp.asarray(flat, dtype=self.dtype))


def draw_batch(problem, cfg, seed, n_coll, n_bnd):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    return (problem.sample_shell(k1, n_coll, cfg),
            problem.sample_sphere(k2, n_bnd, cfg.rho_in),
            problem.sample_sphere(k3, n_bnd, cfg.rho_out))


def decompose(problem, cfg, xs, err, lmax=2):
    """Per angular sector, the two homogeneous radial powers of the flat Laplacian."""
    r = np.asarray(jnp.linalg.norm(xs, axis=1))
    nz = np.asarray(xs[:, 2]) / r
    out = {}
    for l in range(lmax + 1):
        P = np.polynomial.legendre.Legendre.basis(l)(nz)
        A = np.stack([r ** l * P, r ** (-(l + 1)) * P], axis=1)
        coef, *_ = np.linalg.lstsq(A, err, rcond=None)
        out[l] = {"grow": float(coef[0]), "decay": float(coef[1]),
                  "resid": float(np.max(np.abs(A @ coef - err)))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=1, help="order of the Robin condition")
    ap.add_argument("--s1", type=float, default=0.0, help="inner dipole datum (default 0)")
    ap.add_argument("--s2", type=float, default=0.05, help="inner quadrupole datum")
    ap.add_argument("--optimizer", choices=("ssbroyden", "dsgnar"), default="ssbroyden")
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--n-coll", type=int, default=0, help="0 = the test's RATIO4_COLL")
    ap.add_argument("--n-bnd", type=int, default=0, help="0 = the test's RATIO4_BND")
    ap.add_argument("--log-frac", type=float, default=0.4,
                    help="fraction of the shell points drawn log-uniformly; the rest volume-uniform")
    ap.add_argument("--resample-every", type=int, default=100, help="DSGNAR only")
    ap.add_argument("--dsgnar-steps", type=int, default=400)
    ap.add_argument("--dsgnar-sketch", type=int, default=64)
    ap.add_argument("--dsgnar-chunk", type=int, default=16)
    ap.add_argument("--maxiter", type=int, default=0, help="SSBroyden; 0 = the test's QN_MAXITER")
    args = ap.parse_args()

    P = load_problem()
    n_coll = args.n_coll or P.RATIO4_COLL
    n_bnd = args.n_bnd or P.RATIO4_BND
    print(f"flat Laplacian, ratio-4 shell, order {args.order}, S1 = {args.s1:g}, S2 = {args.s2:g}")
    settings = (f"{args.optimizer} steps={args.dsgnar_steps} s={args.dsgnar_sketch} "
                f"resample={args.resample_every}" if args.optimizer == "dsgnar"
                else f"{args.optimizer} maxiter={args.maxiter or P.QN_MAXITER}")
    print(f"optimiser {args.optimizer}, {n_coll} collocation + {2 * n_bnd} boundary points, "
          f"radial hybrid with log fraction {args.log_frac}, {args.batches} independent batches")
    print(f"settings {settings}")

    for b in range(args.batches):
        P.S1, P.S2 = args.s1, args.s2
        cfg = P.cfg_ratio4()
        cfg.radial = "hybrid"
        cfg.radial_log_frac = args.log_frac
        cfg._probe_order = args.order
        cfg._probe_w_bc = P.W_BC
        batch = draw_batch(P, cfg, args.seed0 + b, n_coll, n_bnd)
        model = P.LapNet()
        params = model.init(jax.random.PRNGKey(1), jnp.ones((1, 3)))

        if args.optimizer == "ssbroyden":
            def loss(p):
                lam = lambda y: model.apply(p, y[None, :])[0]
                lap = jax.vmap(lambda y: jnp.trace(jax.hessian(lam)(y)))(batch[0])
                r_in = jax.vmap(lambda y: lam(y) - P.lam_inner_bc(y, cfg))(batch[1])
                r_out = jax.vmap(lambda y: P.robin_operator(lam, y, P.BASE, args.order, P.LAM0))(
                    batch[2])
                return (jnp.mean((jnp.sum(batch[0] * batch[0], axis=1) * lap) ** 2)
                        + P.W_BC * jnp.mean(r_in ** 2) + P.W_BC * jnp.mean(r_out ** 2))

            flat0, unflatten = jax.flatten_util.ravel_pytree(params)
            res = P.crunch_minimize(lambda flat: loss(unflatten(flat)), flat0, args=(),
                                    method="BFGS",
                                    options={"maxiter": args.maxiter or P.QN_MAXITER,
                                             "gtol": P.RATIO4_GTOL,
                                             "initial_H": jnp.eye(flat0.size),
                                             "update_method": "ssbroyden2",
                                             "initial_scale": False})
            solved, iters = unflatten(res.x), int(res.nit)
            detail = "converged" if res.success else f"SSBroyden status {int(res.status)}"
        else:
            from stationary.dsgnar import dsgnar_phase

            cfg.dsgnar_steps = args.dsgnar_steps
            cfg.dsgnar_sketch = args.dsgnar_sketch
            cfg.dsgnar_chunk = args.dsgnar_chunk
            cfg.resample_every = args.resample_every

            def make_obj(seed):
                return FlatObjective(P, cfg, draw_batch(P, cfg, seed, n_coll, n_bnd), model, params)

            obj = make_obj(args.seed0 + b)
            counter = {"k": 0}

            def resample():
                counter["k"] += 1
                return make_obj(args.seed0 + 1000 * b + counter["k"])

            flat, hist, info = dsgnar_phase(obj, obj.flat0, cfg, verbose=False, resample=resample)
            solved, iters = obj.unflatten(flat), int(info["iterations"])
            # DSGNAR stops on its budget or on the radius; it has no convergence flag of
            # its own, so report what it actually did rather than assuming
            detail = (f"stopped: {info['stopped']}, accepted {info['accepted']}/"
                      f"{info['accepted'] + info['rejected']}")

        # measured on FRESH points, never the ones trained on
        xs = P.sample_shell(jax.random.PRNGKey(9 + b), 4096, P.cfg_ratio4())
        got = np.asarray(jax.vmap(lambda y: model.apply(solved, y[None, :])[0])(xs))
        exact = np.asarray(jax.vmap(P.lam_exact_ratio4)(xs))
        err = got - exact
        d = decompose(P, cfg, xs, err)
        print(f"  batch {b}: {iters:5d} iterations ({detail}), max|err| {np.max(np.abs(err)):.3e}")
        for l in (0, 1, 2):
            print(f"      l={l}:  growing rho^{l} {d[l]['grow']:+.3e}   "
                  f"decaying rho^-{l + 1} {d[l]['decay']:+.3e}   (sector fit {d[l]['resid']:.1e})")


if __name__ == "__main__":
    main()
