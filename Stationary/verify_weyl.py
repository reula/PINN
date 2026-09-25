"""Production check: does a Weyl two-black-hole solution solve this code's system?

    python -m verify_weyl                      # two equal rods, half-length 1, half-gap 0.5
    python -m verify_weyl --half-length 2 --half-gap 1 --n-quad 400

This is a *production* check, not a unit test: it needs only jax (no Crunch, no GPU, no
training) and about 10 s, and it answers the question "can this code be pointed at a
genuinely non-spherical, two-black-hole configuration at all?".  It is run by
`run_hub.sh --check` and documented in README.md section 7.

The configuration is the symmetric two-rod Israel-Khan solution: two black holes of equal
mass on the axis, held apart by the conical strut between them.  The mapping into this
code's fields, and the exact gauge source, are derived in stationary/weyl.py.

What is checked, with the tolerances the code has to meet:

  1. the mapping is right: the Weyl fields satisfy the code's two geometric equations,
     `Ric(h)_ab = (1/(2 lam^2)) d_a lam d_b lam` and `Delta_h lam = (1/lam)|d lam|^2`,
     in the Weyl chart (ricci and lam_eq residuals at machine precision), while
     compatibility holds by construction since Gamma is built from h;
  2. the gauge is the only failure: the harmonic condition `Gamma = 0` is violated (that
     is the chart, not the physics), and the closed-form inhomogeneous source
     `Gamma^i = (h_rhorho - 1) h^{ij} d_j ln rho` drives it to machine precision, so the
     Weyl solution solves the FULL system with no coordinate transformation;
  3. that closed form agrees with the independent autodiff connection;
  4. the decay exponents are the ones the Robin conditions assume: h - I ~ rho^-2,
     lam - 1 ~ rho^-1, Gamma ~ rho^-3;
  5. the strut exists: k on the axis between the rods differs from its value outside them;
  6. the PRODUCTION loss vanishes on this solution.  Checks 1-5 are statements about the
     fields; the last one closes the loop with the trainer: a candidate that IS the Weyl
     solution is fed to `total_loss` -- the exact function train.py minimises -- with the
     inner data and the Robin source from the reference and the gauge source built from the
     candidate's own metric, and the returned loss must be machine zero.  The same loss with
     the harmonic gauge must NOT vanish, or a loss that ignored the gauge group could pass
     the check for the wrong reason.

Exit status is 0 if every check passes, 1 otherwise.  The Weyl quadrature needs float64, so
this script enables x64 itself (see Config.__post_init__ for why float32 fails).
"""
from __future__ import annotations

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from stationary.geometry import Fields, residuals_batch
from stationary.weyl import (Rods, fields_of, gauge_source_from_metric, gauge_vector,
                             h_cart, k_of, lam_of)

I3 = jnp.eye(3)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--half-length", type=float, default=1.0, help="half-length of each rod")
    p.add_argument("--half-gap", type=float, default=0.5, help="half the gap between the rods")
    p.add_argument("--n-quad", type=int, default=400, help="Gauss-Legendre nodes for k")
    p.add_argument("--n-points", type=int, default=24, help="sample points for the residuals")
    p.add_argument("--rho-in", type=float, default=None,
                   help="inner sampling radius (default 3 x axis extent)")
    p.add_argument("--rho-out", type=float, default=None,
                   help="outer sampling radius (default 300 x axis extent)")
    p.add_argument("--robin-order", type=int, default=2,
                   help="Robin order for the manufactured loss (order n passes multipoles "
                        "l <= n-1; the Weyl solution is l = 0,1,2 so n = 3 is exact)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def sample_shell(key, n, rho_in, rho_out):
    """Log-uniform in rho, uniform directions, so no point lands on the axis."""
    k1, k2 = jax.random.split(key)
    rho = jnp.exp(jax.random.uniform(k1, (n,), minval=jnp.log(rho_in), maxval=jnp.log(rho_out)))
    u = jax.random.normal(k2, (n, 3))
    u = u / jnp.linalg.norm(u, axis=-1, keepdims=True)
    return rho[:, None] * u


def maxabs(v):
    return float(jnp.max(jnp.abs(v)))


def decay_exponent(rhos, values):
    """Slope of log|value| against log rho (least squares)."""
    v = jnp.abs(jnp.asarray(values))
    x = jnp.log(jnp.asarray(rhos))
    y = jnp.log(v)
    xm, ym = x.mean(), y.mean()
    return float(jnp.sum((x - xm) * (y - ym)) / jnp.sum((x - xm) ** 2))


def main(argv=None):
    a = parse_args(argv)
    rods = Rods.symmetric(a.half_length, a.half_gap)
    extent = rods.axis_extent
    rho_in = a.rho_in if a.rho_in is not None else 3.0 * extent
    rho_out = a.rho_out if a.rho_out is not None else 300.0 * extent
    print(f"configuration: {rods}")
    print(f"  two black holes: rods on the axis, mass m = {a.half_length:g} each, "
          f"gap {2 * a.half_gap:g} between them; the strut is on the axis inside the sphere")
    print(f"  inner sphere must clear rho = {extent:g}; sampling rho in [{rho_in:g}, {rho_out:g}]")
    print(f"  quadrature: {a.n_quad} Gauss-Legendre nodes for k\n")

    key = jax.random.PRNGKey(a.seed)
    xs = sample_shell(key, a.n_points, rho_in, rho_out)
    rho = jnp.linalg.norm(xs, axis=1)

    # ---------------------------------------------------------------- residuals
    checks = []
    harmonic, sourced = {}, {}
    for n_quad in (a.n_quad, 2 * a.n_quad):
        f = fields_of(rods, n_quad)
        src = gauge_source_from_metric(lambda y: h_cart(y, rods, n_quad))
        t0 = time.time()
        r = residuals_batch(f, xs)
        r_src = residuals_batch(f, xs, src)
        print(f"residuals with n_quad = {n_quad}  ({time.time() - t0:.1f} s)")
        for name in ("compat", "ricci", "gauge", "lam_eq"):
            v = r[name]
            w = rho if name in ("compat", "gauge") else rho**2
            scaled = maxabs(v * w.reshape(w.shape + (1,) * (v.ndim - 1)))
            print(f"    {name:7s} max|.| = {maxabs(v):.3e}   rms = "
                  f"{float(jnp.sqrt(jnp.mean(v**2))):.3e}   rho-scaled max = {scaled:.3e}")
        print(f"    gauge with the closed-form source: max|.| = {maxabs(r_src['gauge']):.3e}\n")
        harmonic[n_quad], sourced[n_quad] = r, r_src
    r, r_src = harmonic[a.n_quad], sourced[a.n_quad]

    checks.append(("compat: Gamma built from h", maxabs(r["compat"]) < 1e-14))
    checks.append(("RICCI: h and lam solve the Einstein equation", maxabs(r["ricci"]) < 1e-12))
    checks.append(("LAM_EQ: lam is h-harmonic", maxabs(r["lam_eq"]) < 1e-12))
    checks.append(("the Weyl chart is NOT harmonic (expected)", maxabs(r["gauge"]) > 1e-4))
    checks.append(("closed-form gauge source -> machine zero", maxabs(r_src["gauge"]) < 1e-12))
    checks.append(("residuals converged in n_quad",
                   abs(maxabs(harmonic[a.n_quad]["ricci"])
                       - maxabs(harmonic[2 * a.n_quad]["ricci"])) < 1e-14))

    # ------------------------------------------------------------- gauge vector
    gv = gauge_vector(rods, a.n_quad)
    closed = gauge_source_from_metric(lambda y: h_cart(y, rods, a.n_quad))
    gdiff = []
    print("the gauge source Gamma^i_{jk} h^{jk} (the chart's only failure)")
    print("  rho        |Gamma|      rho*|Gamma|    Gamma_x      Gamma_y      Gamma_z")
    for rr in (rho_in, 2 * rho_in, 10 * rho_in, rho_out):
        x = jnp.array([rr / jnp.sqrt(2.0), rr / jnp.sqrt(2.0), 0.3 * rr])
        g = gv(x)
        gdiff.append(float(jnp.max(jnp.abs(g - closed(x)))))
        print(f"  {rr:9.4g}  {float(jnp.linalg.norm(g)):.4e}   "
              f"{rr * float(jnp.linalg.norm(g)):.4e}   "
              + "  ".join(f"{float(c): .3e}" for c in g))
    print(f"\n  closed form (h_rhorho - 1) h^ij d_j ln rho vs the AD connection: "
          f"max|diff| = {max(gdiff):.2e}")
    checks.append(("closed form == autodiff connection", max(gdiff) < 1e-12))

    # --------------------------------------------------- decay of h - I and lam - 1
    rhos = jnp.exp(jnp.linspace(jnp.log(rho_in), jnp.log(rho_out), 24))
    direction = jnp.array([1.0, 1.0, 0.6]) / jnp.linalg.norm(jnp.array([1.0, 1.0, 0.6]))
    pts = rhos[:, None] * direction
    ph = decay_exponent(rhos, jnp.max(
        jnp.abs(jax.vmap(lambda y: h_cart(y, rods, a.n_quad) - I3)(pts)), axis=(1, 2)))
    pl = decay_exponent(rhos, jax.vmap(lambda y: lam_of(y, rods) - 1.0)(pts))
    pg = decay_exponent(rhos, jax.vmap(lambda y: jnp.linalg.norm(gv(y)))(pts))
    print("\ndecay along a ray (the exponents the Robin conditions assume)")
    print(f"    h - I   : rho^{ph:+.3f}   (robin_exps h = 2)")
    print(f"    lam - 1 : rho^{pl:+.3f}   (robin_exps lam = 1)")
    print(f"    |Gamma| : rho^{pg:+.3f}   (robin_exps G = 3)")
    checks.append(("h - I decays like rho^-2", abs(ph + 2.0) < 0.1))
    checks.append(("lam - 1 decays like rho^-1", abs(pl + 1.0) < 0.1))
    checks.append(("|Gamma| decays like rho^-3", abs(pg + 3.0) < 0.1))

    # ------------------------------------------------------------------- strut
    rho_small = 1e-4 * extent
    g_, l_ = a.half_gap, a.half_length
    print("\nk on the axis: the conical strut between the rods")
    kvals = {}
    for tag, z in (("gap centre (z = 0)", 0.0),
                   ("just inside the gap (z = 0.9 half_gap)", 0.9 * g_),
                   ("just off the rod's outer end (z = 1.1 extent)", 1.1 * extent),
                   ("far outside the rods (z = 3 extent)", 3.0 * extent)):
        kv = float(k_of(rho_small, z, rods, a.n_quad))
        kvals[tag] = kv
        print(f"    {tag:44s} k = {kv:+.6e}   cone factor 2 pi e^-k = "
              f"{float(2.0 * jnp.pi * jnp.exp(-kv)):.6e}")
    gap = abs(kvals["gap centre (z = 0)"])
    outside = abs(kvals["far outside the rods (z = 3 extent)"])
    checks.append(("strut: k is nonzero between the rods", gap > 0.1))
    checks.append(("strut: k -> 0 outside the rods", outside < 0.05))

    # ------------------------------------------ the production loss on this solution
    # Everything above is a statement about the fields.  This section is the one that
    # closes the loop with the trainer: build the reference, point the PRODUCTION loss
    # (`total_loss`, the exact function train.py minimises) at a candidate that IS the
    # Weyl solution, and check that it vanishes.  If it does, the loss has a genuine zero
    # at a two-black-hole configuration -- so a run that converges is solving the same
    # system this script just verified, not a spherical stand-in.
    import flax.linen as nn

    from stationary.losses import total_loss
    from stationary.model import point_fields
    from stationary.problem import Config, sample_sphere

    class WeylFieldNet(nn.Module):
        """A flax module whose forward pass is the exact Weyl solution.

        It is a candidate like any other -- the loss, the boundary terms and the gauge
        source all see only a callable h, Gamma, lambda.  Nothing in the trainer needs to
        know that this one is exact, which is the point: the same network-free path is
        what a trained `FieldNet` would be evaluated through.
        """

        rods: Rods
        n_quad: int = 200

        @nn.compact
        def __call__(self, x):
            x = jnp.atleast_2d(x)
            d = self.param("dummy", nn.initializers.zeros, (1,), jnp.float64)
            f = jax.vmap(fields_of(self.rods, self.n_quad))(x)
            return Fields(f.h + d, f.G + d, f.lam + d)

    cfg = Config(rho_in=rho_in, rho_out=rho_out, outer_bc="robin", robin_order=a.robin_order,
                 robin_source=True, inner_bc="reference", gauge_source="cylindrical",
                 weyl=True, weyl_half_length=a.half_length, weyl_half_gap=a.half_gap,
                 lam_inf=1.0, n_coll=512, n_bnd=96)
    model = WeylFieldNet(rods, a.n_quad)
    key = jax.random.PRNGKey(a.seed + 1)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 3)))
    ref = fields_of(rods, a.n_quad)
    k1, k2, k3 = jax.random.split(key, 3)
    batch = {"coll": sample_shell(k1, cfg.n_coll, rho_in, rho_out),
             "inner": sample_sphere(k2, cfg.n_bnd, rho_in),
             "outer": sample_sphere(k3, cfg.n_bnd, rho_out)}

    loss, parts = total_loss({"net": params}, batch, cfg, model, exact_fields=ref,
                             lam_inf=cfg.lam_inf)
    print("manufactured loss on the Weyl solution "
          f"(inner data from the reference, Robin source, cylindrical gauge source)")
    for k in sorted(parts):
        print(f"    {k:14s} = {float(parts[k]):.3e}")
    print(f"    {'TOTAL':14s} = {float(loss):.3e}\n")

    # The same loss with the HARMONIC gauge: it must NOT vanish, because the Weyl chart is
    # not harmonic.  Without this, a loss that ignores the gauge group entirely would pass
    # the check above for the wrong reason.
    cfg_h = Config(**{**cfg.__dict__, "gauge_source": "none"})
    loss_h, parts_h = total_loss({"net": params}, batch, cfg_h, model, exact_fields=ref,
                                 lam_inf=cfg.lam_inf)

    checks.append(("manufactured loss on the Weyl solution is zero", float(loss) < 1e-20))
    checks.append(("... and is NOT zero in the harmonic gauge (the check has teeth)",
                   float(loss_h) > 1e-8))

    # ------------------------------------------------------------------ verdict
    print("\n" + "=" * 72)
    bad = 0
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        bad += 0 if ok else 1
    print("=" * 72)
    if bad:
        print(f"WEYL CHECK FAILED: {bad} of {len(checks)} checks")
        return 1
    print(f"WEYL CHECK OK: {len(checks)} checks; the Weyl two-black-hole solution solves "
          f"the full system with the closed-form gauge source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
