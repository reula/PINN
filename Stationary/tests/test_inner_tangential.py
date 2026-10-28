"""--inner-tangential: the tangential derivatives of the inner residual (losses._tangential_gradient).

Three statements: the exact solution still has zero loss with the extra terms (its inner residual
vanishes on the whole sphere, so do its tangential derivatives); the term IS rho (I - n n) grad of
the value residual (against a central difference along a tangent); and it is only switched on by
the flag.  The residual vector carrying it squares to the total loss in tests/test_dsgnar.py.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import dataclasses

import jax.numpy as jnp
import numpy as np

from stationary.losses import inner_bc_raw, total_loss
from stationary.model import point_fields
from stationary.problem import sample_shell, sample_sphere
from stationary.train import build, parse_args
from test_pipeline import ExactModel, cfg_m1, exact_point_fields

BASE = ["--arch", "axisym_hybrid", "--n-coll", "64", "--n-bnd", "24",
        "--width", "8", "--depth", "2", "--no-figures",
        "--R0", "0.5773502691896258", "--rho-in", "1.0", "--rho-out", "100.0",
        "--inner-radius", "1.0", "--lam0", "0.33333333333333337", "--lam-inf", "1.0",
        "--lam-bc-S2", "-0.08333333333333333", "--no-robin-G"]


def test_exact_solution_still_has_zero_loss():
    cfg = dataclasses.replace(cfg_m1(), inner_tangential=True)
    model = ExactModel()
    key = jax.random.PRNGKey(2)
    state = {"net": model.init(key, jnp.ones((1, 3)))}
    batch = {"coll": sample_shell(key, 512, cfg),
             "inner": sample_sphere(key, 64, cfg.rho_in),
             "outer": sample_sphere(key, 64, cfg.rho_out)}
    loss, parts = total_loss(state, batch, cfg, model, exact_fields=exact_point_fields())
    assert {"inner_lam_dtan", "inner_h_tan_dtan"} <= set(parts)
    assert float(loss) < 1e-16, (float(loss), {k: float(v) for k, v in parts.items()})


def test_it_is_the_tangential_gradient_of_the_residual(tmp_path):
    cfg = parse_args(BASE + ["--outdir", str(tmp_path / "run"), "--inner-tangential"])
    model, state, _ = build(cfg)
    pf = point_fields(model, state["net"])
    xs = sample_sphere(jax.random.PRNGKey(3), 5, cfg.rho_in)
    R = inner_bc_raw(pf, xs, cfg)
    eps = 1e-5
    for i, x in enumerate(np.asarray(xs)):
        n = x / np.linalg.norm(x)
        t = np.cross(n, [0.3, -0.5, 0.8])
        t /= np.linalg.norm(t)
        plus = inner_bc_raw(pf, jnp.asarray(x + eps * t)[None], cfg)
        minus = inner_bc_raw(pf, jnp.asarray(x - eps * t)[None], cfg)
        rho = np.linalg.norm(x)
        for k in ("lam", "h_tan"):
            fd = rho * (np.asarray(plus[k][0]) - np.asarray(minus[k][0])) / (2 * eps)
            got = np.asarray(R[f"{k}_dtan"][i]) @ t / np.sqrt(3.0)
            np.testing.assert_allclose(got, fd, rtol=1e-6, atol=1e-9)
        # and nothing along the normal
        assert np.max(np.abs(np.asarray(R["lam_dtan"][i]) @ n)) < 1e-12


def test_it_sees_only_lambda_and_h_AB(tmp_path):
    """h_rr and h_rA are free on the inner sphere, and the derivative term must not see them either.

    The metric part is the derivative along the sphere of M = P h P - c P, whose restriction to
    the sphere is the field h_AB alone; adding an arbitrary a(x) n n + n w + w n (w tangential,
    both varying from point to point) to h must leave every *_dtan term, and the values, unchanged.
    """
    cfg = parse_args(BASE + ["--outdir", str(tmp_path / "run"), "--inner-tangential"])
    model, state, _ = build(cfg)
    pf = point_fields(model, state["net"])

    def pf_extra(x):
        f = pf(x)
        n = x / jnp.linalg.norm(x)
        a = 0.3 * jnp.sin(2.0 * x[0]) + 0.1 * x[2]
        w = (jnp.eye(3) - jnp.outer(n, n)) @ jnp.array([jnp.cos(x[1]), 0.2 * x[2], x[0] ** 2])
        return f._replace(h=f.h + a * jnp.outer(n, n) + jnp.outer(n, w) + jnp.outer(w, n))

    xs = sample_sphere(jax.random.PRNGKey(4), 16, cfg.rho_in)
    R0, R1 = inner_bc_raw(pf, xs, cfg), inner_bc_raw(pf_extra, xs, cfg)
    assert set(R0) == {"lam", "h_tan", "lam_dtan", "h_tan_dtan"}     # no h_rr: it is free
    for k in R0:
        np.testing.assert_allclose(np.asarray(R1[k]), np.asarray(R0[k]), rtol=0, atol=1e-12)
    # while a change of h_AB is seen
    def pf_tan(x):
        f = pf(x)
        P = jnp.eye(3) - jnp.outer(x, x) / (x @ x)
        return f._replace(h=f.h + 0.1 * jnp.sin(x[0]) * P)
    assert np.max(np.abs(np.asarray(inner_bc_raw(pf_tan, xs, cfg)["h_tan_dtan"])
                         - np.asarray(R0["h_tan_dtan"]))) > 1e-3


def test_off_by_default(tmp_path):
    cfg = parse_args(BASE + ["--outdir", str(tmp_path / "run")])
    assert cfg.inner_tangential is False
    model, state, _ = build(cfg)
    R = inner_bc_raw(point_fields(model, state["net"]),
                     sample_sphere(jax.random.PRNGKey(0), 4, cfg.rho_in), cfg)
    assert not any(k.endswith("_dtan") for k in R)
