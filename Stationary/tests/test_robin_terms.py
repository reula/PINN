"""The Robin combination as a sum of terms: `robin_coefficients` and its use in the report.

Why this exists: a Robin residual of 1e-06 can be two numbers of order 1 cancelling, and then
it carries no information about the branch.  Measured on runs/production_quad_quarter (the run
that converged to lambda = const): its order-3 lambda residual at rho_out was 6.5e-06 while the
terms of the combination were

    rho^3 d3(lam) = +3.796      9 rho^2 d2(lam) = +0.1645
    18 rho d1(lam) = +0.1420    6 (lam - 1)     = -4.102        sum = +4.4e-05

i.e. a boundary layer in the last 1% of the domain (lam''' rises from -0.05 at rho = 0.9 to
+3.8 at rho = 1.0) doing the cancelling.  Neither the residual nor the coarse lambda(rho) table
could show that, which is why the report now prints the terms.
"""
from __future__ import annotations

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from stationary import exact
from stationary.geometry import pack_sym
from stationary.losses import OFFW, robin_coefficients, robin_operator
from stationary.model import Fields
from stationary.problem import Config, sample_sphere

R0 = 0.005773502691896258


def test_the_coefficients_are_the_known_ones():
    assert robin_coefficients(1, 1) == [1.0, 1.0]                    # rho d + 1
    assert robin_coefficients(1, 2) == [2.0, 4.0, 1.0]               # rho^2 d2 + 4 rho d + 2
    assert robin_coefficients(1, 3) == [6.0, 18.0, 9.0, 1.0]
    assert robin_coefficients(1, 4) == [24.0, 96.0, 72.0, 16.0, 1.0]
    assert robin_coefficients(2, 3) == [24.0, 36.0, 12.0, 1.0]
    assert robin_coefficients(3, 2) == [12.0, 8.0, 1.0]              # (rho d + 3)(rho d + 4)


@pytest.mark.parametrize("base,order", [(1, 1), (1, 2), (1, 3), (2, 3), (1, 4)])
def test_the_window_is_annihilated(base, order):
    """prod_i (rho d + base + i) annihilates rho^-base ... rho^-(base+order-1) and not the next.

    NB for rho^-n: rho^j d^j (rho^-n) = (-1)^j n(n+1)...(n+j-1) rho^-n, a RISING factorial --
    writing (-n)^j here is the classic slip and gives a test that fails on correct code.
    """
    c = robin_coefficients(base, order)
    value = lambda n: sum(c[j] * (-1) ** j * math.prod(range(n, n + j))
                          for j in range(order + 1))
    for n in range(base, base + order):
        assert abs(value(n)) < 1e-9, (base, order, n, value(n))
    assert abs(value(base + order)) > 1e-9, (base, order, base + order, value(base + order))


@pytest.mark.parametrize("base,order,kind", [(1, 3, "lam"), (2, 3, "h"), (1, 1, "lam")])
def test_the_terms_sum_to_the_operator(base, order, kind):
    """The decomposition must reproduce robin_operator exactly, or the report would lie."""
    cfg = Config(R0=R0, rho_in=0.01, inner_radius=0.01, rho_out=1.0, lam_inf=1.0,
                 ref_asymptotic=1.0, ref_solution=True, outer_bc="robin",
                 robin_orders=dict(h=order, lam=order), robin_include_G=False)
    ref, _ = exact.reference_fields_asymptotic(cfg.R0, 1.0, cfg.rho_in,
                                               r_areal=cfg.inner_radius)
    shift = 0.017                                  # a field with a real residual
    inf_val = float(cfg.lam_inf) if kind == "lam" else jnp.eye(3)

    def fields(x):
        f = ref(x)
        return Fields(f.h, f.G, f.lam + shift) if kind == "lam" else \
            Fields(f.h + shift * jnp.eye(3), f.G, f.lam)

    packer = (lambda v: jnp.reshape(v, (1,))) if kind == "lam" else \
        (lambda M: pack_sym(M) * OFFW)
    x = sample_sphere(jax.random.PRNGKey(0), 4, cfg.rho_out)
    c = robin_coefficients(base, order)
    for xi in x:
        n = xi / jnp.linalg.norm(xi)
        rho = float(jnp.linalg.norm(xi))
        f0 = lambda r: (fields(r * n).lam - inf_val) if kind == "lam" else \
            (fields(r * n).h - inf_val)
        total = 0.0
        for j in range(order + 1):
            dj = f0
            for _ in range(j):
                dj = jax.jacfwd(dj)
            total = total + jnp.ravel(packer(dj(rho))) * (rho ** j) * c[j]
        if kind == "lam":
            want = robin_operator(lambda y: fields(y).lam, xi, base, order, inf_val)
            assert float(total[0]) == pytest.approx(float(want), rel=1e-9, abs=1e-12)
        else:
            want = robin_operator(lambda y: fields(y).h, xi, base, order, inf_val)
            assert jnp.allclose(total, pack_sym(want) * OFFW, rtol=1e-9, atol=1e-12)
