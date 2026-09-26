"""The rho-scaled RADIAL DERIVATIVE of the lambda equation (`Config.w_lam_eq_radial`).

Why the term exists: the lambda-equation is second order, so a loss built from it is blind to
lambda''', and lambda''' at rho_out is exactly where the order-3 Robin condition lets a wrong
far-field level hide (the condition's kernel is rho^-1, rho^-2, rho^-3, so its residual is a
cancellation of terms of order 0.1 that leaves 1.3e-08; runs/production_quad_quarter finished
with an outer Robin residual of 7e-06 and lambda(rho_out) = 0.31 against 0.9885193).

These tests pin down what the term is for, not just that it runs:

  * the exact reference is a zero of it (1e-30) -- it costs the solution nothing;
  * for the two deviations the Robin condition cannot see (a shift of the level, a kernel-mode
    a/rho), it is 13-15x LARGER than the plain lambda-equation term, i.e. it is more sensitive
    to exactly the content that was being hidden;
  * for a radial wiggle it grows like the wavenumber k (703x at k = 32, 2.5e4 at k = 200),
    which is what makes hiding the mismatch in high radial derivatives expensive;
  * it is invariant under the shell rescaling rho -> rho/s with the fields relabelled, which
    is what the rho^3 factor (inherited from scale_exps['lam_eq'] = 2) is for;
  * w_lam_eq_radial = 0 returns nothing, so no existing run pays for it.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from stationary import exact
from stationary.geometry import radial_derivative_residual_batch
from stationary.losses import pde_radial_terms, pde_terms
from stationary.model import Fields
from stationary.problem import Config, sample_shell

R0 = 0.005773502691896258


def production_cfg(**over):
    cfg = Config(R0=R0, rho_in=0.01, inner_radius=0.01, rho_out=1.0, lam_inf=1.0,
                 ref_asymptotic=1.0, ref_solution=True, outer_bc="robin",
                 robin_orders=dict(h=3, lam=3), robin_include_G=False,
                 w_lam_eq_radial=1.0)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture(scope="module")
def ref():
    cfg = production_cfg()
    fields, _ = exact.reference_fields_asymptotic(cfg.R0, 1.0, cfg.rho_in,
                                                  r_areal=cfg.inner_radius)
    return cfg, fields


@pytest.fixture(scope="module")
def xs():
    return sample_shell(jax.random.PRNGKey(0), 4096, production_cfg())


def _shifted(ref_fields, dl):
    def f(x):
        g = ref_fields(x)
        return Fields(g.h, g.G, g.lam + dl(x))
    return f


def test_the_weight_zero_switch_costs_nothing(ref, xs):
    cfg, ref_fields = ref
    assert pde_radial_terms(ref_fields, xs, production_cfg(w_lam_eq_radial=0.0)) == {}
    assert pde_radial_terms(ref_fields, xs, production_cfg(w_lam_eq_radial=None)) == {}


def test_the_reference_is_a_zero_of_it(ref, xs):
    """The exact solution is a solution: both the equation and its radial derivative vanish."""
    cfg, ref_fields = ref
    out = pde_radial_terms(ref_fields, xs, cfg)
    assert set(out) == {"lam_eq_radial"}
    assert float(out["lam_eq_radial"]) < 1e-20
    assert float(pde_terms(ref_fields, xs, cfg)["lam_eq"]) < 1e-20


def test_it_is_more_sensitive_than_the_equation_to_the_hidden_deviations(ref, xs):
    """A shifted level and a kernel mode a/rho: invisible to the Robin condition, and the
    radial term is an order of magnitude larger than the plain lambda-equation term."""
    cfg, ref_fields = ref
    for name, f in (("level shift", _shifted(ref_fields, lambda x: 0.1)),
                    ("kernel mode a/rho", _shifted(ref_fields,
                                                   lambda x: 0.05 / jnp.linalg.norm(x)))):
        plain = float(pde_terms(f, xs, cfg)["lam_eq"])
        rad = float(pde_radial_terms(f, xs, cfg)["lam_eq_radial"])
        assert plain > 1e-6, name
        assert rad > 10.0 * plain, (name, plain, rad)


def test_it_grows_with_the_radial_wavenumber(ref, xs):
    """Hiding the mismatch in higher radial derivatives gets more expensive, not less."""
    cfg, ref_fields = ref
    ratios = []
    for k in (16.0, 64.0):
        f = _shifted(ref_fields, lambda x, k=k: 1e-3 * jnp.sin(k * (jnp.linalg.norm(x) - 1.0)))
        ratios.append(float(pde_radial_terms(f, xs, cfg)["lam_eq_radial"])
                      / float(pde_terms(f, xs, cfg)["lam_eq"]))
    assert ratios[1] > 5.0 * ratios[0]           # ~4x for a 4x wavenumber, plus the tail


def test_it_is_invariant_under_the_shell_rescaling(xs):
    """rho -> rho/s with the fields relabelled leaves it unchanged: the rho^3 factor."""
    s = 100.0
    cfg = production_cfg()
    cfg2 = production_cfg(R0=R0 / s, rho_in=cfg.rho_in / s, rho_out=cfg.rho_out / s)
    ref, _ = exact.reference_fields_asymptotic(cfg.R0, 1.0, cfg.rho_in,
                                               r_areal=cfg.inner_radius)
    xs2 = sample_shell(jax.random.PRNGKey(0), 4096, cfg2)
    shifted = _shifted(ref, lambda x: 0.1)
    a = float(pde_radial_terms(shifted, xs, cfg)["lam_eq_radial"])
    b = float(pde_radial_terms(lambda x: shifted(x * s), xs2, cfg2)["lam_eq_radial"])
    assert a > 1e-9
    assert b == pytest.approx(a, rel=1e-9)


def test_the_raw_batch_matches_the_scaling_by_hand(ref, xs):
    """rho^3 d/drho of the raw residual, checked against a finite difference in rho.

    On the exact reference both sides are round-off (~1e-31), so the comparison is made on a
    field with a real residual: lambda_ref + 0.1, whose lambda-equation term is ~7e-04.
    """
    from stationary.geometry import lam_eq_at

    cfg, ref_fields = ref
    shifted = _shifted(ref_fields, lambda x: 0.1)
    r = radial_derivative_residual_batch(shifted, xs, cfg.scale_exps, cfg.scale_ref)
    x = xs[0]
    rho = float(jnp.linalg.norm(x))
    n = x / rho
    h = 1e-5
    fd = (lam_eq_at(shifted, x + h * n) - lam_eq_at(shifted, x - h * n)) / (2.0 * h)
    assert float(r[0]) == pytest.approx(float(fd * rho ** 3), rel=1e-4)
