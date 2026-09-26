"""The far-field VALUE pins (`Config.pin_lam`, `pin_h_tan`, `pin_h_rr`).

Why these exist, in one line each: a Robin condition is a differential combination whose
kernel contains the leading decaying modes of the exact solution, so it says nothing about
their amplitude -- and the far-field amplitude is the branch.  Measured on the production
geometry at rho_out = 1 (see the Config comment): the order-3 lambda combination is a
cancellation of terms of order 0.1 (rho^3 lam''' = +6.77e-02, 9rho^2 lam'' = -2.04e-01,
18rho lam' = +2.05e-01, 6(lam-1) = -6.89e-02) that leaves 1.3e-08, and the exact h deviation
from delta IS the rho^-2 kernel mode of the h condition, so that condition is satisfied
identically (3e-31) whatever the amplitude.  Both failed quadrupole runs confirm it:
lambda(rho_out) = 0.047 and 0.31 flat, with outer Robin residuals <= 1e-05.

These tests pin down the properties that make the terms useful rather than decorative:
the reference satisfies them exactly; a flat lambda is rejected by ~0.46; a wrong-amplitude
metric tail is rejected; the lambda pin sees the MONOPOLE and not the quadrupole (that is
what leaves the l >= 1 tails for the Robin conditions to fix later); and the flags off, or a
missing reference, behave.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from stationary import exact
from stationary.losses import outer_pin_terms
from stationary.model import Fields
from stationary.problem import Config, sample_sphere

R0 = 0.005773502691896258          # 1/sqrt(3) / 100: the production geometry
LAM_REF = 0.988519280582           # the reference's lambda at rho_out


def production_cfg(**over):
    cfg = Config(R0=R0, rho_in=0.01, inner_radius=0.01, rho_out=1.0, lam_inf=1.0,
                 ref_asymptotic=1.0, ref_solution=True, outer_bc="robin",
                 robin_exps=dict(h=2.0, G=3.0, lam=1.0),
                 robin_orders=dict(h=3, lam=3), robin_include_G=False,
                 pin_lam=True, pin_h_tan=True, pin_h_rr=True)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture(scope="module")
def ref():
    cfg = production_cfg()
    fields, _ = exact.reference_fields_asymptotic(cfg.R0, 1.0, cfg.rho_in,
                                                  r_areal=cfg.inner_radius)
    return cfg, fields


def _xs(cfg, n=512, seed=0):
    return sample_sphere(jax.random.PRNGKey(seed), n, cfg.rho_out)


def test_the_reference_satisfies_the_pins(ref):
    """The pins are differences from the reference, so it must be a zero of all three."""
    cfg, ref_fields = ref
    out = outer_pin_terms(ref_fields, _xs(cfg), cfg, ref_fields)
    assert set(out) == {"lam", "h_tan", "h_rr"}
    for k, v in out.items():
        assert float(v) < 1e-20, (k, float(v))


def test_a_flat_lambda_is_rejected(ref):
    """The branch runs/production_quad_quarter converged to: lambda ~ 0.31 everywhere."""
    cfg, ref_fields = ref

    def flat(x):
        f = ref_fields(x)
        return Fields(f.h, f.G, 0.31 + 0.0 * f.lam)

    out = outer_pin_terms(flat, _xs(cfg), cfg, ref_fields)
    assert float(out["lam"]) == pytest.approx((0.31 - LAM_REF) ** 2, rel=1e-3)
    assert float(out["lam"]) > 0.4                      # ~0.46: decisive, where the Robin
    assert float(out["h_tan"]) < 1e-20                  # residual was 4e-11
    assert float(out["h_rr"]) < 1e-20


def test_a_wrong_metric_tail_is_rejected(ref):
    """The h condition cannot see the rho^-2 mode's amplitude; the pin can."""
    cfg, ref_fields = ref

    def doubled(x):
        # delta + 2 R0^2/rho^2 (nn - delta): the right mode with twice the amplitude
        rho = jnp.linalg.norm(x)
        n = x / rho
        nn = jnp.outer(n, n)
        c = 2.0 * R0 ** 2 / rho ** 2
        h = (1.0 - c) * jnp.eye(3) + c * nn
        f = ref_fields(x)
        return Fields(h, f.G, f.lam)

    out = outer_pin_terms(doubled, _xs(cfg), cfg, ref_fields)
    # the difference is one R0^2/rho^2, i.e. 3.3e-05 at rho_out, so ~1.1e-09 squared
    assert float(out["h_tan"]) == pytest.approx(R0 ** 4, rel=0.2)
    # h_rr is 1 identically for ANY amplitude of that mode (h = (1-c) delta + c nn), which
    # is exactly why it needs its own pin: only a genuine radial-gauge drift moves it.
    assert float(out["h_rr"]) == pytest.approx((1.0 - 0.999999302739) ** 2, rel=0.3)

    def gauge_drift(x):
        rho = jnp.linalg.norm(x)
        n = x / rho
        nn = jnp.outer(n, n)
        c = R0 ** 2 / rho ** 2
        f = ref_fields(x)
        return Fields((1.0 - c) * jnp.eye(3) + c * nn + 0.01 * nn, f.G, f.lam)

    d = outer_pin_terms(gauge_drift, _xs(cfg), cfg, ref_fields)
    assert float(d["h_rr"]) == pytest.approx(1e-4, rel=0.05)      # h_rr = 1.01
    # A term along nn is purely radial: it moves h_rr and leaves g2 alone (that is why the
    # two metric pins are separate flags), so h_tan only sees the 3.5e-07 chart offset.
    assert float(d["h_tan"]) < 1e-12
    assert float(out["lam"]) < 1e-20                    # the metric does not disturb lambda


def test_the_lambda_pin_sees_the_monopole_not_the_quadrupole(ref):
    """l = 0 only: a zero-mean P2 perturbation must be invisible, a monopole must not."""
    cfg, ref_fields = ref
    xs = _xs(cfg, n=8192)

    def quadrupole(x):
        n = x / jnp.linalg.norm(x)
        f = ref_fields(x)
        return Fields(f.h, f.G, f.lam + 1e-3 * (3.0 * n[2] ** 2 - 1.0) / 2.0)

    def monopole(x):
        f = ref_fields(x)
        return Fields(f.h, f.G, f.lam + 1e-3)

    q = float(outer_pin_terms(quadrupole, xs, cfg, ref_fields)["lam"])
    m = float(outer_pin_terms(monopole, xs, cfg, ref_fields)["lam"])
    assert q < 1e-8            # (1e-3)^2 = 1e-6 is the monopole-sized signal; MC noise <<
    assert m == pytest.approx(1e-6, rel=0.05)


def test_pins_off_returns_nothing(ref):
    cfg, ref_fields = ref
    cfg = production_cfg(pin_lam=False, pin_h_tan=False, pin_h_rr=False)
    assert outer_pin_terms(ref_fields, _xs(cfg), cfg, ref_fields) == {}


def test_pins_without_a_reference_fail_loudly(ref):
    """The reference is the pinned data; without it the loss must not silently drop them."""
    cfg, ref_fields = ref
    with pytest.raises(ValueError, match="no reference"):
        outer_pin_terms(ref_fields, _xs(cfg), cfg, None)
    # ... and the Config refuses the combination at construction time
    with pytest.raises(ValueError, match="builds no reference"):
        Config(R0=R0, rho_in=0.01, inner_radius=0.01, rho_out=1.0, outer_bc="robin",
               ref_solution=False, robin_source=False, pin_lam=True)
