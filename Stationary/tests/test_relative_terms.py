"""Relative residuals (`Config.relative_terms` / `--relative-terms`).

Each equation group's residual is a signed sum of terms of the same dimension.  Dividing the
sum by the sum of the ABSOLUTE VALUES of those terms gives a dimensionless number in [-1, 1]
that measures the local relative cancellation error.  Two things follow, and they are the point
of the mode:

  * no length has to be chosen and no rho^d weight is applied -- `scaled_residuals_batch` is a
    no-op -- so the far field is not silently de-emphasised (the failure of
    runs/pq_c100_vac6_fixref, where scale_ref = rho_in flattened the weights to constants and
    the far field rotted);
  * a residual that is small only because ITS TERMS are small is charged in full: the lambda
    equation is homogeneous of degree one in lambda, so lam -> 0 makes the "lambda" residual
    vanish; relative to |Delta_h lam| + |d lam|^2/lam, the same collapse is O(1).

These tests pin: the default is off, the ratio really is residual/denominator, the bound, the
scale invariance under lam -> c lam, the exact solution being a zero, the no-op in
`scaled_residuals_batch`, and the rejection of combining it with the log form (a single-term
residual has no relative form).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from stationary import exact
from stationary.geometry import (lam_eq_at, radial_derivative_residual_batch, residuals_at,
                                 residuals_batch, scaled_residuals_batch, set_lam_eq_form,
                                 set_relative_terms)
from stationary.model import Fields
from stationary.problem import Config


@pytest.fixture(autouse=True)
def _restore_knobs():
    set_lam_eq_form("lambda")
    set_relative_terms(False)
    yield
    set_lam_eq_form("lambda")
    set_relative_terms(False)


def flat_field(amp: float = 0.1, c: float = 1.0):
    """h = delta, Gamma = 0, lam = c (1 + amp sin x0): a field with a real lambda residual."""
    def fields(x):
        return Fields(jnp.eye(3), jnp.zeros((3, 3, 3)), c * (1.0 + amp * jnp.sin(x[0])))
    return fields


def exact_fields_and_cfg():
    cfg = Config(R0=1.0, rho_in=1.0, rho_out=100.0, lam_inf=1.0, ref_asymptotic=1.0,
                 ref_solution=True)
    fields, _ = exact.reference_fields_asymptotic(cfg.R0, 1.0, cfg.rho_in, r_areal=1.0)
    return cfg, fields


XS = [jnp.array([1.3, 0.4, 0.7]), jnp.array([-2.0, 0.9, 1.5]), jnp.array([10.0, -3.0, 4.0])]


def test_the_default_is_off():
    """With the knob off nothing changes: the residual is the raw, dimensionful one."""
    f = flat_field()
    set_relative_terms(False)
    for x in XS:
        raw = float(residuals_at(f, x)["lam_eq"])
        d2 = jax.hessian(lambda y: f(y).lam)(x)
        dl = jax.jacfwd(lambda y: f(y).lam)(x)
        want = float(jnp.trace(d2) - jnp.dot(dl, dl) / f(x).lam)
        assert raw == pytest.approx(want, rel=1e-12, abs=1e-15)
        assert abs(raw) < 1.0        # for this field; the raw value carries no bound in general


def test_the_ratio_is_the_residual_over_the_size_of_its_terms():
    """lam_eq: denominator = |Delta_h lam| + |d lam|^2/lam.  Checked by hand, both knobs."""
    f = flat_field()
    for x in XS:
        raw = float(residuals_at(f, x)["lam_eq"])
        d2 = jax.hessian(lambda y: f(y).lam)(x)
        dl = jax.jacfwd(lambda y: f(y).lam)(x)
        lap = float(jnp.trace(d2))
        src = float(jnp.dot(dl, dl) / f(x).lam)
        set_relative_terms(True)
        try:
            rel = float(residuals_at(f, x)["lam_eq"])
        finally:
            set_relative_terms(False)
        assert rel == pytest.approx((lap - src) / (abs(lap) + abs(src)), rel=1e-9)
        assert abs(rel) <= 1.0 + 1e-12


def test_every_group_is_bounded_by_one():
    """A badly wrong field: each relative residual is a ratio, so |r| <= 1 (Frobenius for
    tensors).  This is what removes the 'weight x huge residual' blow-up."""
    def wrong(x):
        return Fields(2.0 * jnp.eye(3), jnp.zeros((3, 3, 3)), 1.0 + x[0])

    set_relative_terms(True)
    try:
        for x in XS:
            r = residuals_at(wrong, x)
            for k, v in r.items():
                assert float(jnp.max(jnp.abs(v))) <= 1.0 + 1e-12, (k, float(jnp.max(jnp.abs(v))))
            assert float(jnp.max(jnp.abs(r["ricci"]))) > 1e-3     # and not trivially zero
    finally:
        set_relative_terms(False)


def test_shrinking_lambda_is_charged_identically():
    """lam -> c lam leaves the relative lambda residual unchanged (the collapse is not free)."""
    ref = None
    for c in (1.0, 1e-3, 1e-6, 1e-9, 1e-12):
        f = flat_field(c=c)
        vals = []
        set_relative_terms(True)
        try:
            vals = [float(residuals_at(f, x)["lam_eq"]) for x in XS]
        finally:
            set_relative_terms(False)
        if ref is None:
            ref = vals
            assert max(abs(v) for v in vals) > 0.05
        else:
            assert vals == pytest.approx(ref, rel=1e-9, abs=1e-12), (c, vals, ref)


def test_the_exact_solution_is_a_zero():
    cfg, fields = exact_fields_and_cfg()
    set_relative_terms(True)
    try:
        for x in [jnp.array([0.0, 0.0, r]) for r in (1.0, 2.5, 10.0, 100.0)]:
            for k, v in residuals_at(fields, x).items():
                assert float(jnp.max(jnp.abs(v))) < 1e-9, (k, float(jnp.max(jnp.abs(v))))
    finally:
        set_relative_terms(False)


def test_scaled_residuals_batch_is_a_no_op_in_relative_mode():
    """No rho^d factor is applied: the relative residual is already dimensionless.  Checked at
    two radii whose weights would differ by 100^p."""
    cfg, fields = exact_fields_and_cfg()

    def shifted(x):
        g = fields(x)
        return Fields(g.h, g.G, g.lam + 0.1)

    xs = jnp.stack([jnp.array([0.0, 0.0, 2.0]), jnp.array([0.0, 0.0, 100.0])])
    exps = dict(compat=1.0, ricci=2.0, gauge=1.0, lam_eq=2.0)
    set_relative_terms(True)
    try:
        plain = residuals_batch(shifted, xs)
        scaled = scaled_residuals_batch(shifted, xs, exps, ref=None)
    finally:
        set_relative_terms(False)
    for k in plain:
        assert jnp.allclose(plain[k], scaled[k], rtol=0, atol=0), k


def test_the_radial_derivative_uses_rho_once_in_relative_mode():
    """The residual is dimensionless, so its radial derivative carries length^-1 (rho^1, not
    rho^3), checked against a finite difference in rho."""
    cfg, fields = exact_fields_and_cfg()

    def shifted(x):
        g = fields(x)
        return Fields(g.h, g.G, g.lam + 0.1)

    x = jnp.array([0.0, 0.0, 3.0])
    rho = float(jnp.linalg.norm(x))
    n = x / rho
    set_relative_terms(True)
    try:
        r = radial_derivative_residual_batch(
            shifted, jnp.stack([x]), dict(compat=1.0, ricci=2.0, gauge=1.0, lam_eq=2.0), None)
        h = 1e-4
        fd = (lam_eq_at(shifted, x + h * n) - lam_eq_at(shifted, x - h * n)) / (2.0 * h)
    finally:
        set_relative_terms(False)
    assert float(r[0]) == pytest.approx(float(fd * rho), rel=1e-4)


def test_relative_terms_and_the_log_form_are_alternatives():
    set_lam_eq_form("log")
    with pytest.raises(ValueError):
        set_relative_terms(True)
    set_lam_eq_form("lambda")
    set_relative_terms(True)          # fine on its own
    with pytest.raises(ValueError):   # and the other order is caught too
        set_lam_eq_form("log")
