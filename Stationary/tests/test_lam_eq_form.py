"""The two forms of the lambda equation (`Config.lam_eq_form` / `--lam-eq-form`).

They are the SAME equation.  Writing phi = log lam,

    Delta_h phi = Delta_h lam / lam - |d lam|^2_h / lam^2
                = (1/lam) [ Delta_h lam - (1/lam) |d lam|^2_h ],

so the bracket -- the "lambda" residual -- is lam x Delta_h phi, and the two have exactly the
same zeros.  What differs is the LOSS:

  * the "lambda" residual is homogeneous of degree one in lam, so lam -> c lam scales it by c;
    in particular lam -> 0 makes it vanish without solving anything.  That is the direction
    runs/pq_c100_vac6_fixref took: with the rho^d weights replaced by constants (scale_ref =
    rho_in) lam fell from 0.25 at the inner sphere to 1e-7 by rho = 1.8 while the loss read
    2.8e-13, and the run ended with a 50x wrong far-field level (lambda(rho_out) = 0.43 against
    0.9885) that the order-3 Robin condition and the order-1 pin cannot see, because both
    annihilate rho^-1 whatever its coefficient.
  * the "log" residual divides by lam (`Delta_h phi`), so it is INVARIANT under lam -> c lam and
    charges in full for a collapse, at any c -- the guard only covers the single point lam = 0.

These tests pin the identity between the two forms, the invariance that is the point of the log
form, that a sign change stays finite, and that the default is unchanged (so every existing run
is unaffected).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from stationary import exact
from stationary.geometry import lam_eq_of, residuals_at, set_lam_eq_form
from stationary.model import Fields
from stationary.problem import Config


@pytest.fixture(autouse=True)
def _restore_form():
    """The form is module-level state (like the Ricci source); put it back after each test."""
    set_lam_eq_form("lambda")
    yield
    set_lam_eq_form("lambda")


def flat_metric_field(amp: float = 0.1, c: float = 1.0):
    """h = delta, Gamma = 0, lam = c (1 + amp sin x) -- a field with a real residual."""
    def fields(x):
        lam = c * (1.0 + amp * jnp.sin(x[0]))
        return Fields(jnp.eye(3), jnp.zeros((3, 3, 3)), lam)
    return fields


XS = [jnp.array([0.3, 0.2, -0.4]), jnp.array([-1.1, 0.7, 0.9]), jnp.array([2.0, -0.5, 0.1])]


def test_the_default_form_is_the_lambda_one():
    f = flat_metric_field()
    for x in XS:
        got = float(residuals_at(f, x)["lam_eq"])
        d2 = jax.hessian(lambda y: f(y).lam)(x)
        dl = jax.jacfwd(lambda y: f(y).lam)(x)
        want = float(jnp.trace(d2) - jnp.dot(dl, dl) / f(x).lam)
        assert got == pytest.approx(want, rel=1e-12, abs=1e-15)


def test_the_log_form_is_the_lambda_residual_over_lambda():
    """The identity Delta_h phi = (1/lam)[Delta_h lam - (1/lam)|d lam|^2]."""
    f = flat_metric_field()
    for x in XS:
        a = float(residuals_at(f, x)["lam_eq"])
        set_lam_eq_form("log")
        try:
            b = float(residuals_at(f, x)["lam_eq"])
        finally:
            set_lam_eq_form("lambda")
        assert a != 0.0
        assert b == pytest.approx(a / float(f(x).lam), rel=1e-11)


def test_scaling_lambda_is_free_in_one_form_and_charged_in_the_other():
    """lam -> c lam: the lambda residual scales by c (=> it can be driven to zero by shrinking
    the field), the log residual does not move."""
    f1, f2 = flat_metric_field(c=1.0), flat_metric_field(c=0.01)
    for x in XS:
        a1 = float(residuals_at(f1, x)["lam_eq"])
        a2 = float(residuals_at(f2, x)["lam_eq"])
        assert a2 == pytest.approx(0.01 * a1, rel=1e-10)         # vanishes as c -> 0
        set_lam_eq_form("log")
        try:
            b1 = float(residuals_at(f1, x)["lam_eq"])
            b2 = float(residuals_at(f2, x)["lam_eq"])
        finally:
            set_lam_eq_form("lambda")
        assert b1 == pytest.approx(b2, rel=1e-10)                # invariant
        assert abs(b1) > 0.1 * abs(a1)                           # and not small


def test_the_log_form_charges_a_shrunk_field_all_the_way_down():
    """The property the guard must not destroy: lam -> c lam leaves Delta_h(log lam) unchanged,
    so a collapse is charged in full even for tiny c.  (Clamping |lam| up to a floor would send
    base/lam -> base/floor ~ c and hide the collapse below the floor -- the degenerate direction
    would merely move.)"""
    for c in (1e-3, 1e-6, 1e-9, 1e-12):
        f1, f2 = flat_metric_field(c=1.0), flat_metric_field(c=c)
        for x in XS:
            set_lam_eq_form("log")
            try:
                b1 = float(residuals_at(f1, x)["lam_eq"])
                b2 = float(residuals_at(f2, x)["lam_eq"])
            finally:
                set_lam_eq_form("lambda")
            assert b1 == pytest.approx(b2, rel=1e-9), (c, b1, b2)
            assert abs(b2) > 0.05 * abs(b1), (c, b1, b2)


def test_a_field_crossing_zero_stays_finite_and_expensive():
    """lam = x0 changes sign; the ungarded 1/lam would be inf and NaN the whole loss.  The guard
    keeps it finite, and large: a genuine collapse is not a free direction in this form."""
    def f(x):
        return Fields(jnp.eye(3), jnp.zeros((3, 3, 3)), x[0])

    xs = [jnp.array([0.0, 0.0, 0.0]), jnp.array([1e-9, 0.2, 0.1]), jnp.array([-1.0, 0.3, 0.2])]
    set_lam_eq_form("log")
    try:
        vals = [float(residuals_at(f, x)["lam_eq"]) for x in xs]
    finally:
        set_lam_eq_form("lambda")
    assert all(jnp.isfinite(v) for v in vals), vals
    assert max(abs(v) for v in vals) > 1.0, vals


def test_a_bad_form_is_rejected():
    with pytest.raises(ValueError):
        set_lam_eq_form("phi")


def test_both_forms_vanish_on_the_exact_solution():
    """The reference solves both -- the floor and the 1/lam factor change nothing there."""
    cfg = Config(R0=1.0, rho_in=1.0, rho_out=100.0, lam_inf=1.0, ref_asymptotic=1.0,
                 ref_solution=True)
    fields, _ = exact.reference_fields_asymptotic(cfg.R0, 1.0, cfg.rho_in, r_areal=1.0)
    xs = [jnp.array([0.0, 0.0, r]) for r in (1.0, 2.5, 10.0, 100.0)]
    for x in xs:
        a = abs(float(residuals_at(fields, x)["lam_eq"]))
        set_lam_eq_form("log")
        try:
            b = abs(float(residuals_at(fields, x)["lam_eq"]))
        finally:
            set_lam_eq_form("lambda")
        assert a < 1e-9 and b < 1e-9, (a, b)


def test_lam_eq_of_matches_the_inline_formula():
    """`lam_eq_of` is the single place the two forms are decided: check it against the algebra."""
    Hinv = jnp.diag(jnp.array([1.0, 2.0, 0.5]))
    hess = jnp.array([[1.0, 0.3, 0.0], [0.3, -0.7, 0.1], [0.0, 0.1, 0.4]])
    dlam = jnp.array([0.2, -0.5, 0.7])
    lam = 1.3
    base = float(jnp.einsum("ij,ij->", Hinv, hess)
                 - (1.0 / lam) * jnp.einsum("ij,i,j->", Hinv, dlam, dlam))
    assert float(lam_eq_of(Hinv, hess, dlam, lam)) == pytest.approx(base, rel=1e-14)
    set_lam_eq_form("log")
    try:
        assert float(lam_eq_of(Hinv, hess, dlam, lam)) == pytest.approx(base / lam, rel=1e-14)
    finally:
        set_lam_eq_form("lambda")
