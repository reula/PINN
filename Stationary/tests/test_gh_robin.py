"""The scalar outer condition on the Geroch-Hansen potential P = (1 - lam^2)/(4 lam).

Why not on lambda itself.  lam = e^{2U} with U harmonic, so

    lam = 1 - 2M/rho + 2M^2/rho^2 - ...,      P = -sinh(2U)/2 = M/rho + O(rho^-3).

The rho^-2 POWER of lambda therefore carries two things: the l = 1 DIPOLE, and the l = 0 term
2M^2/rho^2 which the field equations require.  A radial condition cannot tell them apart, so
dropping the dipolar factor from lambda's kernel would forbid the physical 2M^2 as well -- it
would put the boundary condition in conflict with the equations and drive M^2 -> 0.  In P the
l = 0 part has no rho^-2 at all, so P's rho^-2 power is the dipole alone and dropping it costs
nothing.

Measured on the production reference (M ~ 0.574 from lam(99) = 0.988404): the l = 0 coefficients
are 2M^2 = 0.659 for lambda (6.6e-05 at rho = 100) and zero for P.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from stationary.losses import drop_power, gh_potential, robin_operator_exps

jax.config.update("jax_enable_x64", True)

M = 0.574
Y10 = lambda y: y[2] / jnp.linalg.norm(y)          # P_1(mu)


def lam_mass(y):
    """lambda = e^{2U} truncated to the powers that matter: 1 - 2M/rho + 2M^2/rho^2."""
    r = jnp.linalg.norm(y)
    return 1.0 - 2 * M / r + 2 * M ** 2 / r ** 2


def lam_mass_and_dipole(y, d=1e-2):
    r = jnp.linalg.norm(y)
    return lam_mass(y) + d * Y10(y) / r ** 2


def test_the_potential_of_a_mass_only_lambda_has_no_l0_rho_minus_two():
    """The point of using P: its l = 0 tail is rho^-1, rho^-3, ... -- no rho^-2."""
    y = jnp.array([60.0, 0.0, 80.0])               # mu = 0.8, off the equator
    p = gh_potential(lam_mass(y))
    # the dipole-free kernel annihilates P (no rho^-2 content) but NOT lambda (which has 2M^2/rho^2)
    kern = drop_power([1.0, 2.0, 3.0], 2.0)
    assert kern == [1.0, 3.0, 4.0]
    p_res = float(jnp.abs(robin_operator_exps(lambda z: gh_potential(lam_mass(z)), y, kern, 0.0)))
    l_res = float(jnp.abs(robin_operator_exps(lambda z: lam_mass(z), y, kern, 1.0)))
    assert p_res < 1e-9, p_res                      # P: nothing physical was killed
    assert l_res > 1e-5, l_res                      # lambda: the physical 2M^2/rho^2 is hit
    assert l_res > 1e5 * p_res, (l_res, p_res)      # and P is untouched by comparison


def test_the_gh_condition_sees_a_dipole_that_the_lambda_condition_tolerates():
    y = jnp.array([60.0, 0.0, 80.0])
    kern = drop_power([1.0, 2.0, 3.0], 2.0)
    gh = abs(float(robin_operator_exps(lambda z: gh_potential(lam_mass_and_dipole(z)), y, kern, 0.0)))
    naive = abs(float(robin_operator_exps(lambda z: lam_mass_and_dipole(z), y, [1.0, 2.0, 3.0], 1.0)))
    assert gh > 1e-7, gh                            # the dipole is visible
    assert naive < 1e-3 * gh, (naive, gh)           # the naive window accepts it (kernel)


def test_the_gh_residual_vanishes_on_the_dipole_free_field():
    y = jnp.array([60.0, 0.0, 80.0])
    kern = drop_power([1.0, 2.0, 3.0], 2.0)
    gh = abs(float(robin_operator_exps(lambda z: gh_potential(lam_mass(z)), y, kern, 0.0)))
    assert gh < 1e-9, gh


def test_the_flags_default_to_the_gh_variant():
    from stationary.train import parse_args
    a = parse_args(["--outdir", "/tmp/x"])
    assert a.robin_lam_potential is True and a.robin_lam_drop_dipole is True
    b = parse_args(["--outdir", "/tmp/x", "--robin-lam-lambda", "--keep-lam-dipole"])
    assert b.robin_lam_potential is False and b.robin_lam_drop_dipole is False
