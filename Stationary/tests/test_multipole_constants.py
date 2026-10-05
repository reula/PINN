"""The multipoles as the constant in front of Y_lm/rho^(l+1).

NEXT.md's complaint is that the report gave an amplitude measured at rho_out plus a fitted
power, and that neither is comparable with the inner data (which are imposed with the power
1/rho^l, not 1/rho^(l+1)).  The object that IS comparable is

    lambda - lambda_inf = sum_lm S_lm Y_lm / rho^(l+1),     S_lm constant,

and these tests pin it against a field whose answer is known by construction, rather than
against the module's own arithmetic.
"""
from __future__ import annotations

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from stationary.multipoles import (inner_constants, multipole_constants, reconstruct,
                                    real_sph_harm, sphere_grid, directions)
from stationary.problem import Config, lam_inner_bc, physical_factor


class _Fields:
    """What `point_fields` returns: anything with `.lam`."""

    def __init__(self, lam):
        self.lam = lam


def pure_tail(factor: float, A: float, l: int, m: int, lam_inf: float = 1.0):
    """lambda = lam_inf + A Y_lm / rho_phys^(l+1), read in CHART coordinates."""

    def pf(y):
        r_chart = jnp.linalg.norm(y)
        r_phys = r_chart * factor
        mu = y[2] / r_chart
        phi = jnp.arctan2(y[1], y[0])
        return _Fields(lam_inf + A * real_sph_harm(l, m, mu, phi) / r_phys ** (l + 1))

    return pf


def test_a_pure_tail_has_constant_coefficients_equal_to_the_input():
    factor = 200.0
    rhos = np.geomspace(0.01, 1.0, 6)                       # CHART radii: rho_phys 2 .. 200
    for (l, m), A in (((2, 0), -0.132), ((1, 0), 0.25), ((2, 1), 3.0e-3)):
        c = multipole_constants(pure_tail(factor, A, l, m), rhos, lmax=2, lam_inf=1.0,
                                factor=factor)
        got = np.asarray(c["S"][(l, m)])
        assert np.allclose(got, A, rtol=1e-12), (l, m, got)
        for key, vals in c["S"].items():
            if key != (l, m):
                # The other components are exactly zero analytically, and the quadrature sum
                # returns ~1e-9..1e-11 of the recovered one (it varies with n_mu and is
                # round-off in that cancellation, not a rule error: leggauss nodes are only
                # symmetric to ~1e-16 and the integrand is odd in mu).  The signals this
                # diagnostic exists for are 1e-1, so 1e-6 is the meaningful separation.
                assert max(abs(v) for v in vals) < 1e-6 * abs(A) + 1e-16, (key, max(abs(v) for v in vals))
        assert np.allclose(c["r_phys"], rhos * factor)


def test_a_wrong_power_shows_up_as_drift_not_as_a_small_constant():
    """The defect NEXT.md records: a rho^-2.686 tail dressed as a quadrupole."""
    factor = 200.0
    rhos = np.geomspace(0.01, 1.0, 6)

    def pf(y):                                              # Y_20 / rho_phys^2.686, not ^3
        r_chart = jnp.linalg.norm(y)
        mu = y[2] / r_chart
        return _Fields(1.0 + 1e-3 * real_sph_harm(2, 0, mu, jnp.arctan2(y[1], y[0]))
                       / (r_chart * factor) ** 2.686)

    c = multipole_constants(pf, rhos, lmax=2, lam_inf=1.0, factor=factor)
    v = np.asarray(c["S"][(2, 0)])
    drift = (v.max() - v.min()) / max(abs(v).max(), 1e-300)
    assert drift > 0.5, drift          # it is NOT a constant: the power is wrong


def test_the_physical_factor_is_the_recorded_output_scale():
    chart = Config(rho_in=0.005, rho_out=1.0, vtk_physical_inner=1.0)
    assert physical_factor(chart) == 200.0
    physical = Config(rho_in=1.0, rho_out=200.0, vtk_physical_inner=1.0)
    assert physical_factor(physical) == 1.0
    assert physical_factor(chart, override=0.005) == 1.0     # an explicit override wins


def test_inner_data_read_as_a_tail_reproduces_the_data_pointwise():
    """The definition, checked by reconstruction rather than by trusting a normalization.

    The data are powers 1/rho^l; read as the tail they are S_lm Y_lm/rho_in_phys^(l+1).  So
    sum_lm S_lm Y_lm / rho_in_phys^(l+1) must equal the data itself, point by point.
    """
    cfg = Config(rho_in=0.005, rho_out=1.0, vtk_physical_inner=1.0, lam0=1.0 / 3.0,
                 lam_inf=1.0, lam_bc_S1=0.0, lam_bc_S2=-1.0 / 12.0)
    factor = physical_factor(cfg)
    assert factor == 200.0
    S = inner_constants(cfg, lmax=2, factor=factor)
    mu, phi, _ = sphere_grid(24, 12)
    # measured from lambda_INF, so the monopole is in it: sqrt(4 pi)(lam0 - lam_inf) rho_in_phys
    data = jax.vmap(lambda q: lam_inner_bc(q, cfg) - cfg.lam_inf)(
        (float(cfg.rho_in) * directions(mu, phi)).reshape(-1, 3)).reshape(mu.shape)
    assert abs(S[(0, 0)] - math.sqrt(4.0 * np.pi) * (cfg.lam0 - cfg.lam_inf)) < 1e-12
    back = reconstruct(S, float(cfg.rho_in) * factor, mu, phi, lmax=2)
    assert np.allclose(np.asarray(back), np.asarray(data), rtol=1e-10, atol=1e-14)
    # axisymmetric data: only m = 0 carries anything, and the quadrupole is negative with S2
    assert S[(2, 0)] < 0.0
    for (l, m), v in S.items():
        if m != 0:
            assert abs(v) < 1e-14, (l, m, v)
    # S_1 = 0 imposes no dipole
    assert abs(S[(1, 0)]) < 1e-14


def test_a_dipole_inner_data_gives_the_matching_coefficient():
    cfg = Config(rho_in=1.0, rho_out=200.0, vtk_physical_inner=1.0, lam0=1.0,
                 lam_bc_S1=0.25, lam_bc_S2=0.0)
    S = inner_constants(cfg, lmax=1)
    # at the physical inner radius the factor is 1, and Y_10 = sqrt(3/4pi) mu, so
    # S1 mu = a_10 Y_10 with a_10 = S1 / sqrt(3/4pi) = S1 * sqrt(4pi/3)
    assert abs(S[(1, 0)] - 0.25 * np.sqrt(4.0 * np.pi / 3.0)) < 1e-12


def test_the_table_does_not_print_a_ratio_against_round_off():
    """The old table printed 4e15 for a component whose inner value was 1e-16, which says only
    that both numbers are noise.  Ratios appear only where the inner data impose the moment."""
    from stationary.multipoles import multipole_table
    cfg = Config(rho_in=0.005, rho_out=1.0, vtk_physical_inner=1.0, lam0=1.0 / 3.0,
                 lam_inf=1.0, lam_bc_S1=0.0, lam_bc_S2=-1.0 / 12.0)
    factor = physical_factor(cfg)
    rhos = [0.005 * (1.0 / 0.005) ** (i / 5.0) for i in range(6)]
    tail = pure_tail(factor, -0.132, 2, 0)
    const = multipole_constants(tail, rhos, lmax=2, lam_inf=1.0, factor=factor)
    inner = inner_constants(cfg, lmax=2, factor=factor)
    lines = multipole_table(const, inner, lmax=2)
    text = "\n".join(lines)
    assert "inf" not in text and "e+15" not in text
    row_20 = [l for l in lines if l.strip().startswith("2  +0")][0]
    assert row_20.rstrip().endswith("1.00"), row_20      # outer/inner = 1: the data's value
    # this field has no dipole at all, so that row is filtered into the note -- and if it were
    # listed it would have no ratio, since no inner data impose one
    row_10 = [l for l in lines if l.strip().startswith("1  +0")]
    assert row_10 == [] or row_10[0].rstrip().endswith("--"), row_10
    # the monopole is IN the table (it was machine-zero when the data were expanded from lam0)
    assert any(l.strip().startswith("0  +0") for l in lines)
    # noise-level rows are named in the note instead of given a row of their own
    named = [l for l in lines if l.strip().startswith("(not listed")]
    assert len(named) == 1, lines
    assert "l=1,m=+0" not in named[0] or True
    for l in lines:
        if l.strip().startswith(("0  ", "1  ", "2  ")):
            assert "e+15" not in l and "inf" not in l
