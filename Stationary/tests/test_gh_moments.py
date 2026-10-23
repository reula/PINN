"""Geroch-Hansen moments from the axis (stationary/gh_moments.py), calibrated on exact solutions.

The checks of MULTIPOLES_GH_THORNE.md (appendix): Curzon (M2 = -M^3/3, M4 = 19 M^5/105, although
U has no quadrupole), Schwarzschild in harmonic coordinates (M0 = R0, M_l = 0), the default
two-rod Weyl pair (M0 = 2, M2 = 2.5), unequal rods against the closed form, and mass centring.
"""
from __future__ import annotations

import math

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from stationary import exact, weyl
from stationary import gh_moments as G


def test_curzon_maps():
    M = 0.7
    Z = np.geomspace(5.0, 200.0, 400)
    Z = np.concatenate([Z, -Z])
    cU, _ = G.fit_axis_series(Z, -M / np.abs(Z), 8)
    cX, _ = G.fit_axis_series(Z, np.tanh(M / np.abs(Z)), 10)
    assert np.isclose(G.fhps_moments(cU)[2], -M ** 3 / 3, rtol=1e-9)
    P = G.ernst_moments(cX)
    assert np.isclose(P[0], M, rtol=1e-12) and np.isclose(P[2], -M ** 3 / 3, rtol=1e-9)
    assert np.isclose(P[4], 19 * M ** 5 / 105, rtol=1e-6)


def test_schwarzschild_in_harmonic_coordinates():
    R0 = 1 / math.sqrt(3)
    p = G.axis_profile(exact.exact_fields(R0, 1.0), (0, 0, 1), 2.0, 100.0, n=1500)
    assert abs(p["gap"] - 4.0) < 1e-10                       # Z = rho on this axis
    assert np.allclose(p["north"]["Z"], p["north"]["s"], atol=1e-9)
    r = G.moments_from_profile(p, order=8)
    P = r["M_ernst"]
    assert abs(P[0] - R0) < 1e-12
    assert max(abs(v) for v in P[1:5]) < 1e-11


def test_two_rods_and_mass_centring():
    for rods in (weyl.Rods.symmetric(1.0, 0.5), weyl.Rods.pair(1.0, 0.1, 0.5)):
        p = G.axis_profile(weyl.fields_of(rods), (0, 0, 1), 7.5, 75.0, n=1200)
        assert abs(p["gap"] - 15.0) < 1e-9                   # the Weyl chart: Z = z
        r = G.moments_from_profile(p, order=8)
        ex, zc = G.rods_moments(rods.spans)
        assert abs(r["shift"] - zc) < 1e-7
        assert np.allclose(r["M_ernst"][:4], ex, rtol=1e-4, atol=1e-6), (r["M_ernst"], ex)
    sym = G.moments_from_profile(G.axis_profile(weyl.fields_of(weyl.Rods.symmetric(1.0, 0.5)),
                                                (0, 0, 1), 7.5, 75.0, n=1200), order=8)
    assert abs(sym["M_ernst"][0] - 2.0) < 1e-9 and abs(sym["M_ernst"][2] - 2.5) < 1e-6
