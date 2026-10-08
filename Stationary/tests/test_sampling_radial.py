"""The three radial samplers, pinned by the distribution they produce.

`sample_shell` draws a radius and an isotropic direction; the three settings differ only in the
radial law, and that difference decides how much say the far field has in the loss once the
residuals are weighted by rho^d (the loss averages (weight x residual)^2 over the sampled
points, so it estimates the integral against the sampling density):

    "log"      rho log-uniform  ->  constant count per decade, density per volume ~ rho^-3
    "uniform"  rho uniform      ->  constant count per unit rho, density per volume ~ rho^-2
    "volume"   rho^3 uniform    ->  count per unit rho ~ rho^2, density per volume constant

The last one is what "the number of points grows like rho^2 on each sphere" means: with rho^3
uniform on [rho_in^3, rho_out^3] the fraction of points inside rho = r is

    P(r) = (r^3 - rho_in^3) / (rho_out^3 - rho_in^3).

These tests check that fraction (and the per-shell counts) rather than a histogram shape, so
they do not depend on binning.
"""
from __future__ import annotations

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from stationary.problem import Config, sample_shell

RHO_IN, RHO_OUT = 1.0, 100.0
N = 200_000


def cfg_with(radial: str, frac: float | None = None) -> Config:
    cfg = Config(rho_in=RHO_IN, rho_out=RHO_OUT, radial=radial)
    if frac is not None:
        cfg.radial_log_frac = float(frac)
    return cfg


def radii(radial: str, seed: int = 0, frac: float | None = None):
    xs = sample_shell(jax.random.PRNGKey(seed), N, cfg_with(radial, frac))
    return np.asarray(jnp.linalg.norm(xs, axis=-1))


def test_directions_are_unit_vectors():
    xs = sample_shell(jax.random.PRNGKey(3), 5000, cfg_with("volume"))
    r = np.asarray(jnp.linalg.norm(xs, axis=-1))
    assert np.allclose(r, np.linalg.norm(np.asarray(xs), axis=-1))     # radius = |x| by construction
    assert np.all(r > RHO_IN * 0.999) and np.all(r < RHO_OUT * 1.001)


def test_volume_is_uniform_in_volume():
    """The new law: P(r) = (r^3 - rho_in^3)/(rho_out^3 - rho_in^3)."""
    r = radii("volume")
    for frac in (0.25, 0.5, 0.75):
        r_target = (RHO_IN**3 + frac * (RHO_OUT**3 - RHO_IN**3)) ** (1.0 / 3.0)
        got = float(np.mean(r < r_target))
        assert abs(got - frac) < 0.01, (frac, r_target, got)


def test_log_is_uniform_in_log_rho():
    r = radii("log")
    for frac in (0.25, 0.5, 0.75):
        r_target = RHO_IN * (RHO_OUT / RHO_IN) ** frac
        got = float(np.mean(r < r_target))
        assert abs(got - frac) < 0.01, (frac, r_target, got)


def test_uniform_is_uniform_in_rho():
    r = radii("uniform")
    for frac in (0.25, 0.5, 0.75):
        r_target = RHO_IN + frac * (RHO_OUT - RHO_IN)
        got = float(np.mean(r < r_target))
        assert abs(got - frac) < 0.01, (frac, r_target, got)


def test_the_count_per_sphere_grows_like_rho_squared_for_volume():
    """Restated as counts per shell, which is how it was asked for: with "volume" the number of
    points in [r, r+dr] is proportional to r^2 dr (flat per unit volume); with "log" it is the
    same for every decade.  Compared only between shells with enough points that Poisson noise
    is a per-cent effect -- the sparsest shells of a 200k sample hold a handful of points.
    """
    r = radii("volume")
    lo, hi = np.histogram(r, bins=[16.0, 32.0, 64.0])[0]
    vol_lo = 32.0**3 - 16.0**3
    vol_hi = 64.0**3 - 32.0**3
    assert lo > 3000 and hi > 20_000, (lo, hi)
    assert (hi / lo) == pytest.approx(vol_hi / vol_lo, rel=0.03), (lo, hi)

    rl = radii("log")
    a, b = np.histogram(rl, bins=[1.0, 10.0, 100.0])[0]
    assert a > 50_000 and b > 50_000, (a, b)
    assert b / a == pytest.approx(1.0, rel=0.05), (a, b)


def test_the_three_laws_are_actually_different():
    """A sampler that silently ignored the flag would pass nothing here: compared on the outer
    half of the shell, volume > uniform > log in the fraction of points."""
    frac = {k: float(np.mean(radii(k) > 50.0)) for k in ("log", "uniform", "volume")}
    assert frac["log"] < frac["uniform"] < frac["volume"], frac


def test_hybrid_is_the_mixture_of_the_two_laws():
    """CDF(r) = frac P_log(r) + (1-frac) P_volume(r), for several fractions and radii."""
    for frac in (0.25, 0.5, 0.75):
        r = radii("hybrid", frac=frac)
        for r_t in (1.5, 2.0, 5.0, 20.0, 60.0):
            p_log = math.log(r_t / RHO_IN) / math.log(RHO_OUT / RHO_IN)
            p_vol = (r_t**3 - RHO_IN**3) / (RHO_OUT**3 - RHO_IN**3)
            want = frac * p_log + (1.0 - frac) * p_vol
            got = float(np.mean(r < r_t))
            assert abs(got - want) < 0.01, (frac, r_t, want, got)


def test_hybrid_keeps_the_near_field_that_pure_volume_loses():
    """The reason the hybrid exists: at a shell ratio of 100, pure volume leaves the inner
    boundary layer (rho < 2, where lambda_0 and the steep rise live) with ~1 point in 200k,
    while half-log keeps it at the log law's density."""
    near_h = float(np.mean(radii("hybrid", frac=0.5) < 2.0))
    near_v = float(np.mean(radii("volume") < 2.0))
    p_log_2 = math.log(2.0 / RHO_IN) / math.log(RHO_OUT / RHO_IN)
    assert near_h == pytest.approx(0.5 * p_log_2, rel=0.05), (near_h, near_v)
    assert near_h > 1000.0 * max(near_v, 1.0 / N), (near_h, near_v)
    # ... and the far field is still represented: half the points are volume-uniform
    far_h = float(np.mean(radii("hybrid", frac=0.5) > 10.0))
    assert far_h > float(np.mean(radii("log") > 10.0)), far_h


def test_hybrid_endpoints_are_the_pure_laws():
    a = radii("hybrid", frac=0.0)
    b = radii("volume")
    c = radii("hybrid", frac=1.0)
    d = radii("log")
    for r_t in (2.0, 10.0):
        assert abs(float(np.mean(a < r_t)) - float(np.mean(b < r_t))) < 0.005
        assert abs(float(np.mean(c < r_t)) - float(np.mean(d < r_t))) < 0.005
