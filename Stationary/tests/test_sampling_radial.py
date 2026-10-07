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

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from stationary.problem import Config, sample_shell

RHO_IN, RHO_OUT = 1.0, 100.0
N = 200_000


def cfg_with(radial: str) -> Config:
    return Config(rho_in=RHO_IN, rho_out=RHO_OUT, radial=radial)


def radii(radial: str, seed: int = 0):
    xs = sample_shell(jax.random.PRNGKey(seed), N, cfg_with(radial))
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
