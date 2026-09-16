"""Exact spherically symmetric solution of the stationary system.

In the HARMONIC chart x^i = rho n^i (the chart in which the coordinate functions
are harmonic, i.e. Gamma^i_{jk} h^{jk} = 0) the general spherically symmetric
solution is the two-parameter family

    h_ij   = (1 - R0^2/rho^2) delta_ij + (R0^2/rho^2) n_i n_j
    lambda = k (rho - R0)/(rho + R0)                     (R0 >= 0, k != 0)

equivalently  h = d rho^2 + (rho^2 - R0^2) dOmega^2.  R0 = 0 is flat space with
constant lambda.  lambda -> 1/lambda is also a symmetry of the system.

Geometric (areal) radius of the coordinate sphere rho = const:

    r_areal = sqrt(rho^2 - R0^2)   (the metric is Riemannian for rho > R0)

so the coordinate sphere carrying the round metric of radius 2 sits at
rho_in = sqrt(4 + R0^2) and, to have lambda = lambda_0 there,

    k = lambda_0 (rho_in + R0)/(rho_in - R0).

The harmonic gauge leaves the residual freedom of harmonic diffeomorphisms; for
spherically symmetric data the harmonic radial coordinate is any

    F(rho) = c1 rho + c2 rho log(1 - R0^2/rho^2)

both members satisfying (f^2 F')' = 2F with f^2 = rho^2 - R0^2.  This freedom is
what allows e.g. the geometric sphere of areal radius 2 to be placed at any
chosen coordinate radius while keeping the gauge harmonic.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from .geometry import Fields, christoffel

I3 = jnp.eye(3)


# ----------------------------------------------------------------- base family
def exact_metric(R0: float) -> Callable:
    def h(x):
        r2 = jnp.dot(x, x)
        n = x / jnp.sqrt(r2)
        c = R0**2 / r2
        return (1.0 - c) * I3 + c * jnp.outer(n, n)

    return h


def exact_lambda(R0: float, k: float) -> Callable:
    def lam(x):
        r = jnp.sqrt(jnp.dot(x, x))
        return k * (r - R0) / (r + R0)

    return lam


def exact_fields(R0: float, k: float) -> Callable[[jnp.ndarray], Fields]:
    """Fields callable (h, Gamma, lambda) in the canonical harmonic chart."""
    h = exact_metric(R0)
    lam = exact_lambda(R0, k)

    def fields(x):
        return Fields(h(x), christoffel(h, x), lam(x))

    return fields


# ------------------------------------------------------------------- geometry
def rho_in(R0: float) -> float:
    """Coordinate radius (in the canonical harmonic chart) of the areal-radius-2 sphere."""
    return float(jnp.sqrt(4.0 + R0**2))


def k_from_lambda0(R0: float, lam0: float) -> float:
    """k such that lambda = lam0 on the areal-radius-2 sphere."""
    r = rho_in(R0)
    return lam0 * (r + R0) / (r - R0)


def areal_radius(rho, R0: float):
    return jnp.sqrt(jnp.asarray(rho) ** 2 - R0**2)


def lambda_infinity(k: float) -> float:
    """Asymptotic value lambda -> k as rho -> infinity."""
    return k


# ------------------------------------------------- residual gauge freedom
def F2(R0: float, rho):
    """Second solution of (f^2 F')' = 2F, f^2 = rho^2 - R0^2   (F1 = rho).

    Reduction of order with F = rho G gives G' = c/(rho^2 (rho^2-R0^2)) and hence
        F2 = 2 R0 + rho log((rho - R0)/(rho + R0)) ,
    which is a solution *only with* the additive constant 2R0 (that constant is
    not a multiple of the homogeneous solution F1 = rho).
    """
    return 2.0 * R0 + rho * jnp.log((rho - R0) / (rho + R0))


def F2_prime(R0: float, rho):
    return jnp.log((rho - R0) / (rho + R0)) + 2.0 * R0 * rho / (rho**2 - R0**2)


def chart_map(R0: float, rho, c1: float = 1.0, c2: float = 0.0):
    """F(rho) = c1 rho + c2 F2(rho): radial part of a harmonic coordinate change."""
    return c1 * rho + c2 * F2(R0, rho)


def chart_map_prime(R0: float, rho, c1: float = 1.0, c2: float = 0.0):
    return c1 + c2 * F2_prime(R0, rho)


def monotonic_rho_lo(R0: float, c1: float, c2: float, rho_hi: float = 100.0,
                     n: int = 40000) -> float:
    """Smallest rho such that F = c1 rho + c2 F2 is strictly increasing beyond it.

    F2' -> -inf as rho -> R0, so for c2 > 0 the map is non-monotonic in a
    neighbourhood of R0; this returns a safe lower end for the interpolation grid.
    """
    rg = jnp.linspace(R0 * (1.0 + 1e-6) + 1e-9, rho_hi, n)
    Fp = chart_map_prime(R0, rg, c1, c2)
    bad = jnp.nonzero(Fp <= 0.0)[0]
    if bad.size == 0:
        return float(rg[0])
    i = int(bad.max()) + 1
    if i >= n:
        raise ValueError("chart map is not increasing anywhere on the scanned range")
    return float(rg[i])


def _invert_chart(R0: float, rho_t, c1: float, c2: float, rg, Fg, iters: int = 12):
    """Solve F(rho) = rho_t for rho > R0 (Newton, interpolated guess).

    Piecewise-linear interpolation of the inverse would limit the accuracy of the
    reconstructed metric (and hence show up as a spurious gauge residual), so the
    interpolation is used only as an initial guess.
    """
    rho = jnp.interp(rho_t, Fg, rg)
    for _ in range(iters):
        f = chart_map(R0, rho, c1, c2) - rho_t
        fp = chart_map_prime(R0, rho, c1, c2)
        rho = rho - f / fp
    return rho


def exact_fields_in_harmonic_chart(R0: float, k: float, c1: float = 1.0,
                                   c2: float = 0.0, rho_lo: float | None = None,
                                   rho_hi: float = 400.0, ngrid: int = 40001):
    """The same exact solution written in the harmonic chart x~^i = F(rho) n^i.

    Useful for constructing exactly solvable boundary data on a shell whose inner
    boundary is placed at a prescribed coordinate radius.  Requires F strictly
    increasing on [rho_lo, rho_hi].
    """
    if rho_lo is None:
        rho_lo = monotonic_rho_lo(R0, c1, c2) * (1.0 + 1e-9)
    rg = jnp.linspace(rho_lo, rho_hi, ngrid)
    Fg = chart_map(R0, rg, c1, c2)
    Fpg = chart_map_prime(R0, rg, c1, c2)
    if not bool(jnp.all(jnp.diff(Fg) > 0)):
        raise ValueError("chart map F is not strictly increasing on the grid")

    def rho_of(x):
        return _invert_chart(R0, jnp.linalg.norm(x), c1, c2, rg, Fg)

    def h_of(x):
        rho_t = jnp.linalg.norm(x)
        n = x / rho_t
        rho = _invert_chart(R0, rho_t, c1, c2, rg, Fg)
        Fp = chart_map_prime(R0, rho, c1, c2)
        H = 1.0 / Fp**2                       # h_{rho~rho~}
        g2 = (rho**2 - R0**2) / rho_t**2      # tangential coefficient
        nn = jnp.outer(n, n)
        return H * nn + g2 * (I3 - nn)

    def lam_of(x):
        r = rho_of(x)
        return k * (r - R0) / (r + R0)

    def fields(x):
        return Fields(h_of(x), christoffel(h_of, x), lam_of(x))

    return fields


def chart_with_inner_at(R0: float, rho_coord: float, h_rr: float = 1.0):
    """Harmonic chart (c1, c2) in which the areal-radius-2 sphere sits at rho = rho_coord
    and the normal-normal component there is h_rr.

    F = c1 rho + c2 F2 satisfies F(rho_geom) = rho_coord and F'(rho_geom) = 1/sqrt(h_rr),
    with rho_geom = sqrt(4+R0^2) the canonical coordinate of the areal-radius-2 sphere
    and h_{rho~rho~} = 1/F'^2.
    """
    rg = rho_in(R0)
    A = jnp.array([[rg, float(F2(R0, rg))], [1.0, float(F2_prime(R0, rg))]])
    b = jnp.array([rho_coord, 1.0 / jnp.sqrt(h_rr)])
    c1, c2 = jnp.linalg.solve(A, b)
    return float(c1), float(c2)


def reference_fields(R0: float = 1.0, lam0: float = 1.0, rho_coord: float = 2.0,
                     h_rr: float = 1.0):
    """Exact solution placed so that it satisfies the inner boundary conditions of the
    Milestone-2 shell (areal radius 2, h_rr = h_rr, lambda = lam0 at rho = rho_coord).

    Its asymptotic value is k = lam0 (rho_geom + R0)/(rho_geom - R0), which is what the
    outer Robin condition on lambda should use; the Robin conditions themselves are then
    satisfied to leading order in 1/rho, i.e. to a few percent at rho = 20.
    """
    c1, c2 = chart_with_inner_at(R0, rho_coord, h_rr)
    k = k_from_lambda0(R0, lam0)
    return exact_fields_in_harmonic_chart(R0, k, c1=c1, c2=c2), dict(c1=c1, c2=c2, k=k)


def reference_fields_asymptotic(R0: float = 1.0, k: float = 1.0, rho_coord: float = 2.0,
                                c1: float = 1.0):
    """Exact solution with prescribed asymptotic value lambda -> k, placed in the
    harmonic chart where the areal-radius-2 sphere sits at rho = rho_coord.

    c1 = 1 keeps h -> delta_ij at infinity; c2 is then fixed by the inner condition.
    lambda on the inner sphere is k (rho_geom-R0)/(rho_geom+R0), rho_geom = sqrt(4+R0^2).
    """
    rg = rho_in(R0)
    c2 = (rho_coord - c1 * rg) / float(F2(R0, rg))
    return exact_fields_in_harmonic_chart(R0, k, c1=c1, c2=c2), dict(c1=c1, c2=c2, k=k)
