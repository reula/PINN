"""Weyl solutions: black holes on the axis, in their own (non-harmonic) chart.

The static axisymmetric vacuum metric in Weyl coordinates is

    ds^2 = -e^{2U} dt^2 + e^{-2U} [ e^{2k}(drho^2 + dz^2) + rho^2 dphi^2 ] ,

with U axisymmetric and harmonic with respect to the FLAT 3-metric, and k fixed by

    k_rho = rho (U_rho^2 - U_z^2) ,   k_z = 2 rho U_rho U_z ,   k -> 0 at infinity .

A black hole is a *rod* of length 2m on the axis: a rod spanning z in [a, b] contributes

    U_rod = 1/2 log[ (R_a + R_b - L) / (R_a + R_b + L) ] ,  L = b - a ,  R_a = |(rho, z-a)| ,

whose far field is -m/r, i.e. mass m = L/2.  Two rods is the Israel-Khan solution: two
black holes held apart by the conical strut on the axis between them.  `Rods.symmetric`
builds the equal-mass case; `Rods(spans)` takes any set of rods for later use.

Mapping into the fields this code solves for.  The code's system is the static vacuum
system in conformastatic form: with N = e^U the spatial metric of the Weyl spacetime is
gamma = e^{-2U}[...], and putting h = N^2 gamma, lam = N^2 gives exactly the code's

    Ric(h)_ab = (1/(2 lam^2)) d_a lam d_b lam ,   Delta_h lam = (1/lam) |d lam|^2_h .

Since h = e^{2U} gamma the U-dependence cancels in the metric:

    h   = e^{2k} (drho^2 + dz^2) + rho^2 dphi^2        (Cartesian components below)
    lam = e^{2U}

so the Weyl solutions are solutions of this code's system, with lam = e^{2U} (not e^U:
that factor is what makes the trace of the Ricci equation come out right).

Caveat, and the reason `gauge_vector` exists: this chart is NOT harmonic for h.  The
code's gauge residual is Gamma^i_{jk} h^{jk}, and here it does not vanish -- cylindrical
coordinates never are harmonic (for flat space Delta_h rho = 1/rho != 0).  Imposing
Gamma^i_{jk} h^{jk} = gauge_vector(Weyl) instead of = 0 makes the Weyl solution an exact
solution of the full system in this chart, with no coordinate transformation needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import Fields, christoffel


@dataclass(frozen=True)
class Rods:
    """Rods on the axis (each is a black hole of mass half its length)."""

    spans: tuple[tuple[float, float], ...]

    @staticmethod
    def symmetric(half_length: float = 1.0, half_gap: float = 0.5) -> "Rods":
        """Two equal rods, one on each side of z = 0, gap 2*half_gap between them."""
        if half_length <= 0.0:
            raise ValueError("half_length must be positive")
        if half_gap <= 0.0:
            raise ValueError("half_gap must be positive (half_gap -> 0 merges the holes)")
        return Rods(((-half_gap - 2.0 * half_length, -half_gap),
                     (half_gap, half_gap + 2.0 * half_length)))

    @property
    def masses(self) -> tuple[float, ...]:
        return tuple(0.5 * (b - a) for a, b in self.spans)

    @property
    def total_mass(self) -> float:
        return float(sum(self.masses))

    @property
    def axis_extent(self) -> float:
        """Largest |z| occupied by a rod (the inner sphere must clear this)."""
        return float(max(abs(z) for a, b in self.spans for z in (a, b)))

    def __str__(self) -> str:
        return (f"Rods(spans={self.spans}, masses={self.masses}, "
                f"total_mass={self.total_mass:g}, axis_extent={self.axis_extent:g})")


# --------------------------------------------------------------------- potentials
def U_of(rho, z, rods: Rods):
    """Weyl potential: sum of the rod potentials (they superpose because each is flat-harmonic).

    Written as 1/2 [log(s^2 - L^2) - 2 log(s + L)] with s = Ra + Rb rather than
    1/2 log[(s - L)/(s + L)]: the two agree (since s - L = (s^2-L^2)/(s+L)), but the first
    avoids the cancellation in s - L far from the rod.
    """
    out = jnp.zeros_like(jnp.asarray(rho) + jnp.asarray(z))
    for a, b in rods.spans:
        L = b - a
        Ra = jnp.sqrt(rho**2 + (z - a) ** 2)
        Rb = jnp.sqrt(rho**2 + (z - b) ** 2)
        s = Ra + Rb
        out = out + 0.5 * (jnp.log(jnp.maximum(s**2 - L**2, 1e-300)) - 2.0 * jnp.log(s + L))
    return out


def U_grad(rho, z, rods: Rods):
    """(U_rho, U_z) -- analytic-free, by forward-mode AD."""
    return jax.grad(U_of, 0)(rho, z, rods), jax.grad(U_of, 1)(rho, z, rods)


def _k_rho(rho, z, rods: Rods):
    Ur, Uz = U_grad(rho, z, rods)
    return rho * (Ur**2 - Uz**2)


_GL_NODES: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gauss_legendre(n: int):
    """Nodes and weights on [0, 1] (cached)."""
    if n not in _GL_NODES:
        x, w = np.polynomial.legendre.leggauss(n)
        _GL_NODES[n] = (0.5 * (x + 1.0), 0.5 * w)
    return _GL_NODES[n]


def k_of(rho, z, rods: Rods, n_quad: int = 400):
    """k(rho, z), with k -> 0 at infinity, as the quadrature of k_rho along z = const.

    k(rho) = int_inf^rho k_rho drho' = -int_0^{1/rho} k_rho(1/s, z) ds / s^2 : the
    substitution s = 1/rho' turns the half-infinite range into a finite one and the
    integrand falls off like s as s -> 0, so a Gauss-Legendre rule handles it.  Nothing
    here comes near a rod: the path sits at the target rho > 0, and rods are at rho = 0.
    """
    u, w = _gauss_legendre(int(n_quad))
    smax = 1.0 / jnp.where(rho > 0.0, rho, 1.0)     # the axis is handled below
    s = smax * u                                    # nodes in s, ascending from ~0
    rp = 1.0 / s
    kr = jax.vmap(lambda a, b: _k_rho(a, b, rods))(rp, jnp.full_like(rp, z))
    quad = -jnp.sum(w * smax * kr / s**2)
    # rho = 0 is the axis, where the quadrature cannot be evaluated at all -- its nodes sit
    # at s = u/rho -- so return the exact limit instead of inf/NaN.  This is not a fudge: k
    # vanishes on the axis BEYOND THE OUTERMOST ROD ENDS (the same normalisation that makes
    # k -> 0 at infinity), which is why the metric there is flat in Cartesian components.
    # It matters well beyond tidiness: without it the exact Cartesian h is non-finite on the
    # axis, and ONE non-finite number in an ASCII VTK file stops VisIt from reading every
    # variable that follows it -- a ten-field file then looks like a two-field one.
    # Note "beyond the outermost ends", not "outside every rod": on the axis BETWEEN two
    # rods k is the strut constant (nonzero), and inside a rod it diverges.  No grid point
    # lands there, so NaN is the honest answer for that whole inner stretch.
    extent = max(abs(v) for a, b in rods.spans for v in (a, b))
    return jnp.where(rho > 0.0, quad, jnp.where(jnp.abs(z) >= extent, 0.0, jnp.nan))


# ------------------------------------------------------------------- the code's fields
def h_cart(x, rods: Rods, n_quad: int = 400):
    """The code's h_ij at a Cartesian point x = (x, y, z).

    h = A (drho^2 + dz^2) + rho^2 dphi^2 with A = e^{2k}, transformed with
    drho = (x dx + y dy)/rho and dphi = (x dy - y dx)/rho^2.
    """
    x0, x1, x2 = x[0], x[1], x[2]
    rho2 = x0**2 + x1**2
    rho = jnp.sqrt(rho2)
    A = jnp.exp(2.0 * k_of(rho, x2, rods, n_quad))
    inv = jnp.where(rho2 > 0.0, 1.0 / jnp.maximum(rho2, 1e-300), 0.0)
    # On the axis (rho_cyl = 0) the polar formula is 0/0 and its limit is NOT zero: the
    # angular part collapses, d(rho)^2 + rho^2 d(phi)^2 = dx^2 + dy^2, so h = A (dx^2 +
    # dy^2) + A dz^2 = A * delta there.  Getting this wrong is invisible off the axis but
    # shows up as an isolated O(1) error exactly on it, which is what a VTK export of
    # h(computed) - h(exact) makes obvious and nothing else does.
    on_axis = rho2 <= 0.0
    hxx = jnp.where(on_axis, A, A * x0**2 * inv + x1**2 * inv)
    hyy = jnp.where(on_axis, A, A * x1**2 * inv + x0**2 * inv)
    hxy = (A - 1.0) * x0 * x1 * inv                     # already 0 on the axis
    zero = jnp.zeros_like(hxx)
    return jnp.array([[hxx, hxy, zero],
                      [hxy, hyy, zero],
                      [zero, zero, A]])


def lam_of(x, rods: Rods):
    """lam = e^{2U} at a Cartesian point."""
    rho = jnp.sqrt(x[0] ** 2 + x[1] ** 2)
    return jnp.exp(2.0 * U_of(rho, x[2], rods))


def fields_of(rods: Rods, n_quad: int = 400):
    """A `fields(x) -> Fields` callable for the code's residual machinery."""
    def fields(x):
        h = lambda y: h_cart(y, rods, n_quad)
        return Fields(h(x), christoffel(h, x), lam_of(x, rods))
    return fields


def gauge_source_from_metric(h):
    """The gauge source in closed form, from the metric alone:

        Gamma^i = (h_rhorho - 1) h^{ij} d_j ln rho ,      rho = sqrt(x^2 + y^2) .

    Exact for any h = A(drho^2 + dz^2) + rho^2 dphi^2 written in Cartesian components.
    Derivation: with h^{jk}Gamma^i_{jk} = -(1/sqrt(det h)) d_m(sqrt(det h) h^{im}), the
    (x,y) block of this h has determinant exactly A (so det h = A^2, sqrt(det h) = A), the
    zz term A h^{zz} = 1 is constant -- hence Gamma^z = 0 identically -- and in the x
    component every A_rho term cancels, leaving x (A-1)/(A rho^2).

    Unlike `gauge_vector`, this needs no knowledge of U, of k, or of the solution: it is a
    function of the metric components at the point, so it can be imposed as a gauge
    condition on a candidate solution.  Both agree to 6e-17 on the Weyl fields.
    """
    def one(x):
        H = h(x)
        rho2 = x[0] ** 2 + x[1] ** 2
        rhohat = jnp.array([x[0], x[1], 0.0]) / jnp.sqrt(rho2)
        A = rhohat @ H @ rhohat                       # h_rhorho
        dlnrho = jnp.array([x[0], x[1], 0.0]) / rho2
        return (A - 1.0) * jnp.linalg.solve(H, dlnrho)
    return one


def gauge_vector(rods: Rods, n_quad: int = 400):
    """Gamma^i_{jk} h^{jk} computed independently: from the connection, by autodiff.

    This is the definition, evaluated on the Weyl fields (U by formula, k by quadrature,
    Christoffel symbols and derivatives by AD).  It agrees with `gauge_source_from_metric`
    to machine precision, which is what makes that closed form trustworthy; use this one to
    check, that one to impose.
    """
    def one(x):
        h = lambda y: h_cart(y, rods, n_quad)
        G = christoffel(h, x)
        return jnp.einsum("ijk,jk->i", G, jnp.linalg.inv(h(x)))
    return one
