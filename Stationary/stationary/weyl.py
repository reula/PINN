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
    def pair(mass_above: float = 1.0, mass_below: float | None = None,
             half_gap: float = 0.5) -> "Rods":
        """Two rods on the axis, of masses `mass_above` (at z > 0) and `mass_below`.

        A rod of length 2m is a black hole of mass m, so the argument IS the mass: the
        symmetric configuration of the earlier runs is `pair(1, None, 0.5)`.  Unequal masses
        break the z -> -z symmetry, which shows up in the fields (a nonzero dipole in
        lambda: with two unequal holes the coordinate origin is no longer the centre of
        mass) and nowhere in the equations, which is the point of trying it.  Note that a
        rod is also the horizon, so `half_gap` is half the proper distance between the two
        horizons and must stay positive.
        """
        if mass_below is None:
            mass_below = mass_above
        for name, v in (("mass_above", mass_above), ("mass_below", mass_below)):
            if v <= 0.0:
                raise ValueError(f"{name} must be positive")
        if half_gap <= 0.0:
            raise ValueError("half_gap must be positive (half_gap -> 0 merges the holes)")
        # the gap is symmetric about z = 0, so the rods' ends sit at +-(half_gap + 2m)
        return Rods(((-half_gap - 2.0 * mass_below, -half_gap),
                     (half_gap, half_gap + 2.0 * mass_above)))

    @staticmethod
    def symmetric(half_length: float = 1.0, half_gap: float = 0.5) -> "Rods":
        """Two equal rods, one on each side of z = 0, gap 2*half_gap between them."""
        return Rods.pair(half_length, half_length, half_gap)

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
    # same tolerance as h_cart: sin(pi) is not zero, so the pole of a spherical grid arrives
    # just off the axis and must still get the limit rather than a quadrature at rho ~ 1e-18
    return jnp.where(rho > 1e-12 * extent, quad,
                     jnp.where(jnp.abs(z) >= extent, 0.0, jnp.nan))


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
    #
    # The test is a TOLERANCE, not an equality, because sin(pi) is 1.2e-16 and not 0: a node
    # at the south pole of a spherical grid arrives with rho_cyl ~ 1e-18, takes the other
    # branch, and 1/rho_cyl^2 ~ 1e36 turns its second derivatives into ~1e72 -- which is how
    # a curvature map came back with 1e41 in an otherwise smooth field.  Anything within
    # 1e-12 of the rod scale is on the axis for every purpose, and the formula there is
    # numerically meaningless anyway.
    on_axis = rho2 <= (1e-12 * rods.axis_extent) ** 2
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


def rotation_matrix(angle_deg: float):
    """Rotation by `angle_deg` in the z-x plane, i.e. about +y:

        x' =  x cos(a) + z sin(a),   y' = y,   z' = -x sin(a) + z cos(a).

    Chosen so that the rotated chart is the unrotated one seen from a frame turned by the
    same angle; the axis of the Weyl rods becomes n = R z.
    """
    a = jnp.deg2rad(angle_deg)
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotated_fields(rods: Rods, n_quad: int = 400, rot=None):
    """The Weyl solution in rotated coordinates.

    h -> R h R^T and Gamma^i_{jk} -> R^i_a R^j_b R^k_c Gamma^a_{bc} (one index up, two down),
    both evaluated at R^T x; lambda is a scalar and is unchanged.  This is EXACTLY the same
    solution, and it solves the same equations, because they are generally covariant.  What is
    not covariant is the gauge: the chart is no longer adapted to the z axis, so the
    cylindrical condition has to be rotated with it -- `gauge_source_from_metric(h, R z)`.

    A rotated Weyl configuration is the first one in this project with no symmetry at all, so
    it is the first that the axisymmetric ansatz cannot represent and the 3-D one must.
    """
    base = fields_of(rods, n_quad)
    if rot is None:
        return base
    Rt = jnp.asarray(rot).T

    def fields(x):
        f = base(Rt @ x)
        return Fields(rot @ f.h @ Rt,
                      jnp.einsum("ia,jb,kc,abc->ijk", rot, rot, rot, f.G), f.lam)
    return fields


def curvature_invariant(rods: Rods, n_quad: int = 400, axis=None):
    """R_ab R^ab of the exact solution, evaluated in a chart where it is well conditioned.

    The Cartesian components of the Weyl metric are smooth off the axis but their SECOND
    derivatives are not: the chart formula (A x^2 + y^2)/rho_cyl^2 has a 1/rho_cyl^4
    amplification near it, so a curvature computed from them comes out ~1e+04 where the answer
    is ~5e-02 on a slice that contains the axis -- which the rotated runs' slices do, since the
    singular line is the ROTATED axis.

    R_ab R^ab is a scalar, so it may be computed in any chart.  Do it in the cylindrical one,
    h = A(drho^2 + dz^2) + rho^2 dphi^2 with A = e^{2k}, whose components are perfectly smooth:
    the Christoffel symbols and their derivatives come from the same AD as everywhere else, and
    the only conditioning left is the exact 1/rho^2 in the inverse, which the invariant's own
    cancellation handles down to rho ~ 1e-6 of the rod scale.  Closer than that the point is
    blanked (NaN) rather than reported wrong.

    `axis` is the rod axis (None means z), so this works for the rotated configurations too.
    """
    from .geometry import christoffel as _christoffel, ricci_from_gamma  # local: avoid a cycle
    n = jnp.array([0.0, 0.0, 1.0]) if axis is None else jnp.asarray(axis, jnp.float64)
    n = n / jnp.linalg.norm(n)

    def h_cyl(y):
        r, zz = y[0], y[2]
        r = jnp.abs(r)
        A = jnp.exp(2.0 * k_of(r, zz, rods, n_quad))
        return jnp.diag(jnp.array([A, r**2, A]))

    def one(x):
        z = x @ n
        rho = jnp.linalg.norm(x - z * n)
        y = jnp.array([rho, 0.0, z])          # phi = 0: nothing depends on it
        G = _christoffel(h_cyl, y)
        dG = jax.jacfwd(lambda w: _christoffel(h_cyl, w))(y)
        ric = ricci_from_gamma(G, dG)
        hinv = jnp.linalg.inv(h_cyl(y))
        val = jnp.trace(ric @ hinv @ ric @ hinv)
        return jnp.where(rho > 1e-9 * rods.axis_extent, val, jnp.nan)
    return one


def fields_of(rods: Rods, n_quad: int = 400):
    """A `fields(x) -> Fields` callable for the code's residual machinery."""
    def fields(x):
        h = lambda y: h_cart(y, rods, n_quad)
        return Fields(h(x), christoffel(h, x), lam_of(x, rods))
    return fields


def gauge_source_from_metric(h, axis=None):
    """The gauge source in closed form, from the metric alone:

        Gamma^i = (h_rhorho - 1) h^{ij} d_j ln rho_n ,    rho_n = |x - (x.n) n| ,

    the cylindrical radius about the unit axis `n` (`axis=None` means z, the case every run
    before the rotated one used).  Rotating the axis rotates the condition with it, which is
    what a rotated configuration needs: the metric may be rotated freely -- the equations are
    generally covariant -- but the GAUGE is a statement about a chart, so it must be rotated
    by the same amount.

    Exact for any h = A(drho_n^2 + dz_n^2) + rho_n^2 dphi_n^2 written in Cartesian
    components, i.e. for the Weyl family about any axis.  The code below is the z-axis
    version with (x, y, 0) -- the direction perpendicular to z -- replaced by the
    perpendicular part about `n`; d(rho_n)^2 + rho_n^2 d(phi_n)^2 is the flat metric of that
    perpendicular plane whichever plane it is.
    Derivation: with h^{jk}Gamma^i_{jk} = -(1/sqrt(det h)) d_m(sqrt(det h) h^{im}), the
    (x,y) block of this h has determinant exactly A (so det h = A^2, sqrt(det h) = A), the
    zz term A h^{zz} = 1 is constant -- hence Gamma^z = 0 identically -- and in the x
    component every A_rho term cancels, leaving x (A-1)/(A rho^2).

    Unlike `gauge_vector`, this needs no knowledge of U, of k, or of the solution: it is a
    function of the metric components at the point, so it can be imposed as a gauge
    condition on a candidate solution.  Both agree to 6e-17 on the Weyl fields.
    """
    n = jnp.array([0.0, 0.0, 1.0]) if axis is None else jnp.asarray(axis, jnp.float64)
    n = n / jnp.linalg.norm(n)

    def one(x):
        H = h(x)
        perp = x - (x @ n) * n                        # the part perpendicular to the axis
        rho2 = perp @ perp
        rhohat = perp / jnp.sqrt(rho2)
        A = rhohat @ H @ rhohat                       # h_rhorho
        dlnrho = perp / rho2
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
