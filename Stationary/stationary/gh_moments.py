"""Geroch-Hansen mass moments of a static, axisymmetric solution, from the fields on its axis.

Route (MULTIPOLES_GH_THORNE.md sections 4.1-4.3, 6.2 item 3): in canonical Weyl coordinates a
static axisymmetric vacuum metric is

    g = -e^{2U} dt^2 + e^{-2U} [ e^{2k}(dW^2 + dZ^2) + W^2 dphi^2 ],

and the moments follow from the axis expansion U(W=0, Z) = sum_r U^(r) / |Z|^(r+1) (FHPS:
M0 = -U0, M1 = -U1, M2 = U0^3/3 - U2, M3 = U0^2 U1 - U3), or from the static Ernst potential
xi = -tanh U, xi(0, Z) = sum_j m_j / |Z|^(j+1) (Sotiriou-Apostolatos: P0..P3 = m0..m3,
P4 = m4 - m0(m0 m2 - m1^2)/7, P5 = ...).  The two maps are algebraically unrelated, so their
agreement for l <= 3 is a check of the extraction.  Normalisation N1: M0 is the mass.

No Weyl chart has to be built, because only the AXIS is needed:

* in this code's variables g = -lam dt^2 + lam^{-1} h, so U = (1/2) ln lam and the Weyl radius is
  W = |d_phi|_h (the h-norm of the axial Killing vector d_phi = e x x);
* on the part of the axis that reaches infinity k = 0 (elementary flatness, k -> 0 at infinity),
  so along the axis dZ = ds_h, the proper h-length: Z is the arc length of the axis in h;
* the axis is cut by the inner ball, so the offset between its north and south branches is not an
  arc length.  It is the change of Z along any meridian curve outside the sources, with
  dZ = *dW (W and Z are conjugate harmonic functions of the conformally flat meridian metric
  q = h restricted to the meridian plane): `meridian_gap` integrates it over a semicircle.

The origin of Z is fixed afterwards by mass centring (M1 = 0), which for these solutions is the
centre-of-mass condition of COM_GAUGE.md.  Each fit is a least-squares fit of the axis values to
the series in 1/|Z| over [rho_lo, rho_hi] on both branches; `scan` varies the window and the order
and returns the spread as an error bar.  Convergence of the series needs |Z| beyond the sources.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _perp(e):
    """A unit vector orthogonal to e (the meridian plane is span(e, u))."""
    t = np.array([1.0, 0.0, 0.0]) if abs(e[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = t - (t @ e) * e
    return u / np.linalg.norm(u)


# ------------------------------------------------------------------ the axis and its Weyl Z
def axis_values(fields, e, s):
    """(lam, h_ee) at the points s*e (s may be negative: the south branch)."""
    e = jnp.asarray(e)

    def one(si):
        f = fields(si * e)
        return f.lam, e @ f.h @ e
    lam, hee = jax.vmap(one)(jnp.asarray(s))
    return np.asarray(lam), np.asarray(hee)


def meridian_gap(fields, e, rho, n: int = 256):
    """Z(north pole) - Z(south pole) of the sphere of radius rho, from dZ = *dW on a semicircle.

    Meridian plane span(e, u), coordinates (a, b): x = a e + b u.  W = |e x x|_h, the metric of
    the plane is q_ij = h(e_i, e_j), and (*dW)_i = sqrt(det q) eps_ij q^jk d_k W with eps_ab = +1,
    which in flat Weyl coordinates (a = Z, b = W) is dZ = da.  The semicircle runs from the north
    pole (theta = 0) to the south pole (theta = pi); the result is minus that integral.
    """
    e = jnp.asarray(_unit(e))
    u = jnp.asarray(_perp(np.asarray(e)))

    def W(ab):
        x = ab[0] * e + ab[1] * u
        k = jnp.cross(e, x)
        return jnp.sqrt(k @ fields(x).h @ k)

    def star_dW_dot_t(theta):
        ab = rho * jnp.array([jnp.cos(theta), jnp.sin(theta)])
        x = ab[0] * e + ab[1] * u
        h = fields(x).h
        E = jnp.stack([e, u])
        q = E @ h @ E.T
        g = jax.grad(W)(ab)
        v = jnp.sqrt(jnp.linalg.det(q)) * (jnp.linalg.inv(q) @ g)    # sqrt(q) q^{-1} dW
        star = jnp.array([v[1], -v[0]])                             # eps_ij v^j, eps_ab = +1
        t = rho * jnp.array([-jnp.sin(theta), jnp.cos(theta)])      # d(ab)/d(theta)
        return star @ t

    xg, wg = np.polynomial.legendre.leggauss(n)
    th = 0.5 * math.pi * (xg + 1.0)
    vals = np.asarray(jax.vmap(star_dW_dot_t)(jnp.asarray(th)))
    return -float(0.5 * math.pi * np.sum(wg * vals))


def axis_profile(fields, axis=(0.0, 0.0, 1.0), rho_lo: float = 2.0, rho_hi: float = 100.0,
                 n: int = 4000, gap_rho: float | None = None):
    """Weyl Z and U = ln(lam)/2 along both branches of the axis, rho in [rho_lo, rho_hi].

    Z_north(s) = G/2 + int_{gap_rho}^{s} sqrt(h_ee) ds,  Z_south(s) = -G/2 - int ..., with G the
    meridian gap at gap_rho (default rho_lo); the origin (the midpoint) is provisional.
    """
    e = _unit(axis)
    g_rho = rho_lo if gap_rho is None else gap_rho
    s = np.geomspace(min(rho_lo, g_rho), rho_hi, n)
    G = meridian_gap(fields, e, g_rho)
    out = {"axis": e.tolist(), "gap": G, "gap_rho": g_rho}
    for name, sign in (("north", 1.0), ("south", -1.0)):
        lam, hee = axis_values(fields, e, sign * s)
        ds = np.sqrt(hee)
        arc = np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(s))])
        arc -= np.interp(g_rho, s, arc)
        Z = sign * (0.5 * G + arc)
        keep = s >= rho_lo
        out[name] = {"s": s[keep], "Z": Z[keep], "U": 0.5 * np.log(lam[keep]), "lam": lam[keep],
                     "h_ee": hee[keep]}
    return out


# ------------------------------------------------------------------ fits and the moment maps
def fit_axis_series(Z, F, order: int):
    """Least squares F(Z) = sum_{r=0}^{order} c_r sgn(Z)^r / |Z|^(r+1); returns c_0..c_order."""
    Z = np.asarray(Z, dtype=float)
    F = np.asarray(F, dtype=float)
    z0 = float(np.min(np.abs(Z)))
    A = np.stack([np.sign(Z) ** r * (z0 / np.abs(Z)) ** (r + 1) for r in range(order + 1)], axis=1)
    sol, *_ = np.linalg.lstsq(A, F, rcond=None)
    c = sol * np.array([z0 ** (r + 1) for r in range(order + 1)])
    resid = float(np.sqrt(np.mean((A @ sol - F) ** 2)))
    return c, resid


def fhps_moments(U):
    """FHPS / Guerlebeck: M0..M3 from the Weyl axis coefficients U^(0..3) (N1 normalisation)."""
    U0, U1, U2, U3 = (list(U) + [0.0] * 4)[:4]
    return [-U0, -U1, U0 ** 3 / 3.0 - U2, U0 ** 2 * U1 - U3]


def ernst_moments(m):
    """Sotiriou-Apostolatos (static, no EM): P0..P5 from the Ernst axis coefficients m_0..m_5."""
    m = list(m) + [0.0] * 6
    m0, m1, m2, m3, m4, m5 = m[:6]
    P4 = m4 - m0 * (m0 * m2 - m1 ** 2) / 7.0
    P5 = m5 - m0 * (m0 * m3 - m1 * m2) / 3.0 - m1 * (m0 * m2 - m1 ** 2) / 21.0
    return [m0, m1, m2, m3, P4, P5]


def moments_from_profile(prof, order: int = 8, centre: bool = True, iters: int = 6):
    """The moments from an `axis_profile`, optionally mass-centred (Z shifted until M1 = 0)."""
    Z = np.concatenate([prof["north"]["Z"], prof["south"]["Z"]])
    U = np.concatenate([prof["north"]["U"], prof["south"]["U"]])
    shift = 0.0
    for _ in range(iters if centre else 1):
        Zc = Z - shift
        cU, rU = fit_axis_series(Zc, U, order)
        M = fhps_moments(cU)
        if not centre or abs(M[1]) < 1e-15 * max(1.0, abs(M[0])):
            break
        shift += M[1] / M[0]
    cX, rX = fit_axis_series(Z - shift, -np.tanh(U), order)
    return {"shift": shift, "U_coeffs": cU.tolist(), "fit_rms_U": rU, "M_fhps": M,
            "xi_coeffs": cX.tolist(), "fit_rms_xi": rX, "M_ernst": ernst_moments(cX)}


def gh_moments(fields, axis=(0.0, 0.0, 1.0), rho_lo: float = 2.0, rho_hi: float = 100.0,
               order: int = 8, centre: bool = True, n: int = 4000):
    """Mass-centred GH moments (two routes) plus the provenance: gap, shift, coefficients, fit rms."""
    prof = axis_profile(fields, axis, rho_lo, rho_hi, n=n)
    out = moments_from_profile(prof, order=order, centre=centre)
    out.update({"axis": prof["axis"], "gap": prof["gap"], "rho_lo": rho_lo, "rho_hi": rho_hi,
                "order": order})
    return out


def scan(fields, axis=(0.0, 0.0, 1.0), rho_los=(2.0, 3.0, 5.0), orders=(6, 8, 10),
         rho_hi: float = 100.0, n: int = 4000):
    """The moments over a grid of fit windows and orders: their spread is the error bar."""
    rows = []
    for lo in rho_los:
        prof = axis_profile(fields, axis, lo, rho_hi, n=n)
        for o in orders:
            r = moments_from_profile(prof, order=o)
            r.update({"rho_lo": lo, "order": o, "gap": prof["gap"]})
            rows.append(r)
    M = np.array([r["M_ernst"] for r in rows])
    return {"rows": rows, "M_ernst_median": np.median(M, axis=0).tolist(),
            "M_ernst_spread": (M.max(axis=0) - M.min(axis=0)).tolist()}


def rods_moments(spans, lmax: int = 3):
    """Closed form for Weyl rods (exact asset): U^(r) = -sum_i (b_i^(r+1) - a_i^(r+1)) / (2(r+1)),
    then FHPS; mass-centred by shifting the rods."""
    spans = [(float(a), float(b)) for a, b in spans]
    M0 = sum(0.5 * (b - a) for a, b in spans)
    zc = sum(0.25 * (b * b - a * a) for a, b in spans) / M0          # M1/M0 before centring
    U = [-sum((b - zc) ** (r + 1) - (a - zc) ** (r + 1) for a, b in spans) / (2.0 * (r + 1))
         for r in range(lmax + 1)]
    return fhps_moments(U), zc
