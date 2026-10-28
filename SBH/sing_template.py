"""Singular template for sing_pinn.py: the spherical single-puncture solution as a smooth function. (2026-10-28)

The 1D reference (reference_1d/sph_reference.py) stores one puncture of mass m as 10 radial functions
    gt = a delta + b nn,  At = c delta + d nn,  chi, Khat, Theta,  Ghat = g n,  alpha,  beta = B n
on the staggered grid r_k = (k + 1/2) dr at snapshot times t_j. This module turns such a table into a
C2-in-r, C2-in-t function (tensor-product cubic spline; the r grid is mirrored through r = 0 with each
function's parity first, so the spline is not one-sided at the inner edge) and lifts it to the 22 CCZ4
fields at a Cartesian offset xi = x - a from the puncture. Everything is plain jnp, so autodiff gives
the template's d_t, d_xyz and d_xyz d_xyz.

Two kinds:
  'bl'       the time-dependent table T(r, t) of an evolution from Brill-Lindquist data (alpha = 1/psi):
             at t = 0 it is the BL puncture, and it follows the collapse to the trumpet;
  'trumpet'  one late snapshot T(r, t_s) of the same evolution, frozen in time (the trumpet the
             1+log / integrated Gamma-driver gauge settles to; t_s = 10 = 20 M for m = 0.5).
"""
import numpy as np
from scipy.interpolate import CubicSpline

FIELDS = ("a", "b", "c", "d", "chi", "Khat", "Theta", "g", "alpha", "B")
PARITY = np.array([1, 1, 1, 1, 1, 1, 1, -1, 1, -1], dtype=float)
FLAT = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=float)     # flat space in radial functions
I_CHI, I_ALPHA = 4, 8


def _mirror(r, U, n_mirror):
    """Extend (.., 10, n_r) through r = 0 by parity: r_-k = -r_k."""
    rm = -r[:n_mirror][::-1]
    Um = (U[..., :n_mirror] * PARITY[:, None])[..., ::-1]
    return np.concatenate([rm, r]), np.concatenate([Um, U], axis=-1)


def load_table(path, kind, r_max, t_snap=None, n_mirror=8):
    """Spline coefficients of the reference table in `path`.

    Returns a dict with the jnp-ready numpy arrays:
      kind 'bl':      C (4, n_t-1, 4, n_r-1, 10), t0, dt, r0, dr
      kind 'trumpet': C (4, n_r-1, 10), r0, dr, and t_s
    plus the run parameters (mass, eta, ...) stored in the npz."""
    D = np.load(path)
    r, t, U = D["r"], D["t"], D["U"]                       # U: (n_t, 10, n_r)
    keep = r <= r_max + 4 * (r[1] - r[0])
    r, U = r[keep], U[..., keep]
    meta = {k: float(D[k]) for k in ("mass", "eta", "f_alpha", "f_beta", "kappa_z", "kappa_c", "kappa_2",
                                     "chi_floor") if k in D.files}
    if kind == "trumpet":
        j = int(np.argmin(np.abs(t - t_snap)))
        re, Ue = _mirror(r, U[j], n_mirror)                # (10, n)
        sp = CubicSpline(re, Ue, axis=1)
        C = sp.c                                           # (4, n-1, 10)
        return dict(kind=kind, C=C, r0=float(re[0]), dr=float(re[1] - re[0]), n_r=len(re), t_s=float(t[j]), **meta)
    re, Ue = _mirror(r, U, n_mirror)                       # (n_t, 10, n)
    cr = CubicSpline(re, Ue, axis=2).c                     # (4, n_r-1, n_t, 10)
    ct = CubicSpline(t, cr, axis=2).c                      # (4, n_t-1, 4, n_r-1, 10)
    dt = np.diff(t)
    assert np.allclose(dt, dt[0], rtol=1e-6), "template times must be uniform"
    return dict(kind=kind, C=ct, t0=float(t[0]), dt=float(dt[0]), n_t=len(t), r0=float(re[0]),
                dr=float(re[1] - re[0]), n_r=len(re), t_end=float(t[-1]), **meta)


def make_radial(jnp, jax, tab):
    """radial(t, r) -> (10,) template radial functions at one point (scalars in, jnp)."""
    C = jnp.asarray(tab["C"])
    r0, dr, n_r = tab["r0"], tab["dr"], tab["n_r"]

    def horner(c, h):                                      # scipy's c[k] multiplies h^(3-k); no 0^0 in autodiff
        return ((c[0] * h + c[1]) * h + c[2]) * h + c[3]

    def r_cell(r):
        k = jax.lax.stop_gradient(jnp.clip(jnp.floor((r - r0) / dr), 0, n_r - 2)).astype(int)
        return k, r - (r0 + k * dr)

    if tab["kind"] == "trumpet":
        def radial(t, r):
            k, h = r_cell(r)
            return horner(C[:, k, :], h)
        return radial

    t0, dt, n_t = tab["t0"], tab["dt"], tab["n_t"]

    def radial(t, r):
        k, h = r_cell(r)
        j = jax.lax.stop_gradient(jnp.clip(jnp.floor((t - t0) / dt), 0, n_t - 2)).astype(int)
        s = t - (t0 + j * dt)
        return horner(horner(C[:, j, :, k, :], s), h)
    return radial


def lift(jnp, pack, rad, xi):
    """(10,) radial functions at offset xi (3,), r = |xi| > 0 -> the 22 CCZ4 fields."""
    a, b, c, d, chi, Kh, Th, g, al, B = rad
    r = jnp.sqrt(jnp.sum(xi ** 2))
    n = xi / r
    nn = jnp.outer(n, n)
    I = jnp.eye(3)
    return jnp.concatenate([pack((a * I + b * nn)[None])[0], pack((c * I + d * nn)[None])[0],
                            jnp.stack([chi, Kh, Th]), g * n, al[None], B * n])
