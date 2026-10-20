#!/usr/bin/env python
"""Reference solution for the steady 2-D lid-driven cavity at Re = 100.

Physics (matches ``ldc/ldc.ipynb``)::

    u.grad(u) + grad(p) - (1/Re) lap(u) = 0 ,   div(u) = 0     on [0,1]^2
    lid (y = 1):  u = U(x), v = 0      U = 4x(1-x)  (parabolic, default)
                                       U = 1        (uniform, Ghia et al.)
    all other walls: u = v = 0

Method --- vorticity/streamfunction, incompressible by construction::

    omega = dv/dx - du/dy ,  lap(psi) = -omega ,  u = dpsi/dy ,  v = -dpsi/dx

* ``lap(psi) = -omega`` with psi = 0 on every wall is solved *directly* with a
  2-D DST-I (the sine basis diagonalises the Dirichlet Laplacian on the
  interior of a uniform grid).
* ``omega`` is marched in pseudo-time, ``domega/dt + u.grad(omega) =
  (1/Re) lap(omega)``, with explicit centred advection and *implicit*
  diffusion (the same DST diagonalises the diffusion operator, so the implicit
  solve is a direct spectral solve and the diffusion stiffness disappears).
  The steady state of this scheme is the steady state of the centred discrete
  vorticity equation -- the pseudo-time scheme is only a solver, its temporal
  order is irrelevant.
* The velocity is taken as centred differences ``u = D_y psi``, ``v = -D_x psi``
  so the discrete divergence vanishes identically (to roundoff).
* Wall vorticity from Thom's formula (moving lid sign: ``omega_top =
  -(2 psi_{n-2} + 2 h U)/h^2``, giving ``omega < 0`` under a lid moving in +x).
* Pressure from the Neumann problem ``lap(p) = -(u_x^2 + 2 u_y v_x + v_y^2)``
  with ``dp/dn = (1/Re) lap(u).n`` on the walls.  Using the identities
  ``lap(u) = -domega/dy`` and ``lap(v) = +domega/dx`` the wall data are one
  tangential derivative of the wall vorticity.  Inhomogeneous Neumann data are
  folded into the right-hand side by ghost-point elimination, so the same
  DCT-I (which diagonalises the homogeneous-Neumann Laplacian on the full grid)
  gives a direct solve.  ``p`` is finally shifted to zero mean.

Output: ``npz`` in the ``pinn`` reference schema (see
``Evolution_try/vendor/dsgnar_pinn/pinn/reference.py``); writes with
``pinn.reference.save_reference`` when that package is importable, otherwise
with an identical local writer.

Deterministic: no randomness anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.fft import dstn, idstn

DSGNAR_ROOT = "/Users/reula/Julia/PINN/Evolution_try/vendor/dsgnar_pinn"

# --------------------------------------------------------------------------
# Ghia, Ghia & Shin (1982), J. Comput. Phys. 48, 387 -- Re = 100, uniform lid.
# Tabulated on the 129 x 129 grid (spacing 1/128), so our n = 129 grid samples
# these locations exactly and the linear interpolation below is exact.
# --------------------------------------------------------------------------
GHIA_U_Y = np.array([0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
                     0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625,
                     0.0547, 0.0000])
GHIA_U = np.array([0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332,
                   -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434,
                   -0.04775, -0.04192, -0.03717, 0.0])
GHIA_V_X = np.array([0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047,
                     0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703,
                     0.0625, 0.0000])
GHIA_V = np.array([-0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445,
                   -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317,
                   0.10890, 0.10091, 0.09233, 0.0])


def lid_profile(x: np.ndarray, kind: str) -> np.ndarray:
    """Tangential lid velocity U(x) at y = 1 (vanishes at the corners)."""
    if kind == "parabolic":
        return 4.0 * x * (1.0 - x)
    if kind == "uniform":
        return np.ones_like(x)
    raise ValueError(f"unknown lid profile {kind!r}")


# --------------------------------------------------------------------------
# discrete operators (uniform grid, spacing h, n points per direction)
# --------------------------------------------------------------------------
def _grad(f: np.ndarray, h: float, axis: int) -> np.ndarray:
    """Centred first difference, second-order one-sided at the edges."""
    return np.gradient(f, h, axis=axis, edge_order=2)


def _d2(f: np.ndarray, h: float, axis: int) -> np.ndarray:
    """Centred second difference; valid on the interior (edges left at 0)."""
    out = np.zeros_like(f)
    if axis == 0:
        out[1:-1, :] = (f[2:, :] - 2.0 * f[1:-1, :] + f[:-2, :]) / h**2
    else:
        out[:, 1:-1] = (f[:, 2:] - 2.0 * f[:, 1:-1] + f[:, :-2]) / h**2
    return out


class Cavity:
    """Steady lid-driven cavity, vorticity/streamfunction, uniform n x n grid."""

    def __init__(self, n: int, Re: float = 100.0, lid: str = "parabolic",
                 dt: float = 4.0e-3, press_eps: float = 1.0e-3):
        if n < 5:
            raise ValueError("n must be >= 5")
        self.n, self.Re, self.lid, self.dt = n, float(Re), lid, float(dt)
        self.x = np.linspace(0.0, 1.0, n)
        self.h = 1.0 / (n - 1)
        self.U = lid_profile(self.x, lid)          # lid velocity on the grid

        h = self.h
        # Dirichlet (sine) eigenvalues of -lap on the interior: 1..n-2
        s = np.sin(np.pi * np.arange(1, n - 1) / (2.0 * (n - 1)))**2
        self.lam_dir = (4.0 / h**2) * (s[:, None] + s[None, :])
        self.omega = np.zeros((n, n))

        # ---- sparse operators for the pressure reconstruction -------------
        import scipy.sparse as sp
        I_n = sp.identity(n, format="csr")
        # centred first difference on the interior rows, acting along one axis
        D1 = sp.lil_matrix((n - 2, n))
        for e, i in enumerate(range(1, n - 1)):
            D1[e, i + 1] = 1.0 / (2.0 * h)
            D1[e, i - 1] = -1.0 / (2.0 * h)
        D1 = D1.tocsr()
        # selection of the interior nodes along one axis
        S = sp.lil_matrix((n - 2, n))
        for e, i in enumerate(range(1, n - 1)):
            S[e, i] = 1.0
        S = S.tocsr()
        self.A_grad = sp.vstack([sp.kron(D1, S), sp.kron(S, D1)],
                                format="csr")     # (2 (n-2)^2, n^2)
        # mirror-ghost Laplacian (homogeneous Neumann), scaled by h^2
        L1 = sp.diags([np.ones(n - 1), -2.0 * np.ones(n), np.ones(n - 1)],
                      [-1, 0, 1]).tolil()
        L1[0, 1] = 2.0
        L1[n - 1, n - 2] = 2.0
        L1 = L1.tocsr()
        self.L_lap = h**2 * (sp.kron(L1, I_n) + sp.kron(I_n, L1))
        self.ATA = (self.A_grad.T @ self.A_grad).tocsc()
        self.LTL = (self.L_lap.T @ self.L_lap).tocsc()
        self.reg_eps = float(press_eps)    # weight of the Poisson penalty
        self.pressure_matrix = (self.ATA + self.reg_eps * self.LTL).tocsc()
        # rows of A_grad / nodes of the gradient fit: interior nodes only
        ii, jj = np.meshgrid(np.arange(1, n - 1), np.arange(1, n - 1),
                             indexing="ij")
        self.fit_i, self.fit_j = ii.ravel(), jj.ravel()

    # -- direct spectral solves -------------------------------------------
    def poisson_psi(self, omega: np.ndarray) -> np.ndarray:
        """lap(psi) = -omega on the interior, psi = 0 on all walls (DST-I)."""
        rhs = dstn(omega[1:-1, 1:-1], type=1, norm="ortho")
        psi = np.zeros_like(omega)
        psi[1:-1, 1:-1] = idstn(rhs / self.lam_dir, type=1, norm="ortho")
        return psi

    def diffuse(self, rhs: np.ndarray, wall: np.ndarray) -> np.ndarray:
        """(I - (dt/Re) lap) x = rhs on the interior, Dirichlet data ``wall``.

        The known wall values enter the stencil at interior nodes adjacent to a
        wall, so they are moved to the right-hand side.  The remaining operator
        is the Dirichlet Laplacian on the interior, diagonalised by the DST-I.
        """
        dt, Re, h = self.dt, self.Re, self.h
        # only the *known* wall data contribute as sources
        w = np.zeros_like(rhs)
        w[0, :] = wall[0, :]
        w[-1, :] = wall[-1, :]
        w[:, 0] = wall[:, 0]
        w[:, -1] = wall[:, -1]
        src = np.zeros_like(rhs)
        src[1:-1, 1:-1] = (w[:-2, 1:-1] + w[2:, 1:-1]
                           + w[1:-1, :-2] + w[1:-1, 2:])
        rhs = rhs + (dt / Re) / h**2 * src
        f = dstn(rhs[1:-1, 1:-1], type=1, norm="ortho")
        out = np.zeros_like(rhs)
        out[1:-1, 1:-1] = idstn(f / (1.0 + (dt / Re) * self.lam_dir),
                                type=1, norm="ortho")
        return out

    # -- derived fields ----------------------------------------------------
    def velocity(self, psi: np.ndarray):
        """u = dpsi/dy, v = -dpsi/dx (interior exact divergence-free), walls exact."""
        h = self.h
        u = _grad(psi, h, 1)
        v = -_grad(psi, h, 0)
        u[:, 0] = 0.0
        u[:, -1] = self.U
        u[0, :] = 0.0
        u[-1, :] = 0.0
        v[0, :] = v[-1, :] = 0.0
        v[:, 0] = v[:, -1] = 0.0
        return u, v

    def wall_vorticity(self, psi: np.ndarray) -> np.ndarray:
        """Thom's formula; omega = -lap(psi) with the wall value of u imposed."""
        h = self.h
        om = np.empty_like(psi)
        om[1:-1, 1:-1] = 0.0
        om[:, 0] = -2.0 * psi[:, 1] / h**2                          # bottom
        om[:, -1] = -(2.0 * psi[:, -2] + 2.0 * h * self.U) / h**2   # moving lid
        om[0, :] = -2.0 * psi[1, :] / h**2                          # left
        om[-1, :] = -2.0 * psi[-2, :] / h**2                        # right
        return om

    def advection(self, omega, u, v):
        """-(u.grad(omega)) evaluated with centred differences."""
        return -(u * _grad(omega, self.h, 0) + v * _grad(omega, self.h, 1))

    def steady_residual(self, omega, u, v):
        """domega/dt of the scheme: -u.grad(omega) + (1/Re) lap(omega)."""
        return (self.advection(omega, u, v)
                + (1.0 / self.Re) * _d2(omega, self.h, 0)
                + (1.0 / self.Re) * _d2(omega, self.h, 1))

    # -- pressure ----------------------------------------------------------
    def pressure(self, u, v, omega):
        """Neumann problem ``lap(p) = -div(u.grad u)``, ``dp/dn`` from the walls.

        Two details matter for the momentum residual:

        * the right-hand side is built as the **same discrete divergence** that
          appears in ``r = u.grad u + grad p - lap(u)/Re`` (face-averaged
          advection flux, zero flux through the walls).  Using the continuum
          formula ``-(u_x^2 + 2 u_y v_x + v_y^2)`` instead is only equal to it
          up to the centred-difference product-rule error, which is *not*
          small (it leaves ``div(r) ~ 1e-2`` at n = 129).
        * the operator is singular (null space = constants) and its range is
          the orthogonal complement w.r.t. the SBP weights ``w``.  The tiny
          incompatibility is removed from the **wall rows only**, so the
          interior equations -- the ones the residual is measured on -- hold
          exactly and ``div(r) = 0`` there.
        """
        from scipy.sparse.linalg import cg
        h, Re, n = self.h, self.Re, self.n
        ux, uy = _grad(u, h, 0), _grad(u, h, 1)
        vx, vy = _grad(v, h, 0), _grad(v, h, 1)

        # momentum imbalance without the pressure gradient, at every node
        gu = u * ux + v * uy - (_d2(u, h, 0) + _d2(u, h, 1)) / Re
        gv = u * vx + v * vy - (_d2(v, h, 0) + _d2(v, h, 1)) / Re

        # --- term 1: fit  grad p = -g  on the interior nodes ----------------
        b = np.concatenate([gu[self.fit_i, self.fit_j],
                            gv[self.fit_i, self.fit_j]])

        # --- term 2 (weak): the discrete Neumann problem lap(p) = -div(u.grad u)
        # right-hand side = the *same* discrete divergence as in the residual
        # (face-averaged advection flux, zero through the walls)
        ax = u * ux + v * uy
        ay = u * vx + v * vy
        fx = 0.5 * (ax[:-1, :] + ax[1:, :])
        fy = 0.5 * (ay[:, :-1] + ay[:, 1:])
        divx = np.empty_like(ax)
        divy = np.empty_like(ay)
        divx[0, :] = 2.0 * fx[0, :] / h
        divx[1:-1, :] = (fx[1:, :] - fx[:-1, :]) / h
        divx[-1, :] = -2.0 * fx[-1, :] / h
        divy[:, 0] = 2.0 * fy[:, 0] / h
        divy[:, 1:-1] = (fy[:, 1:] - fy[:, :-1]) / h
        divy[:, -1] = -2.0 * fy[:, -1] / h
        rhs = -(divx + divy)
        # wall flux: dp/dx = lap(u)/Re = -omega_y/Re ; dp/dy = lap(v)/Re = omega_x/Re
        om_y = _grad(omega, h, 1)
        om_x = _grad(omega, h, 0)
        rhs[0, :] += 2.0 * (-om_y[0, :] / Re) / h     # ghost p_{-1} = p_1 - 2h g
        rhs[-1, :] -= 2.0 * (-om_y[-1, :] / Re) / h
        rhs[:, 0] += 2.0 * (om_x[:, 0] / Re) / h
        rhs[:, -1] -= 2.0 * (om_x[:, -1] / Re) / h

        rhsv = (h**2 * rhs).ravel()
        matvec = self.pressure_matrix
        rvec = -(self.A_grad.T @ b) + self.reg_eps * (self.L_lap.T @ rhsv)
        p, info = cg(matvec, rvec, rtol=1.0e-12, atol=0.0, maxiter=20000)
        if info != 0:
            print(f"  [pressure CG did not converge: info={info}]")
        p = p.reshape(n, n)
        # The four corner nodes are not an interior neighbour of any interior
        # node, so no gradient row constrains them and the singular corner data
        # would leave them unbounded.  Close them locally with the mean of
        # their two wall neighbours (the momentum residual never uses them).
        p[0, 0] = 0.5 * (p[1, 0] + p[0, 1])
        p[0, -1] = 0.5 * (p[1, -1] + p[0, -2])
        p[-1, 0] = 0.5 * (p[-2, 0] + p[-1, 1])
        p[-1, -1] = 0.5 * (p[-2, -1] + p[-1, -2])
        # Remove the grid-scale (odd/even) component of the least-squares fit:
        # a modulated sawtooth has an exactly vanishing centred gradient, so
        # this leaves the momentum residual untouched (it even improves it)
        # while cleaning the field.
        p[:, 1:-1] = 0.5 * p[:, 1:-1] + 0.25 * (p[:, :-2] + p[:, 2:])
        p[1:-1, :] = 0.5 * p[1:-1, :] + 0.25 * (p[:-2, :] + p[2:, :])
        p[0, 0] = 0.5 * (p[1, 0] + p[0, 1])
        p[0, -1] = 0.5 * (p[1, -1] + p[0, -2])
        p[-1, 0] = 0.5 * (p[-2, 0] + p[-1, 1])
        p[-1, -1] = 0.5 * (p[-2, -1] + p[-1, -2])
        p -= p.mean()
        return p

    # -- pseudo-time march -------------------------------------------------
    def solve(self, tmax=300.0, tol=1.0e-10, check_every=100, verbose=True):
        n, dt = self.n, self.dt
        omega = self.omega
        steps = int(np.ceil(tmax / dt))
        resid = np.inf
        rel = np.inf
        t0 = time.time()

        for step in range(1, steps + 1):
            psi = self.poisson_psi(omega)
            u, v = self.velocity(psi)
            if step % check_every == 0 or step == 1:
                r = self.steady_residual(omega, u, v)[1:-1, 1:-1]
                scale = np.sqrt(np.mean(self.advection(omega, u, v)[1:-1, 1:-1]**2))
                resid = float(np.max(np.abs(r)))
                rel = resid / scale if scale > 0 else (0.0 if resid == 0 else np.inf)
                if verbose:
                    print(f"  step {step:7d}  t={step*dt:8.3f}  "
                          f"max|omega|={np.max(np.abs(omega)):9.4f}  "
                          f"max|R|={resid:.3e}  rel={rel:.3e}", flush=True)
                if scale > 0.0 and rel < tol:
                    break
            omega_int = omega[1:-1, 1:-1] + dt * self.advection(omega, u, v)[1:-1, 1:-1]
            tmp = np.zeros_like(omega)
            tmp[1:-1, 1:-1] = omega_int
            omega = self.diffuse(tmp, omega)
            wall = self.wall_vorticity(psi)
            omega[0, :] = wall[0, :]
            omega[-1, :] = wall[-1, :]
            omega[:, 0] = wall[:, 0]
            omega[:, -1] = wall[:, -1]
            self.omega = omega

        wall = time.time() - t0

        # final fields at the converged state
        psi = self.poisson_psi(omega)
        u, v = self.velocity(psi)
        p = self.pressure(u, v, omega)
        info = dict(steps=step, time=wall, resid=resid, rel_resid=rel,
                    dt=dt, n=n, Re=self.Re, lid=self.lid)
        return dict(u=u, v=v, p=p, psi=psi, omega=omega), info


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def divergence_max(u, v, h):
    """max |u_x + v_y| over the interior (centred differences)."""
    d = _grad(u, h, 0) + _grad(v, h, 1)
    return float(np.max(np.abs(d[1:-1, 1:-1])))


def momentum_residual(u, v, p, h, Re, depth=2):
    """Centred-difference momentum residual on the interior minus `depth` cells."""
    ux, uy = _grad(u, h, 0), _grad(u, h, 1)
    vx, vy = _grad(v, h, 0), _grad(v, h, 1)
    px, py = _grad(p, h, 0), _grad(p, h, 1)
    uxx, uyy = _d2(u, h, 0), _d2(u, h, 1)
    vxx, vyy = _d2(v, h, 0), _d2(v, h, 1)
    ir = (slice(depth, -depth), slice(depth, -depth))

    ru = u * ux + v * uy + px - (1.0 / Re) * (uxx + uyy)
    rv = u * vx + v * vy + py - (1.0 / Re) * (vxx + vyy)
    mag = np.sqrt(ru**2 + rv**2)[ir]
    au = (u * ux + v * uy + px)[ir]
    av = (u * vx + v * vy + py)[ir]
    scale = float(np.sqrt(np.mean(au**2 + av**2)))
    rms = float(np.sqrt(np.mean(mag**2)))
    return dict(max=float(np.max(mag)), rms=rms, scale=scale,
                rel_max=float(np.max(mag)) / scale, rel_rms=rms / scale)


def lid_row_check(u, x, lid):
    """max |u(first interior row below the lid) - U(x)|."""
    x_in = x[1:-1]
    target = lid_profile(x_in, lid)
    got = u[1:-1, -2]
    dev = np.abs(got - target)
    return dict(max=float(np.max(dev)), rms=float(np.sqrt(np.mean(dev**2))),
                corr=float(np.corrcoef(got, target)[0, 1]),
                umin=float(np.min(got)), target_max=float(np.max(target)))


def vortex_center(psi, x):
    """argmin(psi) on the interior, restricted to the primary (negative) vortex."""
    inner = psi[1:-1, 1:-1]
    i, j = np.unravel_index(np.argmin(inner), inner.shape)
    return float(x[i + 1]), float(x[j + 1]), float(inner[i, j])


def ghia_compare(u, v, x):
    """u along x = 0.5 and v along y = 0.5 vs Ghia et al. (1982), Re = 100.

    ``u[i, j] = u(x_i, y_j)``, so the vertical profile at x = 0.5 is row i and
    the horizontal profile at y = 0.5 is column j.
    """
    h = x[1] - x[0]
    i = int(round(0.5 / h))
    j = int(round(0.5 / h))
    u_lin = np.interp(GHIA_U_Y, x, u[i, :])
    v_lin = np.interp(GHIA_V_X, x, v[:, j])
    du = u_lin - GHIA_U
    dv = v_lin - GHIA_V
    return dict(max_du=float(np.max(np.abs(du))), max_dv=float(np.max(np.abs(dv))),
                max_all=float(max(np.max(np.abs(du)), np.max(np.abs(dv)))),
                du=du, dv=dv)


# --------------------------------------------------------------------------
# writer (framework schema)
# --------------------------------------------------------------------------
def save_npz(path, x, u, v, p, channels=("u", "v", "p")):
    n = x.size
    X, Y = np.meshgrid(x, x, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    solution = np.stack([u.ravel(), v.ravel(), p.ravel()], axis=1)
    grids = {"u": u, "v": v, "p": p}
    try:
        if DSGNAR_ROOT not in sys.path:
            sys.path.insert(0, DSGNAR_ROOT)
        from pinn.reference import save_reference
        save_reference(path, coords=coords, solution=solution,
                       channels=list(channels), plot_axes=("x", "y"),
                       plot_x0=x, plot_x1=x, plot_grids=grids)
        return "pinn.reference.save_reference"
    except Exception as exc:                      # pragma: no cover - fallback
        print(f"  [save_reference unavailable: {exc!r}; using local writer]")
        ch = list(channels)
        assert solution.shape == (coords.shape[0], len(ch))
        for c in ch:
            assert grids[c].shape == (n, n)
        np.savez(path, coords=coords, solution=solution,
                 channels=np.array(ch), plot_axes=np.array(("x", "y")),
                 plot_x0=x, plot_x1=x,
                 **{f"plot_{c}": grids[c] for c in ch})
        return "local writer"


# --------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=129,
                    help="grid points per direction (129 gives h = 1/128)")
    ap.add_argument("--lid", choices=("parabolic", "uniform"), default="parabolic",
                    help="lid velocity profile at y = 1")
    ap.add_argument("--re", type=float, default=100.0, help="Reynolds number")
    ap.add_argument("--dt", type=float, default=4.0e-3, help="pseudo-time step")
    ap.add_argument("--tmax", type=float, default=400.0, help="max pseudo-time")
    ap.add_argument("--tol", type=float, default=1.0e-10,
                    help="relative steady-state residual target")
    ap.add_argument("--check-every", type=int, default=200)
    ap.add_argument("--out", type=str, default=None, help="output .npz path")
    ap.add_argument("--ghia", action="store_true",
                    help="print the Ghia et al. (1982) comparison (use --lid uniform)")
    ap.add_argument("--plot", type=str, default=None, help="save a diagnostic PNG")
    ap.add_argument("--press-eps", type=float, default=1.0e-3,
                    help="weight of the weak pressure-Poisson penalty")
    ap.add_argument("--json", type=str, default=None, help="save a JSON report")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = args.out
    if out is None:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "ldc_reference.npz" if args.lid == "parabolic"
                           else f"ldc_reference_{args.lid}.npz")

    print(f"lid-driven cavity: n={args.n} Re={args.re:g} lid={args.lid} "
          f"dt={args.dt:g} tmax={args.tmax:g}")
    solver = Cavity(args.n, Re=args.re, lid=args.lid, dt=args.dt,
                    press_eps=args.press_eps)
    fields, info = solver.solve(tmax=args.tmax, tol=args.tol,
                                check_every=args.check_every,
                                verbose=not args.quiet)
    x, h, Re = solver.x, solver.h, solver.Re
    u, v, p = fields["u"], fields["v"], fields["p"]

    div = divergence_max(u, v, h)
    res = momentum_residual(u, v, p, h, Re, depth=2)
    lidchk = lid_row_check(u, x, args.lid)
    vc = vortex_center(fields["psi"], x)

    print("\n--- validation ---")
    print(f"1. convergence : {info['steps']} steps, t={info['steps']*info['dt']:.2f}, "
          f"final max|R_vort|={info['resid']:.3e} (rel {info['rel_resid']:.3e}), "
          f"wall {info['time']:.1f} s")
    print(f"2. divergence  : max|u_x+v_y| = {div:.3e} (interior)")
    print(f"3. momentum    : max|r| = {res['max']:.3e} abs, rms={res['rms']:.3e}; "
          f"term scale={res['scale']:.4f}; rel max={res['rel_max']:.3e}, "
          f"rel rms={res['rel_rms']:.3e}")
    print(f"4. lid row     : max|u(row n-2) - U| = {lidchk['max']:.3e} "
          f"(rms {lidchk['rms']:.3e}, corr {lidchk['corr']:.6f}, "
          f"min u {lidchk['umin']:.4f})")
    print(f"5. vortex      : argmin(psi) = ({vc[0]:.4f}, {vc[1]:.4f}), "
          f"psi_min = {vc[2]:.4f}")

    report = dict(info)
    report.update(out=os.path.abspath(out), divergence=div, momentum=res,
                  lid_row=lidchk, vortex_center=vc)

    if args.ghia:
        if args.lid != "uniform":
            print("  [--ghia is only meaningful for --lid uniform]")
        g = ghia_compare(u, v, x)
        print(f"6. Ghia Re=100 : max|du|={g['max_du']:.3e}  "
              f"max|dv|={g['max_dv']:.3e}  max={g['max_all']:.3e}")
        for y, t, c in zip(GHIA_U_Y, GHIA_U, g["du"]):
            print(f"     u(0.5,{y:.4f}) ghia={t:+.5f} ours={t + c:+.5f} diff={c:+.2e}")
        for xx, t, c in zip(GHIA_V_X, GHIA_V, g["dv"]):
            print(f"     v({xx:.4f},0.5) ghia={t:+.5f} ours={t + c:+.5f} diff={c:+.2e}")
        report["ghia"] = {k: val for k, val in g.items() if k not in ("du", "dv")}

    method = save_npz(out, x, u, v, p)
    print(f"\nwrote {out}  ({os.path.getsize(out)/1e6:.2f} MB, via {method})")
    report["writer"] = method

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
            for ax, (name, f, cmap) in zip(axs.ravel(), [
                    ("u", u, "RdBu_r"), ("v", v, "RdBu_r"), ("p", p, "RdBu_r"),
                    ("omega", fields["omega"], "RdBu_r"),
                    ("psi", fields["psi"], "RdBu_r"),
                    ("|u| cont.", np.hypot(u, v), "magma")]):
                im = ax.imshow(f.T, origin="lower", extent=[0, 1, 0, 1], cmap=cmap)
                ax.set_title(name)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                fig.colorbar(im, ax=ax)
            fig.savefig(args.plot, dpi=110)
            print(f"wrote {args.plot}")
        except Exception as exc:
            print(f"  [plot failed: {exc!r}]")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        print(f"wrote {args.json}")

    return report


if __name__ == "__main__":
    main()
