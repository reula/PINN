"""Spherically symmetric reference evolution for a single puncture.

Validation only: the output is a reference to compare the PINN against
(compare_pinn.py). It is never used for training.

What it evolves
---------------
The same vacuum CCZ4 system, gauge and initial data as EEccz4_BH_puncture.py
(1+log slicing, integrated Gamma-driver, Brill-Lindquist puncture with
alpha = 1/psi, beta = 0), for one non-boosted, non-spinning puncture at the
origin. The right-hand side is NOT re-derived in spherical coordinates. It is
the BH code's own r_ccz4_evolution (verified against an independent CCZ4
implementation by tests/test_random.py), evaluated on the x-axis:

  - In spherical symmetry every field is fixed by 10 radial functions:
        gt_ij = a(r) delta_ij + b(r) n_i n_j      At_ij = c(r) delta_ij + d(r) n_i n_j
        chi, Khat, Theta, alpha (scalars)          Ghat^i = g(r) n^i,  beta^i = B(r) n^i
    with n = x/r.
  - At each grid point r_k the radial functions are replaced by their local
    quadratic Taylor model (value, first and second radial derivative from
    4th-order centred finite differences). The Cartesian fields built from
    that model are passed to r_ccz4_evolution as a stand-in "network" with no
    time dependence, so RHS = -residual. Autodiff of the model at (r_k, 0, 0)
    gives exactly the Cartesian first and second derivatives that spherical
    symmetry implies (the "cartoon" idea).
  - The radial RHS are read off on the axis: a_t = (gt_yy)_t,
    b_t = (gt_xx - gt_yy)_t, likewise c, d; g_t = (Ghat^x)_t, B_t = (beta^x)_t.

Numerics: staggered grid r_k = (k + 1/2) dr (the puncture sits between grid
points, standard for 1D moving punctures), parity ghost zones at r = 0
(scalars and tensor coefficients even, vector coefficients odd), 4th-order
centred differences (no upwinding), 6th-difference Kreiss-Oliger dissipation,
classical RK4. The outer boundary is far away (default R_out = 60) and
causally disconnected from r <= 24 for t <= 10 (fastest characteristic speed
sqrt(2)): the reference is the free-space solution, not the solution with the
PINN's Sommerfeld condition at r = 24. Ghost zones at R_out are filled by
4th-order extrapolation.

Divisions by chi use the BH code's max(chi, chi_floor) with the same default,
chi_floor = 1e-5, so the reference solves exactly the PINN's equations. The
floor only acts on the innermost grid points (chi < 1e-5 means r < 0.015 at
t = 0). With a much smaller floor (1e-12) the run fails within t ~ 0.05: the
dissipation at the innermost points pushes chi (~1e-7 there) below zero.

Accuracy (conv_check.py, dr = 0.02/0.01/0.005, t <= 10): 4th-order
convergence (Q ~ 16) for r >= 0.1, with the dr = 0.005 run accurate to
<~1e-5 there (most fields <~1e-6). Near the puncture (0.03 <= r < 0.1) the
convergence drops to ~1st-2nd order and the error of the dr = 0.005 run is
up to ~1e-3 (Khat, At around t = 2). alpha at r = 0.03 agrees between dr =
0.01 and 0.005 to <= 0.004 at all times.

Usage (CPU is enough):
    JAX_PLATFORMS=cpu python3 sph_reference.py --dr 0.005 --tfinal 10 --out ref_single_m0.5_dr0.005.npz
Output npz: r (grid), t (snapshot times), U (n_t, 10, n_r) in the order of
FIELDS, plus the run parameters.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))

FIELDS = ("a", "b", "c", "d", "chi", "Khat", "Theta", "g", "alpha", "B")
PARITY = np.array([1, 1, 1, 1, 1, 1, 1, -1, 1, -1], dtype=float)
NG = 3   # ghost zones (6th-difference dissipation needs 3)


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mass", type=float, default=0.5, help="bare mass m of the puncture (BH default per hole: 0.5)")
    p.add_argument("--dr", type=float, default=0.01)
    p.add_argument("--R_out", type=float, default=60.0)
    p.add_argument("--tfinal", type=float, default=10.0)
    p.add_argument("--cfl", type=float, default=0.25, help="dt = cfl * dr")
    p.add_argument("--ko_sigma", type=float, default=0.1, help="Kreiss-Oliger strength")
    p.add_argument("--snap_dt", type=float, default=0.25, help="snapshot interval (BH window width by default)")
    p.add_argument("--chi_floor", type=float, default=1e-5, help="same default as the BH code; see the module docstring")
    p.add_argument("--f_alpha", type=float, default=1.0)
    p.add_argument("--f_beta", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--kappa_z", type=float, default=0.0)
    p.add_argument("--kappa_c", type=float, default=0.0)
    p.add_argument("--kappa_2", type=float, default=0.0)
    p.add_argument("--r_save", type=float, default=30.0, help="store only r <= r_save (the PINN domain is r <= 24)")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def build(args):
    """Load the BH module with the gauge/damping flags of this run and return
    (rhs_fn, helpers). Everything physics-related comes from the BH code."""
    from load_module import load
    ns = load(["--chi_floor", repr(args.chi_floor), "--f_alpha", repr(args.f_alpha),
               "--f_beta", repr(args.f_beta), "--eta", repr(args.eta),
               "--kappa_z", repr(args.kappa_z), "--kappa_c", repr(args.kappa_c),
               "--kappa_2", repr(args.kappa_2),
               "--n_stars", "1", "--BL_masses", repr(args.mass),
               "--BL_momenta", "0", "--BL_momenta_y", "0", "--BL_spins", "0,0,0"])
    import jax
    import jax.numpy as jnp
    pack = ns["pack_sym3"]
    r_ccz4 = ns["r_ccz4_evolution"]

    def fields_from_radial(rad, xyz):
        """rad: (10,) radial-function values at r=|xyz|; xyz: (3,). Returns the
        22 Cartesian outputs in the BH network's order."""
        a, b, c, d, chi, Kh, Th, g, al, B = rad
        r = jnp.sqrt(jnp.sum(xyz**2))
        n = xyz / r
        nn = jnp.outer(n, n)
        I = jnp.eye(3)
        gt = a * I + b * nn
        At = c * I + d * nn
        return jnp.concatenate([pack(gt[None])[0], pack(At[None])[0],
                                jnp.stack([chi, Kh, Th]), g * n, al[None], B * n])

    def apply_fn(params, t, x, y, z, ws, we):
        coef, r_first, dr = params            # coef: (n_r, 3, 10) = value, d/dr, d2/dr2
        xyz = jnp.concatenate([x, y, z], axis=1)          # (1,3)

        def one(p):
            r = jnp.sqrt(jnp.sum(p**2))
            k = jax.lax.stop_gradient(jnp.clip(jnp.round((r - r_first) / dr), 0, coef.shape[0] - 1)).astype(int)
            rk = r_first + k * dr
            h = r - rk
            c0, c1, c2 = coef[k, 0], coef[k, 1], coef[k, 2]
            return fields_from_radial(c0 + c1 * h + 0.5 * c2 * h * h, p)
        return jax.vmap(one)(xyz)

    return ns, jax, jnp, apply_fn, r_ccz4


def make_solver(args):
    """Grid, initial data and the semi-discrete RHS (no time stepping)."""
    ns, jax, jnp, apply_fn, r_ccz4 = build(args)

    dr = args.dr
    n_r = int(round(args.R_out / dr))
    r = (np.arange(n_r) + 0.5) * dr
    m = args.mass

    # Brill-Lindquist, alpha = 1/psi, beta = 0, gt = delta, At = 0, Ghat = 0.
    psi = 1.0 + m / (2.0 * r)
    U0 = np.zeros((10, n_r))
    U0[0] = 1.0                      # a
    U0[4] = psi**-4                  # chi
    U0[8] = 1.0 / psi                # alpha

    par = jnp.asarray(PARITY)[:, None]
    # 4th-order centred stencils
    c1 = jnp.array([1.0, -8.0, 0.0, 8.0, -1.0]) / (12.0 * dr)
    c2 = jnp.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / (12.0 * dr * dr)
    ko = jnp.array([1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0]) * (args.ko_sigma / (64.0 * dr))

    def pad(U):
        left = (U[:, :NG] * par)[:, ::-1]                       # f(-r_k) = p f(r_k)
        # 4th-order extrapolation at the outer edge
        e = [U[:, -1]]
        tail = U[:, -4:]
        right = []
        for _ in range(NG):
            new = 4 * tail[:, -1] - 6 * tail[:, -2] + 4 * tail[:, -3] - tail[:, -4]
            right.append(new)
            tail = jnp.concatenate([tail[:, 1:], new[:, None]], axis=1)
        return jnp.concatenate([left, U, jnp.stack(right, axis=1)], axis=1)

    def stencil(P, w, half):
        n = P.shape[1] - 2 * NG
        return sum(w[j] * P[:, NG - half + j: NG - half + j + n] for j in range(2 * half + 1))

    r_j = jnp.asarray(r)
    x_pts = r_j[:, None]
    zeros = jnp.zeros_like(x_pts)
    r_first = float(r[0])

    def rhs(U):
        P = pad(U)
        d1 = stencil(P, c1, 2)
        d2 = stencil(P, c2, 2)
        coef = jnp.stack([U.T, d1.T, d2.T], axis=1)           # (n_r, 3, 10)
        res = r_ccz4(apply_fn, (coef, r_first, dr), zeros, x_pts, zeros, zeros, 0.0, 1.0)
        R = -res                                              # d_t field (the model has no t-dependence)
        out = jnp.stack([R[:, 3], R[:, 0] - R[:, 3],          # a, b   (gt_yy, gt_xx - gt_yy)
                         R[:, 9], R[:, 6] - R[:, 9],          # c, d   (At_yy, At_xx - At_yy)
                         R[:, 12], R[:, 13], R[:, 14],        # chi, Khat, Theta
                         R[:, 15], R[:, 18], R[:, 19]])       # g, alpha, B
        return out + stencil(P, ko, 3)

    return r, U0, rhs, jax, jnp


def main():
    args = parse()
    r, U0, rhs, jax, jnp = make_solver(args)
    dr = args.dr
    n_r = len(r)
    m = args.mass
    dt = args.cfl * dr
    n_snap = int(round(args.tfinal / args.snap_dt))
    steps_per_snap = int(round(args.snap_dt / dt))
    dt = args.snap_dt / steps_per_snap

    @jax.jit
    def advance(U):
        def step(U, _):
            k1 = rhs(U)
            k2 = rhs(U + 0.5 * dt * k1)
            k3 = rhs(U + 0.5 * dt * k2)
            k4 = rhs(U + dt * k3)
            return U + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), None
        return jax.lax.scan(step, U, None, length=steps_per_snap)[0]

    print(f"single puncture m={m}, dr={dr}, n_r={n_r}, R_out={args.R_out}, dt={dt:.3e}, "
          f"{steps_per_snap} steps per snapshot of {args.snap_dt}, tfinal={args.tfinal}, "
          f"KO sigma={args.ko_sigma}, chi_floor={args.chi_floor}", flush=True)
    U = jnp.asarray(U0)
    snaps, times = [np.asarray(U)], [0.0]
    t0 = time.time()
    for s in range(1, n_snap + 1):
        U = advance(U)
        Un = np.asarray(U)
        if not np.all(np.isfinite(Un)):
            bad = np.where(~np.all(np.isfinite(Un), axis=0))[0]
            print(f"non-finite state at t = {s * args.snap_dt}; first bad r = {r[bad[:3]]}, last good max-|U| at r = {r[np.argmax(np.abs(snaps[-1]).max(axis=0))]:.3f}; stopping", flush=True)
            break
        snaps.append(Un)
        times.append(s * args.snap_dt)
        print(f"t={times[-1]:6.2f}  alpha(r0)={Un[8, 0]:.4f}  alpha(r={r[np.argmin(abs(r-0.03))]:.3f})="
              f"{Un[8, np.argmin(abs(r - 0.03))]:.4f}  max|Theta|={np.max(abs(Un[6])):.2e}  "
              f"max|B|={np.max(abs(Un[9])):.3f}  wall {time.time() - t0:.0f}s", flush=True)

    out = args.out or f"ref_single_m{m}_dr{dr}.npz"
    keep = r <= args.r_save
    np.savez_compressed(out, r=r[keep], t=np.array(times), U=np.array(snaps)[:, :, keep], fields=np.array(FIELDS),
                        mass=m, dr=dr, R_out=args.R_out, ko_sigma=args.ko_sigma, chi_floor=args.chi_floor,
                        f_alpha=args.f_alpha, f_beta=args.f_beta, eta=args.eta,
                        kappa_z=args.kappa_z, kappa_c=args.kappa_c, kappa_2=args.kappa_2)
    print("wrote", out)


if __name__ == "__main__":
    main()
