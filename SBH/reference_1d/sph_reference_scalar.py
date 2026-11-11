"""Spherically symmetric reference: a puncture plus a massless scalar-field shell that falls in. (2026-11-10)

sph_reference.py (vacuum) with two more radial functions, phi and Pi (scalar_field.py), and the matter sources
in the BH code's own residual (`_ccz4_residual_one(..., matter=...)`). Same grid, stencils, dissipation, RK4,
gauge and chi floor. The purpose (the user, 2026-11-10): a 1D test where the black hole's mass really grows, for
the template with a changing mass and a changing excised core.

Initial data (time-symmetric, conformally flat; the momentum constraint is trivial):
    phi = A exp(-((r - r0)/sigma)^2),  Pi = 0,  gt = delta,  At = 0,  Khat = Theta = 0,  Ghat = 0,  beta = 0,
    chi = psi^-4,  alpha = 1/psi,  psi = 1 + m/(2r) + u(r),
    Hamiltonian constraint:  Lap psi = -pi psi phi'^2,  i.e. with v = r u:  v'' = -pi phi'^2 (r + m/2 + v),
    v(0) = 0, v'(R_out) = 0  (u -> v_inf / r), solved by shooting (linear: particular + c homogeneous, DOP853).
    ADM mass M = m + 2 v_inf.  Half of the shell falls in, half leaves.

Diagnostics per snapshot (also by tools/scalar_ref_diag.py from the npz): apparent horizon (outermost zero of
s^r d_r R - R K^th_th, R = r sqrt(a/chi) the areal radius), its mass R_AH/2, and the Misner-Sharp mass profile.

    JAX_PLATFORMS=cpu python sph_reference_scalar.py --dr 0.01 --tfinal 8 --amp 0.02 --r0 3 --sigma 0.75 --out X.npz
Output npz: r, t, U (n_t, 12, n_r) in the order of FIELDS (the 10 of sph_reference.py, then phi, Pi), r_ah, m_ah
(n_t,), M_adm, plus the run parameters.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
import sph_reference as sr                                           # noqa: E402

FIELDS = sr.FIELDS + ("phi", "Pi")
PARITY = np.concatenate([sr.PARITY, [1.0, 1.0]])
NG = sr.NG


def parse():
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mass", type=float, default=0.5)
    p.add_argument("--amp", type=float, default=0.02, help="A of phi = A exp(-((r - r0)/sigma)^2)")
    p.add_argument("--r0", type=float, default=3.0)
    p.add_argument("--sigma", type=float, default=0.75)
    p.add_argument("--dr", type=float, default=0.01)
    p.add_argument("--R_out", type=float, default=60.0)
    p.add_argument("--tfinal", type=float, default=8.0)
    p.add_argument("--cfl", type=float, default=0.25)
    p.add_argument("--ko_sigma", type=float, default=0.1)
    p.add_argument("--snap_dt", type=float, default=0.25)
    p.add_argument("--chi_floor", type=float, default=1e-5)
    p.add_argument("--f_alpha", type=float, default=1.0)
    p.add_argument("--f_beta", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--kappa_z", type=float, default=0.0)
    p.add_argument("--kappa_c", type=float, default=0.0)
    p.add_argument("--kappa_2", type=float, default=0.0)
    p.add_argument("--r_save", type=float, default=30.0)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def conformal_u(m, A, r0, s, R_out, r_eval):
    """u(r_eval) of psi = 1 + m/(2r) + u for the shell (r_eval increasing, r_eval[0] may be 0), and the ADM mass."""
    from scipy.integrate import solve_ivp
    dphi2 = lambda x: (A * np.exp(-((x - r0) / s) ** 2) * (-2.0 * (x - r0) / s ** 2)) ** 2
    part = lambda x, y: [y[1], -np.pi * dphi2(x) * (x + m / 2 + y[0])]
    homo = lambda x, y: [y[1], -np.pi * dphi2(x) * y[0]]
    kw = dict(method="DOP853", rtol=1e-13, atol=1e-16, t_eval=np.concatenate([r_eval, [R_out]]))
    yp = solve_ivp(part, (0.0, R_out), [0.0, 0.0], **kw).y
    yh = solve_ivp(homo, (0.0, R_out), [0.0, 1.0], **kw).y
    c = -yp[1, -1] / yh[1, -1]
    v, dv = (yp[0] + c * yh[0])[:-1], (yp[1] + c * yh[1])[:-1]
    u = np.where(r_eval > 0, v / np.where(r_eval > 0, r_eval, 1.0), dv)      # u(0) = v'(0)
    return u, m + 2.0 * (yp[0, -1] + c * yh[0, -1])


def initial_data(args, r):
    """U0 (12, n_r) and the ADM mass."""
    u, M_adm = conformal_u(args.mass, args.amp, args.r0, args.sigma, args.R_out, r)
    psi = 1.0 + args.mass / (2.0 * r) + u
    U0 = np.zeros((12, len(r)))
    U0[0] = 1.0
    U0[4] = psi ** -4
    U0[8] = 1.0 / psi
    U0[10] = args.amp * np.exp(-((r - args.r0) / args.sigma) ** 2)
    return U0, M_adm


def diagnostics(r, U):
    """(r_ah, m_ah, Misner-Sharp mass profile) of one snapshot U (12, n_r); r_ah = nan if no horizon."""
    a, b, c, chi, Kh, Th = U[0], U[1], U[2], np.maximum(U[4], 1e-300), U[5], U[6]
    with np.errstate(all="ignore"):                      # the innermost points (chi ~ 1e-7) overflow; not used
        R = r * np.sqrt(a / chi)
        dR = np.gradient(R, r, edge_order=2)
        g_rr = (a + b) / chi
        ratio = (c + a * (Kh + 2.0 * Th) / 3.0) / a
        exp_out = dR / np.sqrt(g_rr) - R * ratio
        ms = 0.5 * R * (1.0 - dR ** 2 / g_rr + R ** 2 * ratio ** 2)
    ok = r > 0.05                                        # the innermost points sit on the chi floor
    idx = np.where(ok[:-1] & (exp_out[:-1] < 0) & (exp_out[1:] >= 0))[0]
    if len(idx) == 0:
        return np.nan, np.nan, ms
    k = idx[-1]
    w = -exp_out[k] / (exp_out[k + 1] - exp_out[k])
    return r[k] + w * (r[k + 1] - r[k]), 0.5 * (R[k] + w * (R[k + 1] - R[k])), ms


def make_solver(args):
    ns, jax, jnp, _, _ = sr.build(args)
    import scalar_field as sf
    import adm_constraints as admc
    pack = ns["pack_sym3"]
    DD = jnp.concatenate([ns["_DD_OUTPUT_IDXS"], jnp.array([22])])   # the 11 of the BH code, then phi
    res_one = ns["_ccz4_residual_one"]

    dr = args.dr
    n_r = int(round(args.R_out / dr))
    r = (np.arange(n_r) + 0.5) * dr
    U0, M_adm = initial_data(args, r)

    par = jnp.asarray(PARITY)[:, None]
    c1 = jnp.array([1.0, -8.0, 0.0, 8.0, -1.0]) / (12.0 * dr)
    c2 = jnp.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / (12.0 * dr * dr)
    ko = jnp.array([1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0]) * (args.ko_sigma / (64.0 * dr))

    def pad(U):
        left = (U[:, :NG] * par)[:, ::-1]
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

    def cart(rad, xyz):
        """The 22 Cartesian fields of the BH code, then phi and Pi, from the 12 radial functions at xyz."""
        a, b, c, d, chi, Kh, Th, g, al, B, phi, Pi = rad
        n = xyz / jnp.sqrt(jnp.sum(xyz ** 2))
        nn, I = jnp.outer(n, n), jnp.eye(3)
        return jnp.concatenate([pack((a * I + b * nn)[None])[0], pack((c * I + d * nn)[None])[0],
                                jnp.stack([chi, Kh, Th]), g * n, al[None], B * n, jnp.stack([phi, Pi])])

    def point(c0, c1_, c2_, rk):
        """Stacks at (rk, 0, 0) of the local quadratic Taylor model (the cartoon idea of sph_reference.py)."""
        def f(p):
            h = jnp.sqrt(jnp.sum(p ** 2)) - rk
            return cart(c0 + c1_ * h + 0.5 * c2_ * h * h, p)
        p0 = jnp.array([1.0, 0.0, 0.0]) * rk
        return f(p0), jax.jacfwd(f)(p0), jax.jacfwd(jax.jacfwd(lambda p: f(p)[DD]))(p0)

    def rhs_point(c0, c1_, c2_, rk):
        val, d, dd = point(c0, c1_, c2_, rk)
        v22, d22, dd11 = val[:22], d[:22], dd[:11]
        sv, sd, phi_dd = val[22:], d[22:], dd[11]
        R = -res_one(v22, jnp.zeros(22), d22, dd11, matter=sf.matter_one(ns, v22, d22, sv, sd))
        rp, rP = sf.kg_rhs_one(ns, v22, d22, sv, sd, phi_dd)
        return jnp.stack([R[3], R[0] - R[3], R[9], R[6] - R[9], R[12], R[13], R[14], R[15], R[18], R[19], rp, rP])

    def con_point(c0, c1_, c2_, rk):
        """(H - 16 pi rho, M_x - 8 pi S_x) at (rk, 0, 0)."""
        val, d, dd = point(c0, c1_, c2_, rk)
        v22, d22 = val[:22], d[:22]
        rho, Si, _ = sf.matter_one(ns, v22, d22, val[22:], d[22:])
        H, M = admc.constraints_one(ns, v22, d22, dd[:11])
        return jnp.stack([H - 16.0 * jnp.pi * rho, M[0] - 8.0 * jnp.pi * Si[0]])

    r_j = jnp.asarray(r)

    def coefs(U):
        P = pad(U)
        return P, U.T, stencil(P, c1, 2).T, stencil(P, c2, 2).T

    def rhs(U):
        P, v, d1, d2 = coefs(U)
        return jax.vmap(rhs_point)(v, d1, d2, r_j).T + stencil(P, ko, 3)

    def constraints(U):
        _, v, d1, d2 = coefs(U)
        return jax.vmap(con_point)(v, d1, d2, r_j).T              # (2, n_r)

    return r, U0, M_adm, rhs, constraints, jax, jnp


def main():
    args = parse()
    r, U0, M_adm, rhs, constraints, jax, jnp = make_solver(args)
    dt = args.cfl * args.dr
    n_snap = int(round(args.tfinal / args.snap_dt))
    steps = int(round(args.snap_dt / dt))
    dt = args.snap_dt / steps

    @jax.jit
    def advance(U):
        def step(U, _):
            k1 = rhs(U)
            k2 = rhs(U + 0.5 * dt * k1)
            k3 = rhs(U + 0.5 * dt * k2)
            k4 = rhs(U + dt * k3)
            return U + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), None
        return jax.lax.scan(step, U, None, length=steps)[0]

    con = jax.jit(constraints)
    band = (r > 0.2) & (r < 20.0)
    far = (r > 1.0) & (r < 20.0)

    def line(t, Un):
        ra, ma, ms = diagnostics(r, Un)
        C = np.asarray(con(jnp.asarray(Un)))
        return ra, ma, (f"t={t:6.2f}  r_AH={ra:.4f}  M_AH={ma:.6f}  MS(r=20)={np.interp(20.0, r, ms):.6f}  "
                        f"max|H-16pi rho|, |M-8pi S| in r=0.2-20: {np.abs(C[0, band]).max():.1e} {np.abs(C[1, band]).max():.1e}, "
                        f"r=1-20: {np.abs(C[0, far]).max():.1e} {np.abs(C[1, far]).max():.1e}  "
                        f"max|phi|={np.abs(Un[10]).max():.3e}")

    print(f"m={args.mass}, shell A={args.amp} r0={args.r0} sigma={args.sigma}, M_ADM={M_adm:.8f}, dr={args.dr}, "
          f"n_r={len(r)}, dt={dt:.3e}, tfinal={args.tfinal}", flush=True)
    U = jnp.asarray(U0)
    snaps, times, rah, mah = [np.asarray(U)], [0.0], [], []
    ra, ma, txt = line(0.0, snaps[0])
    rah.append(ra); mah.append(ma)
    print(txt, flush=True)
    t0 = time.time()
    for s in range(1, n_snap + 1):
        U = advance(U)
        Un = np.asarray(U)
        if not np.all(np.isfinite(Un)):
            print(f"non-finite state at t = {s * args.snap_dt}; stopping", flush=True)
            break
        snaps.append(Un); times.append(s * args.snap_dt)
        ra, ma, txt = line(times[-1], Un)
        rah.append(ra); mah.append(ma)
        print(txt + f"  wall {time.time() - t0:.0f}s", flush=True)

    out = args.out or f"ref_scalar_m{args.mass}_A{args.amp}_dr{args.dr}.npz"
    keep = r <= args.r_save
    np.savez_compressed(out, r=r[keep], t=np.array(times), U=np.array(snaps)[:, :, keep], fields=np.array(FIELDS),
                        r_ah=np.array(rah), m_ah=np.array(mah), M_adm=M_adm, mass=args.mass, amp=args.amp,
                        r0=args.r0, sigma=args.sigma, dr=args.dr, R_out=args.R_out, ko_sigma=args.ko_sigma,
                        chi_floor=args.chi_floor, f_alpha=args.f_alpha, f_beta=args.f_beta, eta=args.eta,
                        kappa_z=args.kappa_z, kappa_c=args.kappa_c, kappa_2=args.kappa_2)
    print("wrote", out)


if __name__ == "__main__":
    main()
