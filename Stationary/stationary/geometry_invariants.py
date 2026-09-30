"""Gauge-invariant geometry of a static solution: curvature, algebra, and global charges.

WHY THIS MODULE EXISTS

The residual system this repo solves is stated in `geometry.py` and, in the 4-d reading of
`README.md` section 1, is **static Einstein + a massless scalar field**:

    g = -dt^2 + h_ij dx^i dx^j ,        phi = log lambda ,
    R_ij(h) = (1/2) phi_i phi_j         (and Delta_h phi = 0 by the Bianchi identity)

Metric components are chart dependent -- the harmonic gauge leaves the freedom of harmonic
diffeomorphisms -- so two runs cannot be compared component by component.  Scalars can, and
this module computes them.  It is additive: nothing here is used by training.

THE SECOND READING, WHICH IS THE PHYSICAL ONE

The SAME pair (h, lambda) also defines a static **vacuum** space-time through the completion

    g = -lambda^a dt^2 + lambda^-a h_ij dx^i dx^j ,      a = 1 ,

because the vacuum Weyl metric is ds^2 = -e^{2U}dt^2 + e^{-2U}[...] and this code's lambda is
e^{2U}.  Both completions solve their own equations exactly whenever (h, lambda) solves the
residual system, so both are offered through `reading`:

    reading="vacuum"  (a = 1)   the black-hole geometry -- masses, horizons, tidal field
    reading="scalar"  (a = 0)   the literal residual system, g = -dt^2 + h

Verified numerically, not assumed (`tests/test_geometry_invariants.py`):

  * the spherical asset `exact.exact_fields(R0, k)` is, in the vacuum reading, **Schwarzschild
    of mass M = R0 in harmonic (de Donder) coordinates**: the areal radius of the coordinate
    sphere rho is `rho + R0` -- NOT sqrt(rho^2 - R0^2), which is the areal radius in the
    SCALAR reading -- its lapse is exactly 1 - 2M/r_a, and its Kretschmann scalar is
    48 M^2/r_a^6 to machine precision;
  * the Weyl two-rod asset is the vacuum Israel-Khan solution of total mass 1.1;
  * C_abcd *C^abcd = 0 for both, which is the gauge-invariant certificate of staticity.

WHAT IS COMPUTED

  curvature        R, R_ab R^ab, R_abcd R^abcd, C_abcd C^abcd, C_abcd *C^abcd
  algebra          electric and magnetic parts of the Weyl tensor for the static observer
                   (H = 0 for a static metric), the tidal eigenvalues, the Newman-Penrose
                   scalars psi_0..psi_4, I, J and the speciality index S = 27 J^2/I^3
                   (S = 1 exactly for type D)
  slice            R_ij, its eigenvalues, the **rank-one defect** R_ij R^ij - R^2 (zero on a
                   solution: R_ij = (1/2) phi_i phi_j is rank one with eigenvalue R), the 3-d
                   identity R_abcd R^abcd = 4 R_ij R^ij - R^2, and the Cotton norm
  global           area and areal radius of a coordinate sphere, its mean curvature, and the
                   Hawking mass (validated: 0 in flat space, M for Schwarzschild, and -> 1.1
                   for Israel-Khan).  `hawking_profile` gives m_H(r) as a function of radius
                   -- the mass profile of the solution -- and `hawking_mass_field` interpolates
                   it onto a grid for the VTK export.  The quadrature is Gauss-Legendre in mu,
                   which makes the sphere integrals exact to round-off even at 6 x 4 nodes
                   (the midpoint rule was 0.2% off at 10 x 6, larger than the error being
                   measured); phi stays a midpoint rule, where it is already spectral.

NOT HERE YET (each needs its own care): the horizon rod area and surface gravity, Geroch-Hansen
multipoles, geodesic observables (ISCO, photon ring), a `geometry.json` next to the run, the
`plane.py` curvature panel and the VTK export.  (`stationary.report` already prints the summary
this module produces, so the text side of the plumbing is done.)  Also **points exactly on the
axis**: the Cartesian components of an axisymmetric metric have direction-dependent second
derivatives at rho_cyl = 0, so the invariants there come back NaN rather than wrong.
`weyl.curvature_invariant` documents the same limitation and evaluates in the cylindrical chart
instead, which is the route to copy when the axis itself has to be reported (it is where the
rods are).
"""
from __future__ import annotations

import math
from typing import Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

READINGS = {"scalar": 0.0, "vacuum": 1.0}


def alpha_of(reading) -> float:
    return READINGS[reading] if isinstance(reading, str) else float(reading)


# ------------------------------------------------------------------- 4-d metric layer
def reading_metric(h: Callable, lam: Callable, reading="vacuum"):
    """g = -lambda^a dt^2 + lambda^-a h, as a (4,4) array function of x4 = (t, x, y, z)."""
    a = alpha_of(reading)

    def g(x4):
        x = x4[1:]
        H, L = h(x), lam(x)
        G = jnp.zeros((4, 4), dtype=H.dtype)
        G = G.at[1:, 1:].set(H * L ** (-a))
        G = G.at[0, 0].set(-(L**a))
        return G

    return g


def _unit_lambda(h, lam, reading):
    """hhat = lambda^-a h and the unchanged lambda, i.e. the pair to hand to the machinery."""
    a = alpha_of(reading)
    if a == 0.0:
        return h, lam
    return (lambda x: h(x) * lam(x) ** (-a)), lam


def christoffel4(g, x):
    G4 = g(x)
    dg = jax.jacfwd(g)(x)                       # dg[a,b,c] = d_c g_ab
    Ginv = jnp.linalg.inv(G4)
    T = dg.transpose(1, 2, 0) + dg - dg.transpose(2, 0, 1)
    return 0.5 * jnp.einsum("ad,dbc->abc", Ginv, T)


def _levi_civita4():
    """eps_abcd with eps_0123 = -1 (mostly-plus signature)."""
    eps = jnp.zeros((4, 4, 4, 4))
    for a in range(4):
        for b in range(4):
            rest = [i for i in range(4) if i not in (a, b)]
            eps = eps.at[a, b, rest[0], rest[1]].set(1.0)
            eps = eps.at[a, b, rest[1], rest[0]].set(-1.0)
    return -eps


ETA = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


def frame_vectors(g4):
    """Columns: an orthonormal frame of the metric g4 -- static observer first, then the triad.

    The spatial block is turned into an orthonormal triad by its symmetric inverse square root,
    h^{-1/2}, so that E^T h E = I; for the chart metrics used here h is diagonal and this is
    just diag(1/sqrt(h_ii)), i.e. the chart's own (rho, phi, z) directions.
    """
    sp = g4[1:, 1:]
    w, V = jnp.linalg.eigh(sp)
    Shalf = V @ jnp.diag(1.0 / jnp.sqrt(w)) @ V.T
    E = jnp.zeros((4, 4))
    E = E.at[0, 0].set(1.0 / jnp.sqrt(-g4[0, 0]))
    E = E.at[1:, 1:].set(Shalf)
    return E


def curvature_at(h, lam, x, reading="vacuum", tetrad=True, frame="coordinate"):
    """Everything curvature at the spatial point x, for one completion.

    Returns a dict with the Ricci tensor and scalar, the Kretschmann scalar, the Weyl tensor
    (lowered and raised), C^2, the Pontryagin density, the electric/magnetic parts of the Weyl
    tensor and their eigenvalues, and the Newman-Penrose data.

    A point where the spatial metric is not positive definite gives NaN rather than a wrong
    number -- the spherical asset, for instance, is only Riemannian for |x| > R0, and the
    tetrad cannot be built inside that.
    """
    g = reading_metric(h, lam, reading)
    x4 = jnp.concatenate([jnp.zeros(1), x])
    G = christoffel4(g, x4)
    dG = jax.jacfwd(lambda y: christoffel4(g, y))(x4)
    Rm = (dG.transpose(0, 1, 3, 2) - dG
          + jnp.einsum("ace,ebd->abcd", G, G)
          - jnp.einsum("ade,ebc->abcd", G, G))                 # R^a_bcd
    G4 = g(x4)
    if frame == "orthonormal":
        # Contract in an orthonormal frame.  The scalars do not care which frame they are
        # computed in ANALYTICALLY, but numerically they care a great deal: in the chart
        # (rho, phi, z) one has h_phiphi = rho^2 M_phiphi, so the coordinate components of a
        # four-index contraction carry 1/rho^4 factors and the answer comes out of a
        # cancellation that loses ~1/rho^2 digits for the Kretschmann and the Weyl scalars
        # (measured: 1e+09 where the answer is 1e+05, at rho = 1e-4 of the rod scale).  In an
        # orthonormal frame every component is O(1) and the same contraction is stable.  This
        # is what makes the invariants usable ON the axis, where the physics is.
        E = frame_vectors(G4)
        Rm = jnp.einsum("ap,qb,rc,sd,pqrs->abcd", jnp.linalg.inv(E), E, E, E, Rm)
        G4 = ETA
    Ginv = jnp.linalg.inv(G4)
    Rlow = jnp.einsum("ae,ebcd->abcd", G4, Rm)
    Ric = jnp.einsum("cbcd->bd", Rm)
    R = jnp.einsum("bd,bd->", Ginv, Ric)
    K = jnp.einsum("abcd,abcd->", Rlow,
                   jnp.einsum("ae,bf,cg,dh,efgh->abcd", Ginv, Ginv, Ginv, Ginv, Rlow))
    gaR = jnp.einsum("ac,bd->abcd", G4, Ric)
    C = (Rlow
         - 0.5 * (gaR - gaR.transpose(0, 1, 3, 2) + gaR.transpose(2, 3, 0, 1)
                  - gaR.transpose(2, 3, 1, 0))
         + (R / 6.0) * (jnp.einsum("ac,bd->abcd", G4, G4)
                        - jnp.einsum("ad,bc->abcd", G4, G4)))
    Cup = jnp.einsum("ae,bf,cg,dh,efgh->abcd", Ginv, Ginv, Ginv, Ginv, C)
    C2 = jnp.einsum("abcd,abcd->", C, Cup)
    star = 0.5 * jnp.einsum("abef,efcd->abcd", _levi_civita4(), C)
    CdotC = jnp.einsum("abcd,abcd->", star, Cup)
    out = dict(Ric=Ric, R=R, K=K, C=C, C2=C2, CdotC=CdotC, g=G4, x4=x4,
               RijRij=jnp.einsum("bd,be,de->", Ric, Ric, Ginv))
    if tetrad:
        if frame == "orthonormal":
            # the triad IS the frame: pass the identity metric and the radial frame direction
            out.update(_algebra(lambda y: jnp.eye(3), lambda y: 1.0, "scalar",
                                jnp.array([1.0, 0.0, 0.0]), C, G4))
        else:
            out.update(_algebra(h, lam, reading, x, C, G4))
    out["frame"] = frame
    return out


def _algebra(h, lam, reading, x, C, G4):
    """Electric/magnetic split, tidal eigenvalues and the NP scalars."""
    u = jnp.zeros(4).at[0].set(1.0)
    u = u / jnp.sqrt(-(u @ G4 @ u))          # contravariant: u^t = lambda^(-a/2)
    # E_ab = C_acbd u^c u^d and H_ab = *C_acbd u^c u^d contract the CONTRAVARIANT u twice.
    # Contracting the lowered u_c u_d instead costs a factor lambda^(-a) per slot -- for the
    # vacuum reading at rho = 1.5 that is a silent factor of 25 in every tidal eigenvalue.
    E = jnp.einsum("acbd,c,d->ab", C, u, u)
    star = 0.5 * jnp.einsum("abef,efcd->abcd", _levi_civita4(), C)
    H = jnp.einsum("acbd,c,d->ab", star, u, u)
    # spatial triad: n the outward normal, (e1, e2) tangent, all hhat-orthonormal
    hhat, _ = _unit_lambda(h, lam, reading)
    H3 = hhat(x)
    n = jnp.linalg.solve(H3, x / jnp.linalg.norm(x))
    n = n / jnp.sqrt(n @ H3 @ n)
    # tangent pair: seed Gram-Schmidt with the coordinate direction LEAST aligned with n.
    # A fixed seed (z, then y) fails outright on the axis, where n IS z: e1 comes out None and
    # the cross product raises.  The axis is not a corner case here -- it is where the rods
    # live and where the solution is axisymmetric, hence type D by symmetry.
    eye = jnp.eye(3)
    align = jnp.abs(jnp.einsum("ai,ij,j->a", eye, H3, n))
    ref = eye[jnp.argmin(align)]
    v = ref - (ref @ H3 @ n) * n
    e1 = v / jnp.sqrt(v @ H3 @ v)
    e2 = jnp.linalg.solve(H3, jnp.cross(n, e1))
    e2 = e2 / jnp.sqrt(e2 @ H3 @ e2)
    # eigenvalues of E as a mixed spatial tensor (orthonormal frame -> usual eigenvalues)
    # E_ab is lowered and spatial, so the frame components are a^i E_ij b^j for the
    # hhat-orthonormal triad (a, b) -- NOT the inner product a.H.b.
    E3 = E[1:, 1:]
    Ei = jnp.array([[a @ E3 @ b for b in (n, e1, e2)] for a in (n, e1, e2)])
    #  psi_0..psi_4 in the static null tetrad l,n radial and m from (e1,e2)
    zero = jnp.zeros(1)
    l = (u + jnp.concatenate([zero, n])) / jnp.sqrt(2.0)
    nn = (u - jnp.concatenate([zero, n])) / jnp.sqrt(2.0)
    m = (jnp.concatenate([zero, e1]) + 1j * jnp.concatenate([zero, e2])) / jnp.sqrt(2.0)
    mb = jnp.conj(m)
    # C is stored LOWERED, so every slot contracts with a CONTRAVARIANT tetrad leg.  Lowering
    # the legs first (as this code did) multiplies psis by metric factors and silently scales
    # psi_2 by ~1/lambda^2 -- the same mistake as in the electric part above.
    psi = jnp.array([jnp.einsum("abcd,a,b,c,d->", C, l, m, l, m),
                     jnp.einsum("abcd,a,b,c,d->", C, l, nn, l, m),
                     jnp.einsum("abcd,a,b,c,d->", C, l, m, mb, nn),
                     jnp.einsum("abcd,a,b,c,d->", C, l, nn, mb, nn),
                     jnp.einsum("abcd,a,b,c,d->", C, nn, mb, nn, mb)])
    I = psi[0] * psi[4] - 4.0 * psi[1] * psi[3] + 3.0 * psi[2] ** 2
    M = jnp.array([[psi[0], psi[1], psi[2]],
                   [psi[1], psi[2], psi[3]],
                   [psi[2], psi[3], psi[4]]])
    J = jnp.linalg.det(M)
    S = 27.0 * J**2 / I**3
    code = petrov_code(I, J, S, jnp.max(jnp.abs(psi)))
    return dict(E=E, H=H, E_eigs=jnp.linalg.eigvalsh(Ei), H_norm=jnp.linalg.norm(H),
                psi=psi, I=I, J=J, S=S, petrov_code=code, petrov=petrov_name(code))


PETROV_NAMES = ("O/N/III", "D", "I")


def petrov_code(I, J, S, psi_scale=1.0, tol=1e-6):
    """Petrov label as a NUMBER (0 = flat/special, 1 = D, 2 = I), safe under vmap/jit.

    The classification has to be branch-free: this module is called both eagerly (reports,
    tests) and inside `vmap` over grid points (the VTK export, the half-plane maps), and a
    Python `if` on a traced value raises `TracerBoolConversionError`.  The test for I = 0 is
    RELATIVE -- |I| ~ |psi|^2 falls off with radius, so an absolute tolerance turns a distant
    Schwarzschild point into 'O/N/III' (which it did, in the first version of this function).
    """
    scale2 = jnp.maximum(jnp.asarray(psi_scale) ** 2, 1e-300)
    return jnp.where(jnp.abs(I) < tol * scale2, 0,
                     jnp.where(jnp.abs(S - 1.0) < 1e-3, 1, 2))


def petrov_name(code):
    """The label for a CONCRETE code (reports only); '?' when the code is a tracer."""
    try:
        return PETROV_NAMES[int(code)]
    except Exception:
        return "?"


def petrov_type(I, J, S, psi_scale=1.0, tol=1e-6):
    """The label, eagerly: `petrov_name(petrov_code(...))`."""
    return petrov_name(petrov_code(I, J, S, psi_scale, tol))


# ------------------------------------------------------------------ spatial slice
def spatial_at(h, x):
    """3-d Ricci, R, curvature invariants and the Cotton norm of the slice metric h.

    Computed through the product g = -dt^2 + h, whose spatial block IS the 3-d Ricci and
    whose Kretschmann IS the 3-d one -- one piece of machinery, already validated on
    Schwarzschild, instead of two.
    """
    c = curvature_at(h, lambda y: 1.0, x, "scalar", tetrad=False)
    H = h(x)
    Hinv = jnp.linalg.inv(H)
    Ric3 = c["Ric"][1:, 1:]
    R3 = c["R"]
    RijRij = jnp.einsum("bd,be,de->", Ric3, Ric3, Hinv)
    # Cotton: C_ijk = D_k R_ij - D_j R_ik + (1/4)(h_ik D_j R - h_ij D_k R)
    dRic = jax.jacfwd(lambda y: curvature_at(h, lambda z: 1.0, y, "scalar",
                                             tetrad=False)["Ric"][1:, 1:])(x)
    dR = jax.jacfwd(lambda y: curvature_at(h, lambda z: 1.0, y, "scalar",
                                           tetrad=False)["R"])(x)
    Cott = (dRic - dRic.transpose(0, 2, 1)
            + 0.25 * (jnp.einsum("ik,j->ijk", H, dR) - jnp.einsum("ij,k->ijk", H, dR)))
    Cott2 = jnp.einsum("ijk,ilm,jl,km->", Cott, Cott, Hinv, Hinv)
    return dict(Ric=Ric3, R=R3, K3=c["K"], RijRij=RijRij, rank1_defect=RijRij - R3**2,
                cotton2=Cott2, eigs=jnp.linalg.eigvalsh(jnp.einsum("ik,kj->ij", Hinv, Ric3)))


# --------------------------------------------------------------- global quantities
_GL_MU: dict = {}


def _mu_rule(n_mu):
    """Gauss-Legendre nodes and weights on mu in [-1, 1] (cached).

    A midpoint rule in mu is only second order; the sphere integrals here -- the area, the mean
    curvature and the Hawking mass, which needs k^2 -- are smooth functions of mu, so
    Gauss-Legendre is spectrally accurate at the same cost.  It also has no node at the poles,
    which matters because the Cartesian components of an axisymmetric metric are not smooth
    there (see the module docstring).  The phi integral stays a midpoint rule: the integrand is
    2pi-periodic and smooth, where the trapezoidal rule is already spectral.
    """
    if n_mu not in _GL_MU:
        import numpy as np
        x, w = np.polynomial.legendre.leggauss(int(n_mu))
        _GL_MU[int(n_mu)] = (jnp.asarray(x), jnp.asarray(w))
    return _GL_MU[int(n_mu)]


def _sphere_points(rho, n_mu, n_phi):
    """Gauss-Legendre nodes in mu = cos(theta) and midpoint nodes in phi.

    The area element sqrt(det g2) carries a factor sin(theta) that the 1/sin(theta) in `one`
    removes, so the remaining integrand is smooth in mu and the GL rule converges fast.
    """
    mu, w_mu = _mu_rule(n_mu)
    ph = 2.0 * jnp.pi * (jnp.arange(n_phi) + 0.5) / n_phi
    st = jnp.sqrt(1.0 - mu**2)
    pts = rho * jnp.stack([st[:, None] * jnp.cos(ph)[None, :],
                           st[:, None] * jnp.sin(ph)[None, :],
                           jnp.broadcast_to(mu[:, None], (n_mu, n_phi))], axis=-1)
    return pts, mu, w_mu, ph, st


def sphere_geometry(h, lam, rho, reading="vacuum", n_mu=48, n_phi=32):
    """Area, areal radius, mean curvature, Hawking mass and mean lambda on |x| = rho.

    With K_ij = 0 (static), the Hawking mass is

        m_H = sqrt(A/16 pi) [ 1 - (1/16 pi) oint k^2 dA ] ,

    k being the mean curvature of the sphere in the physical spatial metric hhat = lambda^-a h.
    Validated against the two exact cases: flat space gives 0, Schwarzschild gives M.
    """
    hhat, _ = _unit_lambda(h, lam, reading)
    pts, mu, w_mu, ph, st = _sphere_points(rho, n_mu, n_phi)
    dphi = 2.0 * jnp.pi / n_phi
    th = jnp.arccos(mu)

    def induced(y, thv, phv):
        # tangents of the ACTUAL coordinate lines of the sphere x = rho n(theta,phi):
        # their lengths carry the rho and sin(theta) factors, which is exactly what the area
        # element needs -- building them from unit vectors and integrating with dtheta dphi
        # loses a factor rho^2 sin(theta) (this was a bug, caught by the flat-space test).
        r = jnp.linalg.norm(y)
        e_th = r * jnp.array([jnp.cos(thv) * jnp.cos(phv), jnp.cos(thv) * jnp.sin(phv),
                              -jnp.sin(thv)])
        e_ph = r * jnp.sin(thv) * jnp.array([-jnp.sin(phv), jnp.cos(phv), 0.0])
        H = hhat(y)
        return jnp.array([[e_th @ H @ e_th, e_th @ H @ e_ph],
                          [e_th @ H @ e_ph, e_ph @ H @ e_ph]])

    def one(y, thv, phv, stv):
        g2 = induced(y, thv, phv)
        G = christoffel4(reading_metric(hhat, lambda z: 1.0, "scalar"),
                         jnp.concatenate([jnp.zeros(1), y]))
        G3 = G[1:, 1:, 1:]
        H3 = hhat(y)
        Hinv = jnp.linalg.inv(H3)
        # outward unit normal 1-form n_j = (x_j/|x|) / |grad |x||_hhat
        x = y
        r = jnp.linalg.norm(x)
        grad = x / r
        norm = jnp.sqrt(jnp.einsum("ij,i,j->", Hinv, grad, grad))
        nj = grad / norm
        dnj = jax.jacfwd(lambda z: (z / jnp.linalg.norm(z))
                         / jnp.sqrt(jnp.einsum("ij,i,j->", jnp.linalg.inv(hhat(z)),
                                               z / jnp.linalg.norm(z), z / jnp.linalg.norm(z))))(y)
        k = jnp.einsum("ij,ij->", Hinv, dnj) - jnp.einsum("ij,kij,k->", Hinv, G3, nj)
        return jnp.sqrt(jnp.linalg.det(g2)) / stv, k

    dens_k = jax.vmap(jax.vmap(one))(
        pts,
        jnp.broadcast_to(th[:, None], pts.shape[:2]),
        jnp.broadcast_to(ph[None, :], pts.shape[:2]),
        jnp.broadcast_to(st[:, None], pts.shape[:2]))
    dens, k = dens_k[0], dens_k[1]
    lam_vals = jax.vmap(jax.vmap(lambda y: lam(y)))(pts)
    A = jnp.einsum("ij,i->", dens, w_mu) * dphi
    k2 = jnp.einsum("ij,ij,ij,i->", k, k, dens, w_mu) * dphi
    k1 = jnp.einsum("ij,ij,i->", k, dens, w_mu) * dphi
    lam_int = jnp.einsum("ij,ij,i->", lam_vals, dens, w_mu) * dphi
    r_areal = jnp.sqrt(A / (4.0 * jnp.pi))
    return dict(area=A, r_areal=r_areal, k_mean=float(k1 / A),
                hawking_mass=r_areal / 2.0 * (1.0 - k2 / (16.0 * jnp.pi)),
                lam_mean=lam_int / A)


def hawking_profile(h, lam, radii, reading="vacuum", n_mu=12, n_phi=8, reference=None):
    """The Hawking mass as a function of the coordinate sphere: the mass profile of a solution.

    For a static solution (K_ij = 0) the Hawking mass of the sphere |x| = r, in the physical
    spatial metric hhat = lambda^-a h, is

        m_H(r) = sqrt(A/16 pi) [ 1 - (1/16 pi) oint k^2 dA ] ,

    with k the mean curvature of that sphere in hhat.  Flat space gives 0 and Schwarzschild
    gives M at EVERY radius, so the PROFILE is what carries information:

      * a configuration whose source is entirely inside the innermost sphere has m_H flat at
        the total mass from there outward -- a slope means either a strut/interaction energy
        (Israel-Khan holds its two holes apart with a conical strut, so its m_H approaches the
        sum of the horizon masses from ABOVE, as the strut's share falls off with radius) or
        the solution's own error;
      * m_H(rho_out) against the exact value is a check on the run that uses no curvature at
        all, which makes it a good cross-check on the gauge-invariant ones;
      * `m_lapse = r_a (1 - lambda)/2` is the same quantity read off the lapse instead of the
        slice's mean curvature, so the two disagreeing is a warning that the reading is wrong.

    The Gauss-Legendre rule in mu makes this exact to round-off on smooth data even at n_mu = 6
    (checked on Schwarzschild), so the cost is the field evaluations, not the quadrature.
    """
    rows = []
    for r in radii:
        sc = sphere_geometry(h, lam, r, reading, n_mu=n_mu, n_phi=n_phi)
        ra, lm = float(sc["r_areal"]), float(sc["lam_mean"])
        row = [float(r), float(sc["hawking_mass"]), ra, lm, 0.5 * ra * (1.0 - lm)]
        if reference is not None:
            sce = sphere_geometry(reference[0], reference[1], r, reading,
                                  n_mu=n_mu, n_phi=n_phi)
            row += [float(sce["hawking_mass"]), float(sce["r_areal"])]
        rows.append(row)
    cols = list(zip(*rows))
    out = dict(radii=jnp.asarray(cols[0]), m_h=jnp.asarray(cols[1]), r_areal=jnp.asarray(cols[2]),
               lam_mean=jnp.asarray(cols[3]), m_lapse=jnp.asarray(cols[4]))
    if reference is not None:
        out["m_h_ref"] = jnp.asarray(cols[5])
        out["r_areal_ref"] = jnp.asarray(cols[6])
    return out


def hawking_mass_field(h, lam, xs, reading="vacuum", n_radii=24, n_mu=8, n_phi=6):
    """m_H(|x|) interpolated to the grid points: the mass profile as a 3-d scalar field.

    The profile is a function of the radius alone, so it is computed on `n_radii` levels
    spanning the radii present in xs and interpolated linearly to each node -- for a spherical
    grid that is exact (its nodes ARE at those radii), and for a box it is the profile of the
    coordinate sphere through the node.  Cost: n_radii sphere integrals, independent of the
    number of nodes, which is why this is cheap even on a 30k-point export.
    """
    r = jnp.linalg.norm(xs, axis=-1)
    r_lo, r_hi = float(jnp.min(r)), float(jnp.max(r))
    radii = jnp.linspace(r_lo, r_hi, int(n_radii))
    prof = jnp.asarray([float(sphere_geometry(h, lam, float(rr), reading,
                                              n_mu=n_mu, n_phi=n_phi)["hawking_mass"])
                        for rr in radii])
    return {"hawking_mass": jnp.interp(r, radii, prof)}


# ------------------------------------------------------- axisymmetric chart route
def axis_from_config(cfg):
    """The axis an axisymmetric run is symmetric about: z, or the ROTATED axis for --weyl.

    A `--weyl --weyl-rotate-deg 45` run is still axisymmetric -- the rods lie on the rotated
    axis -- but the Cartesian components' bad direction is that axis, not z, so everything that
    needs to know where the axis is must ask here rather than assume z.
    """
    if getattr(cfg, "weyl", False) and getattr(cfg, "weyl_rotate_deg", 0.0):
        from .weyl import rotation_matrix
        return rotation_matrix(cfg.weyl_rotate_deg) @ jnp.array([0.0, 0.0, 1.0])
    return jnp.array([0.0, 0.0, 1.0])


def axis_frame(axis=None):
    """(n, e1, e2): the unit axis and an orthonormal pair perpendicular to it."""
    n = jnp.array([0.0, 0.0, 1.0]) if axis is None else jnp.asarray(axis, jnp.float64)
    n = n / jnp.linalg.norm(n)
    # a perpendicular seed that is never parallel to n
    seed = jnp.array([1.0, 0.0, 0.0])
    seed = jnp.where(jnp.abs(n[0]) > 0.9, jnp.array([0.0, 1.0, 0.0]), seed)
    e1 = seed - (seed @ n) * n
    e1 = e1 / jnp.linalg.norm(e1)
    e2 = jnp.cross(n, e1)
    return n, e1, e2


def cylindrical_metric(h, lam, reading="vacuum", axis=None):
    """The 4-metric in the chart (t, rho, phi, z) about `axis`, as functions of y = (rho, phi, z).

    This is a genuine change of coordinates, azimuth included: the field is evaluated at

        X(rho, phi, z) = rho (cos phi e1 + sin phi e2) + z n ,

    and its components are re-expressed in the chart's triad (rhohat, phihat, n),

        h_chart = D (B H B^T) D ,   B = [rhohat, phihat, n] ,  D = diag(1, rho, 1) ,

    with the D because the phi coordinate basis vector is rho * phihat, not phihat -- forgetting
    that D is the classic way to get this wrong.  Nothing here assumes axisymmetry: the returned
    functions have whatever phi dependence the field has, and the module differentiates in
    whatever coordinates it is handed.  (For an axisymmetric field the phi derivatives simply
    vanish.)

    WHY THIS CHART AT ALL.  Near the axis the Cartesian components of an axisymmetric metric are
    not smooth -- their second derivatives carry 1/rho^2, which no amount of autodiff can fix --
    while the chart components are.  This is the same device `weyl.curvature_invariant` uses for
    the exact solution's R_ab R^ab.
    """
    n, e1, e2 = axis_frame(axis)

    def X(y):
        rho, phi = y[0], y[1]
        return rho * (jnp.cos(phi) * e1 + jnp.sin(phi) * e2) + y[2] * n

    def h_chart(y):
        rho, phi = y[0], y[1]
        rh = jnp.cos(phi) * e1 + jnp.sin(phi) * e2
        ph = -jnp.sin(phi) * e1 + jnp.cos(phi) * e2
        B = jnp.stack([rh, ph, n])
        M = B @ h(X(y)) @ B.T
        D = jnp.diag(jnp.array([1.0, rho, 1.0]))
        return D @ M @ D

    def lam_chart(y):
        return lam(X(y))

    return h_chart, lam_chart


def curvature_at_cylindrical(h, lam, x, reading="vacuum", axis=None, tetrad=True,
                             frame="orthonormal"):
    """`curvature_at` at the physical point x, evaluated in the cylindrical chart about `axis`.

    This exists because the Cartesian components of an axisymmetric metric have
    direction-dependent second derivatives at rho_cyl = 0: the invariants computed from them
    blow up (2.9e+08 for the Kretschmann of a solution whose true value there is ~1), even
    though the components themselves are finite.  In the chart (rho, phi, z) the components are
    smooth, the connection's explicit 1/rho factors cancel in the scalars, and the axis comes
    out at its true value.  `weyl.curvature_invariant` uses the same device for R_ab R^ab.

    The contraction is done in the orthonormal frame of the chart by default, which is what
    makes the four-index scalars usable close to the axis (see `curvature_at`).

    Accuracy: fine down to rho ~ 1e-3 of the rod scale, and no further -- the 1/rho^2 loss
    happens when the coordinate Christoffel symbols are formed, before any frame change can help.
    Evaluate at a floored rho and mask below that rather than trusting the last decade.
    """
    n, _, _ = axis_frame(axis)
    x = jnp.asarray(x, jnp.float64)
    rho = jnp.linalg.norm(x - (x @ n) * n)
    z = x @ n
    hc, lc = cylindrical_metric(h, lam, reading, axis)
    return curvature_at(hc, lc, jnp.array([rho, 0.0, z]), reading, tetrad, frame=frame)


# the old name: the azimuth is now included, so the field need not be axisymmetric
curvature_at_axisym = curvature_at_cylindrical


# ------------------------------------------------------- the exact Weyl solution, in the chart
def weyl_rods(cfg):
    """The rods of a Weyl run (local import: geometry must not depend on weyl)."""
    from .weyl import Rods
    return Rods.pair(cfg.weyl_half_length, cfg.weyl_half_length_b, cfg.weyl_half_gap)


def weyl_horizon_table(cfg):
    """Each rod is a horizon: (mass, (z1, z2), area, surface gravity) -- all exact.

    A rod of mass m is a black hole of mass m, so its area is 16 pi m^2 and its surface gravity
    1/(4m).  The spans are NOT symmetric about z = 0 when the masses differ, which is why they
    are returned rather than reconstructed from the masses.
    """
    return [(float(m), (float(a), float(b)), float(16.0 * jnp.pi * m**2), float(1.0 / (4.0 * m)))
            for m, (a, b) in zip(weyl_rods(cfg).masses, weyl_rods(cfg).spans)]


def weyl_chart_fields(cfg, n_quad: int = 200):
    """(h, lam) of the exact Weyl solution in the chart (rho, phi, z) about the RODS' axis.

        h = diag(A, rho^2, A) with A = e^{2k},        lambda = e^{2U}

    which is the same space-time as `train.exact_asset` -- and cheap, smooth at the axis, and
    valid for a ROTATED configuration too, because rotating the rods only relabels which
    direction the axis points (the chart functions are the unrotated ones).  This is the form
    the curvature panels and the report need; `h_cart` is the Cartesian form, which is what the
    residual machinery and the sphere integrals need.
    """
    from .weyl import U_of, k_of
    rods = weyl_rods(cfg)

    def h_chart(y):
        A = jnp.exp(2.0 * k_of(y[0], y[2], rods, n_quad))
        return jnp.diag(jnp.array([A, y[0] ** 2, A]))

    def lam_chart(y):
        return jnp.exp(2.0 * U_of(y[0], y[2], rods))

    return h_chart, lam_chart


# ------------------------------------------------------------- report summary
def _fibonacci_directions(n_dir):
    """A deterministic, reasonably uniform set of directions on the sphere."""
    i = jnp.arange(n_dir) + 0.5
    mu = 1.0 - 2.0 * i / n_dir
    ph = jnp.pi * (1.0 + 5.0**0.5) * i
    st = jnp.sqrt(jnp.maximum(1.0 - mu**2, 0.0))
    return jnp.stack([st * jnp.cos(ph), st * jnp.sin(ph), mu], axis=-1)


def _median(vals):
    v = sorted(vals)
    if not v:
        return float("nan")
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def geometry_report(h, lam, reference=None, rho_lo=1.0, rho_hi=1.0, reading="vacuum",
                    n_dir=24, fractions=(0.40, 0.70, 0.95), n_mu=16, n_phi=8, ref_stride=1,
                    hawking_fractions=(0.5, 0.75, 0.95)):
    """Flat, JSON-ready summary of the gauge-invariant geometry (README section 11).

    Robust reductions (medians, maxima) over a Fibonacci set of directions at a few radii of
    the shell.  Nothing here assumes a symmetry, so it applies to the 3-D runs as well as to
    the axisymmetric ones.  Points where the chart cannot deliver a curvature -- exactly on the
    axis, where the Cartesian components have direction-dependent second derivatives -- come
    back NaN and are DROPPED; `geom_n_used` says how many survived, so a small number is a
    warning that the sample is not representative.

    `reference` is an optional second `(h, lam)`, normally `train.exact_asset(cfg)`.  With it,
    the Kretschmann difference becomes a gauge-invariant error and the Hawking mass can be
    compared like for like.

    Two combinations are the informative ones.  `vacuum_defect = |R_ab| / sqrt(|K|)` is
    dimensionless and vanishes for an exact vacuum solution, so it measures how far a solution
    of the residual system is from the *physical* reading.  `rank1_defect / R^2` is the same
    kind of number for the residual system itself: it vanishes whenever `R_ij = (1/2) phi_i
    phi_j` holds, and it needs no reference solution at all.
    """
    dirs = _fibonacci_directions(n_dir)
    radii = [rho_lo + f * (rho_hi - rho_lo) for f in fractions]
    ref = reference if (reference and reference[0] is not None) else None
    vac, cc, sdev, types, r1, kerr, eig = [], [], [], [], [], [], []
    n_used = 0
    k_pt = 0
    for r in radii:
        for nn in dirs:
            x = r * nn
            c = curvature_at(h, lam, x, reading)
            K = float(c["K"])
            if math.isnan(K):                       # on the axis: chart not usable
                continue
            n_used += 1
            vac.append(float(jnp.linalg.norm(c["Ric"])) / math.sqrt(abs(K)) if K else float("nan"))
            cc.append(abs(float(c["CdotC"])))
            sdev.append(abs(complex(c["S"]) - 1.0))
            types.append(c["petrov"])
            # The residual system's own identities, from ONE more evaluation: in the scalar
            # reading the 4-d spatial block IS R_ij(h) and R is R(h).  They are TWO tests:
            #   rank-one defect    R_ij R^ij - R^2 = 0    constrains h alone (at most one
            #                      eigenvalue of R_ij can be non-zero);
            #   eigenvalue defect  R - (1/2)|grad phi|^2 = 0   is the one a wrong lambda breaks,
            #                      so a good report needs both.
            c2 = curvature_at(h, lam, x, "scalar", tetrad=False)
            R3 = float(c2["R"])
            Ric3 = c2["Ric"][1:, 1:]
            Hinv = jnp.linalg.inv(h(x))
            RijRij = float(jnp.einsum("bd,be,de->", Ric3, Ric3, Hinv))
            r1.append(abs(RijRij - R3**2) / R3**2 if R3 else float("nan"))
            dphi = jax.jacfwd(lambda y: jnp.log(lam(y)))(x)
            grad2 = float(jnp.einsum("ij,i,j->", Hinv, dphi, dphi))
            eig.append(abs(R3 - 0.5 * grad2) / abs(R3) if R3 else float("nan"))
            if ref is not None and k_pt % max(1, int(ref_stride)) == 0:
                Ke = float(curvature_at(ref[0], ref[1], x, reading)["K"])
                if not math.isnan(Ke) and Ke:
                    kerr.append(abs(K / Ke - 1.0))
            k_pt += 1
    out = dict(geom_reading=reading, geom_n_used=n_used, geom_n_points=len(radii) * n_dir,
               geom_vacuum_defect_median=_median(vac), geom_vacuum_defect_max=max(vac, default=float("nan")),
               geom_pontryagin_max=max(cc, default=float("nan")),
               geom_speciality_dev_median=_median(sdev),
               geom_type_D_fraction=(types.count("D") / len(types)) if types else float("nan"),
               geom_rank1_defect_median=_median(r1),
               geom_eigenvalue_defect_median=_median(eig))
    if kerr:
        out.update(geom_kretschmann_error_median=_median(kerr),
                   geom_kretschmann_error_max=max(kerr))
    # the Hawking mass profile: three spheres for the network, and -- because a reference
    # sphere costs ~2.5x a network one -- the exact profile at the OUTER radius only, which is
    # where the comparison against the total mass is made anyway
    hawking = []
    for f in hawking_fractions:
        r = rho_lo + f * (rho_hi - rho_lo)
        sc = sphere_geometry(h, lam, r, reading, n_mu=n_mu, n_phi=n_phi)
        ra, lm = float(sc["r_areal"]), float(sc["lam_mean"])
        hawking.append([r, float(sc["hawking_mass"]), ra, 0.5 * ra * (1.0 - lm)])
    out["geom_spheres"] = hawking
    if ref is not None:
        r = hawking[-1][0]
        sce = sphere_geometry(ref[0], ref[1], r, reading, n_mu=n_mu, n_phi=n_phi)
        out["geom_m_h_ref_outer"] = float(sce["hawking_mass"])
        out["geom_r_areal_ref_outer"] = float(sce["r_areal"])
    return out


def format_geometry(out, indent="    "):
    """The lines a report prints for a `geometry_report` dict (kept next to the numbers)."""
    lines = []
    lines.append(f"{indent}reading {out['geom_reading']}: {out['geom_n_used']} of "
                 f"{out['geom_n_points']} sample points off the axis")
    lines.append(f"{indent}vacuum defect |R_ab|/sqrt|K|  median {out['geom_vacuum_defect_median']:.3e}"
                 f"   max {out['geom_vacuum_defect_max']:.3e}      (0 for exact vacuum)")
    lines.append(f"{indent}rank-one defect / R^2       median {out['geom_rank1_defect_median']:.3e}"
                 f"                  (0 for a solution of the system)")
    lines.append(f"{indent}eigenvalue defect / |R|     median {out['geom_eigenvalue_defect_median']:.3e}"
                 f"                  (0 same; this one needs lambda too)")
    lines.append(f"{indent}Pontryagin |C.C~| max       {out['geom_pontryagin_max']:.3e}"
                 f"                        (0: static)")
    lines.append(f"{indent}speciality |S-1| median     {out['geom_speciality_dev_median']:.3e}"
                 f"   type D at {100 * out['geom_type_D_fraction']:.0f}% of the points")
    if "geom_kretschmann_error_median" in out:
        lines.append(f"{indent}Kretschmann vs reference    median {out['geom_kretschmann_error_median']:.3e}"
                     f"   max {out['geom_kretschmann_error_max']:.3e}   (gauge invariant error)")
    lines.append(f"{indent}Hawking mass profile  (m_H flat at the total mass = the source is "
                 f"inside; m_lapse = r_a(1-lambda)/2 converges only as rho -> infinity)")
    lines.append(f"{indent}{'rho':>8} {'m_Hawking':>12} {'r_areal':>10} {'M_lapse':>10}"
                 + (f"   {'m_H(ref)':>12}" if "geom_m_h_ref_outer" in out else ""))
    for row in out["geom_spheres"]:
        s = f"{indent}{row[0]:8.4f} {row[1]:12.5e} {row[2]:10.5f} {row[3]:10.5f}"
        if "geom_m_h_ref_outer" in out and row is out["geom_spheres"][-1]:
            s += f"   {out['geom_m_h_ref_outer']:12.5e}"
        lines.append(s)
    return lines
