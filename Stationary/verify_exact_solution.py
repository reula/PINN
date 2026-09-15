"""Verification of the exact spherically symmetric solution of

    Ricci(h)_ab = (1/(2 lambda^2)) grad_a lambda grad_b lambda
    h^{ab} grad_a grad_b lambda = (1/lambda) h^{ab} grad_a lambda grad_b lambda
    gauge:  Gamma^a_bc h^{bc} = 0   (harmonic coordinates)

Claimed exact solution in the HARMONIC chart x^i = rho n^i:
    h_ij = (1 - R0^2/rho^2) delta_ij + (R0^2/rho^2) n_i n_j
    lambda = k (rho - R0)/(rho + R0)
i.e.  h = d rho^2 + (rho^2 - R0^2) dOmega^2 :  the coordinate functions x^i are harmonic.

INDEX CONVENTIONS (checked in test 0 below -- do not change casually):
    J = jacfwd(h)(x)          J[i,j,a]  = d_a h_ij        (input axis LAST)
    G = christoffel(h,x)      G[i,j,k]  = Gamma^i_{jk}
    H = h(x)                  Hinv[i,j] = h^{ij}
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jacfwd

I3 = jnp.eye(3)


# --------------------------------------------------------------------------- fields
def make_fields(R0, k):
    def h(x):
        r2 = jnp.dot(x, x)
        n = x / jnp.sqrt(r2)
        c = R0**2 / r2
        return (1.0 - c) * I3 + c * jnp.outer(n, n)

    def lam(x):
        r = jnp.sqrt(jnp.dot(x, x))
        return k * (r - R0) / (r + R0)

    return h, lam


def christoffel(h, x):
    """G[i,j,k] = Gamma^i_{jk} = 1/2 h^{il}(d_j h_lk + d_k h_lj - d_l h_jk)."""
    J = jacfwd(h)(x)  # J[i,j,a] = d_a h_ij
    hinv = jnp.linalg.inv(h(x))
    T = J.transpose(0, 2, 1) + J - J.transpose(2, 1, 0)  # T[l,j,k]
    return 0.5 * jnp.einsum("il,ljk->ijk", hinv, T)


def ricci(h, x):
    """R_ij = d_k G^k_ij - d_i G^k_kj + G^k_kl G^l_ij - G^k_il G^l_kj."""
    G = christoffel(h, x)
    dG = jacfwd(lambda y: christoffel(h, y))(x)  # dG[i,j,k,a] = d_a Gamma^i_{jk}
    R = jnp.einsum("kijk->ij", dG) - jnp.einsum("kkji->ij", dG)
    R = R + jnp.einsum("kkl,lij->ij", G, G) - jnp.einsum("kil,lkj->ij", G, G)
    return R, G


def laplacian(h, f, x):
    """Delta_h f = (1/sqrt h) d_a ( sqrt h  h^{ab} d_b f ), independent route."""
    def V(y):
        H = h(y)
        return jnp.sqrt(jnp.linalg.det(H)) * jnp.einsum("ab,b->a", jnp.linalg.inv(H), jacfwd(f)(y))

    return jnp.trace(jacfwd(V)(x)) / jnp.sqrt(jnp.linalg.det(h(x)))


# --------------------------------------------------------------------------- residuals
def residuals(x, R0, k):
    h, lam = make_fields(R0, k)
    H = h(x)
    hinv = jnp.linalg.inv(H)
    R, G = ricci(h, x)

    lam_v = lam(x)
    dlam = jacfwd(lam)(x)
    d2lam = jacfwd(jacfwd(lam))(x)  # [a,b]

    ric_res = R - (1.0 / (2.0 * lam_v**2)) * jnp.outer(dlam, dlam)
    hess = d2lam - jnp.einsum("cij,c->ij", G, dlam)
    lam_res = jnp.einsum("ij,ij->", hinv, hess) - (1.0 / lam_v) * jnp.einsum(
        "ij,i,j->", hinv, dlam, dlam)
    gauge_res = jnp.einsum("ijk,jk->i", G, hinv)

    J = jacfwd(h)(x)  # [b,c,a] = d_a h_bc
    compat = J.transpose(2, 0, 1) - jnp.einsum("dab,dc->abc", G, H) - jnp.einsum(
        "dac,bd->abc", G, H)

    return dict(ric=ric_res, lam=lam_res, gauge=gauge_res, compat=compat,
                H=H, hinv=hinv, G=G, dlam=dlam)


def show(name, arr):
    print(f"    {name:26s} max|.| = {float(jnp.max(jnp.abs(jnp.asarray(arr)))):.3e}")


def all_residuals(r):
    show("Ricci equation", r["ric"])
    show("lambda equation", r["lam"])
    show("harmonic gauge", r["gauge"])
    show("metric compatibility", r["compat"])


# --------------------------------------------------------------------------- tests
def test0_conventions():
    """Regression tests for the index conventions of christoffel/ricci."""
    print("=" * 78)
    print("0. Conventions regression tests")
    print("=" * 78)

    # (a) flat metric in POLAR chart: Gamma^theta must be -cos(theta)/(r^2 sin theta)
    def h_pol(y):
        r, th, _ = y
        f2 = r**2
        return jnp.diag(jnp.stack([jnp.ones_like(r), f2, f2 * jnp.sin(th) ** 2]))

    for y in [jnp.array([3.0, 0.7, 0.2]), jnp.array([10.0, 2.1, 1.3])]:
        r_, th_ = float(y[0]), float(y[1])
        G = christoffel(h_pol, y)
        gam = jnp.einsum("ijk,jk->i", G, jnp.linalg.inv(h_pol(y)))
        want = -jnp.cos(th_) / (r_**2 * jnp.sin(th_))
        print(f"  flat metric, polar chart r={r_:5.2f} th={th_:5.3f}:")
        print(f"    Gamma^theta = {float(gam[1]): .6e}   analytic -cos/(r^2 sin) = {float(want): .6e}")

    # (b) round S^3 of radius a via stereographic coords: R_ij = (2/a^2) h_ij
    #     ds^2 = (2a^2/(a^2+r^2))^2 |dx|^2
    a = 1.7

    def h_s3(x):
        r2 = jnp.dot(x, x)
        Om = 2 * a**2 / (a**2 + r2)
        return Om**2 * I3

    for x in [jnp.array([0.0, 0.0, 0.0]), jnp.array([0.4, -1.1, 0.9]), jnp.array([3.0, 0.5, -2.0])]:
        R, _ = ricci(h_s3, x)
        err = float(jnp.max(jnp.abs(R - (2.0 / a**2) * h_s3(x))))
        print(f"  round S^3 (radius {a}) in stereographic coords, x={[float(v) for v in x]}: "
              f"max|R_ij - (2/a^2)h_ij| = {err:.3e}")

    # (c) Ricci of flat space in polar chart must vanish
    R, _ = ricci(h_pol, jnp.array([3.0, 0.7, 0.2]))
    print(f"  flat polar Ricci: max|R_ij| = {float(jnp.max(jnp.abs(R))):.3e}")

    # (d) the two independent routes to Gamma^i = -Delta x^i must agree
    R0, k = 1.0, 0.7
    h, lam = make_fields(R0, k)
    xc = jnp.array([1.3, 0.6, -0.9])
    G = christoffel(h, xc)
    gam = jnp.einsum("ijk,jk->i", G, jnp.linalg.inv(h(xc)))
    lap = jnp.array([laplacian(h, lambda z: z[i], xc) for i in range(3)])
    print(f"  Gamma^i (Christoffel route) = {[float(v) for v in gam]}")
    print(f"  -Delta x^i (Laplace route)  = {[float(v) for v in -lap]}")
    print(f"  max discrepancy = {float(jnp.max(jnp.abs(gam + lap))):.3e}")


def test_flat():
    print()
    print("=" * 78)
    print("1. Flat branch R0=0 (h = delta, lambda = const)")
    print("=" * 78)
    for x in [jnp.array([1.3, -2.1, 3.7]), jnp.array([11.0, 0.0, -19.0])]:
        print(f"  x = {[float(v) for v in x]}")
        all_residuals(residuals(x, 0.0, 1.0))


def test_exact():
    print()
    print("=" * 78)
    print("2. Non-trivial branch R0=1, k=0.7  (rho > R0)")
    print("=" * 78)
    R0, k = 1.0, 0.7
    pts = [jnp.array([2.0, 0.0, 0.0]), jnp.array([0.0, 2.0, 0.0]), jnp.array([0.0, 0.0, 5.0]),
           jnp.array([1.0, 2.0, 3.0]), jnp.array([-7.0, 2.5, -11.0]),
           jnp.array([19.7, -3.0, 1.5])]
    for p in pts:
        rho = float(jnp.sqrt(jnp.dot(p, p)))
        print(f"  x = {[round(float(v),2) for v in p]}, rho = {rho:.3f}, "
              f"areal radius = {float(jnp.sqrt(rho**2-R0**2)):.4f}, lambda = {float(k*(rho-R0)/(rho+R0)):.4f}")
        all_residuals(residuals(p, R0, k))


def test_structure():
    print()
    print("=" * 78)
    print("3. Structure of the exact solution")
    print("=" * 78)
    R0, k = 1.0, 0.7
    h, lam = make_fields(R0, k)
    for p in [jnp.array([1.0, 2.0, 3.0]), jnp.array([-7.0, 2.5, -11.0]), jnp.array([15.0, 4.0, -2.0])]:
        rho = float(jnp.sqrt(jnp.dot(p, p)))
        n = p / jnp.sqrt(jnp.dot(p, p))
        f2 = rho**2 - R0**2
        R, _ = ricci(h, p)
        c = float(jnp.einsum("ij,i,j->", R, n, n))
        tang = R - c * jnp.outer(n, n)
        print(f"  rho={rho:6.3f}: R_nn={c: .8e}  predicted 2R0^2/f^4={2*R0**2/f2**2: .8e}"
              f"   tangential part={float(jnp.max(jnp.abs(tang))):.3e}")
    print()
    print("  eigenvalues of h (must be positive for rho > R0):")
    for rho in [1.001, 2.0, 5.0, 20.0]:
        ev = jnp.linalg.eigvalsh(h(jnp.array([rho, 0.0, 0.0])))
        print(f"    rho={rho:6.3f}: {[round(float(v),6) for v in ev]}   "
              f"(areal radius {float(jnp.sqrt(max(rho**2-R0**2,0.0))):.4f})")

    print()
    print("  lambda -> 1/lambda is a symmetry of the system:")

    def lam_inv(x):
        r = jnp.sqrt(jnp.dot(x, x))
        return (r + R0) / (k * (r - R0))

    for p in [jnp.array([2.0, 0.0, 0.0]), jnp.array([1.0, 2.0, 3.0])]:
        H = h(p)
        hinv = jnp.linalg.inv(H)
        R, G = ricci(h, p)
        lv = lam_inv(p)
        dl = jacfwd(lam_inv)(p)
        d2l = jacfwd(jacfwd(lam_inv))(p)
        hess = d2l - jnp.einsum("cij,c->ij", G, dl)
        rr = R - (1.0 / (2 * lv**2)) * jnp.outer(dl, dl)
        lr = jnp.einsum("ij,ij->", hinv, hess) - (1.0 / lv) * jnp.einsum("ij,i,j->", hinv, dl, dl)
        print(f"    rho={float(jnp.sqrt(jnp.dot(p,p))):6.3f}: max|Ricci res|={float(jnp.max(jnp.abs(rr))):.3e}"
              f"  |lambda res|={float(jnp.abs(lr)):.3e}")

    print()
    print("  constant scaling h -> s^2 h leaves the system invariant (s=2.5):")
    s = 2.5
    hs = lambda x: s**2 * h(x)
    for p in [jnp.array([1.0, 2.0, 3.0])]:
        R, G = ricci(hs, p)
        lv = lam(p)
        dl = jacfwd(lam)(p)
        d2l = jacfwd(jacfwd(lam))(p)
        hinv_s = jnp.linalg.inv(hs(p))
        hess = d2l - jnp.einsum("cij,c->ij", G, dl)
        rr = R - (1.0 / (2 * lv**2)) * jnp.outer(dl, dl)
        lr = jnp.einsum("ij,ij->", hinv_s, hess) - (1.0 / lv) * jnp.einsum("ij,i,j->", hinv_s, dl, dl)
        print(f"    max|Ricci res|={float(jnp.max(jnp.abs(rr))):.3e}  |lambda res|={float(jnp.abs(lr)):.3e}")


def test_chart_trap():
    print()
    print("=" * 78)
    print("4. The chart trap: harmonic gauge is NOT usable in polar coordinates")
    print("=" * 78)
    R0, k = 1.0, 0.7

    def h_pol(y):
        r, th, _ = y
        f2 = r**2 - R0**2
        return jnp.diag(jnp.stack([jnp.ones_like(r), f2, f2 * jnp.sin(th) ** 2]))

    for y in [jnp.array([2.0, 1.0, 0.4]), jnp.array([9.0, 2.2, 1.1])]:
        G = christoffel(h_pol, y)
        gam = jnp.einsum("ijk,jk->i", G, jnp.linalg.inv(h_pol(y)))
        r_, th_ = float(y[0]), float(y[1])
        print(f"  exact solution in POLAR chart r={r_:5.2f} th={th_:5.3f}:")
        print(f"    Gamma^a = {[float(v) for v in gam]}")
        print(f"    (Gamma^theta should equal -cos/(h_thth sin) = "
              f"{float(-jnp.cos(th_)/((r_**2-R0**2)*jnp.sin(th_))): .6e} -- never zero)")
    print()
    print("  => any metric that is spherically symmetric IN POLAR COORDINATES has")
    print("     Gamma^theta = -cos(theta)/(h_theta_theta sin(theta)) != 0, so the")
    print("     harmonic gauge forces a Cartesian-like chart x^i = rho n^i.")


if __name__ == "__main__":
    test0_conventions()
    test_flat()
    test_exact()
    test_structure()
    test_chart_trap()
