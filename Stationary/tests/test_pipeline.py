"""End-to-end tests of the residual/loss code path.

The key idea: the loss machinery is fed the EXACT solution (packed exactly as the
network packs its outputs), so every residual and every boundary term must vanish
to machine precision.  This validates the whole pipeline independently of whether
the optimiser can converge.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import flax.linen as nn
import jax.numpy as jnp
import pytest

from stationary import exact
from stationary.geometry import (Fields, christoffel, gamma3, laplacian,
                                 pack_gamma, pack_sym, residuals_batch, sym3)
from stationary.losses import (inner_bc_terms, outer_bc_terms, pde_terms,
                               reference_consistency, total_loss)
from stationary.model import HybridNet, SymHybridNet
from stationary.problem import Config, sample_shell, sample_sphere

R0 = 1.0
# Every run sits on the branch with lambda -> 1; for R0 = 1 and the round sphere of areal
# radius 2 that means lambda_0 = 1/phi^2 (see Config.__post_init__ and exact.lambda0_from_k).
LAM0 = exact.lambda0_from_k(R0, 1.0, 2.0)
K = exact.k_from_lambda0(R0, LAM0)          # = 1


# --------------------------------------------------------------------- fixtures
class ExactModel(nn.Module):
    """Network-shaped module whose output IS the exact solution (no real params)."""
    R0: float = R0
    k: float = K

    @nn.compact
    def __call__(self, x):
        x = jnp.atleast_2d(x)
        dummy = self.param("dummy", nn.initializers.zeros, (1,))
        ef = exact.exact_fields(self.R0, self.k)
        f = jax.vmap(ef)(x)
        d = dummy.sum() * 0.0
        return Fields(f.h + d, f.G + d, f.lam + d)


def exact_point_fields():
    return exact.exact_fields(R0, K)


def cfg_m1() -> Config:
    # round metric of areal radius 2 on the inner sphere (which sits at rho = sqrt5)
    c = Config(R0=R0, lam0=LAM0, inner_radius=2.0, robin_order=1)
    return c


# ------------------------------------------------------------------------ tests
def test_conventions_flat_polar():
    """Gamma^theta of the flat metric in polar coordinates = -cos(theta)/(r^2 sin)."""
    def h_pol(y):
        r, th, _ = y
        return jnp.diag(jnp.stack([jnp.ones_like(r), r**2, r**2 * jnp.sin(th) ** 2]))

    y = jnp.array([3.0, 0.7, 0.2])
    G = christoffel(h_pol, y)
    gam = jnp.einsum("ijk,jk->i", G, jnp.linalg.inv(h_pol(y)))
    want = -jnp.cos(0.7) / (9.0 * jnp.sin(0.7))
    assert jnp.allclose(gam[1], want, atol=1e-12)


def test_conventions_round_s3():
    """Ricci of the round S^3 of radius a is (2/a^2) h."""
    from stationary.geometry import ricci_from_gamma
    a = 1.7

    def h_s3(x):
        r2 = jnp.dot(x, x)
        om = 2 * a**2 / (a**2 + r2)
        return om**2 * jnp.eye(3)

    x = jnp.array([0.4, -1.1, 0.9])
    G = christoffel(h_s3, x)
    dG = jax.jacfwd(lambda y: christoffel(h_s3, y))(x)
    R = ricci_from_gamma(G, dG)
    assert jnp.max(jnp.abs(R - (2.0 / a**2) * h_s3(x))) < 1e-12


def test_gamma_equals_minus_laplacian():
    """Two independent routes to Gamma^i = -Delta_h x^i must agree."""
    h = exact.exact_metric(R0)
    x = jnp.array([1.3, 0.6, -0.9])
    G = christoffel(h, x)
    gam = jnp.einsum("ijk,jk->i", G, jnp.linalg.inv(h(x)))
    lap = jnp.array([laplacian(h, lambda z: z[i], x) for i in range(3)])
    assert jnp.max(jnp.abs(gam + lap)) < 1e-12


@pytest.mark.parametrize("R0v,kv", [(0.0, 1.0), (1.0, K), (0.7, 2.0), (2.5, -1.3)])
def test_exact_residuals(R0v, kv):
    """Exact solution: all four residual groups vanish."""
    pf = exact.exact_fields(R0v, kv)
    xs = jnp.array([[2.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-7.0, 2.5, -11.0], [0.0, 0.0, 19.0]])
    if R0v > 0:
        xs = xs * jnp.maximum(1.0, 1.05 * R0v / jnp.linalg.norm(xs, axis=-1, keepdims=True))
    r = residuals_batch(pf, xs)
    for k, v in r.items():
        assert jnp.max(jnp.abs(v)) < 1e-11, (k, float(jnp.max(jnp.abs(v))))


def test_packing_roundtrip():
    pf = exact.exact_fields(R0, K)
    x = jnp.array([1.0, 2.0, 3.0])
    f = pf(x)
    assert jnp.allclose(sym3(pack_sym(f.h)), f.h)
    assert jnp.allclose(gamma3(pack_gamma(f.G)), f.G)


def test_exact_satisfies_inner_bc():
    """Areal radius 2, h_rr = 1 and lambda = lambda_0 hold for the exact solution."""
    cfg = cfg_m1()
    pf = exact_point_fields()
    xs = sample_sphere(jax.random.PRNGKey(0), 128, cfg.rho_in)
    terms = inner_bc_terms(pf, xs, cfg)
    for k, v in terms.items():
        assert float(v) < 1e-20, (k, float(v))


def test_exact_satisfies_outer_dirichlet_bc():
    cfg = cfg_m1()
    pf = exact_point_fields()
    xs = sample_sphere(jax.random.PRNGKey(1), 128, cfg.rho_out)
    terms = outer_bc_terms(pf, xs, cfg, exact_fields=pf)
    for k, v in terms.items():
        assert float(v) < 1e-24, (k, float(v))


def test_loss_is_zero_on_exact_solution():
    """Full loss (PDE + both BCs) evaluated on the exact solution."""
    cfg = cfg_m1()
    model = ExactModel()
    key = jax.random.PRNGKey(2)
    params = model.init(key, jnp.ones((1, 3)))
    state = {"net": params}
    batch = {"coll": sample_shell(key, 512, cfg),
             "inner": sample_sphere(key, 64, cfg.rho_in),
             "outer": sample_sphere(key, 64, cfg.rho_out)}
    loss, parts = total_loss(state, batch, cfg, model, exact_fields=exact_point_fields())
    assert float(loss) < 1e-16, (float(loss), {k: float(v) for k, v in parts.items()})


def test_inner_radius_default_is_the_areal_radius_2():
    """The default inner sphere is the round sphere of areal radius 2, NOT rho_in.

    rho_in_of_R0(R0) = sqrt(4+R0^2) is by definition the coordinate radius at which the
    canonical-chart exact solution has areal radius 2, so its tangential metric there is
    1 - R0^2/rho_in^2 = 4/(4+R0^2) (0.8 for R0 = 1), not flat.  Defaulting inner_radius to
    rho_in therefore asks for a flat tangential metric on the inner sphere, which the
    exact solution violates by 20%; with the Dirichlet outer data taken from that same
    solution the two boundary conditions contradict each other and the loss cannot reach
    zero.  That was the behaviour between commit 67a802a and the fix, and it is invisible
    in every run whose rho_in happens to equal 2, which is why it survived so long.
    """
    cfg = Config(R0=R0, lam0=LAM0)                     # everything defaulted
    assert cfg.inner_radius == 2.0
    assert abs(cfg.rho_in - float(jnp.sqrt(4.0 + R0**2))) < 1e-12
    chk = reference_consistency(exact_point_fields(), cfg)
    assert max(chk.values()) < 1e-14, chk

    # the wrong value must be *detected*, not quietly absorbed
    bad = Config(R0=R0, lam0=LAM0, inner_radius=float(cfg.rho_in))
    assert reference_consistency(exact_point_fields(), bad)["h_tan"] > 1e-4


def test_default_config_is_the_lambda_to_1_branch():
    """lambda_0 is derived so that lambda -> 1, and h_rr is not constrained.

    `lambda -> c lambda` is an exact symmetry, so the asymptotic value is a free
    normalisation; every run uses k = 1, which makes lambda_0 a function of the geometry:
    1/phi^2 = 0.381966 for R0 = 1 with the areal-radius-2 sphere, 1/3 for the M2 control
    geometry (R0 = 1/sqrt3, areal radius 1).  The second half of the test is the CLI's
    build-then-override pattern: Config() first, --R0/--inner-radius afterwards, which
    must still end up with k = 1.
    """
    c = Config()
    assert c.inner_radius == 2.0
    assert c.inner_h_rr is None                       # free: imposing it over-determines the gauge
    assert abs(c.lam0 - exact.lambda0_from_k(1.0, 1.0, 2.0)) < 1e-15     # = 1/phi^2 = 0.381966
    assert abs(exact.k_from_lambda0(c.R0, c.lam0, r_areal=c.inner_radius) - 1.0) < 1e-12

    c2 = Config()                                     # the CLI pattern
    c2.R0 = 1.0 / jnp.sqrt(3.0)
    c2.rho_in = 1.0
    c2.inner_radius = 1.0
    c2.__post_init__()
    assert abs(c2.lam0 - 1.0 / 3.0) < 1e-12
    assert abs(exact.k_from_lambda0(c2.R0, c2.lam0, r_areal=c2.inner_radius) - 1.0) < 1e-12

    c3 = Config(lam0=1.0)                             # an explicit value still wins
    assert c3.lam0 == 1.0 and c3.lam0_auto is False
    assert abs(exact.k_from_lambda0(c3.R0, c3.lam0, r_areal=c3.inner_radius) - 2.6180340) < 1e-6


def test_robin_terms_are_finite_and_shaped():
    """The Robin outer BC runs and gives the expected number of residual entries."""
    cfg = cfg_m1()
    cfg.outer_bc = "robin"
    cfg.lam_inf = K
    pf = exact_point_fields()
    xs = sample_sphere(jax.random.PRNGKey(3), 32, cfg.rho_out)
    terms = outer_bc_terms(pf, xs, cfg, lam_inf=K)
    assert set(terms) == {"h", "G", "lam"}
    for v in terms.values():
        assert jnp.isfinite(v)


@pytest.mark.parametrize("c2", [-0.5, -0.2, 0.2])
def test_harmonic_chart_freedom(c2):
    """The same solution in a different harmonic chart still solves every equation.

    This checks both the residual gauge freedom and the chart utility: the metric
    is non-trivially rewritten, yet residuals and gauge stay zero.
    """
    rho_in_geom = exact.rho_in(R0)          # geometric sphere of areal radius 2
    # choose c1 so that F(rho_in_geom) = 2  (inner sphere at coordinate radius 2)
    F2v = float(exact.F2(R0, rho_in_geom))
    c1 = (2.0 - c2 * F2v) / rho_in_geom
    pf = exact.exact_fields_in_harmonic_chart(R0, K, c1=c1, c2=c2)
    xs = jnp.array([[2.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-7.0, 2.5, -11.0], [3.0, -2.0, 18.0]])
    r = residuals_batch(pf, xs)
    for k, v in r.items():
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(v)))
        assert jnp.max(jnp.abs(v)) < 1e-8 * scale, k

    # and the inner sphere really is at coordinate radius 2 carrying areal radius 2
    cfg = Config(R0=R0, lam0=LAM0, rho_in=2.0)
    terms = inner_bc_terms(pf, sample_sphere(jax.random.PRNGKey(4), 64, 2.0), cfg)
    assert float(terms["h_tan"]) < 1e-8


class ExactHybridModel(HybridNet):
    """Exact solution through the hybrid (metric-only) code path."""
    R0: float = R0
    k: float = K

    @nn.compact
    def __call__(self, x):
        x = jnp.atleast_2d(x)
        dummy = self.param("dummy", nn.initializers.zeros, (1,))
        ef = exact.exact_fields(self.R0, self.k)
        f = jax.vmap(ef)(x)
        d = dummy.sum() * 0.0
        return Fields(f.h + d, jnp.zeros(x.shape[:-1] + (3, 3, 3)), f.lam + d)


def test_hybrid_path_gives_zero_loss_on_exact_solution():
    """Gamma is derived from h: compatibility is automatic and the loss must vanish."""
    from stationary.model import HybridNet, SymHybridNet, point_fields as make_pf
    cfg = cfg_m1()
    model = ExactHybridModel()
    key = jax.random.PRNGKey(11)
    params = model.init(key, jnp.ones((1, 3)))
    state = {"net": params}
    batch = {"coll": sample_shell(key, 512, cfg),
             "inner": sample_sphere(key, 64, cfg.rho_in),
             "outer": sample_sphere(key, 64, cfg.rho_out)}
    loss, parts = total_loss(state, batch, cfg, model, exact_fields=exact_point_fields())
    assert float(loss) < 1e-16, (float(loss), {k: float(v) for k, v in parts.items()})
    # and the residuals themselves, on a random point set
    pf = make_pf(model, params)
    r = residuals_batch(pf, sample_shell(key, 256, cfg))
    for k, v in r.items():
        assert float(jnp.max(jnp.abs(v))) < 1e-11, k


class ExactSymHybridModel(SymHybridNet):
    """Exact solution through the symmetric metric-only (Gamma derived) path."""
    R0: float = R0
    k: float = K

    @nn.compact
    def __call__(self, x):
        x = jnp.atleast_2d(x)
        dummy = self.param("dummy", nn.initializers.zeros, (1,))
        ef = exact.exact_fields(self.R0, self.k)
        f = jax.vmap(ef)(x)
        d = dummy.sum() * 0.0
        return Fields(f.h + d, jnp.zeros(x.shape[:-1] + (3, 3, 3)), f.lam + d)


def test_sym_hybrid_zero_loss_on_exact_solution():
    """Symmetric metric-only path: compat is automatic, whole loss must vanish."""
    from stationary.model import SymHybridNet, point_fields as make_pf
    cfg = cfg_m1()
    model = ExactSymHybridModel()
    key = jax.random.PRNGKey(21)
    params = model.init(key, jnp.ones((1, 3)))
    batch = {"coll": sample_shell(key, 512, cfg),
             "inner": sample_sphere(key, 64, cfg.rho_in),
             "outer": sample_sphere(key, 64, cfg.rho_out)}
    loss, parts = total_loss({"net": params}, batch, cfg, model,
                             exact_fields=exact_point_fields())
    assert float(loss) < 1e-18, (float(loss), {k: float(v) for k, v in parts.items()})
    pf = make_pf(model, params)
    r = residuals_batch(pf, sample_shell(key, 256, cfg))
    for k, v in r.items():
        assert float(jnp.max(jnp.abs(v))) < 1e-11, (k, float(jnp.max(jnp.abs(v))))


def test_sym_hybrid_manufactured_robin_is_exact():
    """Milestone-2 setup with the manufactured source: exact solution hits loss 0."""
    ref, info = exact.reference_fields_asymptotic(1.0, 1.0, 2.0)
    lam_inner = float(ref(jnp.array([2.0, 0.0, 0.0])).lam)   # = 0.381966 for k = 1
    cfg = Config(R0=1.0, lam0=lam_inner, rho_in=2.0, rho_out=100.0, outer_bc="robin",
                 inner_h_rr=None, robin_source=True, ref_asymptotic=1.0)
    cfg.__post_init__()
    cfg.lam_inf = 1.0
    model = ExactSymHybridModel()
    key = jax.random.PRNGKey(22)
    params = model.init(key, jnp.ones((1, 3)))

    class RefModel(SymHybridNet):
        @nn.compact
        def __call__(self, x):
            x = jnp.atleast_2d(x)
            d = self.param("dummy", nn.initializers.zeros, (1,)) * 0.0
            f = jax.vmap(ref)(x)
            return Fields(f.h + d, jnp.zeros(x.shape[:-1] + (3, 3, 3)), f.lam + d)

    rm = RefModel()
    rp = rm.init(key, jnp.ones((1, 3)))
    batch = {"coll": sample_shell(key, 512, cfg),
             "inner": sample_sphere(key, 64, cfg.rho_in),
             "outer": sample_sphere(key, 64, cfg.rho_out)}
    for model_i, state in ((rm, {"net": rp}), (RefModel(), {"net": rp})):
        loss, parts = total_loss(state, batch, cfg, model_i, exact_fields=ref, lam_inf=1.0)
        assert float(loss) < 1e-16, (float(loss), {k: float(v) for k, v in parts.items()})


# ------------------------------------------------------- new-problem tests
def test_robin_operator_annihilates_powers():
    """(rho d_rho + a) kills rho^-a; the product kills the first `order` powers only."""
    from stationary.losses import robin_operator
    for base in (1, 2, 3):
        for kk in (base, base + 1, base + 2, base + 3):
            for order in (1, 2, 3):
                f = lambda y, kk=kk: jnp.linalg.norm(y) ** (-kk)
                x = jnp.array([3.0, -4.0, 12.0])          # |x| = 13
                val = robin_operator(f, x, base, order)
                killed = base <= kk <= base + order - 1
                if killed:
                    assert abs(float(val)) < 1e-10 * 13.0 ** (-kk), (base, kk, order, float(val))
                else:
                    assert abs(float(val)) > 1e-12, (base, kk, order, float(val))


def _radial_derivatives(f, x, order):
    """d_rho^j f at x for j = 0..order, by differentiating along the ray through x."""
    n = x / jnp.linalg.norm(x)
    g = lambda t: f(x + t * n)
    vals, cur = [g(0.0)], g
    for _ in range(order):
        cur = jax.jacfwd(cur)
        vals.append(cur(0.0))
    return vals


def test_robin_second_order_lambda_form():
    """For lambda with base 1 and order 2 the condition is rho^2 lam'' + 4 rho lam' + 2(lam-1)."""
    from stationary.losses import robin_operator
    lam = lambda y: 1.0 + 2.0 / jnp.linalg.norm(y)
    x = jnp.array([0.0, 0.0, 7.0])
    assert abs(float(robin_operator(lam, x, 1, 2, 1.0))) < 1e-12     # 1/rho is killed
    lam2 = lambda y: 1.0 + 2.0 / jnp.linalg.norm(y) ** 3
    # rho^-3 must survive (it is the next multipole, l = 2)
    assert abs(float(robin_operator(lam2, x, 1, 2, 1.0))) > 1e-6

    # and the expansion itself, on a field that is not a single power
    g = lambda y: 1.0 + 0.4 / jnp.linalg.norm(y) + 0.9 * y[2] / jnp.linalg.norm(y) ** 3
    f0, f1, f2 = _radial_derivatives(g, x, 2)
    want = 7.0**2 * f2 + 4 * 7.0 * f1 + 2 * (f0 - 1.0)
    assert abs(float(robin_operator(g, x, 1, 2, 1.0)) - float(want)) < 1e-12


def test_robin_operator_expansions_match_the_documented_forms():
    """The explicit ODEs quoted in losses.py, problem.py and HUB.md are the implemented ones.

    The operator is built in `theta = rho d_rho` (where the factors commute, so the product
    annihilates a window of powers); the *expansions* in ordinary radial derivatives that the
    docstrings quote are a separate statement, and until this test nothing checked them --
    a formula in a comment could drift away from the code and no test would notice.

    The field below is three decaying harmonics plus an exp term, so no accident of the
    power-law basis can hide a wrong coefficient.  With psi = field - inf_val and
    rho = |x| = 7:

        n=1, base 1:  rho psi' + psi
        n=2, base 1:  rho^2 psi'' + 4 rho psi' + 2 psi
        n=3, base 1:  rho^3 psi''' + 9 rho^2 psi'' + 18 rho psi' + 6 psi
        n=4, base 1:  rho^4 psi'''' + 16 rho^3 psi''' + 72 rho^2 psi'' + 96 rho psi' + 24 psi
        n=2, base 2:  rho^2 psi'' + 6 rho psi' + 6 psi
        n=4, base 2:  rho^4 psi'''' + 20 rho^3 psi''' + 120 rho^2 psi'' + 240 rho psi' + 120 psi
        n=1, base 3:  rho psi' + 3 psi
        n=2, base 3:  rho^2 psi'' + 8 rho psi' + 12 psi

    The base-2 and base-3 rows cover the fields whose conditions production actually changes
    (`robin_exps h=2, G=3`; `--robin-orders h=4` is the base-2 order-4 row).

    For contrast, the form quoted in prompts.txt (`psi'' + 3 psi' + psi/rho`, the
    characteristic polynomial of theta^2 + 3 theta + 2 evaluated at d_rho instead of at
    theta) annihilates no power law at all: at rho = 7 it leaves 8.2e-2 on rho^-1 and 2.7e-2
    on rho^-2, and it fails rho^-2 even though that is the power an order-2 condition exists
    to admit.  Multiplying in theta is what makes the difference.
    """
    from stationary.losses import robin_operator

    x = jnp.array([2.0, 3.0, 6.0])                      # |x| = 7, not axis aligned
    rho = float(jnp.linalg.norm(x))

    def field(y):
        r = jnp.linalg.norm(y)
        return (1.0 + 0.3 / r + 0.7 * y[2] / r**3
                + 0.4 * (y[2] ** 2 - 0.5 * (y[0] ** 2 + y[1] ** 2)) / r**5
                + 0.05 * jnp.exp(-r / 4.0))

    f = _radial_derivatives(field, x, 5)
    p0 = f[0] - 1.0                                     # inf_val = 1
    cases = [
        (1.0, 1, rho * f[1] + p0),
        (1.0, 2, rho**2 * f[2] + 4 * rho * f[1] + 2 * p0),
        (1.0, 3, rho**3 * f[3] + 9 * rho**2 * f[2] + 18 * rho * f[1] + 6 * p0),
        (1.0, 4, rho**4 * f[4] + 16 * rho**3 * f[3] + 72 * rho**2 * f[2] + 96 * rho * f[1]
         + 24 * p0),
        # base 2 and base 3 are the h and Gamma fields (robin_exps), and h at order 4 is what
        # `--robin-orders h=4` runs in production: without these rows the formulas printed in
        # losses.py / problem.py / HUB.md for those fields had nothing checking them.
        (2.0, 2, rho**2 * f[2] + 6 * rho * f[1] + 6 * p0),
        (2.0, 4, rho**4 * f[4] + 20 * rho**3 * f[3] + 120 * rho**2 * f[2] + 240 * rho * f[1]
         + 120 * p0),
        (3.0, 1, rho * f[1] + 3 * p0),
        (3.0, 2, rho**2 * f[2] + 8 * rho * f[1] + 12 * p0),
    ]
    for base, order, want in cases:
        got = float(robin_operator(field, x, base, order, 1.0))
        assert abs(got - float(want)) <= 1e-10 * max(1.0, abs(float(want))), \
            (base, order, got, float(want))


def test_inner_lambda_bc_formula():
    """lam0 + S1 n_z + S2 (3 n_z^2 - 1)/2 on the inner sphere."""
    from stationary.problem import lam_inner_bc
    c = Config(R0=1.0, rho_in=1.0, lam0=1.0 / 3, lam_bc_S1=0.1, lam_bc_S2=0.25,
               inner_radius=1.0, robin_order=2)
    c.__post_init__()
    for th in (0.0, 0.7, 1.57, 2.6, 3.14159):
        x = jnp.array([jnp.sin(th), 0.0, jnp.cos(th)])
        want = 1.0 / 3 + 0.1 * jnp.cos(th) + 0.25 * (3 * jnp.cos(th) ** 2 - 1) / 2
        assert abs(float(lam_inner_bc(x, c)) - float(want)) < 1e-14


def test_multipole_roundtrip():
    from stationary.multipoles import decompose, real_sph_harm, sphere_grid
    mu, phi, w = sphere_grid(40, 32)
    truth = {(0, 0): 1.3, (1, 0): -0.25, (2, 0): 0.07, (2, 1): 0.03, (3, 0): -0.011}
    f = sum(v * real_sph_harm(l, m, mu, phi) for (l, m), v in truth.items())
    coef = decompose(f, mu, phi, w, 3)
    for k, v in coef.items():
        assert abs(v - truth.get(k, 0.0)) < 1e-12, (k, v, truth.get(k, 0.0))


def test_robin_fourth_order_kills_only_first_four_powers():
    """Order 4 must annihilate rho^-(base..base+3) and nothing beyond."""
    from stationary.losses import robin_operator
    x = jnp.array([6.0, 8.0, 24.0])                      # |x| = 26
    for base in (1, 2):
        for kk in range(base, base + 7):
            f = lambda y, kk=kk: jnp.linalg.norm(y) ** (-kk)
            val = float(robin_operator(f, x, base, 4))
            if kk < base + 4:
                assert abs(val) < 1e-12, (base, kk, val)
            else:
                assert abs(val) > 1e-14, (base, kk, val)
