"""Manufactured flat-Laplacian test of the higher-order Robin conditions.

Same shell and chart as the production problem -- two concentric spheres, Cartesian
coordinates, rho_in = 1 ... rho_out = 100 -- but with the Einstein system replaced by the
flat Laplacian, so the exact solution is known in closed form and nothing about the
optimiser's struggle with Ricci can hide a wrong boundary operator:

    Delta lam   = 0                                       rho_in < |x| < rho_out
    lam         = lam0 + S1 z + S2 (z^2 - (x^2+y^2)/2)     |x| = rho_in  = 1
    B_n[lam - lam0] = B_n[lam_exact - lam0]                |x| = rho_out = 100

with B_n = prod_{i<n} (rho d_rho + 1 + i) the order-n Euler product for a field whose
leading decay is 1/rho -- i.e. `losses.robin_operator` with base = 1 -- and the right
hand side the manufactured Robin source.

The exact solution is the decaying multipole sum

    lam_exact(x) = lam0 + S1 z/rho^3 + S2 (z^2 - (x^2+y^2)/2)/rho^5 .

It is harmonic: z/rho^3 and (z^2-(x^2+y^2)/2)/rho^5 are the l = 1 and l = 2 decaying
harmonics, i.e. combinations of derivatives of 1/rho.  It matches the prescribed inner
data because rho = 1 there (z/rho^3 = z and (z^2-(x^2+y^2)/2)/rho^5 = z^2-(x^2+y^2)/2 on
the unit sphere).  And it decays as rho^-2, rho^-3 with no rho^-1 part, so it lies exactly
in the window that B_3 annihilates:

    order 1  kills rho^-1 only          -> dipole AND quadrupole survive
    order 2  kills rho^-1, rho^-2       -> quadrupole survives
    order 3  kills all three            -> source identically zero
    base 2, order 2 (window rho^-2..3)  -> also annihilates it (its leading decay is
                                           rho^-2, not rho^-1)

That is what makes this a test OF the higher-order conditions: the *source* is what
discriminates the orders.  With the manufactured source the same exact solution satisfies
EVERY order by construction, so a solve at any order must reproduce it -- the solve checks
the operator and the source end to end, while the source magnitudes themselves check that
the order is the one that actually admits this solution.

Two things worth knowing when reading the assertions.

1. The ansatz is a *separable* basis (angular harmonics times radial powers) rather than a
   tanh MLP.  Over two decades in rho a plain MLP cannot represent a decaying multipole to
   better than ~1e-2 in max norm -- measured: a supervised fit of lam_exact to 256
   log-uniform points reaches MSE ~1e-7 while the max error on fresh points stays ~1e-2,
   and neither more width/depth nor L-BFGS improves it.  That floor is the same size as the
   solution's whole structure, so it would swamp the comparison.  The basis used here spans
   exactly what the data and the condition allow -- l <= 2 (the inner data is a degree-2
   polynomial) times rho^-1, rho^-2, rho^-3 (the window of B_3) -- which is the same
   "learn only the coefficients of a known tensor structure" device the production models
   use for the metric (h = alpha delta + beta n n).  It still contains spurious modes the
   exact solution does not use (the rho^-1 monopole, rho^-1 and rho^-2 dipoles, ...), so
   it is the conditions, not the ansatz, that select the solution.

2. At rho_out = 100 the solution's out-of-window content is rho^-2 ~ 1e-4 and
   rho^-3 ~ 1e-6, so an order-1 condition that kept them would still be satisfied to ~1e-5
   by this solution and the *solution* of the BVP differs from lam_exact by only ~1e-6
   between orders.  The order discrimination is therefore sharp in the residual (which is
   exact, see the source tests below) and numerically invisible in the solved field at
   this radius -- which is why no test here tries to separate orders by solving.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax
import pytest

from stationary.losses import robin_operator
from stationary.problem import Config, lam_inner_bc, sample_shell, sample_sphere

# --------------------------------------------------------------- problem data
LAM0 = 0.5
S1 = 0.1
S2 = 0.05
RHO_IN = 1.0
RHO_OUT = 100.0
BASE = 1.0              # lambda - lam0 ~ 1/rho is the decay the outer condition assumes

W_BC = 100.0            # inner/outer weight against the (rho^2-scaled) PDE residual
N_COLL = 256
N_BND = 48
TOL_SOLVE = 1e-9        # the collocation solve is exact to ~1e-15 here
TOL_GRADIENT = 5e-3     # the gradient solve is limited by its optimiser, not the setup


def lam_exact(x):
    """Exact solution at one point (vmap for batches)."""
    r = jnp.linalg.norm(x)
    z = x[2]
    quad = z * z - 0.5 * (x[0] ** 2 + x[1] ** 2)
    return LAM0 + S1 * z / r**3 + S2 * quad / r**5


def cfg_laplace() -> Config:
    """The same Config the production Robin branch reads, with the shell of this test."""
    c = Config(rho_in=RHO_IN, rho_out=RHO_OUT, lam0=LAM0, inner_radius=RHO_IN,
               lam_bc_S1=S1, lam_bc_S2=S2, outer_bc="robin", robin_source=True,
               robin_order=3, robin_orders={"h": 3, "G": 3, "lam": 3},
               robin_include_G=False, lam_inf=LAM0)
    c.__post_init__()
    return c


# ------------------------------------------------------------- the ansatz
def _raw_basis(x):
    """Separable block: {1, n_i, n_i n_j} x {d, d^2, d^3}, d = rho_in/rho.

    Angular content up to l = 2 (what the degree-2 inner data can excite) times the three
    radial powers the order-3 Robin condition admits.  Normalised below so that every
    column is O(1) over the shell -- the raw d^3 column spans six decades, which would
    otherwise make the least-squares matrix hopelessly ill-conditioned.
    """
    rho = jnp.linalg.norm(x, axis=-1, keepdims=True)
    n = x / rho
    d = RHO_IN / rho
    feats = [d, d**2, d**3, n * d, n * d**2, n * d**3]
    quad = [n[..., 0:1] * n[..., 0:1], n[..., 1:2] * n[..., 1:2], n[..., 2:3] * n[..., 2:3],
            n[..., 0:1] * n[..., 1:2], n[..., 0:1] * n[..., 2:3], n[..., 1:2] * n[..., 2:3]]
    for q in quad:
        feats += [q * d, q * d**2, q * d**3]
    return jnp.concatenate(feats, axis=-1)


_SCALE = jnp.sqrt(jnp.mean(
    _raw_basis(sample_shell(jax.random.PRNGKey(3), 8192, cfg_laplace())) ** 2, axis=0))
N_BASIS = _SCALE.shape[0]


def basis(x):
    return _raw_basis(x) / _SCALE


def lam_of(c, y):
    """lam = lam0 + sum_k c_k phi_k(y) at a single point."""
    return LAM0 + basis(y[None, :])[0] @ c


# --------------------------------------------------- the assembled conditions
def design(order, seed=0, n_coll=N_COLL, n_bnd=N_BND, w_bc=W_BC):
    """Least-squares system for the three conditions, linear in the coefficients.

    Every row is the derivative of the corresponding residual with respect to c -- the
    residuals are linear in c, so a reverse-mode gradient of the residual at c = 0 returns
    the row itself, which is far cheaper than evaluating the basis function by function.
    The right hand side carries the data and, at the outer boundary, the Robin source.
    """
    cfg = cfg_laplace()
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    coll = sample_shell(k1, n_coll, cfg)
    inner = sample_sphere(k2, n_bnd, cfg.rho_in)
    outer = sample_sphere(k3, n_bnd, cfg.rho_out)
    zero = jnp.zeros(N_BASIS)

    def pde_row(y):
        # rho^2 * Delta: the scaling that makes the Laplacian residual a lambda-scale
        # error rather than a length-scale-dependent one (the production loss scales the
        # lambda equation the same way, scale_exps {'lam_eq': 2}).
        return jax.grad(lambda c: jnp.sum(y * y) * jnp.trace(
            jax.hessian(lambda z: lam_of(c, z))(y)))(zero)

    def inner_row(y):
        return jax.grad(lambda c: lam_of(c, y) - lam_inner_bc(y, cfg))(zero)

    def outer_row(y):
        return jax.grad(lambda c: robin_operator(lambda z: lam_of(c, z), y, BASE, order,
                                                 0.0))(zero)

    s = jnp.sqrt(w_bc)
    A = jnp.concatenate([
        jax.vmap(pde_row)(coll),
        s * jax.vmap(inner_row)(inner),
        s * jax.vmap(outer_row)(outer),
    ], axis=0)
    b = jnp.concatenate([
        jnp.zeros(n_coll),
        s * jax.vmap(lambda y: lam_inner_bc(y, cfg) - LAM0)(inner),
        # the manufactured Robin source: B_n of the exact decaying part
        s * jax.vmap(lambda y: robin_operator(lam_exact, y, BASE, order, LAM0))(outer),
    ], axis=0)
    return A, b


def solve_collocation(order, **kw):
    A, b = design(order, **kw)
    c, *_ = jnp.linalg.lstsq(A, b)
    return c


def max_error(c, n=4096, seed=9):
    """max |lam_c - lam_exact| over a fresh sample of the shell."""
    xs = sample_shell(jax.random.PRNGKey(seed), n, cfg_laplace())
    got = jax.vmap(lambda y: lam_of(c, y))(xs)
    return float(jnp.max(jnp.abs(got - jax.vmap(lam_exact)(xs))))


# ---------------------------------------------- the same loss, solved by descent
def grad_batch(order, seed, n_coll=N_COLL, n_bnd=N_BND):
    """Collocation points plus the manufactured Robin source for one order."""
    cfg = cfg_laplace()
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    coll = sample_shell(k1, n_coll, cfg)
    inner = sample_sphere(k2, n_bnd, cfg.rho_in)
    outer = sample_sphere(k3, n_bnd, cfg.rho_out)
    source = jax.vmap(lambda y: robin_operator(lam_exact, y, BASE, order, LAM0))(outer)
    return coll, inner, outer, jnp.sum(coll * coll, axis=1), source


def grad_loss(order, w_bc=W_BC):
    """Residual loss for one order.  The batch is an argument, not a closure constant."""
    cfg = cfg_laplace()

    def loss(c, batch):
        coll, inner, outer, w2, source = batch
        lap = jax.vmap(lambda y: jnp.trace(jax.hessian(lambda z: lam_of(c, z))(y)))(coll)
        r_in = jax.vmap(lambda y: lam_of(c, y) - lam_inner_bc(y, cfg))(inner)
        # inf_val must match the source's: both sides are B[. - lam0], so the background
        # cancels.  Using 0 here would leave a constant B[lam0] = 6*lam0 in the residual.
        r_out = jax.vmap(lambda y: robin_operator(lambda z: lam_of(c, z), y, BASE, order,
                                                  LAM0))(outer) - source
        return (jnp.mean((w2 * lap) ** 2) + w_bc * jnp.mean(r_in**2)
                + w_bc * jnp.mean(r_out**2))

    return loss


def solve_gradient(orders=(3,), steps_per=2000, lr=1e-2, seed=0, n_coll=N_COLL,
                   n_bnd=N_BND, w_bc=W_BC):
    """Train through `orders` in sequence, warm-starting the parameters.

    A single order is the ordinary fixed-order solve.  A sequence is the order ramp
    (curriculum): train first on the cheapest and best-conditioned condition, then raise
    the order as the fit improves.  Parameters carry over between phases; the Adam state,
    the learning-rate schedule and the collocation points do not -- each phase restarts at
    the full lr with a fresh sample.

    `steps_per` is either one budget for every phase or one per phase.  The latter matters:
    a phase that starts at the full lr initially makes the residual of the *new* condition
    worse, so the final (highest) order needs most of the budget -- measured at a fixed
    2400-step total, 400/400/1600 reaches 4.8e-4 while 800/800/800 stalls at 3.5e-3.
    Returns (coefficients, per-phase final losses, initial loss).
    """
    n_phases = len(orders)
    if isinstance(steps_per, int):
        steps_per = [steps_per] * n_phases
    if len(steps_per) != n_phases:
        raise ValueError("steps_per must be an int or one entry per order")

    c = jnp.zeros(N_BASIS)
    trace, first = [], None
    for j, (order, steps) in enumerate(zip(orders, steps_per)):
        loss = grad_loss(order, w_bc)
        batch = grad_batch(order, seed + 777 * (j + 1), n_coll, n_bnd)
        opt = optax.chain(optax.clip_by_global_norm(1.0),
                          optax.adam(optax.cosine_decay_schedule(lr, steps, alpha=0.01)))
        state = opt.init(c)

        @jax.jit
        def step(c, state, batch):
            value, grads = jax.value_and_grad(loss)(c, batch)
            updates, state = opt.update(grads, state, c)
            return optax.apply_updates(c, updates), state, value

        if first is None:
            first = float(loss(c, batch))
        for _ in range(steps):
            c, state, value = step(c, state, batch)
        trace.append((order, float(value)))
    return c, trace, first


# --------------------------------------------------- properties of the setup
def test_manufactured_solution_is_harmonic_and_matches_the_inner_data():
    """The reference must solve the PDE it is used for, and carry the prescribed data."""
    cfg = cfg_laplace()
    xs = sample_shell(jax.random.PRNGKey(0), 128, cfg)
    lap = jax.vmap(lambda y: jnp.trace(jax.hessian(lam_exact)(y)))(xs)
    assert float(jnp.max(jnp.abs(lap))) < 1e-12, float(jnp.max(jnp.abs(lap)))

    # on rho = 1 the decaying harmonics reduce to the polynomial of the problem
    # statement, which is exactly what lam_inner_bc(cfg) prescribes there
    d = sample_sphere(jax.random.PRNGKey(1), 64, RHO_IN)
    prescribed = jax.vmap(lambda y: lam_inner_bc(y, cfg))(d)
    value = jax.vmap(lam_exact)(d)
    quad = d[:, 2] ** 2 - 0.5 * (d[:, 0] ** 2 + d[:, 1] ** 2)
    want = LAM0 + S1 * d[:, 2] + S2 * quad
    assert float(jnp.max(jnp.abs(prescribed - want))) < 1e-15
    assert float(jnp.max(jnp.abs(value - want))) < 1e-15


@pytest.mark.parametrize("rho", [1.5, 7.0, 100.0])
def test_robin_source_vanishes_only_from_order_three_onwards(rho):
    """The order is what decides whether this solution passes: 1 and 2 leave a residual.

    This is the sharp statement of "higher order": B_3 annihilates the whole solution, so
    its source vanishes, while the lower-order conditions still see the dipole (order 1)
    and the quadrupole (order 2).  B_n is homogeneous in rho, so the statement does not
    depend on where the condition is imposed -- hence the three radii.
    """
    x = rho * jnp.array([1.0, 2.0, 3.0]) / jnp.sqrt(14.0)

    def src(order, base=BASE):
        return abs(float(robin_operator(lam_exact, x, base, order, LAM0)))

    # the solution has rho^-2 and rho^-3 content only: survived by orders 1 and 2
    assert src(1) > 1e-9, src(1)
    assert src(2) > 1e-9, src(2)
    # order 3 is the first that admits it -- to machine precision
    assert src(3) < 1e-13, src(3)
    assert src(4) < 1e-13, src(4)
    # and the same solution sits in the base-2 window too (its leading decay is rho^-2)
    assert src(2, base=2.0) < 1e-13, src(2, base=2.0)
    assert src(1, base=2.0) > 1e-9, src(1, base=2.0)


# ------------------------------------------------------------------- solving
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_collocation_solve_ladder_reproduces_the_exact_solution(order):
    """Solve PDE + inner data + order-`order` Robin (with source) at every order.

    With the manufactured source every order admits this solution, so the ladder is a
    check of the operator and the source *at each order* -- an error at any single order
    would leave the least-squares system inconsistent and show up as a non-solution.  It is
    not a test of which order is necessary; that is the source-magnitude test above, and it
    cannot be done by solving at rho_out = 100 (see the module docstring).

    What the source has to carry differs by order, which is the useful part of the ladder:
    at rho_out the exact solution's residual is 1.0e-5 at order 1, 1.0e-7 at order 2 and
    ~1e-16 at orders 3 and 4.  So the order-1 system genuinely leans on the source, and the
    order-3 case is the condition admitting the solution unaided.  The least-squares system
    is consistent (the exact solution zeroes every row), so the weights do not bias it.

    The outer block of the system states the same thing about the whole ansatz space instead
    of about one field: orders 1 and 2 constrain it (largest row ~1e-2), while orders 3 and
    4 annihilate every basis function, so their outer blocks vanish identically (~1e-15).
    That also means the order-3 and order-4 solves are effectively PDE + inner data alone --
    consistent with lam_exact, but not evidence about the condition.
    """
    x = jnp.array([0.0, 0.0, RHO_OUT])
    src = abs(float(robin_operator(lam_exact, x, BASE, order, LAM0)))
    assert (src > 1e-9) if order < 3 else (src < 1e-13), src

    A, b = design(order)
    outer_rows = A[-N_BND:]                    # the outer-boundary block
    seen = float(jnp.max(jnp.abs(outer_rows)))
    if order < 3:
        # the ansatz reaches outside the window, so the condition genuinely constrains it
        assert seen > 1e-12, (order, seen)
    else:
        # every basis function already lies in the B_n window, so B_n annihilates the whole
        # ansatz space and the outer block is identically zero -- the annihilation property
        # checked on all 30 basis functions at once rather than on a single field
        assert seen < 1e-12, (order, seen)

    c, *_ = jnp.linalg.lstsq(A, b)
    err = max_error(c)
    assert err < TOL_SOLVE, (order, err, src, seen)


def test_gradient_solve_reproduces_the_exact_solution():
    """The same loss at order 3, minimised by Adam instead of solved directly.

    The direct solve above is limited only by linear algebra; this one checks that the
    loss is actually minimisable by descent, i.e. that the assembled conditions have no
    nearby spurious minimum that a gradient method would fall into.
    """
    c, trace, first = solve_gradient((3,), steps_per=2000)
    err = max_error(c)
    assert trace[-1][1] < 1e-3 * first, (first, trace)
    assert err < TOL_GRADIENT, (err, first, trace)


def test_gradient_solve_with_an_order_ramp_reproduces_the_exact_solution():
    """Curriculum: train at order 1, then 2, then 3, warm-starting each phase.

    The order-n residual costs n nested radial derivatives of the ansatz and its operator
    has the steepest spectrum at the largest n, so starting from the cheapest condition and
    raising the order is the natural way to train it.  With the source every phase has the
    same exact solution, so the phases cannot disagree about the target -- what the test
    checks is that the ramp stays convergent and lands on the same field as the fixed-order
    run.

    The budget split is the part that matters: each phase restarts Adam at the full lr, and
    the first steps of a phase make the residual of the newly imposed condition worse, so
    the last (highest) order needs most of the steps.  Measured at a fixed 2400-step total:
    400/400/1600 reaches 4.8e-4, 800/800/800 stalls at 3.5e-3, and repeating order 3 as a
    4th phase (600x4) is worse still at 6.8e-3.  For comparison, 2400 steps at order 3
    alone give 1.1e-3, and 2400 at order 1 give the same field in a fifth of the time --
    at this radius the order barely changes the solve (see the ladder test), and the
    higher orders earn their keep in the source-free problem, where they decide which
    multipoles pass.
    """
    c, trace, first = solve_gradient((1, 2, 3), steps_per=(400, 400, 1600))
    assert [o for o, _ in trace] == [1, 2, 3], trace
    err = max_error(c)
    assert trace[-1][1] < 1e-3 * first, (first, trace)
    assert err < TOL_GRADIENT, (err, first, trace)
