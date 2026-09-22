"""Manufactured flat-Laplacian test of the higher-order Robin conditions.

Flat Laplacian between two concentric spheres in Cartesian coordinates, with the outer
sphere normalised to unit radius (rho_final = 1) and the same shell ratio as production
(rho_in = 0.01 ... rho_out = 1):

    Delta lam   = 0                                        rho_in < |x| < rho_out
    lam         = lam0 + S1 z + S2 (z^2 - (x^2+y^2)/2)      |x| = rho_in  = 0.01
    B_n[lam - lam0] = B_n[lam_exact - lam0]                 |x| = rho_out = 1

with `lam0 = 1/2`, `S1 = 0.1`, `S2 = 0.05`.  The third line is the higher-order Robin
condition with the manufactured source: the right-hand side is `robin_operator` applied to
the exact solution, so the exact solution satisfies the condition by construction whatever
the order.  `B_n` is `robin_operator(., base=1, order=n)` -- the production code path.

The exact solution is the decaying multipole sum

    lam_exact(x) = lam0 + (S1 rho_in^2) z/rho^3 + (S2 rho_in^3) (z^2-(x^2+y^2)/2)/rho^5 ,

written with those amplitudes so that on `rho = rho_in` it reduces to the prescribed data
`lam0 + S1 n_z + S2 (3 n_z^2 - 1)/2`.  That is the *same boundary-value problem* as in the
un-normalised shell [1, 100]: rho -> rho/rho_out is an exact invariance of the whole
statement (lam is unchanged at corresponding points, `rho^2 Delta lam` is unchanged because
two derivatives contribute `L^2` and the `rho^2` prefactor `L^-2`, and `B_n` is
dimensionless), which `test_the_statement_is_invariant_under_rescaling` checks.  It is also
why the Euler-product form of the condition is the right one -- a condition written in
`d_rho` would not survive the rescaling.

The solution is harmonic (both terms are derivative combinations of `1/rho`), matches the
inner data, decays as `rho^-2, rho^-3` with no `rho^-1` part, and therefore lies exactly in
the window that `B_3` annihilates:

    order 1  kills rho^-1 only          -> dipole AND quadrupole survive
    order 2  kills rho^-1, rho^-2       -> quadrupole survives
    order 3  kills all three            -> source identically zero
    base 2, order 2 (window rho^-2..3)  -> also annihilates it (its leading decay is rho^-2)

That is what makes this a test OF the higher-order conditions: the *source* is what
discriminates the orders.  With the manufactured source the same exact solution satisfies
EVERY order by construction, so a solve at any order must reproduce it.

The solver is a plain network -- **6 hidden layers of 20 neurons, tanh** -- minimised by
Crunch's **SSBroyden** (the self-scaling Broyden recurrence `ssbroyden2` in
`PINN/Jax/Crunch/Optimizers`, imported from the sibling checkout; the solve tests skip when
that checkout is absent, at `CRUNCH_ROOT`).  Inputs are the gauge-adapted features of the
production models -- the unit direction `n = x/rho`, the normalised log radius
`t = log(rho/rho_in)/log(rho_out/rho_in)` and the decay variable `rho_in/rho` --, and the
head is a single linear unit added to `lam0`.  Parameters are float64: flax defaults to
float32 even with `jax_enable_x64`, and float32 is also what makes Crunch's two `lax.cond`
branches disagree on dtype.

Run to convergence (`|g|_inf < 1e-8`) on 1056 points (768 interior + 144 + 144; see the point
count note below) the three solves reach 5.2e-07..2.3e-06 max error against `lam_exact` over
seeds 0/3/11/21, against `TOL = 1e-05`.  The boundary weight is `w_bc = 1`: the order-n Robin
residual grows
steeply with n, and even under a line search the stiffer condition is worse (order 3: 8.5e-04
at w = 1, 1.0e-03 at w = 10, 2.2e-03 at w = 100).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from jax import flatten_util

import os
import sys

import flax.linen as nn
import jax.numpy as jnp
import pytest

from stationary.losses import robin_operator
from stationary.problem import Config, lam_inner_bc, sample_shell, sample_sphere

# ------------------------------------------------------ the SSBroyden optimiser
# Crunch's self-scaling Broyden lives in the sibling checkout (PINN/Jax/Crunch) and is not a
# dependency of this repo, so it is imported defensively: when it is absent the solve tests
# skip with a reason naming the path (set CRUNCH_ROOT if the checkout is elsewhere).  The
# optimiser works on a flat parameter vector and carries a dense inverse-Hessian estimate.
CRUNCH_ROOT = os.environ.get("CRUNCH_ROOT", "/Users/reula/Julia/PINN/Jax")
if CRUNCH_ROOT not in sys.path:
    sys.path.append(CRUNCH_ROOT)
try:
    from Crunch.Optimizers.minimize_backtracking import minimize as crunch_minimize
    HAVE_SSBROYDEN = True
except Exception:                                    # pragma: no cover - import guard
    crunch_minimize = None
    HAVE_SSBROYDEN = False

needs_ssbroyden = pytest.mark.skipif(
    not HAVE_SSBROYDEN,
    reason=f"Crunch.Optimizers.minimize not importable (looked in {CRUNCH_ROOT}; "
           "set CRUNCH_ROOT to the directory holding Crunch/)")

# --------------------------------------------------------------- problem data
LAM0 = 0.5
S1 = 0.1
S2 = 0.05
RHO_OUT = 1.0                       # rho_final: the outer sphere is the unit sphere
RHO_IN = RHO_OUT / 100.0            # 0.01: same shell ratio as the production geometry
BASE = 1.0                          # lambda - lam0 ~ 1/rho is the decay assumed by B_n
L_TWIN = RHO_OUT / RHO_IN           # 100: the un-normalised shell [1, 100], for the
                                    # rescaling-invariance test

# solver settings (measured; see the module docstring for the boundary weight)
W_BC = 1.0
# 3x the obvious 256/48/48.  The converged field error is set by how many points the residual
# is enforced on, and it saturates by 3x.  Worst over the three solves, seed 0 (total time for
# the three): 1.19e-04 at 352 points (62 s), 1.01e-05 at 704 (67 s), 2.33e-06 at 1056 (88 s),
# 1.97e-06 at 1760 (97 s), 2.27e-06 at 3520 (152 s) -- i.e. 10x the points buys nothing over
# 3x but costs 1.7x the time.  1056 points also keeps the max error measured on 40960 fresh
# points within 5% of the 4096-point value, so the field is uniformly good, not sample-fitted.
N_COLL = 768
N_BND = 144
QN_MAXITER = 2500       # cap only: every phase in this file converges well inside it
QN_GTOL = 1e-8          # stop on ||grad||_inf < this, so the solves are not cut off
# max |lam_net - lam_exact| over 4096 fresh points, with the stop above and 1056 points:
# 5.2e-07..2.3e-06 across the three solves over seeds 0/3/11/21, so 1e-05 leaves >= 4.3x
# margin -- and that worst case is 0.002% of the solution's amplitude (0.15).
TOL = 1e-5


def lam_exact(x):
    """Exact solution at one point (vmap for batches)."""
    r = jnp.linalg.norm(x)
    z = x[2]
    quad = z * z - 0.5 * (x[0] ** 2 + x[1] ** 2)
    return LAM0 + S1 * RHO_IN**2 * z / r**3 + S2 * RHO_IN**3 * quad / r**5


def lam_twin(y):
    """The same solution written in the un-normalised shell [1, 100]."""
    r = jnp.linalg.norm(y)
    z = y[2]
    quad = z * z - 0.5 * (y[0] ** 2 + y[1] ** 2)
    return LAM0 + S1 * z / r**3 + S2 * quad / r**5


def cfg_laplace() -> Config:
    """The same Config the production Robin branch reads, with the shell of this test."""
    c = Config(rho_in=RHO_IN, rho_out=RHO_OUT, lam0=LAM0, inner_radius=RHO_IN,
               lam_bc_S1=S1, lam_bc_S2=S2, outer_bc="robin", robin_source=True,
               robin_order=3, robin_orders={"h": 3, "G": 3, "lam": 3},
               robin_include_G=False, lam_inf=LAM0)
    c.__post_init__()
    return c


# ------------------------------------------------------------------- the network
class LapNet(nn.Module):
    """6 hidden layers of 20 neurons, tanh, one linear output; lam = lam0 + u.

    Features are the production ones: the unit direction n, the normalised log radius t and
    rho_in/rho (the natural decay variable -- without it the network has to synthesise the
    whole four-decade radial range from `t` alone).
    """
    width: int = 20
    depth: int = 6

    @nn.compact
    def __call__(self, x):
        x = jnp.atleast_2d(x)
        rho = jnp.linalg.norm(x, axis=-1, keepdims=True)
        n = x / rho
        t = jnp.log(rho / RHO_IN) / jnp.log(RHO_OUT / RHO_IN)
        z = jnp.concatenate([n, t, RHO_IN / rho], axis=-1)
        # flax defaults to float32 parameters even with jax_enable_x64 set (flax 0.12), which
        # would make this a float32 solve inside an otherwise float64 test -- and it is what
        # breaks Crunch's SSBroyden, whose two lax.cond branches would then disagree on dtype.
        dense = dict(param_dtype=jnp.float64, dtype=jnp.float64)
        for _ in range(self.depth):
            z = jnp.tanh(nn.Dense(self.width, **dense)(z))
        u = nn.Dense(1, kernel_init=nn.initializers.normal(1e-2),
                     bias_init=nn.initializers.zeros, **dense)(z)
        return LAM0 + u[..., 0]


def lam_of(params, model, y):
    """lam at a single point."""
    return model.apply(params, y[None, :])[0]


# -------------------------------------------------------------------- the loss
def net_batch(order, seed, n_coll=N_COLL, n_bnd=N_BND):
    """Collocation points plus the manufactured Robin source for one order."""
    cfg = cfg_laplace()
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    coll = sample_shell(k1, n_coll, cfg)
    inner = sample_sphere(k2, n_bnd, cfg.rho_in)
    outer = sample_sphere(k3, n_bnd, cfg.rho_out)
    source = jax.vmap(lambda y: robin_operator(lam_exact, y, BASE, order, LAM0))(outer)
    return coll, inner, outer, source


def net_loss(order, w_bc=W_BC):
    """Residual loss for one order.  The batch is an argument, not a closure constant."""
    cfg = cfg_laplace()

    def loss(params, model, batch):
        coll, inner, outer, source = batch
        lam = lambda y: lam_of(params, model, y)
        # rho^2 * Delta: the scaling that makes the Laplacian residual a lambda-scale error
        # rather than a length-scale-dependent one (production scales the lambda equation
        # the same way, scale_exps {'lam_eq': 2}) -- and it is what makes the residual
        # invariant under rho -> rho/L, so this test is independent of rho_final.
        lap = jax.vmap(lambda y: jnp.trace(jax.hessian(lam)(y)))(coll)
        r_in = jax.vmap(lambda y: lam(y) - lam_inner_bc(y, cfg))(inner)
        # inf_val must match the source's: both sides are B[. - lam0], so the background
        # cancels.  Using 0 here would leave a constant B[lam0] = 6*lam0 in the residual.
        r_out = jax.vmap(lambda y: robin_operator(lam, y, BASE, order,
                                                  LAM0))(outer) - source
        return (jnp.mean((jnp.sum(coll * coll, axis=1) * lap) ** 2)
                + w_bc * jnp.mean(r_in**2) + w_bc * jnp.mean(r_out**2))

    return loss


def solve_network(orders=(3,), maxiter=QN_MAXITER, gtol=QN_GTOL, seed=0, w_bc=W_BC,
                  n_coll=N_COLL, n_bnd=N_BND):
    """Minimise the loss with Crunch's SSBroyden, one fixed batch per phase.

    No Adam phase and no learning-rate schedule.  The optimiser works on a flat 1-D vector, so
    the parameters go through `jax.flatten_util` and are unflattened inside the loss; it
    carries a dense inverse-Hessian estimate (`initial_H = I`; 2241^2 float64 = 40 MB here)
    and takes its steps from the two-stage line search of `minimize_backtracking`
    (primary c1 = 1e-4, c2 = 0.9, <=15 evaluations; fallback with c2 = 0.8 then 0.5).
    `update_method = "ssbroyden2"` selects the self-scaling Broyden recurrence; `initial_H`
    is passed inside `options`, which is where the SciPy-style wrapper forwards it.

    The batch is drawn once per phase and held fixed -- a line search needs one objective.

    Each phase stops on `||grad||_inf < gtol` or at `maxiter`.  Measured here every phase
    converges (status 0) in 218..1337 iterations against a 2500 cap, so the field errors the
    tests assert are properties of the method rather than of an iteration budget.  Halving the
    tolerance and dropping the cap to 600 was the earlier configuration: it left the scheme
    still improving by ~4x per 500 iterations, i.e. it measured the budget.

    A sequence of orders is the curriculum: parameters carry over, the inverse Hessian and the
    batch do not.  Returns (model, params, trace), one dict per phase with its order, the
    iterations used, the loss before and after, and the converged flag.
    """
    model = LapNet()
    params = model.init(jax.random.PRNGKey(seed + 1), jnp.ones((1, 3)))
    trace = []
    for j, order in enumerate(orders):
        loss = net_loss(order, w_bc)
        batch = net_batch(order, seed + 777 * (j + 1), n_coll, n_bnd)
        flat0, unflatten = flatten_util.ravel_pytree(params)
        fun = lambda flat: loss(unflatten(flat), model, batch)
        f0 = float(fun(flat0))
        res = crunch_minimize(
            fun, flat0, args=(), method="BFGS",
            options={"maxiter": maxiter, "gtol": gtol, "initial_H": jnp.eye(flat0.size),
                     # initial_scale would engage SSBroyden's tau^A first-step rescaling; it
                     # is off here because the network starts at lam0 and the loss is already
                     # small (measured at order 3: 4.6e-04 with it, 1.8e-04 without).  In
                     # train.py it is ON: there the Adam warm-up leaves a large gradient and
                     # the Wolfe line search cannot bracket the unscaled first step at all.
                     "update_method": "ssbroyden2", "initial_scale": False})
        params = unflatten(res.x)
        trace.append({"order": order, "nit": int(res.nit), "f0": f0,
                      "f": float(res.fun), "converged": bool(res.success)})
    return model, params, trace


def max_error(model, params, n=4096, seed=9):
    """max |lam_net - lam_exact| over a fresh sample of the shell."""
    xs = sample_shell(jax.random.PRNGKey(seed), n, cfg_laplace())
    got = jax.vmap(lambda y: lam_of(params, model, y))(xs)
    return float(jnp.max(jnp.abs(got - jax.vmap(lam_exact)(xs))))


# --------------------------------------------------- properties of the setup
def test_manufactured_solution_is_harmonic_and_matches_the_inner_data():
    """The reference must solve the PDE it is used for, and carry the prescribed data."""
    cfg = cfg_laplace()
    xs = sample_shell(jax.random.PRNGKey(0), 128, cfg)
    lap = jax.vmap(lambda y: jnp.trace(jax.hessian(lam_exact)(y)))(xs)
    assert float(jnp.max(jnp.abs(lap))) < 1e-10, float(jnp.max(jnp.abs(lap)))

    # on rho = rho_in the decaying harmonics reduce to the polynomial of the problem
    # statement, which is exactly what lam_inner_bc(cfg) prescribes there
    d = sample_sphere(jax.random.PRNGKey(1), 64, RHO_IN)
    prescribed = jax.vmap(lambda y: lam_inner_bc(y, cfg))(d)
    value = jax.vmap(lam_exact)(d)
    quad = d[:, 2] ** 2 - 0.5 * (d[:, 0] ** 2 + d[:, 1] ** 2)
    want = LAM0 + S1 * d[:, 2] / RHO_IN + S2 * quad / RHO_IN**2
    assert float(jnp.max(jnp.abs(prescribed - want))) < 1e-15
    assert float(jnp.max(jnp.abs(value - want))) < 1e-15


def test_the_statement_is_invariant_under_rescaling():
    """rho -> rho/rho_out changes nothing, which is why rho_final = 1 is not a new problem.

    Checked against the un-normalised twin shell [1, 100] at corresponding points: the same
    lam, the same `rho^2 Delta lam` (two derivatives give `L^2`, the `rho^2` prefactor
    `L^-2`) and the same `B_n`, for every order.  This is also the property that a
    condition written in `d_rho` rather than in `theta = rho d_rho` would fail.
    """
    cfg = cfg_laplace()
    xs = sample_shell(jax.random.PRNGKey(4), 64, cfg)
    ys = L_TWIN * xs

    dl = jax.vmap(lam_exact)(xs) - jax.vmap(lam_twin)(ys)
    assert float(jnp.max(jnp.abs(dl))) < 1e-15, float(jnp.max(jnp.abs(dl)))

    def scaled_lap(f, y):
        return jnp.sum(y * y) * jnp.trace(jax.hessian(f)(y))

    lap_new = jax.vmap(lambda y: scaled_lap(lam_exact, y))(xs)
    lap_old = jax.vmap(lambda y: scaled_lap(lam_twin, y))(ys)
    assert float(jnp.max(jnp.abs(lap_new - lap_old))) < 1e-12

    for order in (1, 2, 3, 4):
        new = jax.vmap(lambda y: robin_operator(lam_exact, y, BASE, order, LAM0))(xs)
        old = jax.vmap(lambda y: robin_operator(lam_twin, y, BASE, order, LAM0))(ys)
        scale = max(1.0, float(jnp.max(jnp.abs(old))))
        assert float(jnp.max(jnp.abs(new - old))) < 1e-12 * scale, (order, scale)


@pytest.mark.parametrize("frac", [0.015, 0.07, 1.0])
def test_robin_source_vanishes_only_from_order_three_onwards(frac):
    """The order is what decides whether this solution passes: 1 and 2 leave a residual.

    This is the sharp statement of "higher order": B_3 annihilates the whole solution, so
    its source vanishes, while the lower-order conditions still see the dipole (order 1)
    and the quadrupole (order 2).  B_n is homogeneous in rho, so the statement does not
    depend on where the condition is imposed -- hence the three radii (rho_in, mid-shell,
    rho_final).
    """
    x = frac * jnp.array([1.0, 2.0, 3.0]) / jnp.sqrt(14.0)

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
@needs_ssbroyden
def test_network_solve_at_order_one_matches_the_exact_solution():
    """The cheapest condition, solved with SSBroyden from a random init at w_bc = 1."""
    model, params, trace = solve_network((1,))
    err = max_error(model, params)
    assert all(t["converged"] for t in trace), trace
    assert trace[0]["f"] < 1e-2 * trace[0]["f0"], trace
    assert err < TOL, (err, trace)


@needs_ssbroyden
def test_network_solve_at_order_three_matches_the_exact_solution():
    """The condition that admits the quadrupole unaided (its source is ~1e-16)."""
    model, params, trace = solve_network((3,))
    err = max_error(model, params)
    assert all(t["converged"] for t in trace), trace
    assert trace[0]["f"] < 1e-2 * trace[0]["f0"], trace
    assert err < TOL, (err, trace)


@needs_ssbroyden
def test_network_solve_with_an_order_ramp_matches_the_exact_solution():
    """Curriculum: order 1, then 2, then 3, warm-starting the parameters.

    The order-n residual costs n nested radial derivatives of the network and its operator has
    the steepest spectrum at the largest n, so training the cheapest condition first and
    raising the order is the natural schedule.  Each phase restarts the quasi-Newton state
    (identity inverse Hessian) and draws its own batch; only the parameters carry over, and
    the warm start shows up as a cheaper last phase (608 iterations for order 3 here, against
    1337 when it is solved alone).
    """
    model, params, trace = solve_network((1, 2, 3))
    assert [t["order"] for t in trace] == [1, 2, 3], trace
    err = max_error(model, params)
    assert all(t["converged"] for t in trace), trace
    assert trace[-1]["f"] < 1e-2 * trace[-1]["f0"], trace
    assert err < TOL, (err, trace)
