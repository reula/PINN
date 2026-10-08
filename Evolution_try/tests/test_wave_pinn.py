"""Tests for the wave PINN framework.

``pytest`` is not installed in the project's interpreter, so these are plain
``unittest`` cases:

    cd Evolution_try
    /Users/reula/jax_env/bin/python -m unittest discover -s tests -v

Every test here is fast (seconds): they check the maths and the plumbing, not
convergence.  ``test_ssbroyden_descends`` is the only one that runs an
optimiser.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wave_pinn.config import Config, apply_overrides                     # noqa: E402
from wave_pinn.features import feature_dim                               # noqa: E402
from wave_pinn.losses import Objective                                   # noqa: E402
from wave_pinn.model import ansatz_u, init_params, n_parameters          # noqa: E402
from wave_pinn.problem import exact_solution, initial_data, pde_residual, profile   # noqa: E402
from wave_pinn.sampling import make_batch                                # noqa: E402
from wave_pinn.train import configure_jax                                # noqa: E402

configure_jax(Config())


def u_fn(params, cfg):
    return lambda t, x: ansatz_u(params, cfg, t, x)


class TestConfig(unittest.TestCase):
    def test_roundtrip(self):
        cfg = Config(n_modes=9, label="x", optimizer="dsgnar")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            cfg.to_json(path)
            back = Config.from_json(path)
        self.assertEqual(back, cfg)

    def test_overrides_coerce(self):
        cfg = apply_overrides(Config(), ["n_modes=10", "qn_gtol=1e-9", "periodic=false",
                                         "label=hello"])
        self.assertEqual(cfg.n_modes, 10)
        self.assertAlmostEqual(cfg.qn_gtol, 1e-9)
        self.assertIs(cfg.periodic, False)
        self.assertEqual(cfg.label, "hello")

    def test_unknown_key_rejected(self):
        with self.assertRaises(KeyError):
            apply_overrides(Config(), ["not_a_field=1"])
        with self.assertRaises(ValueError):
            Config(features="nope").validate()

    def test_defaults_are_the_requested_problem(self):
        cfg = Config()
        self.assertEqual((cfg.L, cfg.c, cfg.T), (1.0, 1.0, 2.0))
        self.assertEqual((cfg.n_layers, cfg.n_neurons), (6, 20))
        self.assertEqual(cfg.equation, "wave2")
        self.assertTrue(cfg.periodic)
        self.assertEqual(cfg.optimizer, "ssbroyden")
        cfg.validate()


class TestInitialCondition(unittest.TestCase):
    """The whole point of the ansatz: the initial data is exact, not fitted."""

    def setUp(self):
        self.x = jnp.linspace(-1.0, 1.0, 33)

    def _check(self, cfg, tol=0.0):
        configure_jax(cfg)
        params = init_params(cfg, jax.random.PRNGKey(0))
        u0, v0 = initial_data(cfg, self.x)
        got0 = ansatz_u(params, cfg, jnp.zeros_like(self.x), self.x)
        dp = np.asarray(got0) - np.asarray(u0)
        self.assertLessEqual(np.max(np.abs(dp)), tol if tol else 1e-14)
        # u_t(0, x) = v0(x) for every parameter vector
        ut = jax.vmap(lambda xx: jax.grad(lambda tt: ansatz_u(params, cfg, tt, xx))(0.0))(self.x)
        dq = np.asarray(ut) - np.asarray(v0)
        self.assertLessEqual(np.max(np.abs(dq)), tol if tol else 1e-13)

    def test_gaussian_periodised(self):
        self._check(Config())

    def test_sin(self):
        self._check(Config(u0="sin", n_modes=4))

    def test_two_mode_fourier(self):
        self._check(Config(features="fourier_ic", n_modes=4))

    def test_unperiodised_ic_still_hard_coded(self):
        self._check(Config(periodize_ic=False))


class TestPeriodicity(unittest.TestCase):
    def test_ansatz_is_periodic_to_machine_precision(self):
        cfg = Config()
        params = init_params(cfg, jax.random.PRNGKey(3))
        t = jnp.linspace(0.0, cfg.T, 11)
        left = jnp.stack([ansatz_u(params, cfg, tt, jnp.array(-cfg.L)) for tt in t])
        right = jnp.stack([ansatz_u(params, cfg, tt, jnp.array(cfg.L)) for tt in t])
        self.assertLess(float(jnp.max(jnp.abs(left - right))), 1e-13)

    def test_periodised_profile_is_smooth_and_periodic(self):
        cfg = Config()
        x = jnp.linspace(-1.0, 1.0, 201)
        p = profile(cfg, x)
        # exactly periodic values and slopes at the two ends
        self.assertAlmostEqual(float(profile(cfg, jnp.array(-1.0))),
                               float(profile(cfg, jnp.array(1.0))), places=12)
        dp = jax.grad(lambda z: profile(cfg, z))
        self.assertAlmostEqual(float(dp(jnp.array(-1.0))), float(dp(jnp.array(1.0))), places=10)
        self.assertAlmostEqual(float(np.max(np.asarray(p))), 1.0, places=6)
        # away from machine zero at the ends, so "periodic" is a real statement
        self.assertGreater(float(p[0]), 1e-7)


class TestPdeResidual(unittest.TestCase):
    def test_exact_solution_has_zero_residual(self):
        for eq in ("wave2", "advection"):
            cfg = Config(equation=eq)
            t = jnp.array([0.0, 0.3, 1.0, 1.7, 2.0])
            x = jnp.array([-0.9, -0.31, 0.0, 0.42, 0.95])
            r = pde_residual(cfg, lambda tt, xx: exact_solution(cfg, tt, xx), t, x)
            self.assertLess(float(jnp.max(jnp.abs(r))), 1e-10, msg=f"{eq}: {r}")

    def test_exact_solution_is_periodic_in_time_far_past_the_image_window(self):
        """The reference solution must stay valid at large t.

        The periodised profile is a *truncated* image sum, so it is only
        meaningful near the fundamental domain; ``exact_solution`` evaluates it at
        ``x - c t``, which reaches -21 at T = 20.  If the fold into [-L, L) is
        missing, the reference solution is numerical zero for t >~ 7 and every
        error reported there is a ratio against nothing -- which is exactly what
        happened, and why this test exists.
        """
        cfg = Config(T=20)
        x = jnp.linspace(-cfg.L, cfg.L, 201)
        ref = None
        for t in (0.0, 2.0, 6.5, 10.0, 13.3, 17.0, 20.0):
            ex = exact_solution(cfg, jnp.full_like(x, t), x)
            self.assertGreater(float(jnp.sqrt(jnp.mean(ex ** 2))), 0.3)
            if ref is None:
                ref = ex
            # u(t) == u(t - 2L) for the periodic problem, exactly
            ex_prev = exact_solution(cfg, jnp.full_like(x, t - 2.0 * cfg.L), x)
            self.assertLess(float(jnp.max(jnp.abs(ex - ex_prev))), 1e-12)
        # ...and its residual still vanishes, at large t
        t = jnp.array([5.0, 10.0, 15.0, 19.9])
        xs = jnp.array([-0.7, -0.2, 0.1, 0.55])
        r = pde_residual(cfg, lambda tt, xx: exact_solution(cfg, tt, xx), t, xs)
        self.assertLess(float(jnp.max(jnp.abs(r))), 1e-10)

    def test_residual_is_second_order_for_the_wave_equation(self):
        """A linear function in t and x is not a solution, but a linear-in-t, quadratic-in-x
        polynomial has a constant residual -- a cheap check that the right derivatives are taken."""
        cfg = Config(equation="wave2")
        # u = x^2 has u_tt = 0, u_xx = 2, so the residual is -2c^2 everywhere
        r = pde_residual(cfg, lambda tt, xx: xx ** 2, jnp.array([0.5]), jnp.array([0.3]))
        self.assertAlmostEqual(float(r[0]), -2.0 * cfg.c ** 2, places=10)
        # u = t^2 has u_tt = 2, u_xx = 0
        r = pde_residual(cfg, lambda tt, xx: tt ** 2, jnp.array([0.5]), jnp.array([0.3]))
        self.assertAlmostEqual(float(r[0]), 2.0, places=10)


class TestObjective(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(n_coll=256, n_modes=4)
        key = jax.random.PRNGKey(1)
        self.params = init_params(self.cfg, key)
        self.obj = Objective(self.cfg, make_batch(self.cfg, key), self.params)
        self.flat, _ = jax.flatten_util.ravel_pytree(self.params)

    def test_gradient_matches_the_gauss_newton_identity(self):
        """For L = mean(r^2), grad L = (2/M) J^T r exactly."""
        r = np.asarray(self.obj.residual(self.flat))
        J = np.asarray(self.obj.jacobian(self.flat))
        g_ref = 2.0 / r.size * (J.T @ r)
        g = np.asarray(self.obj.grad(self.flat))
        self.assertLess(np.max(np.abs(g - g_ref)) / max(np.max(np.abs(g_ref)), 1e-30), 1e-10)

    def test_residual_scale_normalises_the_initial_loss(self):
        """With residual_norm='auto' the loss at the zero network is O(1), not O(1/sigma^4)."""
        zero = jnp.zeros((self.obj.n,), self.obj.dtype)
        cfg_off = self.cfg.replace(residual_norm="none")
        obj_off = Objective(cfg_off, self.obj.batch._replace(scale=jnp.asarray(1.0)), self.params)
        self.assertLess(self.obj.loss(zero), 10.0)
        self.assertGreater(obj_off.loss(zero), 100.0)


class TestFeatures(unittest.TestCase):
    def test_dimensions(self):
        self.assertEqual(feature_dim(Config()), 3)                       # the default, periodic
        self.assertEqual(feature_dim(Config(features="periodic_ic")), 5)
        self.assertEqual(feature_dim(Config(features="fourier", n_modes=6)), 13)
        self.assertEqual(feature_dim(Config(features="fourier_ic", n_modes=6)), 15)
        self.assertEqual(feature_dim(Config(features="plain")), 2)
        self.assertEqual(feature_dim(Config(features="plain_ic")), 4)

    def test_parameter_count(self):
        params = init_params(Config(), jax.random.PRNGKey(0))
        # the default 6 x 20 MLP on (t, cos, sin): 3*20+20 + 5*(20*20+20) + 20*1+1
        self.assertEqual(n_parameters(params), 2201)
        params = init_params(Config(features="fourier", n_modes=6), jax.random.PRNGKey(0))
        self.assertEqual(n_parameters(params), 2401)


class TestSubagentFreeOptimisers(unittest.TestCase):
    def test_adam_descends(self):
        cfg = Config(n_coll=256, n_modes=4, optimizer="adam", steps=30, log_every=10)
        from wave_pinn.optim.adam import adam_phase
        key = jax.random.PRNGKey(2)
        params = init_params(cfg, key)
        obj = Objective(cfg, make_batch(cfg, key), params)
        flat, _ = jax.flatten_util.ravel_pytree(params)
        before = obj.loss(flat)
        flat, _hist, _info = adam_phase(obj, flat, cfg, verbose=False)
        self.assertLess(obj.loss(flat), before)

    def test_ssbroyden_descends(self):
        cfg = Config(n_coll=256, n_modes=4, optimizer="ssbroyden", qn_steps=40, qn_block=40)
        from wave_pinn.optim.ssbroyden import ssbroyden_phase
        key = jax.random.PRNGKey(2)
        params = init_params(cfg, key)
        obj = Objective(cfg, make_batch(cfg, key), params)
        flat, _ = jax.flatten_util.ravel_pytree(params)
        before = obj.loss(flat)
        flat, hist, info = ssbroyden_phase(obj, flat, cfg, verbose=False)
        self.assertLess(obj.loss(flat), 0.5 * before)
        self.assertGreaterEqual(info["iterations"], 1)
        self.assertAlmostEqual(hist[-1]["loss"], obj.loss(flat), places=12)


class TestDsgnarMachinery(unittest.TestCase):
    """The places where a DSGNAR port goes wrong silently.

    Each of these was a real bug during development, and none of them shows up as
    a crash or as a loss that fails to fall.
    """

    def setUp(self):
        from wave_pinn.optim import dsgnar as D
        self.D = D
        self.cfg = Config(n_coll=48, dsgnar_sketch=11)
        key = jax.random.PRNGKey(5)
        self.params = init_params(self.cfg, key)
        self.obj = Objective(self.cfg, make_batch(self.cfg, key), self.params)
        self.flat, _ = jax.flatten_util.ravel_pytree(self.params)
        self.srct = D.make_srct(jax.random.PRNGKey(6), self.obj.n, 11)
        self.B = D.srct_columns(self.srct, self.obj.n, 11, self.obj.dtype)

    def test_srct_lift_is_the_exact_adjoint_and_an_isometry(self):
        n, s = self.obj.n, 11
        x = jax.random.normal(jax.random.PRNGKey(7), (n,))
        y = jax.random.normal(jax.random.PRNGKey(8), (s,))
        applied = self.D.apply_srct(x[None, :], self.srct)[0]
        lhs = float(jnp.dot(applied, y))
        rhs = float(jnp.dot(x, self.D.lift_srct(y, self.srct, n)))
        self.assertAlmostEqual(lhs, rhs, places=9)
        lifted = self.D.lift_srct(y, self.srct, n)
        self.assertLess(float(jnp.max(jnp.abs(self.B.T @ y - lifted))), 1e-11)
        back = self.D.apply_srct(lifted[None, :], self.srct)[0]
        self.assertLess(float(jnp.max(jnp.abs(back - y))), 1e-11)
        self.assertAlmostEqual(float(jnp.linalg.norm(lifted)),
                               float(jnp.linalg.norm(y)), places=11)

    def test_sketched_jacobian_matches_the_full_jacobian(self):
        """``J(flat) B^T`` by batched JVPs, against ``jacfwd`` -- at a *moved* point."""
        moved = self.flat + 0.01 * jax.random.normal(jax.random.PRNGKey(9), self.flat.shape)
        got = self.D.jvp_columns(self.obj.residual, moved, self.B).T   # (M, s)
        ref = self.obj.jacobian(moved) @ self.B.T
        scale = float(jnp.max(jnp.abs(ref)))
        self.assertLess(float(jnp.max(jnp.abs(got - ref))) / scale, 1e-11)
        # ...and it must depend on where it is linearised: a frozen linearisation
        # point is the bug this test exists for.
        at_start = self.D.jvp_columns(self.obj.residual, self.flat, self.B).T
        self.assertGreater(float(jnp.max(jnp.abs(got - at_start))) / scale, 1e-6)

    def test_predicted_decrease_equals_the_quadratic_model(self):
        """``pred(lam) = m(0) - m(p~(lam))`` for ``m(p) = ||r~ + J~ p||^2`` (M = 1)."""
        s = 24
        J = jax.random.normal(jax.random.PRNGKey(10), (s, s))
        r = jax.random.normal(jax.random.PRNGKey(11), (s,))
        u, sing, v_t = jnp.linalg.svd(J)
        g = sing * (u.T @ r)
        for lam in (0.0, 0.01, 0.5, 5.0, 500.0):
            p, pred = self.D.step_and_pred(sing, g, v_t, lam)
            lhs = float(jnp.sum(r * r) - jnp.sum((r + J @ p) ** 2))
            self.assertGreater(lhs, 0.0)
            self.assertLess(abs(lhs - float(pred)) / lhs, 1e-9, msg=f"lam={lam}")

    def test_secular_solve_hits_the_target_radius(self):
        s = 16
        sing = jnp.linspace(1e-3, 1e2, s)
        g = jax.random.normal(jax.random.PRNGKey(12), (s,))
        for delta in (1e-3, 1e-1, 1.0, 10.0):
            lam = self.D.solve_subproblems(sing, g, jnp.array([delta]), 60, 0.0)[0]
            nrm = float(jnp.linalg.norm(g / (sing ** 2 + lam)))
            self.assertLess(abs(nrm - delta) / delta, 1e-6, msg=f"delta={delta}")

    def test_dsgnar_phase_descends(self):
        from wave_pinn.optim.dsgnar import dsgnar_phase
        # a deterministic grid, so this measures DSGNAR and not the sampler
        cfg = Config(n_coll=128, sampler="uniform", optimizer="dsgnar", dsgnar_steps=6,
                     dsgnar_sketch=64, resample_every=0, log_every=100)
        key = jax.random.PRNGKey(13)
        params = init_params(cfg, key)
        obj = Objective(cfg, make_batch(cfg, key), params)
        flat, _ = jax.flatten_util.ravel_pytree(params)
        before = obj.loss(flat)
        flat, hist, info = dsgnar_phase(obj, flat, cfg, verbose=False)
        self.assertLess(obj.loss(flat), 0.5 * before)
        self.assertEqual(info["iterations"], 6)
        self.assertGreaterEqual(info["accepted"], 1)
        self.assertTrue(all(np.isfinite(h["loss"]) for h in hist))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestTrustRegionMachinery(unittest.TestCase):
    """The third optimiser: Xu & Darve's trust region on the exact dense Hessian.

    The paper delegates its loop to ``scipy.optimize.minimize(method="trust-exact")``
    and its subproblem to ``scipy.optimize._trustregion_exact.IterativeSubproblem``.
    The module therefore *calls* SciPy's solver rather than re-deriving it, and the
    test that matters is that the outer loop built around it reproduces SciPy's own
    trust-region loop step for step.  The other test is the one the paper's authors
    insist on before optimising: that the Hessian is genuinely the second derivative
    of the loss, and not the Gauss-Newton matrix the other optimisers use.
    """

    def test_outer_loop_matches_scipy_trust_exact(self):
        from scipy.optimize import minimize
        from wave_pinn.optim.trustregion import make_hessian, trust_region_phase

        cfg = Config(n_coll=48, sampler="uniform", n_modes=2, resample_every=0,
                     tr_maxiter=12, tr_chunk=64, tr_gtol=1e-12, log_every=1000)
        key = jax.random.PRNGKey(41)
        params = init_params(cfg, key)
        obj = Objective(cfg, make_batch(cfg, key), params)
        flat0, _ = jax.flatten_util.ravel_pytree(params)
        loss, grad, hess = make_hessian(obj, cfg)
        x0 = np.asarray(flat0, dtype=float)

        flat, hist, info = trust_region_phase(obj, flat0, cfg, verbose=False)
        ours = np.asarray(flat, dtype=float)

        res = minimize(loss, x0, jac=grad, hess=hess, method="trust-exact",
                       tol=0.0, options={"maxiter": 12, "gtol": 1e-12})

        denom = max(1.0, float(np.max(np.abs(res.x))))
        self.assertLess(float(np.max(np.abs(ours - res.x))) / denom, 1e-9,
                        msg=f"our loop diverged from scipy's: {info['stopped']}")
        self.assertAlmostEqual(float(loss(ours)), float(res.fun), places=10)

    def test_exact_hessian_is_second_order_consistent_and_differs_from_gauss_newton(self):
        """The paper's contribution is the indefinite exact Hessian, not J^T J."""
        cfg = Config(n_coll=24, sampler="uniform", n_modes=2)
        key = jax.random.PRNGKey(21)
        params = init_params(cfg, key)
        obj = Objective(cfg, make_batch(cfg, key), params)
        flat, _ = jax.flatten_util.ravel_pytree(params)
        H = np.asarray(jax.hessian(obj._loss)(flat), dtype=float)
        self.assertLess(float(np.max(np.abs(H - H.T))), 1e-10 * max(1.0, float(np.max(np.abs(H)))))
        eps = 1e-6
        for j in (0, 5, 17):
            e = np.zeros(flat.size)
            e[j] = eps
            gp = np.asarray(jax.grad(obj._loss)(flat + jnp.asarray(e)), dtype=float)
            gm = np.asarray(jax.grad(obj._loss)(flat - jnp.asarray(e)), dtype=float)
            fd = (gp - gm) / (2 * eps)
            scale = max(1.0, float(np.max(np.abs(H[:, j]))))
            self.assertLess(float(np.max(np.abs(H[:, j] - fd))) / scale, 1e-4)
        res = np.asarray(obj.residual(flat), dtype=float)
        J = np.asarray(obj.jacobian(flat), dtype=float)
        H_gn = (2.0 / res.size) * (J.T @ J)
        self.assertGreater(float(np.max(np.abs(H - H_gn))),
                           1e-6 * max(1.0, float(np.max(np.abs(H)))))

    def test_subproblem_is_optimal(self):
        """PD H with an inactive constraint: p = -H^{-1} g.  Indefinite H: the step
        lies (nearly) on the boundary and lowers the model."""
        from wave_pinn.optim.trustregion import Subproblem
        rng = np.random.default_rng(11)
        A = rng.normal(size=(20, 20))
        H = A @ A.T + 20.0 * np.eye(20)
        g = rng.normal(size=20)
        p, hits = Subproblem(H, g).solve(1e9)
        self.assertFalse(hits)
        self.assertLess(float(np.linalg.norm(p + np.linalg.solve(H, g))), 1e-8)

        H_indef = H - 60.0 * np.eye(20)
        delta = 0.5
        p, hits = Subproblem(H_indef, g).solve(delta)
        self.assertTrue(hits)
        self.assertLess(float(np.linalg.norm(p)) / delta, 1.0 + 0.1 + 1e-9)   # 1 + k_easy slack
        self.assertLess(float(g @ p + 0.5 * p @ H_indef @ p), 0.0)   # the model decreased

    def test_trust_region_phase_descends(self):
        from wave_pinn.optim.trustregion import trust_region_phase
        cfg = Config(n_coll=64, sampler="uniform", n_modes=2, tr_maxiter=8,
                     tr_chunk=64, resample_every=0, log_every=100)
        key = jax.random.PRNGKey(31)
        params = init_params(cfg, key)
        obj = Objective(cfg, make_batch(cfg, key), params)
        flat, _ = jax.flatten_util.ravel_pytree(params)
        before = obj.loss(flat)
        flat, hist, info = trust_region_phase(obj, flat, cfg, verbose=False)
        self.assertLess(obj.loss(flat), 0.9 * before)
        self.assertEqual(info["hessian"], "exact")
        self.assertGreaterEqual(info["accepted"], 1)
        self.assertTrue(all(np.isfinite(h["loss"]) for h in hist))
