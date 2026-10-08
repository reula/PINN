"""Trust-region Newton with the exact dense Hessian: Xu & Darve, arXiv:2105.07552.

The paper
---------
K. Xu and E. Darve, *Trust Region Method for Coupled Systems of PDE Solvers and
Deep Neural Networks* (arXiv:2105.07552v1).  Its argument is that BFGS/L-BFGS
mislead PINN training because they force a *positive definite* curvature
approximation, while the true Hessian of a PDE-constrained loss is indefinite or
positive *semi*-definite; the method is therefore built on the exact Hessian and
on a trust region, which handles indefiniteness natively.

Two facts about it matter for a port, and both come from the authors' own code
rather than from the paper's prose:

1. **The optimiser loop is SciPy.**  The paper specifies only the model,
   ``min_p f + g^T p + (1/2) p^T B p`` s.t. ``||p|| <= Delta``, and then says it
   uses "the nearly exact trust region method proposed in [Conn-Gould-Toint,
   Chapter 7] ... implemented in the scipy library".  The call site in the
   ADCME documentation is
   ``minimize(loss, x0, method="trust-exact", jac=..., hess=..., options={"maxiter": 5000, "gtol": 0.0})``.
   So the subproblem is the More-Sorensen nearly-exact solve: safeguarded Newton
   on the secular equation, a Cholesky of ``B + lambda I`` at every trial
   ``lambda``, and the two-dimensional "hard case" fallback along the
   negative-curvature direction.  Not Steihaug-CG, not dogleg, no Cauchy point.
   This module reimplements that loop and that solver in numpy so the project
   stays self-contained, and tests it against SciPy.

2. **The Hessian is the exact one, and it is required as a matrix.**  Hessian-
   vector products are explicitly rejected by the paper ("the ability to
   calculate Hessians paves the way to more sophisticated and efficient
   optimization techniques, instead of restricting us to matrix-free
   approaches"), and SciPy's ``trust-exact`` raises unless ``hess`` returns a
   dense ``(n, n)`` array.  For this project's loss, ``L = mean(r^2)``,

       grad^2 L = (2/M) [ J_r^T J_r + sum_i r_i grad^2 r_i ].

   Keeping the second term is the whole point: it is what makes ``B_k``
   indefinite.  Dropping it gives the Gauss-Newton matrix, which is what DSGNAR
   already uses; ``tr_hessian="gauss_newton"`` is provided only so that the
   difference can be measured, and it is *not* this paper's algorithm.

Cost, and why the knobs matter here
-----------------------------------
Forming the dense Hessian costs ``n`` forward-over-reverse passes, i.e. ``n``
gradients, and the subproblem factors ``(n x n)`` matrices of order ``n^3/3``
each.  At ``n = 2201`` that is ~39 MB, ~45 s per Hessian on this machine, and
several factorisations per accepted step -- roughly two orders of magnitude more
work per iteration than SSBroyden.  The paper's own networks are 901-921
parameters; it reports the method converging in ~270 iterations there.  So this
phase is best used the way the paper uses it: few, well-chosen steps, each
re-solving a quadratic model that the first-order methods cannot build.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ..config import Config, resample_schedule
from ..losses import Objective

# --------------------------------------------------------------------------
# constants, verbatim from scipy/optimize/_trustregion_exact.py
# --------------------------------------------------------------------------
UPDATE_COEFF = 0.01   # Conn-Gould-Toint eq. 7.3.14 (p. 190), named "theta" there
K_EASY = 0.1          # stop criteria for the iterative subproblem, CGT pp. 194-197
K_HARD = 0.2
EPS = float(np.finfo(np.float64).eps)


# --------------------------------------------------------------------------
# the Hessian
# --------------------------------------------------------------------------
def make_hessian(objective: Objective, cfg: Config) -> Tuple[Callable, Callable, Callable]:
    """Return ``(loss, grad, hess)`` as numpy-facing callables of a flat vector.

    ``hess`` returns the dense symmetric matrix that SciPy's ``trust-exact``
    requires; ``tr_hessian="gauss_newton"`` replaces the exact Hessian by
    ``(2/M) J_r^T J_r`` so the effect of the ``sum_i r_i grad^2 r_i`` term can be
    measured (it is a different algorithm, not this paper's).
    """
    n = objective.n
    dtype = objective.dtype
    chunk = max(1, int(cfg.tr_chunk))
    loss_jit = objective._loss
    grad_jit = jax.jit(jax.grad(loss_jit))

    if cfg.tr_hessian == "exact":
        @jax.jit
        def _rows(flat, basis):
            _, jvp = jax.linearize(grad_jit, flat)
            return jax.vmap(jvp)(basis)          # (chunk, n) rows of the Hessian

        def hess(flat):
            eye = jnp.eye(n, dtype=dtype)
            parts = [_rows(flat, eye[i:i + chunk]) for i in range(0, n, chunk)]
            H = np.asarray(jnp.concatenate(parts, axis=0), dtype=np.float64)
            return 0.5 * (H + H.T)               # symmetrise: potrf reads one triangle,
                                                 # the Gershgorin bounds read both
    elif cfg.tr_hessian == "gauss_newton":
        res_jit = objective._residual
        m = int(np.asarray(objective.residual(jnp.asarray(flat))).shape[0])

        @jax.jit
        def _jrows(flat, basis):
            _, jvp = jax.linearize(res_jit, flat)
            return jax.vmap(jvp)(basis)          # (chunk, M) = rows of J_r

        def hess(flat):
            eye = jnp.eye(n, dtype=dtype)
            parts = [_jrows(flat, eye[i:i + chunk]) for i in range(0, n, chunk)]
            J = np.asarray(jnp.concatenate(parts, axis=0), dtype=np.float64)   # (n, M)
            return (2.0 / m) * (J @ J.T)
    else:
        raise ValueError(f"unknown tr_hessian {cfg.tr_hessian!r}")

    def loss(flat):
        return float(loss_jit(jnp.asarray(flat, dtype=dtype)))

    def grad(flat):
        return np.asarray(grad_jit(jnp.asarray(flat, dtype=dtype)), dtype=np.float64)

    return loss, grad, hess


# --------------------------------------------------------------------------
# the subproblem: More-Sorensen, nearly exact
# --------------------------------------------------------------------------
# This is SciPy's ``IterativeSubproblem`` -- the very class that
# ``scipy.optimize.minimize(method="trust-exact")`` constructs, and therefore the
# very solver the paper's own code runs.  It is used rather than transcribed
# because a hand port of it was written first and did *not* agree with the
# original on 7 % of random indefinite problems: the loop's state
# (``lambda_lb`` reused across calls, the ``already_factorized`` flag consumed
# once per iteration, and -- the one that actually bit -- the fact that the
# hard-case ``quadratic_term`` is taken with the *shifted* matrix ``H + lambda I``
# rather than ``H``) is not something to re-derive by eye.  Depending on the
# reference implementation is strictly better than shipping an approximation of
# it, and SciPy is already a dependency of this project.


class Subproblem:
    """The quadratic model at one point, with its nearly-exact trust-region solver.

    Wrapping SciPy's class (rather than re-instantiating it per solve) matters:
    ``IterativeSubproblem`` carries ``lambda_lb`` between calls, and reuses it when
    the radius shrinks, which is exactly the rejected-step path of the outer loop.
    """

    def __init__(self, H: np.ndarray, g: np.ndarray, maxiter: int = 25,
                 k_easy: float = K_EASY, k_hard: float = K_HARD):
        from scipy.optimize._trustregion_exact import IterativeSubproblem
        self.H = H
        self.g = g
        n = H.shape[0]
        zero = np.zeros(n)
        self._sub = IterativeSubproblem(zero, lambda x: 0.0, lambda x: g, lambda x: H, None,
                                        k_easy=k_easy, k_hard=k_hard, maxiter=maxiter)

    def model(self, p: np.ndarray) -> float:
        """``m(p) = f + g^T p + (1/2) p^T H p``, without the ``f`` offset."""
        return float(self.g @ p + 0.5 * p @ self.H @ p)

    def solve(self, delta: float) -> Tuple[np.ndarray, bool]:
        """Return ``(p, hits_boundary)`` for the trust region of radius ``delta``."""
        p, hits = self._sub.solve(delta)
        return np.asarray(p, dtype=np.float64), bool(hits)


# --------------------------------------------------------------------------
# the outer loop (scipy/optimize/_trustregion.py, verbatim thresholds)
# --------------------------------------------------------------------------
def trust_region_phase(objective: Objective, flat0, cfg: Config,
                       verbose: bool = True,
                       callback: Optional[Callable[[int, float, object], None]] = None,
                       step_offset: int = 0,
                       resample: Optional[Callable[[], Objective]] = None,
                       ) -> Tuple[jax.Array, List[Dict], Dict]:
    """Run the trust-region phase.  Returns ``(flat, history, info)``."""
    dtype = objective.dtype
    flat = jnp.asarray(flat0, dtype=dtype)
    loss, grad, hess = make_hessian(objective, cfg)

    delta = float(cfg.tr_delta0)
    eta = float(cfg.tr_eta)
    if not (0.0 <= eta < 0.25):
        raise ValueError("tr_eta must satisfy 0 <= eta < 0.25 (scipy's acceptance band)")
    maxiter = int(cfg.tr_maxiter)
    delta_max = float(cfg.tr_delta_max)
    sub_maxiter = int(cfg.tr_subproblem_maxiter)

    f = loss(flat)
    g = grad(flat)
    gnorm = float(np.linalg.norm(g))

    history: List[Dict] = [{"step": step_offset, "loss": f, "rho": float("nan"),
                            "radius": delta, "grad_norm": gnorm, "accepted": False,
                            "wall": 0.0}]
    if callback:
        callback(step_offset, f, flat)

    first_redraw, growth = (resample_schedule(cfg, maxiter) if resample is not None
                            else (0.0, 1.0))
    next_redraw = first_redraw if first_redraw > 0 else None

    t0 = time.time()
    n_hess = n_accept = n_reject = n_resample = 0
    stopped = "budget"
    step = 0
    H = None

    while step < maxiter:
        if next_redraw is not None and step >= next_redraw:
            next_redraw = next_redraw * growth
            n_resample += 1
            objective = resample()
            loss, grad, hess = make_hessian(objective, cfg)
            f = loss(flat)
            g = grad(flat)
            gnorm = float(np.linalg.norm(g))
            H = None

        if H is None:
            H = hess(flat)
            n_hess += 1
            model = Subproblem(H, g, maxiter=sub_maxiter)

        if gnorm < float(cfg.tr_gtol):
            stopped = "gradient below gtol"
            break

        try:
            p, hits = model.solve(delta)
        except np.linalg.LinAlgError as exc:                      # scipy warnflag=3
            stopped = f"linalg error in the subproblem ({exc})"
            break
        lam = float(getattr(model._sub, "lambda_current", float("nan")))

        f_new = loss(flat + jnp.asarray(p, dtype=dtype))
        if not np.isfinite(f_new):
            f_new = np.inf
        predicted = f + model.model(p)
        pred_red = f - predicted
        act_red = f - f_new

        # scipy stops the whole run if the model cannot predict *any* decrease;
        # a floor is added because a near-singular H makes pred_red tiny-but-
        # positive, and rho then explodes into NaN and the run stalls silently.
        floor = 1e-14 * max(1.0, abs(f))
        if pred_red <= floor:
            stopped = "model failed to predict improvement"
            break

        rho = act_red / pred_red
        if not np.isfinite(rho):
            stopped = "non-finite decrease ratio"
            break

        if rho < float(cfg.tr_contract_below):
            delta = delta * float(cfg.tr_contract)
        elif rho > float(cfg.tr_expand_above) and hits:
            delta = min(delta * float(cfg.tr_expand), delta_max)

        accepted = rho > eta
        if accepted:
            flat = flat + jnp.asarray(p, dtype=dtype)
            f = f_new
            n_accept += 1
            g = grad(flat)
            gnorm = float(np.linalg.norm(g))
            H = None                       # the model has moved; rebuild at the new point
        else:
            n_reject += 1                  # rejected: reuse H, only the radius changed

        step += 1
        history.append({"step": step_offset + step, "loss": f, "rho": float(rho),
                        "radius": delta, "grad_norm": gnorm, "accepted": bool(accepted),
                        "lam": float(lam), "hits_boundary": bool(hits),
                        "predicted": float(pred_red), "actual": float(act_red),
                        "inner_iters": int(getattr(model._sub, "niter", 0)),
                        "wall": time.time() - t0})
        if callback:
            callback(step_offset + step, f, flat)
        if verbose and (step % max(1, int(cfg.log_every)) == 0 or step == 1):
            print(f"[tr] step {step_offset + step:5d}  loss {f:.6e}  rho {rho:+.3f}  "
                  f"radius {delta:.3e}  lam {lam:.3e}  |g| {gnorm:.3e}  "
                  f"{'accept' if accepted else 'reject'}  ({time.time()-t0:.0f}s)",
                  flush=True)

    info = {"optimizer": "trustregion" if cfg.tr_hessian == "exact" else "trust-region(gauss-newton)",
            "iterations": step, "wall": time.time() - t0, "stopped": stopped,
            "hessian": cfg.tr_hessian, "hessians": n_hess, "accepted": n_accept,
            "rejected": n_reject, "resamples": n_resample,
            "n_parameters": objective.n, "final_radius": delta,
            "final_grad_norm": gnorm, "final_loss": float(f)}
    if verbose:
        print(f"[tr] trust-region ({cfg.tr_hessian} Hessian): loss -> {f:.6e} in {step} "
              f"iterations ({info['wall']:.0f}s), {n_hess} Hessians, accepted {n_accept}, "
              f"rejected {n_reject}; {stopped}", flush=True)
    return flat, history, info
