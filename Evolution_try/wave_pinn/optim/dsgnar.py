"""DSGNAR, the doubly-sketched Gauss-Newton method with adaptive ratio.

The algorithm is Webb, Jerad and Cartis, *An optimisation framework for the
well-conditioned training of physics-informed neural networks*
(arXiv:2607.02194v1); the reference implementation is
``github.com/wephy/physics-informed-neural-networks`` (``src/pinn/optimiser.py``).

What the method does
--------------------
A Gauss-Newton step needs ``(J^T J + lam I)^-1 J^T r``, and a trust-region
method needs that step for many candidate radii ``Delta`` -- i.e. for many
``lam``.  DSGNAR buys both at once with a *square* sketch of the Jacobian,
``J~ = C J Omega S`` in ``R^{s x s}``:

* ``C`` is a CountSketch on the residual dimension (``K`` signed hashes, each
  row going to one of ``s`` buckets, summed over ``K``);
* ``Omega S`` is a subsampled randomised cosine transform on the parameter
  dimension: ``D`` (random signs) then ``Pi`` (random permutation) then an
  orthonormal DCT-II, keeping ``s`` columns.

One SVD of ``J~`` then gives the LM step in closed form for *every* ``lam``, so
the probe loop of Algorithm 3 costs ``O(q s^2)`` rather than ``q`` linear
solves.  Steps are lifted back with the exact adjoint of ``Omega S`` (which the
orthonormal DCT makes an isometry), evaluated in the full space, and the probe
whose measured decrease ratio ``rho`` sits just above the target ``rho*`` sets
the next trust-region radius.

Deviations from the paper and from the reference code
-----------------------------------------------------
1. **Loss units.**  This project's objective is ``L = mean(r**2) = ||r||^2 / M``,
   whereas the paper and the reference code use ``1/2 ||r||^2``.  The sketched
   quadratic model therefore carries a ``1/M``::

       pred(lam) = (1/M) sum_i g_i^2 (S_i^2 + 2 lam) / (S_i^2 + lam)^2

   with ``g = S * (U^T r~)``.  This is exactly ``m(0) - m(p~(lam))`` for
   ``m(p) = (1/M) ||r~ + J~ p||^2``, i.e. the textbook Eq. 14 denominator, and it
   is *twice* the reference code's ``pred``.  Consequently the config's
   ``dsgnar_stage1_ratio = 0.15`` and ``dsgnar_stage2_ratio = 0.5`` -- which are
   stated in textbook units -- are used unchanged and are NOT the reference
   code's ``0.075``: the two expressions differ by exactly the factor of two.
   Because acceptance tests only the *sign* of ``rho``, this choice affects the
   radius schedule but not whether a descent step is taken.
2. **Resampling.**  ``C``, ``D``, ``Pi`` and ``S`` are redrawn every iteration
   from a PRNG key carried in the loop's state.  This follows Algorithm 1
   line 2 and the reference code; section 3.2.2 of the paper instead describes a
   single fixed sketch (``InitSketch``).  Resampling is the unbiased choice and
   makes the sketch error average out over the run.
3. **No per-condition weighting.**  Algorithm 2 scales condition ``m``'s rows by
   ``alpha_m = sqrt(w_m / |X_m|)`` and Algorithm 4 re-weights the conditions
   against each other.  This problem has a single condition (the PDE residual,
   with the initial and periodic conditions hard-coded into the ansatz and the
   optional penalties folded in as extra rows of the same vector), so
   :func:`update_weights` is a no-op whenever ``M == 1`` -- the general rule is
   implemented, the degenerate case simply changes nothing.
4. **Probe count.**  ``q = 25`` probes on ``[Delta/3, 3 Delta]`` (24 intervals,
   over the 25 grid points), as in the reference code's ``n_probes = 24``.
5. **Stage switch.**  Algorithm 5 is implemented over a log10 ``lam`` window
   with the reference code's ``slope > 1e-4`` and ``corr > 0.1`` tests.  The
   reference code additionally waits ``lambda_grace_period = 80`` iterations;
   that has no config field here, so a shorter grace period of
   :data:`LAMBDA_GRACE` iterations is used and the switch is one-way, exactly as
   in the paper.
6. **Stopping.**  The loop stops once the radius falls below
   ``10 * cfg.dsgnar_delta_min`` (Algorithm 1 says ``Delta < Delta_min``; the
   reference code uses the 10x margin) or after ``cfg.dsgnar_steps`` iterations.

Everything else -- the CountSketch definition, the SRCT adjoint, the secular
Newton solve for ``lam`` (using ``g``, never ``g**2``), the monotonised ratio
envelope, the PCHIP last-crossing radius choice and the accept-iff-``rho > 0``
rule -- follows Algorithm 3 and the reference code.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.fft
import numpy as np

from ..config import Config, resample_schedule
from ..losses import Objective

__all__ = ["dsgnar_phase", "update_weights", "update_target_ratio"]


# --------------------------------------------------------------------------
# module constants
# --------------------------------------------------------------------------
N_PROBES: int = 25          # q: probe radii in [Delta/3, 3 Delta], geometrically spaced
N_NEWTON: int = 60          # secular-Newton iterations per probe (reference: 60-80)
WINDOW_SCALE: float = 3.0   # probes span [Delta / WINDOW_SCALE, Delta * WINDOW_SCALE]
PCHIP_GRID: int = 512       # resolution of the crossing search in log-radius
N_HASHES: int = 4           # K: CountSketch hashes (paper: 2 or 4 typical)
RHO_HI_MARGIN: float = 0.1  # reject-but-overshoot band; see Algorithm 1 line 11
STOP_MARGIN: float = 10.0   # radius < STOP_MARGIN * delta_min terminates
LAMBDA_GRACE: int = 20      # iterations before the stage-2 switch may fire
LAMBDA_WINDOW: int = 30     # W: length of the log-lambda window
MIN_SLOPE: float = 1.0e-4   # tau_s, per iteration of log10(lambda)
MIN_CORRELATION: float = 0.1  # tau_c
_LAM_STEP_CLIP: float = 1.0e6  # guard against a Newton step dividing by an underflowed phi'


# --------------------------------------------------------------------------
# CountSketch: C J and C r   (paper Eq. 16, reference `_count_sketch`)
# --------------------------------------------------------------------------
def count_sketch(mat, vec, key, s: int, k_hashes: int = N_HASHES):
    """Apply the CountSketch ``C`` to the rows of ``mat`` and to ``vec``.

    ``C`` has exactly ``K`` non-zeros per column, at ``(h_k(j), j)`` with value
    ``epsilon_k(j) / sqrt(K)``; summing ``K`` independent hashes is what keeps
    the estimator's variance down (``K = 1`` is the plain CountSketch, whose
    per-draw variance from hash collisions is large).  The sketch is never
    materialised: only the ``K`` sign vectors and bucket indices are drawn, and
    the bucketed sum is a ``segment_sum``.  ``mat`` has one row per residual, so
    passing ``(J Omega S).T`` and ``r`` together sketches both with the *same*
    hash functions, which is what makes ``r~`` and ``J~`` consistent.  Passing
    ``mat=None`` sketches only the vector, so the full residual never has to be
    padded into a matrix just to borrow the hash draws.
    """
    rows = vec.shape[0]
    acc_mat = None if mat is None else jnp.zeros((s,) + mat.shape[1:], dtype=mat.dtype)
    acc_vec = jnp.zeros((s,), dtype=vec.dtype)
    for _ in range(k_hashes):
        key, k_sign, k_bucket = jax.random.split(key, 3)
        signs = jax.random.rademacher(k_sign, (rows,), dtype=vec.dtype) / jnp.sqrt(k_hashes)
        buckets = jax.random.randint(k_bucket, (rows,), 0, s)
        if mat is not None:
            acc_mat = acc_mat + jax.ops.segment_sum(
                mat * signs.reshape((-1,) + (1,) * (mat.ndim - 1)), buckets, num_segments=s)
        acc_vec = acc_vec + jax.ops.segment_sum(vec * signs, buckets, num_segments=s)
    return acc_mat, acc_vec


# --------------------------------------------------------------------------
# SRCT: Omega S and its exact adjoint   (paper Eq. 17, reference `_make_srct`)
# --------------------------------------------------------------------------
def make_srct(key, n: int, s: int) -> Tuple:
    """Draw ``(signs, perm)`` for ``Omega S = (D Pi F) S`` with ``S`` a scatter.

    ``F`` is the orthonormal DCT-II, so ``D Pi F`` is orthogonal; ``Pi`` is a full
    permutation and ``S`` keeps the first ``s`` columns, which is the "permute
    then subsample" equivalent of selecting ``s`` distinct columns of ``D Pi F``.
    Because ``F`` is orthonormal, the lift below is the exact adjoint and
    ``||lift(y)|| == ||y||``.
    """
    k_sign, k_perm = jax.random.split(key)
    signs = jax.random.rademacher(k_sign, (n,))
    perm = jax.random.permutation(k_perm, n)[:s]
    return signs, perm


def _dct_matrix(n: int, dtype):
    """The orthonormal DCT-II matrix ``M`` with ``dct(x) == M @ x``.

    Closed form rather than ``vmap(dct, eye)``; the values agree with
    ``jax.scipy.fft.dct(..., type=2, norm="ortho")`` to machine precision.
    """
    k = jnp.arange(n, dtype=dtype).reshape((-1, 1))
    i = jnp.arange(n, dtype=dtype).reshape((1, -1))
    scale = jnp.where(k == 0.0, jnp.sqrt(1.0 / n), jnp.sqrt(2.0 / n))
    return scale * jnp.cos(jnp.pi * (i + 0.5) * k / n)

def srct_columns(srct, n: int, s: int, dtype, dct_matrix=None):
    """The explicit ``(s, n)`` matrix ``B`` whose transpose is ``Omega S``.

    With ``Omega S = D Pi F S`` and ``S`` the scatter that keeps frequency
    ``k`` at column ``k``, the composite is the ``(n, s)`` matrix

        ``(Omega S)[i, k] = signs[i] * M[perm[k], i]``

    where ``M`` is the DCT-II matrix.  Returning the transpose, ``B[k, i]``, lets
    a batch of JVPs along the rows of ``B`` produce ``J @ B.T = J Omega S`` -- the
    doubly-sketched Jacobian -- without ever forming the full ``M x n`` Jacobian.

    ``dct_matrix`` lets the caller pass in an ``M`` it built once: the matrix is
    ``n x n`` and constant, so rebuilding it every iteration would recompute a
    few million cosines for nothing.
    """
    signs, perm = srct
    M = _dct_matrix(n, dtype) if dct_matrix is None else dct_matrix
    return M[jnp.asarray(perm)] * signs[None, :]


def apply_srct(mat, srct):
    """Rows ``mat``: ``mat -> mat (Omega S)``, shape ``(rows, n) -> (rows, s)``.

    Reference form: scale each column by its sign, transform, then permute and
    keep the first ``s`` frequency columns.  This is ``mat @ srct_columns(...).T``
    to machine precision.
    """
    signs, perm = srct
    return jax.scipy.fft.dct(mat * signs[None, :], type=2, norm="ortho",
                             axis=-1)[:, jnp.asarray(perm)]


def lift_srct(y, srct, n: int):
    """Lift ``y in R^s`` to ``(Omega S) y in R^n``: the exact adjoint of ``B``.

    ``(Omega S) = (D Pi F S)^T = D F^T Pi^T S^T``.  Applied to a column ``y``:
    scatter component ``k`` to coordinate ``perm[k]`` (the ``Pi^T S^T`` pair),
    apply the inverse orthonormal transform (``idct``, the adjoint of the ``dct``
    used on rows), and scale by the signs.  The scatter must come *first*: doing
    the inverse transform on the short vector and scattering afterwards is a
    different, non-isometric map.  Because ``F`` is orthonormal this lift is an
    isometry -- ``||lift_srct(y)|| == ||y||`` -- which is what makes the
    trust-region norm and the secular equation on ``p~`` meaningful.
    """
    signs, perm = srct
    y = jnp.asarray(y)
    full = jnp.zeros((n,), dtype=y.dtype).at[jnp.asarray(perm)].set(y)
    full = jax.scipy.fft.idct(full, type=2, norm="ortho")
    return full * signs


def jvp_columns(res_fn, flat_now, basis):
    """``J(flat_now) @ basis.T`` for a batch of tangents: ``(s, n) -> (s, M)``.

    The linearisation is rebuilt at ``flat_now`` on every call.  That is not a
    stylistic choice: ``jax.linearize`` fixes its expansion point, so hoisting it
    out of the iteration loop silently trains every step against the Jacobian of
    the *initial* parameters -- measured at 12 % relative error after a single
    1e-2 step, large enough to make the Gauss-Newton model meaningless while
    still producing a plausible-looking loss curve.
    """
    _, jvp = jax.linearize(res_fn, flat_now)
    return jax.vmap(jvp)(basis)


# --------------------------------------------------------------------------
# Algorithm 3, first half: the secular Newton solve   (reference _solve_subproblems)
# --------------------------------------------------------------------------
def solve_subproblems(sing, g, radii, n_newton: int = N_NEWTON, omega: float = 0.0):
    """Return the ``lam >= 0`` with ``||g / (sing^2 + lam)|| == radii``.

    Newton on ``phi(lam) = ||p~(lam)|| - delta``.  ``sing`` is ``Sigma`` and
    ``g = Sigma * (U^T r~)``, so ``p~ = g / (Sigma^2 + lam)`` in the ``V`` basis
    and ``||p~|| = ||g / (Sigma^2 + lam)||`` -- using ``g`` (never ``g**2``)
    keeps a factor of ``Sigma`` out of the derivative and is what makes
    Algorithm 3 line 7 consistent.  Steps are clipped only to stop a Newton
    iterate running away when ``phi'`` underflows.
    """
    tiny = jnp.finfo(jnp.asarray(sing).dtype).tiny
    omega = jnp.asarray(omega, dtype=jnp.asarray(sing).dtype)
    radii = jnp.asarray(radii, dtype=jnp.asarray(sing).dtype).reshape((-1, 1))
    lam = jnp.full((radii.shape[0],), omega, dtype=radii.dtype)
    sing2 = sing * sing
    for _ in range(n_newton):
        denom = sing2[None, :] + lam[:, None]
        p = g[None, :] / denom
        p_norm = jnp.linalg.norm(p, axis=1)
        phi = p_norm - radii[:, 0]
        d_phi = -(jnp.sum(p * p / (denom + tiny), axis=1) + tiny) / (p_norm + tiny)
        step = jnp.clip(phi / d_phi, -_LAM_STEP_CLIP, _LAM_STEP_CLIP)
        lam = jnp.maximum(lam - step, 0.0)
    return lam


def step_and_pred(sing, g, v_t, lam):
    """LM step ``p~(lam) = -V (g / (Sigma^2 + lam))`` and the model decrease.

    The returned reduction is *unscaled*: the caller multiplies by ``1/M``.
    This project's objective is ``mean(r**2)``, so the sketched model is
    ``m(p) = (1/M) ||r~ + J~ p||^2`` and

        ``m(0) - m(p~) = (1/M) sum g_i^2 (S_i^2 + 2 lam) / (S_i^2 + lam)^2``

    which is exactly the textbook Eq. 14 denominator (see the module docstring).
    The reference code omits the ``1/M`` because its loss already carries a ``1/2``
    and its model is ``(1/2)||r~ + J~ p||^2``; the two conventions differ by a
    factor of two, which is why the config's ratios are the textbook ones.

    The negative convention of Algorithm 1 line 6 (and the reference code) is
    used, not the positive one of Algorithm 3 line 10; only the former is a
    descent direction.
    """
    tiny = jnp.finfo(jnp.asarray(sing).dtype).tiny
    denom = sing * sing + lam
    p_sk = -(v_t.T @ (g / (denom + tiny)))
    pred = jnp.sum(g * g * (sing * sing + 2.0 * lam) / ((denom + tiny) ** 2))
    return p_sk, pred


# --------------------------------------------------------------------------
# PCHIP: monotone cubic interpolation of rho against log(Delta)
# --------------------------------------------------------------------------
def _pchip_edge(h0, h1, m0, m1):
    """One-sided three-point endpoint derivative with the shape limiter.

    This is Moler's ``pchiptx`` rule, as used by ``scipy.interpolate``: a plain
    harmonic-mean endpoint (the textbook Fritsch-Carlson interior rule continued
    past the last knot) is not shape-preserving and lets the interpolant bulge
    around the ends of the probe window, which is exactly where the crossing
    search must be trustworthy.
    """
    d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    mask = np.sign(d) != np.sign(m0)
    mask2 = (np.sign(m0) != np.sign(m1)) & (np.abs(d) > 3.0 * np.abs(m0))
    d = np.where(mask, 0.0, d)
    d = np.where((~mask) & mask2, 3.0 * m0, d)
    return d


def pchip_slopes(x, y):
    """Fritsch-Carlson derivatives at the knots (``numpy``; called outside jit).

    The shape-preserving limiter -- harmonic means in the interior, Moler's
    limited one-sided estimate at the ends -- is what makes the interpolant
    usable for a crossing search: a plain cubic spline can overshoot and invent a
    crossing that the data does not contain.  The values match
    ``scipy.interpolate.PchipInterpolator``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    h = np.diff(x)
    delta = np.diff(y) / np.where(h == 0.0, 1.0, h)
    m = np.zeros_like(y)
    if y.size < 2:
        return m
    if y.size == 2:
        m[:] = delta[0]
        return m
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    same = (np.sign(delta[:-1]) * np.sign(delta[1:]) > 0.0) & (delta[:-1] != 0.0) & (delta[1:] != 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        harmonic = (w1 + w2) / (w1 / delta[:-1] + w2 / delta[1:])
    m[1:-1] = np.where(same, harmonic, 0.0)
    m[0] = _pchip_edge(h[0], h[1], delta[0], delta[1])
    m[-1] = _pchip_edge(h[-1], h[-2], delta[-1], delta[-2])
    return m


def pchip_eval(x, y, xq):
    """Evaluate the PCHIP interpolant of ``(x, y)`` at ``xq`` (outside jit)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xq = np.asarray(xq, dtype=np.float64)
    m = pchip_slopes(x, y)
    idx = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, x.size - 2)
    h = x[idx + 1] - x[idx]
    h = np.where(h == 0.0, 1.0, h)
    t = (xq - x[idx]) / h
    h00 = (1.0 + 2.0 * t) * (1.0 - t) ** 2
    h10 = t * (1.0 - t) ** 2
    h01 = t * t * (3.0 - 2.0 * t)
    h11 = t * t * (t - 1.0)
    return h00 * y[idx] + h10 * h * m[idx] + h01 * y[idx + 1] + h11 * h * m[idx + 1]


# --------------------------------------------------------------------------
# Algorithm 3: LambdaSolve -- pick Delta* from the probe window
# --------------------------------------------------------------------------
def lambda_solve(probe_radii, probe_rhos, target_rho: float,
                 r_lo: float, r_hi: float, grid_size: int = PCHIP_GRID) -> float:
    """The radius whose fitted ratio is (just) above ``target_rho``.

    ``probe_rhos`` are clipped and then monotonised downwards -- a single
    spurious ratio (from a near-zero predicted decrease) would otherwise
    dominate the fit -- and the *last* crossing of the ``>= target`` indicator
    is taken, i.e. the largest radius that still reaches the target.  When the
    whole envelope sits on one side of the target, the nearest extreme of the
    window is used rather than extrapolating beyond it.

    The interpolant is evaluated in ``log``-radius: the probes are geometrically
    spaced, so ``log``-space is where the nodes are uniform and well
    conditioned.
    """
    tiny = np.finfo(np.float64).tiny
    log_probe = np.log(np.maximum(np.asarray(probe_radii, dtype=np.float64), tiny))
    order = np.argsort(log_probe)
    x = log_probe[order]
    y = np.minimum.accumulate(np.clip(np.asarray(probe_rhos, dtype=np.float64), -1.0, 1.0)[order])
    fine = np.linspace(np.log(max(r_lo, tiny)), np.log(max(r_hi, tiny)), grid_size)
    fine_rho = pchip_eval(x, y, fine)
    above = fine_rho >= target_rho
    if not above.any():
        return float(r_lo)
    if above.all():
        return float(r_hi)
    crossings = np.flatnonzero(above[:-1] & ~above[1:])
    last = int(crossings[-1])
    return float(np.exp(0.5 * (fine[last] + fine[last + 1])))


# --------------------------------------------------------------------------
# Algorithm 4: UpdateWeights -- a no-op for a single residual condition
# --------------------------------------------------------------------------
def update_weights(condition_losses, weights, alpha: float = 0.05, eps: float = 1.0e-8):
    """Residual-based condition re-weighting (paper Appendix A.1).

    No-op when there is one condition: with ``M == 1`` both the loss ratios and
    the normalisation are ``1``, so the weights are returned unchanged (up to the
    floating-point round trip).  This problem has exactly one condition, so in
    practice this function never fires.
    """
    if len(condition_losses) <= 1:
        return list(weights)
    losses = np.asarray(condition_losses, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mean_loss = float(np.mean(losses))
    new = w * (mean_loss / (losses + eps)) ** alpha
    return list(new / np.mean(new))


# --------------------------------------------------------------------------
# Algorithm 5: UpdateTargetRatio -- raise rho* once lambda has bottomed out
# --------------------------------------------------------------------------
def update_target_ratio(lam_history: List[float], current: float, stage2: float,
                        target_stage1: float, window: int = LAMBDA_WINDOW,
                        min_slope: float = MIN_SLOPE,
                        min_correlation: float = MIN_CORRELATION) -> float:
    """Stage-1 -> stage-2 switch when ``log10(lambda)`` has turned upwards.

    Both the fitted slope and its correlation with the iteration index must
    clear their thresholds: the slope says ``lambda`` is climbing on average,
    the correlation says it is doing so consistently rather than noisily.  The
    window is in ``log10`` because ``lambda`` moves over tens of orders of
    magnitude; the switch is one-way.
    """
    if len(lam_history) < window or len(lam_history) < LAMBDA_GRACE:
        return current
    if current >= stage2:
        return current
    w = np.maximum(np.asarray(lam_history[-window:], dtype=np.float64), np.finfo(np.float64).tiny)
    log_w = np.log10(w)
    x = np.arange(window, dtype=np.float64)
    slope = float(np.polyfit(x, log_w, 1)[0])
    var = float(np.var(x) * np.var(log_w))
    if var <= 0.0:
        return current
    corr = float(np.cov(x, log_w, bias=True)[0, 1] / np.sqrt(var))
    if np.isnan(corr):
        return current
    if slope > min_slope and corr > min_correlation:
        return float(stage2)
    return float(target_stage1)


# --------------------------------------------------------------------------
# the phase
# --------------------------------------------------------------------------
def dsgnar_phase(objective: Objective, flat0, cfg: Config,
                 verbose: bool = True,
                 callback: Optional[Callable[[int, float, object], None]] = None,
                 step_offset: int = 0,
                 resample: Optional[Callable[[], Objective]] = None,
                 ) -> Tuple[jax.Array, List[Dict], Dict]:
    """Run DSGNAR for ``cfg.dsgnar_steps`` iterations.  Returns ``(flat, history, info)``.

    ``history`` has one dict per iteration with at least ``{"step", "loss"}``,
    plus ``resid``, ``rho``, ``lam``, ``radius``, ``accepted`` and ``wall``.
    ``callback`` is called once per iteration with the loss at the *current*
    iterate (after any accepted step), so that the driver's loss curve is the
    loss of the parameters actually carried forward.

    The residual object is jitted once; a single ``jax.linearize`` provides the
    Jacobian-vector product used to fill the ``(s, n)`` column-sketch matrix
    ``B = (Omega S)^T``, and ``jax.vmap`` over that product gives
    ``J @ B.T = J Omega S`` in one batched call -- without ever materialising the
    ``M x n`` Jacobian, which the full ``jax.jacfwd`` route would build by paying
    for ``n`` forward tangents instead of ``s``.
    """
    dtype = objective.dtype
    n = int(objective.n)
    s = int(cfg.dsgnar_sketch) if cfg.dsgnar_sketch and cfg.dsgnar_sketch > 0 else max(1, n // 3)
    s = min(s, n)
    q = N_PROBES
    n_newton = N_NEWTON
    scale = 1.0 / float(objective.residual(jnp.asarray(flat0, dtype)).shape[0])

    res_fn = objective._residual                      # jitted scalar-to-residual map
    loss_fn = objective._loss                         # jitted scalar loss
    dct = _dct_matrix(n, dtype)                       # constant: built once, not per iteration
    loss_batch = jax.jit(jax.vmap(loss_fn))
    sketched_jacobian = jax.jit(lambda flat_now, basis: jvp_columns(res_fn, flat_now, basis))

    target_stage1 = float(cfg.dsgnar_stage1_ratio)
    target_stage2 = float(cfg.dsgnar_stage2_ratio)
    target_rho = target_stage1
    delta_min = float(cfg.dsgnar_delta_min)
    omega = float(cfg.dsgnar_omega)

    key = jax.random.PRNGKey(int(cfg.seed))
    flat = jnp.asarray(flat0, dtype=dtype)
    radius = float(cfg.dsgnar_delta0)
    r_lo = r_hi = radius
    lam_final = float(omega)
    step = 0

    loss = objective.loss(flat)
    history: List[Dict] = [{"step": step_offset, "loss": loss, "radius": radius,
                            "lam": lam_final, "rho": 0.0, "resid": float("nan"),
                            "accepted": False, "wall": 0.0}]
    if callback:
        callback(step_offset, loss, flat)

    lam_history: List[float] = []
    t0 = time.time()
    t_jvp = t_probe = 0.0
    stopped = "budget"
    accepted_count = rejected_count = resample_count = 0

    first_redraw, growth = (resample_schedule(cfg, int(cfg.dsgnar_steps)) if resample is not None
                            else (0.0, 1.0))
    next_redraw = first_redraw if first_redraw > 0 else None

    while step < int(cfg.dsgnar_steps):
        flat = jnp.asarray(flat, dtype=dtype)
        if not np.isfinite(loss):
            stopped = "diverged"
            break

        # A redraw of the collocation sample changes the estimated objective, not
        # the objective itself, so the trust-region radius and the regularisation
        # carry over; only the quantities that depend on the points are rebuilt.
        #
        # KNOWN COST.  `resample()` builds a fresh `Objective`, whose jitted
        # functions close over the batch, so every redraw pays a recompilation of
        # the batched Jacobian-vector product -- tens of seconds at s = 733.  On
        # the fresh 400-iteration run that is roughly a fifth of the wall time.
        # The fix is to make the batch an argument of the jitted residual rather
        # than a closure constant (`_residual(flat, batch)`), which would retrace
        # only on a change of shape; it is not done here because the SSBroyden
        # path needs a one-argument callable for Crunch's `minimize` and would
        # still close over the batch.
        if next_redraw is not None and step >= next_redraw:
            next_redraw = next_redraw * growth
            resample_count += 1
            objective = resample()
            res_fn = objective._residual
            loss_fn = objective._loss
            loss_batch = jax.jit(jax.vmap(loss_fn))
            sketched_jacobian = jax.jit(lambda f, b: jvp_columns(res_fn, f, b))
            scale = 1.0 / float(objective.residual(flat).shape[0])
            loss = objective.loss(flat)

        # --- Algorithm 1 line 2: fresh sketch operators this iteration --------
        key, k_srct, k_ck = jax.random.split(key, 3)
        srct = make_srct(k_srct, n, s)
        B = srct_columns(srct, n, s, dtype, dct_matrix=dct)   # (s, n); B.T is Omega S

        residual = res_fn(flat)                       # r in R^M, in full

        # --- Algorithm 2 lines 10-12: J~ = C J (Omega S) and r~ = C r ---------
        # jvp_batch's output has the s sketch components along its first axis, so
        # its transpose has one row per residual point, which is what C eats; the
        # same hash draws then give r~ as well, keeping the two sketches identical.
        tic = time.time()
        j_cols = sketched_jacobian(flat, B).T         # (M, s) = J (Omega S)
        j_sketch, r_tilde = count_sketch(j_cols, residual, k_ck, s, k_hashes=N_HASHES)
        t_jvp += time.time() - tic

        # --- Algorithm 1 line 4: one SVD serves every lambda ------------------
        u_mat, sing, v_t = jnp.linalg.svd(j_sketch, full_matrices=False)
        g = sing * (u_mat.T @ r_tilde)

        # --- Algorithm 3 line 2: geometrically spaced radius probes -----------
        cur_r = max(radius, delta_min)
        r_lo = max(cur_r / WINDOW_SCALE, delta_min)
        r_hi = max(cur_r * WINDOW_SCALE, delta_min)
        probe_radii = jnp.geomspace(r_lo, r_hi, q)
        probe_lams = solve_subproblems(sing, g, probe_radii, n_newton, omega)

        # --- Algorithm 3 lines 10-12: lift and evaluate every probe -----------
        # The model decrease comes back unscaled; `scale = 1/M` puts it in the
        # units of the mean-squared objective the ratio divides.
        probe_sk, probe_pred = jax.vmap(lambda lam: step_and_pred(sing, g, v_t, lam))(probe_lams)
        probe_pred = probe_pred * scale
        probe_steps = jax.vmap(lambda p: lift_srct(p, srct, n))(probe_sk)
        tic = time.time()
        probe_losses = loss_batch(flat[None, :] + probe_steps)
        t_probe += time.time() - tic

        finite = jnp.all(jnp.isfinite(probe_losses))
        act_red = loss - probe_losses
        rhos = jnp.where(finite, act_red / (probe_pred + jnp.finfo(dtype).tiny),
                         -jnp.inf)

        # --- final (PCHIP) radius and its lambda ------------------------------
        rho_host = np.asarray(rhos, dtype=np.float64)
        delta_star = lambda_solve(probe_radii, rho_host, target_rho, r_lo, r_hi)
        lam_star = float(solve_subproblems(sing, g, jnp.atleast_1d(delta_star),
                                           n_newton, omega)[0])
        # Reuse the probe prediction when the PCHIP radius happens to be one of
        # the probe radii; otherwise evaluate the model once more rather than
        # opening a trust-region radius whose denominator is not the one used to
        # choose it.
        match = jnp.argmin(jnp.abs(probe_lams - lam_star))
        if bool(jnp.abs(probe_lams[match] - lam_star) <= 1.0e-12 * max(lam_star, omega)):
            final_sk, final_pred = probe_sk[match], probe_pred[match]
        else:
            final_sk, final_pred = step_and_pred(sing, g, v_t, lam_star)
            final_pred = final_pred * scale
        final_step = lift_srct(final_sk, srct, n)
        new_loss = float(loss_fn(flat + final_step))
        final_act = loss - new_loss
        final_rho = final_act / (float(final_pred) + float(jnp.finfo(dtype).tiny))
        if not np.isfinite(new_loss) or not np.isfinite(final_rho):
            final_rho = -np.inf
            new_loss = loss

        # --- Algorithm 1 lines 8-11: accept iff the objective strictly fell ---
        # (equivalently rho > 0; NOT rho >= target), then move the trust region.
        accepted = bool(final_act > 0.0 and np.isfinite(final_rho))
        if accepted:
            flat = flat + final_step
            loss = new_loss
            radius = delta_star
            lam_final = lam_star
            accepted_count += 1
        else:
            rejected_count += 1
            radius = r_hi if final_rho > target_rho + RHO_HI_MARGIN else r_lo

        step += 1
        lam_history.append(max(lam_final, omega))

        # --- Algorithm 1 lines 12-15: stopping and the target-ratio update ----
        if radius < STOP_MARGIN * delta_min:
            stopped = "radius below threshold"
        elif step >= int(cfg.dsgnar_steps):
            stopped = "budget"
        else:
            target_rho = update_target_ratio(lam_history, target_rho,
                                             target_stage2, target_stage1)

        wall = time.time() - t0
        history.append({"step": step_offset + step, "loss": loss,
                        "resid": float(jnp.linalg.norm(residual)) * float(np.sqrt(scale)),
                        "rho": float(final_rho), "lam": lam_star, "radius": radius,
                        "accepted": accepted, "wall": wall})
        if callback:
            callback(step_offset + step, loss, flat)
        if verbose and (step % max(1, int(cfg.log_every)) == 0 or step == 1):
            print(f"[dsgnar] step {step_offset + step:6d}  loss {loss:.6e}  "
                  f"rho {float(final_rho):+.3f}  lam {lam_star:.3e}  radius {radius:.3e}  "
                  f"target {target_rho:.2f}  ({wall:.1f}s)", flush=True)
        if stopped != "budget":
            break

    info = {
        "optimizer": "dsgnar",
        "iterations": step,
        "wall": time.time() - t0,
        "stopped": stopped,
        "sketch": s,
        "hashes": N_HASHES,
        "probes": q,
        "n_parameters": n,
        "n_residuals": int(round(1.0 / scale)),
        "loss_final": float(loss),
        "accepted": accepted_count,
        "rejected": rejected_count,
        "stage2_reached": bool(target_rho >= target_stage2),
        "resamples": resample_count,
        "final_radius": float(radius),
        "final_lambda": float(lam_final),
        "final_rho": float(history[-1]["rho"]),
        "time_jvp": t_jvp,
        "time_probe": t_probe,
    }
    if verbose:
        print(f"[dsgnar] DSGNAR: loss {history[0]['loss']:.6e} -> {float(loss):.6e} in "
              f"{step} iterations ({info['wall']:.1f}s); {stopped}; s={s}, "
              f"accepted {accepted_count}, rejected {rejected_count}, "
              f"stage2={info['stage2_reached']}, resamples={resample_count}", flush=True)
        print(f"[dsgnar] timing: batched JVPs {t_jvp:.1f}s, probe evaluations "
              f"{t_probe:.1f}s, other {info['wall'] - t_jvp - t_probe:.1f}s", flush=True)
    return flat, history, info
