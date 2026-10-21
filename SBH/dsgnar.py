"""DSGNAR -- doubly-sketched Gauss-Newton with adaptive regularisation.

Vendored (2026-10-18) from Stationary/stationary/dsgnar.py, which is itself
vendored from Evolution_try/wave_pinn/optim/dsgnar.py. The algorithm and code
are unchanged; the only change is that `cfg` is duck-typed (any object with the
`dsgnar_*`, `seed`, `resample_every` and `log_every` attributes, built by
EEccz4_BH_puncture.py from its --dsgnar_* flags) instead of Stationary's
problem configuration class. The objective must provide `n`, `dtype`,
`_residual`, `_loss`, `residual`, `loss` and optionally `blocks`. History of
every choice below: Stationary/DSGNAR.md. The original module docstring follows.

, the doubly-sketched Gauss-Newton method with adaptive ratio.

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
1. **Loss units.**  The objective handed in is ``L = 0.5 * mean(r**2) = ||r||^2 / (2M)``
   (``flat_objective.FlatObjective`` scales the residual by ``sqrt(2M)`` to make that equal
   to ``total_loss``), whereas the paper and the reference code use ``1/2 ||r||^2``.  The
   sketched quadratic model therefore carries a ``1/(2M)``::

       pred(lam) = (1/(2M)) sum_i g_i^2 (S_i^2 + 2 lam) / (S_i^2 + lam)^2

   with ``g = S * (U^T r~)``.  This is exactly ``m(0) - m(p~(lam))`` for
   ``m(p) = (1/(2M)) ||r~ + J~ p||^2``, i.e. the reference code's ``pred`` divided by
   ``M``, and the config's ``dsgnar_stage1_ratio = 0.15`` / ``dsgnar_stage2_ratio = 0.5``
   are ratios of the actual to this decrease.  (Until 2026-10 the factor here was ``1/M``,
   from an earlier ``mean(r**2)`` convention, which over-predicted every decrease by
   exactly 2 and halved every ``rho``; the calibration below had been absorbing it.)

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
def count_sketch(mat, vec, key, s: int, k_hashes: int = N_HASHES, row_chunk: int = 0):
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
    # `row_chunk` splits the rows into blocks and sums the partial bucket sums.  Exactly the same
    # number -- segment_sum is a sum over rows and the hash draws are untouched -- but the
    # temporary `mat * signs` and the segment_sum workspace are `row_chunk x s` instead of
    # `rows x s`.  THIS is what the GPU refused next, after the JVP was chunked: at rows ~ 4e5
    # and s = 128 the temporaries are several GiB (measured: `Out of memory while trying to
    # allocate 8.87GiB`, executable `jit__where`, runs/pq_c100_vacF_dsgnar_probe_s128).
    block = int(row_chunk or 0)
    use_blocks = block > 0 and rows % block == 0 and block < rows
    for _ in range(k_hashes):
        key, k_sign, k_bucket = jax.random.split(key, 3)
        signs = jax.random.rademacher(k_sign, (rows,), dtype=vec.dtype) / jnp.sqrt(k_hashes)
        buckets = jax.random.randint(k_bucket, (rows,), 0, s)
        if not use_blocks:
            if mat is not None:
                acc_mat = acc_mat + jax.ops.segment_sum(
                    mat * signs.reshape((-1,) + (1,) * (mat.ndim - 1)), buckets, num_segments=s)
            acc_vec = acc_vec + jax.ops.segment_sum(vec * signs, buckets, num_segments=s)
            continue
        nb = rows // block
        sc = signs.reshape(nb, block)
        bc = buckets.reshape(nb, block)
        vc = vec.reshape(nb, block)
        if mat is not None:
            mc = mat.reshape((nb, block) + mat.shape[1:])
            scm = sc.reshape((nb, block) + (1,) * (mat.ndim - 1))

            def one_mat(i):
                return jax.ops.segment_sum(mc[i] * scm[i], bc[i], num_segments=s)

            acc_mat = acc_mat + jax.lax.map(one_mat, jnp.arange(nb)).sum(0)

        def one_vec(i):
            return jax.ops.segment_sum(vc[i] * sc[i], bc[i], num_segments=s)

        acc_vec = acc_vec + jax.lax.map(one_vec, jnp.arange(nb)).sum(0)
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


def jvp_columns(res_fn, flat_now, basis, chunk: int = 0):
    """``J(flat_now) @ basis.T`` for a batch of tangents: ``(s, n) -> (s, M)``.

    The linearisation is rebuilt at ``flat_now`` on every call.  That is not a
    stylistic choice: ``jax.linearize`` fixes its expansion point, so hoisting it
    out of the iteration loop silently trains every step against the Jacobian of
    the *initial* parameters -- measured at 12 % relative error after a single
    1e-2 step, large enough to make the Gauss-Newton model meaningless while
    still producing a plausible-looking loss curve.

    ``chunk > 0`` pushes the tangents through in blocks of that size instead of all
    ``s`` at once.  The reference implementation vmaps the whole sketch in one call, and on
    this problem that is what the GPU refused: the tangents of the *whole* residual --
    every collocation point, every equation component, and the second derivatives of
    lambda -- are materialised simultaneously, and at ``s = 128`` that is a single
    4.78 GiB allocation (measured, runs/pq_c100_vacF_dsgnar_probe_s128:
    ``RESOURCE_EXHAUSTED: Out of memory while trying to allocate 4.78GiB``).  The arithmetic
    is unchanged -- same columns, same order -- and the peak is set by ``chunk``, not by
    ``s``, which is what makes a useful sketch affordable.
    """
    _, jvp = jax.linearize(res_fn, flat_now)
    s = int(basis.shape[0])
    if not chunk or chunk >= s:
        return jax.vmap(jvp)(basis)
    # lax.map is a real sequential loop in the compiled program: the intermediates of one block
    # are freed before the next is traced, which is the whole point.  A Python loop would be
    # unrolled into one huge computation and XLA may keep all of it live -- so the block count
    # must NOT depend on the chunk dividing s.  It usually does not: the default sketch is
    # floor(n/3), which is 761 here (prime) and was 733 in Evolution_try.  Pad with ZERO
    # tangents instead, run the loop, and slice the padding off: J @ 0 = 0, so the columns are
    # the ones asked for.
    pad = (-s) % chunk
    if pad:
        basis = jnp.concatenate([basis, jnp.zeros((pad, basis.shape[1]), dtype=basis.dtype)], 0)
    blocks = basis.reshape((s + pad) // chunk, chunk, basis.shape[1])
    return jax.lax.map(lambda b: jax.vmap(jvp)(b), blocks).reshape(s + pad, -1)[:s]


def count_sketch_draws(key, rows: int, s: int, k_hashes: int, dtype):
    """The ``K`` sign vectors and bucket indices of :func:`count_sketch`, drawn once.

    Identical draws to ``count_sketch(mat, vec, key, s, k_hashes)`` -- same key splits, same
    order -- so the fused path below sketches with the very same ``C``.  Shapes ``(K, rows)``.
    """
    signs, buckets = [], []
    for _ in range(k_hashes):
        key, k_sign, k_bucket = jax.random.split(key, 3)
        signs.append(jax.random.rademacher(k_sign, (rows,), dtype=dtype) / jnp.sqrt(k_hashes))
        buckets.append(jax.random.randint(k_bucket, (rows,), 0, s))
    return jnp.stack(signs), jnp.stack(buckets)


def apply_count_sketch(mat, signs, buckets, s: int):
    """``C @ mat`` for ``mat`` of shape ``(rows, c)``, with pre-drawn hashes: ``(s, c)``."""
    out = jnp.zeros((s, mat.shape[1]), dtype=mat.dtype)
    for k in range(signs.shape[0]):
        out = out + jax.ops.segment_sum(mat * signs[k][:, None], buckets[k], num_segments=s)
    return out


def sketched_jvp_columns(res_fn, flat_now, basis, signs, buckets, s: int, chunk: int = 0):
    """``C J (Omega S)`` in one pass: ``(s_cols, n) -> (s, s_cols)``, never holding ``M x s``.

    :func:`jvp_columns` followed by :func:`count_sketch` materialises the ``M x s`` column
    matrix ``J Omega S`` -- at the production row count (~4.2e5) that is 0.4 GiB at
    ``s = 128`` and ~3.7 GiB at ``s = 1100 ~ d/2``, which is what kept the sketch at d/18
    here while Evolution_try measured ``d/3 -> d/2`` as worth three orders of magnitude.
    CountSketch is linear and acts column by column, so each block of ``chunk`` tangents
    can be pushed through the JVP and sketched immediately: the live workspace is
    ``M x chunk``, independent of ``s``.  Same draws, same columns -- the result equals the
    two-step path up to the grouping of sums (tests/test_dsgnar.py).
    """
    _, jvp = jax.linearize(res_fn, flat_now)
    n_cols = int(basis.shape[0])
    chunk = int(chunk) if chunk and chunk < n_cols else n_cols
    pad = (-n_cols) % chunk
    if pad:
        basis = jnp.concatenate([basis, jnp.zeros((pad, basis.shape[1]), dtype=basis.dtype)], 0)
    blocks = basis.reshape((n_cols + pad) // chunk, chunk, basis.shape[1])

    def one(b):
        cols = jax.vmap(jvp)(b)                          # (chunk, M)
        return apply_count_sketch(cols.T, signs, buckets, s)   # (s, chunk)

    sk = jax.lax.map(one, blocks)                        # (n_blocks, s, chunk)
    return jnp.transpose(sk, (1, 0, 2)).reshape(s, n_cols + pad)[:, :n_cols]


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

    The returned reduction is ``||r~||^2 - ||r~ + J~ p~||^2``, *unscaled*: the
    caller multiplies by ``1/(2M)``, the units of the objective
    ``0.5 * mean(r**2)`` it divides (see the module docstring).

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
def dsgnar_phase(objective, flat0, cfg,
                 verbose: bool = True,
                 callback: Optional[Callable[[int, float, object], None]] = None,
                 step_offset: int = 0,
                 resample: Optional[Callable[[], object]] = None,
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
    # 1/M per residual row; the model decrease of the objective 0.5 * mean(r**2) is
    # 0.5 * inv_m * (||r~||^2 - ||r~ + J~ p||^2), which is what `scale` multiplies.
    inv_m = 1.0 / float(objective.residual(jnp.asarray(flat0, dtype)).shape[0])
    scale = 0.5 * inv_m

    res_fn = objective._residual                      # jitted scalar-to-residual map
    # SBH addition (2026-10-20): 'rows' builds C J row-wise in the objective (the paper code's
    # construction; objective.sketch_rows), then applies the SRCT to the s x n result; 'cols'
    # (default) is the JVP construction below. Same J~ to round-off.
    rows_mode = (str(getattr(cfg, "dsgnar_jacobian", "cols")) == "rows"
                 and hasattr(objective, "sketch_rows"))
    max_seconds = float(getattr(cfg, "dsgnar_max_seconds", 0.0) or 0.0)
    # SBH addition (2026-10-21): 'none' drops the CountSketch on the residual rows (needs rows mode).
    # Near a nonzero-residual floor the sketched gradient J^T C^T C r is dominated by the sketch
    # noise J^T (C^T C - I) r ~ |J||r|/sqrt(s) once |J^T r| falls below it, and the step goes uphill
    # in ~half the draws however small the radius (tools/sketch_noise.py): the trust region
    # collapses. With the exact residual, (J Omega S)^T r is the projected TRUE gradient.
    exact_residual = (rows_mode and str(getattr(cfg, "dsgnar_residual_sketch", "count")) == "none"
                      and hasattr(objective, "gram_rows"))
    srct_apply = jax.jit(apply_srct)
    dct = None if rows_mode else _dct_matrix(n, dtype)   # constant: built once, not per iteration
    # The 25 trust-region probes are evaluated SEQUENTIALLY, not in one vmap.  A vmap of the
    # loss over the probes materialises the whole residual computation -- every collocation
    # point, every equation component, the second derivatives of lambda -- 25 times over: XLA's
    # memory analysis gives 0.49 GiB against 0.057 GiB for lax.map at 1/13 of the production
    # collocation count, i.e. the 8.87 GiB the GPU refused (the failure surfaced at
    # `rho_host = np.asarray(rhos)`, because the probe losses were computed lazily).  Same 25
    # evaluations, same numbers, one workspace.
    chunk = int(getattr(cfg, "dsgnar_chunk", 0) or 0)
    # no longer used by the phase: the fused sketch never forms the M x s matrix whose
    # CountSketch this blocked; kept in `info` so old and new records stay comparable
    row_chunk = int(getattr(cfg, "dsgnar_row_chunk", 0) or 0)
    k_hashes = max(1, int(getattr(cfg, "dsgnar_hashes", N_HASHES) or N_HASHES))

    # ---- per-condition weights (paper Algorithm 1 line 14, Appendix A.1) -----------------
    # CountSketch mixes the rows of EVERY condition into the same s buckets, so the paper keeps
    # the conditions on a comparable scale with a slow multiplicative update (alpha = 0.05).
    # Applied here as a per-row scale `rs` on the residual, passed to the jitted maps as an
    # ARGUMENT, so a change of weights costs no recompilation.  The objective each step is
    # judged on is the reweighted one, as in the paper; the history keeps the loss at the
    # ORIGINAL weights as well, which is the number comparable with the other optimisers.
    alpha_w = float(getattr(cfg, "dsgnar_reweight_alpha", 0.0) or 0.0)
    blocks = list(getattr(objective, "blocks", None) or [])
    n_rows = int(round(1.0 / inv_m))
    reweight = alpha_w > 0.0 and len(blocks) > 1 and sum(b[1] for b in blocks) == n_rows
    cond_w = np.ones(len(blocks)) if reweight else None
    sizes = [b[1] for b in blocks]
    ones_rs = jnp.ones((n_rows,), dtype)

    def rows_of(w):
        return jnp.asarray(np.repeat(np.sqrt(w), sizes), dtype)

    rs = rows_of(cond_w) if reweight else ones_rs

    def build(res_fn):
        """The jitted maps of one objective; `w` is the per-row scale."""
        loss_s = jax.jit(lambda f, w: 0.5 * jnp.mean((res_fn(f) * w) ** 2))
        batch_s = jax.jit(lambda flats, w: jax.lax.map(lambda x: loss_s(x, w), flats))
        # J~ = C J (Omega S) without the M x s intermediate; see sketched_jvp_columns
        jac_s = jax.jit(lambda f, b, sg, bk, w: sketched_jvp_columns(
            lambda x: res_fn(x) * w, f, b, sg, bk, s, chunk))
        return loss_s, batch_s, jac_s

    loss_s, loss_batch, sketched_jacobian = build(res_fn)
    fixed_sketch = bool(getattr(cfg, "dsgnar_fixed_sketch", False))
    reject_overshoot = bool(getattr(cfg, "dsgnar_reject_overshoot", False))
    line_scan = bool(getattr(cfg, "dsgnar_line_scan", True))
    overshoot_count = 0

    target_stage1 = float(cfg.dsgnar_stage1_ratio)
    target_stage2 = float(cfg.dsgnar_stage2_ratio)
    target_rho = target_stage1
    delta_min = float(cfg.dsgnar_delta_min)
    omega = float(cfg.dsgnar_omega)

    key = jax.random.PRNGKey(int(cfg.seed))
    # the paper's InitSketch (section 3.2.2): one sketch for the whole phase, if asked for
    _, fixed_srct_key, fixed_ck_key = jax.random.split(jax.random.PRNGKey(int(cfg.seed) + 1), 3)
    flat = jnp.asarray(flat0, dtype=dtype)
    radius = float(cfg.dsgnar_delta0)
    # HIGH-WATER MARK of the trust region, and the reference for the floor below.  It must be the
    # largest radius reached so far, not the latest accepted one: rejected steps shrink the radius
    # and accepted steps only preserve it, so "a fraction of the last accepted radius" is a
    # ratchet -- it decays with the radius it is supposed to protect, which is exactly what the
    # first version did (pq_r100_phihyb025_floor: 33 rejections interleaved with accepts took the
    # radius from 1.557 to 6.8e-14 anyway, and the phase stopped at 508 of 2000 iterations with
    # the same loss as the run without any floor).  Anchoring to the high-water mark makes the
    # collapse impossible: the radius can never fall below floor * delta0.
    hwm_radius = float(cfg.dsgnar_delta0)
    r_lo = r_hi = radius
    lam_final = float(omega)
    step = 0

    loss = float(loss_s(flat, rs))                    # the objective steps are judged on
    loss_base = float(loss_s(flat, ones_rs)) if reweight else loss
    history: List[Dict] = [{"step": step_offset, "loss": loss_base, "radius": radius,
                            "lam": lam_final, "rho": 0.0, "resid": float("nan"),
                            "accepted": False, "wall": 0.0}]
    if callback:
        callback(step_offset, loss_base, flat)
    if verbose and (reweight or fixed_sketch or reject_overshoot or not line_scan):
        print(f"[dsgnar] paper options: reweight alpha={alpha_w if reweight else 0:g} over "
              f"{len(blocks)} conditions, fixed sketch {fixed_sketch}, reject overshoot "
              f"{reject_overshoot}, line scan {line_scan}, calibrate "
              f"{bool(getattr(cfg, 'dsgnar_calibrate', True))}, targets "
              f"{cfg.dsgnar_stage1_ratio:g}/{cfg.dsgnar_stage2_ratio:g}, delta_min "
              f"{cfg.dsgnar_delta_min:g}", flush=True)

    lam_history: List[float] = []
    t0 = time.time()
    t_jvp = t_probe = 0.0
    stopped = "budget"
    accepted_count = rejected_count = 0

    resample_every = int(cfg.resample_every) if resample is not None else 0

    while step < int(cfg.dsgnar_steps):
        flat = jnp.asarray(flat, dtype=dtype)
        if not np.isfinite(loss):
            stopped = "diverged"
            break

        # A redraw of the collocation sample changes the estimated objective, not
        # the objective itself, so the trust-region radius and the regularisation
        # carry over; only the quantities that depend on the points are rebuilt.
        if resample_every and step > 0 and step % resample_every == 0:
            objective = resample()
            res_fn = objective._residual
            loss_s, loss_batch, sketched_jacobian = build(res_fn)
            inv_m = 1.0 / float(objective.residual(flat).shape[0])
            scale = 0.5 * inv_m
            loss = float(loss_s(flat, rs))

        # --- Algorithm 1 line 2: fresh sketch operators this iteration --------
        key, k_srct, k_ck = jax.random.split(key, 3)
        if fixed_sketch:
            k_srct, k_ck = fixed_srct_key, fixed_ck_key
        srct = make_srct(k_srct, n, s)
        B = None if rows_mode else srct_columns(srct, n, s, dtype, dct_matrix=dct)   # (s, n); B.T is Omega S

        residual = res_fn(flat) * rs                  # r in R^M, in full (reweighted)

        # --- Algorithm 2 lines 10-12: J~ = C J (Omega S) and r~ = C r ---------
        # jvp_batch's output has the s sketch components along its first axis, so
        # its transpose has one row per residual point, which is what C eats; the
        # same hash draws then give r~ as well, keeping the two sketches identical.
        tic = time.time()
        signs, buckets = count_sketch_draws(k_ck, int(residual.shape[0]), s, k_hashes, dtype)
        if exact_residual:
            # G = A^T A, b = A^T r with A = J Omega S over all rows; an equivalent square system
            # J~ = Sigma V^T, r~ = Sigma^-1 V^T b has J~^T J~ = G and J~^T r~ = b, so everything below
            # (secular solve, model decrease, PCHIP radius) is unchanged and exact in the subspace.
            G_ex, b_ex = objective.gram_rows(flat, srct, s, rs)
            ev, V_ex = jnp.linalg.eigh(0.5 * (G_ex + G_ex.T))
            sig = jnp.sqrt(jnp.maximum(ev, 0.0))
            j_sketch = sig[:, None] * V_ex.T
            vb = V_ex.T @ b_ex
            r_tilde = jnp.where(sig > 1e-14 * jnp.max(sig), vb / jnp.where(sig > 0, sig, 1.0), 0.0)
        elif rows_mode:
            j_sketch = srct_apply(objective.sketch_rows(flat, signs, buckets, s, rs), srct)   # (s, s)
        else:
            j_sketch = sketched_jacobian(flat, B, signs, buckets, rs)      # (s, s)
        if not exact_residual:
            r_tilde = apply_count_sketch(residual[:, None], signs, buckets, s)[:, 0]
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
        # The model decrease comes back unscaled; `scale = 1/(2M)` puts it in the
        # units of the objective 0.5 * mean(r**2) the ratio divides.
        probe_sk, probe_pred = jax.vmap(lambda lam: step_and_pred(sing, g, v_t, lam))(probe_lams)
        probe_pred = probe_pred * scale
        probe_steps = jax.vmap(lambda p: lift_srct(p, srct, n))(probe_sk)
        tic = time.time()
        probe_losses = loss_batch(flat[None, :] + probe_steps, rs)
        t_probe += time.time() - tic

        finite = jnp.all(jnp.isfinite(probe_losses))
        act_red = loss - probe_losses

        # ------------------------------------------------------------ model calibration
        # The sketched model is systematically OVER-OPTIMISTIC on this problem, and by a lot:
        # measured at n = 2285, s = 128 against the EXACT quadratic model of the same step
        # (loss 3.76e-02), the linear term is -4.94e-03 and the true model decrease 1.25e-03,
        # while the sketch predicts 1.51e-02 -- a factor 12.  Every ratio then lands at ~0.02
        # against the 0.15 target, the PCHIP always picks the smallest radius, and the trust
        # region walks 3^-28 = 1e-13 to delta_min in ~28 iterations: that, not the step or the
        # accept rule, is what stopped all three runs.  The 25 probes already measure the TRUE
        # loss, so the over-estimate is measurable for free -- divide the predictions by the
        # median measured/predicted ratio.  Only over-optimism is corrected (the factor is
        # clamped to >= 1): an under-confident model needs no help, and scaling it up would make
        # the trust region reckless.
        calib = 1.0
        if bool(getattr(cfg, "dsgnar_calibrate", True)):
            act_np = np.asarray(act_red, dtype=np.float64)
            pred_np = np.asarray(probe_pred, dtype=np.float64)
            ok = np.isfinite(act_np) & np.isfinite(pred_np) & (act_np > 0.0) & (pred_np > 0.0)
            if bool(ok.any()):
                calib = float(np.median(pred_np[ok] / act_np[ok]))
                if not np.isfinite(calib) or calib < 1.0:
                    calib = 1.0
                calib = min(calib, 1.0e6)
            probe_pred = probe_pred / calib

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
        final_pred = final_pred / calib      # same calibration as the probes
        final_step = lift_srct(final_sk, srct, n)

        # ---------------------------------------------------------------- safeguarded length
        # The trust region lives in the SKETCH space, so the length that reaches the parameters
        # is that of `Omega S p`, which the model does not control.  Measured on this problem
        # (pq_c100_vacH_dsgnar_fromadam, pq_r10_phihyb025_dsgnar): the first proposed step raises
        # the loss by about the predicted decrease (rho ~ -1), the next 27 are rejected as well,
        # and 3^-28 = 1e-13 is exactly the walk from delta0 = 1 to delta_min -- so the phase
        # exits after ~28 iterations having accepted almost nothing.  That is a LENGTH mismatch,
        # not a direction one: the same direction, scaled down, is a descent step.
        #
        # So scan fractions of the SAME lifted direction and take the first that lowers the loss.
        # The algorithm's accept rule is unchanged ("the objective strictly fell", Algorithm 1
        # line 8) and rho is used only where the paper uses it, to move the trust region; the
        # accepted fraction then says by how much the model overestimated the usable step.
        bt = float(getattr(cfg, "dsgnar_backtrack", 0.5) or 0.5)
        if not (0.0 < bt < 1.0):
            bt = 0.5
        # Include two very short lengths: they separate a curvature error (rho -> 1 as the step
        # shrinks, because the model is exact in the linear regime) from a scale/convention error
        # (rho stays far from 1 however short the step, so the gradient projection or the
        # prediction's units are wrong).
        fracs = [1.0]
        for _ in range(max(0, int(getattr(cfg, "dsgnar_backtrack_steps", 0) or 0))):
            if fracs[-1] * bt > 0.0:
                fracs.append(fracs[-1] * bt)
        fracs += [1.0e-3, 1.0e-2]
        if not line_scan:
            fracs = [1.0]                 # the paper: the step, evaluated once, nothing else

        new_loss, final_act, final_rho, used = loss, -np.inf, -np.inf, 0.0
        scan = []
        for f in fracs:
            cand = float(loss_s(flat + f * final_step, rs))
            scan.append((f, cand))
            if np.isfinite(cand) and cand < loss:
                new_loss, final_act, used = cand, loss - cand, f
                # the model decrease is linear in the step to leading order, so the predicted
                # decrease at fraction f is f * pred(1): enough for the radius move below
                denom = f * float(final_pred) + float(jnp.finfo(dtype).tiny)
                final_rho = final_act / denom
                if not np.isfinite(final_rho):
                    final_rho = -np.inf
                break
        if verbose and (step < 3 or (step + 1) % max(1, int(cfg.log_every)) == 0):
            pred1 = float(final_pred)
            bits = []
            for f, l in scan:
                rho_f = (loss - l) / (f * pred1 + float(jnp.finfo(dtype).tiny))
                bits.append(f"{f:g}:{l:.3e}(rho {rho_f:+.3f})")
            print("[dsgnar]   step scan: " + "  ".join(bits)
                  + f"   current {loss:.3e}, accepted {used:g}, model pred {pred1:.3e}, "
                    f"calibration {calib:.2f}", flush=True)

        # --- Algorithm 1 lines 8-11: accept iff the objective strictly fell ---
        accepted = bool(final_act > 0.0)
        # The paper (section 3.4, "Accepting the step"): a step whose ratio OVERSHOOTS the target
        # by more than a small margin is rejected too, and the radius is EXPANDED -- an
        # over-aggressive step says the trust region is too small, and taking it would trade
        # the low-regularisation region the first stage is looking for for a quick decrease.
        if accepted and reject_overshoot and final_rho > target_rho + RHO_HI_MARGIN:
            accepted = False
            overshoot_count += 1
        if accepted:
            flat = flat + used * final_step
            loss = new_loss
            # The radius follows the ACCEPTED length -- but an accepted step never SHRINKS it.
            # The paper moves the radius to the PCHIP radius `delta_star`, which is right when the
            # model's ratio is trustworthy; here it is not (the calibration factor runs 60-80 in
            # the middle of a run), so "the objective fell but the model thinks it should have
            # fallen more" shrank the radius on *success* and walked it to delta_min after 76
            # iterations -- loss 8.3e-06 with 72 accepted / 4 rejected, i.e. descent that stopped
            # for no reason the loss knew about.  With the safeguarded length a too-large radius
            # is safe (the scan trims it), so the radius only shrinks on a REJECTED step.
            nxt = delta_star if used >= 1.0 else max(delta_min, used * delta_star)
            radius = max(radius, nxt) if bool(getattr(cfg, "dsgnar_grow_on_accept", True)) else nxt
            hwm_radius = max(hwm_radius, float(radius))
            lam_final = lam_star
            accepted_count += 1
        else:
            rejected_count += 1
            radius = r_hi if final_rho > target_rho + RHO_HI_MARGIN else r_lo
            # ... but not without a floor.  Rejections are the only thing that shrinks the radius
            # now, and each one divides it by WINDOW_SCALE: 32 of them walk 1.557 to 1e-14 and the
            # phase stops on `radius below threshold` having spent only a fraction of its budget --
            # measured on both finished runs (r10: 266 of 600 iterations, 30 rejections; c100: 508
            # of 2000, 32 rejections), i.e. 3^32 ~ 1e15.  With the guarded length there is no need
            # for the radius to go to zero: it may shrink to a small fraction of the radius that
            # last produced an accepted step, and the line search does the rest.
            floor = float(getattr(cfg, "dsgnar_radius_floor", 0.0) or 0.0)
            if floor > 0.0:
                radius = max(radius, floor * hwm_radius)

        step += 1
        lam_history.append(max(lam_final, omega))

        # --- Algorithm 1 line 14: UpdateWeights from this iteration's residual -------------
        if reweight:
            r_np = np.asarray(residual, dtype=np.float64)
            ells = np.array([float(np.sum(c * c)) for c in np.split(r_np, np.cumsum(sizes)[:-1])])
            # SBH fix (2026-10-20): a condition whose residual is identically zero (the positivity
            # penalty while chi, alpha > 0) must not take part. Otherwise (mean/(0+eps))^alpha
            # makes its weight grow every iteration and the normalisation drives every REAL
            # condition's weight to zero (w8_gn, job 1181118: weights 1e-11..7 after 40
            # iterations, the reweighted loss 1e-13 while the actual loss did not move).
            live = ells > 0.0
            if live.sum() >= 2:
                w_hat = cond_w.copy()
                w_hat[live] = cond_w[live] * (np.mean(ells[live]) / (ells[live] + 1.0e-8)) ** alpha_w
                w_hat[live] = w_hat[live] / np.mean(w_hat[live]) * np.mean(cond_w[live])
                cond_w = w_hat
            rs = rows_of(cond_w)
            loss = float(loss_s(flat, rs))            # the objective just changed
            loss_base = float(loss_s(flat, ones_rs))
        else:
            loss_base = loss

        # --- Algorithm 1 lines 12-15: stopping and the target-ratio update ----
        if radius < STOP_MARGIN * delta_min:
            stopped = "radius below threshold"
        elif max_seconds > 0 and time.time() - t0 >= max_seconds:
            stopped = "time budget"
        elif step >= int(cfg.dsgnar_steps):
            stopped = "budget"
        else:
            target_rho = update_target_ratio(lam_history, target_rho,
                                             target_stage2, target_stage1)

        wall = time.time() - t0
        history.append({"step": step_offset + step, "loss": loss_base, "loss_w": loss,
                        "resid": float(jnp.linalg.norm(residual)) * float(np.sqrt(inv_m)),
                        "rho": float(final_rho), "lam": lam_star, "radius": radius,
                        "accepted": accepted, "frac": used, "wall": wall})
        if callback:
            callback(step_offset + step, loss_base, flat)
        if verbose and (step % max(1, int(cfg.log_every)) == 0 or step == 1):
            wtxt = (f"  w {cond_w.min():.2g}..{cond_w.max():.2g}" if reweight else "")
            print(f"[dsgnar] step {step_offset + step:6d}  loss {loss_base:.6e}{wtxt}  "
                  f"rho {float(final_rho):+.3f}  lam {lam_star:.3e}  radius {radius:.3e}  "
                  f"target {target_rho:.2f}  ({wall:.1f}s)", flush=True)
        if stopped != "budget":
            break

    info = {
        "optimizer": "dsgnar",
        "jacobian": "rows" if rows_mode else "cols",
        "residual_sketch": "none" if exact_residual else "count",
        "chunk": chunk,
        "row_chunk": row_chunk,
        "iterations": step,
        "wall": time.time() - t0,
        "stopped": stopped,
        "sketch": s,
        "hashes": k_hashes,
        "probes": q,
        "n_parameters": n,
        "n_residuals": int(round(1.0 / inv_m)),
        "loss_final": float(loss_base),
        "loss_final_reweighted": float(loss),
        "reweight_alpha": alpha_w if reweight else 0.0,
        "condition_weights": ({b[0]: float(w) for b, w in zip(blocks, cond_w)} if reweight
                              else None),
        "fixed_sketch": fixed_sketch,
        "reject_overshoot": reject_overshoot,
        "overshoot_rejections": overshoot_count,
        "line_scan": line_scan,
        "accepted": accepted_count,
        "rejected": rejected_count,
        "stage2_reached": bool(target_rho >= target_stage2),
        "final_radius": float(radius),
        "final_lambda": float(lam_final),
        "final_rho": float(history[-1]["rho"]),
        "time_jvp": t_jvp,
        "time_probe": t_probe,
    }
    if verbose:
        print(f"[dsgnar] DSGNAR: loss {history[0]['loss']:.6e} -> {float(loss_base):.6e} in "
              f"{step} iterations ({info['wall']:.1f}s); {stopped}; s={s}, "
              f"accepted {accepted_count}, rejected {rejected_count}, "
              f"stage2={info['stage2_reached']}", flush=True)
        print(f"[dsgnar] timing: batched JVPs {t_jvp:.1f}s, probe evaluations "
              f"{t_probe:.1f}s, other {info['wall'] - t_jvp - t_probe:.1f}s", flush=True)
    return flat, history, info
