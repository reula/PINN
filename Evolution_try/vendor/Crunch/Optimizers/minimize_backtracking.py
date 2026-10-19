"""JAX-compatible BFGS optimizer with SSBroyden inverse-Hessian update.

MERGED VERSION (2026-10-17) of the three copies that had diverged:
  * C. Palenzuela's file of 2026-10-16 (md5 77cad12e): strong Wolfe with a_init,
    SCORE, initial_scale, per-call ``stats``.  This is the base; with every new
    option off the result is bit-identical to it.
  * Boson's fork (md5 a3e48534): ``reuse_first_trial`` (Armijo), the counters
    ``ngev / nfull / nskip / nreset / h_scale`` and ``scale_invariant``.
  * BH (md5 59960127): ``rescale_carried``.
All new options default to off.  See CRUNCH_MERGE_2026-10-17.md.

The BFGS loop and line search run as Python for-loops so no XLA loop is
ever compiled.  The caller must supply two thin closure wrappers around
module-level @jax.jit functions:

    fun_val(w)      -> scalar loss         (JIT-compiled once, for line search)
    fun_val_grad(w) -> (scalar, gradient)  (JIT-compiled once, for BFGS updates)

Because the underlying @jax.jit objects are fixed Python objects, JAX's
dispatch cache is hit on every call — no re-tracing per iteration.
"""
import math
import jax
import jax.numpy as jnp
from typing import NamedTuple, Any


class OptimizeResult(NamedTuple):
    x: Any
    fun: Any
    nit: Any
    hess_inv: Any
    nfev: int = 0        # loss-only evaluations (line search)
    njev: int = 0        # loss+gradient evaluations (line search + BFGS)
    stats: Any = None    # per-call diagnostics (dict of Python floats/ints), see minimize()
    # Boson-fork fields (same meaning as there):
    ngev: Any = 0        # = njev (loss+gradient evaluations), Boson's name
    nfull: Any = 0       # iterations whose accepted step was exactly alpha = 1
    h_scale: Any = 1.0   # factor applied to the carried H at it == 0 (rescale_carried; 1.0 = none)
    nskip: Any = 0       # iterations whose H update was skipped (y.s below the curvature guard)
    nreset: Any = 0      # 1 if H was reset to the identity at it == 0 (y.s below the guard)


# Line searches this module implements. Callers that request one by name
# should check membership here rather than trusting an unknown option to be
# honoured (an older copy of this file silently ignores unknown keys).
LINE_SEARCHES = ("armijo", "wolfe")
# Inverse-Hessian updates this module implements ('ssbroyden' is an alias of
# 'ssbroyden2'). 'score' = SSBroyden with SCORE's decrement-coupled shifted
# secant pair and self-concordant first trial (Si, An, Ma & Yan,
# arXiv:2608.04206, Algorithm 1); it requires line_search='wolfe'.
UPDATE_METHODS = ("bfgs", "ssbroyden", "ssbroyden2", "score")

# Feature flags for callers (check with getattr(module, NAME, False)).
RESCALE_CARRIED = True   # options["rescale_carried"] (BH, PINN_LESSONS 17.4)
SCALE_INVARIANT = True   # options["scale_invariant"] (Boson, PINN_LESSONS 18)
REUSE_FIRST_TRIAL = True # options["reuse_first_trial"] (Boson, Armijo only)


# ---------------------------------------------------------------------------
# SSBroyden inverse-Hessian update  (arXiv:2405.04230, eqs. 10–23)
# ---------------------------------------------------------------------------
def _ssbroyden_update(H, s, y, g_prev, alpha):
    n = s.shape[0]
    eps = jnp.array(1e-14, dtype=s.dtype)

    ys  = jnp.dot(y, s)
    Hy  = H @ y
    yHy = jnp.dot(y, Hy)

    sHs = -alpha * jnp.dot(s, g_prev)
    b   = sHs / (ys + eps)
    h   = yHy / (ys + eps)
    a   = h * b - 1.0

    a_pos = jnp.maximum(a, 0.0)
    c     = jnp.sqrt(a_pos / (a_pos + 1.0 + eps))
    rho_m   = jnp.minimum(1.0, h * (1.0 - c))
    theta_m = (rho_m - 1.0) / (a_pos + eps)
    theta_p = 1.0 / (rho_m + eps)
    theta_n = jnp.where(jnp.abs(b) > eps, (1.0 - b) / b, jnp.zeros((), dtype=b.dtype))
    theta   = jnp.clip(theta_n, theta_m, theta_p)
    sigma   = 1.0 + a_pos * theta

    tau1 = jnp.minimum(1.0, ys / (sHs + eps))

    abs_sigma   = jnp.abs(sigma) + eps
    sigma_power = abs_sigma ** (-1.0 / (n - 1))
    tau2_pos    = tau1 * jnp.minimum(sigma_power, 1.0 / (jnp.abs(theta) + eps))
    tau2_neg    = jnp.minimum(tau1 * sigma_power, abs_sigma)
    tau2        = jnp.where(theta > 0.0, tau2_pos, tau2_neg)
    tau2        = jnp.maximum(tau2, eps)

    phi1 = (1.0 - theta) / (sigma + eps)

    sqrt_yHy = jnp.sqrt(jnp.maximum(yHy, eps))
    v        = sqrt_yHy * (s / (ys + eps) - Hy / (yHy + eps))

    H_new = ((H - jnp.outer(Hy, Hy) / (yHy + eps)
                + phi1 * jnp.outer(v, v)) / tau2
             + jnp.outer(s, s) / (ys + eps))

    return jnp.where(ys > eps, H_new, H)


def _bfgs_update(H, s, y):
    eps = jnp.array(1e-14, dtype=s.dtype)
    ys  = jnp.dot(y, s)
    Hy  = H @ y
    yHy = jnp.dot(y, Hy)
    rho = 1.0 / (ys + eps)
    H_new = (H
             - rho * jnp.outer(H @ y, s)
             - rho * jnp.outer(s, y @ H)
             + rho * (yHy * rho + 1.0) * jnp.outer(s, s))
    return jnp.where(ys > eps, H_new, H)


# ---------------------------------------------------------------------------
# Strong-Wolfe line search  (Nocedal & Wright, Algorithms 3.5 + 3.6)
# ---------------------------------------------------------------------------
def _cubic_min(a, fa, dfa, b, fb, dfb):
    """Minimiser of the cubic interpolating (a, fa, dfa) and (b, fb, dfb).

    Returns None when the cubic has no usable minimiser in the interval
    (caller falls back to bisection).  N&W eq. (3.59).
    """
    d1 = dfa + dfb - 3.0 * (fa - fb) / (a - b)
    rad = d1 * d1 - dfa * dfb
    if rad < 0.0:
        return None
    d2 = math.copysign(math.sqrt(rad), b - a)
    denom = dfb - dfa + 2.0 * d2
    if denom == 0.0:
        return None
    return b - (b - a) * (dfb + d2 - d1) / denom


def _quad_min(a_lo, f_lo, d_lo, a_hi, f_hi):
    """Minimiser of the quadratic through (a_lo, f_lo, d_lo) and (a_hi, f_hi)
    (N&W eq. 3.58).  Returns None when the quadratic is not convex."""
    h = a_hi - a_lo
    denom = 2.0 * (f_hi - f_lo - d_lo * h)
    if denom <= 0.0:
        return None
    return a_lo - d_lo * h * h / denom


def _cubic_min_values(a_lo, f_lo, d_lo, a1, f1, a2, f2):
    """Minimiser of the cubic through (a_lo, f_lo) with slope d_lo there and
    the two value-only points (a1, f1), (a2, f2) (N&W eq. 3.43, the
    backtracking cubic).  Returns None when it has no usable minimiser."""
    t1, t2 = a1 - a_lo, a2 - a_lo
    if t1 == 0.0 or t2 == 0.0 or t1 == t2:
        return None
    r1 = f1 - f_lo - d_lo * t1
    r2 = f2 - f_lo - d_lo * t2
    det = t1 * t1 * t2 * t2 * (t2 - t1)
    if det == 0.0:
        return None
    b = (t2 ** 3 * r1 - t1 ** 3 * r2) / det
    c = (t1 ** 2 * r2 - t2 ** 2 * r1) / det
    if abs(c) < 1e-300:
        return None if b <= 0.0 else a_lo - d_lo / (2.0 * b)
    rad = b * b - 3.0 * c * d_lo
    if rad < 0.0:
        return None
    t = (-b + math.copysign(math.sqrt(rad), c)) / (3.0 * c)   # root with m'' > 0
    return a_lo + t


# Smallest fraction of the bracket width a zoom trial may sit from either
# end.  N&W's textbook 10% band turned a bracket [0, 1] whose minimiser is at
# ~1e-5 (an overshooting quasi-Newton step on a stiff loss) into 15 rounds of
# plain bisection.  1e-3 keeps the "bracket must shrink" guarantee while
# letting the interpolant jump straight to a tiny step; much smaller bands
# (1e-5, 1e-6) let the zoom stall next to the low end on Rosenbrock (5-7x
# more evaluations), so this is a tuned value, not a formality.
_WOLFE_SAFEGUARD = 1e-3


def _secant_min(a1, d1, a2, d2):
    """Zero of the slope from the two slopes (a1, d1), (a2, d2): the
    minimiser of the quadratic they define.  None unless convex."""
    if a1 == a2 or d2 - d1 == 0.0:
        return None
    if (d2 - d1) / (a2 - a1) <= 0.0:
        return None
    return a2 - d2 * (a2 - a1) / (d2 - d1)


def _interp_safeguarded(a_lo, f_lo, d_lo, a_hi, f_hi, d_hi, extra=None, backoff=False,
                        prev_lo=None):
    """Interpolated trial point inside [a_lo, a_hi] (either order; a_lo is
    the end with the lower loss and a known slope).

    Candidates, first one inside the bracket wins: the cubic through both
    ends when the slope at a_hi is known; the cubic through the low end's
    value and slope, the high end's value and a further value-only point
    `extra` (a, f); the quadratic through the low end's value and slope and
    the high end's value; bisection.  When the high end has blown up (loss
    orders of magnitude above the low end -- an overshooting step on a stiff
    loss) the polynomials through it are meaningless: retreating (`backoff`)
    the quadratic is used first, since its minimiser scales like 1/f_hi;
    advancing (the low end passed sufficient decrease but still descends)
    the secant through the slopes at the low end and at the previous low
    end `prev_lo` (a, f, d) is used, i.e. the function is extrapolated from
    the good side instead of interpolated towards the blown-up side.

    The result is kept _WOLFE_SAFEGUARD x width from both ends, and in
    `backoff` mode (the last trial failed sufficient decrease, so the search
    is retreating towards a_lo) it is also kept within the half of the
    bracket next to a_lo, so each failed trial at least halves the bracket
    (Dennis & Schnabel's backtracking safeguard)."""
    lo, hi = (a_lo, a_hi) if a_lo < a_hi else (a_hi, a_lo)
    width = hi - lo
    blowup = f_hi - f_lo > 100.0 * (abs(f_lo) + 1e-300)

    def cubic_slopes():
        if d_hi is not None and math.isfinite(d_hi):
            return _cubic_min(a_lo, f_lo, d_lo, a_hi, f_hi, d_hi)

    def cubic_values():
        if extra is not None and math.isfinite(extra[1]):
            return _cubic_min_values(a_lo, f_lo, d_lo, a_hi, f_hi, extra[0], extra[1])

    def quad():
        return _quad_min(a_lo, f_lo, d_lo, a_hi, f_hi)

    def secant():
        if prev_lo is not None and math.isfinite(prev_lo[2]):
            return _secant_min(prev_lo[0], prev_lo[2], a_lo, d_lo)

    if blowup:
        order = (quad, cubic_slopes, cubic_values) if backoff else (secant, quad, cubic_slopes, cubic_values)
    else:
        order = (cubic_slopes, cubic_values, quad)
    a_new = 0.5 * (lo + hi)
    for cand in order:
        a_c = cand()
        if a_c is not None and math.isfinite(a_c) and lo <= a_c <= hi:
            a_new = a_c
            break
    a_min, a_max = lo + _WOLFE_SAFEGUARD * width, hi - _WOLFE_SAFEGUARD * width
    if backoff:
        if a_lo < a_hi:
            a_max = min(a_max, a_lo + 0.5 * width)
        else:
            a_min = max(a_min, a_lo - 0.5 * width)
    return min(max(a_new, a_min), a_max)


def _strong_wolfe(fun_val, fun_val_grad, x, f, g, p, c1, c2, max_ls, alphamin, alphamax,
                  grad_first=True, a_init=1.0):
    """Strong-Wolfe line search along p from x (Nocedal & Wright Alg. 3.5/3.6).

    a_init is the first trial step (1 for a plain quasi-Newton search; SCORE
    passes its self-concordant candidate 1/(1+lambda_k)).

    Returns (alpha, x_new, f_new, g_new).  If no point satisfying both Wolfe
    conditions is found within max_ls trials, returns the best
    sufficient-decrease point seen; if none, the lowest-loss point seen.

    Evaluation policy -- the gradient is only computed where the algorithm
    can use it:
      * the first trial (alpha = 1) is a loss+gradient evaluation when
        grad_first is True (the previous iteration accepted a step >= 1, so
        the full step will most likely be accepted again and its gradient is
        needed anyway), otherwise a cheap loss-only probe, upgraded to
        loss+gradient only if it passes sufficient decrease;
      * expansion trials (alpha doubled) need the slope and use loss+gradient;
      * zoom trials are loss-only until one satisfies sufficient decrease,
        and only that point gets a gradient (for the curvature test).
    Points that fail sufficient decrease therefore never cost a gradient.
    """
    phi0  = float(f)
    dphi0 = float(jnp.dot(g, p))
    dtype = x.dtype

    def x_at(a):
        return x + jnp.array(a, dtype=dtype) * p

    def eval_val(a):
        return float(fun_val(x_at(a)))

    def eval_grad(a):
        xa = x_at(a)
        fa, ga = fun_val_grad(xa)
        phi_a = float(fa)
        dphi_a = float(jnp.dot(ga, p)) if math.isfinite(phi_a) else math.nan
        return phi_a, dphi_a, (a, xa, fa, ga)

    nfev = 0
    best = None          # best sufficient-decrease point (lowest phi); has a gradient
    lowest = None        # (phi, a) lowest finite loss seen at all (last-resort fallback)

    def armijo_ok(a, phi_a):
        return phi_a <= phi0 + c1 * a * dphi0

    def wolfe_ok(dphi_a):
        return abs(dphi_a) <= -c2 * dphi0

    def note_value(a, phi_a):
        nonlocal lowest
        if math.isfinite(phi_a) and (lowest is None or phi_a < lowest[0]):
            lowest = (phi_a, a)

    def record(a, phi_a, pt):
        nonlocal best
        note_value(a, phi_a)
        if math.isfinite(phi_a) and armijo_ok(a, phi_a) and (best is None or phi_a < best[0]):
            best = (phi_a, pt)

    def fallback():
        if best is not None:
            return best[1]
        # No sufficient-decrease point: lowest loss seen (gradient computed
        # now), else a tiny step so the caller's curvature guard can reject it
        # instead of propagating NaN.
        a = lowest[1] if lowest is not None else alphamin
        return eval_grad(a)[2]

    # zoom() only ever returns a freshly evaluated trial point (or the
    # fallback), so the bracket ends are tracked by value only; a_lo = 0 (the
    # current x) is therefore a valid low end.  The low end always has a
    # slope; the high end's slope is None when it was a loss-only trial.
    #
    # Two regimes, told apart by the last trial's outcome:
    #   * backing off an overshoot (last trial failed sufficient decrease):
    #     trials are loss-only -- a failed trial never needs its gradient, and
    #     the cubic through the low end's slope and two values is accurate;
    #   * refining inside the sufficient-decrease region (last trial passed
    #     but failed the curvature test): trials are loss+gradient straight
    #     away -- an accepted trial needs its gradient anyway, and the slope
    #     at both ends gives the better cubic.  Evaluating such a trial
    #     loss-only first would just add a wasted loss evaluation.
    def zoom(a_lo, phi_lo, dphi_lo, a_hi, phi_hi, dphi_hi):
        nonlocal nfev
        extra = None                 # most recently discarded value-only trial (a, phi)
        prev_lo = None               # the low end before it last moved (a, phi, dphi)
        want_grad = a_lo > 0.0       # entered with a sufficient-decrease point in hand
        while nfev < max_ls:
            if abs(a_hi - a_lo) <= alphamin:
                break
            if math.isfinite(phi_hi):
                a_j = _interp_safeguarded(a_lo, phi_lo, dphi_lo, a_hi, phi_hi, dphi_hi, extra,
                                          backoff=not want_grad, prev_lo=prev_lo)
            else:
                a_j = 0.5 * (a_lo + a_hi)
            if want_grad:
                phi_j, dphi_j, pt_j = eval_grad(a_j)
                nfev += 1
                record(a_j, phi_j, pt_j)
            else:
                phi_j = eval_val(a_j)
                nfev += 1
                note_value(a_j, phi_j)
                dphi_j, pt_j = None, None
            if (not math.isfinite(phi_j)) or (not armijo_ok(a_j, phi_j)) or phi_j >= phi_lo:
                if math.isfinite(phi_hi):
                    extra = (a_hi, phi_hi)
                a_hi, phi_hi, dphi_hi = a_j, phi_j, dphi_j
                want_grad = False
                continue
            if pt_j is None:
                # Loss-only trial with sufficient decrease: now worth a gradient.
                phi_j, dphi_j, pt_j = eval_grad(a_j)
                nfev += 1
                record(a_j, phi_j, pt_j)
                if (not math.isfinite(phi_j)) or (not armijo_ok(a_j, phi_j)) or phi_j >= phi_lo:
                    a_hi, phi_hi, dphi_hi = a_j, phi_j, dphi_j     # only if the two evaluations disagree
                    want_grad = False
                    continue
            want_grad = True
            if wolfe_ok(dphi_j):
                return pt_j
            if dphi_j * (a_hi - a_lo) >= 0.0:
                a_hi, phi_hi, dphi_hi = a_lo, phi_lo, dphi_lo
            prev_lo = (a_lo, phi_lo, dphi_lo)
            a_lo, phi_lo, dphi_lo = a_j, phi_j, dphi_j
        return fallback()

    a_prev, phi_prev, dphi_prev = 0.0, phi0, dphi0
    a_i = float(a_init)
    first = True
    while nfev < max_ls:
        if first and not grad_first:
            phi_i = eval_val(a_i)
            nfev += 1
            note_value(a_i, phi_i)
            if (not math.isfinite(phi_i)) or (not armijo_ok(a_i, phi_i)):
                return zoom(0.0, phi0, dphi0, a_i, phi_i, None)
        phi_i, dphi_i, pt_i = eval_grad(a_i)
        nfev += 1
        first = False
        record(a_i, phi_i, pt_i)
        if (not math.isfinite(phi_i)) or (not armijo_ok(a_i, phi_i)) \
                or (a_prev > 0.0 and phi_i >= phi_prev):
            return zoom(a_prev, phi_prev, dphi_prev, a_i, phi_i, dphi_i)
        if wolfe_ok(dphi_i):
            return pt_i
        if dphi_i >= 0.0:
            return zoom(a_i, phi_i, dphi_i, a_prev, phi_prev, dphi_prev)
        if a_i >= alphamax:
            return pt_i          # slope still negative at alphamax: take it
        a_prev, phi_prev, dphi_prev = a_i, phi_i, dphi_i
        a_i = min(2.0 * a_i, alphamax)
    return fallback()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def minimize(fun_val, fun_val_grad, x0, options=None):
    """Minimize using BFGS with an Armijo backtracking or strong-Wolfe line search.

    Parameters
    ----------
    fun_val : callable  w -> scalar
        Loss only.  Used for the line search.  Should be a thin wrapper
        around a module-level @jax.jit function so JAX compiles it once.
    fun_val_grad : callable  w -> (scalar, array)
        Loss + gradient w.r.t. w.  Used for BFGS updates.  Same requirement.
    x0 : jnp.ndarray
        Initial flat parameter vector.
    options : dict, optional
        maxiter       – BFGS iterations (default 100)
        gtol          – gradient-norm stopping tolerance (default 1e-9)
        alphamin      – minimum line-search step (default 1e-15)
        max_iter      – max line-search trials per iteration (default 50):
                        Armijo halvings, or Wolfe loss+gradient evaluations
        line_search   – 'armijo' (default): backtracking by halving from
                        alpha=1 until sufficient decrease (c1=1e-4); one
                        loss evaluation per trial.
                        'wolfe': strong-Wolfe bracketing/zoom search with
                        cubic/quadratic interpolation (Nocedal & Wright
                        Alg. 3.5/3.6); steps may exceed 1, and the curvature
                        condition y.s > 0 is guaranteed whenever the search
                        succeeds.  Gradients are computed only at the first
                        trial (when the previous step was >= 1), at expansion
                        trials and at sufficient-decrease points; all other
                        trials are loss-only (see _strong_wolfe).
        c2            – Wolfe curvature constant (default 0.9; 'wolfe' only)
        alphamax      – largest step the Wolfe search may try (default 1e3)
        update_method – 'ssbroyden2' | 'ssbroyden' | 'bfgs' | 'score' (default 'bfgs')
                        'score': SCORE (arXiv:2608.04206, Alg. 1). Each
                        iteration computes the quasi-Newton decrement
                        lambda_k = sqrt(max(g.H.g, 0)); the strong-Wolfe
                        search starts from the candidate step 1/(1+lambda_k)
                        instead of 1 (accepted when it satisfies both Wolfe
                        conditions, as in the paper; otherwise the search
                        brackets/zooms from there), and the SSBroyden update
                        uses the shifted pair (s, y + mu_k s) with
                        mu_k = clip(score_eta*lambda_k, score_mu_min,
                        score_mu_max). Loss, gradient and Wolfe tests use the
                        original objective. Requires line_search='wolfe'.
        score_eta, score_mu_min, score_mu_max – SCORE shift parameters
                        (defaults 0.1, 1e-8, 1e-4; the paper does not
                        publish its values). mu has units of curvature
                        (loss / parameter^2). On a scaled-down copy of the
                        paper's Burgers benchmark (Oct 2026), eta = 1 /
                        mu_max = 1e-2 stalled ~30x above SSBroyden; these
                        defaults tie SSBroyden at equal evaluations.
        initial_H     – initial inverse Hessian (default identity)
        initial_scale – if True and initial_H is the identity, rescale it to
                        (y.s / y.y) I after the first line search, before the
                        first quasi-Newton update (Nocedal & Wright eq. 6.20).
                        Gives a well-scaled first step instead of the raw
                        identity's unit step; a non-identity initial_H
                        (e.g. carried over from a previous call) is left as is.
        rescale_carried – if True, rescale a NON-identity initial_H (e.g. carried
                        over from the previous call after a resample) by
                        y.s / y.H.y from the first curvature pair of this call,
                        before the first update (PINN_LESSONS 17.4). Default False.
        scale_invariant – if True, make the optimizer independent of the loss
                        scale (PINN_LESSONS 18): the curvature guards become
                        relative, y.s <= 1e-14 |s||y|, and the H update is
                        computed on the rescaled pair, H_new = c U(H/c, s, c y, c g)
                        with c = 1/(|s||y|) (exact for BFGS-type updates), so the
                        internal 1e-14 regularisers act at a relative level.
                        Without it, every update is skipped once y.s < 1e-14
                        (loss ~1e-12). Default False.
        reuse_first_trial – (Armijo only) evaluate the alpha = 1 trial with
                        fun_val_grad; if it passes Armijo, reuse its loss and
                        gradient (saves one forward pass). Default False.
        collect_stats – if True (default), fill ``stats`` (costs two scalar
                        device syncs per iteration). False skips them unless
                        SCORE needs the decrement.

    The result's ``stats`` is a dict of Python scalars for logging:
    n_iter, n_update (curvature pairs used), n_skip (pairs rejected by the
    y.s > 0 guard), n_first (iterations whose first line-search trial was
    accepted), lam_med (median quasi-Newton decrement sqrt(-g.p)), curv_med
    (median raw y.s/s.s) and, for 'score', mu_mean (mean shift).
    """
    if options is None:
        options = {}

    maxiter       = int(options.get('maxiter',       100))
    gtol          = float(options.get('gtol',        1e-9))
    alphamin      = float(options.get('alphamin',    1e-15))
    max_ls        = int(options.get('max_iter',      50))
    update_method = str(options.get('update_method', 'bfgs'))
    line_search   = str(options.get('line_search',   'armijo'))
    c2            = float(options.get('c2',          0.9))
    alphamax      = float(options.get('alphamax',    1e3))
    H             = options.get('initial_H', jnp.eye(x0.shape[0], dtype=x0.dtype))
    initial_scale = bool(options.get('initial_scale', False))
    rescale_carried = bool(options.get('rescale_carried', False))
    scale_invariant = bool(options.get('scale_invariant', False))
    reuse_first   = bool(options.get('reuse_first_trial', False))
    collect_stats = bool(options.get('collect_stats', True))
    score_eta     = float(options.get('score_eta',     0.1))
    score_mu_min  = float(options.get('score_mu_min',  1e-8))
    score_mu_max  = float(options.get('score_mu_max',  1e-4))

    if line_search not in LINE_SEARCHES:
        raise ValueError(f"line_search must be one of {LINE_SEARCHES}, got {line_search!r}")
    if update_method not in UPDATE_METHODS:
        raise ValueError(f"update_method must be one of {UPDATE_METHODS}, got {update_method!r}")
    use_score = update_method == 'score'
    if use_score:
        if line_search != 'wolfe':
            raise ValueError("update_method='score' requires line_search='wolfe'")
        if not (score_eta >= 0.0 and 0.0 <= score_mu_min <= score_mu_max):
            raise ValueError(f"SCORE needs score_eta >= 0 and 0 <= score_mu_min <= score_mu_max, got "
                             f"{score_eta}, {score_mu_min}, {score_mu_max}")

    c1 = 1e-4

    # Evaluation counters (reported in OptimizeResult.nfev / njev).
    n_val = 0
    n_vg  = 0

    def fun_val_c(w):
        nonlocal n_val
        n_val += 1
        return fun_val(w)

    def fun_val_grad_c(w):
        nonlocal n_vg
        n_vg += 1
        return fun_val_grad(w)

    # Backtracking Armijo line search.
    # float() calls sync one scalar GPU→CPU per backtrack step (unavoidable
    # for Python-level branching); typically 1–5 steps in practice.
    def backtrack(x, f, g, p, a_init=1.0):
        gp    = float(jnp.dot(g, p))
        f_ref = float(f)
        alpha = 1.0
        n_ls  = max_ls
        if reuse_first:
            # Full step evaluated WITH its gradient; x_try is built exactly as
            # x_new below would be, so an accepted step is bit-identical.
            x_try        = x + jnp.array(1.0, dtype=x.dtype) * p
            f_try, g_try = fun_val_grad_c(x_try)
            if float(f_try) <= f_ref + c1 * gp:
                return 1.0, x_try, f_try, g_try
            alpha = 0.5
            n_ls  = max_ls - 1
        for _ in range(n_ls):
            if float(fun_val_c(x + alpha * p)) <= f_ref + c1 * alpha * gp or alpha <= alphamin:
                break
            alpha *= 0.5
        x_new        = x + jnp.array(alpha, dtype=x.dtype) * p
        f_new, g_new = fun_val_grad_c(x_new)
        return alpha, x_new, f_new, g_new

    # Strong Wolfe.  grad_first adapts to the recent history: after a step
    # >= 1 the next full step is evaluated with its gradient straight away
    # (one evaluation if accepted); after a backtracked step the next full
    # step is first probed loss-only, so a repeated overshoot costs a cheap
    # loss evaluation instead of a gradient.
    ls_state = {"grad_first": True}

    def wolfe(x, f, g, p, a_init=1.0):
        res = _strong_wolfe(fun_val_c, fun_val_grad_c, x, f, g, p, c1, c2, max_ls,
                            alphamin, alphamax, grad_first=ls_state["grad_first"],
                            a_init=a_init)
        # "the first trial (or a longer step) was accepted": for a_init = 1
        # this is the historical res[0] >= 1 test.
        ls_state["grad_first"] = res[0] >= a_init
        return res

    search = wolfe if line_search == 'wolfe' else backtrack

    # BFGS loop — pure Python, no XLA loop compiled.
    x = x0
    f, g = fun_val_grad_c(x)
    # Is the initial H the plain identity?  For a symmetric H, trace(H) = n
    # and ||H||_F^2 = n together force every eigenvalue to 1, so two
    # reductions decide it without materialising an n x n comparison.
    if initial_scale:
        n_dim = H.shape[0]
        h_is_identity = (abs(float(jnp.trace(H)) - n_dim) <= 1e-9 * n_dim
                         and abs(float(jnp.vdot(H, H)) - n_dim) <= 1e-9 * n_dim)
    else:
        h_is_identity = False
    nit  = 0
    eps  = 1e-14
    n_update = n_skip = n_first = 0
    n_full = n_reset = 0
    h_scale = 1.0
    lam_hist, curv_hist, mu_hist = [], [], []

    for it in range(maxiter):
        if float(jnp.linalg.norm(g)) <= gtol:
            break
        p = -(H @ g)
        # Quasi-Newton decrement lambda_k = sqrt(g.H.g) = sqrt(-g.p).
        lam = math.sqrt(max(-float(jnp.dot(g, p)), 0.0)) if (collect_stats or use_score) else float('nan')
        a_init = 1.0 / (1.0 + lam) if use_score else 1.0
        alpha, x_new, f_new, g_new = search(x, f, g, p, a_init)
        s = x_new - x
        y = g_new - g
        ys = float(jnp.dot(y, s))
        ss = float(jnp.dot(s, s)) if (collect_stats or use_score) else float('nan')
        if collect_stats:
            lam_hist.append(lam)
            if ss > 0.0:
                curv_hist.append(ys / ss)
        if alpha >= a_init:
            n_first += 1
        if alpha == 1.0:
            n_full += 1
        if use_score:
            # Shifted secant pair (s, y + mu s): the inverse update learns the
            # positive shifted metric Hess + mu I along the accepted step.
            mu = min(max(score_eta * lam, score_mu_min), score_mu_max)
            mu_hist.append(mu)
            y_upd  = y + jnp.array(mu, dtype=y.dtype) * s
            ys_upd = ys + mu * ss
        else:
            y_upd, ys_upd = y, ys
        # Curvature guard: absolute 1e-14 (historical) or, with scale_invariant,
        # relative to |s||y| (PINN_LESSONS 18).
        if scale_invariant:
            sc = float(jnp.linalg.norm(s)) * float(jnp.linalg.norm(y_upd))
            thr = eps * sc if sc > 0.0 else math.inf
        else:
            sc, thr = 1.0, eps
        if it == 0 and ys <= thr:
            # H (e.g. carried over as initial_H from a previous, differently
            # -sampled call) fails the curvature condition on its very first
            # application to THIS loss realization: it isn't a trustworthy
            # preconditioner here. The ordinary curvature guard below would
            # just skip the update and silently keep this same stale H for
            # every remaining iteration; instead, discard it and rebuild
            # from the identity so the rest of this call's curvature is
            # self-consistent with its own (resampled) batch.
            H = jnp.eye(H.shape[0], dtype=H.dtype)
            n_reset = 1
        else:
            if it == 0 and h_is_identity:
                # Scale the identity by y.s/y.y (N&W 6.20) so the inverse
                # Hessian starts with the right magnitude along the first
                # curvature pair; the update below then refines it.
                yy = float(jnp.dot(y_upd, y_upd))
                if yy > (0.0 if scale_invariant else eps):
                    H = (ys_upd / yy) * H
            elif it == 0 and rescale_carried:
                # Carried-over H (e.g. after a resample): restore its scale
                # to the new point set's curvature, y.s / y.H.y.
                yHy = float(jnp.dot(y_upd, H @ y_upd))
                if yHy > (0.0 if scale_invariant else eps) and ys_upd > thr:
                    h_scale = ys_upd / yHy
                    H = h_scale * H
            if ys_upd > thr:
                n_update += 1
            else:
                n_skip += 1
            if scale_invariant:
                if ys_upd > thr:
                    c = 1.0 / sc
                    if update_method in ('ssbroyden2', 'ssbroyden', 'score'):
                        H = c * _ssbroyden_update(H * sc, s, c * y_upd, c * g, alpha)
                    else:
                        H = c * _bfgs_update(H * sc, s, c * y_upd)
            elif update_method in ('ssbroyden2', 'ssbroyden', 'score'):
                H = _ssbroyden_update(H, s, y_upd, g, alpha)
            else:
                H = _bfgs_update(H, s, y_upd)
        x, g, f = x_new, g_new, f_new
        nit += 1

    def _median(v):
        return float(sorted(v)[len(v) // 2]) if v else float('nan')

    stats = {"n_iter": nit, "n_update": n_update, "n_skip": n_skip, "n_first": n_first,
             "lam_med": _median(lam_hist), "curv_med": _median(curv_hist),
             "h_scale": h_scale, "n_reset": n_reset}
    if use_score:
        stats["mu_mean"] = float(sum(mu_hist) / len(mu_hist)) if mu_hist else float('nan')

    return OptimizeResult(
        x=x,
        fun=f,
        nit=jnp.array(nit, dtype=jnp.int32),
        hess_inv=H,
        nfev=n_val,
        njev=n_vg,
        stats=stats,
        ngev=n_vg,
        nfull=n_full,
        h_scale=h_scale,
        nskip=n_skip,
        nreset=n_reset,
    )
