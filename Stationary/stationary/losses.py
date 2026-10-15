"""Residuals, boundary conditions and the total loss.

Inner boundary (rho = rho_in):
    * lambda = lam0 + S1 z/rho_in + S2 (z^2-(x^2+y^2)/2)/rho_in^2   (see problem.py)
    * the induced metric on the sphere is the round metric of areal radius
      `inner_radius` (default rho_in)
    * optionally h_rr = const (off by default: it over-determines the radial gauge)

Outer boundary (rho = rho_out):
    * "dirichlet_exact": h and lambda from the exact solution (manufactured test)
    * "robin": higher-multipole decay condition.  With n = cfg.robin_order and the
      leading decay exponent k_f of each field,

          prod_{i=0}^{n-1} (rho d_rho + k_f + i) (field - field_inf) = 0 .

      The Euler operators (rho d_rho + a) commute and annihilate exactly rho^-a, so
      n = 1 is the familiar (field-field_inf)/rho + d_rho field = 0 and n = 2 lets the
      next multipole through, e.g. for lambda: rho^2 lam'' + 4 rho lam' + 2 (lam-1) = 0.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .geometry import (pack_gamma, pack_sym, radial_derivative_residual_batch,
                       radial_derivative_residuals_batch,
                       residuals_batch, scaled_residuals_batch)
from .problem import lam_inner_bc, sample_sphere

I3 = jnp.eye(3)
OFFW = jnp.array([1.0, jnp.sqrt(2.0), jnp.sqrt(2.0), 1.0, jnp.sqrt(2.0), 1.0])


def robin_operator(field_fun, x, base, order, inf_val=0.0):
    """prod_{i<order} (rho d_rho + base+i) applied to (field - inf_val) at x.

    (rho d_rho + a) annihilates exactly rho^-a, and these Euler operators commute, so
    the product annihilates the powers rho^-base ... rho^-(base+order-1) and lets the
    next multipoles through.  `order=1` is the familiar first-order decay condition.
    """
    cur = lambda y: field_fun(y) - inf_val
    for i in range(order):
        k = base + i
        prev = cur

        def cur(y, prev=prev, k=k):
            r = jnp.linalg.norm(y)
            n = y / r
            d = jnp.einsum("a,...a->...", n, jax.jacfwd(prev)(y))       # d_rho prev
            return r * d + k * prev(y)

    return cur(x)


def gh_potential(lam):
    """The Geroch-Hansen potential P = (1 - lam^2)/(4 lam).

    Its expansion is the multipole expansion: for lam = e^{2U} with U harmonic,
    P = -sinh(2U)/2 = -U - (2/3)U^3 + ..., so P's l = 0 part has powers rho^-1, rho^-3, ...
    and NO rho^-2 term.  In lambda itself the l = 0 rho^-2 term is 2M^2 (from the exponential),
    so a radial condition acting on lambda cannot drop the dipolar factor without also killing a
    physical term; on P the rho^-2 power IS the dipole alone.
    """
    return (1.0 - lam ** 2) / (4.0 * lam)


def robin_operator_exps(field_fun, x, exps, inf_val=0.0):
    """prod_{a in exps} (rho d_rho + a) applied to (field - inf_val) at x.

    Same operator as `robin_operator`, with the exponents given explicitly so the KERNEL can be
    chosen instead of being the consecutive window base ... base+order-1.  Dropping the exponent 2
    is what removes the dipolar power rho^-2 from the kernel: it stops being annihilated, so the
    condition forces that coefficient to vanish.
    """
    cur = lambda y: field_fun(y) - inf_val
    for k in exps:
        prev = cur

        def cur(y, prev=prev, k=k):
            r = jnp.linalg.norm(y)
            n = y / r
            d = jnp.einsum("a,...a->...", n, jax.jacfwd(prev)(y))
            return r * d + k * prev(y)

    return cur(x)


def drop_power(exps, power=2.0):
    """The same window with one power removed from the kernel and one higher appended."""
    kept = [a for a in exps if abs(a - float(power)) > 1e-12]
    return list(exps) if len(kept) == len(exps) else kept + [max(exps) + 1.0]


def robin_coefficients(base: float, order: int) -> list[float]:
    """Coefficients c_j of prod_i (rho d_rho + base + i) = sum_j c_j rho^j d_rho^j.

    The Euler operators commute, so the product is a polynomial in L = rho d_rho, and
    L^k = sum_j S(k, j) rho^j d_rho^j with S the Stirling numbers of the SECOND kind (the
    falling powers are the other convention).  With the coefficients in hand a Robin residual
    can be written as the SUM OF ITS TERMS, which is the diagnostic that matters here: a
    residual of 1e-06 can be two numbers of order 1 cancelling.  Measured on
    runs/production_quad_quarter, whose order-3 lambda combination at rho_out is

        rho^3 d3(lam) = +3.796      9 rho^2 d2(lam) = +0.1645
        18 rho d1(lam) = +0.1420    6 (lam - 1)    = -4.102     sum = +4.4e-05

    -- that is how a run kept lambda(rho_out) = 0.3163 against 0.9885193 while reporting an
    outer Robin residual of 6.5e-06.  base 1, order 3 gives [6, 18, 9, 1] and base 2, order 3
    gives [24, 36, 12, 1]; `tests/test_robin_terms.py` checks the annihilation properties.
    """
    a = [1.0]                                  # product of (x + base + i), a[k] = coeff of x^k
    for i in range(order):
        s_ = float(base) + i
        new = [0.0] * (len(a) + 1)
        for k, c in enumerate(a):
            new[k] += c * s_
            new[k + 1] += c
        a = new
    S = [[0.0] * (order + 1) for _ in range(order + 1)]      # Stirling, second kind
    S[0][0] = 1.0
    for k in range(1, order + 1):
        for j in range(1, k + 1):
            S[k][j] = j * S[k - 1][j] + S[k - 1][j - 1]
    c = [0.0] * (order + 1)
    for k in range(order + 1):
        for j in range(k + 1):
            c[j] += a[k] * S[k][j]
    return c


# ------------------------------------------------------------------ PDE terms
def gauge_source_of(cfg, point_fields):
    """The inhomogeneous gauge source cfg.gauge_source names, or None for harmonic.

    Built from the CANDIDATE's own metric: the point of a gauge condition is that it does
    not presuppose the solution, so nothing here knows about rods, U or k.
    """
    if cfg.gauge_source == "none":
        return None
    if cfg.gauge_source == "cylindrical":
        from .weyl import gauge_source_from_metric, rotation_matrix
        axis = None
        if getattr(cfg, "weyl_rotate_deg", 0.0):
            # the condition is about a chart: rotate its axis with the configuration
            axis = rotation_matrix(cfg.weyl_rotate_deg) @ jnp.array([0.0, 0.0, 1.0])
        return gauge_source_from_metric(lambda x: point_fields(x).h, axis=axis)
    raise ValueError(f"unknown gauge_source {cfg.gauge_source!r}")


def pde_raw(point_fields, xs, cfg, gauge_src=None) -> dict:
    """Per-point (rho-scaled) residual COMPONENTS of the equation groups -- no reduction.

    `pde_terms` is the mean of their squares.  The unreduced arrays are what a least-squares
    optimiser needs: the loss is a weighted sum of means of squares, so it is also the squared
    norm of a residual vector (see `residual_vector` and stationary/dsgnar.py).

    `gauge_src(x)` (optional) replaces the harmonic gauge condition with the inhomogeneous
    Gamma^i_{jk} h^{jk} = gauge_src^i, which is what lets a non-harmonic chart (the Weyl
    chart of a two-black-hole solution) be an exact solution of the whole system; see
    geometry.residuals_at and stationary/weyl.py.
    """
    if gauge_src is None:
        gauge_src = gauge_source_of(cfg, point_fields)
    return scaled_residuals_batch(point_fields, xs, cfg.scale_exps, cfg.scale_ref, gauge_src)


def pde_terms(point_fields, xs, cfg, gauge_src=None) -> dict:
    """Mean squared (rho-scaled) residuals of the four equation groups."""
    return {k: jnp.mean(v ** 2) for k, v in pde_raw(point_fields, xs, cfg, gauge_src).items()}


def pde_radial_raw(point_fields, xs, cfg):
    """The unreduced rho-scaled radial derivative of the lambda-equation residual, or None.

    None (and free, nothing is differentiated) unless `cfg.w_lam_eq_radial` is nonzero, so
    every existing run and every existing test is unaffected.  It exists because the
    lambda-equation is second order: the loss cannot see lambda''', and the order-3 Robin
    condition can be satisfied with a wrong far-field level by putting the mismatch exactly
    there (measured on runs/production_quad_quarter: outer Robin residual 7e-06 while
    lambda(rho_out) = 0.31 against 0.9885193; the reference's own order-3 combination is a
    cancellation of terms of order 0.1 leaving 1.3e-08).  One radial derivative makes the
    loss sensitive to that content; `rho^3` is the dimensionally consistent factor, inherited
    from `scale_exps['lam_eq'] = 2` rather than hard-coded.
    """
    if not float(getattr(cfg, "w_lam_eq_radial", 0.0) or 0.0):
        return None
    return radial_derivative_residual_batch(point_fields, xs, cfg.scale_exps, cfg.scale_ref)


def pde_radial_groups_raw(point_fields, xs, cfg) -> dict:
    """The rho-scaled radial derivative of EVERY group's residual, or {}.

    Nothing is differentiated unless `cfg.pde_radial` (--pde-radial), so every existing run and
    test is unaffected.  `compat` is left out on purpose: it holds identically in the metric-only
    scheme and is not in the loss there, so there is nothing for its derivative to constrain.

    The weight is not decided here: the callers add these arrays to their own group, so the term
    inherits the equation's weight (see `total_loss`, `residual_vector` and `group_terms`).
    """
    if not bool(getattr(cfg, "pde_radial", False)):
        return {}
    out = radial_derivative_residuals_batch(point_fields, xs, cfg.scale_exps, cfg.scale_ref)
    return {k: v for k, v in out.items() if k != "compat"}


def pde_radial_groups_terms(point_fields, xs, cfg) -> dict:
    """Mean squared rho-scaled radial derivative, per equation group."""
    return {k: jnp.mean(v ** 2) for k, v in pde_radial_groups_raw(point_fields, xs, cfg).items()}


def pde_radial_terms(point_fields, xs, cfg) -> dict:
    """Mean squared rho-scaled RADIAL DERIVATIVE of the lambda-equation residual."""
    r = pde_radial_raw(point_fields, xs, cfg)
    return {} if r is None else {"lam_eq_radial": jnp.mean(r ** 2)}


# -------------------------------------------------------------- inner boundary
def inner_bc_raw(point_fields, xs, cfg, exact_fields=None) -> dict:
    """Per-point inner boundary residuals, split by field and unreduced.

    "spherical" (the default) imposes the problem statement: lambda from lam_inner_bc and
    the round metric of areal radius inner_radius on the sphere.  "reference" imposes
    whatever the exact reference carries there, which is what a manufactured check of a
    non-spherical solution needs -- the Weyl two-black-hole configuration has neither a
    round inner metric nor that polynomial lambda.

    THE METRIC CONDITION IS ON THE INDUCED METRIC ONLY: `P h P` with `P = I - n n`, the
    tangential block.  `h_rr` and the mixed components `h_rA` are NOT boundary data and must not
    be imposed -- in either mode.  Until 2026-10-15 this imposed `h - h_rr n n = c (I - n n)`,
    whose mixed part silently set `h_rA = 0` on the sphere as well.  For the spherical solution
    `h_rA` vanishes by symmetry, so nothing showed; for any l >= 1 datum it is one condition too
    many per mode: in harmonic gauge the l = 2 sector has five decaying constants and the inner
    sphere already fixes five (the lambda datum, the trace and trace-free parts of the induced
    metric, the two gauge components), so with `h_rA = 0` added no decaying solution exists.  The
    spectral reference solve (stationary/linref.py) under the old condition showed what that
    looks like: Newton stalled at a residual floor (loss 1.8e-5 at l <= 6, 24 radial modes) and
    kept a NON-DECAYING l = 2 metric part (h_rr's P_2 coefficient ~ -0.045 from rho = 10 to 100)
    -- the "forbidden rho^0 mode" of PERTURBATION.md, section 3, which the networks carried too.  `h_rA` is a coordinate
    statement that nothing in the problem asks for, and the harmonic gauge already uses up the
    coordinate freedom.  `model._blend_inner` (the hard version) projects the same way.
    """
    if cfg.inner_bc == "reference":
        if exact_fields is None:
            raise ValueError("inner_bc = 'reference' needs exact_fields")

        def one(x):
            f, e = point_fields(x), exact_fields(x)
            n = x / jnp.linalg.norm(x)
            P = I3 - jnp.outer(n, n)
            # the induced metric only (see the docstring): never h_rr, never h_rA
            return jnp.concatenate([pack_sym(P @ (f.h - e.h) @ P) * OFFW,
                                    jnp.array([f.lam - e.lam])])

        R = jax.vmap(one)(xs)
        drop = getattr(cfg, "drop_inner_ls", ()) or ()
        return {"h": _drop_legendre_modes(R[:, :6], xs, drop),
                "lam": _drop_legendre_modes(R[:, 6][:, None], xs, drop)[:, 0]}

    def one(x):
        f = point_fields(x)
        rho = jnp.linalg.norm(x)
        n = x / rho
        nn = jnp.outer(n, n)
        h_rr = n @ f.h @ n
        c = cfg.inner_radius**2 / rho**2          # round metric of areal radius inner_radius
        P = I3 - nn
        # The INDUCED metric P h P, and nothing else: h_rA is not boundary data (docstring).  The
        # rr and rA entries of M vanish identically; they stay in the packed rows so the row
        # count -- and every residual-vector consumer -- is unchanged.
        M = P @ f.h @ P - c * P
        hrr_ref = 1.0 if cfg.inner_h_rr is None else cfg.inner_h_rr
        return jnp.concatenate([
            jnp.array([f.lam - lam_inner_bc(x, cfg), h_rr - hrr_ref]),
            pack_sym(M) * OFFW,
        ])

    R = jax.vmap(one)(xs)
    drop = getattr(cfg, "drop_inner_ls", ()) or ()
    out = {"lam": _drop_legendre_modes(R[:, 0][:, None], xs, drop)[:, 0],
           "h_tan": _drop_legendre_modes(R[:, 2:], xs, drop)}
    if cfg.inner_h_rr is not None:
        out["h_rr"] = _drop_legendre_modes(R[:, 1][:, None], xs, drop)[:, 0]
    return out


def inner_datum_of(cfg, exact_fields=None):
    """`x (N,3) -> (h_datum (N,3,3), lam_datum (N,))`: the inner data as a function of DIRECTION.

    Evaluate the datum at the point of the inner sphere with the same direction, `rho_in * x/|x|`.
    That restriction is on purpose.  A datum is an inner-sphere quantity, so its value is what the
    inner condition prescribes and there is nothing to "extend"; the polynomial `lam_inner_bc` in
    particular grows like `z^2 / rho_in^2` and would reach ~800 at rho_out = 100 if it were simply
    evaluated at x.  Keeping it direction-only makes the datum bounded, and the hard-constraint
    blend in `model._blend_inner` then leaks it outward with weight `rho_in/r` -- 1% at rho_out =
    100 -- which the network absorbs.

    Both inner_bc modes return direction-only data, so a run can switch the inner condition from
    soft to hard without touching anything else.

    Only the TANGENTIAL block of `h_datum` is boundary data: `model._blend_inner` imposes `P h P`
    and leaves `h_rr` and `h_rA` to the network (see `inner_bc_raw`).  The full tensor is returned
    because it is what a hybrid datum needs to be, but its rr and rA parts are never imposed.
    """
    if getattr(cfg, "inner_bc", "spherical") == "reference":
        if exact_fields is None:
            raise ValueError("inner_bc = 'reference' needs exact_fields")

        def datum(x):
            r = jnp.linalg.norm(x, axis=-1, keepdims=True)
            e = jax.vmap(exact_fields)(cfg.rho_in * x / r)
            return e.h, e.lam

        return datum

    def datum(x):
        r = jnp.linalg.norm(x, axis=-1, keepdims=True)
        n = x / r
        nn = n[..., :, None] * n[..., None, :]
        hrr = 1.0 if cfg.inner_h_rr is None else cfg.inner_h_rr
        # c is the round sphere of areal radius `inner_radius` evaluated ON the inner sphere, so
        # c = (inner_radius / rho_in)^2 -- NOT 1: the two differ whenever the config's inner
        # radius is not rho_in, and the residual then reads exactly 3|1 - c| (1.73 for the
        # default inner_radius = 2 with rho_in = 1), which is how this was caught.
        c = (float(cfg.inner_radius) / float(cfg.rho_in)) ** 2
        return hrr * nn + c * (I3 - nn), lam_inner_bc(cfg.rho_in * n, cfg)

    return datum


def inner_bc_terms(point_fields, xs, cfg, exact_fields=None) -> dict:
    """Inner boundary residuals, reduced to means of squares."""
    return {k: jnp.mean(v ** 2)
            for k, v in inner_bc_raw(point_fields, xs, cfg, exact_fields).items()}


def reference_consistency(point_fields, cfg, n: int = 64, seed: int = 7,
                          exact_fields=None) -> dict:
    """How much the exact reference itself violates the imposed INNER boundary data.

    Zero means the inner data and the outer data (the Dirichlet values, or the
    manufactured Robin source) come from one and the same exact solution, so that
    solution is a genuine zero of the loss.  Anything else means they come from two
    different solutions: no metric can satisfy both, the run converges to a compromise,
    and the numbers it reports near the inner sphere do not mean what they look like.
    The usual cause is `inner_radius` not being the reference's own inner areal radius
    (see Config.__post_init__).
    """
    xs = sample_sphere(jax.random.PRNGKey(seed), n, cfg.rho_in)
    return {k: float(v)
            for k, v in inner_bc_terms(point_fields, xs, cfg, exact_fields).items()}


# -------------------------------------------------------------- outer boundary
def outer_bc_raw(point_fields, xs, cfg, exact_fields=None, lam_inf=None) -> dict:
    """Per-point outer boundary residuals, split by field and unreduced."""
    if cfg.outer_bc == "dirichlet_exact":
        if exact_fields is None:
            raise ValueError("dirichlet_exact outer BC needs exact_fields")

        def one(x):
            f = point_fields(x)
            e = exact_fields(x)
            return jnp.concatenate([
                pack_sym(f.h - e.h) * OFFW,
                jnp.array([f.lam - e.lam]),
            ])

        R = jax.vmap(one)(xs)
        return {"h": R[:, :6], "lam": R[:, 6]}

    if cfg.outer_bc == "robin":
        ph, pG, pl = cfg.robin_exps["h"], cfg.robin_exps["G"], cfg.robin_exps["lam"]
        orders = cfg.robin_orders or {k: cfg.robin_order for k in ("h", "G", "lam")}
        if lam_inf is None:
            lam_inf = cfg.lam_inf_init
        if cfg.robin_source and exact_fields is None:
            raise ValueError("robin_source needs exact_fields (set --ref-solution)")

        # The SCALAR condition acts on the Geroch-Hansen potential, not on lambda: P's l = 0 part
        # carries no rho^-2 (lambda = e^{2U} puts 2M^2 there), so P's rho^-2 power is the dipole
        # alone and the dipolar factor can be dropped from the kernel without touching the mass.
        # The metric condition is left alone: its rho^-2 content (g2 - 1 = -M^2/rho^2 + ...) is
        # physical too, and the dipole there is a gauge mode, to be handled by a centre-of-mass
        # condition rather than by a decay kernel.
        use_gh = bool(getattr(cfg, "robin_lam_potential", True))
        exps_lam = [pl + i for i in range(orders["lam"])]
        if use_gh and bool(getattr(cfg, "robin_lam_drop_dipole", True)):
            exps_lam = drop_power(exps_lam, 2.0)
        if use_gh:
            lam_field = lambda y: gh_potential(point_fields(y).lam)
            lam_ref = lambda y: gh_potential(exact_fields(y).lam) if exact_fields else None
            lam_target = gh_potential(jnp.asarray(lam_inf))
        else:
            lam_field = lambda y: point_fields(y).lam
            lam_ref = lambda y: exact_fields(y).lam if exact_fields else None
            lam_target = lam_inf

        def one(x):
            f = point_fields(x)
            rh = robin_operator(lambda y: point_fields(y).h, x, ph, orders["h"], I3)
            rl = robin_operator_exps(lam_field, x, exps_lam, lam_target)
            sh = sl = 0.0
            if cfg.robin_source:
                sh = robin_operator(lambda y: exact_fields(y).h, x, ph, orders["h"], I3)
                sl = robin_operator_exps(lam_ref, x, exps_lam, lam_target)
            if cfg.robin_include_G:
                rG = robin_operator(lambda y: point_fields(y).G, x, pG, orders["G"],
                                    jnp.zeros((3, 3, 3)))
                sG = 0.0
                if cfg.robin_source:
                    sG = robin_operator(lambda y: exact_fields(y).G, x, pG, orders["G"],
                                        jnp.zeros((3, 3, 3)))
                return jnp.concatenate([pack_sym(rh - sh) * OFFW,
                                        pack_gamma(rG - sG),
                                        jnp.array([rl - sl])])
            return jnp.concatenate([pack_sym(rh - sh) * OFFW, jnp.array([rl - sl])])

        R = jax.vmap(one)(xs)
        drop = getattr(cfg, "drop_robin_ls", ()) or ()
        out = {"h": _drop_legendre_modes(R[:, :6], xs, drop),
               "lam": _drop_legendre_modes(R[:, -1][:, None], xs, drop)[:, 0]}
        if cfg.robin_include_G:
            out["G"] = _drop_legendre_modes(R[:, 6:24], xs, drop)
        return out

    raise ValueError(f"unknown outer_bc {cfg.outer_bc!r}")


def _legendre(l, mu):
    """P_l(mu) by the standard recurrence (jax has no Legendre module)."""
    if l == 0:
        return jnp.ones_like(mu)
    p0, p1 = jnp.ones_like(mu), mu
    for k in range(2, l + 1):
        p0, p1 = p1, ((2 * k - 1) * mu * p1 - (k - 1) * p0) / k
    return p1


def _drop_legendre_modes(R, xs, ls):
    """Remove the `l in ls` angular content of every column of `R`, sampled on the sphere `xs`.

    `cfg.drop_robin_ls` is how a run says "do not let the outer condition act on this multipole":
    the residual handed to the loss keeps the l = 0 and l >= 2 parts untouched and loses the l = 1
    one, so the dipole is left completely free at the outer boundary.  That is a diagnostic, not a
    physical statement: with the Robin window's action on the dipole removed, whatever the interior
    pushes into that sector is unopposed, and the multipole it settles into can be read directly.

    The outer batch is a uniform draw on the sphere, so the coefficient of `P_l(mu)` is estimated
    without bias by `(2l+1)/N sum_i R_i P_l(mu_i)`, since `E[P_l P_l'] = delta_ll'/(2l+1)` for a
    uniform `mu`.  Every column (a metric component, or lambda) is treated separately.
    """
    ls = [int(l) for l in (ls or ())]
    if not ls:
        return R
    mu = xs[:, 2] / jnp.linalg.norm(xs, axis=-1)
    for l in ls:
        P = _legendre(l, mu)
        c = (2 * l + 1) / R.shape[0] * jnp.sum(R * P[:, None], axis=0)
        R = R - c[None, :] * P[:, None]
    return R


def outer_bc_terms(point_fields, xs, cfg, exact_fields=None, lam_inf=None) -> dict:
    """Outer boundary residuals, reduced to means of squares."""
    return {k: jnp.mean(v ** 2)
            for k, v in outer_bc_raw(point_fields, xs, cfg, exact_fields, lam_inf).items()}


def outer_pin_raw(point_fields, xs, cfg, exact_fields=None):
    """Per-point pin residuals, in the two groups the reduction distinguishes.

    Returns `(averaged, value)`: the first are conditions on the spherical MEAN (reduced by
    mean-then-square), the second are statements about every angle (reduced by
    mean-of-squares).  `outer_pin_terms` reduces them; `residual_vector` needs them apart.

    See `outer_pin_terms` for the meaning of each pin.
    """
    robin_lam = bool(getattr(cfg, "pin_lam_robin", False))
    robin_h = bool(getattr(cfg, "pin_h_robin", False))
    if not (cfg.pin_lam or robin_lam or cfg.pin_h_tan or cfg.pin_h_rr or robin_h):
        return {}, {}
    if exact_fields is None and (cfg.pin_lam or cfg.pin_h_tan or cfg.pin_h_rr):
        raise ValueError(
            "the far-field value pins are differences from the exact reference at rho_out, "
            "and this run has no reference: add --ref-solution (with --ref-asymptotic k) or "
            "drop the pin flags.  --pin-lam-robin and --pin-h-robin are the exceptions: the "
            "averaged Robin conditions need lam_inf only.")

    lam_inf = cfg.lam_inf if getattr(cfg, "lam_inf", None) is not None else cfg.lam_inf_init

    def h_rr_of(y, h=None):
        h = point_fields(y).h if h is None else h
        n = y / jnp.linalg.norm(y)
        return n @ h @ n

    def g2_of(y, h=None):
        h = point_fields(y).h if h is None else h
        n = y / jnp.linalg.norm(y)
        return (jnp.trace(h) - n @ h @ n) / 2.0

    def one(x):
        """One point of the outer sphere, as a dict of the pin residuals required here.

        The averaged conditions evaluate the POINTWISE operator and let the reduction average
        it, which is the same thing as the operator applied to the mean, because averaging
        commutes with rho d_rho.  Doing it this way is what lets them reuse `robin_operator`
        unchanged, so a pin and the pointwise boundary condition cannot drift apart.
        """
        f = point_fields(x)
        v = {}
        if robin_lam:
            v["lam"] = robin_operator(lambda y: point_fields(y).lam, x, 1.0, 1, lam_inf)
        elif cfg.pin_lam:
            v["lam"] = f.lam - exact_fields(x).lam
        if robin_h:
            # order 1 at EACH quantity's own leading decay power: on the exact solution
            # h_rr - 1 ~ rho^-3 and g2 - 1 ~ rho^-2 (measured; see Config.pin_h_robin).  This
            # is not the base 2 the POINTWISE h condition uses, and it cannot be.
            b = cfg.h_robin_bases or {}
            v["h_rr_robin"] = robin_operator(h_rr_of, x, float(b.get("h_rr", 3.0)), 1, 1.0)
            v["h_tan_robin"] = robin_operator(g2_of, x, float(b.get("g2", 2.0)), 1, 1.0)
        elif exact_fields is not None:
            e = exact_fields(x)
            if cfg.pin_h_tan:
                v["h_tan_value"] = g2_of(x, f.h) - g2_of(x, e.h)
            if cfg.pin_h_rr:
                v["h_rr_value"] = h_rr_of(x, f.h) - h_rr_of(x, e.h)
        return v

    R = jax.vmap(one)(xs)       # a dict of batched arrays
    # Two different reductions, and the difference is the whole point: an AVERAGED condition is
    # a statement about the mean, so it squares the mean; the metric's VALUE pins are
    # statements about every angle, so they average the squares.  `lam` is on the mean in both
    # of its forms, which is why it appears only in the first group.
    averaged = {name: R[k] for k, name in (("lam", "lam"), ("h_rr_robin", "h_rr"),
                                           ("h_tan_robin", "h_tan")) if k in R}
    value = {name: R[k] for k, name in (("h_tan_value", "h_tan"),
                                        ("h_rr_value", "h_rr")) if k in R}
    return averaged, value


def outer_pin_terms(point_fields, xs, cfg, exact_fields=None) -> dict:
    """Asymptotic-VALUE pins at rho_out (`cfg.pin_lam`, `cfg.pin_h_tan`, `cfg.pin_h_rr`).

    Each term is the mean square of (candidate - reference) at the outer sphere, so a value
    cannot be traded against a derivative the way a Robin residual can (see the Config
    comment: the order-3 lambda combination is a cancellation of terms of order 0.1, and the
    exact h deviation is a kernel mode of the h condition).  The comparison is to the exact
    reference the run already carries, so no constant is hard-coded.

    * `lam`   -- the spherical MEAN of the difference: the monopole, which is the branch.
                 The l >= 1 content is deliberately left free, so that a later Robin-only
                 relaxation phase can fix the multipoles from the equations.
    * `h_tan` -- the tangential metric, angle by angle: g2 = (tr h - h_rr)/2, i.e. the areal
                 radius content.
    * `h_rr`  -- the radial gauge component, angle by angle.

    `cfg.pin_lam_robin` pins the same monopole through the order-1 Robin COMBINATION rather
    than its value:

        rho d_rho <lam> + (<lam> - lam_inf) = 0

    Averaging commutes with rho d_rho, so this is the mean of the pointwise order-1 condition
    `robin_operator(lam, x, base=1, order=1, inf_val=lam_inf)` -- exactly what
    `--robin-orders lam=1` imposes at every angle, imposed on the spherical mean.  It needs
    lam_inf and NO reference, which is what makes it the one pin available to a run whose
    inner data are angular (S1/S2): for those the spherical reference is not a solution at
    all, so every other pin here would be pinning a departure.  The mean is pinned and the
    l >= 1 content of the combination is left free, the same deliberate blind spot as above.
    """
    averaged, value = outer_pin_raw(point_fields, xs, cfg, exact_fields)
    out = {k: jnp.mean(v) ** 2 for k, v in averaged.items()}
    out.update({k: jnp.mean(v ** 2) for k, v in value.items()})
    return out


# ------------------------------------------------------------------ total loss
GROUP_KEYS = ("compat", "ricci", "gauge", "lam_eq", "inner", "outer")

# Which equation groups can constrain anything, per formulation.  A metric-only ("hybrid")
# model derives Gamma from h, so compatibility holds IDENTICALLY: it is a constraint on
# nothing, and the loss used to carry it anyway.  Its value there is ~1e-36, so dropping it
# changes no number -- what it changes is that the reported loss stops counting a fourth
# equation that is not one, and the group list stops implying the first-order formulation.
# The first-order models (FieldNet, SymFieldNet) output Gamma independently and need it.
FIRST_ORDER_PDE_KEYS = ("compat", "ricci", "gauge", "lam_eq")
METRIC_ONLY_PDE_KEYS = ("ricci", "gauge", "lam_eq")


def equation_keys(model=None) -> tuple:
    """The equation groups `model`'s formulation actually imposes.

    Keyed off the model's own `derives_gamma`, not off `cfg.arch`, so a caller that hands over
    a metric-only model gets the metric-only loss whatever the config says.
    """
    if model is not None and getattr(model, "derives_gamma", False):
        return METRIC_ONLY_PDE_KEYS
    return FIRST_ORDER_PDE_KEYS



def default_weights(cfg):
    w = {k: jnp.asarray(cfg.eq_weights[k]) for k in ("compat", "ricci", "gauge", "lam_eq")}
    w["inner"] = jnp.asarray(cfg.w_inner)
    w["outer"] = jnp.asarray(cfg.w_outer)
    # PER-FIELD boundary weights, taken from the same eq_weights dict.  `inner`/`outer` are one
    # weight per boundary and carry lambda and the metric together, so a run that needs the
    # metric undriven while lambda's data stays imposed could not be expressed.  A key like
    # `inner_h` or `outer_lam` overrides the group weight for that field alone:
    #     --eq-weights compat=0,ricci=0,gauge=0,lam_eq=1,inner_h=0,outer_h=0
    for _k, _v in dict(cfg.eq_weights).items():
        if _k.startswith(("inner_", "outer_")):
            w[_k] = jnp.asarray(_v)
    return w


def group_terms(state, batch, cfg, model, exact_fields=None, lam_inf=None) -> dict:
    """Unweighted mean-squared residuals of the six residual groups.

    Used both for diagnostics and for gradient-norm adaptive weighting.  Note that `compat` is
    returned even for a metric-only model, where `equation_keys` keeps it out of the loss: it is
    then a structural check (Gamma is h's Christoffel symbol, so the residual should be at the
    round-off floor) rather than a term, and the reweighting never sees it because the caller
    passes the model's own keys.
    """
    from .model import point_fields as make_point_fields

    pf = make_point_fields(model, state["net"])
    out = dict(pde_terms(pf, batch["coll"], cfg))
    # The radial-derivative term rides with the lambda-equation group here: this function
    # feeds the gradient-norm reweighting, which balances the equation groups, and a term the
    # reweighting cannot see would let it drive lam_eq's weight down while the unseen term
    # carries the loss.  Its value is not used as a diagnostic anywhere (the loss parts are).
    out["lam_eq"] = out["lam_eq"] + sum(pde_radial_terms(pf, batch["coll"], cfg).values())
    for k, v in pde_radial_groups_terms(pf, batch["coll"], cfg).items():
        out[k] = out.get(k, 0.0) + v
    out["inner"] = sum(inner_bc_terms(pf, batch["inner"], cfg, exact_fields).values())
    # The outer group carries the Robin conditions AND the value pins: the plateau rule and
    # the per-block log watch this number, and a run is not converged while a pin is unmet.
    if batch.get("outer_extra") is not None:
        out["outer_extra"] = sum(outer_bc_terms(pf, batch["outer_extra"], cfg, exact_fields,
                                                lam_inf).values())
    out["outer"] = (sum(outer_bc_terms(pf, batch["outer"], cfg, exact_fields,
                                       lam_inf).values())
                    + sum(outer_pin_terms(pf, batch["outer"], cfg, exact_fields).values()))
    return out


def residual_vector(state, batch, cfg, model, exact_fields=None, weights=None,
                    pde_scale=1.0, lam_inf=None, _blocks=None) -> jnp.ndarray:
    """The residual vector whose SQUARED NORM is `total_loss`.

    Every term of the loss is a weighted mean of squares, so the loss is itself a sum of
    squares:

        L = sum_i r_i^2,     r_i = sqrt(weight_i / n_i) * (the i-th unreduced component),

    with `n_i` the number of components that term averages over.  This function produces that
    vector, which is exactly what a Gauss-Newton / least-squares method operates on (see
    stationary/dsgnar.py, and `dsgnar_phase` in train.py).  Nothing about the objective
    changes: same weights, same points, same terms -- only the reduction is deferred.

    The two pin reductions are kept apart on purpose: an AVERAGED pin is `mean(R)**2`, so its
    residual is the single number `mean(R)`, while a VALUE pin is `mean(R**2)` and keeps one
    row per point.

    `tests/test_dsgnar.py::test_the_residual_vector_squares_to_the_total_loss` asserts
    `jnp.sum(residual_vector(...)**2) == total_loss(...)[0]`.

    `_blocks`, if a list, receives one `(label, n_rows)` per term, in order -- the CONDITIONS
    a per-condition reweighting acts on (DSGNAR's UpdateWeights).  Sizes are static, so
    `jax.eval_shape` fills it without evaluating anything.
    """
    from .model import point_fields as make_point_fields

    pf = make_point_fields(model, state["net"])
    if lam_inf is None:
        lam_inf = state.get("lam_inf", None)
    if weights is None:
        weights = default_weights(cfg)

    rows = []

    def push(label, row):
        rows.append(row)
        if _blocks is not None:
            _blocks.append((label, int(row.size)))

    def add(label, w, v):
        """One term: `w` its loss weight, `v` the unreduced components (any shape)."""
        push(label, jnp.sqrt(jnp.asarray(w) * float(pde_scale) / v.size) * v.reshape(-1))

    pde = pde_raw(pf, batch["coll"], cfg)
    for k in equation_keys(model):
        add(f"pde_{k}", weights[k], pde[k])
    # --pde-radial: the same weight as the group, because these are that group's residual,
    # differentiated -- see RADIAL_DERIVATIVES.md.
    for k, v in pde_radial_groups_raw(pf, batch["coll"], cfg).items():
        add(f"pde_{k}_radial", weights[k], v)
    rad = pde_radial_raw(pf, batch["coll"], cfg)
    if rad is not None:
        add("pde_lam_eq_radial", float(cfg.w_lam_eq_radial), rad)
    for k, v in inner_bc_raw(pf, batch["inner"], cfg, exact_fields).items():
        push(f"inner_{k}", jnp.sqrt(jnp.asarray(weights.get(f"inner_{k}", weights["inner"]))
                                    / v.size) * v.reshape(-1))
    for k, v in outer_bc_raw(pf, batch["outer"], cfg, exact_fields, lam_inf).items():
        push(f"outer_{k}", jnp.sqrt(jnp.asarray(weights.get(f"outer_{k}", weights["outer"]))
                                    / v.size) * v.reshape(-1))
    # the same condition on the inner shells of the ladder (see train.make_batch).  The PIN stays on
    # rho_out alone: it is an average and belongs at the outer sphere.
    if batch.get("outer_extra") is not None:
        for k, v in outer_bc_raw(pf, batch["outer_extra"], cfg, exact_fields, lam_inf).items():
            push(f"outer_extra_{k}",
                 jnp.sqrt(jnp.asarray(weights.get(f"outer_{k}", weights["outer"]))
                          / v.size) * v.reshape(-1))
    averaged, value = outer_pin_raw(pf, batch["outer"], cfg, exact_fields)
    w_pin = jnp.asarray(cfg.w_pin)
    for k, v in averaged.items():
        push(f"pin_mean_{k}", jnp.sqrt(w_pin) * jnp.mean(v).reshape(1))
    for k, v in value.items():
        push(f"pin_{k}", jnp.sqrt(w_pin / v.size) * v.reshape(-1))
    return jnp.concatenate(rows)


def total_loss(state, batch, cfg, model, exact_fields=None, pde_scale=1.0,
               weights=None, lam_inf=None):
    """state = {'net': params[, 'lam_inf': scalar]};  batch = dict of point arrays.

    `lam_inf` overrides the state leaf, which is how a FROZEN asymptotic value is
    imposed: leaving it optimisable lets the trivial (flat, lambda = const) branch
    re-select itself, since that branch satisfies every Robin condition whenever
    lambda_inf is allowed to equal lambda.
    """
    from .model import point_fields as make_point_fields

    pf = make_point_fields(model, state["net"])
    if lam_inf is None:
        lam_inf = state.get("lam_inf", None)
    if weights is None:
        weights = default_weights(cfg)

    parts = {}
    for k, v in pde_terms(pf, batch["coll"], cfg).items():
        parts[f"pde_{k}"] = v
    for k, v in pde_radial_groups_terms(pf, batch["coll"], cfg).items():
        parts[f"pde_{k}"] = parts.get(f"pde_{k}", 0.0) + v
    rad = pde_radial_terms(pf, batch["coll"], cfg)
    for k, v in rad.items():
        parts[f"pde_{k}"] = v
    inner = inner_bc_terms(pf, batch["inner"], cfg, exact_fields)
    for k, v in inner.items():
        parts[f"inner_{k}"] = v
    extra = (outer_bc_terms(pf, batch["outer_extra"], cfg, exact_fields, lam_inf)
             if batch.get("outer_extra") is not None else {})
    if extra:
        parts["outer_extra"] = sum(extra.values())
    outer = outer_bc_terms(pf, batch["outer"], cfg, exact_fields, lam_inf)
    for k, v in outer.items():
        parts[f"outer_{k}"] = v
    pin = outer_pin_terms(pf, batch["outer"], cfg, exact_fields)
    for k, v in pin.items():
        parts[f"pin_{k}"] = v

    loss = 0.0
    for k in equation_keys(model):
        loss = loss + pde_scale * weights[k] * parts[f"pde_{k}"]
    # own weight, and ramped in with the other equation terms (pde_scale): it is an interior
    # equation, not a boundary datum, so it should not dominate before the ramp finishes
    for k, v in rad.items():
        loss = loss + pde_scale * jnp.asarray(cfg.w_lam_eq_radial) * v
    # per-field override where --eq-weights set one, group weight otherwise
    loss = loss + sum(v * weights.get(f"inner_{k}", weights["inner"]) for k, v in inner.items())
    # per-field override where --eq-weights set one, group weight otherwise
    loss = loss + sum(v * weights.get(f"outer_{k}", weights["outer"]) for k, v in outer.items())
    # The ladder of shells (train.make_batch).  It was in `parts` and in `residual_vector` but NOT
    # here, so SSBroyden -- whose objective is this function -- never felt it: the run "on four
    # shells" trained exactly as if there were one, and the 0.22 it reported for bc_out was a
    # diagnostic of a condition that was not being imposed.  Same weights as the outer group.
    loss = loss + sum(v * weights.get(f"outer_{k}", weights["outer"]) for k, v in extra.items())
    if pin:
        # its own weight: the pins are data, not a decay condition, and the run that needs
        # them is exactly the one where w_outer could not see the level at all
        loss = loss + jnp.asarray(cfg.w_pin) * sum(pin.values())

    return loss, parts
