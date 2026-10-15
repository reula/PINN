# The radial-derivative terms (`--pde-radial`)

**Status: implemented, wired, OFF by default.**  Recorded here so the design does not have to be
rederived; with the inner boundary corrected (commit `0773b6a`, the `h_rA = 0` datum) it is
probably not needed, but it is one flag away.

## What it adds

For every equation group `g` in the loss, and at the collocation points, the RADIAL DERIVATIVE of
that group's residual, with the group's own weight:

    L  +=  sum_g  weights[g] * mean( ( rho**(scale_exps[g] + 1) * d rho R_g )**2 )

`R_g` is the same residual the group already uses (Ricci, harmonic gauge, the lambda equation --
not `compat`, which holds identically in the metric-only scheme), and `d rho` is the derivative
along `n = x/|x|`, evaluated with a single `jax.jvp` per point.

## Why, and what it is for

A loss built from an equation at scattered points cannot see how the residual varies BETWEEN them.
The original motivation was the lambda equation: it is second order, so a loss built from it alone
is blind to `lambda'''`, and an order-3 Robin condition can be satisfied while the far-field level
is wrong -- `runs/production_quad_quarter` reported an outer Robin residual of 7e-06 with
`lambda(rho_out) = 0.31` against 0.9885193, the order-3 combination being a cancellation of terms
of order 0.1 leaving 1.3e-08.  One radial derivative makes the loss see that content.

The same argument is what exposed the non-decaying `l = 2` metric part and the boundary layer: a
condition satisfied in a thin shell is satisfied at the sampled points.

## The four decisions, and why each

1. **Radial only.**  The operators that can be satisfied in a layer are the radial ones (the outer
   Robin ladder, the decay windows), so it is `n . grad`, not the full gradient.  One `jvp`, not
   three.
2. **The same weight as the group, no weight of its own.**  The term IS that group's residual,
   differentiated, so `--eq-weights` sets it.  Consequence to be aware of: to emphasise the
   derivative you must raise the group's weight, which emphasises the residual itself by the same
   factor.  (The older `--w-lam-eq-radial` does the opposite for the lambda equation alone: its own
   weight, defaulting to 0; it remains, unchanged, for the runs that used it.)
3. **One extra rho.**  `R_g` carries `length**(-scale_exps[g])` and its radial derivative one more
   inverse length, so `rho**(scale_exps[g] + 1)` is exactly the factor that makes the term
   dimensionally the same as the group.  In relative mode the residuals are dimensionless ratios and
   the factor is `rho**1`.
4. **No compat.**  `compat` holds identically in the metric-only scheme and is not in the loss, so
   there is nothing for its derivative to constrain.

## Cost

The residual already carries second derivatives of the fields, so this pushes to third.  Measured in
the code for the lambda-only version: about 1.5-1.9x the plain PDE cost per value-and-gradient at
16384 points (the full Jacobian would be ~3x).  All groups come from ONE `jvp`, which shares the
field derivatives, so the cost does not grow with the number of groups.

## How to use it

    --pde-radial                                   # switch it on
    --eq-weights compat=0,ricci=1,gauge=1,lam_eq=4 # and the group weights it rides on

`cfg.pde_radial` is recorded in the run's `config.json`, so the provenance is automatic.  The
diagnostic groups it feeds (`group_terms`) add the term into its own group, so a report reads
`pde_gauge` / `pde_ricci` / `pde_lam_eq` as *residual plus its radial derivative*.

## Where the code is

    stationary/geometry.py   radial_derivative_residuals_batch   -- one jvp, all groups
    stationary/losses.py     pde_radial_groups_raw / _terms      -- gated by cfg.pde_radial
                             the three call sites: group_terms, residual_vector, total_loss
    stationary/problem.py    pde_radial: bool = False
    stationary/train.py      --pde-radial
    tests/test_pde_radial.py off by default; no compat; the objective grows; and with every
                             equation weight at zero the term changes NOTHING (that is what
                             "the same weight as the group" means); the residual vector still
                             squares to the total loss with the terms on

## How to judge whether it helps

Compare a run with and without on the ASYMPTOTIC amplitudes (`S_l / rho**(l+1)` at rho = 25, 50,
100, so the subleading powers are small) and on the bulk residual; a term that only moves the
numbers at the sampled boundary is doing nothing.
