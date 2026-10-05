"""Problem configuration and sampling for the shell  rho_in <= |x| <= rho_out."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from .exact import lambda0_from_k
from .exact import rho_in as rho_in_of_R0


@dataclass
class Config:
    # ---------------------------------------------------------------- physics
    R0: float = 1.0                 # exact-solution parameter used for milestone 1
    rho_out: float = 20.0
    # Constant part of lambda on the inner sphere.  None (the default) DERIVES it so that
    # the solution is the one with lambda -> 1 at infinity:
    #     lambda_0 = k (rho_g - R0)/(rho_g + R0),  rho_g = sqrt(inner_radius^2 + R0^2),  k = 1.
    # lambda -> c lambda is an exact symmetry of the system, so k is a free normalisation
    # and every run uses k = 1 (see exact.lambda0_from_k).  Pass --lam0 to sit on another
    # branch; the banner and the report then print the k it implies.
    lam0: float | None = None
    # True while lambda_0 is derived rather than given: it is recomputed by every
    # __post_init__ call, which is what makes the CLI's build-then-override pattern
    # (Config() first, --R0/--inner-radius after) come out right.  The CLI clears it when
    # --lam0 is passed.  Stored configs from before this field existed default to False,
    # so their recorded lambda_0 keeps its meaning.
    lam0_auto: bool = False
    inner_radius: float | None = None   # areal radius required on the inner sphere
                                        # None -> 2, the radius in the problem statement
    lam_bc_S1: float = 0.0          # dipole amplitude    S1 * z / rho_in
    lam_bc_S2: float = 0.0          # quadrupole amplitude S2 * (z^2-(x^2+y^2)/2)/rho_in^2
    # h_rr on the inner sphere is deliberately NOT constrained by default: the metric on
    # the inner sphere (round, areal radius inner_radius) already fixes the scaling
    # freedom h -> s^2 h, and pinning h_rr as well over-determines the radial gauge --
    # the exact solution does not satisfy it once the inner sphere is placed anywhere
    # other than the canonical chart.  `--inner-h-rr V` still sets it if you want it.
    # Inner boundary data.  "spherical" is the physical problem statement (round sphere of
    # radius inner_radius plus the prescribed polynomial lambda); "reference" takes the data
    # from the exact reference instead, for a manufactured check of a solution whose inner
    # data is not that polynomial -- the Weyl two-black-hole configuration, for instance.
    inner_bc: str = "spherical"      # "spherical" | "reference"
    # Gauge.  "none" is the harmonic (de Donder) condition Gamma^i_{jk} h^{jk} = 0.
    # "cylindrical" imposes the inhomogeneous source
    #     Gamma^i = (h_rhorho - 1) h^{ij} d_j ln rho ,
    # which is what a chart adapted to an axisymmetric solution satisfies (stationary/
    # weyl.py derives it and verify_weyl.py checks it).  The source is built from the
    # CANDIDATE's own metric, so it needs no knowledge of the solution.
    gauge_source: str = "none"       # "none" | "cylindrical"
    # The Weyl two-black-hole reference (symmetric Israel-Khan): two rods on the axis, each
    # of mass weyl_half_length, separated by a gap 2*weyl_half_gap.  rho_in must clear
    # weyl_half_gap + 2*weyl_half_length.
    weyl: bool = False
    weyl_half_length: float = 1.0        # mass of the upper hole (rod length / 2)
    weyl_half_length_b: float | None = None   # the lower hole; None -> equal masses
    weyl_half_gap: float = 0.5
    weyl_n_quad: int = 400
    # Rotate the whole configuration by this angle in the z-x plane (about +y).  The metric
    # and lambda are induced on the inner sphere from the ROTATED exact solution, so the run
    # is a manufactured one like the others, but the solution now has no symmetry at all --
    # the axisymmetric ansatz cannot represent it and the 3-D one must -- and the cylindrical
    # gauge condition rotates with the rods, since it is a statement about the chart and not
    # about the geometry (see stationary.weyl.gauge_source_from_metric).
    weyl_rotate_deg: float = 0.0
    inner_h_rr: float | None = None
    rho_in: float | None = None     # None -> sqrt(4 + R0^2): the areal-radius-2 sphere

    # ------------------------------------------------------- outer boundary
    outer_bc: str = "dirichlet_exact"   # "dirichlet_exact" (milestone 1) | "robin"
    ref_solution: bool = False          # build the exact reference for diagnostics only
    robin_source: bool = False           # inhomogeneous Robin source (manufactured test)
    ref_asymptotic: float | None = None  # if set, reference has lambda -> this value
    lam_inf: float | None = None        # asymptotic lambda; None -> learnable (robin)
    lam_inf_init: float = 1.0           # initial value when lam_inf is learnable
    # decay exponents of the Robin condition  n^i d_i f = -p (f - f_inf)/rho.
    # The system forces h ~ 1/rho^2, Gamma ~ 1/rho^3, lambda ~ 1/rho (measured for
    # the exact solution), so a single exponent p = 1 for every field would impose
    # the wrong asymptotics and create a boundary layer at rho_out.
    robin_exps: dict = field(default_factory=lambda: dict(h=2.0, G=3.0, lam=1.0))
    # Higher-multipole Robin: instead of forcing each field onto its single leading
    # decay power, annihilate the first `robin_order` powers with the Euler operators
    #     prod_{i=0}^{n-1} (rho d_rho + (base+i)) (field - field_inf) = 0 .
    # These operators commute and annihilate exactly rho^-(base+i), so n=1 reproduces
    # the first-order condition above, n=2 allows the next multipole through, etc.
    # Example (lambda, base 1, n=2):  rho^2 lam'' + 4 rho lam' + 2 (lam-1) = 0.
    robin_order: int = 2
    robin_orders: dict | None = None     # per-field override, e.g. dict(h=4, G=1, lam=4)
    robin_include_G: bool = True         # impose the Robin condition on Gamma too.
                                         # In the metric-only (hybrid) schemes Gamma is
                                         # derived from h, so its condition is redundant
                                         # and is what forces fifth derivatives of h.

    # ---------------------------------------------------------------- model
    arch: str = "sym"               # "sym" (spherically symmetric ansatz) | "3d"
    # The Laplacian recipe's network (Laplace_Robin.md section 7.2): 20 wide, 6 deep, and no
    # Fourier features in t.  Small on purpose -- the quasi-Newton phase carries a DENSE
    # inverse Hessian, n_params^2, which is 0.04 GB here and 1.57 GB at the old 64x4 net --
    # and that file measures Fourier features as not worth the extra stiffness the
    # higher-order Robin condition then sees.  Use --width/--depth/--fourier to override.
    width: int = 20
    depth: int = 6
    fourier: int = 0
    decay_feature: bool = False      # add rho_in/rho to the network features

    # ------------------------------------------------------------- sampling
    n_coll: int = 4096
    n_bnd: int = 256
    # Points on the OUTER sphere specifically; None means `n_bnd`.  Its own knob because the
    # outer sphere carries the Robin conditions, the averaged pins and (with --pin-far) the
    # value pins, while the inner sphere carries only the imposed data -- so the two do not
    # want the same resolution, and doubling both in order to double one spends boundary
    # points on the cheaper of the two.
    n_bnd_outer: int | None = None
    resample_every: int = 500
    radial: str = "log"              # "uniform" | "log"

    # --------------------------------------------------- weights / scaling
    ricci_lam_source: float = 1.0     # 0 -> solve Ricci = 0, dropping only lambda's source
    eq_weights: dict = field(default_factory=lambda: dict(
        compat=1.0, ricci=1.0, gauge=1.0, lam_eq=1.0))
    scale_exps: dict = field(default_factory=lambda: dict(
        compat=1.0, ricci=2.0, gauge=1.0, lam_eq=2.0))
    # These exponents are not free weights: sqrt(compat) and Gamma h have dimension
    # 1/length, Ricci and box(lambda) have 1/length^2, so multiplying by rho or rho^2
    # (with scale_ref = None, i.e. the LOCAL rho) makes every residual dimensionless.
    # The consequence is worth knowing before choosing a chart: together with the
    # rho_in- and log-normalised network features and the scale-invariant Euler operator
    # rho d_rho, it makes the WHOLE loss invariant under scaling the shell and whatever
    # lives in it -- measured, not just argued: the manufactured Weyl loss on a covariantly
    # perturbed Weyl candidate is 1.720418e-02 in [7.5, 75], [0.1, 1] and [0.01, 0.1]
    # alike, equal to every printed digit and in every group.  So at a fixed shell RATIO
    # the chart is a no-op here; what a [0.01, 1] recipe really changes is the ratio
    # (100 instead of 10), i.e. how far outside the source the outer sphere sits.
    scale_ref: float | None = None   # None -> use local rho; else a fixed length
    w_inner: float = 10.0
    w_outer: float = 10.0

    # ------------------------------------------------ asymptotic-value pins at rho_out
    # A Robin condition is a DIFFERENTIAL combination, and its kernel contains the leading
    # decaying modes of the exact solution (lambda: rho^-1, rho^-2, rho^-3 with base 1;
    # h: rho^-2, rho^-3, rho^-4 with base 2).  It therefore says nothing about the AMPLITUDE
    # of those modes, and that amplitude is the branch.  Measured on the production geometry:
    # the order-3 lambda combination at rho_out is a cancellation of terms of order 0.1
    # (rho^3 lam''' = +6.77e-02, 9rho^2 lam'' = -2.04e-01, 18rho lam' = +2.05e-01,
    # 6(lam-1) = -6.89e-02) that leaves 1.3e-08 -- so a wrong level is paid for out of the
    # derivatives -- and the exact h deviation from delta IS the rho^-2 kernel mode, so the
    # h condition is satisfied identically (residual 3e-31) whatever that amplitude is.
    # The three quadrupole runs that lost the branch all did so with a lambda outer Robin
    # residual (rms, at rho_out) at or below 2.6e-04, against a level error of 0.68 to 0.96:
    #   production_quad          lambda(rho_out) = 0.0238 (target 0.98852)  rms 2.6e-04
    #   production_quad_half     lambda(rho_out) = 0.0467                  rms 4.6e-06
    #   production_quad_quarter  lambda ~ 0.31 flat (lambda = const)       rms 7.2e-06
    # These pins impose VALUES instead, which cannot be traded against derivatives:
    #   pin_lam    -- the spherical MEAN of lambda at rho_out (the monopole that carries the
    #                 branch); its l >= 1 content stays free for the Robin conditions, which
    #                 is what a later Robin-only relaxation phase needs to fix the multipoles.
    #   pin_h_tan  -- the tangential metric at rho_out, angle by angle (equivalently the areal
    #                 radius there); separates the radial gauge component.
    #   pin_h_rr   -- h_rr at rho_out (its own flag: the inner data deliberately leaves h_rr
    #                 free, and this is the radial gauge component).
    # The pin values are the exact reference's own values at rho_out, so nothing is
    # hard-coded; for the quadrupole problem the true solution differs from them by
    # ~S2 (rho_in/rho_out)^3 = 8e-08, i.e. 800x below the order-1 Robin floor (6.6e-05).
    # Both pins act on the same object, the spherical mean minus the asymptotic value, and
    # they are the first two members of the SAME operator family --robin-orders uses:
    #     order 0:  <lam> - lam_inf = 0                      (the value; prod over no factors)
    #     order 1:  rho d_rho <lam> + (<lam> - lam_inf) = 0   (the first-order combination)
    # Order 0 is what `pin_lam` does and needs the reference's own value at rho_out.  Order 1
    # is `pin_lam_robin` and needs NO reference, only lam_inf, which is what makes it the one
    # pin available to a run whose inner data are angular (S1/S2): for those the spherical
    # reference is not a solution at all, so order 0 would be pinning a departure.  The
    # general order-n combination is `robin_operator(lam, x, base=1, order=n, inf_val=lam_inf)`
    # averaged, so extending this to order 2 or 3 is one parameter if it is ever wanted.
    # Neither is a substitute for the POINTWISE Robin condition at the configured order: the
    # mean fixes the monopole, the pointwise condition is what constrains the local, angular
    # content, and a run needs both.
    pin_lam: bool = False
    # order 1, and it constrains the SHAPE of the monopole rather than its size: `rho d_rho +
    # 1` annihilates rho^-1 exactly, so it is blind to the monopole amplitude a in
    # lam = 1 + a/rho -- which is deliberate.  The amplitude is not known a priori for a run
    # with angular inner data (that is what Robin conditions are for); it is fixed by the
    # inner data and the interior, while this pin fixes the rho^-2 and rho^-3 terms and
    # rejects the constant branch outright (measured: the exact solution gives 1.7e-11, the
    # lambda ~ 0.31 branch 4.8e-01).  Its floor on the exact solution is the order-1 residual
    # -2 R0^2 / rho_out^2 (4.2e-06 at ratio 400), i.e. negligible at the solution and 10
    # orders decisive away from it.
    pin_lam_robin: bool = False
    pin_h_tan: bool = False
    pin_h_rr: bool = False
    # The AVERAGED Robin condition for the metric -- the same construction as pin_lam_robin,
    # applied to the two objects the value pins above act on:
    #     <h_rr>:  rho d_rho <h_rr> + 3 (<h_rr> - 1) = 0
    #     <g2>:    rho d_rho <g2>   + 2 (<g2>   - 1) = 0       g2 = (tr h - h_rr)/2
    # The bases are not chosen, they are measured: the exact solution's <h_rr> - 1 decays as
    # rho^-3.000 and its <g2> - 1 as rho^-1.998, so `rho d_rho + b` annihilates each for
    # b = 3 and b = 2 respectively -- NOT the base 2 the POINTWISE h condition uses, which is
    # why this cannot reuse that exponent.  Measured residuals on the exact solution at
    # rho_out = 4: h_rr 5.5e-14 (machine zero -- the deviation is purely rho^-3, with none of
    # the admixture that leaves the lambda pin a 4.2e-06 floor), g2 5.4e-09.
    #
    # What it holds, and it is exactly what the value pins held, but with no reference: a
    # wrong level c at rho_out moves the pin by b*c, so a 1% error is 3e-02 (h_rr) or 2e-02
    # (g2), eleven orders above the floor.  <h_rr> is the CHART scale -- the harmonic
    # diffeomorphism freedom makes h_rr = 1/F'^2 a constant, and a constant is rejected by 3x
    # itself -- and <g2> is the size of the outer sphere.  This is the reference-free
    # replacement for pin_h_rr and pin_h_tan.
    #
    # Its blind spot is the lambda pin's: `rho d_rho + 3` annihilates a rho^-3 deviation of ANY
    # amplitude, so it fixes the SHAPE and the level, not the size of the deviation.  And it is
    # the averaged value only -- the pointwise order-3 condition above is untouched and is what
    # constrains the angular content.
    pin_h_robin: bool = False
    # The decay powers the two averaged metric conditions use, as `{"h_rr": b, "g2": b}`.
    # None means the measured defaults (3, 2).  THEY ARE A PARAMETER BECAUSE THEY ARE AN
    # ASSUMPTION: those two numbers were measured on the SPHERICALLY SYMMETRIC reference, and
    # `h_rr - 1 ~ rho^-3` comes from its chart freedom (the c2 term of F = c1 rho + c2 F2),
    # which says nothing about a solution whose inner data are angular.  The multipole
    # expansion of THIS problem need not match the spherical one -- taking the angular average
    # is precisely what avoids assuming that -- so the powers have to be settable, and any run
    # that relies on them is relying on a measurement, not on a derivation.
    h_robin_bases: dict | None = None

    # Appends `res(jit=<compiled executables> maps=</proc/self/maps lines>)` to each per-block
    # line, so a run that dies of a host allocation failure says whether it was COMPILING
    # itself to death.  A JIT cache that grows block after block means something in the phase
    # retraces, and every trace leaks LLVM section memory that is never returned: two runs of
    # this problem died at exactly block 23, at 2300 iterations, after 1.5 h, with
    # "LLVM ERROR: Unable to allocate section memory!" -- identical failure points across
    # different code and different thread counts, which is accumulation, not contention.
    log_resources: bool = False
    w_pin: float = 100.0        # weight of the pin group, independent of w_outer

    # ------------------------------------- radial derivative of the lambda equation
    # The lambda-equation is second order, so the loss is blind to lambda''', and lambda'''
    # at rho_out is where the order-3 Robin condition lets a wrong far-field level hide (its
    # kernel is rho^-1, rho^-2, rho^-3, so its residual is a cancellation of terms of order
    # 0.1 leaving 1.3e-08 -- measured on runs/production_quad_quarter: outer Robin 7e-06 with
    # lambda(rho_out) = 0.31 against 0.9885193).  Weighting d(rho)/d rho of that residual,
    # scaled by rho^3 to stay dimensionless (lam_eq is 1/length^2, its radial derivative
    # 1/length^3, and scale_exps['lam_eq'] = 2), makes the loss see exactly that content.
    # 0 (the default) turns the term off entirely: nothing is even differentiated.
    w_lam_eq_radial: float = 0.0

    pde_ramp_steps: int = 0     # ramp the PDE weights in over this many steps (0 = off)
    # Period in OPTIMISER ITERATIONS, counted on one counter that Adam and the
    # quasi-Newton phase both advance, so `--steps 0 --reweight-every 1500` reweights every
    # 1500 quasi-Newton iterations.  It used to be an Adam-step period, which meant a run
    # that skipped or shortened Adam never reweighted at all
    # (runs/production_quad_quarter_pin_r400 asked for 1500, ran 8000 quasi-Newton
    # iterations, and logged no [reweight] line).
    reweight_every: int = 2000  # gradient-norm adaptive reweighting period (0 = off)
    reweight_max_ratio_inv: float = 0.5   # per-update cap: weights move by at most 2x
    # Cumulative band: over a whole run a PDE weight may not move further than this
    # factor from its configured value.  Without it the rule w ~ 1/||grad term|| drifts:
    # in runs/control_ord1 the `compat` weight (identically satisfied in the metric-only
    # schemes, residual ~1e-21) grew 64x while `lam_eq` -- the group with the largest
    # residual -- was driven down 2.1x, i.e. the most violated equation ended up with the
    # least relative weight.  Groups whose gradient is below noise (`reweight_floor`
    # relative to the largest) are excluded from the target and left alone entirely.
    reweight_band: float = 4.0
    reweight_floor: float = 1e-6

    # ---------------------------------------------------------- optimisation
    steps: int = 20000
    lr: float = 1e-3
    lbfgs_steps: int = 300
    # Quasi-Newton phase.  "ssbroyden" runs Crunch's self-scaling Broyden (the JAX fork
    # tracked here as Jax/Crunch, also looked for inside Stationary/ and above the repo --
    # see train._crunch_candidates -- imported lazily and falling back to optax.lbfgs when
    # none of them is present); "lbfgs" forces the optax path.  SSBroyden carries a
    # *dense* inverse-Hessian estimate, n_params^2: the production network (13 828
    # parameters) needs 1.53 GB in float64 and 0.76 GB in float32, which qn_max_H_gb caps.
    qn_method: str = "ssbroyden"    # "ssbroyden" | "lbfgs"
    qn_max_H_gb: float = 2.0        # refuse the dense SSBroyden estimate above this
    # "Run until the loss plateaus": the quasi-Newton phase is done in blocks of
    # `qn_block` iterations (the inverse Hessian is carried across them), and the run stops
    # when `patience` consecutive blocks improve the loss by less than `plateau_tol`
    # relative -- or when the gradient norm falls below `qn_gtol`, or at the `lbfgs_steps`
    # cap, whichever comes first.  The Adam phase stops on the same rule (checked every
    # `log_every` steps).  `plateau_min_iters` keeps it from stopping in the first blocks,
    # where the loss is still falling fast.
    # VTK export for VisIt (stationary/vtk.py, run by postprocess.sh when this is true):
    # a graded Cartesian grid of the shell, in the PHYSICAL coordinates, one file per run.
    make_vtk: bool = False
    vtk_n_half: int = 20           # points per half axis (geometric -> clusters at rho_in)
    vtk_physical_inner: float = 1.0  # the inner radius in the coordinates to write out
    qn_block: int = 100
    qn_gtol: float = 1e-9
    plateau_tol: float = 1e-4
    plateau_patience: int = 3
    plateau_min_iters: int = 100
    log_every: int = 200
    seed: int = 0
    outdir: str = "runs/m1"
    init_from: str | None = None
    ckpt_every: int = 0             # write a resumable checkpoint every N Adam steps (0 = off)
    resume: str | None = None       # checkpoint path, or "auto" for <outdir>/ckpt.pkl
    make_figures: bool = True       # lambda at the inner sphere + outer multipoles

    def __post_init__(self):
        if self.weyl and self.weyl_rotate_deg and self.arch in ("sym", "sym_hybrid",
                                                                 "axisym_hybrid"):
            # a rotated configuration has no axis of symmetry left
            self.arch = "mlp"
        if self.weyl and self.arch in ("sym", "sym_hybrid"):
            # Not a preference: these ansaetze have no angular freedom, so the loss has no
            # zero at a two-black-hole field.  Use the axisymmetric ansatz (valid, since
            # the rods are on the axis) or the general 3-D one.
            self.arch = "axisym_hybrid"
        if self.weyl and not jax.config.jax_enable_x64:
            # The Weyl reference needs the dynamic range of float64: its k comes from a
            # quadrature over s = 1/rho' whose extreme node sits at rho' ~ 1e7, where the
            # integrand's second derivative is ~1e-33 against intermediates ~1e-21.  In
            # float32 that is below the ~7 digits available, U_rho^2 - U_z^2 loses its
            # sign, and the SECOND derivative of h comes out NaN -- which would show up
            # only as a NaN outer Robin term, far from the cause.  Fail here instead.
            raise ValueError(
                "cfg.weyl needs float64 (the Weyl k quadrature is not representable in "
                "float32): run with JAX_ENABLE_X64=1 or jax.config.update("
                "'jax_enable_x64', True) before importing jax.numpy.")
        if self.rho_in is None:
            self.rho_in = rho_in_of_R0(self.R0)
        self.rho_in = float(self.rho_in)
        if self.inner_radius is None:
            # The inner sphere is the round sphere of AREAL RADIUS 2 -- the problem
            # statement -- and `rho_in_of_R0(R0) = sqrt(4+R0^2)` is by construction the
            # coordinate radius at which the canonical-chart solution has areal radius 2.
            #
            # It must NOT default to rho_in: the coordinate sphere |x| = rho_in has
            # tangential metric (rho_in/rho_in)^2 = 1, i.e. flat, while the exact solution
            # has h_tan = 1 - R0^2/rho_in^2 = 4/(4+R0^2) there (0.8 for R0 = 1). Asking for
            # the former makes the inner data contradict the exact solution used for
            # `dirichlet_exact` (and for the Robin source), so no metric can satisfy both:
            # the run then converges to a hybrid that is 25% off near the inner sphere.
            # This default was rho_in between commits 67a802a and this one; the runs made
            # before it (m1_sym, m1_3d) used the areal radius 2.
            self.inner_radius = 2.0
        if self.lam0 is None or self.lam0_auto:
            # Every run sits on the branch with lambda -> 1 (k = 1): the asymptotic value
            # is a free normalisation (lambda -> c lambda is an exact symmetry) and 1 is
            # the physically interesting one.  With R0 = 1 and areal radius 2 this gives
            # lambda_0 = 1/phi^2 = 0.381966; with the M2 control geometry (R0 = 1/sqrt3,
            # areal radius 1) it gives 1/3.  An explicit --lam0 clears lam0_auto and wins.
            self.lam0 = lambda0_from_k(self.R0, 1.0, self.inner_radius)
            self.lam0_auto = True
        if self.pin_lam and self.pin_lam_robin:
            # First, because it is a contradiction between two flags and independent of
            # whether a reference exists: reporting "no reference" for it would send the
            # reader off to add --ref-solution for a pin they should not have asked for.
            raise ValueError(
                "--pin-lam pins the MEAN VALUE of lambda and --pin-lam-robin pins the order-1 "
                "ROBIN COMBINATION of the mean; both act on the same monopole, so ask for one "
                "of them.")
        if self.pin_h_robin and (self.pin_h_tan or self.pin_h_rr):
            # Same reason: the value pins and the averaged Robin conditions act on the same two
            # quantities (h_rr and the tangential part), one against the reference and one
            # against lam_inf alone.  Asking for both would weight the same constraint twice.
            raise ValueError(
                "--pin-h-tan / --pin-h-rr pin the metric at rho_out to the reference's VALUES, "
                "and --pin-h-robin pins the same two quantities through their AVERAGED Robin "
                "combinations; ask for one of them.")
        if self.pin_lam or self.pin_h_tan or self.pin_h_rr:
            # The pins are differences from the exact reference at rho_out, so that object
            # has to exist.  Fail here rather than inside the jitted loss: a missing
            # reference there surfaces as a TypeError from `None(x)` in the middle of a
            # traceback that names neither the flag nor the reason.
            # pin_lam_robin is NOT in this list on purpose: it is the mean of the order-1
            # Robin combination, which needs lam_inf and nothing else, so it is the one pin a
            # run with no exact solution can use.
            has_ref = (self.weyl or self.ref_solution or self.robin_source
                       or self.outer_bc == "dirichlet_exact")
            if not has_ref:
                which = ", ".join(n for n, on in (("pin_lam", self.pin_lam),
                                                  ("pin_h_tan", self.pin_h_tan),
                                                  ("pin_h_rr", self.pin_h_rr)) if on)
                raise ValueError(
                    f"{which} compare the candidate against the exact reference at "
                    f"rho_out, and this run builds no reference.  Add --ref-solution "
                    f"(with --ref-asymptotic k), or drop the pin flags.  For the lambda "
                    f"monopole there is --pin-lam-robin, which pins the order-1 Robin "
                    f"combination of the mean and needs no reference.")


def reference_is_departure_only(cfg) -> bool:
    """True when the exact reference cannot be a solution of THIS run's problem.

    The reference is spherically symmetric, so it cannot carry angular inner data (S1, S2).
    For those runs it is not an exact solution to compare against: it violates the imposed
    inner data by construction, and every difference from it is DEPARTURE -- how far the
    solution has moved from the spherical one, which is the point of the angular data -- not
    error.  Nothing in the numbers distinguishes the two, so the plots must not present them
    as errors (`evaluate`, `profile`) or as an error row (`report`).

    `inner_bc = "reference"` is the exception and the reason this is a function rather than a
    flag test: there the inner data ARE the reference's own, as in the Weyl runs, so the
    comparison is exactly the error.
    """
    if getattr(cfg, "inner_bc", "spherical") == "reference":
        return False
    return bool(getattr(cfg, "lam_bc_S1", 0.0) or getattr(cfg, "lam_bc_S2", 0.0))


def lam_inner_bc(x, cfg):
    """lambda prescribed on the inner sphere:

        lambda = lam0 + S1 z/rho_in + S2 (z^2 - (x^2+y^2)/2)/rho_in^2

    written in the harmonic coordinates (x, y, z); on |x| = rho_in it reduces to
    lam0 + S1 n_z + S2 (3 n_z^2 - 1)/2.
    """
    z = x[..., 2]
    r2 = jnp.sum(x * x, axis=-1)
    return (cfg.lam0 + cfg.lam_bc_S1 * z / cfg.rho_in
            + cfg.lam_bc_S2 * (z * z - 0.5 * (r2 - z * z)) / cfg.rho_in**2)


def sphere_directions(key, n: int) -> jnp.ndarray:
    """Uniform directions on S^2."""
    u = jax.random.normal(key, (n, 3))
    return u / jnp.linalg.norm(u, axis=-1, keepdims=True)


def sample_shell(key, n: int, cfg: Config) -> jnp.ndarray:
    """Collocation points in the shell (uniform in rho or in log rho)."""
    k1, k2 = jax.random.split(key)
    if cfg.radial == "log":
        rho = jnp.exp(jax.random.uniform(k1, (n,), minval=math.log(cfg.rho_in),
                                         maxval=math.log(cfg.rho_out)))
    else:
        rho = jax.random.uniform(k1, (n,), minval=cfg.rho_in, maxval=cfg.rho_out)
    return rho[:, None] * sphere_directions(k2, n)


def sample_sphere(key, n: int, radius: float) -> jnp.ndarray:
    return radius * sphere_directions(key, n)
