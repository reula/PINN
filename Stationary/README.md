# Stationary Einstein system on a spherical shell — JAX PINN

Solve, in the region between two concentric spheres, the stationary system

    Ricci(h)_ab = (1/(2 lambda^2)) grad_a lambda grad_b lambda            (R)
    h^{ab} grad_a grad_b lambda = (1/lambda) h^{ab} grad_a lambda grad_b lambda   (L)

together with the harmonic (de Donder) gauge condition

    Gamma^a_{bc} h^{bc} = 0                                              (G)

as a first-order system in the variables `(h, Gamma, lambda)`.

## 1. Structure of the system (what to keep in mind)

* With `phi = log lambda`, (L) is exactly `Delta_h phi = 0` and (R) becomes
  `R_ab = (1/2) phi_a phi_b`.  The divergence of (R) reproduces (L) — the second
  equation *is* the Bianchi identity of the first, so the pair is consistent
  rather than overdetermined.
* 4D reading: static Einstein + massless scalar field,
  `g = -dt^2 + h`, `phi = log lambda` (normalisation `8 pi G = 1/2`).
* First-order counting: metric compatibility (18) + Ricci (6) = 24 equations for
  the 24 unknowns `(h, Gamma)`, which carry **exactly a 3-parameter gauge
  degeneracy**; the gauge condition (G) removes it.
* (G) is *chart dependent*: it says the coordinates you use are harmonic.  For any
  metric that is spherically symmetric **in polar coordinates** one has
  `Gamma^theta = -cos(theta)/(h_theta_theta sin(theta)) != 0`, so the polar chart
  can never satisfy (G) (flat space is harmonic in Cartesian, not in polar,
  coordinates).  All computation is therefore done in the harmonic
  "Cartesian-like" chart `x^i = rho n^i`, and (G) is imposed there
  (equivalently `Delta_h x^i = 0`, see `geometry.laplacian`).

## 2. Exact solution (validation asset)

In the harmonic chart the general spherically symmetric solution is the
two-parameter family

    h_ij   = (1 - R0^2/rho^2) delta_ij + (R0^2/rho^2) n_i n_j
    lambda = k (rho - R0)/(rho + R0)                      (R0 >= 0, k != 0)

i.e. `h = d rho^2 + (rho^2 - R0^2) dOmega^2`.  `R0 = 0` is flat space with
constant `lambda`; `lambda -> 1/lambda` is another symmetry.  The **areal**
(geometric) radius of a coordinate sphere is `sqrt(rho^2 - R0^2)`, so the sphere
carrying the round metric of radius 2 sits at `rho_in = sqrt(4 + R0^2)`, and
`lambda = lambda_0` there requires `k = lambda_0 (rho_in + R0)/(rho_in - R0)`.

The harmonic gauge leaves the residual freedom of harmonic diffeomorphisms; for
symmetric data the harmonic radial coordinate is any
`F(rho) = c1 rho + c2 [2 R0 + rho log((rho-R0)/(rho+R0))]`
(both members solve `(f^2 F')' = 2F`, `f^2 = rho^2 - R0^2`).  This is why the
geometric sphere of areal radius 2 may be placed at *any* chosen coordinate
radius while keeping the gauge harmonic — see `exact.exact_fields_in_harmonic_chart`
and `tests/test_pipeline.py::test_harmonic_chart_freedom`.

## 3. Layout

    stationary/geometry.py     Christoffel, Ricci and the four residual groups
    stationary/exact.py        exact solution family, chart freedom, boundary data
    stationary/model.py        MLP: (x,y,z) -> h_ij(6), Gamma^i_jk(18), lambda(1)
    stationary/problem.py      Config, domain sampling
    stationary/losses.py       PDE residuals + inner/outer boundary conditions
    stationary/train.py        Adam -> L-BFGS driver, checkpoints, reports
    stationary/evaluate.py     diagnostics and figures for a checkpoint
    stationary/profile.py      lambda (and other fields) as a function of rho
    stationary/report.py       one-screen text report of a run (paste-able)
    stationary/compare.py      side-by-side table of several runs (the analysis step)
    stationary/invariants.py   chart-independent content (r_a, lambda, Ricci scalar)
    stationary/multipoles.py   spherical-harmonic decomposition and figures
    stationary/diagnostics.py  residual/error/boundary-geometry reports
    postprocess.sh             figures + lambda_vs_rho.png + report.txt of a run
    run_hub.sh                 detached run on a JupyterHub (see HUB.md)
    run_ladder.sh              the control ladder: order 1/2/4, x64, capacity, dipole
    tests/test_pipeline.py     end-to-end tests (see below)
    verify_exact_solution.py   standalone regression test of the exact solution

## 4. Index conventions (regression tested — do not change casually)

`jax.jacfwd` puts the **input axis LAST**:

    J  = jacfwd(h)(x)          J[i,j,a]    = d_a h_ij
    JG = jacfwd(G)(x)          JG[i,j,k,a] = d_a G[i,j,k]
    G[i,j,k] = Gamma^i_{jk}    symmetric in (j,k)

Residual groups (all residual machinery is shared by the network and by the
tests through a single `fields(x) -> (h, Gamma, lambda)` callable):

    compat[a,b,c] = d_a h_bc - G^d_{ab} h_dc - G^d_{ac} h_bd          (18)
    ricci[i,j]    = R_ij(G) - (1/(2 lam^2)) d_i lam d_j lam           (6)
    gauge[i]      = G^i_{jk} h^{jk}                                   (3)
    lam_eq        = h^{ij}(d_i d_j lam - G^k_{ij} d_k lam)
                    - (1/lam) h^{ij} d_i lam d_j lam                  (1)

Each group is multiplied by `rho^p` (`p = 3, 4, 3, 3`), which makes the residual
dimensionless *relative to the size of its own terms*, since the exact solution
obeys `compat, gauge, lam_eq ~ rho^-3` and `ricci ~ rho^-4`.

## 5. Boundary conditions

Inner sphere (`rho = rho_in`), as specified by the user:

* `lambda = lambda_0` (plus the optional `S1`, `S2` angular terms).  **`lambda_0` is
  derived so that the solution is the one with `lambda -> 1` at infinity**:
  `lambda_0 = k (rho_g - R0)/(rho_g + R0)` with `rho_g = sqrt(inner_radius^2 + R0^2)` and
  `k = 1`, i.e. `1/phi^2 = 0.381966` for `R0 = 1` with the areal-radius-2 sphere and `1/3`
  for the M2 control geometry (`R0 = 1/sqrt3`, areal radius 1).  `lambda -> c lambda` is an
  exact symmetry of the system (`Ricci` is unchanged and so is `(1/2lambda^2) dlambda
  dlambda`), so `k` is a free normalisation; **every run uses `k = 1`**.  An explicit
  `--lam0` still selects another branch, and the run then prints the `k` it implies.
* the induced metric is the round metric of **areal radius `inner_radius`** (default 2, the
  radius in the problem statement), implemented as
  `h - h_rr n n - (inner_radius^2/rho^2)(I - n n) = 0`.

No condition is imposed on `h_rr`: that round-metric condition already fixes the scaling
freedom `h -> s^2 h`, and pinning `h_rr` as well over-determines the radial gauge — the
exact solution does not satisfy it once the inner sphere is placed anywhere other than the
canonical chart.  (`Config.inner_h_rr` exists for special cases; no documented command uses
it.)

`inner_radius` defaults to **2**, which is the problem statement and is what
`rho_in_of_R0(R0) = sqrt(4+R0^2)` places at `rho_in` in the canonical harmonic chart: the
exact solution has `h_tan = 1 - R0^2/rho_in^2 = 4/(4+R0^2)` there, i.e. `0.8` for
`R0 = 1`, *not* flat.  It must not default to `rho_in` (it did for a while, commit
`67a802a`): that asks for `h_tan = 1`, which contradicts the Dirichlet/Robin-source data
taken from the same exact solution, so no metric can satisfy both boundary conditions and
the run converges to a compromise ~25% off near the inner sphere.  `train.build` now
checks this before starting and prints the offending term and the fix; `report.py` prints
a `reference vs the imposed inner data` line, and `tests/test_pipeline.py` pins the
default.  If you want a scaled solution (as the M2 runs do, at `rho_in = 1`), say so with
`--inner-radius`.

Outer sphere (`rho = rho_out`), two modes:

* `dirichlet_exact` — `h` and `lambda` from the exact solution (milestone 1:
  a manufactured-solution test with a known answer)
* `robin` — `n^i d_i field = -(field - field_inf)/rho_out` on `h` (with
  `field_inf = delta_ij`), on `Gamma` (`field_inf = 0`) and on `lambda`
  (`field_inf = lambda_inf`, optionally a learnable scalar)

`--lam-inf` (the value the Robin condition drives `lambda` to) must equal the `k` implied by
the inner data; with the derived `lambda_0` that is `k = 1` automatically.  `train.build`
warns when they disagree.  That inconsistency is what parked `n2_dipole`: `lambda_0 = 1/3`
with `R0 = 1` on the areal-radius-1 sphere gives `k = 1.943`, while its Robin condition drove
`lambda` to 1.

Normalisation of the runs already in `runs/`, for reading old figures: `m2R2_asym1`,
`m2R3_symhybrid` and `m2R4_realrobin` are on the `k = 1` branch; `m1_sym`, `m1_3d` and
`m2R1_trivial` used `lambda_0 = 1` (`k = phi^2 = 2.618`) and `n2_dipole` `k = 1.943`.  The
metric results are unaffected either way — the symmetry rescales `lambda` only — but `lambda`
values in those reports must be divided by their `k` before comparing with a `k = 1` run.

## 6. Usage

    .venv/bin/python -m pytest tests/ -q                       # test suite
    .venv/bin/python verify_exact_solution.py                  # standalone check
    .venv/bin/python -m stationary.train --steps 20000 --lbfgs-steps 2000 \
        --outdir runs/m1
    .venv/bin/python -m stationary.evaluate runs/m1            # diagnostics + figures
    .venv/bin/python -m stationary.profile  runs/m1            # lambda vs rho (+ table)
    .venv/bin/python -m stationary.report   runs/m1            # text report
    ./postprocess.sh runs/m1                                   # all three of the above
    .venv/bin/python -m stationary.compare runs/m1 runs/m2     # side-by-side table
    .venv/bin/python -m stationary.invariants --outdir runs/m1 # chart-independent content

`postprocess.sh` writes every figure plus `report.txt` into the run directory; it is what
`run_hub.sh` runs automatically when a training process ends (see `HUB.md` §4), so a
finished — or crashed — run is complete without any further command. It caps the CPU
threads (`POST_THREADS`, default 4 — XLA otherwise takes every core of the node, which is
slower for graphs this small) and caches XLA compilations in `.jaxcache`, which is the
difference between ~1 min and ~30 s (or ~10 s for `--only report`).

Useful flags: `--R0`, `--lam0`, `--rho-out`, `--n-coll`, `--width`, `--depth`,
`--outer-bc {dirichlet_exact,robin}`, `--pde-ramp-steps`, `--w-inner`,
`--w-outer`, `--init-from <params.pkl>` (continue a run), `--seed`.

## 7. Tests

`tests/test_pipeline.py` contains the end-to-end acceptance tests.  The crucial
one is `test_loss_is_zero_on_exact_solution`: the exact solution is packed exactly
as the network packs its outputs and fed through the *same* loss function, and the
total loss (all four residual groups + both boundary conditions) must come out at
machine zero.  This validates the whole pipeline independently of whether the
optimiser can converge.  Others check the index conventions against known
curvatures (flat space in polar coordinates, round `S^3`), the equivalence
`Gamma^i = -Delta_h x^i` by two independent routes, and the residual gauge freedom
of the harmonic chart.

### 7.1 Production check: the Weyl two-black-hole solution

`python -m verify_weyl` is the *production* counterpart of the pytest suite: instead of
checking the code against itself it checks it against a genuinely different exact solution —
the symmetric two-rod Weyl (Israel–Khan) family, i.e. **two black holes on the axis, held
apart by the conical strut between them**.  It needs only jax (no Crunch, no GPU, no
training) and about 10 s, it is run by `run_hub.sh --check`, and it exits non-zero if any
check fails.  Fourteen checks, all passing:

* the Weyl fields satisfy this code's two geometric equations in the Weyl chart — `ricci`
  and `lam_eq` residuals at ~1e-17 — while compatibility is exact by construction, since
  `Gamma` is built from `h`;
* that chart is **not** harmonic: `Gamma^i_{jk} h^{jk}` is 3.9e-03 at `rho = 7.5`, which is
  the chart and not the physics.  The inhomogeneous source
  `Gamma^i = (h_rhorho − 1) h^{ij} d_j ln rho` drives it to machine zero, so the Weyl
  solution solves the **full** system — PDE, scalar equation and gauge — with no coordinate
  transformation.  That closed form needs only the metric at the point, so it can be imposed
  on a candidate solution, which is what a production gauge condition has to be;
* the closed form agrees with the independent autodiff connection to 2e-17;
* the decay exponents are exactly the ones the Robin conditions assume: `h − I ~ rho^-2`,
  `lam − 1 ~ rho^-1`, `Gamma ~ rho^-3` (so `robin_exps` needs no retuning for this solution);
* the strut is present: `k`, which sets the cone angle `2 pi e^-k` on the axis, is −0.588
  across the gap and 0 outside the rods.

The mapping is `h = e^{2k}(drho^2 + dz^2) + rho^2 dphi^2`, `lam = e^{2U}`; the derivation of
that and of the gauge source is in `stationary/weyl.py`.  All parameters
(`--half-length`, `--half-gap`, `--n-quad`, `--rho-in`, `--rho-out`, `--n-points`) are on the
command line, and `Rods(spans)` accepts an arbitrary set of rods for later configurations.
Note the inner sphere must clear `rho = axis_extent` (2.5 for the default), and the strut's
`k` near the axis is the least-converged number here, since the quadrature is tuned for the
region outside the holes.

The same script closes the loop with the trainer.  It builds a candidate whose forward pass
*is* the Weyl solution and feeds it to `total_loss` — the exact function `train.py` minimises —
with the inner data and the Robin source taken from the reference and the gauge source built
from the candidate's own metric.  The total loss comes out at **1.3e-28**: the loss has a
genuine zero at a two-black-hole configuration, so a run that converges is solving the system
this script verifies and not a spherical stand-in.  The check has teeth: the same loss with the
harmonic condition (`gauge_source = "none"`) does **not** vanish, because the Weyl chart is not
harmonic.

### 7.2 Production run: training on the Weyl configuration

    python -m stationary.train --weyl --outdir runs/weyl

`--weyl` sets up the manufactured Weyl problem end to end, and every piece of it is an
ordinary production code path rather than a special case:

* `inner_bc = "reference"` takes the inner data — all six components of `h` and `lambda` —
  from the exact reference instead of from the round sphere plus the polynomial `lambda`.
  That is what this configuration needs: at `rho_in` the two black holes make the induced
  metric non-spherical, and `lambda` there is not
  `lam0 + S1 z/rho + S2 (z^2 - (x^2+y^2)/2)/rho^2`.  `inner_bc = "spherical"` stays the
  default and is unchanged;
* `gauge_source = "cylindrical"` imposes `Gamma^i = (h_rhorho - 1) h^{ij} d_j ln rho`, the
  inhomogeneous de Donder condition that a chart adapted to an axisymmetric solution
  satisfies.  The source is built from the **candidate's** own metric (derivatives of its
  `h`), so it presupposes nothing about the solution: the network is still the only thing
  that knows where the black holes are;
* `outer_bc = "robin"` with `robin_source` and `lam_inf = 1` (the Weyl asymptotic value),
  and `robin_exps` unchanged, since §7.1 measures exactly `h ~ rho^-2`, `lambda ~ rho^-1`,
  `Gamma ~ rho^-3` for this family;
* the reference enters the loss *only* as boundary data and as the Robin source; elsewhere it
  is a comparison asset, exactly as in the manufactured scalar-field runs;
* `rho_in` must clear the rods, which reach `rho = half_gap + 2*half_length`, so `--weyl`
  defaults to `rho_in = 3 * axis_extent` (7.5 for the default rods) and
  `rho_out = 10 * rho_in`.  `--weyl-half-length`, `--weyl-half-gap` and `--weyl-n-quad`
  parameterise the configuration, and `--rho-in`/`--rho-out` override the shell;
* the run **must** be in float64 (`JAX_ENABLE_X64=1`), and `--weyl` enforces that with an
  explicit error.  The reason is specific: `k` comes from a quadrature over `s = 1/rho'`
  whose extreme node sits at `rho' ~ 1e7`, where the integrand's second derivative is
  ~1e-33 against intermediates ~1e-21.  In float32 that is below the seven digits
  available, `U_rho^2 - U_z^2` loses its sign, and the **second** derivative of `h` comes
  out NaN — which surfaces only as a NaN outer Robin term, far from its cause.

A run that converges here has solved the Einstein-scalar system on a shell whose inner data is
the field of two black holes — the first configuration in this project with no symmetry.

## 8. Results

### 8.1 Exact solution and pipeline validation

* `verify_exact_solution.py`: all four residual groups vanish to machine precision
  ($10^{-16}$ or better) for $h=d\rho^2+(\rho^2-R_0^2)d\Omega^2$,
  $\lambda=k(\rho-R_0)/(\rho+R_0)$; the Ricci routine reproduces $R_{ij}=(2/a^2)h_{ij}$
  on the round $S^3$; $\Gamma^i$ agrees with $-\Delta_h x^i$ by two independent routes.
* `tests/test_pipeline.py` (16 tests): the *whole* loss — four residual groups plus
  both boundary conditions — evaluates to machine zero when fed the exact solution,
  for both the independent-$\Gamma$ and the hybrid (metric-only) formulations.

### 8.2 Milestone 1 (manufactured solution: BC data from the exact solution)

Configuration: $R_0=1$, inner sphere at $\rho_{\rm in}=\sqrt5$ (the areal-radius-2
sphere), outer at $\rho=20$, Dirichlet data from the exact solution at both spheres.
The runs below were made with $\lambda_0=1$, i.e. on the
$k=(\sqrt5+1)/(\sqrt5-1)=\varphi^2$ normalisation; runs made now derive
$\lambda_0=1/\varphi^2=0.381966$ for the same geometry so that $\lambda\to1$.  Since
$\lambda\to c\lambda$ is an exact symmetry, the metric results ($\max|\Delta h|$ and
friends) are the same in both normalisations, and only the quoted $\lambda$ values
differ by the factor $\varphi^2$.

What was learned about the optimisation (all runs in `runs/`):

1. **Initialisation matters.** With a standard random final layer the initial loss is
   $\sim10^9$ (the metric is far from positive definite); a small final-layer init
   (start near flat space) fixes it.
2. **The boundary conditions must be won first.** The loss has a strong degenerate
   direction: $h=\delta$, $\Gamma=0$ with any *slowly varying* $\lambda$ has almost
   zero residual. With BC weights of order 1 the optimiser satisfies the BCs only at
   the two spheres and leaves a 30 % error in $\lambda$ in between. A BC-first ramp of
   the PDE weights (`--pde-ramp-steps`) plus large BC weights fixes this: the boundary
   data then hold to $10^{-10}$ and the areal radius of the inner sphere comes out as
   $4.0002\pm2\times10^{-15}$ (i.e. exactly 2), $\lambda_{\rm inner}=1.000000$.
3. **Residual scaling.** Multiplying each group by $\rho^p$ makes the far field
   dominate the loss by up to $10^8$; a fixed reference length, or the local
   $\rho^{1},\rho^{2}$ scaling now used by default, is far better conditioned.
4. **Remaining gap.** After 4000 steps (curriculum, BC weights 100, uniform radial
   sampling) the interior residuals are $\sim10^{-3}$ rms while the pointwise error is
   $\sim10^{-2}$: the error is concentrated in the *middle* of the shell
   ($h_{rr}$ bulges by $\sim3.7\,\%$ near $\rho\simeq3$–$5$, while both boundaries are
   matched to $10^{-5}$). This is the signature of weakly constrained long-wavelength
   modes — in particular the residual harmonic-diffeomorphism freedom of the harmonic
   gauge ($\delta h\sim\partial\xi$ but $\delta\Gamma\sim\partial^2\xi$) — and of the
   representation ceiling of the network (a *supervised* fit of the exact solution
   reaches only $\max|\Delta h|\simeq5\times10^{-3}$ for the symmetric net and
   $\simeq3\times10^{-2}$ for the 3-D net, so the PINN cannot beat those).

Longer runs with gradient-norm adaptive reweighting are in progress; see
`runs/m1_sym/report.json` and `runs/m1_3d/report.json`.

Final Milestone-1 numbers (20000 Adam + 1500 L-BFGS, symmetric net):

| quantity | symmetric net | 3-D net (96x5, 15000 steps) |
|---|---|---|
| inner BC residuals | $10^{-13}$ | $10^{-9}$ |
| areal radius of inner sphere | $4.000000\pm10^{-6}$ | $4.0051$ |
| $\max|\Delta h|$ / rms | $6.5\times10^{-3}$ / $1.8\times10^{-3}$ | $3.0\times10^{-2}$ / $8.5\times10^{-3}$ |
| $\max|\Delta\lambda|$ | $1.7\times10^{-3}$ | $1.7\times10^{-2}$ |
| residual rms (Ricci / compat) | $2\times10^{-4}$ / $7.5\times10^{-4}$ | $3\times10^{-4}$ / $8\times10^{-3}$ |

The 3-D net is $\sim5\times$ less accurate, matching its measured supervised
representation ceiling ($3\times10^{-2}$ vs $5\times10^{-3}$), i.e. the limit is the
network, not the physics.

### 8.3 Milestone 2 (Robin outer boundary) — three structural findings

Setting up the shell $2\le\rho\le20$ with your inner data ($\lambda=\lambda_0$, induced
metric round of areal radius 2) and the Robin outer condition
$n^i\partial_i\,\text{field}=-p\,(\text{field}-\text{field}_\infty)/\rho$ produced three
results that change the specification:

1. **The asymptotic value of $\lambda$ must be frozen.** With $\lambda_\infty$
   optimisable (or equal to $\lambda_0$), $h=\delta$, $\lambda=\text{const}$,
   $\lambda_\infty=\lambda$ satisfies *every* condition — the trivial branch. A probe
   run converged exactly there ($\lambda_\infty\to1.057$, residuals at the network
   floor). Freezing $\lambda_\infty\ne\lambda_0$ removes it. Use
   `--lam-inf 2.618033988749895` (the value the exact solution implies for
   $R_0=1,\lambda_0=1$).

2. **The decay exponent is not the same for every field.** Measured on the exact
   solution at $\rho=20$: $h-\delta$ decays as $\rho^{-2}$, $\Gamma$ as $\rho^{-3}$,
   $\lambda-\lambda_\infty$ as $\rho^{-1}$ (the ratio
   $|n^i\partial_i f|\,/\,(|f-f_\infty|/\rho)$ is $2.01$, $3.03$, $0.955$). A single
   exponent $p=1$ imposes the wrong asymptotics on $h$ and $\Gamma$ and creates a
   boundary layer at $\rho_{\rm out}$; `robin_exps = {h: 2, G: 3, lam: 1}` is the
   consistent choice (configurable via `--robin-exps`). The equations *force* these
   rates: $R_{ab}\sim\partial^2\delta h$ must match $(1/2\lambda^2)\lambda_a\lambda_b
   \sim\rho^{-4}$, so $\delta h\sim\rho^{-2}$ and $\Gamma\sim\rho^{-3}$.

3. **$h_{rr}=1$ on the inner sphere is an extra condition** (added by me, not in your
   specification) and it over-determines the symmetric sector: the harmonic chart has
   only two free parameters $F=c_1\rho+c_2F_2$, and "areal radius 2 at $\rho=2$" plus
   "$h_{rr}=1$ at $\rho=2$" fix both, leaving the asymptotic metric at $1.165\,\delta$
   instead of $\delta$.  It was a mistake and it is **gone**: no run constrains $h_{rr}$
   any more (the (default of `inner_h_rr` is now `None`, i.e. free), which makes the exact
   solution satisfy **your entire BC set** ($\lambda=\lambda_0$ and round inner metric at
   $\rho=2$, plus all three Robin conditions at $\rho=20$) to $10^{-10}$ or better ---
   which turns Milestone 2 into a validation problem with a known answer.  The old
   `--no-inner-h-rr` flag is no longer needed anywhere.

Reference chart (inner areal radius 2 at $\rho=2$, $h\to\delta$): $c_1=1$,
$c_2=1.552622$, $k=2.618034$; then $h_{rr}(\rho=2)=0.6487$ comes out of the solution
rather than being imposed. `stationary/invariants.py` checks a numerical solution
against the family using only geometric scalars ($r_a$, $\lambda$, Ricci scalar), which
is the right comparison because harmonic charts are not unique.

First Milestone-2 runs (symmetric net, 12000 Adam + 500 L-BFGS, $\lambda_\infty$ frozen
at 2.618034):

| | M2a (with $h_{rr}=1$) | M2b (spec-faithful, no $h_{rr}$) |
|---|---|---|
| final loss | $2.5\times10^{-4}$ | $3.5\times10^{-5}$ |
| residual rms Ricci / compat / gauge / $\lambda$ | 2.9e-4 / 1.1e-3 / 1.5e-4 / 4.4e-4 | 2.0e-4 / 1.1e-3 / 7.2e-4 / 1.7e-4 |
| areal radius of inner sphere | 3.999992 | 3.9999998 |
| $\lambda_{\rm inner}$ | 0.999998 | 0.999998 |
| Robin residuals at $\rho=20$ ($h$ / $\lambda$) | 7.9e-4 / 4.2e-6 | 1.6e-4 / 2.3e-6 |
| $\lambda(\rho=20)$ | 2.3859 | 2.3913 |

M2b is the better-posed variant (its exact reference satisfies every BC to $10^{-10}$),
and indeed it reaches a 7x smaller loss. Neither run has yet converged to its reference
solution: the metric still differs by $\sim10\%$ (max $|\Delta h|\simeq0.09$–0.12) while
the residuals sit at $10^{-4}$ — the same weakly-constrained long-wavelength behaviour
seen in Milestone 1, which needs longer training and/or an explicit treatment.




### 8.4 Where the remaining error actually comes from

Measured by *supervised* fitting of the exact solution (no PDE involved), i.e. the
representation ceiling of each ansatz:

| ansatz | max dh | max dlambda/lambda | notes |
|---|---|---|---|
| symmetric 64x3, 4000 steps | 4.8e-3 | 1.9e-3 | |
| symmetric 96x4 fourier 20, 8000 + 1500 L-BFGS | 2.1e-3 | 1.3e-3 | error concentrated in the innermost bin |
| 3-D 128x5 fourier 16, 12000 + 1200 L-BFGS | 3.0e-2 | 1.1e-2 | |
| symmetric + feature rho_in/rho | 6.0e-4 | 5.1e-4 | error no longer concentrated at rho_in |

Conclusions:

* The **3-D network, not the physics, is the accuracy limit**: its ceiling (3e-2) is
  essentially the error it achieves on Milestone 1 (3.0e-2).
* The symmetric net has a ceiling of ~1e-3 but the PINN delivers 6.5e-3 - a factor ~5
  gap produced by the optimiser (weakly constrained long-wavelength modes), not by
  capacity.
* Adding the physical decay variable rho_in/rho to the network features improves the
  ceiling by 1.6x and removes the inner-boundary concentration of the error (an artifact
  of building the feature map on log rho, which puts a domain edge exactly at rho_in).
  This is the cheapest available accuracy win and is not yet wired into model.py.

### 8.5 Robin boundary data, asymptotic lambda = 1, and a manufactured source

Following the requested setup (lambda -> 1 at long distances, larger outer radius, and
a Robin *source* so that the known solution satisfies the boundary condition exactly):

* `--lam-inf 1.0` freezes the Robin target at 1 (frozen is essential, see 8.3).
* `--rho-out 100` with `--radial log`: at rho=100 the metric deviation is ~1e-4 and
  lambda is within 2 % of its asymptotic value.
* `--robin-source`: the source is the Robin residual OF the reference solution,
  s_f = n^i d_i f_ref + p (f_ref - f_inf)/rho, so the reference satisfies the
  inhomogeneous condition identically. Verified: the Robin residuals of the reference
  are exactly 0.00e+00 for h, Gamma and lambda (1.7e-10 / 3.7e-13 / 5.2e-8 without it).
  `tests/test_pipeline.py::test_sym_hybrid_manufactured_robin_is_exact` locks this in.

**lambda_0 = 1 together with lambda -> 1 makes the problem trivial.** u = ln lambda
satisfies Delta_h u = 0, so with u = 0 on both boundaries the maximum principle gives
u == 0, hence lambda == 1, R_ab = 0 and (in 3D) a flat metric. A run with those data
(`runs/m2R1_trivial`) confirms it: loss 4.4e-8, max|h-delta| = 3.1e-3 (network floor),
max|lambda-1| = 1.2e-5. Nontrivial content therefore needs lambda_0 != 1; the value
consistent with lambda -> 1 is the exact-family one, lambda_0 = (sqrt5-1)/(sqrt5+1)
= 0.381966 at the areal-radius-2 sphere (equivalently k = 1).

**Diagnostic that settles "wrong" vs "not converged".** Evaluate the loss at the
reference itself (an `ExactModel`-style module returning the reference fields) and
compare with the loss at the trained state. For `runs/m2R2_asym1` (independent-Gamma
ansatz, 20000 Adam + 2000 L-BFGS, rho_out = 100):

    reference solution : loss = 5.0e-30      <- the global minimum
    trained PINN       : loss = 1.2e-5       (pde_compat = 1.1e-5 dominates)
    difference         : max|dh| = 0.33, but lambda and the areal radius agree to 1e-3

The disagreement is confined to h_rr, i.e. the radial (gauge) component, and the
dominant residual is compatibility: the independent-Gamma ansatz is spending its
accuracy on tying Gamma to d h. That motivates `arch="sym_hybrid"`: the network outputs
only (alpha, beta, u) and Gamma is the Christoffel symbol of h, so compatibility holds
identically (`pde_compat ~ 1e-18`) and there are three fewer outputs.

### 8.6 The independent-Gamma ansatz was the bottleneck (main numerical result)

Running the *same* Milestone-2 problem (manufactured Robin source, lambda -> 1,
rho_out = 100, 20000 Adam + 2000 L-BFGS) with the two spherically symmetric ansaetze
differs by three orders of magnitude. Evaluated against the exact reference:

| | `arch="sym"` (network outputs h, Gamma, lambda) | `arch="sym_hybrid"` (outputs alpha, beta, u; Gamma = Christoffel(h)) |
|---|---|---|
| `pde_compat` residual (rms) | 9.3e-4 (dominant term in the loss) | 1.7e-9 (round-off level only) |
| final loss | 1.2e-5 | 5.8e-8 |
| `max|dh|` / rms | 3.3e-1 / 9.2e-2 | 5.3e-5 / 2.3e-5 |
| `max|dGamma|` | 1.7e-1 | 2.3e-4 |
| `max|dlambda|` | 1.3e-2 | 2.9e-5 |
| chart-independent `max|dlambda(r_a)|` | 3.4e-3 | 2.2e-5 |
| inner boundary | areal radius^2 = 4.0000, lambda = 0.38197 | areal radius^2 = 4.0000, lambda = 0.38197 |

Milestone 2 is therefore validated: with the manufactured Robin source, lambda -> 1,
rho_out = 100 and lambda_0 = 0.381966 on the areal-radius-2 sphere, the PINN reproduces
the known exact solution to max|dh| = 5.3e-5 and max|dlambda| = 2.9e-5, with all four
residual groups at 1e-5 or below and both boundary conditions satisfied exactly
(`runs/m2R3_symhybrid/report.json`).

Diagnosis and cure: the independent-connection formulation spends its accuracy budget
tying Gamma to d h (compatibility), and the residual there has a nearly flat
long-wavelength direction, so the solution can slide along a coordinate (gauge) mode
without paying much loss. In the R2 run above, lambda as a function of the areal radius
(a chart-independent invariant) already matched the reference to 3.4e-3 while h_rr was
off by 33 %: the solution was the same geometry written in a different harmonic chart,
with both satisfying the boundary conditions. Deriving Gamma from h removes that entire
degenerate direction: compatibility holds exactly, the compat residual and three of the
six scalar outputs disappear, and the PINN then reproduces the known solution to ~1e-4.

The first-order (h, Gamma, lambda) formulation the project asked for remains available
as `arch="sym"` / `arch="3d"` (`stationary/model.py`), with the hybrid as
`arch="sym_hybrid"` / `arch="hybrid"`.

### 8.7 Milestone 2 solved with the real (non-manufactured) Robin data

`runs/m2R4_realrobin`: `arch="sym_hybrid"`, lambda_0 = 0.381966 on the areal-radius-2
sphere, Robin data with lambda_inf = 1 at rho_out = 100 (no manufactured source), 20000
Adam + 2000 L-BFGS, 7435 s.

Residuals (rms / max): compat 1.6e-9 / 3.0e-8 (round-off, Gamma is derived),
Ricci 4.4e-6 / 9.5e-5, gauge 1.8e-5 / 3.6e-4, lambda-equation 4.5e-5 / 4.6e-4;
boundary conditions: inner 1.4e-14 (lambda) and 1.3e-17 (round metric), outer Robin
1.8e-14 / 1.9e-14 / 4.9e-17 for h / Gamma / lambda.

Boundary values: inner sphere at rho=2 has areal radius 2.00000001, lambda =
0.38196613, h_theta_theta = 1.00000001 and h_rr = 0.648602 (not imposed; the exact
family predicts 0.64868). Outer sphere at rho=100 has areal radius 99.99540,
lambda = 0.98039070 (2 % short of the asymptote, as expected at this radius) and
h_rr = 1.000011.

Chart-independent content - lambda against the areal radius r_a:

| r_a | 2.00000 | 2.94894 | 5.94549 | 9.96043 | 19.97778 | 39.98839 | 99.99540 |
|---|---|---|---|---|---|---|---|
| lambda | 0.38196613 | 0.51387406 | 0.71556635 | 0.81849706 | 0.90494838 | 0.95140748 | 0.98039070 |
| R (Ricci) | 1.258e-1 | 2.644e-2 | 1.598e-3 | 2.061e-4 | 1.226e-5 | 5.8e-7 | ~noise |

Fitting the exact-family relation lambda = k (sqrt(r_a^2+R0^2)-R0)/(sqrt(r_a^2+R0^2)+R0)
gives R0 = 0.999491 and k = 1.000022 with a relative deviation of only 4.8e-4: the
numerical solution IS the exact family member with R0 = 1, lambda_infinity = 1. The
Ricci scalar matches 2 R0^2/r_a^4 at r_a = 2 (0.1258 vs 0.125) but is noise-dominated
further out, so lambda(r_a) is the usable invariant. Against the exact reference:
max|dh| = 8.5e-5, max|dGamma| = 1.9e-4, max|dlambda| = 1.9e-4.

### 8.8 Robin order: the weight, not the order, was the problem (measured)

The order-`n` Robin condition annihilates the first `n` decay powers, which lets higher
multipoles through.  Three runs identical except for the order (20000 Adam + 3000 L-BFGS,
`rho_out = 100`, `R0 = 1/sqrt3`, areal radius 1, `lambda -> 1`, `w_outer = 100`,
`runs/control_ord*`) looked at first like "order 1 wins by two orders of magnitude":

| run | order | precision | `lambda(100)` | error | outer BC rms (`h`, `lambda`) | `lam_eq` rms |
|---|---|---|---|---|---|---|
| `control_ord1b` | 1 | float32 | 0.9882606 | **3.0e-04** | 5.0e-05, 4.5e-05 | 6.4e-07 |
| `control_ord2` | 2 | float32 | 0.8263355 | 1.6e-01 | 1.5e-04, 1.8e-04 | 2.9e-05 |
| `control_ord4_x64` | 4 | float64 | 0.9303993 | 5.9e-02 | 3.5e-04, 7.7e-04 | 5.1e-06 |

That reading was wrong, and the order-2 loss trajectory shows why: its loss at the end is
`7.05e-04` of which `6.84e-04` is the `lam_eq` group, while every boundary term is at
`1e-08`-`1e-09`.  It did not fail on the boundary condition — it stopped solving the
equation, because the *stiff* outer term dominated the globally clipped gradient.  The
`order-n` Robin residual at initialisation is `(gain)^n` larger (`8.8e-04` for `n = 1`,
`2.1e-01` for `n = 2`), so the same `w_outer = 100` is a different weight at every order.

A fixed-budget sweep (6000 Adam + 500 L-BFGS, `n_coll = 512`, same seed) with the weight
matched:

| run | `w_outer` | final loss | `lam_eq` rms | `ricci` rms |
|---|---|---|---|---|
| order 1 | 100 | 3.43e-04 | 1.79e-02 | 4.25e-03 |
| order 2 | 100 | 1.17e-03 | 3.05e-02 | 5.62e-03 |
| order 2 | **10** | **2.29e-04** | **1.20e-02** | **2.33e-03** |
| order 2 | 1 | 4.50e-04 | 7.76e-03 | 7.89e-04 |

With a lower weight the order-2 run's *PDE* residuals improve 2.5x (`lam_eq` 3.05e-02 ->
1.20e-02) and its loss improves 5x -- but part of the loss improvement is just the outer
term being counted less, so the weight-independent numbers are the ones to trust.  The same
sweep at order 4 (x64, 3000 steps, `n_coll = 256`, `runs` reproduced locally) shows that
the weight is a **trade-off, not a bug with a correct setting**:

| `w_outer` | `lam_eq` rms | `ricci` rms | outer `lambda` rms | `lambda(100)` |
|---|---|---|---|---|
| 100 | 2.83e-05 | 2.30e-06 | 6.2e-02 | 0.661 |
| 10 | 1.76e-05 | 2.48e-06 | 1.2e-01 | 0.620 |
| 1 | 1.82e-05 | 1.77e-06 | 9.9e-02 | 0.586 |
| 0.1 | 1.49e-05 | 1.46e-06 | 2.2e-01 | 0.490 |

The equation gets better and the boundary condition and the far field get worse, all
monotonically: at 3000 steps nothing satisfies both.  (These are 1/7-budget probes, so the
absolute numbers are far from the ladder runs; only the trends are meaningful.)  The
apples-to-apples verdict comes from `runs/control_ord2_w10`, which is the order-2 run at the
*full* budget of `control_ord2` with `--w-outer 10` -- and the metric that decides it is the
**outer BC residual**, not the loss, because a smaller weight shrinks the loss by itself.

There is no simple scaling law for that weight — the initial outer residual is
`8.8e-04` (order 1), `2.1e-01` (order 2) and `4.4e+03` (order 4, from
`runs/control_ord4_x64`, where it is 99.99% of the loss at step 1), a factor `5e6` across
three orders — while the best weight at order 2 is `10`, not the `0.4` that matching the
initial magnitudes would suggest.  So **measure it**: a three-point sweep
(`--w-outer 100, 10, 1`) at 3000-6000 steps costs a few minutes and settles it.  The
diagnostic that tells you the weight is too large is in `report.txt`: the PDE residuals
stall while every boundary term sits at `1e-08`-`1e-09`.

Consequences:

* **Order 1 is still what the dipole runs use for now**, and not because higher order is
  worse: the dipole's `l = 1` tail at `rho = 100` is `~S1 (rho_in/rho_out)^2 = 1e-05`, so
  the order-1 condition biases it by about `1e-05`, thirty times below the accuracy the
  control reaches (`3e-04`).
* **Higher order will matter below `~1e-04`**: the exact solution does not satisfy the
  order-1 condition exactly, its residual being `6.6e-05` in `lambda` at `rho = 100` (and
  `7.4e-07` at order 2, `8.3e-10` at order 4).  So once capacity pushes the solution error
  below that, the order-1 condition becomes the limiting factor and the higher-order route
  -- with a matched weight, in x64 -- is the way to go.
* Round-off is a real but later constraint: order 4 is floored at `1.2e-05` in float32
  (versus `8.3e-10` in float64), which is why the float32 order-4 attempt parked at
  `lambda(100) = 0.406` while the x64 one reached `0.930`.  See HUB.md section 6.

### 8.9 The Weyl two-black-hole configuration (first run with no symmetry at all)

The manufactured problem of §7.1/§7.2, run for real: the inner data is the two-black-hole
field on a sphere that clears the rods, the gauge condition is the inhomogeneous cylindrical
one, and the outer condition is the order-2 Robin with its manufactured source.  The run is
`runs/weyl_prod`.

**Configuration.**  `axisym_hybrid 20 x 6`, fourier 0 (2265 parameters — five functions of
`(rho, mu)` rather than twenty-five of three); `rho` in [7.5, 75], shell ratio 10, with the
rods (half-length 1, half-gap 0.5) and therefore both horizons *inside* the inner sphere;
`n_coll = 4096`, `n_bnd = 256`; `w_inner = w_outer = 10`; `scale_exps` as elsewhere (which,
as measured below, makes the loss dimensionless and chart-independent); float64 — and it has
to be, for the reason in §7.2.

**Optimiser: SSBroyden from the random init, with no Adam phase at all** (`--steps 0`, blocks
of 100, `initial_scale` engaged on the first block).  The loss went 1.746e+00 → 2.059e-13 in
3000 iterations (2 h 4 min), monotonically, and was still improving by ~1.4x per block when
the iteration cap stopped it (status 1: never a stalled line search).  Selected blocks:

| iteration | 100 | 500 | 1000 | 2000 | 3000 |
|---|---|---|---|---|---|
| loss | 9.01e-05 | 4.15e-08 | 5.26e-10 | 4.66e-12 | 2.06e-13 |

Final groups: `compat` 3.7e-36 (structural — Γ is derived from `h`), `ricci` 7.6e-14,
`gauge` 5.3e-14, `lam_eq` 4.3e-14, `inner` 2.8e-16, `outer_h` 5.2e-16, `outer_lam` 2.2e-16.
So a cold-started quasi-Newton phase, given the `initial_scale` it was designed with, needs no
Adam warm-up — the warm-up was only fixing the large gradient it created itself.

**Accuracy against the exact Weyl solution.**

| where | `max abs(dh)` | `max abs(dGamma)` | `max abs(dlambda)` |
|---|---|---|---|
| over the shell, 4096 sampled interior points | 4.83e-07 | 8.58e-07 | 8.60e-08 |
| inner sphere `rho_in` (2048 points) | 8.00e-07 | — | — |
| outer sphere `rho_out` (2048 points) | 3.47e-07 | — | 9.63e-08 |

How these are measured, because the first row is easy to misread: it is a **volume max, not a
boundary value**.  The points come from `sample_shell`, i.e. log-uniform in `rho` across
[7.5, 75] with uniform directions, and the max is taken pointwise and then over the sample —
4096 points, fixed seed, none of them on either sphere.  It is a *sampled* max, not a
certified bound: the separate 2048-point inner-sphere probe is the worst number in the table
(8.0e-07), which is what a denser sample near the inner sphere does.  The error is not
concentrated anywhere in particular: 4.3e-07 for `rho <= 1.2 rho_in`, 4.8e-07 in the middle,
3.5e-07 beyond `3 rho_in`; `max abs(dlambda)` = 8.6e-08 is attained at `rho = 71.6`.

**The gauge condition is satisfied, and that needs saying explicitly.**  The diagnostics
report the residual of the condition the run *imposed* — the inhomogeneous cylindrical source
— not `Gamma = 0`.  Measured independently at 512 shell points: imposed condition max 3.34e-07
(rms 2.29e-08), `ricci` max 6.24e-08, `lam_eq` max 2.66e-08, and `lambda` averaged 0.819392
against the reference's 0.819392 — six decimals.  The same probe on the exact solution with
its own source gives 1.3e-15, so 3.3e-07 is the network's error and not a floor of the check.
For scale: the *harmonic* residual of this solution is 8.6e-03 at `rho_in`, a property of the
chart and precisely why the source is imposed at all; it is kept in `report.json` as
`res_gauge_harmonic_*`.

**The radial structure is the right one.**  The outer multipole diagnostic fits `l = 0`:
power −0.926 (expected −1), `l = 1`: −2.05 (−2), `l = 2`: −2.85 (−3), with `l = 3` down in the
1e-9 roundoff floor — exactly the Weyl asymptotics the Robin conditions assume in §7.1.

**Diagnostics that deliberately do not apply here**, so that a reader of the figures is not
misled: no areal radius is imposed (the inner data is the reference's own, and the measured
areal radius there is 7.4115, r^2 = 54.93, against the round-sphere 2 that a spherical-data run
would impose); the family read-off of `R0` and `k` in the chart-independent section assumes
spherical symmetry and reports 1.99/0.9997 against the data's 1/1 — indicative only; and the
inner-sphere figure plots `|lambda_net - lambda_ref| <= 1.1e-07`, not the distance to the
round-sphere polynomial, which would be 0.21 of genuine Weyl-vs-round difference wearing the
network's name.

**Does the chart matter?**  Measured, because it was worth knowing before spending the compute:
the manufactured loss on a covariantly perturbed Weyl candidate is 1.720418e-02 in
[7.5, 75], [0.1, 1] and [0.01, 0.1] alike — equal to every printed digit and in every group.
`scale_exps` makes each residual dimensionless (`compat`, `gauge` ×rho; `ricci`, `lam_eq`
×rho²), the features are `t = log(rho/rho_in)/log(ratio)` and `mu = n_z`, `decay_feature` is
`rho_in/rho`, and `theta = rho d_rho` is scale-invariant, so at a fixed shell RATIO the chart
cancels out of the whole problem.  A [0.01, 1] recipe therefore does not change the
*conditioning*; what it changes is the ratio (100 instead of 10), i.e. how far outside the
source the outer sphere sits.

**Ratio 10 against ratio 100.**  `runs/weyl_rescaled` is the same configuration with the rods
scaled by 1/750, so that `rho_in = 3*axis_extent = 0.01` and `rho_out = 1` — the shell
convention the other runs in this project use.  Both are 3000 cold-start SSBroyden iterations
of the same 2265-parameter ansatz:

| | ratio 10, `[7.5, 75]` | ratio 100, `[0.01, 1]` |
|---|---|---|
| final loss (wall time) | 2.06e-13 (7424 s) | 2.75e-13 (7822 s) |
| `lambda` mean over the shell | 0.5867266 | 0.5867271 |
| areal / coordinate radius at `rho_in` | 7.4115 / 7.5 | 0.00988 / 0.01 |
| `max abs(dh)` over the shell | 6.70e-07 | 1.17e-06 |
| `max abs(dlambda)` | 9.04e-08 | 1.57e-07 |
| inner sphere, `max abs(lambda_net - lambda_ref)` | 1.12e-07 | 3.14e-08 |
| `l = 0` decay fit (expected −1) | −0.926 | −0.998 |

They agree where they must: `lambda` averaged over the shell comes out 0.5867266 against
0.5867271, and the inner sphere's areal radius is 0.988 of its coordinate radius in both —
the same geometry, exactly as the chart-invariance above requires.  What the wider shell buys
is the far field: the `l = 0` decay exponent is −0.998 instead of −0.926, and the inner-sphere
`lambda` error is three times smaller.  What it costs is a somewhat larger shell-wide max,
because the same network budget now has to cover two decades of radius instead of one.  One
number in the diagnostics is **not** comparable between the two columns: `max abs(dGamma)` is
5.1e-04 at ratio 100 against 6.5e-07 at ratio 10, since Gamma has dimension 1/length and the
chart rescales lengths by 75.

**Files.**  `runs/weyl_prod/`: `report.txt`, `report.json`, `report_eval.json`,
`diagnostics.png`, `lambda_inner.png`, `lambda_vs_rho.png`, `lambda_multipoles_outer.png`,
`lambda_multipole_decay.png`, and `vtk/solution.vtk`.

The VTK file is written by

    python -m stationary.vtk --outdir runs/weyl_prod --physical-inner 7.5

and it is the file the error is read from: 81 426 points, 81 920 cells, 17.3 MB, ten point
fields led by **`lambda_err` = lambda_net − lambda_exact**, which over the grid has max
`1.14e-07` and rms `1.53e-08`.  Then come `h_err`, `lambda_exact`, `lambda`,
`lambda_minus_1`, `r_areal`, `ricci_scalar`, `ricci_sq`, `res_ricci`, `res_lam_eq`.
`--physical-inner 7.5` keeps the file in **Weyl units** (scale factor 1), so the rods sit on
the axis at `|z|` in [0.5, 2.5], exactly as the analytic solution has them.

The mesh (`--grid spherical`, the default) is conforming: nodes sit on radial levels
`rho_in * (rho_out/rho_in)^(i/n_rho)` — geometric, so both spheres are hit exactly and the
cells grow by a constant factor outwards — times uniform angles `theta` and periodic `phi`.
Every cell is inside the shell by construction, so there is nothing to blank out, and the
inner sphere is resolved the same amount in **every** direction.  The graded Cartesian box
(`--grid cartesian`, the older default) could not do that: grading each half axis clusters
points near `+-rho_in` on the three *axes* only, leaving the rest of the inner sphere
comparatively bare, where the field varies fastest; and roughly half its cells had to be
discarded because their centres fell outside the shell.  The two polar rings are written as
wedges (VTK 13) rather than hexahedra with two coincident nodes, so no cell is degenerate.

Every field in it is finite, and getting there fixed two real defects in the reference rather
than papering over them.  On the axis both exact expressions were 0/0: the `k` quadrature
divides by `rho_cyl`, and the Cartesian `h` formula `(A x^2 + y^2)/rho_cyl^2` was falling back
to zero, when its limit is `A` — the angular part collapses, `d(rho)^2 + rho^2 d(phi)^2 =
dx^2 + dy^2`, so `h = A * delta` there, and `k = 0` beyond the outermost rod ends makes it
exactly `delta`.  Both now return the limit.  Note that nothing else in the pipeline ever
looks at the axis — training samples `rho >= rho_in`, the Robin conditions sit at `rho_out` —
so an O(1) error there was invisible until a field-by-field export of `h(computed) −
h(exact)` was asked for.  Off the axis every value is unchanged, so no trained number in this
document moves.

The failure mode this exposed is worth recording, because it is silent: **one non-finite
number in a legacy ASCII VTK file stops VisIt from reading every variable that follows it**,
so a ten-field file comes back as a two-field one.  `stationary.vtk` now refuses to write such
a value (it substitutes 0 and reports it) — a guard, not a blanking mechanism.  Current
ranges in `runs/weyl_prod/vtk/solution.vtk`: `lambda_err` in [−1.14e-07, +4.52e-08] (max
1.14e-07, rms 1.53e-08) and `h_err` in [7.2e-09, 8.6e-07].

## 9. Running on a JupyterHub / GPU machine

The hub workflow has its own document: **`HUB.md`** (setup, `run_hub.sh`,
checkpointing/resume, monitoring, gotchas). `run_hub.sh` is the single entry point
there -- use it rather than launching `python -m stationary.train` by hand, because it
detaches the job (`setsid`+`nohup`), keeps outputs on `$HOME`, checkpoints every 500
steps and writes a `resume.sh`. A run started from a notebook cell or a plain terminal
dies with the server; `./run_hub.sh ...` does not.

Two things about *this* project that matter on the hub:

* **Training is float32.** `stationary/train.py` does not enable x64, so every number
  under `runs/` was produced in float32 and the smallest residual it can represent is
  ~1e-7 rms. Setting `JAX_ENABLE_X64=1` changes the optimisation trajectory and makes
  the existing results irreproducible (and costs ~2x throughput on GPU); `run_hub.sh`
  warns about it. The verification suite (`tests/`, `verify_exact_solution.py`) and the
  post-processing modules (`evaluate.py`, `invariants.py`, `multipoles.py`) *do* enable
  x64 -- that is deliberate: train in float32, verify and analyse in float64.
* **Boundary-condition and residual numbers quoted in this README are mean squares**,
  as they come out of the loss. Divide by the number of components and take a square
  root for the rms: a mean square of 1e-14 is an rms of 1e-7, i.e. the float32 floor,
  *not* a statement that the boundary data are satisfied to 1e-14.

The two production runs of this project (the spherical validation control and the
S1 = 0.1 dipole) are written out with their full flag lists in `HUB.md` §7.

## 10. Invalidated results (removed)

Two completed runs, `n1_control` and `n3_order4`, were **void and have been deleted** from
`runs/`. They were produced with the gradient-norm reweighting applied to *all* loss
groups, including the boundary terms. Since that rule sets `w ~ 1/||d term/d theta||`, the
**most violated constraint receives the smallest weight**: in that control run the outer
Robin weight decayed 17.7 -> 12.5 -> 8.84 -> 6.25 over the last rewrites while the inner
weight grew 81 -> 229. The only term enforcing `lambda -> 1` was therefore silenced,
`lambda` stayed at its inner value 1/3 all the way out (mean lambda at rho = 100: 0.3365
instead of the required 0.9885) and the solution drifted onto the trivial branch
(lambda ~ const, nearly flat metric, max|dh| = 0.42).

The reweighting now touches the four interior groups only; the boundary weights stay at
`w_inner`/`w_outer`. Verified on a 400-step control run: the outer Robin mean square falls
1.56e+03 -> 4.57e-02 -> 2.36e-04 while its weight is held at 100, and the 23 tests pass.

Runs affected: anything started before this fix **with `--reweight-every > 0`**.
`m1_sym`, `m1_3d`, `m2R3_symhybrid` and `m2R4_realrobin` used the same old rule, but there
the BC-first ramp satisfied the boundary data *before* the first rewrite, so their BC
weights grew rather than decayed and their results stand. Everything else that was
exploratory (scaler sweeps, schedule comparisons, regression probes, smoke tests) has also
been deleted; `runs/` now holds only the seven runs the documentation refers to.

### 10.1 Status of the dipole run

`runs/n2_dipole` (S1 = 0.1, independent-Gamma ansatz, order-2 Robin) is kept because the
chart-independent analysis in §8.6 uses it, but it is **not converged**: the interior
residuals sit at ~1e-2 and lambda reaches only 0.66 at rho = 100 instead of ~1. It has to
be re-run with the corrected reweighting (and, for the quadrupole, with `--robin-orders
h=4,lam=4`), which is what the hub runs in `HUB.md` §7 are for. Its residual diagnostics
are quoted in §8.6 and its timing in `HUB.md` §6; its figures should not be read as
physics.
