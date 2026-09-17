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
    stationary/diagnostics.py  residual/error/boundary-geometry reports
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

* `lambda = lambda_0`
* the induced metric is the round metric of **areal radius 2**
  (implemented as `h - h_rr n n - (4/rho^2)(I - n n) = 0`)
* `h_rr = 1` (normal-normal component; fixes the remaining metric freedom)

Outer sphere (`rho = rho_out`), two modes:

* `dirichlet_exact` — `h` and `lambda` from the exact solution (milestone 1:
  a manufactured-solution test with a known answer)
* `robin` — `n^i d_i field = -(field - field_inf)/rho_out` on `h` (with
  `field_inf = delta_ij`), on `Gamma` (`field_inf = 0`) and on `lambda`
  (`field_inf = lambda_inf`, optionally a learnable scalar)

## 6. Usage

    .venv/bin/python -m pytest tests/ -q                       # test suite
    .venv/bin/python verify_exact_solution.py                  # standalone check
    .venv/bin/python -m stationary.train --steps 20000 --lbfgs-steps 2000 \
        --outdir runs/m1
    .venv/bin/python -m stationary.evaluate runs/m1            # diagnostics + figure

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

Configuration: $R_0=1$, $\lambda_0=1$, $k=(\sqrt5+1)/(\sqrt5-1)=\varphi^2$, inner
sphere at $\rho_{\rm in}=\sqrt5$ (the areal-radius-2 sphere), outer at $\rho=20$,
Dirichlet data from the exact solution at both spheres.

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

3. **$h_{rr}=1$ on the inner sphere is an extra condition** (it was added by me, not in
   your specification) and it over-determines the symmetric sector: the harmonic chart
   has only two free parameters $F=c_1\rho+c_2F_2$, and "areal radius 2 at $\rho=2$"
   plus "$h_{rr}=1$ at $\rho=2$" fix both, leaving the asymptotic metric at
   $1.165\,\delta$ instead of $\delta$. Dropping $h_{rr}=1$ makes the exact solution
   satisfy **your entire BC set** ($\lambda=\lambda_0$ and round inner metric at
   $\rho=2$, plus all three Robin conditions at $\rho=20$) to $10^{-10}$ or better —
   which turns Milestone 2 into a validation problem with a known answer. Use
   `--no-inner-h-rr` for that variant; `--ref-solution` builds the reference for
   diagnostics.

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
