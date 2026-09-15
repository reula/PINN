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

(filled in as runs complete — see `runs/`.)
