# 1D reference for the single-puncture baseline

Validation-only reference solution for a single non-boosted puncture. Like the exact
Kirchhoff solution in the wave project, it is never used for training, point
selection or tuning. Written 2026-10-10.

## What it is

`sph_reference.py` evolves one Brill-Lindquist puncture (bare mass m, at the origin,
alpha = 1/psi, beta = 0) in spherical symmetry, with **the BH code's own
`r_ccz4_evolution`** as the right-hand side. Same CCZ4 equations, same 1+log and
integrated Gamma-driver gauge, same `--f_alpha/--f_beta/--eta/--kappa_*` and the same
`--chi_floor` (1e-5) as the PINN. It is a method-of-lines finite-difference code:

- 10 radial functions fix every field: gt = a δ + b nn, At = c δ + d nn, Ghat = g n,
  beta = B n, plus chi, Khat, Theta and alpha.
- At each grid point, a local quadratic Taylor model of those functions is passed to
  `r_ccz4_evolution` as a stand-in network on the x-axis. RHS = −residual.
- Staggered grid r_k = (k + ½) dr with parity ghost zones, 4th-order centred
  differences, 6th-difference Kreiss-Oliger dissipation (σ = 0.1), RK4 with dt = dr/4.
- The outer boundary is at R_out = 60 and causally disconnected from r ≤ 24 for
  t ≤ 10. So the reference is the free-space solution. It does **not** include the
  PINN's Sommerfeld condition at R_domain = 24.

The equations are not re-derived, so the reference inherits the verification of
`r_ccz4_evolution` (`tests/test_random.py`: ~1e-16 against an independent CCZ4).
A shared bug in the RHS would therefore not be caught by this comparison.

## Checks

| check | result |
|---|---|
| `test_sph_reduction.py`: 1D reduction vs direct 3D evaluation, generic spherical fields with Z, Θ, β ≠ 0, off-axis points | max difference 3e-16: **PASS** |
| self-convergence, dr = 0.02/0.01/0.005, t ≤ 10 (`conv_check.py`) | Q ≈ 16 (4th order) for r ≥ 0.1. Estimated error of the dr = 0.005 run: ≲ 1e-5 there, most fields ≲ 1e-6 |
| near the puncture, 0.03 ≤ r < 0.1 | ~1st–2nd order, error up to ~1e-3 (Khat, At around t = 2) |
| alpha at r = 0.03 (where `puncture_track.txt` samples it) | dr = 0.01 and 0.005 agree to ≤ 0.004 at all times |
| BH code, smoke test with `--n_stars 1` (tiny network, 2 windows) | runs end to end, no errors; `compare_pinn.py` reproduces the code's own `alpha_0` in `puncture_track.txt` to 4 digits |

Two numerical notes:

- With a tiny chi floor (1e-12) the run fails by t ≈ 0.05: the dissipation drives
  chi (~1e-7 at the innermost point) negative. The code's 1e-5 floor avoids this.
- dr = 0.04 is too coarse (fails at t ≈ 2.5). So is running without dissipation
  (fails at t ≈ 4.5).

## What the reference already says (m = 0.5, see `ref_single_m0.5_dr0.005_alpha.png`)

- **The early rise of alpha near the puncture is physical in this gauge.** At
  r = 0.03, alpha goes 0.107 (t = 0) → 0.28 (t ≈ 2) → 0.017 (t = 5), then creeps up
  slowly (0.034 at t = 10). The old BBH runs showed the same rise (0.14 → ~0.45 by
  t ≈ 1.5) and the same collapse, so that part was not a PINN artefact. Only the
  amplitude can now be checked.
- **The slow late rise of alpha near the puncture is also physical** (trumpet
  settling). So "alpha creeps up" alone is not proof of erasure. Compare with the
  reference before calling erasure.
- Diagnostic inconsistency found in the BH code: the t = 0 analytic row of
  `puncture_track.txt` samples alpha at (x_p, 0.03, 0.03), i.e. r = 0.042
  (alpha = 0.145 for m = 0.5). Every later row samples it at (x_p, y_p, 0.03),
  i.e. r = 0.03 (alpha(t = 0) = 0.107). Not fixed. Keep it in mind when reading the
  first row.

## Files

| file | content |
|---|---|
| `sph_reference.py` | the reference evolution (`--help`) |
| `test_sph_reduction.py` | check of the 1D reduction (run after any change) |
| `conv_check.py` | three-level self-convergence of reference runs |
| `compare_pinn.py` | PINN run vs reference: errors per field and time, alpha near the puncture, ray-to-ray anisotropy, handoff jump at each window start, 3 PNGs |
| `plot_reference.py` | overview figure of a reference run |
| `ref_single_m0.5_dr0.005.npz` | **the reference**: m = 0.5, dr = 0.005, t = 0…10 every 0.25 (the window boundaries), r ≤ 30, 17 MB |
| `ref_single_m0.5_dr0.005_alpha.png` | its overview figure |

Regenerate the reference (CPU, about 30 minutes):

    cd reference_1d
    JAX_PLATFORMS=cpu python3 sph_reference.py --dr 0.005 --tfinal 10 --out ref_single_m0.5_dr0.005.npz

For other gauge or damping flags, pass the same ones to `sph_reference.py`
(`--f_alpha --f_beta --eta --kappa_z --kappa_c --kappa_2 --chi_floor`) and to the PINN
run. For a longer run, raise `--R_out` to at least 24 + 1.5 tfinal.

## Using it with a PINN run

Submit `single_puncture.psubmit` from the BH folder on MN5 (`sbatch single_puncture.psubmit`). Its outputs go to `single_m05/` (`--outdir`). Then:

    JAX_PLATFORMS=cpu python3 reference_1d/compare_pinn.py single_m05 reference_1d/ref_single_m0.5_dr0.005.npz

This writes `<run dir>/compare_ref_*.csv/png`. The comparison is in the same
coordinates and gauge, so fields are compared directly. Read it as:

- `compare_ref_alpha_puncture.png`: the erasure test. It shows alpha at r = 0.03,
  PINN vs reference.
- `compare_ref_errors.csv`, rows with `side = end`: accuracy per field at each window
  end. Rows with `side = start` belong to the next window. The difference between the
  two is the error of the soft-IC handoff.
- `compare_ref_summary.txt`: a scoreboard at t = 0, 0.25, 1, 2, 3, 5, 7.5 and 10, with banded L2 errors (core [0.05, 0.5), mid [0.5, 5), far [5, 20)), Theta, anisotropy and puncture drift. Put it next to `single_m05/analysis/compare_ref_summary.txt`, the baseline run.
- `ray_spread` and `off_form`: how far the PINN is from spherical symmetry. The
  reference has exactly zero for both, so any nonzero value is PINN error.
- Keep `--r_max` well inside R_domain (default 20). Near r = 24 the PINN's Sommerfeld
  boundary adds its own difference.
