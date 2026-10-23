# Verification of the paper's numbers (started 2026-10-23)

Every number in `paper.tex` is recomputed here from the data the runs left, or by rerunning, or
by recomputing from the exact solutions. Each entry gives the paper's value, the recomputed value
and how it was obtained. ✓ means the two agree to the quoted digits; ✎ means the paper is
corrected to the recomputed value; ✗ means the number is removed from the paper.

## Exact solutions and analytic checks (Mac; `Paper/recheck_local.py`, `verify_weyl.py`, tests)

| claim | paper (draft) | recomputed | |
|---|---|---|---|
| exact Schwarzschild through the full loss | < 1e-16 | < 1e-16 (`test_loss_is_zero_on_exact_solution`) | ✓ |
| same, Robin-source chart with the averaged pins | — | 4.3e-10 from the tangential pin; zero without the pins (`test_sym_hybrid_manufactured_robin_is_exact`, updated) | new |
| exact Weyl rods, full loss, cylindrical source | 1.3e-28 | 1.26e-28 without the pins (verify_weyl, 17/17); 1.6e-9 with the h_rr pin | ✓ (pins off) |
| Weyl chart's harmonic residual at rho = 7.5 | 3.9e-3 | max 9.2e-3, rms 7.1e-3 over the sphere | ✎ |
| rotated exact Weyl: compat / Ricci / lam_eq / gauge | 1.9e-16 / 1.6e-15 / 7.1e-17 / 3.4e-16 | max over 256 points in [4, 40]: 8.9e-16 / 2.6e-14 / 5.9e-17 / 1.9e-15 | ✎ |
| rotated exact Weyl in harmonic gauge | 3.0e-2 | 2.6e-2 | ✎ |
| Robin decay ratios h / Gamma / lam at rho = 20 | 2.01 / 3.03 / 0.955 | 2.00 / 3.005 / 0.952 (local log-derivative) | ✎ |
| order-1 lambda pin residual on the exact solution at rho = 100 | -6.6e-5 | -6.590e-5 | ✓ |
| its cost at w_pin = 100 | 4.3e-7 | 4.34e-7 | ✓ |
| the same pin on P | 1.5e-11 | 1.48e-11 | ✓ |
| parameter counts (axisym / + decay / 3-D hybrid / mlp) | 2265 / 2285 / 2347 / 2725 | identical | ✓ |
| R_ab R^ab of the exact rods (1, 0.1) at rho_in = 7.5 / 4 / 3.5 | 3.3e-6 / 5.8e-3 / 5.3e-2 | 3.28e-6 / 5.79e-3 / 5.28e-2 (`weyl.curvature_invariant`, max over the sphere) | ✓ |
| GH moments of exact assets (Schwarzschild, rods, Curzon) | — | `tests/test_gh_moments.py` | new |

## Weyl runs re-evaluated from their saved networks (`stationary.evaluate` on copies)

| run | quantity | paper | recomputed | |
|---|---|---|---|---|
| weyl_prod | final loss | 2.06e-13 | 2.059e-13 | ✓ |
| | max dh / max dlam | 6.7e-7 / 9.0e-8 | 6.70e-7 / 9.04e-8 | ✓ |
| | decay powers l = 0, 1, 2 | -0.926, -2.05, -2.85 | report.txt fit: -0.924, -2.97 (l = 1 is at the 1e-12 noise), -2.84 | ✗ (replaced by the GH moments) |
| weyl_unequal | loss / max dh | 2.4e-14 / 3.5e-7 | 2.37e-14 / 3.41e-7 | ✎ dh |
| | a_l0 coefficients, dipole 1.398415 | (from README) | not recomputed | ✗ (replaced by the GH moments) |
| weyl_close4 | loss / max dh | 7.1e-13 / 1.6e-5 | 7.13e-13 / 1.54e-5 | ✎ dh |
| weyl_close35 | loss / max dh / max dlam | 3.1e-12 / 7.7e-5 / 1.8e-6 | 3.06e-12 / 7.29e-5 / 1.76e-6 | ✎ dh |
| weyl_rot45 | loss / max dh / max dlam | 1.75e-4 / 7.1e-2 / 5.2e-3 | 1.753e-4 / 7.12e-2 / 5.19e-3 | ✓ |
| weyl_rot45_long | loss / max dh / max dlam | 2.28e-5 / 3.6e-1 / 6.4e-2 | 2.282e-5 / 3.57e-1 / 6.44e-2 | ✓ |
| weyl_rot45_hub | loss / max dh / max dlam | 6.93e-8 / 1.1e-3 / 1.0e-4 | 6.928e-8 / 1.13e-3 / 1.04e-4 | ✓ |
| chart-invariance loss | 1.720418e-2 on three shells | not recomputed | ✗ |

## Schwarzschild tests

The original runs (`m2R2_asym1` first order, `m2R3_symhybrid`, `m2R4_realrobin`) were deleted from
`runs/` in commit 92cb622. They are recovered from 92cb622~1 into `runs/_restored/`.

| quantity | paper | restored report.json | |
|---|---|---|---|
| first order: final loss | 1.2e-5 | 6.31e-6 | ✎ (the 1.2e-5 has no source) |
| first order: max dh / dGamma / dlam | 3.3e-1 / 1.7e-1 / 1.3e-2 | 0.328 / 0.173 / 1.28e-2 | ✓ |
| metric only: loss, max dh / dGamma / dlam | 5.8e-8, 5.3e-5 / 2.3e-4 / 2.9e-5 | 5.75e-8, 5.30e-5 / 2.34e-4 / 2.92e-5 | ✓ |
| real Robin data: max dh / dGamma / dlam | 8.5e-5 / 1.9e-4 / 1.9e-4 | 8.50e-5 / 1.88e-4 / 1.93e-4 | ✓ |

These were float32 runs of older code. All three are rerun with the current code and without the
averaged pins (`Paper/commands_schwarzschild.sh`, Mendieta jobs 1182114-1182116); the paper will
quote the reruns. Pending.

## Quadrupole (Mendieta; `Paper/recheck_mendieta.sh`, report_metrics and grade_rotated rerun)

Pending: jobs 1182087-1182094.

## The paper's from-scratch runs (round P of `mendieta_submit.sh`)

`qp_com`, `qp_nocom` and `qp_rot`, each Adam + SSBroyden stages with a final check. Also the
rotated Weyl test with the metric-only network, `weyl_rot45_hyb` (`Paper/commands_weyl.sh`).
Pending.
