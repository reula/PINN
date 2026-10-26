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
| weyl_rot45 -> _long | loss reduction "factor 13" (README) | 1.753e-4 / 2.282e-5 = 7.7 | ✎ (factor 8) |

## Schwarzschild tests

The original runs (`m2R2_asym1` first order, `m2R3_symhybrid`, `m2R4_realrobin`) were deleted from
`runs/` in commit 92cb622. They are recovered from 92cb622~1 into `runs/_restored/`.

| quantity | paper | restored report.json | |
|---|---|---|---|
| first order: final loss | 1.2e-5 | 6.31e-6 | ✎ (the 1.2e-5 has no source) |
| first order: max dh / dGamma / dlam | 3.3e-1 / 1.7e-1 / 1.3e-2 | 0.328 / 0.173 / 1.28e-2 | ✓ |
| metric only: loss, max dh / dGamma / dlam | 5.8e-8, 5.3e-5 / 2.3e-4 / 2.9e-5 | 5.75e-8, 5.30e-5 / 2.34e-4 / 2.92e-5 | ✓ |
| real Robin data: max dh / dGamma / dlam | 8.5e-5 / 1.9e-4 / 1.9e-4 | 8.50e-5 / 1.88e-4 / 1.93e-4 | ✓ |

These were float32 runs of older code. All three were rerun with the current code and without the
averaged pins (`Paper/commands_schwarzschild.sh`, jobs 1182114-1182116, 9-11 min each). The paper
quotes the reruns:

| rerun | loss | max dh | max dGamma | max dlam | M0 (exact 1) |
|---|---|---|---|---|---|
| schw_first_order | 1.50e-9 | 1.71e-2 | 8.61e-3 | 1.27e-3 | |
| schw_metric_manuf | 2.40e-11 | 6.51e-6 | 7.86e-6 | 1.01e-6 | 0.999991 |
| schw_metric_real | 2.42e-11 | 7.00e-6 | 5.97e-6 | 6.32e-7 | 0.999997 |

M2 of the Schwarzschild runs is 1.5e-4 +- 2.7e-4 (consistent with 0; with the inner sphere at 2M
the axis series converges slowly, so the window spread is large).

## Rotated Weyl with the metric-only 3-D network (`Paper/commands_weyl.sh`, job 1182117)

| | loss | max dh | max dlam | iterations / time | M0 / M2 / M3 / centre (exact 0.0314286 / 6.785e-6 / -3.33e-7 / 0.0374026) |
|---|---|---|---|---|---|
| weyl_rot45_hyb | 5.13e-9 | 1.64e-4 | 1.73e-5 | 7000 (plateau) / 16.6 min | 0.0314289 / 6.792e-6 / -3.26e-7 / 0.03740 |
| (old first-order weyl_rot45_hub) | 6.93e-8 | 1.13e-3 | 1.04e-4 | 12165 / 141 min | 0.031446 / 5.3e-6 / -2.5e-7 |

## The spectral reference (`Paper/commands_specref.sh`, job 1182235)

The rerun of the reconstructed command gives loss 2.793e-11 after Newton step 3, identical to the
original (job 1180550). ✓

## Quadrupole (Mendieta; `Paper/recheck_mendieta.sh`, report_metrics and grade_rotated rerun)

Recomputed with jobs 1182087-1182094 (`plots/recheck_report_metrics.json` and the grade logs). The
earlier tables reproduce. For example, m5_ssb_norsmp has lambda error 1.558e-3 at rho 1-3 (table
1.6e-3) and d_z = -0.02567 on all four spheres; m7_dsg_com has 6.67e-8 (6.7e-8); m8_ssb_com has
lambda 3.18e-8 and h 1.43e-7 (3.2e-8 and 1.4e-7). The rotated grades reproduce to the quoted digits:
rot_ssbm 1.56e-5 / 1.16e-3, rot_from_fit 6.49e-6 / 3.23e-4, ctl_hyb_ssbm2 1.17e-5 / 5.52e-4,
rot_coll256k_2g 1.43e-5 / 7.47e-4. These runs are no longer quoted in the paper, which uses the
from-scratch round P instead.

Wolfe line search on the rotated case (round 28): rot_ssbw (from rot_ssb) reaches loss 1.73e-6,
rms dlam 5.6e-6 and dh 8.5e-4, against Armijo rot_ssbm's 2.25e-6, 1.56e-5 and 1.16e-3.

## The paper's from-scratch runs (round P of `mendieta_submit.sh`)

`qp_com`, `qp_nocom` and `qp_rot`, each Adam + SSBroyden stages with a final check. Also the
rotated Weyl test with the metric-only network, `weyl_rot45_hyb` (`Paper/commands_weyl.sh`).
Pending.
