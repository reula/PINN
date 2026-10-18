# shellcheck shell=bash
# The Stationary problem every optimiser comparison runs on (the pq shell at production size),
# sourced by run_batch_sketch.sh (hub) and mendieta_train.sbatch (Mendieta), so the two cannot
# drift apart.  n_coll 16384 / n_bnd 1024 with four outer shells is what fits the hub's 13.5 GiB
# share (DSGNAR.md section 9.2); the inner condition is the induced metric (0773b6a).
S_RECIPE=(--arch axisym_hybrid
          --n-coll 16384 --n-bnd 1024 --n-bnd-outer 1024 --ckpt-every 500
          --R0 0.5773502691896258 --rho-in 1.0 --rho-out 100.0 --inner-radius 1.0
          --vtk-physical-inner 1.0
          --lam0 0.33333333333333337 --lam-inf 1.0 --lam-bc-S2 -0.08333333333333333
          --outer-bc robin --robin-orders h=3,lam=3 --no-robin-G
          --eq-weights compat=0,ricci=1,gauge=1,lam_eq=1
          --pin-lam-robin --pin-h-robin --ref-solution --ref-asymptotic 1.0
          --outer-shell-fracs 1.0,0.9,0.8,0.7
          --resample-every 500 --decay-feature --log-resources --no-figures)

# The settings that made the difference (2026-10-18, REPORT_PIN_COM.md): residuals weighted by
# rho^4, the outer lambda pin on the Geroch-Hansen potential P, and the centre-of-mass gauge on six
# spheres.  Appended after S_RECIPE (later flags win).  The workflow that uses them is README
# "Recommended workflow": Adam -> DSGNAR restart cycles -> SSBroyden (Wolfe) with resampling ->
# scripts/fresh_sample_loss.py.
S_BEST=(--scale-exps 3,4,3,4 --pin-lam-potential --com-pin 1,1.5,3,10,30,100)
