#!/usr/bin/env bash
# Two small CPU runs of the "full situation" at a size that fits a laptop.
#
#   phi = ln(lambda)                      --lam-eq-form log
#   metric pins ON, all multipoles        --eq-weights without inner_h=0,outer_h=0 (so weight 10)
#   the CORRECT Ricci source              --ricci-lam-source 1   (Ricci_ab = d_a lam d_b lam / 2 lam^2)
#   soft inner conditions                 (no --hard-inner-bc)
#   SSBroyden                             (default --qn-method)
#
#   L1: all conditions                 L2: + the dipole exempted from every condition
#
# Reduced for CPU: width 16, depth 4, no Fourier features, 1024 collocation + 256 + 256 boundary
# points, 1000 Adam steps, up to 4000 quasi-Newton iterations (the plateau rule can stop it sooner).
set -u
cd "$(dirname "$0")"
PY=./.venv/bin/python
mkdir -p logs
COMMON=(--arch axisym_hybrid --width 16 --depth 4 --fourier 0
        --n-coll 1024 --n-bnd 256 --n-bnd-outer 256
        --rho-in 1.0 --rho-out 100.0 --R0 0.5773502691896258 --inner-radius 1.0
        --lam0 0.33333333333333337 --lam-inf 1.0 --lam-bc-S2 -0.08333333333333333
        --outer-bc robin --robin-orders h=3,lam=3 --no-robin-G --pin-lam-robin
        --ref-solution --ref-asymptotic 1.0 --decay-feature
        --lam-eq-form log --radial hybrid --radial-log-frac 0.25
        --eq-weights compat=0,ricci=1,gauge=1,lam_eq=1 --ricci-lam-source 1
        --steps 1000 --lbfgs-steps 4000 --qn-block 1500 --resample-every 500 --reweight-every 0
        --ckpt-every 500 --no-figures --log-every 200)
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
for tag in L1 L2; do
    extra=()
    [ "$tag" = L2 ] && extra=(--drop-robin-l 1 --drop-inner-l 1)
    echo "=== $tag: $PY -m stationary.train --outdir runs/mac_$tag ... ${extra[*]:-} ==="
    date
    $PY -m stationary.train --outdir "runs/mac_$tag" "${COMMON[@]}" ${extra[@]+"${extra[@]}"} \
        > "logs/mac_$tag.log" 2>&1
    echo "$tag exit $?"
    date
done
