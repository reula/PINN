#!/usr/bin/env bash
# The Weyl two-rod tests of the paper (Sec. IV C-D).  --weyl takes the inner data, the Robin source
# and the comparison asset from the exact solution and imposes the cylindrical gauge source.
# The axisymmetric runs (weyl_prod, weyl_unequal, weyl_close4/35) are kept in runs/; their
# settings are in each run's config.json.  The rotated test is rerun here with the metric-only 3-D
# network (the original weyl_rot45_hub used the first-order mlp): same rods (masses 1 and 0.1,
# gap 1, shell [3.5, 35] trained in the chart scaled by 1/35), 45 deg rotation, 8192 + 512 points,
# SSBroyden from cold up to 20000 iterations.
ROT=(--outdir runs/weyl_rot45_hyb --weyl --weyl-half-length 0.028571428571 --weyl-half-length-b 0.002857142857
     --weyl-half-gap 0.014285714286 --weyl-rotate-deg 45 --rho-in 0.1 --rho-out 1.0 --arch hybrid
     --n-coll 8192 --n-bnd 512 --w-inner 10 --w-outer 10 --reweight-every 2000 --steps 0
     --qn-method ssbroyden --lbfgs-steps 20000 --ckpt-every 500 --vtk-physical-inner 0.1)
if [ "${SUBMIT:-0}" = "1" ]; then
    echo "weyl_rot45_hyb -> $(sbatch --parsable -t 12:00:00 -J weyl_rot45_hyb mendieta_cmd.sbatch -m stationary.train "${ROT[@]}")"
fi
