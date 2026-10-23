#!/usr/bin/env bash
# Recompute, from the saved networks on Mendieta, the quadrupole numbers the paper quotes.
# Outputs go to plots/recheck_*.json and the logs; nothing existing is overwritten.
set -euo pipefail
cd ~/Julia/PINN/Stationary
REF=runs/ib16k_ssbroyden/specref_nonlinear_fine.npz
sbatch --parsable -J recheck_metrics mendieta_cmd.sbatch scripts/report_metrics.py --runs-dir runs --ref $REF \
    --out plots/recheck_report_metrics.json \
    ib16k_ssbroyden m2_dsg_s2000_t075 m3_dsg_s2000_rho4 m4_ssb_rho4 m5_ssb_norsmp m7_dsg_com m8_ssb_com \
    m8_dsg_com_cont wf_ssbm
for r in rot_ssbm rot_ssbm3 rot_from_fit ctl_hyb_ssbm ctl_hyb_ssbm2 rot_coll64k rot_coll256k_2g; do
    sbatch --parsable -J recheck_grade_$r mendieta_cmd.sbatch scripts/grade_rotated.py runs/$r --ref $REF
done
