#!/usr/bin/env bash
# The correct-Ricci-source batch.  Six cells, run SEQUENTIALLY, each detached by run_hub.sh and
# waited for, so the whole batch survives the terminal:
#
#     nohup setsid bash run_batch_D.sh > logs/batch_D.driver.log 2>&1 &
#
# Relaunching after an interruption: SKIP_DONE=1 leaves cells that already exited 0
# alone and runs the rest:
#     SKIP_DONE=1 nohup setsid bash run_batch_D.sh > logs/batch_D.driver.log 2>&1 &
#
# All cells share: correct Ricci source (--ricci-lam-source 1), the metric's pins and weights as in
# the PREDECESSOR family that predates the zero-source switch (--pin-h-robin, w_inner = w_outer =
# 100, --reweight-every 500, --qn-block 250), and 500 Adam steps then up to 30000 SSBroyden.
#
#   U1  the old family's configuration exactly (lambda form, log sampling, 32768/2048/2048 points,
#       fourier 0)                                     -> reproduces pq_u100_s2_12, which had the
#                                                          correct source already
#   U2  U1 + the dipole exempted from every condition (metric and lambda, both spheres)
#   P1  our current recipe (phi = ln lambda, hybrid 0.25, 27768/4548/4548 points)
#   P2  P1 + the exemption
#   F1  P1 with radial harmonics (--fourier 8)         -> is the floor a resolution effect?
#   F2  F1 + the exemption
#
# Acceptance is NOT the loss and not report.txt's table (that one is on the outer sphere only, and a
# thin boundary layer can satisfy it while the bulk is wrong): judge on the profile of h_rr and on
# the BULK residual that `bulk_residuals.py` prints at the end.
set -u
cd "$(dirname "$0")"
mkdir -p logs
# Belt and braces now that the batch failed exactly this way once: run_hub.sh picks this up,
# and the driver refuses to march on if a cell does not reach exit 0.
export PY="${PY:-$PWD/.venv/bin/python}"
[ -x "$PY" ] || { echo "no interpreter at $PY -- aborting"; exit 1; }
# The physics flags are NOT repeated here on purpose: problem.py now defaults to them
# (ricci_lam_source = 1, pin_h_robin = True, w_inner = w_outer = 100, reweight_every = 500,
# qn_block = 250, resample_every = 500), so a launcher cannot forget them again.  What is left
# below is only what differs BETWEEN the cells.
COMMON=(--arch axisym_hybrid --width 20 --depth 6 --steps 500 --lbfgs-steps 30000
        --R0 0.5773502691896258 --rho-in 1.0 --inner-radius 1.0 --rho-out 100.0
        --ref-solution --ref-asymptotic 1.0 --lam-bc-S2 -0.08333333333333333
        --outer-bc robin --robin-exps 2,3,1 --robin-orders h=3,lam=3 --no-robin-G --lam-inf 1.0
        --decay-feature --pin-lam-robin --pin-h-robin --w-pin 100
        --outer-shell-fracs 1.0,0.9,0.8,0.7 --kill-dipole h   # condition on 4 spheres; no dipole in h
        --eq-weights compat=0,ricci=1,gauge=1,lam_eq=1
        --ckpt-every 500 --log-resources --vtk --vtk-physical-inner 0.01 --log-every 200)
DROP=(--drop-robin-l 1 --drop-inner-l 1)
U=(--radial log --n-coll 32768 --n-bnd 2048 --n-bnd-outer 2048)
P=(--lam-eq-form log --radial hybrid --radial-log-frac 0.25 --n-coll 27768 --n-bnd 4548 --n-bnd-outer 4548)

launch_and_wait () {   # name, extra flags...
    local name=$1; shift
    # SKIP_DONE=1 resumes a batch: a cell that already exited 0 is left alone, so relaunching
    # after the driver dies (or after a cleanup killed it) does not redo the finished cells.
    if [ "${SKIP_DONE:-0}" = "1" ] && [ "$(cat "runs/$name/train.exit" 2>/dev/null)" = "0" ]; then
        echo "=== $name  already finished (exit 0), skipping"
        return
    fi
    echo "=== $name  launched $(date)"
    bash run_hub.sh --outdir "runs/$name" "${COMMON[@]}" "$@"
    local waited=0
    while [ ! -f "runs/$name/train.exit" ]; do
        sleep 60; waited=$((waited + 60))
        if [ "$waited" -gt 21600 ]; then echo "=== $name: no exit after 6 h, moving on"; return; fi
    done
    local st; st=$(cat "runs/$name/train.exit")
    echo "=== $name  done $(date)  exit=$st"
    if [ "$st" != "0" ]; then
        echo "=== $name exited $st -- ABORTING the batch (fix and delete runs/$name/train.exit to retry)"
        tail -25 "logs/$name.log"
        exit 1
    fi
    grep -aE "final loss|gauge +rms|lam_eq +rms|ricci +rms" "runs/$name/report.txt" 2>/dev/null | head -4
}

launch_and_wait pq_b_U1 "${U[@]}"
launch_and_wait pq_b_U2 "${U[@]}" "${DROP[@]}"
launch_and_wait pq_b_P1 "${P[@]}"
launch_and_wait pq_b_P2 "${P[@]}" "${DROP[@]}"
launch_and_wait pq_b_F1 "${P[@]}" --fourier 8
launch_and_wait pq_b_F2 "${P[@]}" --fourier 8 "${DROP[@]}"

echo "=== batch finished $(date); acceptance table:"
./.venv/bin/python bulk_residuals.py runs/pq_b_U1 runs/pq_b_U2 runs/pq_b_P1 runs/pq_b_P2 \
    runs/pq_b_F1 runs/pq_b_F2 2>&1 | tail -20
