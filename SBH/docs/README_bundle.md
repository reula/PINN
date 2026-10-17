# BH PINN production bundle (2026-10-16)

## Contents

| file | role | md5 |
|---|---|---|
| `EEccz4_BH_puncture_gpus.py` | production code: the base code plus multi-GPU and the blow-up guard | 73961c50… |
| `EEccz4_BH_puncture.py` | base single-GPU code (the version exchanged with Carlos) | 9f84e480… |
| `gpus.patch` | difference between the two codes | |
| `single_puncture.psubmit` | MN5 job script: single puncture, m = 0.5, t ≤ 10 | c672618a… |
| `execution.psubmit` | MN5 job script: BBH | 9eda47c4… |
| `Crunch/` | optimiser library (SSBroyden with Wolfe line search). Must sit next to the code. | |

## Running it

Run from the folder that holds the code and `Crunch/`. Both job scripts:

- default to `EEccz4_BH_puncture_gpus.py` on 2 GPUs (`NGPU=2`, `--gres=gpu:2`);
- add the production flags `--fast_hessian --loss_plot_every 10 --blowup_factor 100`.

The settings from the best run so far are **not** defaults. Pass them with EXTRA_ARGS:

    sbatch --export=ALL,OUTDIR=my_run,EXTRA_ARGS="--core_points 48000 --core_ic_points 4800 --core_radius 0.5 --EPOCHS_QN 24000" single_puncture.psubmit

- For a quick test (4 windows, t ≤ 1, about 72 min on 4 GPUs in the debug queue):

      sbatch -q acc_debug -t 02:00:00 --gres=gpu:4 --cpus-per-task=80 --export=ALL,NGPU=4,OUTDIR=test,EXTRA_ARGS="... --tfinal 1 --N_window 4" single_puncture.psubmit

- For 4 GPUs: `--gres=gpu:4 --cpus-per-task=80` plus `NGPU=4` in `--export`.

## Account-specific lines to edit

- `#SBATCH --account=uib108` and `--qos=acc_resa`.
- `ENV=jax09` activates the shared venv `/gpfs/projects/uib108/venvs/pinn` (Python 3.12, JAX 0.9.2). Without access to uib108, point it to your own environment: Python 3.12 with jax[cuda12] 0.9.2, flax, optax, numpy, scipy, matplotlib and tqdm.

## Comparing with the 1D reference (optional)

`reference_1d/` is not included in this bundle; ask for it if needed (`compare_pinn.py` plus the 17 MB reference file).
