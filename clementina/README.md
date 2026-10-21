# Running the PINN code on Clementina XXI

Target: the JAX PINN code in this repo (`Evolution_try/`, then `Stationary/` and
`SBH/`) on **Clementina XXI**, the Argentine national supercomputer at the
Servicio Meteorológico Nacional. Account **`pci_143`**, user **`oreula`**.

Documentation (Spanish): <https://docs.clementinaxxi.org.ar> — the pages this
directory is based on are [Conexión](https://docs.clementinaxxi.org.ar/es/latest/PrimerosPasos/acceso.html),
[Envío de trabajos](https://docs.clementinaxxi.org.ar/es/latest/PrimerosPasos/jobs.html),
[Software instalado](https://docs.clementinaxxi.org.ar/es/latest/PrimerosPasos/modulos.html),
[Ejecución en GPUs](https://docs.clementinaxxi.org.ar/es/latest/Desarrollo/gpus.html),
[Acceso a internet](https://docs.clementinaxxi.org.ar/es/latest/FAQ/acceso_internet_externo.html),
[Hardware](https://docs.clementinaxxi.org.ar/es/latest/Infraestructura/hardware.html),
[Almacenamiento](https://docs.clementinaxxi.org.ar/es/latest/Infraestructura/storage.html).
Support is only through <https://tickets.clementinaxxi.org.ar> (same credentials).

---

## Status (2026-10-19)

* **probe** (job 1547927, cn034): environment mapped — see §4.
* **smoke** (job 1548105, cn038): **18/18 checks passed** on one Max 1550.
  `jax 0.11.2` + `jax-oneapi-plugin 0.11.2`, `backend gpu`, six
  `OneapiDevice`s (3 unfiltered cards × 2 tiles), f32 matmul ≈ 10.7 TFLOP/s,
  f64 ≈ 370 GFLOP/s, f64 6×20 MLP `value_and_grad` at batch 2201 ≈ 2.9 ms/iter.
* **tiny end-to-end** (jobs 1548106 GPU / 1548107 CPU, §7): both `COMPLETED`,
  **identical floats** — `final_loss = 1.303484e-01` in both — which validates
  the port.  But the GPU was *slower*: 43.8 s vs 15.2 s wall, and DSGNAR's own
  breakdown says why (`batched JVPs 7.4 s` on GPU vs `2.6 s` on 64 CPU cores,
  plus ~29 s of per-step host-sync overhead).  At 512 points and 2201 parameters
  the problem is far too small to fill a GPU.
* **GPU-hours only from here.** This allocation has few CPU hours: the one CPU
  job (1548107, the correctness baseline) cost ~0.7 CPU-hours, and a full
  production recipe on 64 cores would cost ~640 CPU-hours versus ~15-25
  GPU-hours. All CPU scripts were removed so they cannot be submitted by
  accident.
* **next**: `06_bench_gpu.sbatch` measures DSGNAR at n_coll = 2201 and 16384 in
  both FLAT and COMPOSITE tile modes, and projects the GPU cost of the full
  10-window recipe from `ms/iter`.

## 0. What is different from Mendieta, in one table

| | Mendieta (what the code was written for) | Clementina XXI |
|---|---|---|
| GPUs per node | 2 × NVIDIA A30 (24 GB, CUDA) | **4 × Intel Data Center GPU Max 1550** (2 tiles each, 128 GB/card, Level Zero/SYCL) |
| JAX GPU install | `jax[cuda12]` | **`jax[oneapi]`** — experimental Intel PJRT plugin, JAX calls the backend **`oneapi`** |
| CPUs per node | 20 | 64 × Xeon Max 9462, 512 GB RAM |
| Scheduler | Slurm, `-A ifeg`, `-p multi/short` | Slurm, `-A pci_143`, `-p gpunode/cpunode/testing` (`testing` ≤ 20 min, rest ≤ 48 h) |
| Internet | login node direct | **proxy `http://172.28.3.3:3128`** on login and compute nodes |
| `/home` | 333 TB NAS | 500 GB, and results belong on `/data/contrib/pci_143` |
| Python | 3.14 venv | whatever exists on Rocky 8 — the oneAPI wheels need **≥ 3.12**, so we bring our own |
| Julia | — | no Julia module is listed; irrelevant for this path |

Three consequences that decide everything below:

1. **The backend still calls itself `gpu`** — JAX 0.11.2 on the Intel plugin
   reports `jax.default_backend() == "gpu"` and devices of class
   `OneapiDevice` with `platform == "gpu"`, so the repo's existing GPU guards
   (`assert jax.default_backend() == "gpu"`, `d.platform == "gpu"`) keep
   working. What does *not* transfer is CUDA: there is no `nvidia-smi`,
   `XLA_PYTHON_CLIENT_*` do nothing, and `/dev/dri` is not filtered per job, so
   a job can see several cards unless `ZE_AFFINITY_MASK` pins it (`env.sh`'s
   `cxxi_pin_gpu` does that from `SLURM_JOB_GPUS`).
2. **The only released `jax-oneapi-plugin` is 0.11.2** (there are 0.11.0/0.11.1
   development builds, no releases), while the repo pins
   `jax==0.11.1` (`Evolution_try/requirements.txt`, `Stationary/requirements.txt`).
   On Clementina we install 0.11.2; step 4 checks that this patch bump does not
   move the numbers (`05_cpu_tiny.sbatch` vs `03_tiny_run.sbatch`).
3. **The plugin is experimental** (JAX upstream labels Intel GPU support
   "experimental"; see the [JAX installation docs](https://docs.jax.dev/en/latest/installation.html#intel-gpu)).
   The plan is therefore staged: probe → environment → smoke test → tiny physics
   run → production. There is **no CPU fallback**, by choice: CPU hours are
   scarce, and the numbers below show why (one production recipe ≈ 640
   CPU-hours on 64 cores vs ≈ 15-25 GPU-hours on one card). The one CPU run that
   was made — the 512-point correctness baseline — is kept as a reference point.

---

## 1. What is in this directory

| File | What it is | Where it runs |
|---|---|---|
| `env.sh` | shared settings: account, proxy, `/data` run root, Intel GPU env vars, interpreter lookup | sourced |
| `00_probe.sbatch` | step 1 — discovers drivers, modules, python, partitions, ZE mask, proxy | `testing`, 1 GPU, 10 min |
| `01_setup_env.sh` | step 2 — builds `~/venvs/pinn-oneapi` with `jax[oneapi]==0.11.2` | **login node** |
| `smoke.py` | step 4 — checks the JAX primitives the DSGNAR optimiser needs; names the one that fails | any node |
| `bench.sh` | DSGNAR micro-benchmark on one GPU, FLAT vs COMPOSITE tiles; projects the full-recipe GPU cost | any GPU node |
| `06_bench_gpu.sbatch` | runs `bench.sh` in flat and composite modes (≈6 GPU-minutes total) | `gpunode`, 40 min |
| `debug_oneapi.sbatch` | when a job prints nothing and runs to its limit: every diagnostic wrapped in `timeout`, so `rc=124` names the hanging step | `gpunode`, 20 min |
| `02_smoke.sbatch` | step 3 wrapper, inside a GPU allocation | `testing`, 1 GPU, 20 min |
| `03_tiny_run.sbatch` | step 4 — the real `wave_pinn` CLI, tiny, on one Intel GPU | `gpunode`, 40 min |
| `ldc_paper_run.py` | **the paper's own trainer** (`pinn.train64` + `RunConfig`, i.e. `ldc_pinn.ipynb`), which is the reference implementation | any GPU node |
| `09_ldc_paper.sbatch` | runs `ldc_paper_run.py` with `MODE=smoke|paper` | `gpunode`, 24 h (override) |
| `ldc_run.py` | `ldc/ldc.ipynb` as a headless script: DSGNAR vs SSBroyden on the lid-driven cavity, graded against the reference field | any GPU node |
| `08_ldc.sbatch` | runs `ldc_run.py` with `MODE=smoke|a30|paper` (default a30) | `gpunode`, 8 h (override) |
| `04_prod_run.sbatch` | step 5 — the Mendieta production recipe, one GPU | `gpunode`, 8 h (override) |

All of them are idempotent enough to re-run: an existing
`runs/<label>/report.md` makes the run scripts exit immediately.

---

## 2. Connect (this Mac)

The VPN is **already up on this machine**: `openfortivpn` is running and the route
to `172.29.3.3` goes through `ppp0`. If it is ever down:

```bash
sudo openfortivpn vpn.clementinaxxi.org.ar:10443 -u oreula -p '<password>' \
     --trusted-cert 990ac2e45fd8180f5c8697bf55a26af069247460fc18de601779fefa7d0499a3
```

Then, from the Mac:

```bash
ssh oreula@ssh.clementinaxxi.org.ar          # password, or the key below
```

Optional convenience (`~/.ssh/config` on the Mac):

```
Host clementina cxxi
    HostName ssh.clementinaxxi.org.ar
    User oreula
    ServerAliveInterval 30
```

From the Mac you can also drive the cluster in one line, which is how the steps
below are meant to be read:

```bash
ssh oreula@ssh.clementinaxxi.org.ar 'cd ~/Julia/PINN/clementina && sbatch 00_probe.sbatch'
scp 'oreula@ssh.clementinaxxi.org.ar:~/Julia/PINN/clementina/probe_*.out' ~/Downloads/
```

Passwordless access: generate an ed25519 key, copy `~/.ssh/id_ed25519.pub` into
`~/.ssh/authorized_keys` on Clementina (the wiki's
[Autenticación sin contraseña](https://docs.clementinaxxi.org.ar/es/latest/PrimerosPasos/acceso.html#autenticacion-sin-contrasena)
section). First login also wants `passwd` — it is the same password as the VPN.

File transfer in either direction is `sftp`/`scp`/`rsync` over the same host.

---

## 3. Get the code and the run directory onto the cluster (login node)

Bootstrap gotcha: `env.sh` lives **inside** this repo, so it cannot be sourced
before the clone — and the login node does not resolve external host names by
itself. Export the proxy inline for the first clone (`fatal: unable to access
... Could not resolve host: github.com` is this, not a dead network):

```bash
export http_proxy=http://172.28.3.3:3128
export https_proxy=http://172.28.3.3:3128

# first time only
mkdir -p ~/Julia
git clone https://github.com/reula/PINN.git ~/Julia/PINN

# optional: make later `git pull` work without re-exporting
git config --global http.proxy  "$http_proxy"
git config --global https.proxy "$https_proxy"

# later, to pick up changes pushed from the Mac
cd ~/Julia/PINN && source clementina/env.sh && git pull --ff-only
```

If the proxy is not the problem and the clone still fails, the offline route is
`rsync` from the Mac with the VPN up (the Mac already has the repo):

```bash
rsync -avz --exclude '.venv' --exclude 'runs' --exclude '__pycache__' \
      ~/Julia/PINN/ oreula@ssh.clementinaxxi.org.ar:~/Julia/PINN/
```

The wiki also documents reaching GitHub over SSH through `ncat --proxy
172.28.3.3:3128` if you prefer that. Working copy here is clean against
`origin/main`, so a clone and an rsync are equivalent.

Results belong on the 1 PB filesystem, not in the 500 GB home:

```bash
mkdir -p /data/contrib/pci_143/pinn
```

`env.sh` exports `CXXI_DATA=/data/contrib/pci_143/pinn`, and falls back to
`~/pinn_runs` if that path is not writable (probe step 1 checks this).

The shared directory belongs to the `pci_143` group; check your groups and that
you can write before trusting it with a long run:

```bash
id
touch /data/contrib/pci_143/.w_oreula && rm -f /data/contrib/pci_143/.w_oreula \
    && echo "pci_143: writable" || echo "pci_143: NOT writable -> use ~/pinn_runs"
```

---

## 4. Step 1 — probe the machine

```bash
cd ~/Julia/PINN/clementina
sbatch 00_probe.sbatch
tail -n +1 probe_*.out probe_*.err        # the testing queue is short; ~2 min
```

The log paths in the `#SBATCH` lines are absolute
(`/home/oreula/Julia/PINN/clementina/...`), so the logs land there no matter
which directory you submit from.  A `tail: cannot open ... No such file` right
after submitting means the job is still **pending** (Slurm creates the file when
it starts) or it was rejected — check `squeue --me` and `sacct -u oreula
--starttime today`.

It prints, without touching anything: the GPU allocation and `ZE_*` variables,
`xpu-smi discovery/topology`, `sycl-ls` with and without Slurm's mask, where
`libze_loader`/`libsycl` live, the real `module avail`/`module spider` output,
which `python3` exists, `/home` vs `/data`, the real partition names and limits,
the `pci_143` association, and whether PyPI is reachable with and without the
proxy.

**What probe job 1547927 (cn034, 2026-10-19) found** — the assumptions held, with
three corrections, and the scripts are written against these answers:

| Question | Answer |
|---|---|
| partitions | `gpunode` is real (67 nodes, 266 GPUs, 2-day limit) and `testing` has only **two** nodes (`cn034`, `cn053`) with the 20-min cap — keep it for probes only |
| Level Zero | `libze_loader.so.1` and `libze_intel_gpu.so.1.6.33578` are in `/lib64`, i.e. the default search path: the pip JAX plugin needs **no** `LD_LIBRARY_PATH` help |
| python | `python3` is 3.9.18 **without pip**, `uv` absent, and `sycl-ls` does not exist even after `module load oneapi` — so the `uv` route in step 2 is the right one (plan D in §5 if its download is refused) |
| Intel GPU visibility | Slurm sets `CUDA_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` but **no** `ZE_AFFINITY_MASK`; `/dev/dri` is **not** filtered, so a `--gpus=1` job can see several Max 1550s (cn034 showed 1, cn038 showed 3). JAX/Level Zero enumerate every visible card, two tile-devices each in FLAT |
| storage | `/home` 47 TB (113 MB used), `/data` 1.1 PB, `/data/contrib/pci_143` exists for group `pci_143` |
| account | `pci_143` with `qos_pci_143`, `MaxJobs=50`, fairshare ~0.77 |
| proxy | PyPI unrestolvable **without** the proxy, HTTP 200 in ~50 ms **with** it |
| modules | `stages/2025`, `stages/2026`, `python/3.14.3` (needs `stages/2026`), `intel/2025.3.0`, `oneapi/2025.1.0`; no JAX module |

The three answers that matter:

* does `xpu-smi`/`sycl-ls` see the 4 Max 1550s (they should; the node is a GPU node);
* where the Level Zero loader is (`ldconfig -p | grep libze`) — the pip plugin
  ships the SYCL/oneMKL runtime but **not** the GPU kernel driver or its Level
  Zero loader, those come from the node image;
* the real partition spelling: the wiki says `gpunode`/`cpunode` on one page and
  `gpunodes`/`cpunodes` on another, and the account association (`pci_143`).
  If `gpunode` is wrong, `sbatch` fails instantly with "invalid partition" —
  just edit the `#SBATCH --partition` line in the other scripts.

---

## 5. Step 2 — build the Python environment (login node, once)

```bash
cd ~/Julia/PINN/clementina
bash 01_setup_env.sh
```

What it does:

1. checks PyPI through the proxy (and falls back to no-proxy if the cluster has
   direct access);
2. installs `uv` (single static binary, no root, no conda — the system `python3`
   on Rocky 8 is too old to `pip install uv`, so the curl installer is preferred);
3. `uv python install 3.12` — a managed CPython 3.12 in `~/.local/share/uv`;
4. creates `~/venvs/pinn-oneapi` and installs **`jax[oneapi]==0.11.2`**, which
   pulls `jaxlib==0.11.2`, `jax-oneapi-plugin==0.11.2` and
   `jax-oneapi-plugin[with-oneapi]` (the oneAPI runtime from PyPI: mkl,
   sycl-rt, tbb, …);
5. installs `numpy scipy optax matplotlib pytest tqdm`, then verifies the import.

Budget ~2 GB in `$HOME`. The PJRT plugin alone is a 521 MB `.so`.

Notes:

* **Do not `module load oneapi` in a job that runs this venv.** The
  `[with-oneapi]` extra ships its own oneMKL/SYCL/TBB; a module prepends a
  different set to `LD_LIBRARY_PATH` and the two fight. The only thing we take
  from the system is the GPU driver and its Level Zero loader.
* `jax[oneapi]` is a moving target: if 0.11.2 has disappeared, `uv pip index
  versions jax-oneapi-plugin` shows what exists, and `JAX_PIN` overrides the
  version. Plugin, PJRT, jaxlib and jax must all be the same version.
* On the login node `jax.devices()` is expected to show the **CPU** (no GPU
  there). That is not a failure; step 3 is the real check. For the same reason
  `python smoke.py --allow-cpu --skip-bench` on the login node only validates the
  install, not the GPU.

If the cluster cannot reach PyPI at all, plan C is to download a Linux
wheelhouse on the Mac and `rsync` it over, then `uv pip install --no-index
--find-links wheelhouse/ ...` inside the venv:

```bash
# on the Mac (Linux x86_64 + CPython 3.12 wheels, no build, no CUDA)
pip download -d wheelhouse --only-binary :all: \
    --platform manylinux_2_28_x86_64 --python-version 3.12 \
    "jax[oneapi]==0.11.2" numpy scipy optax matplotlib pytest tqdm
rsync -avz wheelhouse/ oreula@ssh.clementinaxxi.org.ar:~/wheelhouse/
```

**Alternative (plan D)** if uv's managed CPython cannot be fetched: the cluster
has a `python/3.14.3-ejkoyow` module behind `stages/2026`
(`module purge && module load stages/2026 && module load python`), and a plain
`python -m venv` + `pip install "jax[oneapi]==0.11.2"` does the same job. Use it
only as a fallback: loading the stage module also puts its compiler environment
on `LD_LIBRARY_PATH`, which the pip oneAPI runtime does not need.

**Alternative (plan B)** if the pip runtime refuses to talk to the node's
driver: load the cluster's oneAPI module and install the plugin *without* the
bundled runtime — `uv pip install "jax==0.11.2" "jaxlib==0.11.2"
jax-oneapi-plugin==0.11.2` — then run with `module load oneapi` in the job.
The probe tells us the module name/version to use.

---

## 6. Step 3 — get the code's numbers checked on the GPU

```bash
cd ~/Julia/PINN/clementina
sbatch 02_smoke.sbatch
tail -n +1 smoke_*.out
```

`smoke.py` prints the backend, every device and its `device_kind`, then runs the
primitives `wave_pinn` actually uses — `jit`, `grad`, `vmap`, **batched `jvp`**
(DSGNAR's tangent batches), `jax.scipy.fft.dct/idct` (the sketch), `jnp.linalg.svd`
(tangent truncation), `lax.scan/while_loop`, buffer donation, pytree flattening,
a 256 MiB host↔device copy and the f32/f64 matmul timings — and finally the real
shape: a 6×20 MLP, batch 2201, `value_and_grad`, float64.

Exit code tells you where you are:

| exit | meaning | next |
|---|---|---|
| 0 | everything ran on the Intel GPU | go to step 4 |
| 1 | one named primitive failed — the check name is the missing kernel | send the output back; it is a JAX/plugin limitation to work around, not a code bug |
| 2 | backend is `cpu`: the plugin did not load a device | see Troubleshooting |

---

## 7. Step 4 — tiny end-to-end run (GPU)

```bash
cd ~/Julia/PINN/clementina
sbatch 03_tiny_run.sbatch          # gpunode, one Intel GPU, ~1 min of work
```

Already done, as job 1548106:

* `pinned ZE_AFFINITY_MASK=0,1`, two `OneapiDevice`s, `backend gpu`;
* the real CLI ran the DSGNAR path in float64 and finished in 43.8 s of run time
  (1:05 of job time), exit 0;
* one CPU run on 64 cores (job 1548107) gave **the same loss to every printed
  digit** — `1.303484e-01` — which is the validation that the oneAPI port is
  numerically faithful. Its 15.2 s is also the whole CPU baseline we are
  spending: see §8 for why production cannot go that way.

## 7b. Step 4b — what will the production run cost in GPU hours?

```bash
cd ~/Julia/PINN/clementina
sbatch 06_bench_gpu.sbatch
sleep 60; cat /data/contrib/pci_143/pinn/bench_gpu.csv
```

Four cases, ~6 GPU-minutes: n_coll = 2201 (production) and 16384
(Stationary-sized) × FLAT (one 64 GB tile) and COMPOSITE (one 128 GB device).
Each row reports `ms/iter` for DSGNAR and
`projected_10window_h = ms/iter × 15000 / 3600` — the cost of the full
`T=20`, 10-window, 5-round production recipe.

## 8. Step 5 — production

Same recipe as `Evolution_try/scripts/mendieta_run.sbatch` (`T=20`, 10 windows,
`t2` ansatz, `n_coll=2201`, DSGNAR 300×5, stop audits on), on one Intel GPU:

```bash
cd ~/Julia/PINN/clementina
sbatch 04_prod_run.sbatch cxxi_dsgnar_soft_all soft_all 0
sbatch 04_prod_run.sbatch cxxi_dsgnar_hard_s1500 hard 1500
sbatch -t 24:00:00 04_prod_run.sbatch cxxi_long soft_all 1500 --set dsgnar_steps=600
```

The first three positional arguments are the same as the Mendieta script
(`label`, `window_ic`, `dsgnar_sketch`); everything after them is passed straight
to the CLI as `--set k=v`. Command-line `sbatch` options override the directives
in the file (`-t`, `--gpus`, `-c`, `--mem`).

To submit the whole `mendieta_submit.sh` list, feed its `RUNS` array to this
script, e.g.:

```bash
while IFS='|' read -r label ic sk extra; do
    [ -f "/data/contrib/pci_143/pinn/Evolution_try/runs/$label/report.md" ] && continue
    sbatch -J "$label" 04_prod_run.sbatch "$label" "$ic" "$sk" $extra
done < runs.txt
```

Monitoring:

```bash
squeue --me
squeue -u oreula -o "%.10i %.30j %.8T %.10M %.6D %R"
scancel <jobid>
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize
scontrol show job <jobid> | grep -E 'ExitCode|RunTime|NodeList'
```

### Intel-backend codegen: paper shapes compile, smoke shapes do not

The paper trainer (`09_ldc_paper.sbatch`) hits an XLA-SYCL codegen crash on the
**smoke** configuration:

```
LLVM Fatal Error while compiling target binary for module: jit__step_impl
Reason: OpTypeVector with 32 components requires the following SPIR-V
        extension: SPV_EXT_long_vector
```

It is not the float32 pre-pass (job 1548876 with `--float32-steps 0` fails the
same way) and not our code: the **paper** shapes compile and run
(job 1548877: `Compile complete 398.8 s` for float32, `927.8 s` for float64, then
the trainer stepped). The 32-lane vector is produced by the vectoriser on the
small smoke shapes, which is where the SYCL backend's SPIR-V legaliser gives up.
Consequences:

* use `MODE=paper` for this trainer on Clementina; `MODE=smoke` only on the hub
  or on the CPU (locally it gives mean rel L2 5.8-6.6e-02);
* the persistent compile cache cannot be written on this backend
  (`Unsupported platform ID for XlaExecutableAbiVersion`), so **every run pays
  ~22 min of compile** before training starts — budget it in GPU hours;
* the trainer logs only at pass boundaries, so a long float64 pass is silent.

### Spend GPU hours in stages

1. **reduced-fidelity run of the same recipe** — 10 windows, but 100 DSGNAR steps
   and one resample round instead of 300×5 (~1-1.5 GPU-h).  Enough to exercise
   the whole pipeline, windows, snapshots and report; nothing scientific.
2. **the full recipe** (≈15-25 GPU-h by the bench projection, under the 48 h
   limit) once stage 1 is trusted.

   ```bash
   sbatch -t 04:00:00 04_prod_run.sbatch cxxi_stage1 soft_all 0 \
       --set dsgnar_steps=100 --set resample_rounds=1
   sbatch -t 24:00:00 04_prod_run.sbatch cxxi_full soft_all 0
   ```

Note on GPU visibility in production: the device nodes are not filtered per job.
A `--gpus=1` allocation can still enumerate several Max 1550s, so the *default*
device is whichever card Level Zero lists first — usually the allocated one, but
not guaranteed when other jobs hold other cards. The smoke log prints how many
devices JAX sees; if it is more than the allocation, pin it with
`ZE_AFFINITY_MASK` (FLAT: `0,1` = card 0; COMPOSITE: `0.0,0.1`) via
`sbatch --export=ALL,ZE_AFFINITY_MASK=0,1`.

With the default FLAT hierarchy, single-device code uses the first tile it is
given and has 64 GB; `CXXI_ZE_MODE=COMPOSITE` collapses each card into one
128 GB device if a recipe needs more (nothing in this repo has needed ~24 GB).

---

## 8b. The LDC notebook as a job (`08_ldc.sbatch`)

`ldc/ldc.ipynb` (lid-driven cavity, Re = 100; DSGNAR from the paper's vendored
package vs Crunch SSBroyden, every iterate graded against
`ldc/reference_gen/ldc_reference.npz`) has a batch twin: `ldc_run.py` — same
numerics, headless, no writes into the checkout, outputs under `/data`.

```bash
sbatch clementina/08_ldc.sbatch                                        # a30 preset
sbatch --export=ALL,MODE=smoke clementina/08_ldc.sbatch                # minutes: pipeline check
sbatch -t 24:00:00 --export=ALL,MODE=paper clementina/08_ldc.sbatch    # the paper's numbers
```

| MODE | net | points (pde/bc) | sketch | batch | dense Jacobian |
|---|---|---|---|---|---|
| `smoke` | 40-40-40 | 2^11 / 2^10 | 256 | 256 | 0.02 GiB |
| `a30` | 48-48-48 | 2^13 / 2^12 | 1665 | 1024 | 0.12 GiB |
| `paper` | 60-60-60-60 | 2^15 / 2^14 | 4000 | 4096 | ~1.1 GiB |

Outputs in `/data/contrib/pci_143/pinn/ldc/ldc_<mode>/`: `summary.json` (grades,
losses, timings), `history.json`, `results.npz`, `dsgnar_flat.npy`,
`figures/optimizers_loss_and_relL2.png`, `figures/fields_{u,v,p}.png`.

Budgets can be trimmed without changing the preset:
`--dsgnar-steps`, `--qn-maxiter`, `--float32-steps` (pass them after the sbatch
script, e.g. `sbatch clementina/08_ldc.sbatch --dsgnar-steps 20`), and
`--skip-fields` drops the per-channel field figures.

The paper preset fits a 48 GiB tile comfortably (Jacobian ~1.1 GiB, sketch
0.12 GiB, XLA temporaries 2-3x the Jacobian). What it costs is **time**: the
comparison runs DSGNAR 200 float64 steps and SSBroyden 2 x 1000 quasi-Newton
iterations at 2^15 points, so submit with `-t 24:00:00` and expect hours rather
than minutes.

## 8b-sexies. RESULT: DSGNAR now works on the Intel tile (job 1549619)

With the exact residual rows (default-off `exact_residual=True` in the vendored optimiser,
`--exact-residual` on the drivers) and the float32 pre-pass skipped, the a30 preset **beats the
A30 hub run**:

| run (a30 preset, 48-48-48, 2^13/2^12, sketch 1665) | loss | rel L2 mean | u | v | p |
|---|---|---|---|---|---|
| **Intel, exact residual, float64 only (1549619)** | **3.12e-07** | **0.008634** | 0.00149 | 0.00249 | 0.0219 |
| A30 hub, count sketch (`hub_a30_dsgnar_48x48x48`) | 1.84e-07 | 0.009266 | 0.00177 | 0.00337 | 0.02266 |
| Intel, count sketch, float64 only (1549432) | 6.05e-04 | 0.6617 | 0.553 | 0.820 | 0.611 |
| Intel, count sketch + SSBroyden (1549369, old best) | 3.16e-08 | 0.01139 | 0.00138 | 0.00270 | 0.03007 |

400/400 DSGNAR steps accepted, rho pinned at 0.074-0.076 (the plot's target 0.075), radius
staying ~1e-2, loss still falling at the 400-step budget: 2076 s of DSGNAR + 541.8 s of
SSBroyden polish (10000 iterations), 52 min wall on one tile.  Saved and re-verified:

```bash
python clementina/ldc_params.py   --params ldc/params/intel_a30_exact_48x48x48.npy   --network ldc/params/intel_a30_exact_48x48x48.json
# [score ] {"u": 0.001489, "v": 0.002488, "p": 0.021923, "mean": 0.008634}
```

How to reproduce / rerun:

```bash
# the a30 run (exact residual, float64 only, + SSBroyden)
sbatch -p gpunode -t 01:30:00 clementina/17_gpu_a30_exact.sbatch
# the 8-step smoke of the change
sbatch -p gpunode -t 00:30:00 clementina/16_gpu_replay_exact.sbatch
# or directly
python clementina/ldc_run.py --mode a30 --exact-residual --float32-steps 0
```

Two Intel-specific compile fixes were needed on top of the ported algebra (both committed):
the dense `jacrev` must run on the usual 1024-point macro-batches and the Gram system be
accumulated block by block — a single `jacrev` over a whole condition's 8192 points makes XLA
emit 32-wide vectors and the SYCL backend dies with `SPV_EXT_long_vector` — and the
eigendecomposition of the Gram matrix must be an SVD (`S = sqrt(sigma)`, `Vt = V^T`,
`g = V^T b`): `jnp.linalg.eigh` on the 1665x1665 matrix crashes the same way, the SVD compiles.

Artifacts: `ldc/params/intel_a30_exact_48x48x48.{npy,json}` (parameters + architecture), and
`clementina/results/ldc_a30_exact_{summary,history}.json` / `_results.npz` (full per-step trace).

## 8b-quinquies. The fix to test: exact residual rows (from SBH/06fac11)

The other agent's DSGNAR work (`Stationary`/`SBH`, commit 06fac11) found the mechanism behind
the floor: DSGNAR's step direction is `J^T C^T C r`, whose sketch-noise term
`J^T (C^T C - I) r ~ |J||r|/sqrt(s)` does not shrink with the residual, while the true gradient
`|J^T r|` does as the residual becomes orthogonal to range(J).  Near the floor the step is then
close to random (uphill in 8/20 sketch draws in their small-problem test) and the trust region
collapses — exactly the rho -> 0 / negative / NaN pattern seen here.  Their fix keeps the
parameter sketch but **drops the CountSketch on the residual rows**, accumulating the exact
Gram system `G = (J Omega S)^T (J Omega S)`, `b = (J Omega S)^T r` over every row and
eigendecomposing it (from a stall point: 30/30 accepted vs 2/13).

That is now available in the vendored paper optimiser behind a default-off flag
(`Optimiser(exact_residual=True)`, a LOCAL PATCH in
`Evolution_try/vendor/dsgnar_pinn/pinn/optimiser.py`) and exposed as:

```bash
python clementina/ldc_run.py    --mode a30 --exact-residual ...      # training
python clementina/ldc_replay.py --float32-steps 0 --exact-residual   # 6-step test
```

Why it should also cure the Intel-specific instability: the count-sketch is what leaves the
1665x1665 system with only ~258 singular values above 1e-12*max (down to 1e-17); the exact Gram
system has the spectrum of `J Omega S` squared, with no sketch-induced null space, so the
`S/(S^2+lam)` amplification that makes the large-radius probe steps platform-sensitive
disappears.  Validate on CPU (free) first, then on one Intel GPU; keep the flag off elsewhere
until it is confirmed.

## 8b-quater. Why DSGNAR stalls on the Intel tile: a float32-only overflow

With an identical initial state (`loss 6.3471e-01`, rel L2 4.696) and byte-identical
vendored code (md5 checked on Mac, cluster and hub), the hand-driven DSGNAR behaves
completely differently per platform:

| platform | step 1 (float32) | later |
|---|---|---|
| A30 (hub) | loss 1.0695e-01, rho +0.824, **accept** | rho pinned at the 0.075 target, loss → 2e-07 in 215 steps |
| CPU (Mac, jax 0.11.1 CPU) | 1.0924e-01, rho +0.820, accept | same trajectory as the A30 |
| Intel tile (oneAPI 0.11.2) | loss 6.3501e-01, **rho +NaN, reject** | radius ratchets 3e-1 → 1e-13, loss freezes at 4.6e-04 |

Diagnostics (jobs 1549424/1549425/1549426, all ~1-4 GPU-minutes):

* the dense Jacobian, the count-sketch `B`, the SVD, the secular solve and the
  predicted reductions are **clean** on the Intel tile in both dtypes (no NaN, no inf);
* the faithful replay (`ldc_replay.py`, key for key with `ldc_run.py`) reproduces the
  failure: in the **float32** pass `probe_losses` contain 12-24 NaNs and finite
  remnants of ~1e12 (residual-squared overflow), so `probe_rhos` and the pchip radius
  selector are all NaN, all four float32 steps are rejected and the radius collapses;
* the **float64** pass is healthy: its first replay steps accept with rho 0.891 / 0.963;
* CPU and A30 have probe losses O(0.02-0.2) in the same float32 pass.

So the defect is in the *lifted probe-step evaluation* in single precision on this
SYCL backend — not in the optimiser's logic, the residual, SSBroyden, or the code.
**Workaround: skip the float32 pre-pass on the Intel tile** (`--float32-steps 0`); the
float64 pass is the one that carries the convergence.  The same flag exists for the
paper trainer but does not rescue its smoke shapes, which fail for a different reason
(§8b-ter).

## 8b-ter. Result on the Intel tile (`a30` preset, job 1549369)

`08_ldc.sbatch` (MODE=a30, one pinned Max 1550), **completed in 24 m 36 s**:

| phase | iterations | time | paper loss | rel L2 (u / v / p / mean) |
|---|---|---|---|---|
| DSGNAR (paper code) | 82 | 440 s | 4.43e-04 | 0.511 / 0.665 / 0.535 / **0.5703** |
| SSBroyden from x0 | 10 000 | 1 050 s | **3.16e-08** | 0.00138 / 0.00270 / 0.03007 / **0.01139** |
| SSBroyden polish from DSGNAR | 10 000 | 520 s | 3.17e-08 | 0.01139 |

Best by accuracy: **SSBroyden from x0, mean rel L2 0.01139** — the same class as the
A30 hub run (`0.00927`, loss 1.8e-07), reached with a different route.  Saved and
re-loadable: `ldc/params/intel_a30_48x48x48_best.npy` + `.json`; the loader scores it
at mean 0.011386.  Figures: `clementina/results/ldc_a30_intel_optimizers.png`,
`..._fields_u.png`; full run in `/data/contrib/pci_143/pinn/ldc/ldc_a30/`.

**Open question:** with the *same* initial state (loss 6.3471e-01, rel L2 4.696, identical
to the A30 run) DSGNAR on the Intel tile collapses its trust region to 1e-13 after 82
iterations and stops at 0.57, while on the A30 it ran 238 iterations to 0.0093.  The
SSBroyden phase is unaffected.  Whether DSGNAR's rho estimates are perturbed by the SYCL
reductions (accept iff rho > 0 is a knife-edge) is the thing to check if the paper's
DSGNAR advantage must be reproduced on this hardware.

## 8b-bis. Trained parameters are saved with their architecture

Every run written by `ldc_run.py` and `ldc_paper_run.py` now saves

    <run>/params_flat.npy   the trainable arrays, flattened, float64
    <run>/network.json      hidden_dims, seed, SIREN options, driver, config,
                            jax/jaxlib versions, backend, rel L2 per channel
    <run>/params.eqx        exact equinox leaves (same numbers, with structure)

because a flat vector is meaningless without the net that produced it.  To score or
plot a saved network, with or without having produced it here:

```bash
python clementina/ldc_params.py --run /data/contrib/pci_143/pinn/ldc/ldc_paper_paper
python clementina/ldc_params.py --params ldc/params/hub_a30_dsgnar_48x48x48.npy \
                                --network ldc/params/hub_a30_dsgnar_48x48x48.json --outdir /tmp/eval
```

The A30 solution (hub `ldc.ipynb`, `MODE=a30`, 238 DSGNAR iterations in 203.5 s,
paper loss 1.8e-7) is stored that way in `ldc/params/`: the loader rebuilds the
48-48-48 SIREN and scores it at **u 0.00177, v 0.00337, p 0.02266, mean 0.00927**.

The `a30` preset in `ldc_run.py` now matches that run (`dsgnar_steps=400,
qn_maxiter=10000, min_radius=1e-14`); the previous 200-step / 1e-8 version came from
an older copy of the notebook.  `bench_gpu.csv`-style projection: the hub did 238
iterations in 203.5 s on an A30; the Intel tile is several times slower per iteration
at equal size.

## 8c. Which LDC driver — use the paper's own trainer (`09_ldc_paper.sbatch`)

The hub's `ldc/` folder has four generations of this notebook, and they are not
the same driver. Measured on the hub (A30, jax 0.11.1, CUDA):

| notebook | driver | hub result |
|---|---|---|
| `ldc_pinn.ipynb` | **`pinn.train64` + `pinn.RunConfig`** — the reference implementation, with its `auto_adjust` weight/λ schedule | **final rel error 6.384e-02 in 11.5 s** (SMOKE config, 2^11 pts, 40-40-40) |
| `ldc_before_warmup.ipynb` (`PAPER=True`) | `stationary.dsgnar.dsgnar_phase`, 5×60 cycles at s=128, then SSBroyden | 1.27e-2 / 1.92e-2 / 4.90e-2 (u/v/p) |
| `ldc-paper.ipynb` (PAPER mode) | `dsgnar_phase`, 200 iters at s=1665 — accepted **0** steps — then SSBroyden | 2.21e-2 / 3.54e-2 / 5.59e-2 |
| `ldc.ipynb` | hand-driven `pinn.optimiser.Optimiser` loop + own SSBroyden — what `08_ldc.sbatch` ports | cluster job 1548247: DSGNAR stuck at rel L2 ≈ 0.68 |

So: **the paper's trainer is both the reference implementation and the cheapest
good result**, and the hand-driven variant in `ldc.ipynb` is the one that stalls.
`09_ldc_paper.sbatch` runs the former:

```bash
sbatch --export=ALL,MODE=smoke clementina/09_ldc_paper.sbatch     # minutes
sbatch clementina/09_ldc_paper.sbatch                             # MODE=paper, 2^15 pts
```

Outputs in `/data/contrib/pci_143/pinn/ldc/ldc_paper_<mode>/`: `summary.json`
(trainer rel-L2 per channel and mean, config, wall time), `results.npz`,
`figures/fields_{u,v,p}.png`. `ldc_paper_run.py` locally reproduces the hub smoke
run on CPU: mean rel L2 6.64e-02 (u 3.4e-2, v 3.9e-2, p 1.3e-1).

Note the loss/accuracy question: inside this trainer the two track each other —
it prints the rel error at the end of each precision pass and stops on the trust
radius — which is why the reference implementation is the fair test of the
paper's claim. The decoupling seen in job 1548247 is a property of the
hand-driven `ldc.ipynb` loop (fixed residual weights, no `auto_adjust`), not of
the problem.

## 9. Files back, and where results live

Everything a run writes is under
`/data/contrib/pci_143/pinn/Evolution_try/runs/<label>/` (report, config,
checkpoints, snapshots). Copy it back from the Mac with the VPN up:

```bash
rsync -avz oreula@ssh.clementinaxxi.org.ar:/data/contrib/pci_143/pinn/Evolution_try/runs/cxxi_tiny \
      ~/Julia/PINN/Evolution_try/runs/
```

Keep `/home` for code and the venv only; the wiki says `/home` is 500 GB per user
and heading to 500 GB per IPAC, while `/data/contrib` is ~1 PB shared by the IPAC.

---

## 10. Troubleshooting (mapped to the wiki's own sections)

| Symptom | Cause | Fix |
|---|---|---|
| `ssh: Could not resolve hostname` | DNS, or VPN down | `ssh oreula@172.29.3.3`, or set DNS `8.8.8.8` (wiki: Errores Frecuentes) |
| `Permission denied (publickey,...)` | key not in `authorized_keys` | wiki's *Autenticación sin contraseña*; password login always works |
| `sbatch: error: invalid partition` | `gpunode` vs `gpunodes` | use the name printed by `sinfo` in the probe |
| job inherits the login shell Lmod environment (`LD_LIBRARY_PATH` with ohpc gcc 13 / openmpi5) | Slurm copies the submit environment | jobs now `unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME`; do the same by hand with `srun --pty` |
| job runs to its time limit printing nothing after the header | a hang: `xpu-smi`, python startup, or the oneAPI plugin init | `sbatch -p gpunode -t 00:20:00 clementina/debug_oneapi.sbatch`; the `rc=124` step is the culprit |
| log file not there right after `sbatch` | job still `PD`, or log went to the submit directory (older relative paths) | `squeue --me`; `ls -t ~/Julia/PINN/*.out ~/Julia/PINN/clementina/*.out` |
| `sbatch: error: invalid account` | `pci_143` association | `sacctmgr show assoc user=$USER` in the probe output |
| stderr says `/var/spool/slurmd/job*/env.sh: No such file` | older checkout: the script looked for `env.sh` next to the slurmd spool copy | `git pull --ff-only`; the scripts now locate `env.sh` through an absolute `CXXI_DIR`, overridable with `sbatch --export=ALL,CXXI_DIR=/path/to/clementina` |
| `git clone`: `Could not resolve host: github.com` | proxy not exported **and** no external DNS — expected before `env.sh` is available | export `http_proxy`/`https_proxy` inline (see §3); `rsync` from the Mac otherwise |
| pip/uv/git hang or 403 | proxy missing | `export https_proxy=http://172.28.3.3:3128` (and `http_proxy`) |
| `python3: not found`/3.6 | Rocky 8 base | irrelevant: `01_setup_env.sh` brings CPython 3.12 via uv |
| smoke exit 2, backend `cpu`, message `libze_loader.so.1: cannot open shared object file` | plugin cannot find the Level Zero loader | take the *directory* of `libze_loader.so.1` from the probe (`ldconfig -p`) and prepend only that to `LD_LIBRARY_PATH`; or plan B in §5 |
| smoke exit 2, backend `cpu`, no library error | `ZE_AFFINITY_MASK` hides everything (wiki documents Slurm getting this wrong) | try `CXXI_UNSET_ZE_MASK=1 sbatch --gpus=4 02_smoke.sbatch` on a whole node, or set the mask explicitly for the assigned GPU |
| smoke exit 1 on a specific primitive | missing/incorrect XLA SYCL kernel in 0.11.2 | send the check name back; either avoid that op in the recipe or try another plugin version |
| JAX prints `No GPU/TPU found, falling back to CPU` | plugin not registered (`jax-oneapi-plugin` missing) | `uv pip list | grep oneapi`; reinstall `jax[oneapi]==<jax version>` |
| weird crashes after `module load oneapi` | two oneAPI runtimes on `LD_LIBRARY_PATH` | do not load that module in jobs that use the pip venv |
| very slow job, 100 % CPU | silent CPU fallback | the run scripts refuse this; if you run the CLI by hand, check `jax.default_backend()` first |
| job sits `PENDING (Resources)` | normal on a shared queue | submit and walk away; `sshare -u oreula` shows fair-share |

Do not wait at the terminal for a queued production job; `sbatch` and check later.
The `testing` partition is capped at 20 minutes and shared — keep probes short.

---

## 11. What is verified from here, and what is not

Verified on the Mac: the runbook's facts come from the wiki (linked above); the
JAX/oneAPI packaging was checked against PyPI (0.11.2 plugin/PJRT/jaxlib, the
`oneapi` extra contents, Python ≥ 3.12, backend name `oneapi`, priority 500); the
tiny settings were dry-run through the real CLI (`--dry-run`) so the flags exist;
`smoke.py` was run locally on CPU JAX.

Not verified (needs the cluster): everything in step 1's three questions, whether
`jax[oneapi]` initialises against this node image, which primitives work, and the
GPU/CPU timing ratio. Those are exactly what steps 1, 3 and 4 report.

## 12. Next after Evolution_try

* `Stationary/`: same venv; `mendieta_train.sbatch` needs `-A pci_143 -p gpunode
  --gpus=1`, the CUDA `nvidia-smi` line replaced, and the GPU assert relaxed to
  `backend != "cpu"`. Its `_device_memory()` is already guarded against
  `memory_stats()` returning `None`, which is what oneAPI devices tend to do.
* `SBH/`: same treatment; it is the heaviest recipe (single-puncture PCCZ4), so
  run it after the tiny `wave_pinn` run proves the backend.
* The Julia notebooks in `Jax/` and `PINNS_codes/` are not on this path: there is
  no Julia module on Clementina, `oneAPI.jl` is far less complete than CUDA.jl,
  and `LuxCUDA` cannot drive an Intel GPU — CPU-only would be the only option.
