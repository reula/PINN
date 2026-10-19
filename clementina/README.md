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

1. **`jax.default_backend()` returns `"oneapi"`, not `"gpu"`.** Every existing
   guard in the repo (`assert jax.default_backend() == "gpu"`,
   `[d for d in jax.devices() if d.platform == "gpu"]` in the `mendieta_*.sbatch`
   and `hub_*.sh` scripts) would reject a perfectly working GPU on Clementina.
   The scripts here test `backend != "cpu"` instead.
2. **The only released `jax-oneapi-plugin` is 0.11.2** (there are 0.11.0/0.11.1
   development builds, no releases), while the repo pins
   `jax==0.11.1` (`Evolution_try/requirements.txt`, `Stationary/requirements.txt`).
   On Clementina we install 0.11.2; step 4 checks that this patch bump does not
   move the numbers (`05_cpu_tiny.sbatch` vs `03_tiny_run.sbatch`).
3. **The plugin is experimental** (JAX upstream labels Intel GPU support
   "experimental"; see the [JAX installation docs](https://docs.jax.dev/en/latest/installation.html#intel-gpu)).
   The plan is therefore staged: probe → environment → smoke test → tiny physics
   run → production, with a CPU baseline to fall back on. If the GPU path turns
   out to be unusable for a particular recipe, the same tiny run on 64 Xeon cores
   still produces correct results.

---

## 1. What is in this directory

| File | What it is | Where it runs |
|---|---|---|
| `env.sh` | shared settings: account, proxy, `/data` run root, Intel GPU env vars, interpreter lookup | sourced |
| `00_probe.sbatch` | step 1 — discovers drivers, modules, python, partitions, ZE mask, proxy | `testing`, 1 GPU, 10 min |
| `01_setup_env.sh` | step 2 — builds `~/venvs/pinn-oneapi` with `jax[oneapi]==0.11.2` | **login node** |
| `smoke.py` | step 4 — checks the JAX primitives the DSGNAR optimiser needs; names the one that fails | any node |
| `02_smoke.sbatch` | step 3 wrapper, inside a GPU allocation | `testing`, 1 GPU, 20 min |
| `03_tiny_run.sbatch` | step 4 — the real `wave_pinn` CLI, tiny, on one Intel GPU | `gpunode`, 40 min |
| `04_prod_run.sbatch` | step 5 — the Mendieta production recipe, one GPU | `gpunode`, 8 h (override) |
| `05_cpu_tiny.sbatch` | step 4b — same tiny run on CPU, for correctness + performance baseline | `cpunode`, 40 min |

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

---

## 4. Step 1 — probe the machine

```bash
cd ~/Julia/PINN/clementina
sbatch 00_probe.sbatch
tail -n +1 probe_*.out probe_*.err        # the testing queue is short; ~2 min
```

It prints, without touching anything: the GPU allocation and `ZE_*` variables,
`xpu-smi discovery/topology`, `sycl-ls` with and without Slurm's mask, where
`libze_loader`/`libsycl` live, the real `module avail`/`module spider` output,
which `python3` exists, `/home` vs `/data`, the real partition names and limits,
the `pci_143` association, and whether PyPI is reachable with and without the
proxy.

**Send back `probe_*.out` and `probe_*.err`.** The three answers that matter:

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

## 7. Step 4 — tiny end-to-end run (GPU), plus the CPU baseline

```bash
cd ~/Julia/PINN/clementina
sbatch 03_tiny_run.sbatch          # gpunode, one Intel GPU, ~40 min
sbatch 05_cpu_tiny.sbatch          # cpunode, 64 cores, ~40 min
```

Both run the same command — `python -m wave_pinn.cli`, `T=2`, 512 collocation
points, 10 DSGNAR iterations, sketch 256, float64 — with the same seed and only
the platform different. Outputs:

`/data/contrib/pci_143/pinn/Evolution_try/runs/cxxi_tiny/` and
`.../cxxi_tiny_cpu/`.

Check:

1. the GPU job's final loss and the CPU job's final loss agree to ~1e-8
   (`report.md` / `report.json` in each directory); bigger differences are a
   porting problem, not rounding;
2. the wall time ratio — this is the number that decides whether the production
   runs move to Clementina or stay on Mendieta;
3. if the GPU job dies, the traceback names the failing op, and `smoke.py`
   (step 3) narrows it down further.

---

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

Note on GPU visibility in production: the default is to leave Slurm's
`ZE_AFFINITY_MASK` alone, so a `--gpus=1` job gets what Slurm allocated. If the
probe shows only a 64 GB tile and the run needs the whole 128 GB card, the knob is
`CXXI_ZE_MODE=COMPOSITE` (see `env.sh`) — but read the comment there first: the
mask syntax differs between FLAT and COMPOSITE and a mask from one mode can be
invalid or point at another user's GPU in the other. Only `CXXI_UNSET_ZE_MASK=1`
when you own all four GPUs (`--gpus=4`).

---

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
| `sbatch: error: invalid account` | `pci_143` association | `sacctmgr show assoc user=$USER` in the probe output |
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
