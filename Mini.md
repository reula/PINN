# Mac mini (headless compute box)

Apple M6, 16 GB, macOS 27. Set up 2026-11-04 and run with no monitor or keyboard.

## Reaching it

The mini and the laptop are on different internal networks at the university (10.164.0-7.x
and 10.164.8-15.x), so they cannot see each other directly. They are joined by **Tailscale**
(one account, installed on both). From anywhere:

    ssh reula@macmini-de-oscar          # or ssh reula@100.98.22.2
    open vnc://100.98.22.2              # Screen Sharing

Key login is installed (`ssh-copy-id`), so scripted `ssh -o BatchMode=yes` works. Anything
that needs `sudo` has to be run with `ssh -t` by the user, because a password is required.

## What is on it

| Item | Where |
|---|---|
| Homebrew 7.0.12 (`/opt/homebrew`), git, gh | `gh` is logged in as `reula` |
| Julia 1.13.3 via `juliaup` | no project in this repo uses it; the Julia code lives in Stationary/ |
| Python 3.13 (Homebrew) | the system Python is 3.9 and too old for JAX 0.11 |
| `Evolution_try/.venv` | JAX 0.11.1 **CPU**, scipy, optax, numpy, matplotlib (`requirements.txt`) |
| `Stationary/.venv` | JAX 0.11.1 CPU, flax 0.12.9, optax, scipy, numpy, matplotlib, tqdm; SBH reads this one |
| MacTeX (no GUI) | `latexmk -pdf paper.tex` in `Stationary/Paper` builds the 12-page paper |
| Repo | `~/Julia/PINN`, cloned from github.com/reula/PINN |

Verified: both venvs import and report `CpuDevice`; the paper builds from scratch; SBH `.py`
files parse. No run has been done on the mini yet.

## Settings that keep it reachable

`pmset`: `sleep 0`, `displaysleep 0`, `disksleep 0`, `autorestart 1` (starts again after a
power cut). **FileVault is off** so it boots and reconnects with nobody at the keyboard; the
cost is an unencrypted disk. Tailscale starts at login and automatic login is on. A macOS
update that restarts is therefore safe, but check `tailscale status` afterwards.

## Limits

- **No GPU.** JAX runs on CPU only (the M6 GPU is not used: no `jax-metal`, and the code
  assumes x64, which Metal lacks). Anything that needs the A30 or the Intel Max nodes still
  goes to the hub or Mendieta as described in SBH/README.md and Mendieta.md.
- Results from the mini run in CPU mode; SBH/README.md notes that CPU and GPU results differ
  slightly, so grade them against a baseline made in the same mode.
- Never run `softwareupdate -i` for macOS itself without being able to reach the mini
  afterwards; the Command Line Tools update is separate and was installed with
  `sudo softwareupdate -i "Command Line Tools for Xcode 27.2-27.2"`.
