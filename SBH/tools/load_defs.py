"""Load every definition of EEccz4_BH_puncture.py WITHOUT training.

The script has no main(): parsing, set-up and function definitions run at
module level and training starts at the first top-level
`if resume_checkpoint is not None:`. This executes the source up to that
line in a fresh namespace and returns it, so benchmarks and tests can call
r_ccz4_evolution, compute_bfgs_loss, ... directly.

    from load_defs import load
    ns = load(["--HIDDEN", "40", "--N_LAYERS", "6"])
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "EEccz4_BH_puncture.py")
STOP = "if resume_checkpoint is not None:"


def load(argv=(), script=SCRIPT, outdir=None):
    src = open(script).read()
    cut = src.index("\n" + STOP + "\n")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    argv = list(argv)
    if outdir is not None and "--outdir" not in argv and "--outdir" in src:
        argv += ["--outdir", outdir]
    old = sys.argv
    sys.argv = [script] + argv
    try:
        ns = {"__name__": "bh_defs", "__file__": script}
        exec(compile(src[:cut], script, "exec"), ns)
    finally:
        sys.argv = old
    return ns
