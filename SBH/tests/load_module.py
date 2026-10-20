"""Shim for the collaborators' reference_1d tools (compare_pinn.py, BH_invariants.py, ...):
their `load_module.load(argv)` returns the BH script's namespace, as tools/load_defs.load does."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from load_defs import load as _load


def load(argv):
    return _load(list(argv), outdir=tempfile.mkdtemp(prefix="refcmp_"))
