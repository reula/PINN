# Vendored: the DSGNAR reference implementation

Source: https://github.com/wephy/physics-informed-neural-networks

    J. Webb, S. Jerad, C. Cartis,
    "An Optimisation Framework for the Well-Conditioned Training of Physics-Informed
     Neural Networks", arXiv:2607.02194.

Copied from `src/pinn/` at the commit fetched on 2026-10-04, plus `requirements.txt` and
`pyproject.toml`.  Used rather than reimplemented because a from-scratch transcription in
`evolution/dsgnar.py` fixed four real bugs and still could not solve a LINEAR least-squares
problem, where a Gauss-Newton step is exact and one iteration suffices.

Additional dependencies beyond `Stationary/requirements.txt`: `equinox`, `jaxtyping`,
`ipython`, `ipywidgets` (the last two only because `plotting.py` imports them at prime time).

## Local patches

1. **`pinn/trainer64.py`** -- each `results["slabs"]` entry now also stores `"coords"`, the
   evaluation coordinates its `"u_pred"` was computed at.  Without it a downstream plot has to
   guess where each value came from; with it, an exact `(x, t)` snapshot is possible.  Purely
   additive: nothing reads the extra key except our plotting.

No other file is modified.
