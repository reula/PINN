"""Finalise ``ldc_reference.npz``: extrapolate the two moving-lid corners, recentre p.

Why this exists
---------------
``make_ldc_reference.py`` fits the pressure gradient ``grad p = -(u.grad)u + lap(u)/Re``
at the *interior* nodes and pins the remaining freedom with a weak Laplace penalty
(``press_eps``).  Nothing constrains ``p`` *at* the two top corners (x, y) = (0, 1) and
(1, 1): the gradient fit has no row there and the penalty leaves that null direction
untouched, so the least-squares solve parks the whole null-space component on those two
nodes (measured: -670.27 and +676.10, while every neighbour is |p| < 0.5).  The interior
field is good -- its gradient matches the momentum right-hand side to 0.6-2.6 % RMS,
against ~6 % for the spectral Neumann solve -- so the fix is local: replace a corner value
that is orders of magnitude above the field by the average of its two nearest neighbours
along x and y, then recentre p to zero mean.

Pressure at a lid corner is not determined by the data anyway (the Neumann problem fixes
p only up to a constant, and the corner is where the lid's tangential velocity is
discontinuous in slope), so extrapolating the corner is a statement of convention, not a
change of the solution.

Usage:  python finalize_reference.py [path]        # in place; writes <path>.raw once
"""
from __future__ import annotations

import os
import sys
import numpy as np

SPIKE = 50.0        # a node this many times the interior RMS is the degenerate corner


def fix_corners(p: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    p = p.copy()
    interior_rms = float(np.sqrt(np.mean(p[2:-2, 2:-2] ** 2)))
    limit = SPIKE * max(interior_rms, 1e-12)
    fixed = []
    for i, j in np.argwhere(np.abs(p) > limit):
        neighbours = [p[i - 1, j] if i > 0 else None,
                      p[i + 1, j] if i + 1 < p.shape[0] else None,
                      p[i, j - 1] if j > 0 else None,
                      p[i, j + 1] if j + 1 < p.shape[1] else None]
        vals = [v for v in neighbours if v is not None and abs(v) <= limit]
        new = float(np.mean(vals))
        fixed.append((int(i), int(j), float(p[i, j]), new))
        p[i, j] = new
    return p, fixed


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    path = argv[0] if argv else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "ldc_reference.npz")
    z = dict(np.load(path))                       # the generator's raw output
    p, fixed = fix_corners(z["plot_p"])
    if not fixed:
        print(f"{path}: nothing to fix (no node above {SPIKE:g} x the interior RMS)")
        return 0

    raw = path + ".raw"
    if not os.path.exists(raw):
        np.savez(raw, **z)
        print(f"kept the unmodified generator output at {raw}")

    p = p - p.mean()
    z["plot_p"] = p
    z["solution"] = z["solution"].copy()
    z["solution"][:, [str(c) for c in z["channels"]].index("p")] = p.ravel()
    np.savez(path, **z)
    for i, j, old, new in fixed:
        print(f"  corner node ({i}, {j}): {old:+.4f} -> {new:+.4f}")
    print(f"{path}: pressure recentred to zero mean; "
          f"range now [{p.min():.4f}, {p.max():.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
