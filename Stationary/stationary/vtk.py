"""VTK export of a run for VisIt: a graded Cartesian grid of the shell.

    python -m stationary.vtk --outdir runs/<name>            # writes runs/<name>/vtk/*.vtk
    python -m stationary.vtk --outdir runs/<name> --n-half 30 --lambda-only

Called automatically by postprocess.sh when the run's config has `make_vtk` true
(`./run_hub.sh ... --vtk`), so a run that asks for it produces the files when it ends.

What it writes
--------------
Legacy ASCII VTK, one `UNSTRUCTURED_GRID` of hexahedra per call, in the PHYSICAL
coordinates (`--physical-inner`, default 1.0; the run itself lives in the scaled shell
`[0.01, 1]`, so the factor is 1/rho_in = 100).  Cartesian and NOT equispaced: along each
half axis the points are geometric from the inner radius to the outer one, so they cluster
where the solution varies fastest -- near the inner sphere, where the action is -- and thin
out in the far field.

Only the cells whose centre lies inside the shell are emitted, so the hole and the exterior
cost nothing: on a 41-per-axis box that is roughly half the cells.  Point data:

    lambda          the field itself
    lambda_minus_1  the decaying part, which is what the figures are about
    r_areal         areal radius of the sphere through the point (from the induced metric)
    ricci_scalar    R of h
    ricci_sq        R_ab R^ab -- in three dimensions the Weyl tensor vanishes, so this is
                    the full curvature information beyond the scalar
    res_ricci       |Ricci(h)_ab - (1/2 lambda^2) d_a lambda d_b lambda| (quality map)
    res_lam_eq      |box lambda|  (quality map)

`--lambda-only` writes just the first two, which is ~4x cheaper: the other fields need
second derivatives of the network, and on a slow CPU that dominates the cost.

The files are large (megabytes) and regenerable, so `runs/*/vtk/` is in .gitignore.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from .evaluate import load_run
from .geometry import residuals_at, ricci_from_gamma
from .model import point_fields


# ------------------------------------------------------------------ the grid
def half_axis(n_half: int, r_in: float, r_out: float):
    """Geometric points on one half axis, from r_in out to r_out, plus 0 in the middle.

    Geometric (equally spaced in log r) is the natural grading here: the shell spans two
    decades and the solution's radial variation is itself close to a power law.
    """
    t = jnp.linspace(0.0, 1.0, n_half)
    pos = r_in * (r_out / r_in) ** t
    # keep BOTH ends: the origin sits between the two half axes, and the innermost point of
    # each must be exactly r_in (the earlier `pos[1:]` dropped it, so the grid started at
    # r_in * (r_out/r_in)^(1/(n-1)) and never touched the inner sphere, where the action is)
    return jnp.concatenate([-pos[::-1], jnp.zeros(1), pos])


def shell_cells(coords, r_in: float, r_out: float):
    """Indices (i,j,k) of the cells whose centre lies inside the shell."""
    mid = 0.5 * (coords[:-1] + coords[1:])                    # cell centres per axis
    X, Y, Z = jnp.meshgrid(mid, mid, mid, indexing="ij")
    r = jnp.sqrt(X**2 + Y**2 + Z**2)
    inside = (r >= r_in) & (r <= r_out)
    return jnp.argwhere(inside)                        # (n_cells, 3) of (i, j, k)


# ------------------------------------------------------------------ the fields
def fields_at(pf, xs, want_all: bool = True):
    """Point data at the grid points xs (an (n,3) array)."""
    lam = jax.vmap(lambda x: pf(x).lam)(xs)
    out = {"lambda": lam, "lambda_minus_1": lam - 1.0}

    h = jax.vmap(lambda x: pf(x).h)(xs)
    nn = xs / jnp.linalg.norm(xs, axis=-1, keepdims=True)
    hrr = jnp.einsum("nij,ni,nj->n", h, nn, nn)
    alpha = (jnp.trace(h, axis1=-2, axis2=-1) - hrr) / 2.0
    out["r_areal"] = jnp.linalg.norm(xs, axis=-1) * jnp.sqrt(alpha)

    if not want_all:
        return out

    def one(x):
        f = pf(x)
        _, dG, _ = jax.jacfwd(pf)(x)
        hinv = jnp.linalg.inv(f.h)
        ric = ricci_from_gamma(f.G, dG)
        res = residuals_at(pf, x)          # the same residual groups the loss uses
        return (jnp.trace(hinv @ ric),                             # Ricci scalar of h
                # R_ab R^ab: in three dimensions the Weyl tensor vanishes, so this carries
                # the rest of the curvature information
                jnp.trace(ric @ hinv @ ric @ hinv),
                jnp.max(jnp.abs(res["ricci"])),                    # quality maps
                jnp.abs(res["lam_eq"]))

    R, R2, res_r, res_l = jax.vmap(one)(xs)
    out["ricci_scalar"] = R
    out["ricci_sq"] = R2
    out["res_ricci"] = res_r
    out["res_lam_eq"] = res_l
    return out


# ------------------------------------------------------------------ the writer
def write_vtk(path: str, points, arrays: dict, cells):
    """Legacy ASCII VTK: an unstructured grid of hexahedra with scalar point data."""
    n_pts = points.shape[0]
    n_cells = cells.shape[0]
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\n")
        fh.write("stationary Einstein solution: lambda and invariants (VisIt)\n")
        fh.write("ASCII\n")
        fh.write("DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {n_pts} double\n")
        for p in points:
            fh.write(f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
        # hexahedra: 8 corners each, so each cell line has 9 entries
        fh.write(f"CELLS {n_cells} {9 * n_cells}\n")
        for c in cells:
            fh.write("8 " + " ".join(str(int(i)) for i in c) + "\n")
        fh.write(f"CELL_TYPES {n_cells}\n")
        fh.write("12\n" * n_cells)                                   # 12 = VTK_HEXAHEDRON
        fh.write(f"POINT_DATA {n_pts}\n")
        for name, values in arrays.items():
            fh.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
            for v in values:
                fh.write(f"{float(v):.9g}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--params-file", default="params.pkl")
    p.add_argument("--n-half", type=int, default=None,
                   help="override cfg.vtk_n_half (points per half axis)")
    p.add_argument("--physical-inner", type=float, default=None,
                   help="override cfg.vtk_physical_inner")
    p.add_argument("--lambda-only", action="store_true",
                   help="only lambda and lambda-1 (cheap: no second derivatives)")
    p.add_argument("--subdir", default="vtk")
    a = p.parse_args()
    run_dir = a.outdir or a.run_dir
    if run_dir is None:
        p.error("give the run directory, as `--outdir RUN` or as the first argument")

    cfg, model, state = load_run(run_dir, a.params_file)
    pf = point_fields(model, state["net"])
    n_half = int(a.n_half or cfg.vtk_n_half)
    r_in = float(a.physical_inner or cfg.vtk_physical_inner)
    # the run lives in the scaled shell [rho_in, rho_out]; the physical inner radius is
    # whatever the user asked the file to be written in (1.0 by default)
    factor = r_in / cfg.rho_in
    r_out = cfg.rho_out * factor
    coords = half_axis(n_half, r_in, r_out)
    print(f"[vtk] grid {len(coords)}^3 in [{r_in:g}, {r_out:g}] (geometric, cell centres in "
          f"the shell kept); factor {factor:g} from the run's rho_in {cfg.rho_in:g}")

    # keep the grid points that belong to a kept cell
    cells_ijk = shell_cells(coords, r_in, r_out)
    if cells_ijk.shape[0] == 0:
        raise SystemExit("[vtk] no cells inside the shell -- check --physical-inner")
    n = len(coords)
    corner = jnp.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    # (n_cells, 8, 3) corner indices -> (n_cells, 8) linear indices in the full n^3 grid
    flat = cells_ijk[:, None, :] + corner[None, :, :]
    lin = flat[..., 0] * n * n + flat[..., 1] * n + flat[..., 2]
    used = jnp.unique(lin)                       # which grid points a kept cell touches
    # NB: jnp.full, not -jnp.ones(...).at[...].set(...) -- the latter negates the RESULT of
    # the update (unary minus binds after the method call), leaving every index negative.
    index = jnp.full(n ** 3, -1, dtype=jnp.int32).at[used].set(
        jnp.arange(used.size, dtype=jnp.int32))
    hexes = index[lin]                           # (n_cells, 8) compact indices
    if bool(jnp.any(hexes < 0)):
        raise SystemExit("[vtk] internal error: a cell corner was not registered")

    # physical coordinates of the used points, in the order they were indexed
    ijk = jnp.stack(jnp.unravel_index(used, (n, n, n)), axis=-1)
    xs_phys = jnp.stack([coords[ijk[:, 0]], coords[ijk[:, 1]], coords[ijk[:, 2]]], axis=-1)
    xs_chart = xs_phys / factor
    print(f"[vtk] {used.size} points, {cells_ijk.shape[0]} hexahedra")

    data = fields_at(pf, xs_chart, want_all=not a.lambda_only)
    out_dir = os.path.join(run_dir, a.subdir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "solution.vtk")
    write_vtk(path, xs_phys, data, hexes)
    size = os.path.getsize(path) / 2**20
    print(f"[vtk] wrote {path}  ({size:.1f} MB, fields: {', '.join(data)})")
    with open(os.path.join(out_dir, "grid.json"), "w") as fh:
        json.dump({"n_half": n_half, "physical_inner": r_in, "physical_outer": r_out,
                   "n_points": int(used.size), "n_cells": int(cells_ijk.shape[0]),
                   "fields": list(data)}, fh, indent=2)


if __name__ == "__main__":
    main()
