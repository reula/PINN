"""VTK export of a run for VisIt: a shell-conforming grid, with the error in lambda.

    python -m stationary.vtk --outdir runs/<name>            # writes runs/<name>/vtk/*.vtk
    python -m stationary.vtk --outdir runs/<name> --n-rho 40 --n-theta 32 --n-phi 64
    python -m stationary.vtk --outdir runs/<name> --grid cartesian --n-half 30 --lambda-only

Called automatically by postprocess.sh when the run's config has `make_vtk` true
(`./run_hub.sh ... --vtk`), so a run that asks for it produces the files when it ends.

What it writes
--------------
Legacy ASCII VTK, one `UNSTRUCTURED_GRID` per call, in the PHYSICAL coordinates
(`--physical-inner`, default 1.0; the run itself lives in the scaled shell, so the factor is
physical_inner/rho_in).  Two grids:

* `--grid spherical` (the default): conforming to the two spheres.  Nodes sit on radial
  levels rho = rho_in (rho_out/rho_in)^(i/n_rho) -- geometric, so both spheres are hit
  exactly and the cells grow by a constant factor outwards -- times uniform angles, theta in
  [0, pi] and phi periodic in [0, 2pi).  Every cell is inside the shell BY CONSTRUCTION, so
  there is nothing to blank out, and the inner sphere is resolved the same amount in EVERY
  direction.  That last property is exactly what the Cartesian grid cannot give: grading each
  half axis geometrically clusters points near +-rho_in on the three AXES only, leaving the
  rest of the inner sphere comparatively bare -- and the inner sphere is where the field
  varies fastest.  The polar rings are wedges (VTK 13), not degenerate hexahedra.
* `--grid cartesian`: the older graded box (`--n-half` points per half axis, geometric),
  keeping only the cells whose centre is inside the shell.  Roughly half its cells are
  discarded and the survivors straddle the spheres; kept for comparison.

Point data:

    lambda_err      lambda(computed) - lambda(exact): the error the run is judged by
    lambda_exact    the exact reference's lambda at the same point
    h_err           max_ij |h(computed) - h(exact)|, the metric error in one number
    lambda          the field itself
    lambda_minus_1  the decaying part, which is what the figures are about
    r_areal         areal radius of the sphere through the point (from the induced metric)
    ricci_scalar    R of h
    ricci_sq        R_ab R^ab -- in three dimensions the Weyl tensor vanishes, so this is
                    the full curvature information beyond the scalar
    res_ricci       |Ricci(h)_ab - (1/2 lambda^2) d_a lambda d_b lambda| (quality map)
    res_lam_eq      |box lambda|  (quality map)

The error fields need the exact reference, taken from the single definition the run itself
was built on (`train.exact_asset`); without one they are skipped with a note.

`--lambda-only` writes just the cheap fields, no second derivatives of the network, which is
what makes a large grid affordable: the curvature and quality maps are the expensive part and
on a slow CPU they dominate the cost.

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


def spherical_grid(n_rho: int, n_theta: int, n_phi: int, r_in: float, r_out: float):
    """Nodes of a shell-conforming (rho, theta, phi) grid, with its cells and VTK types.

    Every cell lies inside the shell by construction, the poles are shared and phi is
    periodic, so no node is duplicated and nothing needs blanking.  See the module
    docstring for why the grading is geometric in rho and uniform in angle.
    """
    rho = r_in * (r_out / r_in) ** jnp.linspace(0.0, 1.0, n_rho + 1)
    th = jnp.linspace(0.0, jnp.pi, n_theta + 1)
    ph = 2.0 * jnp.pi * jnp.arange(n_phi) / n_phi
    # one radial level: the north pole, the rings at theta_1 .. theta_{n_theta-1}, the pole
    st = jnp.sin(th)[1:-1, None]
    ct = jnp.cos(th)[1:-1, None]
    dirs = jnp.stack([st * jnp.cos(ph)[None, :],
                      st * jnp.sin(ph)[None, :],
                      jnp.broadcast_to(ct, (n_theta - 1, n_phi))], axis=-1)
    n_ring = (n_theta - 1) * n_phi
    per_level = n_ring + 2
    level = jnp.concatenate([jnp.array([[0.0, 0.0, 1.0]]), dirs.reshape(-1, 3),
                             jnp.array([[0.0, 0.0, -1.0]])], axis=0)
    nodes = (rho[:, None, None] * level[None, :, :]).reshape(-1, 3)

    def nid(i, j, k):
        base = i * per_level
        if j == 0:
            return base                                    # north pole
        if j == n_theta:
            return base + n_ring + 1                       # south pole
        return base + 1 + (j - 1) * n_phi + (k % n_phi)

    cells, types = [], []
    for i in range(n_rho):
        for j in range(n_theta):
            for k in range(n_phi):
                k1 = (k + 1) % n_phi
                if j == 0:                                 # north cap: wedge at theta = 0
                    cells.append([nid(i, 0, 0), nid(i, 1, k), nid(i, 1, k1),
                                  nid(i + 1, 0, 0), nid(i + 1, 1, k), nid(i + 1, 1, k1)])
                    types.append(13)
                elif j == n_theta - 1:                     # south cap, reversed winding
                    cells.append([nid(i, n_theta, 0), nid(i, n_theta - 1, k1),
                                  nid(i, n_theta - 1, k),
                                  nid(i + 1, n_theta, 0), nid(i + 1, n_theta - 1, k1),
                                  nid(i + 1, n_theta - 1, k)])
                    types.append(13)
                else:                                      # hexahedron
                    cells.append([nid(i, j, k), nid(i, j, k1), nid(i, j + 1, k1),
                                  nid(i, j + 1, k),
                                  nid(i + 1, j, k), nid(i + 1, j, k1),
                                  nid(i + 1, j + 1, k1), nid(i + 1, j + 1, k)])
                    types.append(12)
    # cells must stay a plain list: wedges have 6 nodes and hexahedra 8, so it is ragged
    # and jnp.asarray would reject it.  write_vtk handles both by writing the node count.
    return nodes, cells, jnp.asarray(types, dtype=jnp.int32)


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


def error_at(pf, ref, xs):
    """Where the run is wrong: lambda_computed - lambda_exact, and the metric error.

    This is the field the exercise is about, so it is written first and named plainly.  It
    needs the exact reference the boundary data came from (train.exact_asset), not a
    re-derived one.
    """
    def one(x):
        f, e = pf(x), ref(x)
        return f.lam - e.lam, jnp.max(jnp.abs(f.h - e.h)), e.lam

    dl, dh, lam_ex = jax.vmap(one)(xs)
    return {"lambda_err": dl, "h_err": dh, "lambda_exact": lam_ex}


# ------------------------------------------------------------------ the writer
def write_vtk(path: str, points, arrays: dict, cells, cell_types):
    """Legacy ASCII VTK: an unstructured grid with scalar point data.

    Mixed cell types are supported, which is what lets the spherical grid use wedges at the
    poles instead of a hexahedron with two coincident nodes (VisIt accepts those, but some
    filters do not).
    """
    n_pts = points.shape[0]
    n_cells = len(cells)
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\n")
        fh.write("stationary Einstein solution: lambda and invariants (VisIt)\n")
        fh.write("ASCII\n")
        fh.write("DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {n_pts} double\n")
        for p in points:
            fh.write(f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
        # each cell line is its node count followed by the node indices
        fh.write(f"CELLS {n_cells} {int(sum(len(c) + 1 for c in cells))}\n")
        for c in cells:
            fh.write(f"{len(c)} " + " ".join(str(int(i)) for i in c) + "\n")
        fh.write(f"CELL_TYPES {n_cells}\n")
        for t in cell_types:
            fh.write(f"{int(t)}\n")   # 12 hexahedron, 13 wedge
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
                   help="only the cheap fields (no second derivatives)")
    p.add_argument("--grid", choices=("spherical", "cartesian"), default="spherical",
                   help="spherical: conforming to the two spheres, isotropic resolution on "
                        "the inner sphere (default); cartesian: the older graded box")
    p.add_argument("--n-rho", type=int, default=None,
                   help="spherical grid: radial levels (geometric, both spheres hit exactly)")
    p.add_argument("--n-theta", type=int, default=None, help="spherical grid: polar cells")
    p.add_argument("--n-phi", type=int, default=None, help="spherical grid: azimuthal cells")
    p.add_argument("--no-error", action="store_true",
                   help="skip lambda_err / h_err / lambda_exact (no reference needed)")
    p.add_argument("--subdir", default="vtk")
    a = p.parse_args()
    run_dir = a.outdir or a.run_dir
    if run_dir is None:
        p.error("give the run directory, as `--outdir RUN` or as the first argument")

    cfg, model, state = load_run(run_dir, a.params_file)
    pf = point_fields(model, state["net"])
    n_half = int(a.n_half or cfg.vtk_n_half)
    r_in = float(a.physical_inner or cfg.vtk_physical_inner)
    factor = r_in / cfg.rho_in
    r_out = cfg.rho_out * factor
    # the run lives in the scaled shell [rho_in, rho_out]; the physical inner radius is
    # whatever the user asked the file to be written in (1.0 by default)
    if a.grid == "spherical":
        n_rho = int(a.n_rho or 36)
        n_theta = int(a.n_theta or 24)
        n_phi = int(a.n_phi or 48)
        xs_phys, cells, cell_types = spherical_grid(n_rho, n_theta, n_phi, r_in, r_out)
        print(f"[vtk] conforming shell grid {n_rho} x {n_theta} x {n_phi} in "
              f"[{r_in:g}, {r_out:g}]: geometric in rho, uniform in theta and phi, every "
              f"cell inside the shell; factor {factor:g} from rho_in {cfg.rho_in:g}")
        sizes = {"n_rho": n_rho, "n_theta": n_theta, "n_phi": n_phi}
    else:
        coords = half_axis(n_half, r_in, r_out)
        print(f"[vtk] cartesian grid {len(coords)}^3 in [{r_in:g}, {r_out:g}] (geometric, "
              f"cell centres in the shell kept); factor {factor:g} from rho_in "
              f"{cfg.rho_in:g}")
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
        used = jnp.unique(lin)                   # which grid points a kept cell touches
        # NB: jnp.full, not -jnp.ones(...).at[...].set(...) -- the latter negates the RESULT
        # of the update (unary minus binds after the method call), leaving negatives.
        index = jnp.full(n ** 3, -1, dtype=jnp.int32).at[used].set(
            jnp.arange(used.size, dtype=jnp.int32))
        hexes = index[lin]                       # (n_cells, 8) compact indices
        if bool(jnp.any(hexes < 0)):
            raise SystemExit("[vtk] internal error: a cell corner was not registered")
        # physical coordinates of the used points, in the order they were indexed
        ijk = jnp.stack(jnp.unravel_index(used, (n, n, n)), axis=-1)
        xs_phys = jnp.stack([coords[ijk[:, 0]], coords[ijk[:, 1]], coords[ijk[:, 2]]],
                            axis=-1)
        cells = hexes
        cell_types = jnp.full(cells.shape[0], 12, dtype=jnp.int32)
        sizes = {"n_half": n_half}
    xs_chart = xs_phys / factor
    print(f"[vtk] {xs_phys.shape[0]} points, {len(cells)} cells")

    # the error comes first: it is what the file is for
    data = {}
    if not a.no_error:
        from .train import exact_asset
        ref = exact_asset(cfg)
        if ref is None:
            print("[vtk] no exact reference for this run: skipping lambda_err / h_err")
        else:
            data.update(error_at(pf, ref, xs_chart))
            print(f"[vtk] lambda_err: max |lambda_net - lambda_exact| = "
                  f"{float(jnp.max(jnp.abs(data['lambda_err']))):.3e}, "
                  f"rms {float(jnp.sqrt(jnp.mean(data['lambda_err'] ** 2))):.3e}")
            nl = int((~jnp.isfinite(data["lambda_err"])).sum())
            if nl:
                print(f"[vtk] WARNING: lambda_err is non-finite at {nl} nodes")
            print(f"[vtk] h_err: max {float(jnp.max(jnp.abs(data['h_err']))):.3e}")
    data.update(fields_at(pf, xs_chart, want_all=not a.lambda_only))

    # Nothing below should be non-finite (weyl.k_of now returns the exact axis limit rather
    # than inf).  This guard is not a blanking mechanism but a loud one: VisIt stops reading
    # an ASCII VTK file at the first nan/inf, so a single bad value silently truncates the
    # variable list -- which is precisely how a ten-field file came back as a two-field one.
    for _name, _vals in list(data.items()):
        _arr = jnp.asarray(_vals)
        _bad = ~jnp.isfinite(_arr)
        if bool(jnp.any(_bad)):
            print(f"[vtk] WARNING: {_name}: {int(_bad.sum())} of {_bad.size} values are "
                  f"non-finite and are written as 0 -- VisIt would otherwise stop reading "
                  f"the variables after this one")
            data[_name] = jnp.where(_bad, 0.0, _arr)
    out_dir = os.path.join(run_dir, a.subdir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "solution.vtk")
    write_vtk(path, xs_phys, data, cells, cell_types)
    size = os.path.getsize(path) / 2**20
    print(f"[vtk] wrote {path}  ({size:.1f} MB, fields: {', '.join(data)})")
    with open(os.path.join(out_dir, "grid.json"), "w") as fh:
        json.dump({"grid": a.grid, "physical_inner": r_in, "physical_outer": r_out,
                   **sizes, "n_points": int(xs_phys.shape[0]),
                   "n_cells": int(len(cells)), "fields": list(data)}, fh, indent=2)


if __name__ == "__main__":
    main()
