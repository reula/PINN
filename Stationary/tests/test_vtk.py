"""The VTK export: the grid construction and the file format VisIt has to read.

`main()` needs a trained run, so what is tested here are the pieces that can go wrong
silently: the grading of the Cartesian grid, the choice of cells (only those inside the
shell), and the legacy VTK syntax -- written and then parsed back, with the array sizes
checked against the declared counts.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from stationary.vtk import half_axis, shell_cells, write_vtk


def test_grading_is_geometric_and_spans_the_shell():
    """Points per half axis run from the inner to the outer radius, denser inside."""
    c = np.asarray(half_axis(20, 1.0, 100.0))
    assert c[0] == -100.0 and c[-1] == 100.0
    assert np.all(np.diff(c) > 0)                      # strictly increasing
    assert np.any(c == 0.0)                            # one point at the origin
    pos = c[c > 0]
    assert abs(pos[0] - 1.0) < 1e-12                   # starts exactly at the inner radius
    d = np.diff(pos)
    assert d[0] < d[-1]                                # geometric: sparse in the far field
    # equal in log r, which is what "not equispaced" means here
    assert np.allclose(np.diff(np.log(pos)), np.diff(np.log(pos))[0], rtol=1e-12)


def test_only_cells_inside_the_shell_are_kept():
    coords = half_axis(8, 1.0, 100.0)
    cells = shell_cells(coords, 1.0, 100.0)
    n = len(coords)
    assert cells.shape[1] == 3
    # every cell centre is inside the shell
    mid = 0.5 * (np.asarray(coords)[:-1] + np.asarray(coords)[1:])
    for i, j, k in np.asarray(cells)[:200]:
        x, y, z = mid[i], mid[j], mid[k]
        r = np.sqrt(x * x + y * y + z * z)
        assert 1.0 <= r <= 100.0
    # and by count: a 8^3 grid cannot keep everything (the hole and the corners are out)
    assert 0 < cells.shape[0] < (n - 1) ** 3


def test_the_written_file_parses_back(tmp_path):
    """Parse what we wrote the way a reader would: counts first, then the data."""
    coords = half_axis(5, 1.0, 10.0)
    n = len(coords)
    cells = shell_cells(coords, 1.0, 10.0)
    corner = jnp.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    flat = cells[:, None, :] + corner[None, :, :]
    lin = flat[..., 0] * n * n + flat[..., 1] * n + flat[..., 2]
    used = jnp.unique(lin)
    index = jnp.full(n ** 3, -1, dtype=jnp.int32).at[used].set(
        jnp.arange(used.size, dtype=jnp.int32))
    hexes = index[lin]
    ijk = np.stack(np.unravel_index(np.asarray(used), (n, n, n)), axis=-1)
    pts = np.stack([np.asarray(coords)[ijk[:, 0]], np.asarray(coords)[ijk[:, 1]],
                    np.asarray(coords)[ijk[:, 2]]], axis=-1)
    vals = np.linspace(0.0, 1.0, len(pts))
    path = tmp_path / "t.vtk"
    write_vtk(str(path), pts, {"lambda": vals, "r_areal": vals * 2.0}, np.asarray(hexes))

    lines = path.read_text().splitlines()
    assert lines[0].startswith("# vtk DataFile")
    assert lines[2] == "ASCII" and lines[3] == "DATASET UNSTRUCTURED_GRID"
    n_pts = int(lines[4].split()[1])
    assert n_pts == len(pts) == used.size
    i = 5
    for _ in range(n_pts):                                     # POINTS block
        assert len(lines[i].split()) == 3
        i += 1
    n_cells, size = (int(v) for v in lines[i].split()[1:3])
    assert n_cells == len(hexes) and size == 9 * n_cells
    i += 1
    for _ in range(n_cells):
        toks = lines[i].split()
        assert toks[0] == "8" and len(toks) == 9
        assert all(0 <= int(t) < n_pts for t in toks[1:])      # indices inside the point list
        i += 1
    assert int(lines[i].split()[1]) == n_cells                 # CELL_TYPES
    i += 1
    assert all(lines[i + j] == "12" for j in range(n_cells))   # 12 = VTK_HEXAHEDRON
    i += n_cells
    assert int(lines[i].split()[1]) == n_pts                   # POINT_DATA
    i += 1
    for _ in range(2):                                         # two SCALARS blocks
        assert lines[i].startswith("SCALARS")
        assert lines[i + 1].startswith("LOOKUP_TABLE")
        i += 2
        assert len(lines) >= i + n_pts
        i += n_pts
    assert i == len(lines)                                     # nothing left over
