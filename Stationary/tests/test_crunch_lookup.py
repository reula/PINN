"""Where the quasi-Newton optimiser is looked for.

Crunch -- the JAX fork of `jax.scipy.optimize` that adds the self-scaling Broyden (`ssbroyden2`,
a translation of Optim.jl's `SSBroyden`) -- is tracked in this repo as `Jax/Crunch`, so
`<repo>/Jax` is searched first.  But a machine that carries only `Stationary/` (the hub is
synced with `rsync ... Stationary/ hub:.../Stationary/`) has no `<repo>/Jax` at all, so a copy
inside `Stationary/`, in the repo root or above the repo has to be found too.  These tests pin
the search order, that synthetic layouts import, and that one unusable candidate does not stop
the search -- the failure mode that needs care is a half-imported `Crunch` left in
`sys.modules`, which the next candidate's `import Crunch` would happily return.
"""
from __future__ import annotations

import os
import sys

import pytest

from stationary import train as T

LINE_SEARCH = "def backtracking(*a, **k):\n    return None\n\n\nclass BacktrackingResult:\n    pass\n"
MINIMIZE = ("from line_search_backtracking import backtracking   # the fork's top-level import\n"
            "\n\n"
            "def minimize(*a, **k):\n"
            "    return 'ok'\n")
BROKEN = "raise ImportError('this copy of Crunch is incomplete')\n"


def _make_crunch(root, minimize_src=MINIMIZE):
    """A minimal tree that satisfies what the fork's import chain needs."""
    os.makedirs(os.path.join(root, "Crunch", "Optimizers"), exist_ok=True)
    open(os.path.join(root, "line_search_backtracking.py"), "w").write(LINE_SEARCH)
    open(os.path.join(root, "Crunch", "__init__.py"), "w").write("")
    open(os.path.join(root, "Crunch", "Optimizers", "__init__.py"), "w").write("")
    open(os.path.join(root, "Crunch", "Optimizers", "minimize_backtracking.py"), "w").write(
        minimize_src)
    return root


@pytest.fixture(autouse=True)
def _clean_import_state():
    """`_crunch_minimize` appends to sys.path and imports a package called Crunch: undo both,
    or one test's synthetic tree is what the next one imports."""
    saved_path = list(sys.path)
    saved = {m: sys.modules[m] for m in list(sys.modules)
             if m in ("Crunch", "line_search_backtracking") or m.startswith("Crunch.")}
    for m in saved:
        sys.modules.pop(m, None)
    yield
    sys.path[:] = saved_path
    for m in [m for m in list(sys.modules)
              if m in ("Crunch", "line_search_backtracking") or m.startswith("Crunch.")]:
        sys.modules.pop(m, None)
    sys.modules.update(saved)


def test_candidates_put_the_tracked_location_first_and_include_stationary_and_above(tmp_path):
    repo = tmp_path / "PINN"
    station = repo / "Stationary"
    station.mkdir(parents=True)
    cands = T._crunch_candidates(str(station))
    assert cands[0] == str(repo / "Jax")                     # tracked here: unchanged behaviour
    assert str(station / "Jax") in cands                     # inside Stationary
    assert str(station) in cands                             # Stationary itself
    assert str(repo) in cands                                # the repo root
    assert str(tmp_path / "Jax") in cands and str(tmp_path) in cands   # above the repo
    assert len(cands) == len(set(cands))


def test_a_copy_inside_stationary_is_found(tmp_path, monkeypatch):
    """The case the hub needs: only Stationary/, with Crunch inside it."""
    repo = tmp_path / "PINN"
    station = repo / "Stationary"
    (station / "stationary").mkdir(parents=True)
    _make_crunch(str(station))
    monkeypatch.delenv("CRUNCH_ROOT", raising=False)
    monkeypatch.setattr(T, "__file__", str(station / "stationary" / "train.py"))
    minimize, where = T._crunch_minimize()
    assert minimize is not None and minimize() == "ok"
    assert where == str(station)


def test_an_unusable_candidate_does_not_stop_the_search(tmp_path, monkeypatch):
    """`<repo>/Jax` holds an incomplete Crunch; the good copy in Stationary/ must still win."""
    repo = tmp_path / "PINN"
    station = repo / "Stationary"
    (station / "stationary").mkdir(parents=True)
    _make_crunch(str(repo / "Jax"), minimize_src=BROKEN)
    _make_crunch(str(station))
    monkeypatch.delenv("CRUNCH_ROOT", raising=False)
    monkeypatch.setattr(T, "__file__", str(station / "stationary" / "train.py"))
    minimize, where = T._crunch_minimize()
    assert minimize is not None
    assert where == str(station)
    # the half-built Crunch the first candidate left behind is gone: nothing in sys.modules
    # points into <repo>/Jax, so a later `import Crunch` cannot be poisoned by it
    stale = [m for m in sys.modules
             if (m == "Crunch" or m.startswith("Crunch.") or m == "line_search_backtracking")
             and str(repo / "Jax") in str(getattr(sys.modules[m], "__file__", ""))]
    assert stale == []


def test_crunch_root_is_authoritative(tmp_path, monkeypatch):
    """An explicit CRUNCH_ROOT is the ONLY candidate: a typo is reported, not worked around."""
    repo = tmp_path / "PINN"
    station = repo / "Stationary"
    (station / "stationary").mkdir(parents=True)
    _make_crunch(str(station))                                # a good copy that must be ignored
    monkeypatch.setattr(T, "__file__", str(station / "stationary" / "train.py"))
    monkeypatch.setenv("CRUNCH_ROOT", str(tmp_path / "nowhere"))
    minimize, reason = T._crunch_minimize()
    assert minimize is None
    assert "nowhere" in reason and "no Crunch/Optimizers" in reason
    # and the real thing resolves when the override is dropped
    monkeypatch.delenv("CRUNCH_ROOT")
    minimize, where = T._crunch_minimize()
    assert minimize is not None and where == str(station)


def test_the_real_checkout_is_found_without_an_override(monkeypatch):
    """In this working tree the tracked Jax/Crunch is what resolves (and it imports)."""
    monkeypatch.delenv("CRUNCH_ROOT", raising=False)
    minimize, where = T._crunch_minimize()
    if minimize is None:                                      # e.g. a Stationary-only checkout
        pytest.skip(f"no Crunch in this checkout: {where}")
    assert os.path.isdir(os.path.join(where, "Crunch", "Optimizers"))
    assert callable(minimize)


# ------------------------------------------------- the copy that travels with Stationary/
MIN_PATH = ("Crunch/Optimizers/__init__.py", "Crunch/Optimizers/bfgs.py",
            "Crunch/Optimizers/bfgs_backtracking.py", "Crunch/Optimizers/minimize.py",
            "Crunch/Optimizers/minimize_backtracking.py", "Crunch/__init__.py",
            "line_search_backtracking.py")


def test_the_minimal_copy_inside_stationary_is_complete_and_imports():
    """`Stationary/Jax` carries the minimize path, for a machine that syncs only Stationary/.

    Imported through the explicit root, so this exercises the COPY (the search itself prefers
    <repo>/Jax when it exists).
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # <repo>/Stationary
    root = os.path.join(here, "Jax")
    if not os.path.isdir(os.path.join(root, "Crunch", "Optimizers")):
        pytest.skip("this checkout has no Stationary/Jax copy")
    for rel in MIN_PATH:
        assert os.path.exists(os.path.join(root, rel)), rel
    minimize, where = T._crunch_minimize(root=root)
    assert minimize is not None, where
    assert where == os.path.normpath(root)


def test_the_two_copies_do_not_drift():
    """The duplication is deliberate but must not rot: identical bytes, or this fails.

    `Jax/` is the canonical, tracked fork; `Stationary/Jax/` is the subset that travels with a
    Stationary-only sync.  Re-copy with
        cp Jax/Crunch/Optimizers/{...}.py Stationary/Jax/Crunch/Optimizers/
        cp Jax/line_search_backtracking.py Stationary/Jax/
    when the canonical one is updated.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # <repo>/Stationary
    canonical = os.path.dirname(here)                                        # <repo>
    copy = os.path.join(here, "Jax")
    if not os.path.isdir(os.path.join(canonical, "Jax")):
        pytest.skip("no canonical Jax/ in this checkout (a Stationary-only sync)")
    def norm(path):
        # line endings are not drift: `* text=auto` stores both copies with LF in the
        # repository (checked: the seven blobs are identical in HEAD) while this working tree
        # may hold either, depending on how the files were written or checked out
        return open(path, "rb").read().replace(b"\r\n", b"\n")

    differing = []
    for rel in MIN_PATH:
        a = os.path.join(canonical, "Jax", rel)
        b = os.path.join(copy, rel)
        if not os.path.exists(b) or not os.path.exists(a):
            differing.append(rel + " (missing)")
        elif norm(a) != norm(b):
            differing.append(rel)
    assert differing == [], f"the Stationary/Jax copy has drifted: {differing}"
