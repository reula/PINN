"""Session-wide isolation for the module-level knobs in `stationary.geometry`.

`geometry` carries process-global switches that the training program sets from the config:

    _WANT_COMPAT, _RICCI_LAM_SOURCE, _LAM_EQ_FORM (= _LAM_EQ_FLOOR), _RELATIVE_TERMS

They are set once per run and are not part of any function signature, which is right for the
program and wrong for a test session: in a full run of the suite they leak between FILES.  The
concrete failure this fixture exists for: `tests/test_artifacts.py` runs a tiny training with a
model that derives Gamma from h, which calls `set_want_compat(False)`; `_WANT_COMPAT` is then
still False when `tests/test_equation_keys.py` and `tests/test_pipeline.py` run, so
`residuals_at` omits the `compat` group and those files die with `KeyError: 'pde_compat'` --
even though each of them passes on its own.  (Reproduced at the parent commit, so it is not a
regression of any one change: it is the ordering.)

The fixture saves the four globals before every test and restores them afterwards, so a file can
neither depend on nor contaminate what ran before it.  Tests that want a non-default pairing
(e.g. the first-order system, where `compat` IS an equation) set it explicitly with the setters,
which is what the project owner asks for: the usual run has no `compat` equation, and only the
first-order formulation uses it.
"""
from __future__ import annotations

import pytest

from stationary import geometry


def _snapshot():
    return (geometry._WANT_COMPAT, geometry._RICCI_LAM_SOURCE, geometry._LAM_EQ_FORM,
            geometry._LAM_EQ_FLOOR, geometry._RELATIVE_TERMS)


def _restore(state):
    (geometry._WANT_COMPAT, geometry._RICCI_LAM_SOURCE, geometry._LAM_EQ_FORM,
     geometry._LAM_EQ_FLOOR, geometry._RELATIVE_TERMS) = state


@pytest.fixture(autouse=True)
def restore_geometry_knobs():
    saved = _snapshot()
    yield
    _restore(saved)
