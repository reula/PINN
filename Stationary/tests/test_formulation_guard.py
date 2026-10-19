"""A first-order network (independent Gamma) must not train without its compatibility equation.

`rot_mlp_ssb` (2026-10-19) did: the shared recipe's `--eq-weights compat=0`, right for the metric-only
networks, left the mlp's Gamma untied to its h -- loss 2.6e-4, compat residual 3.2e3, h error 9.4.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import pytest

from stationary import train as T
from stationary.train import parse_args

SMALL = ["--n-coll", "32", "--n-bnd", "8", "--width", "4", "--depth", "2", "--no-figures"]


@pytest.mark.parametrize("arch", ["mlp", "sym"])
def test_a_first_order_network_with_zero_compat_weight_is_refused(tmp_path, arch):
    cfg = parse_args(SMALL + ["--arch", arch, "--outdir", str(tmp_path / "r"),
                              "--eq-weights", "compat=0,ricci=1,gauge=1,lam_eq=1"])
    with pytest.raises(SystemExit, match="INDEPENDENT Gamma"):
        T.build(cfg)


def test_a_first_order_network_with_compat_on_builds_but_warns(tmp_path, capsys):
    cfg = parse_args(SMALL + ["--arch", "mlp", "--outdir", str(tmp_path / "r"),
                              "--eq-weights", "compat=1,ricci=1,gauge=1,lam_eq=1"])
    T.build(cfg)
    assert "metric-only networks are superior" in capsys.readouterr().out


@pytest.mark.parametrize("arch", ["hybrid", "sym_hybrid", "axisym_hybrid"])
def test_metric_only_networks_need_no_compat_term(tmp_path, arch):
    cfg = parse_args(SMALL + ["--arch", arch, "--outdir", str(tmp_path / "r"),
                              "--eq-weights", "compat=0,ricci=1,gauge=1,lam_eq=1"])
    T.build(cfg)


def test_a_rotated_weyl_run_picks_the_metric_only_3d_network():
    from stationary.problem import Config
    assert Config(weyl=True, weyl_rotate_deg=45.0, arch="axisym_hybrid").arch == "hybrid"
