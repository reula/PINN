"""Command line entry point.

Examples
--------
    PY=/Users/reula/jax_env/bin/python
    cd Evolution_try
    $PY -m wave_pinn.cli --label ssbroyden_ref --set optimizer=ssbroyden
    $PY -m wave_pinn.cli --label dsgnar_ref   --set optimizer=dsgnar
    $PY -m wave_pinn.cli --config runs/ssbroyden_ref/config.json --set qn_steps=8000

Every ``Config`` field can be overridden with ``--set key=value``; values are
coerced to int/float/bool/JSON as appropriate.
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import Config, apply_overrides, resolve_outdir
from .train import configure_jax, run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wave_pinn", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="JSON config to start from (e.g. a previous run's config.json)")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="override a config field (repeatable)")
    p.add_argument("--label", default=None, help="run label; artefacts go to runs/<label>")
    p.add_argument("--outdir", default=None, help="explicit output directory")
    p.add_argument("--dry-run", action="store_true", help="print the resolved config and stop")
    p.add_argument("--quiet", action="store_true", help="less console output")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config.from_json(args.config) if args.config else Config()
    if args.label:
        cfg = cfg.replace(label=args.label)
    if args.outdir:
        cfg = cfg.replace(outdir=args.outdir)
    cfg = apply_overrides(cfg, args.overrides)
    cfg.validate()

    if args.dry_run:
        print(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
        print(f"# outdir: {resolve_outdir(cfg)}")
        return 0

    configure_jax(cfg)
    if cfg.windows > 1:
        from .windows import run_windows
        run_windows(cfg, verbose=not args.quiet)
    else:
        run(cfg, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
