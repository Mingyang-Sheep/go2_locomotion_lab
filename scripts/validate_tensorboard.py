#!/usr/bin/env python3
"""Validate required scalar tags in one completed training run."""

from __future__ import annotations

import argparse
from pathlib import Path

from go2_locomotion_lab.monitoring import validate_scalar_tags

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_dir", type=Path)
parser.add_argument("--route", choices=("competition", "him"), default="competition")
args = parser.parse_args()


def main() -> int:
    present, missing = validate_scalar_tags(args.run_dir, args.route)
    print(f"[INFO] Found {len(present)} scalar tags in {args.run_dir.resolve()}")
    if missing:
        print("[ERROR] Missing required tags:")
        for tag in sorted(missing):
            print(f"  {tag}")
        return 1
    print(f"[INFO] {args.route} TensorBoard contract is complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
