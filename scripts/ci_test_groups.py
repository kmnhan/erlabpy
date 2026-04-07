from __future__ import annotations

import argparse
import sys

from scripts._ci_test_groups import (
    check_coverage_partition,
    get_group_targets,
    iter_known_groups,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print CI test group targets.")
    parser.add_argument("group", nargs="?", choices=sorted(iter_known_groups()))
    parser.add_argument(
        "--check-partition",
        action="store_true",
        help="Verify that the coverage groups cover every test file exactly once.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.check_partition:
        errors = check_coverage_partition()
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        print("Coverage groups cover every test file exactly once.")
        return 0

    if args.group is None:
        parser.error("either a group name or --check-partition is required")

    print("\n".join(get_group_targets(args.group)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
