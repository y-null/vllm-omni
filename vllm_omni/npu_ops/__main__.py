"""Command-line diagnostics for the vLLM-Omni NPU operator wheel."""

from __future__ import annotations

import argparse
import json

from .runtime import environment, package_info, self_test


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("info", help="print installed payload information")
    subparsers.add_parser("environment", help="print required runtime environment")
    self_test_parser = subparsers.add_parser("self-test", help="load and invoke the packaged A14 operator")
    self_test_parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    if args.command == "info":
        result = package_info()
    elif args.command == "environment":
        result = environment()
    else:
        result = self_test(args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

