#!/usr/bin/env python3
"""CLI for METC data preparation."""

from __future__ import annotations

import argparse
import json

from src.config import load_config
from src.pipeline import COMMANDS


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the METC hand-washing dataset (no training)")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", default="configs/preprocessing.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    result = COMMANDS[args.command](config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
