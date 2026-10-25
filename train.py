#!/usr/bin/env python3
"""Train the METC GRU classifier from prepared frame/window artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.training_data import FEATURE_GROUPS, prepare_training_data


def load_training_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Training YAML must contain an object at its root")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GRU on prepared METC windows")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--groups", nargs="+", choices=FEATURE_GROUPS)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Load a resumable checkpoint and evaluate without running more epochs",
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=range(5),
        help="Use one participant fold as validation and exclude the original test split",
    )
    parser.add_argument(
        "--resume",
        default="none",
        help="none, auto, or a path to last_checkpoint.pt",
    )
    args = parser.parse_args()
    if args.fold is not None and args.evaluate_test:
        parser.error("--evaluate-test is forbidden with --fold; the original test must stay sealed")
    if args.evaluate_only and args.resume == "none":
        parser.error("--evaluate-only requires --resume auto or a checkpoint path")

    config = load_training_config(args.config)
    groups = args.groups or config["data"]["groups"]
    data = prepare_training_data(args.data_dir, config, groups, fold=args.fold)

    # Import PyTorch only after data/config validation so input errors stay clear.
    from src.training import train_and_evaluate

    result = train_and_evaluate(
        data,
        config,
        args.output_dir,
        args.evaluate_test,
        resume=args.resume,
        evaluate_only=args.evaluate_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
