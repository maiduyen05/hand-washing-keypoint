#!/usr/bin/env python3
"""Summarize participant-fold validation metrics without touching the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def summarize_cv(input_dirs: list[Path]) -> tuple[dict[str, Any], pd.DataFrame]:
    if len(input_dirs) != 5:
        raise ValueError("Exactly five fold output directories are required")
    records: list[dict[str, Any]] = []
    seen_folds: set[int] = set()
    for directory in input_dirs:
        result = _read_json(directory / "result.json")
        if result.get("evaluated_split") != "validation" or result.get("test_metrics") is not None:
            raise ValueError(f"Fold output must contain validation-only metrics: {directory}")
        fold = result.get("development_fold")
        if not isinstance(fold, int) or fold not in range(5):
            raise ValueError(f"Invalid or missing development_fold in {directory}")
        if fold in seen_folds:
            raise ValueError(f"Duplicate fold output: {fold}")
        seen_folds.add(fold)
        metrics = result["validation_metrics"]
        row: dict[str, Any] = {
            "fold": fold,
            "accuracy": float(metrics["accuracy"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "weighted_f1": float(metrics["weighted_f1"]),
            "best_epoch": int(result["best_epoch"]),
        }
        for label, values in metrics["per_class"].items():
            row[f"class_{label}_f1"] = float(values["f1"])
        records.append(row)
    if seen_folds != set(range(5)):
        raise ValueError(f"Expected folds 0..4, found {sorted(seen_folds)}")

    frame = pd.DataFrame(records).sort_values("fold").reset_index(drop=True)
    metric_columns = [column for column in frame if column not in {"fold", "best_epoch"}]
    summary = {
        "fold_count": 5,
        "folds": frame.to_dict(orient="records"),
        "mean": {column: float(frame[column].mean()) for column in metric_columns},
        "sample_std": {column: float(frame[column].std(ddof=1)) for column in metric_columns},
        "best_epoch_mean": float(np.mean(frame["best_epoch"])),
    }
    return summary, frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize five participant-fold GRU runs")
    parser.add_argument("inputs", nargs=5, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary, frame = summarize_cv(args.inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cv_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frame.to_csv(args.output_dir / "cv_folds.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
