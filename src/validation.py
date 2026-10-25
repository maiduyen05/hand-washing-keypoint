"""Validation report for manifests and prepared Parquet artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import output_path


def _manifest_checks(config: dict[str, Any]) -> dict[str, Any]:
    dataset = pd.read_csv(output_path(config, "dataset_manifest.csv"))
    split = pd.read_csv(output_path(config, "split_manifest.csv"))
    fold = pd.read_csv(output_path(config, "fold_manifest.csv"))
    participant_sets = {
        name: set(group["participant_id"].astype(int))
        for name, group in split.groupby("new_split", observed=True)
    }
    overlap = {
        "train_validation": sorted(participant_sets.get("train", set()) & participant_sets.get("validation", set())),
        "train_test": sorted(participant_sets.get("train", set()) & participant_sets.get("test", set())),
        "validation_test": sorted(participant_sets.get("validation", set()) & participant_sets.get("test", set())),
    }
    participant_counts = split.groupby("new_split")["participant_id"].nunique().astype(int).to_dict()
    fold_uniqueness = int(fold.groupby("participant_id")["fold"].nunique().max())
    return {
        "zenodo_video_count": int((dataset["local_status"] != "not_in_zenodo").sum()),
        "local_video_count": int((dataset["local_status"] == "available").sum()),
        "missing_local": dataset.loc[dataset["local_status"] == "missing_local", "filename"].tolist(),
        "participant_counts": participant_counts,
        "participant_overlap": overlap,
        "folds_per_participant_max": fold_uniqueness,
        "manifest_valid": not any(overlap.values()) and fold_uniqueness == 1,
    }


def _parquet_checks(config: dict[str, Any]) -> dict[str, Any]:
    feature_path = output_path(config, "features.parquet")
    window_path = output_path(config, "window_index.parquet")
    manifest_path = output_path(config, "feature_manifest.json")
    if not feature_path.exists() or not window_path.exists() or not manifest_path.exists():
        return {"prepared": False, "reason": "prepare artifacts are not present"}

    feature_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = feature_manifest["feature_names"]
    parquet = pq.ParquetFile(feature_path)
    schema_names = parquet.schema_arrow.names
    values_are_finite = True
    for batch in parquet.iter_batches(columns=names, batch_size=10_000):
        numeric = batch.to_pandas().to_numpy(dtype=np.float32)
        if not np.isfinite(numeric).all():
            values_are_finite = False
            break
    windows = pd.read_parquet(window_path)
    window_size = int(config["sequence"]["window_size"])
    window_valid = bool(
        windows.empty
        or (
            ((windows["end_row_exclusive"] - windows["start_row"]) == window_size).all()
            and ((windows["end_frame"] - windows["start_frame"] + 1) == window_size).all()
        )
    )
    return {
        "prepared": True,
        "feature_rows": parquet.metadata.num_rows,
        "feature_count": len(names),
        "feature_names_unique": len(names) == len(set(names)),
        "feature_schema_matches_manifest": all(name in schema_names for name in names),
        "feature_values_finite": values_are_finite,
        "window_count": int(len(windows)),
        "windows_valid": window_valid,
    }


def validate_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    report = {
        "manifest": _manifest_checks(config),
        "prepared_data": _parquet_checks(config),
    }
    manifest_ok = report["manifest"]["manifest_valid"]
    prepared = report["prepared_data"]
    prepared_ok = not prepared.get("prepared", False) or all(
        prepared.get(key, False)
        for key in ("feature_names_unique", "feature_schema_matches_manifest", "feature_values_finite", "windows_valid")
    )
    report["valid"] = bool(manifest_ok and prepared_ok)
    output_path(config, "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
