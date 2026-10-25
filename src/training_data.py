"""Data loading, feature selection and train-only scaling for GRU training."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_GROUPS = ("shape", "relation", "motion")
BASE_FEATURES = (
    "left_present",
    "right_present",
    "left_imputed",
    "right_imputed",
    "cross_valid",
    "left_confidence",
    "right_confidence",
)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_preprocessing_artifacts(data_dir: str | Path, report_name: str) -> dict:
    report_path = Path(data_dir) / report_name
    if not report_path.exists():
        raise FileNotFoundError(f"Missing preprocessing validation report: {report_path}")
    report = _read_json(report_path)
    if report.get("valid") is not True:
        raise ValueError("Preprocessing validation_report.json is not valid=true")
    return report


def select_feature_names(manifest: dict, groups: Iterable[str]) -> list[str]:
    requested = tuple(dict.fromkeys(str(group) for group in groups))
    unknown = sorted(set(requested) - set(FEATURE_GROUPS))
    if unknown:
        raise ValueError(f"Unknown feature groups: {unknown}")
    if not requested:
        raise ValueError("At least one feature group is required")

    all_names = manifest.get("feature_names")
    if not isinstance(all_names, list) or not all(isinstance(name, str) for name in all_names):
        raise ValueError("feature_manifest.json does not contain a valid feature_names list")
    if len(all_names) != len(set(all_names)):
        raise ValueError("Feature names in manifest are not unique")
    missing_base = [name for name in BASE_FEATURES if name not in all_names]
    if missing_base:
        raise ValueError(f"Manifest is missing base features: {missing_base}")

    selected = [name for name in all_names if name in BASE_FEATURES]
    selected.extend(
        name for name in all_names
        if any(name.startswith(f"{group}_") for group in requested)
    )
    return selected


def is_binary_feature(name: str) -> bool:
    return (
        name.endswith("_present")
        or name.endswith("_imputed")
        or name.endswith("_valid")
    )


def _validity_column(name: str) -> str | None:
    if name.startswith("shape_left_"):
        return "left_available"
    if name.startswith("shape_right_"):
        return "right_available"
    if name.startswith("relation_"):
        return "cross_valid"
    if name.startswith("motion_velocity_left_") or name.startswith("motion_speed_left_"):
        return "motion_left_valid"
    if name.startswith("motion_velocity_right_") or name.startswith("motion_speed_right_"):
        return "motion_right_valid"
    if name.startswith("motion_") and not is_binary_feature(name):
        return "motion_cross_valid"
    return None


def _validity_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    left_available = (
        frame["left_present"].to_numpy(dtype=np.float32, copy=False)
        + frame["left_imputed"].to_numpy(dtype=np.float32, copy=False)
    ) > 0
    right_available = (
        frame["right_present"].to_numpy(dtype=np.float32, copy=False)
        + frame["right_imputed"].to_numpy(dtype=np.float32, copy=False)
    ) > 0
    arrays = {
        "left_available": left_available,
        "right_available": right_available,
        "cross_valid": frame["cross_valid"].to_numpy(dtype=np.float32, copy=False) > 0,
    }
    for name in ("motion_left_valid", "motion_right_valid", "motion_cross_valid"):
        if name in frame.columns:
            arrays[name] = frame[name].to_numpy(dtype=np.float32, copy=False) > 0
    return arrays


@dataclass
class FeatureStandardizer:
    """Per-feature scaler fitted only on valid training rows."""

    feature_names: list[str]
    means: np.ndarray
    scales: np.ndarray
    scaled: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, feature_names: list[str]) -> "FeatureStandardizer":
        if "split" not in frame:
            raise ValueError("Feature frame must contain the split column")
        train_mask = frame["split"].to_numpy() == "train"
        if not train_mask.any():
            raise ValueError("No training rows are available for scaler fitting")
        validity = _validity_arrays(frame)
        means = np.zeros(len(feature_names), dtype=np.float32)
        scales = np.ones(len(feature_names), dtype=np.float32)
        scaled = np.zeros(len(feature_names), dtype=bool)

        for index, name in enumerate(feature_names):
            if name in BASE_FEATURES or is_binary_feature(name):
                continue
            values = frame[name].to_numpy(dtype=np.float32, copy=False)
            mask = train_mask.copy()
            validity_name = _validity_column(name)
            if validity_name is not None:
                if validity_name not in validity:
                    raise ValueError(f"Missing validity feature {validity_name} required by {name}")
                mask &= validity[validity_name]
            valid_values = values[mask]
            if valid_values.size == 0:
                continue
            mean = float(valid_values.mean(dtype=np.float64))
            scale = float(valid_values.std(dtype=np.float64))
            if not math.isfinite(mean) or not math.isfinite(scale):
                raise ValueError(f"Non-finite scaler statistics for {name}")
            means[index] = mean
            scales[index] = scale if scale > 1e-6 else 1.0
            scaled[index] = True
        return cls(feature_names=list(feature_names), means=means, scales=scales, scaled=scaled)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise ValueError(f"Feature frame is missing columns: {missing[:5]}")
        matrix = frame[self.feature_names].to_numpy(dtype=np.float32, copy=True)
        validity = _validity_arrays(frame)
        for index, name in enumerate(self.feature_names):
            if not self.scaled[index]:
                continue
            matrix[:, index] = (matrix[:, index] - self.means[index]) / self.scales[index]
            validity_name = _validity_column(name)
            if validity_name is not None:
                matrix[~validity[validity_name], index] = 0.0
        if not np.isfinite(matrix).all():
            raise ValueError("Scaled feature matrix contains NaN or Inf")
        return np.ascontiguousarray(matrix, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            feature_names=np.asarray(self.feature_names, dtype=np.str_),
            means=self.means,
            scales=self.scales,
            scaled=self.scaled,
        )


def filter_reaction_margin(
    windows: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    margin_seconds: float,
    nominal_fps: float,
) -> tuple[pd.DataFrame, int]:
    margin_frames = int(math.ceil(max(0.0, margin_seconds) * nominal_fps))
    required_window = {
        "video_name", "segment_id", "start_row", "end_row_exclusive", "split", "label"
    }
    required_metadata = {"video_name", "segment_id"}
    if not required_window.issubset(windows.columns):
        raise ValueError(f"Window index is missing columns: {sorted(required_window - set(windows.columns))}")
    if not required_metadata.issubset(feature_metadata.columns):
        raise ValueError("Feature metadata must contain video_name and segment_id")

    metadata = feature_metadata[["video_name", "segment_id"]].copy()
    metadata["global_row"] = np.arange(len(metadata), dtype=np.int64)
    bounds = metadata.groupby(["video_name", "segment_id"], sort=False).agg(
        segment_start_row=("global_row", "min"),
        segment_end_row=("global_row", "max"),
    ).reset_index()
    bounds["segment_end_row_exclusive"] = bounds["segment_end_row"] + 1
    merged = windows.merge(
        bounds.drop(columns="segment_end_row"),
        on=["video_name", "segment_id"],
        how="left",
        validate="many_to_one",
    )
    if merged["segment_start_row"].isna().any():
        raise ValueError("Some windows do not match a feature segment")
    keep = (
        (merged["start_row"] >= merged["segment_start_row"] + margin_frames)
        & (merged["end_row_exclusive"] <= merged["segment_end_row_exclusive"] - margin_frames)
    )
    filtered = merged.loc[keep, windows.columns].reset_index(drop=True)
    return filtered, margin_frames


@dataclass
class PreparedTrainingData:
    matrix: np.ndarray
    feature_names: list[str]
    windows: dict[str, pd.DataFrame]
    standardizer: FeatureStandardizer
    margin_frames: int
    development_fold: int | None = None


def _apply_development_fold(
    feature_frame: pd.DataFrame,
    windows: pd.DataFrame,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use one participant fold for validation while keeping the original test sealed."""
    available_folds = sorted(int(value) for value in windows["fold"].dropna().unique())
    if fold not in available_folds:
        raise ValueError(f"Fold {fold} is unavailable; expected one of {available_folds}")

    feature_frame = feature_frame.copy()
    original_feature_split = feature_frame["split"].astype(str)
    development_rows = original_feature_split.isin(["train", "validation"])
    feature_frame["split"] = "excluded"
    feature_frame.loc[development_rows & (feature_frame["fold"] != fold), "split"] = "train"
    feature_frame.loc[development_rows & (feature_frame["fold"] == fold), "split"] = "validation"

    windows = windows.loc[windows["split"].isin(["train", "validation"])].copy()
    windows["split"] = np.where(windows["fold"].astype(int) == fold, "validation", "train")
    train_people = set(windows.loc[windows["split"] == "train", "participant_id"].astype(int))
    validation_people = set(
        windows.loc[windows["split"] == "validation", "participant_id"].astype(int)
    )
    if not train_people or not validation_people:
        raise ValueError(f"Fold {fold} produced an empty train or validation participant set")
    overlap = sorted(train_people & validation_people)
    if overlap:
        raise ValueError(f"Participant leakage in fold {fold}: {overlap}")
    return feature_frame, windows


def prepare_training_data(
    data_dir: str | Path,
    config: dict,
    groups: Iterable[str],
    fold: int | None = None,
) -> PreparedTrainingData:
    data_dir = Path(data_dir)
    data_config = config["data"]
    validate_preprocessing_artifacts(data_dir, data_config["validation_report"])
    manifest = _read_json(data_dir / data_config["feature_manifest"])
    selected = select_feature_names(manifest, groups)

    metadata_columns = ["video_name", "participant_id", "segment_id", "split", "fold"]
    read_columns = list(dict.fromkeys(metadata_columns + selected))
    feature_frame = pd.read_parquet(data_dir / data_config["features"], columns=read_columns)
    windows = pd.read_parquet(data_dir / data_config["window_index"])
    windows, margin_frames = filter_reaction_margin(
        windows,
        feature_frame[metadata_columns],
        float(data_config["reaction_margin_seconds"]),
        float(data_config["nominal_fps"]),
    )
    if fold is not None:
        feature_frame, windows = _apply_development_fold(feature_frame, windows, int(fold))

    expected_size = int(data_config["expected_window_size"])
    lengths = windows["end_row_exclusive"] - windows["start_row"]
    if not (lengths == expected_size).all():
        raise ValueError("Window index contains an unexpected window size")
    allowed = set(int(value) for value in data_config["allowed_labels"])
    observed = set(int(value) for value in windows["label"].unique())
    if not observed.issubset(allowed):
        raise ValueError(f"Unexpected labels in window index: {sorted(observed - allowed)}")

    standardizer = FeatureStandardizer.fit(feature_frame, selected)
    matrix = standardizer.transform(feature_frame)
    split_windows = {
        split: windows.loc[windows["split"] == split].reset_index(drop=True)
        for split in ("train", "validation", "test")
    }
    required_splits = ("train", "validation") if fold is not None else ("train", "validation", "test")
    if any(split_windows[name].empty for name in required_splits):
        raise ValueError(f"Required splits must contain windows: {required_splits}")
    if fold is not None and not split_windows["test"].empty:
        raise ValueError("The original test split must remain excluded from fold training")
    return PreparedTrainingData(
        matrix,
        selected,
        split_windows,
        standardizer,
        margin_frames,
        development_fold=fold,
    )
