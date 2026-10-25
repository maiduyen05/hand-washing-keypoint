"""Training data tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.training_data import (
    BASE_FEATURES,
    FeatureStandardizer,
    _apply_development_fold,
    filter_reaction_margin,
    select_feature_names,
)
from train import load_training_config


def test_feature_group_selection_keeps_base_in_manifest_order() -> None:
    names = [
        *BASE_FEATURES,
        "shape_left_landmark_0_x",
        "relation_palm_distance",
        "motion_speed_left_palm",
    ]
    manifest = {"feature_names": names}
    assert select_feature_names(manifest, ["shape"]) == [*BASE_FEATURES, names[7]]
    assert select_feature_names(manifest, ["shape", "motion"]) == [
        *BASE_FEATURES, names[7], names[9]
    ]
    assert select_feature_names(manifest, ["shape", "relation", "motion"]) == names


def test_training_config_does_not_require_author(tmp_path) -> None:
    path = tmp_path / "training.yaml"
    path.write_text("data: {}\nmodel: {}\ntraining: {}\n", encoding="utf-8")
    assert load_training_config(path)["data"] == {}


def test_actual_full_manifest_has_515_features_when_available() -> None:
    path = Path("artifacts/preprocessing/feature_manifest.json")
    if not path.exists():
        pytest.skip("Local preprocessing artifacts are intentionally not stored in Git")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    selected = select_feature_names(manifest, ["shape", "relation", "motion"])
    assert len(selected) == 515
    assert len(selected) == len(set(selected))


def test_scaler_uses_only_valid_training_values_and_preserves_masks() -> None:
    frame = pd.DataFrame({
        "split": ["train", "train", "validation"],
        "left_present": [1, 1, 1],
        "right_present": [1, 1, 0],
        "left_imputed": [0, 0, 0],
        "right_imputed": [0, 0, 0],
        "cross_valid": [1, 0, 0],
        "left_confidence": [0.8, 0.9, 0.7],
        "right_confidence": [0.8, 0.8, 0.0],
        "shape_left_landmark_0_x": [1.0, 3.0, 100.0],
        "relation_palm_distance": [5.0, 0.0, 0.0],
    })
    features = [*BASE_FEATURES, "shape_left_landmark_0_x", "relation_palm_distance"]
    scaler = FeatureStandardizer.fit(frame, features)
    matrix = scaler.transform(frame)
    shape_index = features.index("shape_left_landmark_0_x")
    relation_index = features.index("relation_palm_distance")
    assert scaler.means[shape_index] == pytest.approx(2.0)
    assert matrix[2, shape_index] == pytest.approx(98.0)
    assert np.array_equal(matrix[:, 0], np.array([1, 1, 1], dtype=np.float32))
    assert matrix[1, relation_index] == 0.0
    assert matrix[2, relation_index] == 0.0


def test_reaction_margin_removes_windows_near_both_segment_edges() -> None:
    metadata = pd.DataFrame({
        "video_name": ["v.mp4"] * 12,
        "segment_id": [0] * 12,
    })
    windows = pd.DataFrame({
        "video_name": ["v.mp4"] * 5,
        "segment_id": [0] * 5,
        "start_row": [0, 2, 4, 6, 8],
        "end_row_exclusive": [4, 6, 8, 10, 12],
        "split": ["train"] * 5,
        "label": [1] * 5,
    })
    filtered, margin_frames = filter_reaction_margin(
        windows, metadata, margin_seconds=0.125, nominal_fps=16.0
    )
    assert margin_frames == 2
    assert filtered["start_row"].tolist() == [2, 4, 6]


def test_actual_margin_counts_when_artifacts_are_available() -> None:
    root = Path("artifacts/preprocessing")
    if not (root / "features.parquet").exists():
        pytest.skip("Local preprocessing artifacts are intentionally not stored in Git")
    metadata = pd.read_parquet(
        root / "features.parquet", columns=["video_name", "segment_id"]
    )
    windows = pd.read_parquet(root / "window_index.parquet")
    filtered, margin_frames = filter_reaction_margin(windows, metadata, 0.5, 16.0)
    assert margin_frames == 8
    assert filtered.groupby("split").size().to_dict() == {
        "test": 7286,
        "train": 33455,
        "validation": 7852,
    }


def test_development_fold_excludes_test_and_keeps_participants_disjoint() -> None:
    features = pd.DataFrame({
        "participant_id": [1, 1, 2, 2, 3, 3],
        "fold": [0, 0, 1, 1, 2, 2],
        "split": ["train", "train", "validation", "validation", "test", "test"],
        "left_present": [1, 1, 1, 1, 1, 1],
        "right_present": [1, 1, 1, 1, 1, 1],
        "left_imputed": [0, 0, 0, 0, 0, 0],
        "right_imputed": [0, 0, 0, 0, 0, 0],
        "cross_valid": [1, 1, 1, 1, 1, 1],
        "shape_left_landmark_0_x": [1.0, 3.0, 100.0, 200.0, 1000.0, 2000.0],
    })
    windows = pd.DataFrame({
        "participant_id": [1, 2, 3],
        "fold": [0, 1, 2],
        "split": ["train", "validation", "test"],
        "label": [0, 1, 2],
    })
    folded_features, folded_windows = _apply_development_fold(features, windows, fold=1)
    assert folded_windows["participant_id"].tolist() == [1, 2]
    assert folded_windows["split"].tolist() == ["train", "validation"]
    assert set(folded_features.loc[folded_features["participant_id"] == 3, "split"]) == {"excluded"}
    scaler = FeatureStandardizer.fit(
        folded_features, ["left_present", "shape_left_landmark_0_x"]
    )
    assert scaler.scaled.tolist() == [False, True]
    assert scaler.means[1] == pytest.approx(2.0)
