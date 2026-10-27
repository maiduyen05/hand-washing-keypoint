"""Zenodo metadata, participant-safe manifests and deterministic splits."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import output_path


MOVEMENT_COLS = [f"movement_{i}" for i in range(7)]

# Các tên không tuân theo mẫu <participant>-<interface>. Bảng này cố ý tường minh.
FILENAME_PARTICIPANT_EXCEPTIONS = {
    "2021-06-30_09-56-19-11labots.mp4": 1,
    "2021-06-30_09-50-42-13.mp4": 1,
}


def participant_id_from_filename(filename: str, interface_id: int) -> int:
    if filename in FILENAME_PARTICIPANT_EXCEPTIONS:
        return FILENAME_PARTICIPANT_EXCEPTIONS[filename]
    match = re.match(
        r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-(?P<participant>\d+)",
        filename,
    )
    if not match:
        raise ValueError(f"Cannot parse participant id from filename: {filename}")
    participant = int(match.group("participant"))
    if participant < 1 or participant > 999:
        raise ValueError(f"Invalid participant id {participant} in {filename}")
    return participant

## Hàm cũ này sẽ down file statistics từ Zenodo, 
# nhưng hiện tại chúng ta đã có file statistics trên local, nên không cần download nữa.
# def download_statistics(config: dict[str, Any]) -> pd.DataFrame:
#     url = config["source"]["statistics_url"]
#     with urllib.request.urlopen(url, timeout=60) as response:
#         payload = response.read()
#     digest = hashlib.md5(payload).hexdigest()  # nosec: checksum supplied by Zenodo
#     expected = str(config["source"]["statistics_md5"]).lower()
#     if digest != expected:
#         raise ValueError(f"statistics.csv checksum mismatch: {digest} != {expected}")
#     destination = output_path(config, "zenodo_statistics.csv")
#     destination.write_bytes(payload)
#     frame = pd.read_csv(io.BytesIO(payload))
#     expected_rows = int(config["source"]["expected_videos"])
#     if len(frame) != expected_rows:
#         raise ValueError(f"Expected {expected_rows} Zenodo rows, found {len(frame)}")
#     return frame


def download_statistics(config: dict[str, Any]) -> pd.DataFrame:
    # 1. Đường dẫn file CSV đã có sẵn trên local
    file_path = Path(config["paths"]["output_dir"]) / "zenodo_statistics.csv"

    # 2. Kiểm tra file có tồn tại không
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file statistics: {file_path}"
        )

    # 3. Đọc dữ liệu từ file local
    payload = file_path.read_bytes()

    # 4. Kiểm tra MD5 để đảm bảo nội dung file không bị thay đổi
    digest = hashlib.md5(payload).hexdigest()
    expected = str(config["source"]["statistics_md5"]).lower()

    if digest != expected:
        raise ValueError(
            f"MD5 không khớp: {digest} != {expected}"
        )

    # 5. Chuyển dữ liệu CSV thành DataFrame
    frame = pd.read_csv(io.BytesIO(payload))

    # 6. Kiểm tra số lượng video trong file
    expected_rows = int(config["source"]["expected_videos"])

    if len(frame) != expected_rows:
        raise ValueError(
            f"Số video không đúng: {len(frame)} != {expected_rows}"
        )

    print(f"Đã đọc statistics từ local: {file_path}")
    print(f"Số video trong statistics: {len(frame)}")

    return frame


def _scan_local_csv(path: str | Path, old_split: str) -> pd.DataFrame:
    required = ["video_name", "motion_label"]
    counts: dict[tuple[str, int], int] = {}
    video_rows: dict[str, int] = {}
    for chunk in pd.read_csv(path, usecols=required, chunksize=100_000):
        invalid = sorted(set(chunk["motion_label"].unique()) - set(range(7)))
        if invalid:
            raise ValueError(f"Unexpected labels in {path}: {invalid}")
        for video, count in chunk["video_name"].value_counts().items():
            video_rows[str(video)] = video_rows.get(str(video), 0) + int(count)
        grouped = chunk.groupby(["video_name", "motion_label"], observed=True).size()
        for (video, label), count in grouped.items():
            key = (str(video), int(label))
            counts[key] = counts.get(key, 0) + int(count)

    records = []
    for video, row_count in video_rows.items():
        record: dict[str, Any] = {
            "filename": video,
            "old_split": old_split,
            "local_frame_count": row_count,
        }
        for label in range(7):
            record[f"local_frames_{label}"] = counts.get((video, label), 0)
        records.append(record)
    return pd.DataFrame(records)


def build_dataset_manifest(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    statistics = download_statistics(config)
    statistics["interface_id"] = statistics["location"].str.extract(r"(\d+)$").astype(int)
    statistics["participant_id"] = [
        participant_id_from_filename(name, interface)
        for name, interface in zip(statistics["filename"], statistics["interface_id"])
    ]

    train = _scan_local_csv(config["paths"]["train_csv"], "train")
    test = _scan_local_csv(config["paths"]["test_csv"], "test")
    local = pd.concat([train, test], ignore_index=True)
    duplicated = local[local["filename"].duplicated(keep=False)]["filename"].tolist()
    if duplicated:
        raise ValueError(f"Videos occur in both local CSV files: {duplicated}")

    manifest = statistics.merge(local, on="filename", how="outer", indicator=True)
    manifest["local_status"] = manifest["_merge"].map(
        {"both": "available", "left_only": "missing_local", "right_only": "not_in_zenodo"}
    )
    manifest = manifest.drop(columns=["_merge"])
    manifest.to_csv(output_path(config, "dataset_manifest.csv"), index=False, encoding="utf-8-sig")

    available = manifest[manifest["local_status"] == "available"]
    old_train_people = set(available.loc[available["old_split"] == "train", "participant_id"].astype(int))
    old_test_people = set(available.loc[available["old_split"] == "test", "participant_id"].astype(int))
    report = {
        "zenodo_rows": int((manifest["local_status"] != "not_in_zenodo").sum()),
        "local_videos": int((manifest["local_status"] == "available").sum()),
        "participant_count": int(statistics["participant_id"].nunique()),
        "missing_local": manifest.loc[manifest["local_status"] == "missing_local", "filename"].tolist(),
        "not_in_zenodo": manifest.loc[manifest["local_status"] == "not_in_zenodo", "filename"].tolist(),
        "old_split_participant_overlap": sorted(old_train_people & old_test_people),
        "old_split_participant_overlap_count": len(old_train_people & old_test_people),
    }
    output_path(config, "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest, report


def _participant_vectors(manifest: pd.DataFrame) -> tuple[list[int], np.ndarray]:
    available = manifest[manifest["local_status"] == "available"].copy()
    feature_cols = ["local_frame_count", *[f"local_frames_{i}" for i in range(7)]]
    interface_dummies = pd.get_dummies(available["interface_id"], prefix="interface", dtype=float)
    augmented = pd.concat([available[["participant_id", *feature_cols]], interface_dummies], axis=1)
    grouped = augmented.groupby("participant_id", observed=True).sum(numeric_only=True)
    people = [int(value) for value in grouped.index]
    vectors = grouped.to_numpy(dtype=np.float64)
    scales = np.maximum(vectors.sum(axis=0), 1.0)
    return people, vectors / scales


def _split_score(vectors: np.ndarray, groups: list[np.ndarray], targets: np.ndarray) -> float:
    total = vectors.sum(axis=0)
    score = 0.0
    for indices, target in zip(groups, targets):
        observed = vectors[indices].sum(axis=0)
        score += float(np.square(observed - total * target).sum())
    return score


def create_split_manifests(config: dict[str, Any], manifest: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if manifest is None:
        manifest = pd.read_csv(output_path(config, "dataset_manifest.csv"))
    people, vectors = _participant_vectors(manifest)
    expected_people = int(config["source"]["expected_participants"])
    if len(people) != expected_people:
        raise ValueError(f"Expected {expected_people} participants, found {len(people)}")

    split_cfg = config["split"]
    sizes = np.array([
        int(split_cfg["train_participants"]),
        int(split_cfg["validation_participants"]),
        int(split_cfg["test_participants"]),
    ])
    if sizes.sum() != len(people):
        raise ValueError(f"Participant split sizes {sizes.tolist()} do not sum to {len(people)}")
    rng = np.random.default_rng(int(split_cfg["seed"]))
    targets = sizes / sizes.sum()
    best: tuple[float, np.ndarray] | None = None
    for _ in range(int(split_cfg.get("search_iterations", 5000))):
        order = rng.permutation(len(people))
        groups = np.split(order, np.cumsum(sizes)[:-1])
        score = _split_score(vectors, groups, targets)
        if best is None or score < best[0]:
            best = (score, order.copy())
    assert best is not None
    groups = np.split(best[1], np.cumsum(sizes)[:-1])
    assignments: dict[int, str] = {}
    for name, indices in zip(("train", "validation", "test"), groups):
        assignments.update({people[int(index)]: name for index in indices})

    # Deterministic, balanced fold assignment at participant level.
    fold_count = int(split_cfg["folds"])
    tie_order = rng.permutation(len(people))
    ranked = sorted(tie_order, key=lambda i: -float(vectors[int(i)].sum()))
    fold_vectors = np.zeros((fold_count, vectors.shape[1]), dtype=np.float64)
    fold_sizes = np.zeros(fold_count, dtype=int)
    folds: dict[int, int] = {}
    for raw_index in ranked:
        index = int(raw_index)
        candidate_scores = []
        for fold in range(fold_count):
            projected = fold_vectors[fold] + vectors[index]
            candidate_scores.append((float(np.square(projected).sum()), fold_sizes[fold], fold))
        _, _, chosen = min(candidate_scores)
        fold_vectors[chosen] += vectors[index]
        fold_sizes[chosen] += 1
        folds[people[index]] = chosen

    available = manifest[manifest["local_status"] == "available"].copy()
    available["new_split"] = available["participant_id"].astype(int).map(assignments)
    available["fold"] = available["participant_id"].astype(int).map(folds).astype(int)
    split_cols = ["filename", "participant_id", "interface_id", "old_split", "new_split"]
    fold_cols = ["filename", "participant_id", "interface_id", "fold"]
    available[split_cols].to_csv(output_path(config, "split_manifest.csv"), index=False, encoding="utf-8-sig")
    available[fold_cols].to_csv(output_path(config, "fold_manifest.csv"), index=False, encoding="utf-8-sig")
    return available[split_cols], available[fold_cols]
