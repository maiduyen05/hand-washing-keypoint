"""Pipeline orchestration for audit, split, prepare and validate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import output_path
from .dataset_metadata import build_dataset_manifest, create_split_manifests
from .feature_engineering import feature_names, process_video
from .sequence_builder import WINDOW_COLUMNS, build_window_index
from .validation import validate_artifacts


def audit(config: dict[str, Any]) -> dict[str, Any]:
    _, report = build_dataset_manifest(config)
    return report


def split(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = output_path(config, "dataset_manifest.csv")
    if not manifest_path.exists():
        manifest, _ = build_dataset_manifest(config)
    else:
        manifest = pd.read_csv(manifest_path)
    split_manifest, fold_manifest = create_split_manifests(config, manifest)
    return {
        "videos": len(split_manifest),
        "participants_by_split": split_manifest.groupby("new_split")["participant_id"].nunique().to_dict(),
        "participants_by_fold": fold_manifest.groupby("fold")["participant_id"].nunique().to_dict(),
    }


def _iter_videos(path: str | Path, chunk_size: int) -> Iterator[tuple[str, pd.DataFrame]]:
    carry: pd.DataFrame | None = None
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = None
        last_video = str(chunk.iloc[-1]["video_name"])
        complete = chunk[chunk["video_name"] != last_video]
        carry = chunk[chunk["video_name"] == last_video].copy()
        for video, frame in complete.groupby("video_name", sort=False):
            yield str(video), frame
    if carry is not None and not carry.empty:
        for video, frame in carry.groupby("video_name", sort=False):
            yield str(video), frame


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    split_path = output_path(config, "split_manifest.csv")
    fold_path = output_path(config, "fold_manifest.csv")
    if not split_path.exists() or not fold_path.exists():
        split(config)
    split_frame = pd.read_csv(split_path)
    fold_frame = pd.read_csv(fold_path)
    metadata = split_frame.merge(
        fold_frame[["filename", "fold"]], on="filename", how="inner", validate="one_to_one"
    ).set_index("filename")

    enabled = {key: bool(value) for key, value in config["features"].items()}
    names = feature_names(enabled)
    feature_manifest = {
        "groups": enabled,
        "feature_count": len(names),
        "feature_names": names,
    }
    output_path(config, "feature_manifest.json").write_text(
        json.dumps(feature_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    feature_path = output_path(config, "features.parquet")
    window_path = output_path(config, "window_index.parquet")
    feature_writer: pq.ParquetWriter | None = None
    window_writer: pq.ParquetWriter | None = None
    row_offset = 0
    video_count = 0
    compression = config["processing"].get("parquet_compression", "zstd")
    try:
        for csv_path in (config["paths"]["train_csv"], config["paths"]["test_csv"]):
            for video, raw in _iter_videos(csv_path, int(config["processing"]["csv_chunk_size"])):

                # Bỏ qua video chưa đối chiếu được với Zenodo
                if video not in metadata.index:
                    print(f"[SKIP] Video chưa có trong split manifest: {video}")
                    continue

                info = metadata.loc[video].to_dict()
                processed = process_video(
                    raw,
                    info,
                    enabled,
                    float(config["temporal"]["interpolation_max_seconds"]),
                )
                windows = build_window_index(
                    processed,
                    row_offset,
                    int(config["sequence"]["window_size"]),
                    int(config["sequence"]["stride"]),
                )
                feature_table = pa.Table.from_pandas(processed, preserve_index=False)
                if feature_writer is None:
                    feature_writer = pq.ParquetWriter(feature_path, feature_table.schema, compression=compression)
                feature_writer.write_table(feature_table)
                if not windows.empty:
                    window_table = pa.Table.from_pandas(windows[WINDOW_COLUMNS], preserve_index=False)
                    if window_writer is None:
                        window_writer = pq.ParquetWriter(window_path, window_table.schema, compression=compression)
                    window_writer.write_table(window_table)
                row_offset += len(processed)
                video_count += 1
    finally:
        if feature_writer is not None:
            feature_writer.close()
        if window_writer is not None:
            window_writer.close()
    if window_writer is None:
        pd.DataFrame(columns=WINDOW_COLUMNS).to_parquet(window_path, index=False)
        
    # Đảm bảo đã xử lý đủ số video có trong split manifest
    expected_videos = len(metadata)

    if video_count != expected_videos:
        raise RuntimeError(
            f"Preprocessing chưa hoàn chỉnh: "
            f"đã xử lý {video_count}/{expected_videos} video"
        )
        
    return {
        "videos": video_count,
        "rows": row_offset,
        "features": len(names),
        "feature_path": str(feature_path),
        "window_path": str(window_path),
    }


COMMANDS = {
    "audit": audit,
    "split": split,
    "prepare": prepare,
    "validate": validate_artifacts,
}
