"""Lightweight sequence index construction without 3-D materialization."""

from __future__ import annotations

import pandas as pd


WINDOW_COLUMNS = [
    "video_name", "participant_id", "split", "fold", "segment_id",
    "start_row", "end_row_exclusive", "start_frame", "end_frame", "label",
]


def build_window_index(
    feature_frame: pd.DataFrame,
    first_global_row: int,
    window_size: int,
    stride: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for segment_id, segment in feature_frame.groupby("segment_id", sort=False):
        segment = segment.sort_values("frame_index")
        frames = segment["frame_index"].to_numpy()
        if len(segment) < window_size:
            continue
        if not (frames[1:] - frames[:-1] == 1).all():
            raise ValueError("Sequence segment contains non-contiguous frame indices")
        base = int(segment.index[0])
        for start in range(0, len(segment) - window_size + 1, stride):
            end = start + window_size
            last = segment.iloc[end - 1]
            records.append({
                "video_name": str(last["video_name"]),
                "participant_id": int(last["participant_id"]),
                "split": str(last["split"]),
                "fold": int(last["fold"]),
                "segment_id": int(segment_id),
                "start_row": first_global_row + base + start,
                "end_row_exclusive": first_global_row + base + end,
                "start_frame": int(segment.iloc[start]["frame_index"]),
                "end_frame": int(last["frame_index"]),
                "label": int(last["motion_label"]),
            })
    return pd.DataFrame(records, columns=WINDOW_COLUMNS)
