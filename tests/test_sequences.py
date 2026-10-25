"""Sequence tests."""

import pandas as pd

from src.sequence_builder import build_window_index


def test_windows_remain_inside_segments() -> None:
    frame = pd.DataFrame({
        "video_name": ["v.mp4"] * 10,
        "participant_id": [1] * 10,
        "split": ["train"] * 10,
        "fold": [0] * 10,
        "segment_id": [0] * 5 + [1] * 5,
        "frame_index": list(range(5)) + list(range(10, 15)),
        "motion_label": [1] * 5 + [2] * 5,
    })
    windows = build_window_index(frame, first_global_row=100, window_size=4, stride=1)
    assert len(windows) == 4
    assert (windows["end_frame"] - windows["start_frame"] + 1 == 4).all()
    assert set(windows["label"]) == {1, 2}
