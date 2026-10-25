"""Imputation tests."""

import numpy as np

from src.imputation import interpolate_short_gaps


def _series(missing: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = missing + 2
    points = np.zeros((length, 21, 3), dtype=np.float32)
    points[-1] = 1.0
    present = np.zeros(length, dtype=bool)
    present[[0, -1]] = True
    timestamps = np.arange(length, dtype=np.float64) / 16.0
    return points, present, timestamps


def test_interpolates_four_missing_frames_at_quarter_second() -> None:
    points, present, timestamps = _series(4)
    filled, flags = interpolate_short_gaps(points, present, timestamps, 0.25)
    assert flags.tolist() == [0, 1, 1, 1, 1, 0]
    assert np.allclose(filled[1, 0], 0.2)


def test_does_not_interpolate_long_gap() -> None:
    points, present, timestamps = _series(5)
    filled, flags = interpolate_short_gaps(points, present, timestamps, 0.25)
    assert not flags.any()
    assert np.allclose(filled[1:-1], 0.0)
