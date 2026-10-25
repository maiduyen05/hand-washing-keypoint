"""Bounded landmark interpolation that preserves original presence masks."""

from __future__ import annotations

import numpy as np


def interpolate_short_gaps(
    points: np.ndarray,
    present: np.ndarray,
    timestamps: np.ndarray,
    max_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate only bounded missing runs whose elapsed time is within the limit.

    The caller must pass one already-contiguous video/label segment, so this
    function cannot cross video, frame-index or label boundaries.
    """
    points = np.asarray(points, dtype=np.float32)
    present = np.asarray(present, dtype=bool)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (21, 3):
        raise ValueError(f"Expected landmark shape (T, 21, 3), got {points.shape}")
    if len(points) != len(present) or len(points) != len(timestamps):
        raise ValueError("points, present and timestamps must have equal length")

    filled = points.copy()
    imputed = np.zeros(len(points), dtype=np.int8)
    index = 0
    while index < len(points):
        if present[index]:
            index += 1
            continue
        start = index
        while index < len(points) and not present[index]:
            index += 1
        end = index  # exclusive
        left = start - 1
        right = end
        if left < 0 or right >= len(points) or not present[left] or not present[right]:
            continue
        anchor_elapsed = float(timestamps[right] - timestamps[left])
        
        # Duration of the missing samples themselves (not the two valid anchors).
        gap_duration = anchor_elapsed * (end - start) / (right - left)
        if anchor_elapsed <= 0 or gap_duration > max_seconds + 1e-9:
            continue
        denominator = anchor_elapsed
        for row in range(start, end):
            weight = float((timestamps[row] - timestamps[left]) / denominator)
            filled[row] = (1.0 - weight) * points[left] + weight * points[right]
            imputed[row] = 1
    return filled, imputed
