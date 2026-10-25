"""Landmark geometry shared by preprocessing stages."""

from __future__ import annotations

import numpy as np


NUM_LANDMARKS = 21
AXES = ("x", "y", "z")
PALM_INDICES = np.array([0, 5, 9, 13, 17])
TIP_INDICES = np.array([4, 8, 12, 16, 20])
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
]
INTRA_HAND_PAIRS = [
    (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),
    (4, 8), (4, 12), (4, 16), (4, 20),
    (8, 12), (12, 16), (16, 20),
]

# Unique, physically motivated pairs. Duplicates in the legacy list were removed.
CROSS_HAND_PAIRS = [
    (0, 0), (9, 9), (5, 5), (13, 13), (17, 17),
    (20, 1), (16, 5), (12, 9), (8, 13), (4, 17),
    (1, 20), (5, 16), (9, 12), (13, 8), (17, 4),
    (8, 5), (16, 13), (20, 17),
    (5, 8), (13, 16), (17, 20),
    (6, 9), (10, 9), (14, 9), (18, 9),
    (9, 6), (9, 10), (9, 14), (9, 18),
    (4, 9), (4, 5), (9, 4), (5, 4),
    (8, 9), (16, 9), (20, 9),
    (9, 8), (9, 16), (9, 20),
]


def hand_scale(points: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    scale = np.linalg.norm(points[:, 9] - points[:, 0], axis=1)
    return np.maximum(scale, epsilon).astype(np.float32)


def normalize_single_hand(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.zeros_like(points, dtype=np.float32)
    if np.any(valid):
        centered = points[valid] - points[valid, 0:1]
        result[valid] = centered / hand_scale(points[valid])[:, None, None]
    return result


def bone_vectors(points: np.ndarray) -> np.ndarray:
    return np.stack([points[:, b] - points[:, a] for a, b in HAND_CONNECTIONS], axis=1)


def pair_distances(points: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.stack([np.linalg.norm(points[:, b] - points[:, a], axis=1) for a, b in pairs], axis=1)


def palm_center(points: np.ndarray) -> np.ndarray:
    return points[:, PALM_INDICES].mean(axis=1)


def palm_normal(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    first = points[:, 5] - points[:, 0]
    second = points[:, 17] - points[:, 0]
    normal = np.cross(first, second)
    norm = np.linalg.norm(normal, axis=1, keepdims=True)
    result = np.zeros_like(normal, dtype=np.float32)
    usable = valid & (norm[:, 0] > 1e-8)
    result[usable] = normal[usable] / norm[usable]
    return result
