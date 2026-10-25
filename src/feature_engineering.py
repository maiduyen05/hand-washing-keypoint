"""Shape, two-hand relation and temporal motion features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .imputation import interpolate_short_gaps
from .normalization import (
    AXES,
    CROSS_HAND_PAIRS,
    HAND_CONNECTIONS,
    INTRA_HAND_PAIRS,
    TIP_INDICES,
    bone_vectors,
    hand_scale,
    normalize_single_hand,
    pair_distances,
    palm_center,
    palm_normal,
)


META_COLUMNS = [
    "video_name", "frame_index", "timestamp", "motion_label",
    "participant_id", "interface_id", "split", "fold", "segment_id",
]


def landmark_columns(side: str) -> list[str]:
    return [f"{side}_p{i}_{axis}" for i in range(21) for axis in AXES]


def _flatten_names(prefix: str, count: int) -> list[str]:
    return [f"{prefix}_{index}_{axis}" for index in range(count) for axis in AXES]


def _base_feature_names() -> list[str]:
    return [
        "left_present", "right_present", "left_imputed", "right_imputed",
        "cross_valid", "left_confidence", "right_confidence",
    ]


def shape_feature_names() -> list[str]:
    names: list[str] = []
    for side in ("left", "right"):
        names += _flatten_names(f"shape_{side}_landmark", 21)
        names += [f"shape_{side}_bone_{a}_{b}_{axis}" for a, b in HAND_CONNECTIONS for axis in AXES]
        names += [f"shape_{side}_distance_{a}_{b}" for a, b in INTRA_HAND_PAIRS]
    return names


def relation_feature_names() -> list[str]:
    names = []

    # Tọa độ chung của hai tay chỉ sử dụng XY
    for side in ("left", "right"):
        for index in range(21):
            for axis in ("x", "y"):
                names.append(
                    f"relation_{side}_common_{index}_{axis}"
                )

    # Vector vị trí tương đối giữa hai lòng bàn tay (XY)
    names += [
        f"relation_palm_vector_{axis}"
        for axis in ("x", "y")
    ]

    # Các khoảng cách giữa hai bàn tay
    names += [
        "relation_palm_distance",
        "relation_tip_min",
        "relation_tip_mean",
        "relation_tip_max"
    ]

    # Khoảng cách giữa các cặp khớp thuộc hai tay
    names += [
        f"relation_cross_L{left}_R{right}"
        for left, right in CROSS_HAND_PAIRS
    ]

    # Giữ nguyên palm normal 3D của DHB
    names += [
        f"relation_left_normal_{axis}"
        for axis in AXES
    ]

    names += [
        f"relation_right_normal_{axis}"
        for axis in AXES
    ]

    names += ["relation_palm_normal_cosine"]

    return names


def motion_feature_names() -> list[str]:
    tracked = ["left_palm", "right_palm", *[f"left_tip_{i}" for i in TIP_INDICES], *[f"right_tip_{i}" for i in TIP_INDICES]]
    names = [f"motion_velocity_{item}_{axis}" for item in tracked for axis in AXES]
    names += [f"motion_speed_{item}" for item in tracked]
    
    # Vận tốc và hướng chuyển động tương đối giữa hai tay chỉ dùng XY
    names += [
        f"motion_relative_velocity_{axis}"
        for axis in ("x", "y")
    ]

    names += [
        f"motion_relative_direction_{axis}"
        for axis in ("x", "y")
    ]
    
    names += ["motion_palm_distance_rate", "motion_palm_angle_rate"]
    names += ["motion_left_valid", "motion_right_valid", "motion_cross_valid"]
    return names


def feature_names(enabled: dict[str, bool]) -> list[str]:
    names = _base_feature_names()
    if enabled.get("shape", True):
        names += shape_feature_names()
    if enabled.get("relation", True):
        names += relation_feature_names()
    if enabled.get("motion", True):
        names += motion_feature_names()
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Duplicate feature names: {duplicates}")
    return names


def assign_segments(frame: pd.DataFrame) -> np.ndarray:
    frame = frame.sort_values("frame_index")
    boundary = (
        frame["frame_index"].diff().ne(1)
        | frame["motion_label"].diff().ne(0)
    )
    boundary.iloc[0] = True
    return (boundary.cumsum() - 1).to_numpy(dtype=np.int32)


def _shape(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    normalized = normalize_single_hand(points, valid)
    bones = bone_vectors(normalized).reshape(len(points), -1)
    distances = pair_distances(normalized, INTRA_HAND_PAIRS)
    result = np.concatenate([normalized.reshape(len(points), -1), bones, distances], axis=1)
    result[~valid] = 0.0
    return result.astype(np.float32)


@dataclass
class RelationState:
    features: np.ndarray
    left_palm: np.ndarray
    right_palm: np.ndarray
    normal_cosine: np.ndarray
    palm_distance: np.ndarray



def _relation(
    left: np.ndarray,
    right: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
) -> RelationState:

    length = len(left)

    # Quan hệ giữa hai tay chỉ hợp lệ khi cả hai tay có dữ liệu
    valid = left_valid & right_valid

    # 1. Tính tâm lòng bàn tay
    # Giữ XYZ để phục vụ motion riêng của từng tay
    left_palm = palm_center(left)
    right_palm = palm_center(right)

    # Chỉ đưa tọa độ về 0 nếu chính bàn tay đó bị thiếu
    left_palm[~left_valid] = 0.0
    right_palm[~right_valid] = 0.0

    # 2. Chỉ lấy tọa độ XY để tính quan hệ hai tay
    left_xy = left[..., :2]
    right_xy = right[..., :2]

    left_palm_xy = left_palm[:, :2]
    right_palm_xy = right_palm[:, :2]

    # Tâm chung của hai lòng bàn tay
    midpoint_xy = (left_palm_xy + right_palm_xy) / 2.0

    # 3. Chuẩn hóa khoảng cách bằng kích thước bàn tay trên XY
    left_scale = np.linalg.norm(
        left_xy[:, 9] - left_xy[:, 0], axis=1
    )

    right_scale = np.linalg.norm(
        right_xy[:, 9] - right_xy[:, 0], axis=1
    )

    scale = np.maximum(
        (left_scale + right_scale) / 2.0,
        1e-6
    )

    # 4. Đưa hai tay về hệ tọa độ chung XY
    left_common = (
        left_xy - midpoint_xy[:, None, :]
    ) / scale[:, None, None]

    right_common = (
        right_xy - midpoint_xy[:, None, :]
    ) / scale[:, None, None]

    # 5. Vector và khoảng cách giữa hai lòng bàn tay
    palm_vector = (
        right_palm_xy - left_palm_xy
    ) / scale[:, None]

    palm_distance = np.linalg.norm(
        palm_vector, axis=1
    )

    # 6. Khoảng cách giữa các đầu ngón tay của hai bàn tay
    tip_distances = np.linalg.norm(
        left_xy[:, TIP_INDICES, None, :]
        - right_xy[:, None, TIP_INDICES, :],
        axis=3
    ).reshape(length, -1) / scale[:, None]

    # Tổng hợp khoảng cách nhỏ nhất, trung bình và lớn nhất
    tip_summary = np.stack(
        [
            tip_distances.min(axis=1),
            tip_distances.mean(axis=1),
            tip_distances.max(axis=1)
        ],
        axis=1
    )

    # 7. Khoảng cách giữa các cặp khớp liên tay trên XY
    cross = np.stack(
        [
            np.linalg.norm(
                left_xy[:, a] - right_xy[:, b],
                axis=1
            ) / scale
            for a, b in CROSS_HAND_PAIRS
        ],
        axis=1
    )

    # 8. Giữ nguyên palm normal 3D của DHB
    # Đây là đặc trưng hướng, không phải khoảng cách 3D
    left_normal = palm_normal(left, valid)
    right_normal = palm_normal(right, valid)

    cosine = np.clip(
        (left_normal * right_normal).sum(axis=1),
        -1.0,
        1.0
    )

    # 9. Ghép các đặc trưng quan hệ giữa hai tay
    features = np.concatenate(
        [
            left_common.reshape(length, -1),
            right_common.reshape(length, -1),
            palm_vector,
            palm_distance[:, None],
            tip_summary,
            cross,
            left_normal,
            right_normal,
            cosine[:, None],
        ],
        axis=1
    ).astype(np.float32)

    # Nếu thiếu một tay, các đặc trưng quan hệ không hợp lệ
    features[~valid] = 0.0
    palm_distance[~valid] = 0.0
    cosine[~valid] = 0.0

    # Không xóa palm của tay còn quan sát được
    return RelationState(
        features,
        left_palm,
        right_palm,
        cosine,
        palm_distance
    )



def _motion(
    left: np.ndarray,
    right: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    timestamps: np.ndarray,
    relation: RelationState,
) -> np.ndarray:
    length = len(left)
    tracked = np.concatenate(
        [
            relation.left_palm[:, None], relation.right_palm[:, None],
            left[:, TIP_INDICES], right[:, TIP_INDICES],
        ],
        axis=1,
    )
    velocities = np.zeros_like(tracked, dtype=np.float32)
    relative_velocity = np.zeros((length, 2), dtype=np.float32)
    relative_direction = np.zeros((length, 2), dtype=np.float32)
    palm_distance_rate = np.zeros(length, dtype=np.float32)
    palm_angle_rate = np.zeros(length, dtype=np.float32)
    left_motion_valid = np.zeros(length, dtype=np.float32)
    right_motion_valid = np.zeros(length, dtype=np.float32)
    cross_motion_valid = np.zeros(length, dtype=np.float32)

    dt = np.diff(timestamps)
    for row in range(1, length):
        if not np.isfinite(dt[row - 1]) or dt[row - 1] <= 0:
            continue
        if left_valid[row - 1] and left_valid[row]:
            velocities[row, 0] = (tracked[row, 0] - tracked[row - 1, 0]) / dt[row - 1]
            velocities[row, 2:7] = (tracked[row, 2:7] - tracked[row - 1, 2:7]) / dt[row - 1]
            left_motion_valid[row] = 1.0
        if right_valid[row - 1] and right_valid[row]:
            velocities[row, 1] = (tracked[row, 1] - tracked[row - 1, 1]) / dt[row - 1]
            velocities[row, 7:12] = (tracked[row, 7:12] - tracked[row - 1, 7:12]) / dt[row - 1]
            right_motion_valid[row] = 1.0
        if left_motion_valid[row] and right_motion_valid[row]:
            # Chỉ tính chuyển động tương đối trên XY
            current_relative = (
                relation.right_palm[row, :2]
                - relation.left_palm[row, :2]
            )

            prior_relative = (
                relation.right_palm[row - 1, :2]
                - relation.left_palm[row - 1, :2]
            )
            
            relative_velocity[row] = (current_relative - prior_relative) / dt[row - 1]
            magnitude = np.linalg.norm(relative_velocity[row])
            if magnitude > 1e-8:
                relative_direction[row] = relative_velocity[row] / magnitude
            palm_distance_rate[row] = (relation.palm_distance[row] - relation.palm_distance[row - 1]) / dt[row - 1]
            current_angle = np.arccos(np.clip(relation.normal_cosine[row], -1.0, 1.0))
            prior_angle = np.arccos(np.clip(relation.normal_cosine[row - 1], -1.0, 1.0))
            palm_angle_rate[row] = (current_angle - prior_angle) / dt[row - 1]
            cross_motion_valid[row] = 1.0
    speeds = np.linalg.norm(velocities, axis=2)
    return np.concatenate(
        [
            velocities.reshape(length, -1), speeds,
            relative_velocity, relative_direction,
            palm_distance_rate[:, None], palm_angle_rate[:, None],
            left_motion_valid[:, None], right_motion_valid[:, None], cross_motion_valid[:, None],
        ],
        axis=1,
    ).astype(np.float32)


def process_video(
    frame: pd.DataFrame,
    metadata: dict[str, object],
    enabled: dict[str, bool],
    interpolation_max_seconds: float,
) -> pd.DataFrame:
    frame = frame.sort_values("frame_index").reset_index(drop=True).copy()
    frame["segment_id"] = assign_segments(frame)
    left_raw = frame[landmark_columns("left")].to_numpy(dtype=np.float32).reshape(-1, 21, 3)
    right_raw = frame[landmark_columns("right")].to_numpy(dtype=np.float32).reshape(-1, 21, 3)
    left_present = frame["left_hand_detected"].to_numpy(dtype=bool)
    right_present = frame["right_hand_detected"].to_numpy(dtype=bool)
    timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
    left = left_raw.copy()
    right = right_raw.copy()
    left_imputed = np.zeros(len(frame), dtype=np.int8)
    right_imputed = np.zeros(len(frame), dtype=np.int8)

    for _, indices in frame.groupby("segment_id", sort=False).groups.items():
        rows = np.asarray(list(indices), dtype=int)
        left[rows], left_imputed[rows] = interpolate_short_gaps(
            left_raw[rows], left_present[rows], timestamps[rows], interpolation_max_seconds
        )
        right[rows], right_imputed[rows] = interpolate_short_gaps(
            right_raw[rows], right_present[rows], timestamps[rows], interpolation_max_seconds
        )

    left_valid = left_present | left_imputed.astype(bool)
    right_valid = right_present | right_imputed.astype(bool)
    cross_valid = left_valid & right_valid
    # Truyền mask của từng tay thay vì mask chung
    relation = _relation(
        left,
        right,
        left_valid,
        right_valid
    )

    blocks = [
        left_present.astype(np.float32)[:, None], right_present.astype(np.float32)[:, None],
        left_imputed.astype(np.float32)[:, None], right_imputed.astype(np.float32)[:, None],
        cross_valid.astype(np.float32)[:, None],
        frame["left_hand_confidence"].to_numpy(dtype=np.float32)[:, None],
        frame["right_hand_confidence"].to_numpy(dtype=np.float32)[:, None],
    ]
    if enabled.get("shape", True):
        blocks += [_shape(left, left_valid), _shape(right, right_valid)]
    if enabled.get("relation", True):
        blocks.append(relation.features)
    if enabled.get("motion", True):
        motion_parts = []
        for _, indices in frame.groupby("segment_id", sort=False).groups.items():
            rows = np.asarray(list(indices), dtype=int)
            sub_relation = RelationState(
                relation.features[rows], relation.left_palm[rows], relation.right_palm[rows],
                relation.normal_cosine[rows], relation.palm_distance[rows],
            )
            motion_parts.append((rows, _motion(
                left[rows], right[rows], left_valid[rows], right_valid[rows], timestamps[rows], sub_relation
            )))
        motion = np.zeros((len(frame), len(motion_feature_names())), dtype=np.float32)
        for rows, values in motion_parts:
            motion[rows] = values
        blocks.append(motion)

    names = feature_names(enabled)
    values = np.concatenate(blocks, axis=1)
    if values.shape[1] != len(names):
        raise AssertionError(f"Feature width {values.shape[1]} != name count {len(names)}")
    result = pd.DataFrame(values, columns=names)
    for column in ("video_name", "frame_index", "timestamp", "motion_label", "segment_id"):
        result.insert(len([c for c in META_COLUMNS if c in result]), column, frame[column].to_numpy())
    result.insert(4, "participant_id", int(metadata["participant_id"]))
    result.insert(5, "interface_id", int(metadata["interface_id"]))
    result.insert(6, "split", str(metadata["new_split"]))
    result.insert(7, "fold", int(metadata["fold"]))
    return result[META_COLUMNS + names]
