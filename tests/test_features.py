"""Feature tests."""

import numpy as np

from src.feature_engineering import (
    feature_names,
    _relation,
    _motion,
    relation_feature_names,
    motion_feature_names,
)

from src.normalization import CROSS_HAND_PAIRS, normalize_single_hand


def test_feature_names_and_cross_pairs_are_unique() -> None:
    assert len(CROSS_HAND_PAIRS) == len(set(CROSS_HAND_PAIRS))
    names = feature_names({"shape": True, "relation": True, "motion": True})
    assert len(names) == len(set(names))


def test_shape_normalization_is_translation_invariant() -> None:
    rng = np.random.default_rng(42)
    points = rng.normal(size=(2, 21, 3)).astype(np.float32)
    valid = np.ones(2, dtype=bool)
    translated = points + np.array([10.0, -3.0, 2.5], dtype=np.float32)
    assert np.allclose(
        normalize_single_hand(points, valid),
        normalize_single_hand(translated, valid),
        atol=1e-5,
    )
    

# Kiểm tra khoảng cách giữa hai tay không phụ thuộc z
def test_relation_uses_xy():

    left = np.zeros((1, 21, 3), dtype=np.float32)
    right = np.zeros((1, 21, 3), dtype=np.float32)

    left[:, :, 0] = 0.2
    right[:, :, 0] = 0.5

    # Tạo kích thước bàn tay khác 0
    left[:, 9, 1] = 0.1
    right[:, 9, 1] = 0.1

    valid = np.array([True])

    result1 = _relation(left, right, valid, valid)

    # Chỉ thay đổi z của hai tay
    left[:, :, 2] = -10.0
    right[:, :, 2] = 20.0

    result2 = _relation(left, right, valid, valid)

    names = relation_feature_names()
    idx = names.index("relation_palm_distance")

    # Khoảng cách XY không được thay đổi
    assert np.allclose(
        result1.features[:, idx],
        result2.features[:, idx]
    )


# Kiểm tra tay còn hợp lệ không bị xóa khi tay kia missing
def test_visible_hand_is_preserved():

    left = np.zeros((3, 21, 3), dtype=np.float32)
    right = np.zeros((3, 21, 3), dtype=np.float32)

    for i in range(3):
        left[i, :, 0] = 0.2 + 0.1 * i
        left[i, 9, 1] = 0.1

        right[i, :, 0] = 0.5
        right[i, 9, 1] = 0.1

    left_valid = np.array([True, True, True])
    right_valid = np.array([True, False, True])

    relation = _relation(
        left, right, left_valid, right_valid
    )

    # Frame 1 mất tay phải nhưng palm trái vẫn được giữ
    assert relation.left_palm[1, 0] > 0

    # Relation phải bằng 0 khi thiếu một tay
    assert np.all(relation.features[1] == 0)

    # Motion tay trái vẫn được tính
    motion = _motion(
        left,
        right,
        left_valid,
        right_valid,
        np.arange(3, dtype=float),
        relation
    )

    names = motion_feature_names()

    left_idx = names.index("motion_velocity_left_palm_x")
    right_mask_idx = names.index("motion_right_valid")

    assert motion[1, left_idx] > 0
    assert motion[1, right_mask_idx] == 0

