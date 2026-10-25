import numpy as np

# phương pháp chuẩn hóa chỉ áp dụng riêng cho từng bàn tay
# tức là mỗi bàn tay có hệ quy chiếu gốc là cổ tay của chính nó

# => nếu đo khoảng cách giữa 2 tay thì phải tính trên tọa độ gốc (raw)

# tương ứng với ngón cái, ngón trỏ, ngón giữa, ngón áp út, ngón út
NUM_LANDMARKS = 21

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (17, 18), (18, 19), (19, 20)]

INTRA_HAND_PAIRS = [
            (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),   # cổ tay -> từng đầu ngón
            (4, 8), (4, 12), (4, 16), (4, 20),           # ngón cái -> các đầu ngón khác
            (8, 12), (12, 16), (16, 20),                 # giữa các đầu ngón liền kề nhau
        ]

CROSS_HAND_PAIRS = [
            # bước 1: chà 2 lòng bàn tay vào nhau
            (0, 0),                      # cổ tay trái <-> cổ tay phải
            (9, 9), (5, 5), (13, 13), (17, 17),            # giữa lòng bàn tay trái <-> giữa lòng bàn tay phải
            
            # tại có thể đặt 2 tay vuông góc để chà 2 lòng bàn tay vào nhau :))
            # (4, 4), (8, 8), (12, 12), (16, 16), (20, 20),  # đầu ngón trái <-> đầu ngón phải cùng loại

            # bước 2: chà lòng bàn tay này lên mu bàn tay kia
            (20, 1), (16, 5), (12, 9), (8, 13), (4, 17),  
            (1, 20), (5, 16), (9, 12), (13, 8), (17, 4),

            # bước 3: chà 2 lòng bàn tay vào nhau, miết ngón
            # Đầu ngón tay này luồn vào kẽ (MCP) của tay kia
            # Đầu ngón Trái luồn kẽ Phải
            (8, 5), (12, 9), (16, 13), (20, 17),
            # Đầu ngón Phải luồn kẽ Trái
            (5, 8), (9, 12), (13, 16), (17, 20),

            # bước 4: chà mặt ngoài các ngón tay
            # Mặt ngoài ngón Trái cọ lòng bàn tay Phải
            (6, 9), (10, 9), (14, 9), (18, 9),
            # Mặt ngoài ngón Phải cọ lòng bàn tay Trái
            (9, 6), (9, 10), (9, 14), (9, 18),

            # bước 5: chà ngón cái
            (4, 9), (4, 5),   # Ngón cái Trái bị tay Phải nắm
            (9, 4), (5, 4),   # Ngón cái Phải bị tay Trái nắm

            # bước 6: chụm đầu ngón tay vào lòng bàn tay
            # Đầu ngón Trái chụm vào lòng bàn tay Phải
            (8, 9), (12, 9), (16, 9), (20, 9),
            # Đầu ngón Phải chụm vào lòng bàn tay Trái
            (9, 8), (9, 12), (9, 16), (9, 20)
        ]

def location_normalize(keypoints: np.ndarray) -> np.ndarray:
    """Trừ tọa độ cổ tay (point 0) để loại bỏ vị trí di chuyển trong khung hình."""
    if keypoints.shape[0] == 0:
        return keypoints.copy()
    wrist = keypoints[0].astype(np.float32)
    return (keypoints - wrist).astype(np.float32)


def location_scale_normalize(keypoints: np.ndarray) -> np.ndarray:
    """Chuẩn hóa vị trí và quy mô bằng cách chia cho giá trị max tuyệt đối."""
    normalized = location_normalize(keypoints)
    max_abs = np.max(np.abs(normalized))
    if max_abs < 1e-8:
        return normalized
    return (normalized / max_abs).astype(np.float32)

def normalize_keypoints(keypoints: np.ndarray, mode: str = "raw") -> np.ndarray:
    """Hàm gộp chọn 1 trong 3 cấu hình, dùng cho dễ so sánh thực nghiệm.
    mode: "raw" | "location" | "location_scale"
    """
    if mode == "raw":
        return keypoints.astype(np.float32).copy()
    elif mode == "location":
        return location_normalize(keypoints)
    elif mode == "location_scale":
        return location_scale_normalize(keypoints)
    raise ValueError(f"mode không hợp lệ: {mode}")

# bone vectors

def compute_bone_vectors(keypoints: np.ndarray, connections=None) -> np.ndarray:
    """Tính vector xương giữa các điểm nối kề nhau."""
    if connections is None:
        connections = HAND_CONNECTIONS

    vectors = []
    for p1, p2 in connections:
        vec = keypoints[p2] - keypoints[p1]
        vectors.append(vec)
    return np.array(vectors, dtype=np.float32).reshape(-1)

# tính khoảng cách giữa các cặp điểm quan trọng

# """Tính khoảng cách giữa các cặp điểm quan trọng.
# Ví dụ khi xoa 2 bàn tay vào nhau thì khoảng cách giữa các đầu ngón tay của cả 2 tay sẽ gần nhau
# Nếu xét theo các bước:
# - palm to palm: 0-0, 1-1, 2-2,...
# - chà lưng bàn tay và đan ngón tay: đầu ngón tay trái sẽ gần với kẽ tay của tay phải.
# - nắm và xoay quanh 1 ngón
# - chụm 5 đầu ngón vào lòng bàn tay còn lại

def compute_intra_hand_distances(keypoints: np.ndarray, pairs=None) -> np.ndarray:
    """Khoảng cách giữa các cặp điểm KHÔNG liền kề trong CÙNG một bàn tay.
 
    Các cặp phản ánh HÌNH DẠNG bàn tay (đầu ngón - đầu ngón, cổ tay - đầu ngón) 
    để phân biệt xòe tay/nắm tay/chụm ngón.
    """
    if pairs is None:
        pairs = INTRA_HAND_PAIRS
    distances = [np.linalg.norm(keypoints[p2] - keypoints[p1]) for p1, p2 in pairs]
    return np.array(distances, dtype=np.float32)
 
# ---------------------------------------------------------------------------
# đặc trưng 2 tay - luôn nhận đầu vào là tọa độ gốc,
# vì mỗi tay không được ở trong hệ quy chiếu riêng khi so khoảng cách.
# ---------------------------------------------------------------------------
 
def compute_cross_hand_distances(
    raw_left: np.ndarray,
    raw_right: np.ndarray,
    pairs=None,
    left_hand_detected: bool = True,
    right_hand_detected: bool = True,
) -> np.ndarray:
    """Khoảng cách liên tay, tính trên tọa độ gốc của cả 2 tay (cùng 1 hệ quy chiếu camera).
 
    Các cặp mặc định lấy theo đúng các 'cặp điểm quan trọng' bạn mô tả cho
    từng bước:
      - cổ tay <-> cổ tay: bước palm-to-palm
      - đầu ngón <-> đầu ngón cùng loại: bước palm-to-palm / ma sát nhịp nhàng
      - đầu ngón <-> kẽ ngón (MCP) tay đối diện: bước đan ngón (fingers interlaced)
      - đầu ngón cái <-> lòng bàn tay đối diện: bước xoay ngón cái
      - chụm 5 đầu ngón <-> giữa lòng bàn tay đối diện: bước fingertips-to-palm
    """
    if pairs is None:
        pairs = CROSS_HAND_PAIRS

    # Nếu thiếu 1 trong 2 tay thì trả mảng 0 cho toàn bộ đặc trưng cross-hand.
    if not left_hand_detected or not right_hand_detected:
        return np.zeros(len(pairs), dtype=np.float32)

    if raw_left.shape[0] == 0 or raw_right.shape[0] == 0:
        return np.zeros(len(pairs), dtype=np.float32)
 
    distances = [np.linalg.norm(raw_right[r] - raw_left[l]) for l, r in pairs]
    return np.array(distances, dtype=np.float32)

 
def build_extended_feature_vector(
    raw_left: np.ndarray,
    raw_right: np.ndarray,
    mode: str = "location_scale",
    left_hand_detected: bool = True,
    right_hand_detected: bool = True,
) -> np.ndarray:
    """Tạo feature vector đầy đủ cho 1 frame gồm 2 tay.
 
    mode: "raw" | "location" | "location_scale"  (3 cấu hình thực nghiệm)
 
    Thứ tự các khối feature:
      [cross-hand distances (raw)]  +
      [left: base_norm + bone + intra-dist]  +
      [right: base_norm + bone + intra-dist]
    """
    # 1) Cross-hand distances PHẢI tính trên RAW, trước khi normalize riêng từng tay
    cross = compute_cross_hand_distances(
        raw_left,
        raw_right,
        left_hand_detected=left_hand_detected,
        right_hand_detected=right_hand_detected,
    )
 
    # 2) Chuẩn hóa riêng từng tay theo cấu hình được chọn
    base_left = normalize_keypoints(raw_left, mode)
    base_right = normalize_keypoints(raw_right, mode)
 
    bone_left = compute_bone_vectors(base_left)
    bone_right = compute_bone_vectors(base_right)
 
    dist_left = compute_intra_hand_distances(base_left)
    dist_right = compute_intra_hand_distances(base_right)
 
    return np.concatenate(
        [
            cross,
            base_left.reshape(-1), bone_left, dist_left,
            base_right.reshape(-1), bone_right, dist_right,
        ],
        axis=0,
    ).astype(np.float32)