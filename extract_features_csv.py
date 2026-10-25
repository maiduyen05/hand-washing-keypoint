"""
Doc file CSV landmark tho (tu MediaPipe) -> tinh feature vector (cross-hand +
bone + intra-hand distance, theo 1 mode chuan hoa) -> ghi thang xuong file
output theo tung dong / tung chunk nho, KHONG load ca file vao RAM.

Cach dung:
    python extract_features_csv.py input.csv output.parquet --mode location_scale
    python extract_features_csv.py input.csv output.csv     --mode location_scale --format csv
"""

import argparse
import csv
import numpy as np

from normalization import (
    NUM_LANDMARKS,
    HAND_CONNECTIONS,
    INTRA_HAND_PAIRS,
    CROSS_HAND_PAIRS,
    normalize_keypoints,
    compute_bone_vectors,
    compute_intra_hand_distances,
    compute_cross_hand_distances,
    build_extended_feature_vector,
)

META_COLS = [
    "video_name", "frame_index", "timestamp", "motion_label",
    "left_hand_detected", "right_hand_detected",
    "left_hand_confidence", "right_hand_confidence",
]

N_CROSS = len(CROSS_HAND_PAIRS)
N_BASE = NUM_LANDMARKS * 3
N_BONE = len(HAND_CONNECTIONS) * 3
N_DIST = len(INTRA_HAND_PAIRS)


def build_feature_column_names() -> list:
    cols = [f"cross_L{l}_R{r}" for l, r in CROSS_HAND_PAIRS]
    for side in ["left", "right"]:
        cols += [f"{side}_norm_p{i}_{axis}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")]
        cols += [f"{side}_bone_{p1}_{p2}_{axis}" for p1, p2 in HAND_CONNECTIONS for axis in ("x", "y", "z")]
        cols += [f"{side}_dist_p{p1}_p{p2}" for p1, p2 in INTRA_HAND_PAIRS]
    return cols


def parse_bool(value: str) -> bool:
    """Doc cot detected tu CSV (co the la '1', '1.0', 'True'...) -> bool."""
    return str(value).strip().lower() in ("1", "1.0", "true")


def parse_hand_coords(row: dict, side: str) -> np.ndarray:
    """Gom 21x3 cot phang thanh mang (21,3), CHI goi khi hand da duoc detect."""
    coords = np.empty((NUM_LANDMARKS, 3), dtype=np.float32)
    for i in range(NUM_LANDMARKS):
        coords[i, 0] = float(row[f"{side}_p{i}_x"])
        coords[i, 1] = float(row[f"{side}_p{i}_y"])
        coords[i, 2] = float(row[f"{side}_p{i}_z"])
    return coords


def run_csv_streaming(input_csv: str, output_path: str, mode: str) -> int:
    """Doc - tinh - ghi TUNG DONG MOT. Bo nho dung on dinh du file to bao nhieu."""
    feature_cols = build_feature_column_names()
    out_fieldnames = META_COLS + feature_cols

    n_rows = 0
    with open(input_csv, "r", newline="") as fin, open(output_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=out_fieldnames)
        writer.writeheader()

        for row in reader:
            left_detected = parse_bool(row["left_hand_detected"])
            right_detected = parse_bool(row["right_hand_detected"])

            # Neu tay khong duoc detect, giu nguyen khoi feature cua tay do = 0.
            if left_detected:
                raw_left = parse_hand_coords(row, "left")
                # base_left = normalize_keypoints(raw_left, mode)
                # bone_left = compute_bone_vectors(base_left)
                # dist_left = compute_intra_hand_distances(base_left)
            else:
                raw_left = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
                # base_left = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
                # bone_left = np.zeros(N_BONE, dtype=np.float32)
                # dist_left = np.zeros(N_DIST, dtype=np.float32)

            if right_detected:
                raw_right = parse_hand_coords(row, "right")
                # base_right = normalize_keypoints(raw_right, mode)
                # bone_right = compute_bone_vectors(base_right)
                # dist_right = compute_intra_hand_distances(base_right)
            else:
                raw_right = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
                # base_right = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
                # bone_right = np.zeros(N_BONE, dtype=np.float32)
                # dist_right = np.zeros(N_DIST, dtype=np.float32)

            feat = build_extended_feature_vector(
                raw_left,
                raw_right,
                mode=mode,
                left_hand_detected=left_detected,
                right_hand_detected=right_detected,
            )

            # # Logic xu ly thieu tay nam trong compute_cross_hand_distances.
            # cross = compute_cross_hand_distances(
            #     raw_left,
            #     raw_right,
            #     left_hand_detected=left_detected,
            #     right_hand_detected=right_detected,
            # )

            # feat = np.concatenate([
            #     cross,
            #     # base_left.reshape(-1), bone_left, dist_left,
            #     # base_right.reshape(-1), bone_right, dist_right,
            # ]).astype(np.float32)

            # Một mẹo nhỏ: Dùng get(col, None) thay vì get(col, "") để tương thích Parquet tốt hơn
            out_row = {col: row.get(col, None) for col in META_COLS}
            out_row.update(zip(feature_cols, feat.tolist()))
            writer.writerow(out_row)
            n_rows += 1

    return n_rows


def run_parquet_chunked(input_csv: str, output_path: str, mode: str, chunk_size: int = 5000) -> int:
    """Doc - tinh - ghi theo TUNG LO NHO (chunk_size dong), dung ParquetWriter
    de ghi noi tiep nhieu lo vao 1 file duy nhat -> bo nho khong phinh to
    bat ke file dau vao lon co nao, khac voi cach load ca file roi to_parquet.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    feature_cols = build_feature_column_names()
    out_fieldnames = META_COLS + feature_cols

    writer = None
    buffer = []
    n_rows = 0

    def flush(buffer):
        nonlocal writer
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
        buffer.clear()

    with open(input_csv, "r", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            left_detected = parse_bool(row["left_hand_detected"])
            right_detected = parse_bool(row["right_hand_detected"])

            # Neu tay khong duoc detect, giu nguyen khoi feature cua tay do = 0.
            if left_detected:
                raw_left = parse_hand_coords(row, "left")
                # base_left = normalize_keypoints(raw_left, mode)
                # bone_left = compute_bone_vectors(base_left)
                # dist_left = compute_intra_hand_distances(base_left)
            else:
                raw_left = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
                # base_left = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
                # bone_left = np.zeros(N_BONE, dtype=np.float32)
                # dist_left = np.zeros(N_DIST, dtype=np.float32)

            if right_detected:
                raw_right = parse_hand_coords(row, "right")
                # base_right = normalize_keypoints(raw_right, mode)
                # bone_right = compute_bone_vectors(base_right)
                # dist_right = compute_intra_hand_distances(base_right)
            else:
                raw_right = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
                # base_right = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
                # bone_right = np.zeros(N_BONE, dtype=np.float32)
                # dist_right = np.zeros(N_DIST, dtype=np.float32)

            feat = build_extended_feature_vector(
                raw_left,
                raw_right,
                mode=mode,
                left_hand_detected=left_detected,
                right_hand_detected=right_detected,
            )
            
            # # Logic xu ly thieu tay nam trong compute_cross_hand_distances.
            # cross = compute_cross_hand_distances(
            #     raw_left,
            #     raw_right,
            #     left_hand_detected=left_detected,
            #     right_hand_detected=right_detected,
            # )

            # feat = np.concatenate([
            #     cross,
            #     # base_left.reshape(-1), bone_left, dist_left,
            #     # base_right.reshape(-1), bone_right, dist_right,
            # ]).astype(np.float32)
            out_row = {col: row.get(col, None) for col in META_COLS}
            out_row.update(zip(feature_cols, feat.tolist()))
            buffer.append(out_row)
            n_rows += 1
            if len(buffer) >= chunk_size:
                flush(buffer)
        flush(buffer)  # ghi not phan con lai

    if writer is not None:
        writer.close()
    return n_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_path")
    parser.add_argument("--mode", default="location_scale",
                         choices=["raw", "location", "location_scale"])
    parser.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    parser.add_argument("--chunk-size", type=int, default=5000,
                         help="Chi dung cho --format parquet")
    args = parser.parse_args()

    if args.format == "csv":
        n = run_csv_streaming(args.input_csv, args.output_path, args.mode)
    else:
        n = run_parquet_chunked(args.input_csv, args.output_path, args.mode, args.chunk_size)

    print(f"Da xu ly {n} dong -> {args.output_path} (mode={args.mode}, format={args.format})")


if __name__ == "__main__":
    main()