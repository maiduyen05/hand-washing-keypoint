# METC Hand-washing Data Preparation

Project chuẩn bị dữ liệu landmark hai bàn tay cho bài toán nhận dạng sáu động tác
rửa tay theo WHO. Project gồm pipeline audit/preprocessing và bộ PyTorch GRU để
huấn luyện trên Kaggle GPU. Preprocessing đã hoàn tất; checkpoint và metrics chỉ
được tạo khi chạy notebook Kaggle.

## Nguồn dữ liệu

- Zenodo: <https://zenodo.org/records/5808789>
- DOI: `10.5281/zenodo.5808789`
- METC subset: 212 video, 72 participant, ba giao diện, FPS xấp xỉ 16.
- `statistics.csv` được kiểm tra bằng MD5
  `e5c66f07be277a5a79bbf11933470968`.

Hai CSV landmark hiện có chứa 211 video. Video có metadata trên Zenodo nhưng chưa
có trong dữ liệu local là `2021-08-12_08-50-43-59-3.mp4`.

### Mapping nhãn chính thức

| Nhãn | Ý nghĩa |
|---:|---|
| 0 | Other/incorrect hand-washing movement |
| 1 | Palm to palm |
| 2 | Palm over dorsum, fingers interlaced |
| 3 | Palm to palm, fingers interlaced |
| 4 | Backs of fingers to opposing palm |
| 5 | Rotational rubbing of thumb |
| 6 | Fingertips to palm |

Không tự động đổi nhãn `7` thành `0`. Pipeline sẽ dừng nếu gặp nhãn ngoài `0–6`.

## Vấn đề của dữ liệu cũ

- Split cũ được tạo ngẫu nhiên theo video. Do mỗi participant có nhiều video,
  32 trong 34 participant của test cũ cũng xuất hiện trong train.
- Presence mask từng tồn tại trong CSV nhưng notebook cũ không đưa mask vào model.
- Nội suy cũ có thể kéo dài qua nhiều frame và qua đoạn không liên tục.
- Một số cross-hand pair trùng tên làm schema Parquet không khớp số feature dự kiến.

Pipeline mới giữ CSV gốc bất biến, chia lại theo participant và không tạo window
bắc qua video, frame gap hoặc ranh giới nhãn.

## Cấu trúc

```text
DHB/
├── README.md
├── requirements.txt
├── run_pipeline.py
├── train.py
├── kaggle_train_gru.ipynb
├── configs/
│   ├── preprocessing.yaml
│   └── training.yaml
├── src/
│   ├── config.py
│   ├── dataset_metadata.py
│   ├── normalization.py
│   ├── imputation.py
│   ├── feature_engineering.py
│   ├── sequence_builder.py
│   ├── validation.py
│   ├── pipeline.py
│   ├── training_data.py
│   └── training.py
└── tests/
```

## Cài đặt và chạy

```bash
python -m pip install -r requirements.txt
python run_pipeline.py audit --config configs/preprocessing.yaml
python run_pipeline.py split --config configs/preprocessing.yaml
python run_pipeline.py prepare --config configs/preprocessing.yaml
python run_pipeline.py validate --config configs/preprocessing.yaml
```

`audit` tải metadata Zenodo và đối chiếu 2 CSV. `split` tạo split participant
50/11/11 và manifest 5-fold. `prepare` tạo feature Parquet theo từng video để giữ
RAM ổn định. `validate` kiểm tra leakage, schema, NaN/Inf và sequence continuity.

## Feature và missing

- `shape`: landmark chuẩn hóa riêng từng tay, bone vector và intra-hand distance.
- `relation`: hệ tọa độ chung hai tay, palm/tip/cross-hand relation và palm normal.
- `motion`: velocity, speed, direction và tốc độ thay đổi relation.
- Landmark chỉ được nội suy trong cùng segment khi gap không quá 0.25 giây.
- `left_present`, `right_present`, `left_imputed`, `right_imputed` và
  `cross_valid` luôn được giữ làm feature.

Các nhóm feature, window và đường dẫn đều chỉnh được trong
`configs/preprocessing.yaml`.

## Artifacts

```text
artifacts/preprocessing/
├── zenodo_statistics.csv
├── dataset_manifest.csv
├── split_manifest.csv
├── fold_manifest.csv
├── audit_report.json
├── features.parquet
├── window_index.parquet
├── feature_manifest.json
└── validation_report.json
```

`window_index.parquet` chỉ lưu vị trí window trong `features.parquet`, không tạo
mảng `(samples, frames, features)` nặng.

Hai file `train_hand_keypoints.csv` và `test_hand_keypoints.csv` là đầu vào gốc
của preprocessing. Model không đọc trực tiếp hai file này; model đọc
`features.parquet` và `window_index.parquet` đã có split participant-safe.

## Train GRU trên Kaggle

GitHub chỉ lưu code. Upload các artifacts đã chuẩn bị thành một Kaggle Dataset
Private, import `kaggle_train_gru.ipynb`, gắn dataset và bật GPU T4 x2. Hướng dẫn từng
bước nằm trong [KAGGLE_TRAINING.md](KAGGLE_TRAINING.md).

Lệnh notebook sử dụng:

```bash
python train.py \
  --config configs/training.yaml \
  --data-dir /kaggle/input/<dataset-slug> \
  --output-dir /kaggle/working/dhb_gru_full \
  --groups shape relation motion \
  --resume auto \
  --evaluate-test
```

Reaction margin 0,5 giây được áp dụng khi đọc window. Model chọn checkpoint theo
validation macro-F1 và chỉ đánh giá test sau khi model selection kết thúc.
`last_checkpoint.pt` và `history.csv` được cập nhật sau mỗi epoch để tiếp tục khi
runtime Kaggle bị ngắt.

`handwashing-preprocessing-data.ipynb` là notebook legacy dùng split/nội suy cũ;
không dùng notebook này để tạo metric mới.
