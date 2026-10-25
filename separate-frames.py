#!/usr/bin/env python3
import cv2
import os
import json
import csv
import random
import mediapipe as mp

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & THÔNG SỐ CŨ
# ==========================================
# input_folder = 'D:/handwashing/data/metc'
# output_folder = './RSU_METC_dataset_preprocessed/frames/'
# video_output_folder = './RSU_METC_dataset_preprocessed/videos/'
# csv_output_file = './RSU_METC_dataset_preprocessed/extracted_hand_keypoints.csv' # File CSV đầu ra

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN DÀNH CHO LINUX (WSL)
# ==========================================
input_folder = '/mnt/d/handwashing/data/metc'
output_folder = './RSU_METC_dataset_preprocessed/frames/'
video_output_folder = './RSU_METC_dataset_preprocessed/videos/'
# csv_output_file = './RSU_METC_dataset_preprocessed/extracted_hand_keypoints.csv' # File CSV đầu ra

train_csv_path = './RSU_METC_dataset_preprocessed/train_hand_keypoints.csv'
test_csv_path = './RSU_METC_dataset_preprocessed/test_hand_keypoints.csv'

TEST_PROPORTION = 0.25
TOTAL_MOVEMENTS = 7
TOTAL_ANNOTATORS = 1
FPS = 16
REACTION_TIME_FRAMES = FPS // 2
subdirs = ["Interface_number_1", "Interface_number_2", "Interface_number_3"]
snippet_extraction_file = "create_snippets.sh"
FULL_PROCESSING = True

# ==========================================
# 2. KHỞI TẠO MEDIAPIPE (CHUẨN TASKS API MỚI)
# ==========================================
# LƯU Ý: Phải đảm bảo file "hand_landmarker.task" nằm cùng thư mục với script này
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
if not os.path.exists(model_path):
    print(f"CẢNH BÁO: Không tìm thấy {model_path}. Vui lòng tải file mô hình về!")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Thay vì static_image_mode=True của API cũ, ta dùng RunningMode.IMAGE
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE, 
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Khởi tạo bộ nhận diện
landmarker = HandLandmarker.create_from_options(options)

# Khởi tạo danh sách lưu toàn bộ dòng dữ liệu
# dataset = []

os.makedirs(os.path.dirname(train_csv_path), exist_ok=True)
os.makedirs(os.path.dirname(test_csv_path), exist_ok=True)

# Mở 2 file CSV để ghi trực tiếp (On-the-fly)
f_train = open(train_csv_path, 'w', newline='', encoding='utf-8')
f_test = open(test_csv_path, 'w', newline='', encoding='utf-8')
writer_train = csv.writer(f_train)
writer_test = csv.writer(f_test)

# Khai báo tên cột cho CSV
columns = ['video_name', 'frame_index', 'timestamp', 'motion_label', 
           'left_hand_detected', 'right_hand_detected', 
           'left_hand_confidence', 'right_hand_confidence']
for i in range(21):
    columns.extend([f'left_p{i}_x', f'left_p{i}_y', f'left_p{i}_z'])
for i in range(21):
    columns.extend([f'right_p{i}_x', f'right_p{i}_y', f'right_p{i}_z'])

# Ghi ngay dòng tiêu đề (Header) vào 2 file
writer_train.writerow(columns)
writer_test.writerow(columns)

# ... [GIỮ NGUYÊN CÁC HÀM CŨ: majority_vote, mk, discount_reaction_indeterminacy, find_frame_labels, select_frames_to_save] ...

def majority_vote(lst):
    """Returns the element present in majority of the list, or -1 otherwise
    """
    counts = [0] * TOTAL_MOVEMENTS
    for el in lst:
        counts[int(el)] += 1
    best = 0
    for i in range(1, TOTAL_MOVEMENTS):
        if counts[best] < counts[i]:
            best = i
    majority = (len(lst) + 2) // 2
    if counts[best] < majority:
        return -1
    return best

def mk(directory):
    os.makedirs(directory, exist_ok=True)

def discount_reaction_indeterminacy(labels):
    new_labels = [u for u in labels]
    n = len(labels) - 1
    for i in range(n):
        if i == 0 or labels[i] != labels[i+1] or i == n - 1:
            start = max(0, i - REACTION_TIME_FRAMES)
            end = i
            for j in range(start, end):
                new_labels[j] = -1
            start = i
            end = min(n + 1, i + REACTION_TIME_FRAMES)
            for j in range(start, end):
                new_labels[j] = -1
    return new_labels


def find_frame_labels(fullpath):
    """Returns `is_washing` status and movement codes for each frame
    """
    filename = os.path.basename(fullpath)
    annotators_dir = os.path.join(os.path.dirname(os.path.dirname(fullpath)), "Annotations")

    annotations = []

    # Load the supplementary info, if present.
    # This info is not part of the public dataset;
    # it is hand presence information, generated using the Mediapipe hand tracking neural network
    supplementary_dir = os.path.join(os.path.dirname(os.path.dirname(fullpath)), "Supplementary")
    supplementary_filename = os.path.join(supplementary_dir, "hands-" + filename + ".txt")
    frames_with_hands = []
    if os.access(supplementary_filename, os.R_OK):
        with open(supplementary_filename, "r") as f:
            for line in f.readlines():
                try:
                    i = int(line)
                except Exception as ex:
                    break
                frames_with_hands.append(i)

    for a in range(1, TOTAL_ANNOTATORS + 1):
        annotator_dir = os.path.join(annotators_dir, "Annotator_" + str(a))
        json_filename = os.path.join(annotator_dir, filename.split(".")[0] + ".json")

        if os.access(json_filename, os.R_OK):
            with open(json_filename, "r") as f:
                try:
                  data = json.load(f)
                except:
                  print("failed to load {}".format(json_filename))
                  continue
                a_annotations = [(data['labels'][i]['is_washing'], data['labels'][i]['code']) for i in range(len(data['labels']))]
                annotations.append(a_annotations)

    num_annotators = len(annotations)
    num_frames = len(annotations[0])
    is_washing = []
    codes = []
    for frame_num in range(num_frames):
        frame_annotations = [annotations[a][frame_num] for a in range(num_annotators)]
        frame_is_washing_any = any([frame_annotations[a][0] for a in range(num_annotators)])
        frame_is_washing_all = all([frame_annotations[a][0] for a in range(num_annotators)])
        frame_codes = [frame_annotations[a][1] for a in range(num_annotators)]
        # treat movement 7 as movement 0
        frame_codes = [0 if code == 7 else code for code in frame_codes]

        if frame_is_washing_all:
            frame_is_washing = 2
        elif frame_is_washing_any:
            frame_is_washing = 1
        else:
            frame_is_washing = 0

        is_washing.append(frame_is_washing)
        if frame_is_washing:
            codes.append(majority_vote(frame_codes))
        else:
            codes.append(-1)

    if len(frames_with_hands):
        if len(frames_with_hands) != len(is_washing):
            if len(frames_with_hands) > len(is_washing):
                print("Incorrect dimensions of the supplementary information: {} vs {}".format(len(frames_with_hands), len(is_washing)))
                frames_with_hands = []
            else:
                # pad with zeroes (no hands detected)
                pad_len = len(is_washing) - len(frames_with_hands)
                frames_with_hands += [0] * pad_len

    if True:
        is_washing = discount_reaction_indeterminacy(is_washing)
        codes = discount_reaction_indeterminacy(codes)

    return is_washing, codes, frames_with_hands, num_annotators


def select_frames_to_save(is_washing, codes):
    old_code = -1
    old_saved = False
    num_snippets = 0
    mapping = {}
    current_snippet = {}
    for i in range(len(is_washing)):
        new_code = codes[i]
        new_saved = (is_washing[i] == 2 and new_code != -1)
        if new_saved != old_saved:
            if new_saved:
                num_snippets += 1
                current_snippet = {}
            else:
                # save current snippet
                for key in current_snippet:
                    mapping[key] = current_snippet[key]

        if new_saved:
            current_snippet_frame = len(current_snippet)
            current_snippet[i] = (current_snippet_frame, num_snippets, new_code)
        old_saved = new_saved
        old_code = new_code

    if old_saved:
        # save current snippet
        for key in current_snippet:
            mapping[key] = current_snippet[key]

    return mapping

# ==========================================
# 3. HÀM MỚI: TRÍCH XUẤT MEDIAPIPE & GHI TRỰC TIẾP CSV
# ==========================================
def process_landmarks_to_dataset(image_bgr, video_name, frame_index, timestamp, label, traintest):
    """
    Hàm nhận ảnh đang nằm trên RAM, chạy MediaPipe (Tasks API) và ghi thẳng xuống file
    """
    row_data = [video_name, frame_index, timestamp, label, 0, 0, 0.0, 0.0] + [0.0] * 126

    # Đổi BGR (OpenCV) sang RGB (MediaPipe) và bọc thành định dạng mp.Image
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Chạy nhận diện
    results = landmarker.detect(mp_image)

    if results.hand_landmarks:
        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            handedness = results.handedness[idx][0]
            hand_label = str(handedness.category_name).lower()
            confidence = round(float(handedness.score), 4)

            if hand_label == 'left':
                row_data[4] = 1 
                row_data[6] = confidence 
                for i, lm in enumerate(hand_landmarks):
                    row_data[8 + i*3] = lm.x
                    row_data[8 + i*3 + 1] = lm.y
                    row_data[8 + i*3 + 2] = lm.z
            elif hand_label == 'right':
                row_data[5] = 1 
                row_data[7] = confidence 
                for i, lm in enumerate(hand_landmarks):
                    row_data[71 + i*3] = lm.x
                    row_data[71 + i*3 + 1] = lm.y
                    row_data[71 + i*3 + 2] = lm.z

    # GHI TRỰC TIẾP VÀO FILE THAY VÌ ĐƯA VÀO LIST
    if traintest == 'test':
        writer_test.writerow(row_data)
    else:
        writer_train.writerow(row_data)


# ==========================================
# 4. SỬA HÀM get_frames ĐỂ NHÚNG MEDIAPIPE
# ==========================================
def get_frames(folder):
    print('Processing folder: ' + folder + ' ...')
    for subdir, dirs, files in os.walk(folder):
        for videofile in files:
            if videofile.endswith(".mp4"):
                fullpath = os.path.join(subdir, videofile)
                is_washing, codes, frames_with_hands, num_annotators = find_frame_labels(fullpath)

                if random.random() < TEST_PROPORTION:
                    traintest = "test"
                else:
                    traintest = "trainval"

                new_frame_numbers = select_frames_to_save(is_washing, codes)

                if not FULL_PROCESSING:
                    continue

                vidcap = cv2.VideoCapture(fullpath)
                is_success, image = vidcap.read()
                frame_number = 0

                while is_success:
                    if frame_number in new_frame_numbers:
                        new_frame_num, snippet_num, code = new_frame_numbers[frame_number]
                        assert code == codes[frame_number]
                        
                        subfolder = str(codes[frame_number])
                        filename = 'frame_{}_snippet_{}_{}.jpg'.format(new_frame_num, snippet_num, os.path.splitext(videofile)[0])
                        save_path_and_name = os.path.join(output_folder, traintest, subfolder, filename)
                        
                        # VẪN LƯU ẢNH RA DISK (Bạn có thể comment dòng này nếu không cần lưu ảnh JPG nữa)
                        cv2.imwrite(save_path_and_name, image)
                        
                        # ---> INJECT MEDIAPIPE TẠI ĐÂY <---
                        # Tính timestamp thực tế theo FPS
                        timestamp = round(frame_number / FPS, 4)
                        
                        # Chạy MediaPipe và đẩy data vào list
                        process_landmarks_to_dataset(
                            image_bgr=image, 
                            video_name=videofile, 
                            frame_index=frame_number, # Dùng frame_number gốc làm index cho chuẩn
                            timestamp=timestamp, 
                            label=code,
                            traintest=traintest
                        )

                    is_success, image = vidcap.read()
                    frame_number += 1

def main():
    random.seed(0)

    for movement in range(TOTAL_MOVEMENTS):
        mk(os.path.join(output_folder, "test", str(movement)))
        mk(os.path.join(output_folder, "trainval", str(movement)))
        mk(os.path.join(video_output_folder, "test", str(movement)))
        mk(os.path.join(video_output_folder, "trainval", str(movement)))

    for subdir in subdirs:
        get_frames(os.path.join(input_folder, subdir))
        
    # ==========================================
    # 5. KẾT THÚC VÀ ĐÓNG FILE
    # ==========================================
    f_train.close()
    f_test.close()
    print(f"\nXong! Dữ liệu đã được ghi an toàn và tách biệt vào:")
    print(f" - {train_csv_path}")
    print(f" - {test_csv_path}")
    
if __name__ == "__main__":
    main()