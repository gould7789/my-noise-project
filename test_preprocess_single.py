import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

# ==========================
# 설정
# ==========================
SEGMENT_DURATION = 5  # 초 단위 세그먼트

# 저장 폴더 생성
os.makedirs("features", exist_ok=True)
os.makedirs("spectrograms", exist_ok=True)

# ==========================
# 특성 추출 및 저장 함수
# ==========================
def extract_features(segment, sr, base_filename, seg_idx):
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(segment)

    # (14, time) 형태로 결합
    features = np.vstack([mfcc, zcr])

    # --- features 저장 (.npy)
    npy_path = f"features/{base_filename}_seg{seg_idx}.npy"
    np.save(npy_path, features)

    # --- 스펙트로그램 이미지 저장
    plt.figure(figsize=(10, 4))
    plt.imshow(features, aspect='auto', origin='lower', cmap='coolwarm')
    plt.colorbar()
    plt.xlabel("Time")
    plt.ylabel("Feature Index")
    plt.title(f"MFCC+ZCR - {base_filename}_seg{seg_idx}")
    plt.tight_layout()
    png_path = f"spectrograms/{base_filename}_seg{seg_idx}_mfcc_zcr.png"
    plt.savefig(png_path, dpi=300)
    plt.close()

# ==========================
# 전체 전처리 함수
# ==========================
def preprocess_single_file(file_path):
    filename = os.path.splitext(os.path.basename(file_path))[0]
    y, sr = librosa.load(file_path, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)

    for i in range(0, int(total_duration), SEGMENT_DURATION):
        start = i * sr
        end = min((i + SEGMENT_DURATION) * sr, len(y))
        segment = y[start:end]

        # padding (마지막 세그먼트 짧을 때)
        if len(segment) < SEGMENT_DURATION * sr:
            padding = SEGMENT_DURATION * sr - len(segment)
            segment = np.pad(segment, (0, padding), mode='constant')

        extract_features(segment, sr, filename, i // SEGMENT_DURATION)

# ==========================
# 실행 예시
# ==========================
if __name__ == "__main__":
    test_file = "data/ambiguous/pen_knock.wav"  # 원하는 파일 경로로 변경
    preprocess_single_file(test_file)