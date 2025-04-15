import os
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

# ================================
# 기본 설정
# ================================
SAMPLE_RATE = 22050
SEGMENT_DURATION = 5.0
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)
N_MFCC = 13
MAX_LEN = 130

# ================================
# 특징 추출 함수 (MFCC + ZCR)
# ================================
def extract_features(segment, sr):
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
    zcr = librosa.feature.zero_crossing_rate(y=segment)
    features = np.vstack([mfcc, zcr])
    if features.shape[1] < MAX_LEN:
        features = np.pad(features, ((0,0), (0, MAX_LEN - features.shape[1])), mode='constant')
    else:
        features = features[:, :MAX_LEN]
    return features.T

# ================================
# 폴더 처리 함수
# ================================
def process_directory(folder_path, label):
    X, y = [], []
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3', '.m4a'))])

    for fname in tqdm(file_list, desc=f"[{label}] Processing"):
        path = os.path.join(folder_path, fname)
        try:
            signal, sr = sf.read(path)
            if sr != SAMPLE_RATE:
                signal = librosa.resample(signal, orig_sr=sr, target_sr=SAMPLE_RATE)
        except Exception as e:
            print(f"❌ 파일 무시됨: {os.path.basename(path)} → {e}")
            continue

        total_segments = len(signal) // SEGMENT_SAMPLES

        for i in range(total_segments):
            start = i * SEGMENT_SAMPLES
            end = start + SEGMENT_SAMPLES
            segment = signal[start:end]
            if len(segment) < SEGMENT_SAMPLES:
                padding = SEGMENT_SAMPLES - len(segment)
                segment = np.pad(segment, (0, padding), mode='constant')

            features = extract_features(segment, SAMPLE_RATE)
            X.append(features)
            y.append(label)

    return X, y

# ================================
# 메인 실행
# ================================
if __name__ == "__main__":
    X, y = [], []

    # 조용한 소리
    X_q, y_q = process_directory("data/quiet", label=0)
    X.extend(X_q)
    y.extend(y_q)

    # 시끄러운 소리
    X_l, y_l = process_directory("data/loud", label=1)
    X.extend(X_l)
    y.extend(y_l)

    # # 애매한 소리 (원할 경우 주석 해제)
    X_a, y_a = process_directory("data/ambiguous", label=2)
    X.extend(X_a)
    y.extend(y_a)

    X = np.array(X)
    y = np.array(y)

    os.makedirs("output", exist_ok=True)
    np.save("output/X_lstm.npy", X)
    np.save("output/y_lstm.npy", y)

    print("\n✅ 전처리 완료!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)