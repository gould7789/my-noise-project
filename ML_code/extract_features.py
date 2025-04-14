import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# 🔧 설정값
SAMPLE_RATE = 22050              # 오디오 샘플링 레이트
SEGMENT_DURATION = 5.0           # 한 segment의 길이 (초 단위)
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)  # segment당 샘플 수
N_MFCC = 13                      # MFCC 개수
MAX_LEN = 130                    # 각 segment의 프레임 수 (패딩을 위해 고정)

# 🔍 특징 추출 함수: MFCC + ZCR → (N, 14) 형태로 변환
def extract_features(segment, sr):
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)             # MFCC (13개)
    zcr = librosa.feature.zero_crossing_rate(y=segment)                      # ZCR (1개)
    features = np.vstack([mfcc, zcr])                                        # (14, 시간 프레임)
    
    # 프레임 수 맞추기 (길이가 다르면 0으로 패딩)
    if features.shape[1] < MAX_LEN:
        features = np.pad(features, ((0,0), (0,MAX_LEN - features.shape[1])), mode='constant')
    else:
        features = features[:, :MAX_LEN]  # 초과 프레임 자르기
    
    return features.T  # (시간 프레임, 14)

# 📂 폴더 하나 처리 (폴더 내 파일을 segment로 나누고 특징 추출)
def process_directory(folder_path, label):
    X, y = [], []
    for fname in tqdm(os.listdir(folder_path)):  # tqdm: 진행률 표시
        if not fname.endswith(('.wav', '.mp3', '.m4a')):
            continue

        path = os.path.join(folder_path, fname)
        try:
            signal, sr = librosa.load(path, sr=SAMPLE_RATE)  # 오디오 로드
        except Exception as e:
            print(f"Error: {fname} → {e}")
            continue
        
        total_segments = len(signal) // SEGMENT_SAMPLES

        for i in range(total_segments):
            start = i * SEGMENT_SAMPLES
            end = start + SEGMENT_SAMPLES
            segment = signal[start:end]

            features = extract_features(segment, sr)
            X.append(features)
            y.append(label)

    return X, y

# 🚀 메인 실행 코드
if __name__ == "__main__":
    X, y = [], []

    # 조용한 소리 처리 (label = 0)
    X_q, y_q = process_directory('../data/quiet', label=0)
    X.extend(X_q)
    y.extend(y_q)

    # 시끄러운 소리 처리 (label = 1)
    X_L, y_L = process_directory('../data/loud', label=1)
    X.extend(X_L)
    y.extend(y_L)

    # # 애매한 소리 처리 (label = 2)
    # X_a, y_a = process_directory('../data/ambiguous', label=2)
    # X.extend(X_a)
    # y.extend(y_a)

    # 최종 배열로 변환
    X = np.array(X)
    y = np.array(y)

    # 💾 저장
    os.makedirs('./output', exist_ok=True)
    np.save('./output/X_lstm.npy', X)
    np.save('./output/y_lstm.npy', y)

    print("✅ Pre-processing completed!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)