import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import math

# ================================
# 기본 설정
# ================================
sr = 22050
n_mfcc = 13
max_len = 130
segment_duration = 5.0
segment_samples = int(sr * segment_duration)
label_map = {'quiet': 0, 'loud': 1, 'ambiguous': 2}

# ================================
# MFCC + ZCR 특징 추출 함수
# ================================
def extract_features(segment, sr):
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    zcr = librosa.feature.zero_crossing_rate(y=segment)
    features = np.vstack([mfcc, zcr])
    if features.shape[1] < max_len:
        features = np.pad(features, ((0,0), (0, max_len - features.shape[1])), mode='constant')
    else:
        features = features[:, :max_len]
    return features.T[..., np.newaxis]  # (130, 14, 1)

# ================================
# 단일 파일 예측 함수
# ================================
def predict_file(model, filepath):
    signal, _ = librosa.load(filepath, sr=sr)
    total_segments = math.ceil(len(signal) / segment_samples)
    preds = []

    for i in range(total_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = signal[start:end]

        # 짧은 구간은 패딩
        if len(segment) < segment_samples:
            padding = segment_samples - len(segment)
            segment = np.pad(segment, (0, padding), mode='constant')

        feature = extract_features(segment, sr)
        feature = np.expand_dims(feature, axis=0)  # 배치 차원 추가

        prob = model.predict(feature, verbose=0)[0][0]
        pred = int(prob >= 0.5)  # sigmoid 기준 이진 분류
        preds.append((pred, prob))

    # 세그먼트들의 평균 확률로 최종 예측
    avg_prob = np.mean([p[1] for p in preds])
    final_pred = int(avg_prob >= 0.5)
    return final_pred, avg_prob

# ================================
# 전체 평가 함수
# ================================
def evaluate(model_path='output/cnn_lstm_model.h5', folder_path='data'):
    model = tf.keras.models.load_model(model_path)
    y_true, y_pred = [], []

    for category in os.listdir(folder_path):
        label = label_map.get(category)
        category_path = os.path.join(folder_path, category)
        if not os.path.isdir(category_path):
            continue

        for fname in os.listdir(category_path):
            if not fname.endswith(".wav"):
                continue

            filepath = os.path.join(category_path, fname)
            pred_label, prob = predict_file(model, filepath)

            print(f"{fname} → 예측={pred_label} (확률={prob:.2f}), 실제={label}")
            y_true.append(label)
            y_pred.append(pred_label)

    if not y_true:
        print("❌ 예측된 데이터가 없습니다. wav 파일이 충분한지 확인하세요.")
        return

    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ 전체 정확도: {acc * 100:.2f}%")

    # 혼동 행렬
    labels = ["quiet", "loud", "ambiguous"]
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, "output")
    os.makedirs(output_path, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("예측 라벨")
    plt.ylabel("실제 라벨")
    plt.title("혼동 행렬")
    plt.savefig(os.path.join(output_path, "confusion_matrix.png"))
    plt.show()

# ================================
# 실행
# ================================
if __name__ == "__main__":
    evaluate()