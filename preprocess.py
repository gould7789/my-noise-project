import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

DATA_DIR = "data"
OUTPUT_FEATURES = "features"
OUTPUT_IMAGES = "spectrograms"
SEGMENT_DURATION = 5  # seconds

os.makedirs(OUTPUT_FEATURES, exist_ok=True)
os.makedirs(OUTPUT_IMAGES, exist_ok=True)

def extract_features(y, sr, filename, segment_idx, label):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    features = np.vstack([mfcc, zcr])

    # 저장
    np.save(f"{OUTPUT_FEATURES}/{filename}_seg{segment_idx}.npy", features)

    # 시각화
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(features, x_axis='time')
    plt.title(f"MFCC+ZCR of {filename} - seg{segment_idx}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_IMAGES}/{filename}_seg{segment_idx}.png", dpi=150)
    plt.close()

def preprocess_audio(file_path, label):
    filename = os.path.splitext(os.path.basename(file_path))[0]
    y, sr = librosa.load(file_path, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)

    for i in range(0, int(total_duration), SEGMENT_DURATION):
        start = i * sr
        end = min((i + SEGMENT_DURATION) * sr, len(y))
        segment = y[start:end]

        if len(segment) < SEGMENT_DURATION * sr:
            padding = SEGMENT_DURATION * sr - len(segment)
            segment = np.pad(segment, (0, padding), mode='constant')

        extract_features(segment, sr, filename, i // SEGMENT_DURATION, label)

def main():
    for label_folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, label_folder)
        if not os.path.isdir(folder_path):
            continue

        for fname in os.listdir(folder_path):
            if not fname.lower().endswith((".wav", ".mp3")):
                continue

            fpath = os.path.join(folder_path, fname)
            preprocess_audio(fpath, label_folder)

if __name__ == "__main__":
    main()