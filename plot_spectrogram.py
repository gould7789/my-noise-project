import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 파일 로드
y, sr = librosa.load("data/ambiguous/pen_knock.wav", sr=None)
                        # 시각화 하고 싶은 파일의 경로 및 이름

# MFCC 계산 (기본 13차원)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# 시각화
plt.figure(figsize=(12, 5))
librosa.display.specshow(mfccs, x_axis='time', cmap='seismic')  # or 'seismic'
plt.colorbar()
plt.title("MFCC of pen_knock.wav") # 저장되는 이미지 파일의 이름
plt.ylabel("Feature Index")
plt.xlabel("Frame / Time")
plt.tight_layout()

# 저장
plt.savefig("data/spectrogram/pen_knock_mfcc.png", dpi=400) # 저장되는 곳 위치
print("✅ 이미지 저장 완료")