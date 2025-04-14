import numpy as np
np.complex = complex  # numpy 최신 버전과 librosa 호환성 문제 해결

import os
import numpy as np
import subprocess
import librosa
import soundfile as sf

# 🔧 설정값
segment_duration = 5  # 초
sample_rate = 22050
segment_samples = segment_duration * sample_rate

# 📁 입력/출력 폴더 설정
input_root = "../data"  # 원본 mp3, m4a, mp4 폴더
output_root = "data"      # 최종 저장 위치

# 🔁 변환 함수: mp3, m4a → wav
def convert_to_wav(input_path, output_path):
    subprocess.run(["ffmpeg", "-y", "-i", input_path, output_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

# ✂️ 5초 단위로 잘라 저장
def split_and_save(y, sr, save_dir, base_name, start_index):
    total_segments = len(y) // segment_samples
    for i in range(total_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = y[start:end]

        save_name = f"{base_name}_{start_index}.wav"
        save_path = os.path.join(save_dir, save_name)
        sf.write(save_path, segment, sr)
        print(f"📁 저장됨: {save_path}")
        start_index += 1
    return start_index

# 🚀 전체 처리
for category in ["loud", "quiet"]:  # 필요 시 "ambiguous" 추가
    input_folder = os.path.join(input_root, category)
    output_folder = os.path.join(output_root, category)

    os.makedirs(output_folder, exist_ok=True)
    print(f"\n🎧 카테고리 처리 중: {category}")

    file_index = 1

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith((".mp3", ".m4a", ".mp4")):
            continue

        input_path = os.path.join(input_folder, fname)
        print(f"🔁 변환 중: {input_path}")

        try:
            # 변환 후 메모리에서 자르기 위해 temp로 저장 안 함
            temp_path = f"{output_folder}/_temp.wav"
            convert_to_wav(input_path, temp_path)

            # 자르기 및 저장
            y, sr = librosa.load(temp_path, sr=sample_rate)
            file_index = split_and_save(y, sr, output_folder, category, file_index)

            os.remove(temp_path)  # temp 제거
        except Exception as e:
            print(f"⚠️ 실패: {fname} → {e}")

print("\n✅ 모든 변환 및 분할 완료!")