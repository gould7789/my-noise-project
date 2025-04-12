import os
import subprocess

input_root = "../data"  # 예: data/loud, data/quiet, data/ambiguous
output_root = "data"

# 변환 함수
def convert_to_wav(input_path, output_path):
    subprocess.run(["ffmpeg", "-y", "-i", input_path, output_path], check=True)

# 각 카테고리(loud/quiet/ambiguous)별 처리
for category in ["loud", "quiet", "ambiguous"]:
    input_folder = os.path.join(input_root, category)
    output_folder = os.path.join(output_root, category)
    os.makedirs(output_folder, exist_ok=True)

    count = 1  # 카테고리별 파일 번호

    for fname in os.listdir(input_folder):
        if fname.endswith((".mp3", ".m4a", ".mp4")):
            input_path = os.path.join(input_folder, fname)
            # 파일명을 loud1.wav, quiet2.wav 이런 식으로 지정
            new_filename = f"{category}_{count}.wav"
            output_path = os.path.join(output_folder, new_filename)

            print(f"🎵 변환 중: {input_path} → {output_path}")
            try:
                convert_to_wav(input_path, output_path)
                count += 1
            except Exception as e:
                print(f"⚠️ 오류 발생: {fname} → {e}")

print("✅ 모든 변환이 완료되었습니다!")