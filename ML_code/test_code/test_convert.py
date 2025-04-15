from pydub import AudioSegment
import os

# 도커 컨테이너 기준 경로
input_root = "/app/test"
output_root = "/app/ML_code/model_test"

categories = {
    "loud": "loud_test",
    "quiet": "quiet_test",
    "ambiguous": "ambiguous_test"
}

# 카테고리마다 처리
for category, out_folder in categories.items():
    input_folder = os.path.join(input_root, category)
    output_folder = os.path.join(output_root, out_folder)
    os.makedirs(output_folder, exist_ok=True)

    file_index = 1
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".mp3", ".m4a", ".mp4", ".wav")):
            continue  # 오디오 파일이 아니면 무시

        file_path = os.path.join(input_folder, filename)
        try:
            audio = AudioSegment.from_file(file_path)
            duration_ms = len(audio)

            for i in range(0, duration_ms, 5000):  # 5초 단위 자르기
                chunk = audio[i:i+5000]
                out_filename = f"{category}_test{file_index}.wav"
                out_path = os.path.join(output_folder, out_filename)
                chunk.export(out_path, format="wav")
                print(f"✅ Saved: {out_filename}")
                file_index += 1

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")