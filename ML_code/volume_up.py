import numpy as np
import soundfile as sf

# 경로 설정 (파일 이름 바꿔도 돼)
input_path = "유튜브 대화 소리.wav"
output_path = "유튜브_대화소리_증폭버전_2.wav"

# 오디오 파일 읽기
data, sr = sf.read(input_path)

# 스테레오일 경우 모노로 변환
if len(data.shape) > 1:
    data = np.mean(data, axis=1)

# +10dB 증폭 (10^(10/20) ≈ 3.16배)
amplified = data * (10 ** (15 / 20))

# 클리핑 방지
amplified = np.clip(amplified, -1.0, 1.0)

# 증폭된 파일 저장
sf.write(output_path, amplified, sr)

print("✅ 증폭 완료! 저장된 파일명:", output_path)