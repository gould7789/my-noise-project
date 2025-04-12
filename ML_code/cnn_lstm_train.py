import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ====================================
# 🔧 출력 폴더 설정 및 생성
# ====================================
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================================
# 1. 데이터 불러오기
# ====================================
X = np.load(os.path.join(OUTPUT_DIR, "X_lstm.npy"))
y = np.load(os.path.join(OUTPUT_DIR, "y_lstm.npy"))

# ====================================
# 2. 입력 형태 변환 (CNN 입력: 4차원 필요)
# (samples, 130, 14) → (samples, 130, 14, 1)
# ====================================
if len(X.shape) == 3:
    X = X[..., np.newaxis]

# ====================================
# 3. CNN + LSTM 모델 구성
# ====================================
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(130, 14, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Reshape((32, -1)))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# ====================================
# 4. 컴파일 설정
# ====================================
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ====================================
# 5. 모델 학습 (EarlyStopping 적용)
# ====================================
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X, y,
                    epochs=30,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=1)

# ====================================
# 6. 모델 저장
# ====================================
model_path = os.path.join(OUTPUT_DIR, "cnn_lstm_model.h5")
model.save(model_path)

# ====================================
# 7. 학습 시각화 그래프 저장
# ====================================
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('CNN + LSTM 학습 추이')
plt.xlabel('Epoch')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "cnn_lstm_train_history.png")
plt.savefig(plot_path)
plt.show()