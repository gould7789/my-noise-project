import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import train_test_split

# ====================================
# 🔧 출력 폴더 설정
# ====================================
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================================
# 1. 데이터 불러오기
# ====================================
X = np.load(os.path.join(OUTPUT_DIR, "X_lstm.npy"))
y = np.load(os.path.join(OUTPUT_DIR, "y_lstm.npy"))

# CNN 입력 형태로 변환 (samples, 130, 14) → (samples, 130, 14, 1)
if len(X.shape) == 3:
    X = X[..., np.newaxis]

# ====================================
# 2. Stratified Train/Validation 분할
# ====================================
X_train, X_val, y_train_raw, y_val_raw = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ====================================
# 3. One-hot 인코딩
# ====================================
y_train = to_categorical(y_train_raw, num_classes=3)
y_val = to_categorical(y_val_raw, num_classes=3)

# ====================================
# 4. 모델 구성
# ====================================
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(130, 14, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Reshape((32, -1)))
model.add(LSTM(64))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

# ====================================
# 5. 컴파일
# ====================================
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ====================================
# 6. 콜백 설정
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    os.path.join(OUTPUT_DIR, "best_model.h5"),
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# ====================================
# 7. 모델 학습
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=16,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# ====================================
# 8. 최종 모델 저장
model.save(os.path.join(OUTPUT_DIR, "cnn_lstm_model.h5"))

# ====================================
# 9. 학습 시각화 (Accuracy / Loss 분리)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'], label='Train Acc', marker='o')
ax1.plot(history.history['val_accuracy'], label='Val Acc', linestyle='--', marker='x')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'], label='Train Loss', marker='o')
ax2.plot(history.history['val_loss'], label='Val Loss', linestyle='--', marker='x')
ax2.set_title('Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.suptitle("CNN + LSTM Training History", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "cnn_lstm_train_history.png"))
plt.show()