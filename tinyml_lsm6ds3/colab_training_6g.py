import os
import gc
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.utils import Sequence
import binascii

print(f"TensorFlow Version: {tf.__version__}")

# ==========================================
# 1. ПОДКЛЮЧЕНИЕ К GOOGLE ДИСКУ
# ==========================================
from google.colab import drive
drive.mount('/content/drive')

FILE_PATH = '/content/drive/MyDrive/WISDM_MULTI_100Hz.csv'
OUTPUT_MODEL_CUT = '/content/drive/MyDrive/model_cut.h'
OUTPUT_HEAD_WEIGHTS = '/content/drive/MyDrive/head_weights.h'

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Файл {FILE_PATH} не найден! Загрузите датасет на Диск.")
print("Диск подключен, файл найден!")

# ==========================================
# 2. ЗАГРУЗКА ДАННЫХ
# ==========================================
columns_to_use = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
df = pd.read_csv(FILE_PATH, usecols=columns_to_use, dtype=np.float32)

acc_data = df[['acc_x', 'acc_y', 'acc_z']].values
gyro_data = df[['gyro_x', 'gyro_y', 'gyro_z']].values

# Освобождаем память
del df
gc.collect()

# ==========================================
# 3. ГЕНЕРАТОР ДАННЫХ ДЛЯ 6 ОСЕЙ
# ==========================================
WINDOW_SIZE = 50
FUTURE_STEP = 5
BATCH_SIZE = 4096

class IMUDataGenerator(Sequence):
    def __init__(self, acc, gyro, window_size, future_step, batch_size, split='train', **kwargs):
        super().__init__(**kwargs)
        self.acc = acc
        self.gyro = gyro
        self.window_size = window_size
        self.future_step = future_step
        self.batch_size = batch_size

        total_valid_samples = len(acc) - window_size - future_step
        split_idx = int(total_valid_samples * 0.8)

        if split == 'train':
            self.start_idx = 0
            self.end_idx = split_idx
        else:
            self.start_idx = split_idx
            self.end_idx = total_valid_samples

        self.length = self.end_idx - self.start_idx

    def __len__(self):
        return int(np.ceil(self.length / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_start = self.start_idx + (idx * self.batch_size)
        batch_end = min(batch_start + self.batch_size, self.end_idx)

        batch_x, batch_y = [], []

        for i in range(batch_start, batch_end):
            window_acc = self.acc[i : i + self.window_size]
            window_gyro = self.gyro[i : i + self.window_size]
            # Вход (50 моментов х 6 осей)
            batch_x.append(np.concatenate((window_acc, window_gyro), axis=1))

            target_acc = self.acc[i + self.window_size + self.future_step]
            target_gyro = self.gyro[i + self.window_size + self.future_step]
            # Предсказываем сразу 6 осей
            batch_y.append(np.concatenate((target_acc, target_gyro)))

        return np.array(batch_x), np.array(batch_y)

train_gen = IMUDataGenerator(acc_data, gyro_data, WINDOW_SIZE, FUTURE_STEP, BATCH_SIZE, split='train')
val_gen = IMUDataGenerator(acc_data, gyro_data, WINDOW_SIZE, FUTURE_STEP, BATCH_SIZE, split='val')
print("Генераторы готовы!")

# ==========================================
# 4. АРХИТЕКТУРА НЕЙРОСЕТИ
# ==========================================
imu_input = Input(shape=(WINDOW_SIZE, 6), name='imu_input')

conv1 = Conv1D(32, kernel_size=3, activation='relu')(imu_input)
pool1 = MaxPooling1D(pool_size=2)(conv1)
flat  = Flatten()(pool1)

# Выход Backbone (Эта часть уйдет в TFLite на микроконтроллер)
embedding = Dense(32, activation='relu', name='feature_embedding')(flat)
drop = Dropout(0.2)(embedding)

# Единая обучаемая голова на устройство (Её скопируем в C++)
head_combined = Dense(6, activation='linear', name='head')(drop)

full_model = Model(inputs=imu_input, outputs=head_combined)
full_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ==========================================
# 5. ОБУЧЕНИЕ
# ==========================================
print("Начинаю предварительное обучение на GPU...")
history = full_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    verbose=1
)

# ==========================================
# 6. ЭКСПОРТ TFLITE BACKBONE (Только "тушка")
# ==========================================
print("\n🗜️ Начинаю INT8 квантование Backbone (тушки)...")
backbone_model = Model(inputs=imu_input, outputs=drop)

converter = tf.lite.TFLiteConverter.from_keras_model(backbone_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
    for i in range(500):
        sample_acc = acc_data[i : i + WINDOW_SIZE]
        sample_gyro = gyro_data[i : i + WINDOW_SIZE]
        sample_x = np.concatenate((sample_acc, sample_gyro), axis=1)
        yield [np.array([sample_x], dtype=np.float32)]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

def convert_to_c_array(contents, name):
    hex_str = binascii.hexlify(contents).decode('ascii')
    hex_list = ['0x' + hex_str[i:i+2] for i in range(0, len(hex_str), 2)]
    c_array = ",\n  ".join([", ".join(hex_list[i:i+12]) for i in range(0, len(hex_list), 12)])
    return f"alignas(16) const unsigned char {name}[] = {{\n  {c_array}\n}};\nconst int {name}_len = {len(hex_list)};"

with open(OUTPUT_MODEL_CUT, 'w') as f:
    f.write(convert_to_c_array(tflite_model, "model_data"))

print(f"Готово! Файл {OUTPUT_MODEL_CUT} сохранен на Диск.")

# ==========================================
# 7. ИЗВЛЕЧЕНИЕ ПРЕДОБУЧЕННЫХ ВЕСОВ В C++ ФАЙЛ
# ==========================================
print("Генерирую head_weights.h...")
weights, biases = full_model.get_layer('head').get_weights()

with open(OUTPUT_HEAD_WEIGHTS, 'w') as f:
    f.write("/*\n * head_weights.h — Предобученные стартовые веса\n")
    f.write(" * Сгенерировано автоматически из Google Colab\n */\n")
    f.write("#ifndef HEAD_WEIGHTS_H\n#define HEAD_WEIGHTS_H\n\n")
    f.write("#include \"config.h\"\n\n")
    f.write("float head_W[EMBEDDING_SIZE][NUM_PREDICTIONS] = {\n")
    for i in range(32):
        row_str = ", ".join([f"{w:.8f}f" for w in weights[i]])
        f.write(f"    {{{row_str}}}" + (",\n" if i < 31 else "\n"))
    f.write("};\n\n")
    
    f.write("float head_B[NUM_PREDICTIONS] = {\n")
    bias_str = ", ".join([f"{b:.8f}f" for b in biases])
    f.write(f"    {{{bias_str}}}\n")
    f.write("};\n\n")
    f.write("#endif // HEAD_WEIGHTS_H\n")

print(f"Готово! Файл {OUTPUT_HEAD_WEIGHTS} сохранен на Диск.")
