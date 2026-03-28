# 6G Federated TinyML — On-Device Trajectory Prediction

<p align="center">
  <b>ESP32 + ADXL345 + TensorFlow Lite Micro + MQTT Federated Averaging</b>
</p>

Прошивка для датчика акселерометра, которая в реальном времени предсказывает
координаты тела на 50 мс вперед, **обучается прямо на устройстве** и
обменивается опытом с другими датчиками через протокол Federated Learning.

---

## Оглавление

1. [Архитектура](#архитектура)
2. [Структура файлов](#структура-файлов)
3. [Необходимое железо](#необходимое-железо)
4. [Установка и запуск](#установка-и-запуск)
5. [⚠️ ОБЯЗАТЕЛЬНЫЕ ФИКСЫ БИБЛИОТЕК](#%EF%B8%8F-обязательные-фиксы-библиотек)
6. [Конвертация модели в Colab](#конвертация-модели-в-colab)
7. [Подробное описание параметров (config.h)](#подробное-описание-параметров-configh)
8. [Как работает обучение на устройстве](#как-работает-обучение-на-устройстве)
9. [Federated Learning (MQTT)](#federated-learning-mqtt)
10. [Слабые места и пути улучшения](#слабые-места-и-пути-улучшения)

---

## Архитектура

```
┌─────────────────────────────────────────────────────────┐
│                      ESP32 (On-Device)                  │
│                                                         │
│  ADXL345 ──► Ring Buffer (50×3) ──► TFLite Backbone     │
│              (сырые данные)         (model_cut.h)       │
│                                         │               │
│                                    32 features          │
│                                    (embedding)          │
│                                         │               │
│              ┌──────────────────────────┤               │
│              │                          ▼               │
│         Delay Buffer              Dense Head            │
│         (5 тиков)              head_W[32×3]+head_B[3]   │
│              │                          │               │
│              │   Ground Truth      Prediction           │
│              │   (real X,Y,Z)     (pred X,Y,Z)          │
│              │         │                │               │
│              ▼         ▼                │               │
│           SGD Backward Pass             │               │
│           (обновление весов)            │               │
│                                         │               │
│  ┌─────────── MQTT (раз в 10 сек) ──────┘               │
│  │  Отправка head_W + head_B (396 байт)                 │
│  │  Приём глобальных весов от сервера                   │
└──┼──────────────────────────────────────────────────────┘
   │
   ▼
┌───────────────────────────────┐
│   Python FedAvg Server        │
│   federated_server.py         │
│                               │
│   Усредняет веса от N клиентов│
│   и публикует global_weights  │
└───────────────────────────────┘
```

**Ключевая идея:**
Тяжёлые для МК слои (Conv1D + LSTM) заморожены и работает только как Feature
Extractor. Обучается только лёгкий Dense-слой (99 float = 396 байт), что
позволяет выполнять градиентный спуск по этим весам прямо на микроконтроллере
без обратного распространения ошибки через весь граф вычислений.

---

## Структура файлов

```
inference/
├── inference.ino      # Главный файл (setup + loop), связывает модули
├── config.h           # ВСЕ настройки и гиперпараметры в одном месте
├── head_weights.h     # Предобученные веса Dense-головы из Colab
├── learner.h          # Forward Pass + Backward Pass (SGD) + Delay Buffer
├── network.h          # Wi-Fi + MQTT (подключение, отправка/приём весов)
├── model_cut.h        # Замороженный TFLite Micro backbone (бинарный массив)
├── ADXL345_ESP32.h    # Заголовок драйвера акселерометра
├── ADXL345_ESP32.cpp  # Реализация драйвера акселерометра
├── visualizer.py      # 2D-визуализатор жестов (Serial → Matplotlib)
└── federated_server.py # Python-сервер агрегации весов (FedAvg)
```

---

## Необходимое железо

| Компонент | Модель | Примечание |
|-----------|--------|------------|
| Микроконтроллер | ESP32 DevKit v1 | Двухъядерный, 520 КБ SRAM, Wi-Fi |
| Акселерометр | ADXL345 | I²C, подключение SDA→GPIO21, SCL→GPIO22 |
| Питание | USB или LiPo 3.7V | Через Vin или USB |

---

## Установка и запуск

### Шаг 1. Установка библиотек в Arduino IDE

Через **Менеджер библиотек** (Ctrl+Shift+I):

1. **TensorFlowLite_ESP32** (он же `tflm_esp32`) — движок нейросети
2. **PubSubClient** (автор Nick O'Leary) — MQTT-клиент

### Шаг 2. Применение фиксов библиотек

> ⚠️ **БЕЗ ЭТОГО ШАГА КОД НЕ СКОМПИЛИРУЕТСЯ!**  
> Подробности в разделе [Обязательные фиксы библиотек](#%EF%B8%8F-обязательные-фиксы-библиотек).

### Шаг 3. Настройка Wi-Fi

Отредактируй `config.h`:
```cpp
#define WIFI_SSID       "ИМЯ_ТВОЕЙ_СЕТИ"
#define WIFI_PASSWORD   "ПАРОЛЬ"
```

### Шаг 4. Прошивка ESP32

1. Открой `inference.ino` в Arduino IDE.
2. Плата: `ESP32 Dev Module`.
3. Нажми **Upload (Загрузка)**.
4. Открой Serial Monitor (115200 бод).

Ожидаемый вывод при успешном запуске:
```
--- 6G TinyML Suit ---
Wi-Fi: connecting to TP-Link_F04D .....
Wi-Fi OK  IP: 192.168.0.5
Init I2C + ADXL345...
Loading TFLite backbone...
Arena: 7576 / 16384 bytes
SYSTEM_ONLINE
0.0780,0.0195,-0.0858,-0.0168,0.0052,0.1056
```

### Шаг 5. Запуск сервера агрегации (на ПК)

```bash
pip install paho-mqtt numpy
python federated_server.py
```

---

## ⚠️ ОБЯЗАТЕЛЬНЫЕ ФИКСЫ БИБЛИОТЕК

Библиотека `TensorFlowLite_ESP32` содержит **два бага**, которые не дадут
скомпилировать проект. Их нужно пропатчить вручную.

### Фикс 1: «Deleted destructor» / `operator delete is private`

**Ошибка:**
```
error: deleted function 'virtual ...::~AllOpsResolver()'
error: 'static void ...::operator delete(void*)' is private
```

**Причина:** Макрос `TF_LITE_REMOVE_VIRTUAL_DELETE` в оригинале
заканчивается словом `private:`, что прячет все последующие публичные
методы класса (`AddBuiltin`, `AddCustom` и т.д.) от компилятора.

**Файл:**
```
Arduino/libraries/TensorFlowLite_ESP32/src/tensorflow/lite/micro/compatibility.h
```

**Исправленный макрос (заменить целиком):**
```cpp
#ifdef TF_LITE_STATIC_MEMORY
#define TF_LITE_REMOVE_VIRTUAL_DELETE \
  public: void operator delete(void* p) {}
#else
#define TF_LITE_REMOVE_VIRTUAL_DELETE
#endif
```

> **ВАЖНО:** Строка заканчивается на `{}`, без `private:` после неё!
> Если оставить `private:`, класс `AllOpsResolver` потеряет доступ к
> методам `AddBuiltin()`, и вы получите ещё более загадочные ошибки.

---

### Фикс 2: Flatbuffers `assignment of read-only member 'count_'`

**Ошибка:**
```
error: assignment of read-only member 'flatbuffers::span<T, Extent>::count_'
```

**Причина:** Члены `data_` и `count_` объявлены как `const`, и компилятор
ESP32 (xtensa-gcc) запрещает прямое присваивание.

**Файл:**
```
Arduino/libraries/TensorFlowLite_ESP32/src/third_party/flatbuffers/stl_emulation.h
```

**Найти (около строки 383) оператор `operator=` и заменить его тело на:**
```cpp
FLATBUFFERS_CONSTEXPR_CPP14 span &operator=(const span &other)
    FLATBUFFERS_NOEXCEPT {
  // data_ and count_ are const; use placement new to reassign
  this->~span();
  new (this) span(other);
  return *this;
}
```

---

### Фикс 3 (Runtime): Модель выдаёт `NaN` (Not a Number)

**Симптом:** На Serial Monitor видно:
```
0.0780,0.0195,-0.0858,nan,nan,nan
```

**Причина:** Модель была сконвертирована в TFLite с оптимизацией
`tf.lite.Optimize.DEFAULT`, которая пережимает веса в `float16`.
ESP32 не поддерживает аппаратный `float16`, и `output_tensor->data.f`
читает мусор из памяти.

**Диагностика (встроена в код):**
```
Input Tensor Type: 1 (0=float32, 1=float16, 9=int8)
>>> FATAL ERROR: Модель не FLOAT32!
```

**Исправление:** В Colab при конвертации **убрать оптимизацию**:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]  # ← ЗАКОММЕНТИРОВАТЬ!
tflite_model = converter.convert()
```

---

## Конвертация модели в Colab

Наша архитектура Split-Learning требует «обрезанную» модель — без
последнего Dense-слоя. Вот правильный алгоритм экспорта:

```python
import tensorflow as tf

# 1. Загружаем полную модель (которая предсказывает X, Y, Z)
model = tf.keras.models.load_model("trajectory_model.h5")

# 2. Обрезаем: создаём новую модель БЕЗ последнего Dense-слоя
#    Выход = предпоследний слой (32 нейрона = embedding)
cut_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.layers[-2].output  # предпоследний слой
)

# 3. Конвертируем в TFLite БЕЗ ОПТИМИЗАЦИИ (иначе будет float16 → NaN!)
converter = tf.lite.TFLiteConverter.from_keras_model(cut_model)
# НЕ ДОБАВЛЯТЬ: converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("model_cut.tflite", "wb") as f:
    f.write(tflite_model)

# 4. Конвертируем .tflite → C-массив (model_cut.h)
import binascii
with open("model_cut.tflite", "rb") as f:
    data = f.read()

hex_array = ", ".join(f"0x{b:02x}" for b in data)
c_code = f"alignas(16) const unsigned char model_data[] = {{\n{hex_array}\n}};\n"
c_code += f"const int model_data_len = {len(data)};\n"

with open("model_cut.h", "w") as f:
    f.write(c_code)

# 5. Извлекаем начальные веса головы для head_weights.h
last_dense = model.layers[-1]
W, B = last_dense.get_weights()
print("Initial Head Weights:", W.tolist())
print("Initial Head Biases:", B.tolist())
```

---

## Подробное описание параметров (config.h)

### Параметры модели

| Параметр | Значение | Описание |
|----------|----------|----------|
| `NUM_INPUTS` | `150` | Размер входного окна TFLite: 50 временных шагов × 3 оси (X, Y, Z). При частоте опроса 100 Гц это окно покрывает 500 мс истории движения. Увеличение окна даёт модели «больше контекста», но линейно увеличивает размер арены и время Invoke(). |
| `EMBEDDING_SIZE` | `32` | Количество признаков (фичей) на выходе замороженного backbone. Это размерность «скрытого пространства» (latent space), в которой модель кодирует паттерны движения. При увеличении — больше информации для головы, но линейно растёт размер обучаемой матрицы и объём передаваемых по MQTT весов. |
| `NUM_PREDICTIONS` | `3` | Количество выходов головы: предсказанные X, Y, Z ускорения. |
| `DELAY_TICKS` | `5` | Глубина буфера задержки для self-supervised learning. При `SAMPLE_INTERVAL_MS = 10` это означает, что предсказание делается на `5 × 10 = 50 мс` вперёд. Увеличение DELAY_TICKS = предсказание дальше в будущее, но ошибка (loss) растёт, так как далёкое будущее менее предсказуемо. |
| `TENSOR_ARENA_KB` | `16` | Размер памяти (в КБ), выделяемой TFLite Micro для хранения промежуточных тензоров во время Invoke(). Если модель не помещается — увеличить. Реальное потребление видно в логе: `Arena: 7576 / 16384 bytes`. |

### Параметры обучения

| Параметр | Значение | Описание |
|----------|----------|----------|
| `LEARNING_RATE` | `0.0005f` | **Самый важный гиперпараметр.** Размер «шага» SGD при обновлении весов. Слишком большой (>0.01) — веса начнут прыгать (diverge), предсказания станут хуже. Слишком маленький (<0.00001) — модель будет учиться неделю. Рекомендуемый диапазон для нашей задачи: `0.0001 – 0.001`. |
| `GRADIENT_CLIP` | `2.0f` | Порог клиппинга градиента. Если ошибка `(pred - actual)` превышает ±2.0, она обрезается до ±2.0. Это предотвращает «взрыв градиентов» (Exploding Gradients), который может испортить веса за один шаг. Уменьшить до `1.0`, если наблюдается нестабильность обучения. |
| `SYNC_INTERVAL_MS` | `10000` | Как часто (в мс) ESP32 отправляет свои локальные веса серверу и получает глобальные. 10 секунд — компромисс между скоростью конвергенции (чаще = быстрее учимся от других) и нагрузкой на сеть/батарею. Для экономии батареи можно увеличить до 30000–60000. |

### Параметры датчика

| Параметр | Значение | Описание |
|----------|----------|----------|
| `SAMPLE_INTERVAL_MS` | `10` | Период опроса акселерометра в миллисекундах. 10 мс = 100 Гц. Это стандартная частота для IMU. Снижение до 20 мс (50 Гц) удвоит время жизни батареи, но ухудшит разрешение быстрых жестов. |
| `ADXL_RANGE` | `ADXL345_RANGE_16G` | Диапазон измерения акселерометра. 16G — максимальный, ловит резкие удары. Для плавных жестов можно снизить до `ADXL345_RANGE_2G` для лучшего разрешения (меньше шума). |

### Сетевые параметры

| Параметр | Значение | Описание |
|----------|----------|----------|
| `MQTT_BROKER` | `test.mosquitto.org` | Бесплатный публичный MQTT-брокер. Для продакшена использовать локальный Mosquitto или облачный HiveMQ. |
| `MQTT_BUF_SIZE` | `512` | Размер буфера PubSubClient. По умолчанию 256, но наш payload весов = 396 байт, поэтому буфер увеличен до 512. |
| `TOPIC_LOCAL` | `6g_lab/suit1/local_weights` | Топик, в который ESP32 публикует свои локальные веса. Для второго костюма: `6g_lab/suit2/local_weights`. |
| `TOPIC_GLOBAL` | `6g_lab/global_weights` | Топик, из которого все костюмы получают усреднённые (глобальные) веса. |

---

## Как работает обучение на устройстве

### Self-Supervised Learning через Delay Buffer

Нейросети обычно нужны «правильные ответы» (labels), размеченные человеком.
Мы обходим это ограничение трюком с задержкой:

```
Время:  t=0         t=50мс
        │           │
        ▼           ▼
      Emb[0] ──► Pred[0]     ← предсказание на 50 мс вперёд
                    ...
                  Actual[5]   ← через 50 мс считываем настоящие координаты
                    │
                    ▼
              Error = Pred[0] - Actual[5]
                    │
                    ▼
              SGD обновление весов
```

1. В момент `t` модель получает embedding и делает предсказание `pred`.
2. И embedding, и prediction сохраняются в кольцевой буфер (`delay_emb`, `delay_pred`).
3. Через 5 тиков (50 мс) настоящие данные с датчика (`ground_truth`) поступают.
4. Считается ошибка: `error = old_pred - ground_truth`.
5. Веса корректируются через SGD: `W -= lr × error × embedding`.

### Математика SGD (Stochastic Gradient Descent)

Для функции потерь MSE (Mean Squared Error):

$$L = \frac{1}{N}\sum_{j=1}^{N}(pred_j - gt_j)^2$$

Градиенты:

$$\frac{\partial L}{\partial W_{ij}} = error_j \cdot emb_i$$

$$\frac{\partial L}{\partial B_j} = error_j$$

Обновление весов:

$$W_{ij} \leftarrow W_{ij} - \eta \cdot \text{clip}(error_j, \pm C) \cdot emb_i$$

$$B_j \leftarrow B_j - \eta \cdot \text{clip}(error_j, \pm C)$$

Где `η = LEARNING_RATE`, `C = GRADIENT_CLIP`.

---

## Federated Learning (MQTT)

### Протокол обмена весами

```
ESP32 (suit1)                    Python Server                    ESP32 (suit2)
     │                                │                                │
     │──── local_weights (396 B) ────►│                                │
     │                                │◄─── local_weights (396 B) ─────│
     │                                │                                │
     │                          ┌─────┴─────┐                          │
     │                          │  FedAvg:  │                          │
     │                          │ W_global =│                          │
     │                          │ mean(W_i) │                          │
     │                          └─────┬─────┘                          │
     │                                │                                │
     │◄── global_weights (396 B) ─────│──── global_weights (396 B) ──► │
     │                                │                                │
```

### Формат payload (396 байт)

```
Байты 0–383:   head_W[32][3]  (32×3×4 = 384 байт, float32, little-endian)
Байты 384–395: head_B[3]      (3×4 = 12 байт, float32, little-endian)
```

### FedAvg (Federated Averaging)

Сервер применяет простое усреднение:

$$W_{global} = \frac{1}{K}\sum_{k=1}^{K} W_k$$

Где `K` — количество костюмов, приславших веса за текущий раунд.

---

## Слабые места и пути улучшения

### 1. Проблема Non-IID данных (Главное слабое место)

**Суть:** Если разные костюмы носят люди с разной моторикой (барабанщик
vs дирижёр), FedAvg усредняет их модели в «ни рыбу ни мясо».

**Решения:**
- **FedPer (Personalized FL):** Усреднять только часть весов, оставляя
  bias (`head_B`) персональным для каждого устройства.
- **FedProx:** Добавить регуляризатор, штрафующий за сильное отклонение
  локальных весов от глобальных: `L += μ/2 × ||W_local - W_global||²`

### 2. Vanilla SGD vs Adam

**Суть:** Наш SGD не имеет моментума и адаптивного шага. Он может
застревать в локальных минимумах и медленно сходиться.

**Решение:** Реализовать Adam на C++. Это потребует хранить дополнительно
два массива (первый и второй моменты) размера `32×3 + 3 = 99` float
каждый (≈800 байт). Вполне помещается в SRAM ESP32.

### 3. Замороженный Backbone

**Суть:** Если модель обучалась на ходьбе, а пользователь танцует —
backbone выдаёт бесполезные фичи, и никакой head не поможет.

**Решения:**
- **TinyTL:** Дообучать bias-ы внутренних слоёв backbone (не касаясь весов).
- **Периодическая пересборка модели:** Раз в неделю собирать данные,
  дообучать backbone в облаке и прошивать OTA.

### 4. Энергопотребление Wi-Fi

**Суть:** Wi-Fi на ESP32 потребляет ~100 мА при активной передаче.
Постоянное подключение к MQTT быстро убивает батарею.

**Решения:**
- **BLE Mesh** вместо Wi-Fi (потребление ~10 мА).
- **Квантование payload:** Передавать веса в `int8` вместо `float32`
  (сжатие пакета с 396 до 99 байт, -75%).
- **Adaptive Sync:** Отправлять веса не по таймеру, а только когда
  loss упал ниже порога (модель реально чему-то научилась).

### 5. Сенсорная слепота (только ADXL345)

**Суть:** Акселерометр не может отличить линейное ускорение от наклона
(гравитация «перетекает» между осями при повороте запястья).

**Решение:** Заменить ADXL345 на **MPU6050** (акселерометр + гироскоп).
Это позволит использовать Sensor Fusion (фильтр Маджвика/Калмана)
для вычитания гравитации и получения чистого линейного ускорения.
