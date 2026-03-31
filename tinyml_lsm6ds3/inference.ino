/*
 * inference.ino — Главный файл проекта 6G Federated TinyML
 *
 * Архитектура:
 *   config.h        → Все настройки и константы
 *   head_weights.h  → Предобученные веса Dense-слоя из Colab
 *   network.h       → Wi-Fi / MQTT (Federated Learning коммуникации)
 *   learner.h       → Forward Pass + Backward Pass (SGD)
 *   model_cut.h     → Замороженный TFLite backbone (Feature Extractor)
 *   ADXL345_ESP32.* → Драйвер акселерометра
 *
 * Этот файл только связывает модули: setup() + loop().
 */
//Импорт модулей 
#include <Wire.h>
#include <FastIMU.h>

#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "config.h"
#include "model_cut.h"
#include "head_weights.h"
#include "learner.h"
#include "mqtt_client.h"

// TFLite Micro
constexpr int kArenaSize = TENSOR_ARENA_KB * 1024;
alignas(16) uint8_t tensor_arena[kArenaSize];

const tflite::Model*       tfl_model  = nullptr;
tflite::MicroInterpreter*  tfl_interp = nullptr;
TfLiteTensor*              tfl_in     = nullptr;
TfLiteTensor*              tfl_out    = nullptr;

tflite::AllOpsResolver     tfl_resolver;
tflite::MicroErrorReporter tfl_reporter;

    //Датчик
    LSM6DS3 IMU;
    calData calib = { 0 };

//Кольцевой буфер входных данных
float ring_buf[NUM_INPUTS];
int   ring_head     = 0;
bool  ring_full     = false;

//Таймеры
unsigned long tick_ms = 0;   // Таймер опроса датчика
unsigned long sync_ms = 0;   // Таймер отправки весов


void setup() {
    Serial.begin(115200);
    delay(2000);
    // 1. Сеть
    network_setup();

    // 2. Датчик (FastIMU)
    Serial.println("Init I2C + FastIMU (LSM6DS3)...");
    Wire.begin();
    
    int err = IMU.init(calib, 0x6A);
    if (err != 0) {
        Serial.println("LSM6DS3 не найден на адресе 0x6A. Пробуем 0x6B...");
        err = IMU.init(calib, 0x6B);
        if (err != 0) {
            Serial.println("Ошибка инициализации LSM6DS3! Проверьте провода SDA/SCL и питание.");
            while (1) delay(1000); // Останавливаем выполнение
        }
    }
    Serial.println("Датчик LSM6DS3 успешно подключен!");
    
    // Настраиваем нужные нам диапазоны
    IMU.setAccelRange(2);  // +- 2G
    IMU.setGyroRange(250); // +- 250 градусов/с
    
    // КАЛИБРОВКА НУЛЕЙ
    Serial.println("Оставьте плату неподвижно на столе на 2-3 секунды для калибровки...");
    IMU.calibrateAccelGyro(&calib);
    Serial.println("Калибровка завершена!");

    // 3. TFLite backbone
    Serial.println("Loading TFLite backbone...");
    tfl_model = tflite::GetModel(model_data);
    if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("FATAL: model schema mismatch");
        while (1) delay(1000);
    }

    tfl_interp = new tflite::MicroInterpreter(
        tfl_model, tfl_resolver, tensor_arena, kArenaSize, &tfl_reporter);

    if (tfl_interp->AllocateTensors() != kTfLiteOk) {
        Serial.println("FATAL: AllocateTensors() failed");
        while (1) delay(1000);
    }

    tfl_in  = tfl_interp->input(0);
    tfl_out = tfl_interp->output(0);

    // Проверка совместимости модели
    if (tfl_out->dims->data[1] != EMBEDDING_SIZE) {
        Serial.printf("FATAL: model outputs %d features, expected %d\n",
                       tfl_out->dims->data[1], EMBEDDING_SIZE);
        while (1) delay(1000);
    }
    if (tfl_in->type != kTfLiteFloat32 || tfl_out->type != kTfLiteFloat32) {
        Serial.println("FATAL: model is not float32! Rebuild without quantization.");
        while (1) delay(1000);
    }

    Serial.printf("Arena: %d / %d bytes\n",
                   tfl_interp->arena_used_bytes(), kArenaSize);
    Serial.println("SYSTEM_ONLINE");
}


void loop() {
    network_loop();

    // Каждые SAMPLE_INTERVAL_MS (10 мс = 100 Гц)
    if (millis() - tick_ms < SAMPLE_INTERVAL_MS) return;
    tick_ms = millis();

    // ── 1. Чтение датчика ──
    IMU.update(); // Обновляем данные с шины I2C
    AccelData accel;
    GyroData gyro;
    
    IMU.getAccel(&accel);
    IMU.getGyro(&gyro);
    
    float raw[6] = {accel.accelX, accel.accelY, accel.accelZ, gyro.gyroX, gyro.gyroY, gyro.gyroZ};
    
    // ── 1.5 Нормализация на лету (Standard Scaler EWMA) ──
    static float mean[6] = {0};
    static float var[6]  = {1, 1, 1, 1, 1, 1};
    const float alpha    = 0.005f; // "Окно" адаптации ~2 секунды
    float norm[6];

    for (int i = 0; i < 6; i++) {
        float diff = raw[i] - mean[i];
        mean[i] += alpha * diff;
        var[i] = (1.0f - alpha) * (var[i] + alpha * diff * diff);
        norm[i] = (raw[i] - mean[i]) / (sqrtf(var[i]) + 1e-6f); // Избегаем деления на ноль
    }
    
    // Для визуализатора и реальных таргетов нужны "сырые" данные:
    float ax = raw[0];
    float ay = raw[1];
    float az = raw[2];
    float gx = raw[3];
    float gy = raw[4];
    float gz = raw[5];

    // В кольцевой буфер (для TFLite) кладем НОРМАЛИЗОВАННЫЕ данные:
    for (int i = 0; i < 6; i++) {
        ring_buf[ring_head++] = norm[i];
    }
    
    if (ring_head >= NUM_INPUTS) {
        ring_head = 0;
        ring_full = true;
    }
    if (!ring_full) return;

    // 2. Подготовка входного тензора (разворот кольцевого буфера)
    float flat[NUM_INPUTS];
    int k = 0;
    for (int i = ring_head; i < NUM_INPUTS; i++) flat[k++] = ring_buf[i];
    for (int i = 0; i < ring_head; i++)          flat[k++] = ring_buf[i];
    memcpy(tfl_in->data.f, flat, sizeof(flat));

    //3. TFLite Invoke (получаем 32 фичи) ──
    if (tfl_interp->Invoke() != kTfLiteOk) {
        Serial.println("ERR: TFLite Invoke failed");
        return;
    }

    //4. Обучение + предсказание (learner.h) ──
    float gt[NUM_PREDICTIONS] = {ax, ay, az, gx, gy, gz};
    float pred[NUM_PREDICTIONS];
    /* float loss = */ learner_step(tfl_out->data.f, gt, pred);

    Serial.printf("%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
              CLIENT_ID, ax, ay, az, gx, gy, gz, 
              pred[0], pred[1], pred[2], pred[3], pred[4], pred[5]);

    //7. Синхронизация весов (раз в SYNC_INTERVAL_MS) ──
    if (millis() - sync_ms > SYNC_INTERVAL_MS) {
        sync_ms = millis();
        network_publish_weights();
    }
}