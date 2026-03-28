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
#include "ADXL345_ESP32.h"

#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "config.h"
#include "model_cut.h"
#include "head_weights.h"
#include "learner.h"
#include "network.h"

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
ADXL345_ESP32 adxl;

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
    Serial.println("\n--- 6G TinyML Suit ---");

    // 1. Сеть
    network_setup();

    // 2. Датчик
    Serial.println("Init I2C + ADXL345...");
    Wire.begin();
    adxl.begin();
    adxl.setRange(ADXL_RANGE);

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
    float x, y, z;
    adxl.readAcceleration(x, y, z);

    ring_buf[ring_head++] = x;
    ring_buf[ring_head++] = y;
    ring_buf[ring_head++] = z;
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
    float gt[NUM_PREDICTIONS] = {x, y, z};
    float pred[NUM_PREDICTIONS];
    /* float loss = */ learner_step(tfl_out->data.f, gt, pred);

    //5. Вывод в Serial (для visualizer.py) ──
    Serial.printf("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                   x, y, z, pred[0], pred[1], pred[2]);

    //6. Телеметрия по MQTT ──
    network_publish_data(x, y, z, pred);

    //7. Синхронизация весов (раз в SYNC_INTERVAL_MS) ──
    if (millis() - sync_ms > SYNC_INTERVAL_MS) {
        sync_ms = millis();
        network_publish_weights();
    }
}