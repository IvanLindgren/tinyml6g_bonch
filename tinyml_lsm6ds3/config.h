/*
 * config.h — Все константы и гиперпараметры проекта 6G TinyML
 * 
 * Здесь собраны ВСЕ настройки, чтобы менять их в одном месте:
 * сеть, размеры модели, параметры обучения.
 */
#ifndef CONFIG_H
#define CONFIG_H

// ====================== СЕТЬ ======================
#define WIFI_SSID       "TP-Link_F04D"
#define WIFI_PASSWORD   "96260272"
#define MQTT_BROKER     "broker.emqx.io"
#define MQTT_PORT       1883
#define MQTT_BUF_SIZE   1024   // PubSubClient default=256, наш payload=792

// MQTT-топики
#define CLIENT_ID       "suit1"
#define TOPIC_LOCAL     "6g_lab/suit1/local_weights"
#define TOPIC_GLOBAL    "6g_lab/global_weights"
#define TOPIC_DATA      "6g_lab/suit1/data"

// ====================== МОДЕЛЬ ======================
#define NUM_INPUTS          300   // 50 тиков × 6 осей
#define EMBEDDING_SIZE      32    // Выход backbone (кол-во фичей)
#define NUM_PREDICTIONS     6     // X, Y, Z, GX, GY, GZ
#define DELAY_TICKS         5     // Буфер задержки (50 мс при 10 мс/тик)
#define TENSOR_ARENA_KB     16    // Размер арены TFLite (КБ)

// ====================== ОБУЧЕНИЕ ======================
#define LEARNING_RATE       0.05f
#define GRADIENT_CLIP       2.0f    // Порог клиппинга для стабильности
#define SYNC_INTERVAL_MS    10000   // Интервал отправки весов (мс)

// ====================== ДАТЧИК ======================
#define SAMPLE_INTERVAL_MS  10      // 10 мс = 100 Гц
#define ADXL_RANGE          ADXL345_RANGE_16G

// ====================== ПРОИЗВОДНЫЕ КОНСТАНТЫ ======================
#define HEAD_WEIGHTS_BYTES  (EMBEDDING_SIZE * NUM_PREDICTIONS * sizeof(float))
#define HEAD_BIAS_BYTES     (NUM_PREDICTIONS * sizeof(float))
#define PAYLOAD_BYTES       (HEAD_WEIGHTS_BYTES + HEAD_BIAS_BYTES)  // 792

#endif // CONFIG_H
