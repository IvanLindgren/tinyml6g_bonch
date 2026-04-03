/*
 * network.h — Модуль Wi-Fi + MQTT для Federated Learning
 *
 * Ответственность:
 *   - Подключение к Wi-Fi (неблокирующее)
 *   - Подключение и переподключение к MQTT-брокеру
 *   - Приём глобальных весов (callback)
 *   - Отправка локальных весов / телеметрии
 */
#ifndef MQTT_CLIENT_H
#define MQTT_CLIENT_H

#include <WiFi.h>
#include <PubSubClient.h>
#include "config.h"

// --- Глобальные сетевые объекты ---
WiFiClient   espClient;
PubSubClient mqttClient(espClient);

static unsigned long lastReconnect = 0;

// Прототипы (чтобы head_W/head_B были видны из callback)
extern float head_W[EMBEDDING_SIZE][NUM_PREDICTIONS];
extern float head_B[NUM_PREDICTIONS];

// ─────────────── MQTT Callback ───────────────
void mqtt_callback(char* topic, byte* payload, unsigned int length) {
    if (strcmp(topic, TOPIC_GLOBAL) != 0) return;

    if (length == PAYLOAD_BYTES) {
        memcpy(head_W, payload, HEAD_WEIGHTS_BYTES);
        memcpy(head_B, payload + HEAD_WEIGHTS_BYTES, HEAD_BIAS_BYTES);
        Serial.println(">>> FEDERATED UPDATE: global weights applied");
    } else {
        Serial.printf(">>> WARN: bad global payload %u bytes (expected %u)\n",
                       length, PAYLOAD_BYTES);
    }
}

// ─────────────── Wi-Fi + MQTT Init ───────────────
void network_setup() {
    Serial.printf("Wi-Fi: connecting to %s ", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    for (int i = 0; i < 15 && WiFi.status() != WL_CONNECTED; i++) {
        delay(500);
        Serial.print(".");
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\nWi-Fi OK  IP: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\nWi-Fi FAIL — running offline");
    }

    mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
    mqttClient.setCallback(mqtt_callback);
    mqttClient.setBufferSize(MQTT_BUF_SIZE);
}

// ─────────────── Неблокирующий MQTT Loop ───────────────
void network_loop() {
    if (WiFi.status() != WL_CONNECTED) return;

    if (!mqttClient.connected()) {
        unsigned long now = millis();
        if (now - lastReconnect > 5000) {
            lastReconnect = now;
            String cid = "6G_" + String(random(0xffff), HEX);
            if (mqttClient.connect(cid.c_str())) {
                Serial.println("MQTT connected");
                mqttClient.subscribe(TOPIC_GLOBAL);
            }
        }
    } else {
        mqttClient.loop();
    }
}

// ─────────────── Отправка весов на сервер ───────────────
void network_publish_weights() {
    if (!mqttClient.connected()) return;
    
    // Используем потоковую передачу (Streaming API), чтобы обойти любые лимиты MQTT_MAX_PACKET_SIZE!
    // Отправляем частями прямо в сокет.
    if (mqttClient.beginPublish(TOPIC_LOCAL, PAYLOAD_BYTES, false)) {
        mqttClient.write((const uint8_t*)head_W, HEAD_WEIGHTS_BYTES);
        mqttClient.write((const uint8_t*)head_B, HEAD_BIAS_BYTES);
        if (mqttClient.endPublish()) {
            Serial.println("GLOBAL WEIGHTS: Отправка на Сервер (792 байта) прошла УСПЕШНО!");
        } else {
            Serial.println("GLOBAL WEIGHTS: Ошибка endPublish()!");
        }
    } else {
        Serial.println("GLOBAL WEIGHTS: Ошибка beginPublish()!");
    }
}

// MQTT телеметрия удалена (перенесено на быстрый UDP)

#endif // MQTT_CLIENT_H
