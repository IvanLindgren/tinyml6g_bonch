import paho.mqtt.client as mqtt
import struct
import numpy as np
import time

# Настройки брокера
BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC_SUBSCRIBE = "6g_lab/+/local_weights"  # Подписка на все костюмы (suit1, suit2 и т.д.)
TOPIC_PUBLISH = "6g_lab/global_weights"

# Формат пакета: W = 32x3 = 96 float. B = 3 float. Итого 99 float.
# '<' означает little-endian (стандартно для ESP32). '99f' - 99 чисел типа float (по 4 байта = 396 байт)
PAYLOAD_FORMAT = '<99f'

clients_weights = {}

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Подключено к MQTT брокеру!")
        client.subscribe(TOPIC_SUBSCRIBE)
        print(f"Слушаю топик: {TOPIC_SUBSCRIBE}")
    else:
        print(f"Ошибка подключения, код {rc}")

def on_message(client, userdata, msg):
    try:
        # Извлекаем ID клиента из топика (например, если топик 6g_lab/suit1/local_weights -> client_id = suit1)
        client_id = msg.topic.split('/')[1]
        
        # Если размер payload ровно 396 байт
        if len(msg.payload) == struct.calcsize(PAYLOAD_FORMAT):
            data = struct.unpack(PAYLOAD_FORMAT, msg.payload)
            
            # Извлекаем W (первые 96 элементов) и восстанавливаем размерность 32x3
            W = np.array(data[:96]).reshape((32, 3))
            # Извлекаем B (последние 3 элемента)
            B = np.array(data[96:])
            
            # Сохраняем в словарь последнюю матрицу от клиента
            clients_weights[client_id] = {'W': W, 'B': B, 'timestamp': time.time()}
            print(f"[{time.strftime('%H:%M:%S')}] Получены локальные веса от '{client_id}'")
        else:
            print(f"Ошибка: неверный размер пакета от {client_id}: {len(msg.payload)} байт")
            
    except Exception as e:
        print(f"Сбой при разборе пакета: {e}")

# Инициализация клиента
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print(f"Подключение к {BROKER}:{PORT}...")
client.connect(BROKER, PORT, 60)
client.loop_start()

print("Federated Server запущен. Ожидание обновлений...\n")

try:
    while True:
        # Пауза между раундами агрегации (каждые 15 секунд)
        time.sleep(15)
        
        current_time = time.time()
        valid_updates = []
        
        # Берем только свежие обновления (не старше 30 секунд)
        for cid, updates in clients_weights.items():
            if current_time - updates['timestamp'] < 30:
                valid_updates.append(updates)
                
        if len(valid_updates) > 0:
            print(f"\n--- Раунд FedAvg ({len(valid_updates)} обновлений) ---")
            
            # Инициализация нулевых массивов
            avg_W = np.zeros((32, 3))
            avg_B = np.zeros(3)
            
            # Простое математическое усреднение (Федеративное)
            for u in valid_updates:
                avg_W += u['W']
                avg_B += u['B']
                
            avg_W /= len(valid_updates)
            avg_B /= len(valid_updates)
            
            # Упаковка усредненных (Глобальных) весов обратно в бинарный формат
            flat_W = avg_W.flatten().tolist()
            flat_B = avg_B.tolist()
            
            payload = struct.pack(PAYLOAD_FORMAT, *(flat_W + flat_B))
            
            # Публикация
            client.publish(TOPIC_PUBLISH, payload)
            print(f"[{time.strftime('%H:%M:%S')}] Глобальные веса опубликованы в '{TOPIC_PUBLISH}'\n")
            
            # Очистка локального кэша, чтобы ждать следующего раунда реальных данных
            clients_weights.clear()

except KeyboardInterrupt:
    print("\nСервер остановлен.")
    client.loop_stop()
    client.disconnect()
