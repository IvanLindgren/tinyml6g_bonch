import serial
import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt
import numpy as np
import time
import threading
import os
import sys

# --- НАСТРОЙКИ ---
SERIAL_PORT = 'COM11'
BAUD_RATE = 115200
TOPIC_WEIGHTS = "6g_lab/suit1/local_weights"
MAX_UPDATES = 50
FUTURE_STEPS = 5

# Создаем папку для отчетов
RESULT_DIR = 'research_report_6g'
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
    os.makedirs(f"{RESULT_DIR}/phases")


class SilentResearcher:
    def __init__(self):
        # Буферы для текущей фазы
        self.current_real = []
        self.current_pred = []

        # Глобальная история для финального отчета
        self.mae_history = []
        self.weights_count = 0
        self.total_bytes = 0
        self.start_time = time.time()

        self.running = True
        self.new_update_event = threading.Event()

    def mqtt_worker(self):
        def on_message(client, userdata, msg):
            if msg.topic == TOPIC_WEIGHTS:
                self.weights_count += 1
                self.total_bytes += len(msg.payload)
                self.new_update_event.set()  # Сигнализируем основной петле

        client = mqtt.Client()
        client.on_message = on_message
        client.connect("test.mosquitto.org", 1883)
        client.subscribe(TOPIC_WEIGHTS)
        client.loop_forever()

    def save_phase_plot(self, update_num, real, pred):
        """Быстрое сохранение графика фазы в файл без вывода на экран"""
        if not real: return

        plt.figure(figsize=(10, 4))
        plt.plot(real, 'b-', label='Реальность', alpha=0.5)
        # Сдвигаем прогноз для визуального совпадения
        shifted = pred[FUTURE_STEPS:] + [None] * FUTURE_STEPS
        plt.plot(shifted, 'r--', label='Прогноз (+50мс)')

        mae = np.mean(np.abs(np.array(real) - np.array(pred)))
        plt.title(f"Фаза обучения #{update_num} | MAE: {mae:.4f}")
        plt.grid(alpha=0.3)
        plt.legend(loc='upper right')
        plt.savefig(f"{RESULT_DIR}/phases/update_{update_num:03d}.png")
        plt.close()  # Важно закрыть, чтобы не жрать RAM
        return mae

    def run(self):
        threading.Thread(target=self.mqtt_worker, daemon=True).start()

        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            print(f"📡 Сбор данных запущен (COM: {SERIAL_PORT}).")
            print(f"📊 Ждем {MAX_UPDATES} апдейтов весов...")
            print("🛑 Нажми Ctrl+C для досрочного завершения.")
        except:
            print("❌ Ошибка порта!");
            return

        try:
            while self.running and self.weights_count < MAX_UPDATES:
                # Читаем данные из Serial максимально быстро
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    parts = line.split(',')
                    if len(parts) == 6:
                        self.current_real.append(float(parts[0]))
                        self.current_pred.append(float(parts[3]))

                # Обработка апдейта весов
                if self.new_update_event.is_set():
                    self.new_update_event.clear()

                    # Считаем и сохраняем текущую фазу
                    mae = self.save_phase_plot(self.weights_count, self.current_real, self.current_pred)
                    self.mae_history.append(mae)

                    print(f"📥 Апдейт {self.weights_count}/{MAX_UPDATES} обработан. MAE: {mae:.4f}")

                    # Очищаем буферы для следующей фазы
                    self.current_real = []
                    self.current_pred = []

        except KeyboardInterrupt:
            print("\n⏹ Сбор остановлен пользователем.")
        finally:
            ser.close()
            self.generate_final_report()

    def generate_final_report(self):
        print("\n" + "=" * 40)
        print("🧬 ГЕНЕРАЦИЯ ФИНАЛЬНОГО ОТЧЕТА...")

        duration = time.time() - self.start_time
        raw_size = duration * 1200  # 3 оси * 4 байта * 100 Гц
        ratio = raw_size / max(1, self.total_bytes)

        # 1. График кривой обучения
        plt.figure(figsize=(12, 6))
        x_axis = range(1, len(self.mae_history) + 1)
        plt.plot(x_axis, self.mae_history, 'g-o', lw=2, label='Средняя ошибка (MAE)')

        # Тренд (линейная регрессия для наглядности)
        if len(self.mae_history) > 1:
            z = np.polyfit(x_axis, self.mae_history, 1)
            p = np.poly1d(z)
            plt.plot(x_axis, p(x_axis), "r--", alpha=0.8, label='Тренд персонализации')

        plt.title("Learning Curve: Адаптация нейросети на устройстве (Federated TinyML)", fontsize=14)
        plt.xlabel("Номер цикла обновления весов (каждые 10 сек)", fontsize=12)
        plt.ylabel("Погрешность MAE (ускорение g)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{RESULT_DIR}/FINAL_LEARNING_CURVE.png")

        # 2. Текстовый отчет
        with open(f"{RESULT_DIR}/summary_report.txt", "w", encoding="utf-8") as f:
            f.write("ФИНАЛЬНЫЙ ОТЧЕТ ЛАБОРАТОРИИ 6G\n")
            f.write("=" * 30 + "\n")
            f.write(f"Длительность теста:  {duration:.1f} сек\n")
            f.write(f"Экономия трафика:    в {ratio:.1f} раз\n")
            f.write(f"Начальная ошибка:    {self.mae_history[0]:.4f} g\n")
            f.write(f"Финальная ошибка:    {self.mae_history[-1]:.4f} g\n")
            f.write(
                f"Улучшение точности:  {((self.mae_history[0] - self.mae_history[-1]) / self.mae_history[0]) * 100:.1f}%\n")

        print(f"✅ Готово! Результаты в папке: {RESULT_DIR}")
        print(f"🚀 Экономия трафика: в {ratio:.1f} раз")
        print(
            f"📉 Точность улучшилась на: {((self.mae_history[0] - self.mae_history[-1]) / self.mae_history[0]) * 100:.1f}%")
        print("=" * 40)
        plt.show()


if __name__ == "__main__":
    researcher = SilentResearcher()
    researcher.run()