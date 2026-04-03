"""
metrics_collector.py — Сбор данных с ESP32 через Serial

Записывает поток данных (actual X,Y,Z + predicted X,Y,Z) в CSV-файл
для дальнейшего анализа метрик в analyze_metrics.py.

Использование:
    python metrics_collector.py            # Запись 60 секунд
    python metrics_collector.py --sec 120  # Запись 120 секунд
"""

import serial
import csv
import time
import argparse
import sys

PORT      = "COM11"
BAUD_RATE = 115200

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sec", type=int, default=60,
                        help="Длительность записи (секунды)")
    parser.add_argument("--out", type=str, default="session_data.csv",
                        help="Имя выходного CSV-файла")
    args = parser.parse_args()

    print(f"🔌 Подключение к {PORT}...")
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=0.5)
    except Exception as e:
        print(f"❌ Ошибка порта: {e}\n   Закрой Serial Monitor в Arduino IDE!")
        sys.exit(1)

    # Пропускаем первые 2 секунды логов инициализации
    time.sleep(2)
    ser.reset_input_buffer()

    out_file = args.out
    duration = args.sec
    rows = 0

    print(f"📝 Запись в {out_file} ({duration} сек)...")
    t0 = time.time()

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ms", "ax", "ay", "az", "px", "py", "pz"])

        while time.time() - t0 < duration:
            try:
                line = ser.readline().decode("utf-8").strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) != 6:
                    continue  # Пропускаем логи инициализации
                # Проверяем, что все 6 значений — числа
                vals = [float(p) for p in parts]
                if any(v != v for v in vals):  # NaN check
                    continue
                ts = int((time.time() - t0) * 1000)
                writer.writerow([ts] + vals)
                rows += 1
                if rows % 100 == 0:
                    elapsed = int(time.time() - t0)
                    print(f"  ... {rows} строк ({elapsed}/{duration} сек)")
            except (ValueError, UnicodeDecodeError):
                continue

    ser.close()
    print(f"✅ Готово! Записано {rows} строк в {out_file}")

if __name__ == "__main__":
    main()
