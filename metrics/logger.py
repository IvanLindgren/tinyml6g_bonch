import serial
import csv

# НАСТРОЙКА
SERIAL_PORT = 'COM11'  # Твой порт!
BAUD_RATE = 115200
OUTPUT_FILE = 'training_data.csv'

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    with open(OUTPUT_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'x', 'y', 'z'])  # Заголовки

        print("Жду старта... Надень датчик на запястье!")
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line == "START_DATA":
                print("ЗАПИСЬ ПОШЛА! Делай разные движения рукой.")
                break

        count = 0
        while count < 6000:  # Соберем 6000 строк = 1 минута записи
            line = ser.readline().decode('utf-8').strip()
            if line:
                data = line.split(',')
                if len(data) == 4:
                    writer.writerow(data)
                    count += 1
                    if count % 100 == 0:
                        print(f"Собрано {count}/6000 строк...")

    print(f"ГОТОВО! Файл {OUTPUT_FILE} создан.")
    ser.close()
except Exception as e:
    print(f"Ошибка: {e}")