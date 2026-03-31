import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
from collections import deque
import sys
import csv
import time

# ================= НАСТРОЙКИ =================
PORT = "COM12"
BAUD_RATE = 115200

MAX_POINTS = 50  # Длина "хвоста" траектории (чем меньше, тем быстрее стирается)
DT = 0.01

DECAY = 0.95  # Затухание (возвращает кисть в центр, убирает вечный дрифт)
SCALE_GYR = 0.5  # Масштаб отрисовки на графике

# =============================================
print(f"Подключение к порту {PORT}...")
try:
    ser = serial.Serial(PORT, BAUD_RATE, timeout=0.1)
except Exception as e:
    print(f"Ошибка открытия порта: {e}\nУбедись, что закрыл Serial Monitor в Arduino IDE!")
    sys.exit()

# ================= НАСТРОЙКА CSV (ДЛЯ EXCEL) =================
csv_filename = f"telemetry_log_{int(time.time())}.csv"
csv_file = open(csv_filename, 'w', newline='')
writer = csv.writer(csv_file, delimiter=';')  # Точка с запятой - стандарт для русскоязычного Excel
# Заголовки столбцов:
writer.writerow([
    "Real_Acc_X", "Real_Acc_Y", "Real_Acc_Z", "Real_Gyr_X", "Real_Gyr_Y", "Real_Gyr_Z",
    "Pred_Acc_X", "Pred_Acc_Y", "Pred_Acc_Z", "Pred_Gyr_X", "Pred_Gyr_Y", "Pred_Gyr_Z",
    "MSE_Loss", "Beam_Error_Deg"
])
print(f"Идет постоянная запись данных: {csv_filename}")
# ==============================================================

real_path = deque(maxlen=MAX_POINTS)
pred_path = deque(maxlen=MAX_POINTS)

# Координаты кисти на экране (с DECAY для красивой отрисовки)
real_pos = np.zeros(2)
pred_pos = np.zeros(2)

# Истинные пространственные углы без затухания (для физики бимформинга)
abs_real_pos = np.zeros(2)
abs_pred_pos = np.zeros(2)

# Последние сырые данные для текст-бокса
latest_real = np.zeros(6)
latest_pred = np.zeros(6)

# Настройка графического окна Pyside/Tkinter
fig = plt.figure(figsize=(14, 8))
fig.canvas.manager.set_window_title('Визуализация работы датчиков и ИИ')
gs = gridspec.GridSpec(4, 4)

# --- ЛЕВАЯ ЧАСТЬ (Траектория) ---
ax_traj = fig.add_subplot(gs[0:3, :3])
line_real, = ax_traj.plot([], [], 'b-', linewidth=4, label='Реальная рука')
line_pred, = ax_traj.plot([], [], 'r--', linewidth=2, label='ИИ (Предсказание)')
point_real, = ax_traj.plot([], [], 'bo', markersize=8)  # Точка конца кисти
point_pred, = ax_traj.plot([], [], 'ro', markersize=6)

ax_traj.legend(loc='upper right')
ax_traj.grid(True, linestyle=':', alpha=0.6)
ax_traj.set_title("Траектория (Гироскоп BMI160)", fontsize=16)
ax_traj.set_xlabel('Рыскание / Слева-Направо (X)')
ax_traj.set_ylabel('Тангаж / Снизу-Вверх (Y)')
ax_traj.set_facecolor('#f4f4f4')

# --- НИЖНЯЯ ЧАСТЬ (Оценка Бимформинга 6G) ---
ax_beam = fig.add_subplot(gs[3, :3])
error_history = deque([0] * 200, maxlen=200)  # История ошибки
line_err, = ax_beam.plot(range(200), error_history, 'r-', linewidth=2, label='Погрешность нацеливания луча')
ax_beam.axhspan(0, 5, facecolor='green', alpha=0.2, label='Зона успешного покрытия 6G (< 5°)')
ax_beam.axhline(5, color='darkgreen', linestyle='--', linewidth=1.5)
ax_beam.set_xlim(0, 200)
ax_beam.set_ylim(0, 10)  # Шкала ошибки (0 - 10 градусов)
ax_beam.set_title("Анализ трекинга антенной ФАР (пространственная метрика в градусах)", fontsize=11, weight='bold')
ax_beam.set_ylabel('Ошибка (°)')
ax_beam.set_xlabel('Фреймы')
ax_beam.legend(loc='upper right', fontsize=9)
ax_beam.grid(True, linestyle=':', alpha=0.6)

# --- ПРАВАЯ ЧАСТЬ (Телеметрия) ---
ax_info = fig.add_subplot(gs[:, 3])
ax_info.axis('off')

# Заголовки
text_real_title = ax_info.text(0.0, 0.95, "РЕАЛЬНЫЙ ДАТЧИК", weight='bold', color='blue', fontsize=12)
text_real_acc = ax_info.text(0.0, 0.78, "ACC:", fontsize=11, family='monospace')
text_real_gyr = ax_info.text(0.0, 0.61, "GYR:", fontsize=11, family='monospace')

text_pred_title = ax_info.text(0.0, 0.45, "ПРЕДСКАЗАННОЕ", weight='bold', color='red', fontsize=12)
text_pred_acc = ax_info.text(0.0, 0.28, "ACC:", fontsize=11, family='monospace')
text_pred_gyr = ax_info.text(0.0, 0.11, "GYR:", fontsize=11, family='monospace')

bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1)
text_loss = ax_info.text(0.0, -0.05, "MSE LOSS: 0.0000", color='green', weight='bold', fontsize=12, bbox=bbox_props)


def update(frame):
    global real_pos, pred_pos, latest_real, latest_pred

    updated = False
    while ser.in_waiting:
        try:
            line = ser.readline().decode('utf-8').strip()

            # Игнорируем системные и текстовые принты
            if not line or "nan" in line.lower() or "Init" in line or line.startswith(("#", "I")):
                continue

            parts = line.split(',')
            if len(parts) == 12:  # Мы ожидаем ровно 12 столбцов от 6 осей!
                vals = np.array([float(p) for p in parts])
                latest_real = vals[0:6]
                latest_pred = vals[6:12]

                # --- ИДЕАЛЬНЫЙ РАСЧЕТ ТРАЕКТОРИИ (Только Гироскоп) ---
                # Использование гироскопа (угловой скорости) дает невероятно плавную "воздушную мышь".
                # gy (Тангаж - Вниз/Вверх), gz (Рыскание - Влево/Вправо)

                real_dx = latest_real[4] * SCALE_GYR * DT  # Ось Y (Pitch) -> Screen X
                real_dy = latest_real[5] * SCALE_GYR * DT  # Ось Z (Yaw) -> Screen Y

                # Отрисовочные координаты с затуханием (чтобы курсор плавно возвращался в центр к нулю)
                real_pos[0] = (real_pos[0] + real_dx) * DECAY
                real_pos[1] = (real_pos[1] + real_dy) * DECAY
                real_path.append(real_pos.copy())

                pred_dx = latest_pred[4] * SCALE_GYR * DT
                pred_dy = latest_pred[5] * SCALE_GYR * DT
                pred_pos[0] = (pred_pos[0] + pred_dx) * DECAY
                pred_pos[1] = (pred_pos[1] + pred_dy) * DECAY
                pred_path.append(pred_pos.copy())

                # Истинное математическое интегрирование углов (без скейлов и затуханий)
                abs_real_pos[0] += latest_real[4] * DT
                abs_real_pos[1] += latest_real[5] * DT

                abs_pred_pos[0] += latest_pred[4] * DT
                abs_pred_pos[1] += latest_pred[5] * DT

                # Мгновенные подсчеты для таблицы
                inst_loss = np.mean((latest_real - latest_pred) ** 2)

                # Реальная ПРОСТРАНСТВЕННАЯ ошибка Бимформинга (в градусах)
                beam_err_deg = np.sqrt((abs_real_pos[0] - abs_pred_pos[0]) ** 2 +
                                       (abs_real_pos[1] - abs_pred_pos[1]) ** 2)

                # Постоянная дозапись в CSV файл
                writer.writerow([
                    round(latest_real[0], 4), round(latest_real[1], 4), round(latest_real[2], 4),
                    round(latest_real[3], 4), round(latest_real[4], 4), round(latest_real[5], 4),
                    round(latest_pred[0], 4), round(latest_pred[1], 4), round(latest_pred[2], 4),
                    round(latest_pred[3], 4), round(latest_pred[4], 4), round(latest_pred[5], 4),
                    round(inst_loss, 4), round(beam_err_deg, 4)
                ])

                updated = True
        except Exception:
            pass

    if updated and len(real_path) > 0:
        rx, ry = zip(*real_path)
        px, py = zip(*pred_path)

        line_real.set_data(rx, ry)
        line_pred.set_data(px, py)
        point_real.set_data([rx[-1]], [ry[-1]])
        point_pred.set_data([px[-1]], [py[-1]])

        max_bound = max(np.max(np.abs(real_path)), np.max(np.abs(pred_path)), 10.0) * 1.2
        ax_traj.set_xlim(-max_bound, max_bound)
        ax_traj.set_ylim(-max_bound, max_bound)

        text_real_acc.set_text(
            f"ACC (ускорение):\n  X: {latest_real[0]: 5.2f} G\n  Y: {latest_real[1]: 5.2f} G\n  Z: {latest_real[2]: 5.2f} G")
        text_real_gyr.set_text(
            f"GYR (вращение):\n  X: {latest_real[3]: 6.1f}°/s\n  Y: {latest_real[4]: 6.1f}°/s\n  Z: {latest_real[5]: 6.1f}°/s")

        text_pred_acc.set_text(
            f"ACC (ускорение):\n  X: {latest_pred[0]: 5.2f} G\n  Y: {latest_pred[1]: 5.2f} G\n  Z: {latest_pred[2]: 5.2f} G")
        text_pred_gyr.set_text(
            f"GYR (вращение):\n  X: {latest_pred[3]: 6.1f}°/s\n  Y: {latest_pred[4]: 6.1f}°/s\n  Z: {latest_pred[5]: 6.1f}°/s")

        # Расчет "ошибки" (MSE по всем осям)
        current_loss = np.mean((latest_real - latest_pred) ** 2)
        text_loss.set_text(f"MSE LOSS: {current_loss:.4f}")

        # График пространственной ошибки бимформинга
        current_beam_err = np.sqrt((abs_real_pos[0] - abs_pred_pos[0]) ** 2 + (abs_real_pos[1] - abs_pred_pos[1]) ** 2)
        error_history.append(current_beam_err)
        line_err.set_ydata(error_history)

    return line_real, line_pred, point_real, point_pred, text_real_acc, text_real_gyr, text_pred_acc, text_pred_gyr, text_loss, line_err


ani = animation.FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)
plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.1)  # Сдвигаем графики для красоты
plt.show()

# Закрываем порты и файлы при закрытии окна крестиком
ser.close()
csv_file.close()
print(f"Файл {csv_filename} успешно сохранен!")

