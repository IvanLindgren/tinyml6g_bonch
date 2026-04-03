import socket
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
from collections import deque
import sys
import csv
import time

# ================= НАСТРОЙКИ =================
BAUD_RATE = 115200

MAX_POINTS = 50  # Длина "хвоста" траектории (чем меньше, тем быстрее стирается)
DT = 0.01

DECAY = 0.95  
SCALE_GYR = 0.5  # Масштаб отрисовки на графике

# ================= НАСТРОЙКА UDP СЕРВЕРА =================
UDP_IP = "0.0.0.0"  # Слушать все
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

print(f"UDP-Сервер запущен! Ожидание данных от костюмов на порту {UDP_PORT}...")
# =========================================================

# Импортируем вынесенную логику парсеров
from parsers_6g import sensor_parsers

# ================= НАСТРОЙКА CSV (ДЛЯ EXCEL) =================
csv_filename = f"telemetry_log_{int(time.time())}.csv"
csv_file = open(csv_filename, 'w', newline='')
writer = csv.writer(csv_file, delimiter=';')  # Точка с запятой - стандарт для русскоязычного Excel
# Заголовки столбцов:
writer.writerow([
    "Client_ID",
    "Real_Acc_X", "Real_Acc_Y", "Real_Acc_Z", "Real_Gyr_X", "Real_Gyr_Y", "Real_Gyr_Z",
    "Pred_Acc_X", "Pred_Acc_Y", "Pred_Acc_Z", "Pred_Gyr_X", "Pred_Gyr_Y", "Pred_Gyr_Z",
    "MSE_Loss", "Beam_Error_Deg"
])
print(f"Идет постоянная запись мульти-данных: {csv_filename}")
# ==============================================================

# Мульти-клиентная структура данных
clients = {}
colors_real = ['b', 'g', 'c', 'm', 'k']  # Цвета для реальных траекторий
colors_pred = ['r', 'orange', 'y', 'pink', 'purple']  # Цвета для ИИ

# Настройка графического окна
fig = plt.figure(figsize=(14, 8))
fig.canvas.manager.set_window_title('6G Federated TinyML - Multi-Client Supervisor')
gs = gridspec.GridSpec(4, 4)

# --- ЛЕВАЯ ЧАСТЬ (Траектория) ---
ax_traj = fig.add_subplot(gs[0:3, :3])
ax_traj.grid(True, linestyle=':', alpha=0.6)
ax_traj.set_title("Траектории координации датчиков (Живой эфир)", fontsize=16)
ax_traj.set_xlabel('Рыскание / Слева-Направо (X)')
ax_traj.set_ylabel('Тангаж / Снизу-Вверх (Y)')
ax_traj.set_facecolor('#f4f4f4')

# --- НИЖНЯЯ ЧАСТЬ (Оценка Бимформинга 6G) ---
ax_beam = fig.add_subplot(gs[3, :3])
ax_beam.axhspan(0, 5, facecolor='green', alpha=0.2, label='Зона захвата 6G (< 5°)')
ax_beam.axhline(5, color='darkgreen', linestyle='--', linewidth=1.5)
ax_beam.set_xlim(0, 200)
ax_beam.set_ylim(0, 15)  # Расширенная шкала ошибки
ax_beam.set_title("Оценка точности наведения для всех устройств", fontsize=11, weight='bold')
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
    updated = False
    artists = []  # Список графических элементов для обновления

    try:
        while True:  # Опустошаем очередь пакетов
            data, addr = sock.recvfrom(2048)
            line = data.decode('utf-8').strip()
            

            parts = line.split(',')
            if len(parts) >= 14:  # (ID, TYPE, 6 real, 6 pred)
                client_id = parts[0].strip()
                sensor_type = parts[1].strip()

                if sensor_type not in sensor_parsers:
                    continue

                vals = np.array([float(p) for p in parts[2:]])

                if client_id not in clients:
                    c_idx = len(clients) % len(colors_real)
                    c_real = colors_real[c_idx]
                    c_pred = colors_pred[c_idx]

                    lr, = ax_traj.plot([], [], color=c_real, linestyle='-', linewidth=3, label=f'{client_id} (Real)')
                    lp, = ax_traj.plot([], [], color=c_pred, linestyle='--', linewidth=2, label=f'{client_id} (Pred)')
                    pr, = ax_traj.plot([], [], color=c_real, marker='o', markersize=8)
                    pp, = ax_traj.plot([], [], color=c_pred, marker='o', markersize=6)
                    le, = ax_beam.plot(range(200), [0] * 200, color=c_pred, linewidth=2, label=f'Ошибка {client_id}')

                    ax_traj.legend(loc='upper right', fontsize=8)
                    ax_beam.legend(loc='upper right', fontsize=8)

                    clients[client_id] = {
                        'real_path': deque(maxlen=MAX_POINTS),
                        'pred_path': deque(maxlen=MAX_POINTS),
                        'real_pos': np.zeros(2),
                        'pred_pos': np.zeros(2),
                        'abs_real_pos': np.zeros(2),
                        'abs_pred_pos': np.zeros(2),
                        'latest_real': np.zeros(6),
                        'latest_pred': np.zeros(6),
                        'error_history': deque([0] * 200, maxlen=200),
                        'artists': (lr, lp, pr, pp, le)
                    }

                c = clients[client_id]

                
                success = sensor_parsers[sensor_type].parse_payload(vals, c)
                if not success: continue

                c['real_path'].append(c['real_pos'].copy())
                c['pred_path'].append(c['pred_pos'].copy())

                inst_loss = np.mean((c['latest_real'] - c['latest_pred']) ** 2)

                beam_err_deg = np.sqrt((c['abs_real_pos'][0] - c['abs_pred_pos'][0]) ** 2 +
                                       (c['abs_real_pos'][1] - c['abs_pred_pos'][1]) ** 2)

                writer.writerow([
                    client_id,
                    round(c['latest_real'][0], 4), round(c['latest_real'][1], 4), round(c['latest_real'][2], 4),
                    round(c['latest_real'][3], 4), round(c['latest_real'][4], 4), round(c['latest_real'][5], 4),
                    round(c['latest_pred'][0], 4), round(c['latest_pred'][1], 4), round(c['latest_pred'][2], 4),
                    round(c['latest_pred'][3], 4), round(c['latest_pred'][4], 4), round(c['latest_pred'][5], 4),
                    round(inst_loss, 4), round(beam_err_deg, 4)
                ])

                updated = True

                # Обновляем текст
                text_real_title.set_text(f"РЕАЛЬНЫЙ ДАТЧИК ({client_id} / {sensor_type})")
                text_real_acc.set_text(
                    f"ACC:\n  X: {c['latest_real'][0]: 5.2f} G\n  Y: {c['latest_real'][1]: 5.2f} G\n  Z: {c['latest_real'][2]: 5.2f} G")

                if sensor_type == "LSM":
                    text_real_gyr.set_text(
                        f"GYR:\n  X: {c['latest_real'][3]: 6.1f}°/s\n  Y: {c['latest_real'][4]: 6.1f}°/s\n  Z: {c['latest_real'][5]: 6.1f}°/s")
                    text_pred_gyr.set_text(
                        f"GYR:\n  X: {c['latest_pred'][3]: 6.1f}°/s\n  Y: {c['latest_pred'][4]: 6.1f}°/s\n  Z: {c['latest_pred'][5]: 6.1f}°/s")
                else:
                    text_real_gyr.set_text("GYR:\n  N/A (Только Аксель)")
                    text_pred_gyr.set_text("GYR:\n  N/A (Только Аксель)")

                text_pred_acc.set_text(
                    f"ACC:\n  X: {c['latest_pred'][0]: 5.2f} G\n  Y: {c['latest_pred'][1]: 5.2f} G\n  Z: {c['latest_pred'][2]: 5.2f} G")
                text_loss.set_text(f"MSE LOSS: {inst_loss:.4f}\nBEAM ERROR: {beam_err_deg:.2f}°")

    except BlockingIOError:
        pass  # Очередь пуста
    except Exception as e:
        pass

    if updated:
        global_max_bound = 10.0
        for client_id, c in clients.items():
            if len(c['real_path']) > 0:
                rx, ry = zip(*c['real_path'])
                px, py = zip(*c['pred_path'])

                c['artists'][0].set_data(rx, ry)
                c['artists'][1].set_data(px, py)
                c['artists'][2].set_data([rx[-1]], [ry[-1]])
                c['artists'][3].set_data([px[-1]], [py[-1]])

                c['error_history'].append(np.sqrt((c['abs_real_pos'][0] - c['abs_pred_pos'][0]) ** 2 +
                                                  (c['abs_real_pos'][1] - c['abs_pred_pos'][1]) ** 2))
                c['artists'][4].set_ydata(c['error_history'])

                artists.extend(c['artists'])

                max_b = max(np.max(np.abs(c['real_path'])), np.max(np.abs(c['pred_path']))) * 1.2
                if max_b > global_max_bound:
                    global_max_bound = max_b

        ax_traj.set_xlim(-global_max_bound, global_max_bound)
        ax_traj.set_ylim(-global_max_bound, global_max_bound)
        artists.extend([text_real_title, text_real_acc, text_real_gyr, text_pred_acc, text_pred_gyr, text_loss])

    return artists


ani = animation.FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)
plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.1)  
plt.show()

sock.close()
csv_file.close()
print(f"Файл {csv_filename} успешно сохранен!")
