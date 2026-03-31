import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
from collections import deque
import csv
import time
import os

# ================= НАСТРОЙКИ =================
PORT = "COM12"
BAUD_RATE = 115200
MAX_SUITS = 3  # Сколько максимум костюмов выводить на экран
MAX_POINTS = 50  # Длина хвоста траектории
DT = 0.01
DECAY = 0.95
SCALE_GYR = 0.5
# =============================================

if not os.path.exists('suit_logs'):
    os.makedirs('suit_logs')


class SuitProcessor:
    """Класс для обработки данных конкретного узла/костюма"""

    def __init__(self, suit_id, index):
        self.suit_id = suit_id
        self.index = index

        # Данные траектории
        self.real_path = deque(maxlen=MAX_POINTS)
        self.pred_path = deque(maxlen=MAX_POINTS)
        self.real_pos = np.zeros(2)
        self.pred_pos = np.zeros(2)
        self.abs_real_pos = np.zeros(2)
        self.abs_pred_pos = np.zeros(2)

        # Ошибки
        self.error_history = deque([0] * 100, maxlen=100)
        self.latest_loss = 0
        self.latest_beam_err = 0

        # Создание персонального CSV
        self.filename = f"suit_logs/log_{suit_id}_{int(time.time())}.csv"
        self.csv_file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.csv_file, delimiter=';')
        self.writer.writerow(["R_AccX", "R_AccY", "R_AccZ", "R_GyrX", "R_GyrY", "R_GyrZ",
                              "P_AccX", "P_AccY", "P_AccZ", "P_GyrX", "P_GyrY", "P_GyrZ", "MSE", "BeamErr"])

    def update_data(self, vals):
        latest_real = vals[0:6]
        latest_pred = vals[6:12]

        # Интегрирование траектории (Гироскоп)
        dx_r, dy_r = latest_real[4] * SCALE_GYR * DT, latest_real[5] * SCALE_GYR * DT
        self.real_pos = (self.real_pos + [dx_r, dy_r]) * DECAY
        self.real_path.append(self.real_pos.copy())

        dx_p, dy_p = latest_pred[4] * SCALE_GYR * DT, latest_pred[5] * SCALE_GYR * DT
        self.pred_pos = (self.pred_pos + [dx_p, dy_p]) * DECAY
        self.pred_path.append(self.pred_pos.copy())

        # Истинные углы (для бимформинга)
        self.abs_real_pos += [latest_real[4] * DT, latest_real[5] * DT]
        self.abs_pred_pos += [latest_pred[4] * DT, latest_pred[5] * DT]

        self.latest_loss = np.mean((latest_real - latest_pred) ** 2)
        self.latest_beam_err = np.sqrt(np.sum((self.abs_real_pos - self.abs_pred_pos) ** 2))
        self.error_history.append(self.latest_beam_err)

        # Запись в лог
        self.writer.writerow(list(latest_real) + list(latest_pred) + [self.latest_loss, self.latest_beam_err])


# --- ИНИЦИАЛИЗАЦИЯ ИНТЕРФЕЙСА ---
fig = plt.figure(figsize=(16, 9))
gs = gridspec.GridSpec(2, MAX_SUITS)
suits = {}  # Словарь для хранения объектов SuitProcessor

# Подготовка пустых осей (на MAX_SUITS устройств)
traj_axes = []
err_axes = []
lines_real = [];
lines_pred = [];
lines_err = []

for i in range(MAX_SUITS):
    # Верхний ряд - траектории
    ax = fig.add_subplot(gs[0, i])
    ax.set_title(f"Suit Slot {i + 1}: Waiting...")
    lr, = ax.plot([], [], 'b-', lw=2, label='Real')
    lp, = ax.plot([], [], 'r--', lw=1, label='AI Pred')
    ax.set_xlim(-5, 5);
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.2)
    traj_axes.append(ax)
    lines_real.append(lr);
    lines_pred.append(lp)

    # Нижний ряд - ошибки бимформинга
    ax_e = fig.add_subplot(gs[1, i])
    le, = ax_e.plot(range(100), [0] * 100, 'g-')
    ax_e.set_ylim(0, 15);
    ax_e.set_title("Beam Error (°)")
    err_axes.append(ax_e)
    lines_err.append(le)

ser = serial.Serial(PORT, BAUD_RATE, timeout=0.01)


def animate(frame):
    while ser.in_waiting:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        parts = line.split(',')

        # Ожидаем формат: ID, R_AccX, ..., P_GyrZ (всего 13 значений)
        if len(parts) == 13:
            suit_id = parts[0]
            vals = np.array([float(p) for p in parts[1:]])

            if suit_id not in suits and len(suits) < MAX_SUITS:
                suits[suit_id] = SuitProcessor(suit_id, len(suits))
                traj_axes[suits[suit_id].index].set_title(f"Device: {suit_id}")

            if suit_id in suits:
                suits[suit_id].update_data(vals)

    # Отрисовка всех активных устройств
    for sid, suit in suits.items():
        if len(suit.real_path) > 0:
            rx, ry = zip(*suit.real_path)
            px, py = zip(*suit.pred_path)

            lines_real[suit.index].set_data(rx, ry)
            lines_pred[suit.index].set_data(px, py)
            lines_err[suit.index].set_ydata(suit.error_history)

            # Автомасштаб
            max_v = max(np.max(np.abs(suit.real_path)), 2.0) * 1.2
            traj_axes[suit.index].set_xlim(-max_v, max_v)
            traj_axes[suit.index].set_ylim(-max_v, max_v)

    return lines_real + lines_pred + lines_err


ani = animation.FuncAnimation(fig, animate, interval=30, blit=False)
plt.tight_layout()
plt.show()

# Закрытие файлов
for s in suits.values():
    s.csv_file.close()
ser.close()
