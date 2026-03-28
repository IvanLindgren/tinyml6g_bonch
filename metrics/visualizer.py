import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque
import sys

# ================= НАСТРОЙКИ =================
PORT = "COM11"
BAUD_RATE = 115200

MAX_POINTS = 350   # 3.5 секунды истории (хватит на целое сердечко!)
DT = 0.01          

DECAY = 0.90       # Жесткое затухание для стабильности "воздушной кисти"
GRAVITY_ALPHA = 0.01
SCALE = 100.0      # Усилитель
# =============================================

print(f"🔌 Подключение к порту {PORT}...")
try:
    ser = serial.Serial(PORT, BAUD_RATE, timeout=0.1)
except Exception as e:
    print(f"❌ Ошибка открытия порта: {e}\nУбедись, что закрыл Serial Monitor!")
    sys.exit()

real_path = deque(maxlen=MAX_POINTS)
pred_path = deque(maxlen=MAX_POINTS)

# Стейт математики
gravity = np.zeros(3)
real_v = np.zeros(2) # Берем только оси X и Y для 2D-рисования
pred_v = np.zeros(2)

# Графика 2D
fig, ax = plt.subplots(figsize=(8, 8))

line_real, = ax.plot([], [], 'b-', linewidth=5, label='Твоя рука')
line_pred, = ax.plot([], [], 'r--', linewidth=2, label='ИИ 6G (Предсказание)')

ax.legend(loc='upper right')
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_title("Живое рисование жестов (2D)")
ax.set_xlabel('Влево - Вправо (X)')
ax.set_ylabel('Вперед - Назад (Y)')

def init():
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    return line_real, line_pred

def update(frame):
    global gravity, real_v, pred_v
    
    while ser.in_waiting:
        try:
            line = ser.readline().decode('utf-8').strip()
            
            if not line or "nan" in line.lower() or "Init" in line or line.startswith("A"):
                if not line.replace("-", "").replace(".", "").replace(",", "").isdigit():
                    continue

            parts = line.split(',')
            if len(parts) == 6:
                # Входные сырые данные (в G)
                acc = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
                p_acc = np.array([float(parts[3]), float(parts[4]), float(parts[5])])
                
                # Вырезаем гравитацию (Low-pass filter)
                if np.sum(gravity) == 0: gravity = acc
                else: gravity = gravity * (1 - GRAVITY_ALPHA) + acc * GRAVITY_ALPHA
                
                # Математика "Воздушной кисти" (Velocity Integration)
                # Берем только 0-й (X) и 1-й (Y) элементы
                lin_acc = (acc[:2] - gravity[:2]) * SCALE
                lin_p_acc = (p_acc[:2] - gravity[:2]) * SCALE
                
                # Интегрируем только один раз (до скорости). Это лучше всего 
                # рисует формы закрытых жестов типа круга или сердца
                real_v = real_v * DECAY + lin_acc * DT
                pred_v = pred_v * DECAY + lin_p_acc * DT
                
                real_path.append(real_v.copy())
                pred_path.append(pred_v.copy())
                
        except Exception:
            pass
            
    if len(real_path) > 0:
        rx, ry = zip(*real_path)
        px, py = zip(*pred_path)
        
        line_real.set_data(rx, ry)
        line_pred.set_data(px, py)
        
        # Динамическое масштабирование для 2D
        max_bound = max(np.max(np.abs(real_path)), np.max(np.abs(pred_path)), 5.0) * 1.2
        ax.set_xlim(-max_bound, max_bound)
        ax.set_ylim(-max_bound, max_bound)
        
    return line_real, line_pred

print("🚀 Запуск 2D-визуализатора! Можно рисовать сердечки...")
ani = animation.FuncAnimation(fig, update, init_func=init, blit=False, interval=20)
plt.show()

ser.close()
