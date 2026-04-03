#include "BMI160_ESP32.h"
#include <Arduino.h>

BMI160_ESP32::BMI160_ESP32(TwoWire &wire) : _wire(wire) {}

bool BMI160_ESP32::begin() {
    _wire.begin();
    
    // Проверка CHIP ID (у BMI160 он равен 0xD1)
    uint8_t chip_id;
    readRegisters(0x00, &chip_id, 1);
    if (chip_id != 0xD1) {
        Serial.printf("❌ BMI160 НЕ НАЙДЕН! Текущий ID: 0x%X (должен быть 0xD1)\n", chip_id);
        Serial.println("Возможно I2C адрес другой (0x68 вместо 0x69). Поменяйте в BMI160_ESP32.h!");
        return false;
    }

    // Инициализация (перевод из режима Suspend в Normal)
    // CMD: Accel Normal Mode
    writeRegister(0x7E, 0x11);
    delay(50);
    // CMD: Gyro Normal Mode
    writeRegister(0x7E, 0x15);
    delay(100);

    // Настраиваем Акселерометр: +-2G
    writeRegister(0x40, 0x28); // 100Hz ODR (Частота выборки)
    writeRegister(0x41, 0x03); // диапазон +-2G

    // Настраиваем Гироскоп: +-250 deg/s
    writeRegister(0x42, 0x28); // 100Hz ODR
    writeRegister(0x43, 0x03); // диапазон +-250
    
    return true;
}

void BMI160_ESP32::readMotion(float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
    uint8_t data[12];
    
    // В BMI160 регистры данных начинаются с 0x0C (сначала гироскоп, потом акселерометр)
    // Читаем сразу 12 байт (6 осей * 2 байта)
    readRegisters(0x0C, data, 12);
    
    // Гироскоп идет первым по регистрам: 0x0C - 0x11
    int16_t rawGx = (data[1] << 8) | data[0];
    int16_t rawGy = (data[3] << 8) | data[2];
    int16_t rawGz = (data[5] << 8) | data[4];
    
    // За ним Акселерометр: 0x12 - 0x17
    int16_t rawAx = (data[7] << 8) | data[6];
    int16_t rawAy = (data[9] << 8) | data[8];
    int16_t rawAz = (data[11] << 8) | data[10];
    
    // Перевод в реальные числа 
    // Масштаб для +-2g -> ~16384 LSB/g
    ax = rawAx / 16384.0f;
    ay = rawAy / 16384.0f;
    az = rawAz / 16384.0f;
    
    // Масштаб для +-250 deg/s -> ~131.2 LSB/(deg/s)
    gx = rawGx / 131.2f;
    gy = rawGy / 131.2f;
    gz = rawGz / 131.2f;
}

void BMI160_ESP32::writeRegister(uint8_t reg, uint8_t value) {
    _wire.beginTransmission(BMI160_ADDRESS);
    _wire.write(reg);
    _wire.write(value);
    _wire.endTransmission();
}

void BMI160_ESP32::readRegisters(uint8_t reg, uint8_t *buffer, uint8_t length) {
    _wire.beginTransmission(BMI160_ADDRESS);
    _wire.write(reg);
    _wire.endTransmission(false);
    _wire.requestFrom((uint8_t)BMI160_ADDRESS, length);
    for (uint8_t i = 0; i < length; i++) {
        buffer[i] = _wire.read();
    }
}
