#include "ADXL345_ESP32.h"

ADXL345_ESP32::ADXL345_ESP32(TwoWire &wire) : _wire(wire) {}

void ADXL345_ESP32::begin() {
    _wire.begin();
    writeRegister(ADXL345_REG_POWER_CTL, 0x08); // Enable measurements
}

void ADXL345_ESP32::setRange(uint8_t range) {
    writeRegister(ADXL345_REG_DATA_FORMAT, range);
}

void ADXL345_ESP32::readAcceleration(float &x, float &y, float &z) {
    uint8_t data[6];
    readRegisters(ADXL345_REG_DATAX0, data, 6);
    
    int16_t rawX = (data[1] << 8) | data[0];
    int16_t rawY = (data[3] << 8) | data[2];
    int16_t rawZ = (data[5] << 8) | data[4];
    
    x = rawX * 0.0039; // Scale factor for 2G
    y = rawY * 0.0039;
    z = rawZ * 0.0039;
}

void ADXL345_ESP32::writeRegister(uint8_t reg, uint8_t value) {
    _wire.beginTransmission(ADXL345_ADDRESS);
    _wire.write(reg);
    _wire.write(value);
    _wire.endTransmission();
}

void ADXL345_ESP32::readRegisters(uint8_t reg, uint8_t *buffer, uint8_t length) {
    _wire.beginTransmission(ADXL345_ADDRESS);
    _wire.write(reg);
    _wire.endTransmission(false);
    _wire.requestFrom(ADXL345_ADDRESS, length);
    for (uint8_t i = 0; i < length; i++) {
        buffer[i] = _wire.read();
    }
}
