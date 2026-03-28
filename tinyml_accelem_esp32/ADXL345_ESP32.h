#ifndef ADXL345_ESP32_H
#define ADXL345_ESP32_H

#include <Wire.h>

#define ADXL345_ADDRESS 0x53
#define ADXL345_REG_POWER_CTL 0x2D
#define ADXL345_REG_DATA_FORMAT 0x31
#define ADXL345_REG_DATAX0 0x32

#define ADXL345_RANGE_2G 0x00
#define ADXL345_RANGE_4G 0x01
#define ADXL345_RANGE_8G 0x02
#define ADXL345_RANGE_16G 0x03

class ADXL345_ESP32 {
public:
    ADXL345_ESP32(TwoWire &wire = Wire);
    void begin();
    void setRange(uint8_t range);
    void readAcceleration(float &x, float &y, float &z);
    
private:
    TwoWire &_wire;
    void writeRegister(uint8_t reg, uint8_t value);
    void readRegisters(uint8_t reg, uint8_t *buffer, uint8_t length);
};

#endif // ADXL345_ESP32_H
