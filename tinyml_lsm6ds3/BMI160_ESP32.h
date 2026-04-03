#ifndef BMI160_ESP32_H
#define BMI160_ESP32_H

#include <Wire.h>

// I2C Адрес BMI160. По умолчанию часто бывает 0x69 (реже 0x68).
// Если датчик не находится, поменяйте на 0x68 
#define BMI160_ADDRESS 0x68

class BMI160_ESP32 {
public:
    BMI160_ESP32(TwoWire &wire = Wire);
    bool begin();
    void readMotion(float &ax, float &ay, float &az, float &gx, float &gy, float &gz);
    
private:
    TwoWire &_wire;
    void writeRegister(uint8_t reg, uint8_t value);
    void readRegisters(uint8_t reg, uint8_t *buffer, uint8_t length);
};

#endif // BMI160_ESP32_H
