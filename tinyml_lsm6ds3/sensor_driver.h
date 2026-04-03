#ifndef SENSOR_DRIVER_H
#define SENSOR_DRIVER_H

#include <Wire.h>
#include <FastIMU.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>

// Базовый абстрактный класс для всех датчиков
class BaseSensor {
public:
    virtual bool init() = 0;
    virtual void update() = 0;
    virtual void getRawData(float* raw) = 0; // Сохраняет актуальные данные в массив
    virtual String getName() = 0;          
};

// Реализация для датчика гироскопа и акселерометра LSM6DS3
class SensorLSM6DS3 : public BaseSensor {
private:
    LSM6DS3 IMU;
    calData calib = { 0 };
public:
    bool init() override {
        int err = IMU.init(calib, 0x6A);
        if (err != 0) err = IMU.init(calib, 0x6B);
        if (err != 0) return false;
        
        IMU.setAccelRange(2);
        IMU.setGyroRange(250);
        
        Serial.println("Оставьте плату LSM неподвижно для калибровки...");
        delay(500);
        IMU.calibrateAccelGyro(&calib);
        return true;
    }
    
    void update() override { 
        IMU.update(); 
    }
    
    void getRawData(float* raw) override {
        AccelData accel; GyroData gyro;
        IMU.getAccel(&accel); IMU.getGyro(&gyro);
        raw[0] = accel.accelX; raw[1] = accel.accelY; raw[2] = accel.accelZ;
        raw[3] = gyro.gyroX;   raw[4] = gyro.gyroY;   raw[5] = gyro.gyroZ;
    }
    
    String getName() override { return "LSM"; }
};

// Реализация для акселерометра SensorADXL345
class SensorADXL345 : public BaseSensor {
private:
    Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);
public:
    bool init() override {
        if (!accel.begin()) return false;
        accel.setRange(ADXL345_RANGE_16_G);
        return true;
    }
    
    void update() override { 
        // Библиотеке Adafruit Unified не нужен метод polling, она читает данные при getEvent
    }
    
    void getRawData(float* raw) override {
        sensors_event_t event;
        accel.getEvent(&event);
        // Библиотека Adafruit возвращает m/s^2. Модель ожидает G (ускорение свободного падения)
        raw[0] = event.acceleration.x / 9.80665f;
        raw[1] = event.acceleration.y / 9.80665f;
        raw[2] = event.acceleration.z / 9.80665f;
        raw[3] = 0; raw[4] = 0; raw[5] = 0; // Заполнение недостающих осей гироскопа нулями
    }
    
    String getName() override { return "ADXL"; }
};


class Sensor {
private:
    BaseSensor* impl;
public:
    Sensor(String name) {
        if (name == "ADXL") {
            impl = new SensorADXL345();
        } else {
            impl = new SensorLSM6DS3();
        }
    }
    ~Sensor() {
        if (impl != nullptr) delete impl;
    }

    bool init() { return impl->init(); }
    void update() { impl->update(); }
    void getRawData(float* raw) { impl->getRawData(raw); }
    String getName() { return impl->getName(); }
};

#endif
