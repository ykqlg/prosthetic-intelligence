#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "struct_typedef.h"
#include "BMI088reg.h"
#include <wiringPi.h>
#include "BMI088driver.h"

#define CHANNEL_GYRO 0
#define CHANNEL_ACC 1
#define CS_PIN 11
#define SPEED 500000
#define BUF_MAX 1024

SPI_HandleTypeDef* hspi_acc;

int main(void) {

  BMI088_init();

  // hspi_acc = BMI088_spi_init(CHANNEL_ACC, SPEED, DEBUG);
  hspi_acc = BMI088_spi_init(CHANNEL_ACC, SPEED, NO_DEBUG);

  fp32 accel[3];


  // ************************

  BMI088_accel_soft_reset();


  uint8_t dummy_data = 0;
  BMI088_accel_read_single_reg(BMI088_ACC_CHIP_ID,&dummy_data);

  BMI088_accel_write_single_reg(BMI088_ACC_PWR_CTRL, BMI088_ACC_ENABLE_ACC_ON); // turn on ACC power control
  BMI088_accel_write_single_reg(BMI088_ACC_PWR_CONF, BMI088_ACC_PWR_ACTIVE_MODE); // switch to active mode

  BMI088_accel_write_single_reg(BMI088_ACC_RANGE, BMI088_ACC_RANGE_3G); // set range to 3g
  
  
  BMI088_accel_self_test();

  while(1){
    uint8_t rxdata = 0;    
    // BMI088_accel_read_single_reg(BMI088_ACC_PWR_CONF,&rxdata);

    // BMI088_accel_write_single_reg(BMI088_ACCEL_XOUT_M,0x00);
    // BMI088_accel_read_single_reg(BMI088_ACCEL_XOUT_M,&rxdata);
    // uint8_t buf[8] = {0, 0, 0, 0, 0, 0,0,0};

    // BMI088_accel_read_muli_reg(BMI088_ACCEL_XOUT_L, buf, 8);
    // printf("received: %02x\n",rxdata);
    
    // BMI088_read(accel);
    // printf("%f %f %f\n",accel[0],accel[1],accel[2]);
    

    // delay(50);


    // 获取传感器数据，假设传感器数据存储在变量 sensor_data 中
    int sensor_data = rand() % 100; // 这里使用随机数模拟传感器数据

    // 将传感器数据通过标准输出打印出来
    printf("%d\n", sensor_data);

    // 每隔1秒更新一次数据
    sleep(1);
  }
  
  return 0;


}
