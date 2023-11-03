#ifndef BMI088MIDDLEWARE_H
#define BMI088MIDDLEWARE_H

#include "struct_typedef.h"

#define BMI088_USE_SPI
//#define BMI088_USE_IIC

#define CHANNEL_GYRO 0
#define CHANNEL_ACC 1
#define CS_PIN 11
#define SPEED 500000

void BMI088_GPIO_init(void);
void BMI088_com_init(void);
void BMI088_delay_ms(uint16_t ms);
void BMI088_delay_us(uint16_t us);


#endif
