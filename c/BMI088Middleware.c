#include "BMI088Middleware.h"
#include "struct_typedef.h"
#include "BMI088driver.h"
#include "BMI088reg.h"
#include <stdio.h>
#include <stdlib.h>
#include <wiringPi.h>
#include <wiringPiSPI.h>
#include <string.h>

SPI_HandleTypeDef* hspi_acc;

void BMI088_GPIO_init(void)
{
    if (wiringPiSetup() == -1) {
      printf("wiringPi setup failed\n");
      exit(1);
    }
}

void BMI088_com_init(void)
{
    int fd_acc = wiringPiSPISetup(CHANNEL_ACC,SPEED);
    if(fd_acc == -1){
        printf("SPI setup failed: Channel %d\n",CHANNEL_ACC);
    }
    hspi_acc = (SPI_HandleTypeDef*)malloc(sizeof(SPI_HandleTypeDef));
    hspi_acc->rfd = CHANNEL_ACC;
    hspi_acc->wfd = fd_acc;
    hspi_acc->debug = NO_DEBUG;

    // if(wiringPiSPISetup(CHANNEL_GYRO,SPEED) == -1){
    //     printf("SPI setup failed: Channel %d\n",CHANNEL_GYRO);
    // }
}

void BMI088_delay_ms(uint16_t ms)
{
    delay(ms);
}

void BMI088_delay_us(uint16_t us)
{
    delayMicroseconds(us);
}

