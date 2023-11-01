#include "BMI088driver.h"
#include "BMI088reg.h"
#include <stdio.h>
#include <stdlib.h>
#include <wiringPi.h>
#include <wiringPiSPI.h>
#include "struct_typedef.h"
#include <string.h>

fp32 BMI088_ACCEL_SEN = BMI088_ACC_3G_SEN;
fp32 BMI088_GYRO_SEN = BMI088_GYRO_2000_SEN;

extern SPI_HandleTypeDef* hspi_acc;

SPI_HandleTypeDef* BMI088_spi_init(int channel,int speed, int debug){
  int fd = wiringPiSPISetup(channel,speed);
  if (fd == -1) {
    printf("SPI setup failed: Channel %d\n",channel);
    return NULL;
  }
  SPI_HandleTypeDef* hspi = malloc(sizeof(SPI_HandleTypeDef));
  hspi->wfd = fd;
  hspi->rfd = channel;
  hspi->debug = debug;

  return hspi;
}


void log_print(uint8_t *data, uint8_t len){
    printf("Received: ");
    while(len--){
        printf("%f ",data++);
    }
    printf("\n");
}
// **********************************
void BMI088_init(){
    if (wiringPiSetup() == -1) {
      printf("wiringPi setup failed\n");
      exit(1);
    }
}

void BMI088_read(fp32 accel[3])
{
    uint8_t buf[7];
    int16_t bmi088_raw_temp;

    BMI088_accel_read_muli_reg(BMI088_ACCEL_XOUT_L, buf, 7);

    bmi088_raw_temp = (int16_t)((buf[2]) << 8) | buf[1];
    accel[0] = bmi088_raw_temp * BMI088_ACCEL_SEN;
    bmi088_raw_temp = (int16_t)((buf[4]) << 8) | buf[3];
    accel[1] = bmi088_raw_temp * BMI088_ACCEL_SEN;
    bmi088_raw_temp = (int16_t)((buf[6]) << 8) | buf[5];
    accel[2] = bmi088_raw_temp * BMI088_ACCEL_SEN;

}

// *********** BMI088 I/O *****************

void BMI088_read_single_reg(SPI_HandleTypeDef* hspi, uint8_t reg, uint8_t *return_data)
{
    uint8_t buffer[] = {reg | 0x80, 0};
    wiringPiSPIDataRW(hspi->rfd, buffer, 2);
    *return_data = buffer[1];
    if(hspi->debug){
        // printf("Single reg: %02x %02x\n",buffer[0],buffer[1]);
        printf("Single reg: %02x\n",buffer[1]);
    }
}

void BMI088_write_single_reg(SPI_HandleTypeDef* hspi, uint8_t reg, uint8_t data)
{
    uint8_t buffer[2] = {reg, data};
    wiringPiSPIDataRW(hspi->rfd , buffer, 2);
    delay(50);
}

void BMI088_read_muli_reg(SPI_HandleTypeDef* hspi, uint8_t reg, uint8_t *buf, uint8_t len)
{
    uint8_t data;
    uint8_t *p = buf;
    uint8_t l = len;
    while (len--)
    {
        uint8_t buffer[] = {reg | 0x80, 0};
        wiringPiSPIDataRW(hspi->rfd, buffer, 2);
        *buf = buffer[1];
        buf++;
        reg++;
    }
    if(hspi->debug){
        printf("Multi reg: ");
        while(l--){
            printf("%02x ",*(p++));
        }
        printf("\n");

    }
}

void BMI088_write_muli_reg(SPI_HandleTypeDef* hspi, uint8_t reg, uint8_t* buf, uint8_t len )
{
   while(len != 0 ){
        BMI088_write_single_reg(hspi, reg, *buf);
        reg ++;
        buf ++;
        len --;
   }
}

void soft_reset(SPI_HandleTypeDef* hspi){
    uint8_t data[] = {BMI088_ACC_SOFTRESET, BMI088_ACC_SOFTRESET_VALUE}; 
    wiringPiSPIDataRW(hspi->rfd,data,2);
    delay(1);
}

// ********** Accelemeter ******************
void BMI088_accel_write_single_reg(uint8_t reg, uint8_t data){
    BMI088_write_single_reg(hspi_acc, reg, data);  
}

void BMI088_accel_read_single_reg(uint8_t reg, uint8_t* data){
    BMI088_read_single_reg(hspi_acc, reg, data);  
}

void BMI088_accel_read_muli_reg(uint8_t reg, uint8_t *data, uint8_t len ){
    BMI088_read_muli_reg(hspi_acc, reg, data, len);
}

void BMI088_accel_soft_reset(){
    soft_reset(hspi_acc);
}


// *******************************
void self_test(){
    BMI088_accel_write_single_reg(BMI088_ACC_RANGE, BMI088_ACC_RANGE_24G);
    BMI088_accel_write_single_reg(BMI088_ACC_CONF, BMI088_ACC_1600_HZ);

    BMI088_accel_write_single_reg(BMI088_ACC_SELF_TEST, BMI088_ACC_SELF_TEST_POSITIVE_SIGNAL);
    delay(10);

    fp32 accel_pos[3];
    BMI088_read(accel_pos);

    BMI088_accel_write_single_reg(BMI088_ACC_SELF_TEST, BMI088_ACC_SELF_TEST_NEGATIVE_SIGNAL);
    delay(10);

    fp32 accel_neg[3];
    BMI088_read(accel_neg);

    BMI088_accel_write_single_reg(BMI088_ACC_SELF_TEST, BMI088_ACC_SELF_TEST_OFF);
    // cal
    int len = 3;
    fp32 result[len];
    for(int i=0;i<len;i++){
        result[i] = accel_pos[i] - accel_neg[i];
    }

    printf("Received: ");
    for(int i = 0; i < len; i++){
        printf("%f ",result[i]);
    }
    printf("\n");
    


}

