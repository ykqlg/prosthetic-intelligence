#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "struct_typedef.h"
#include "BMI088reg.h"
#include <wiringPi.h>
#include <signal.h>
#include "BMI088driver.h"

#define CHANNEL_GYRO 0
#define CHANNEL_ACC 1
#define CS_PIN 11
#define SPEED 500000
#define BUF_MAX 1024

SPI_HandleTypeDef* hspi_acc = NULL;
FILE * file = NULL;
void test(){
  while(1){
    int data = rand() % 100;
    printf("%d\n", data);
    usleep(10);  // 模拟传感器每秒产生一次数据
  }
}

void signal_handler(int signum) {
    if (file != NULL) {
        fclose(file); // 关闭文件
    }
    exit(signum); // 退出程序
}

int main(void) {
  signal(SIGINT, signal_handler);
  BMI088_init();

  // hspi_acc = BMI088_spi_init(CHANNEL_ACC, SPEED, DEBUG);
  hspi_acc = BMI088_spi_init(CHANNEL_ACC, SPEED, NO_DEBUG);

  fp32 accel[3];



  BMI088_accel_soft_reset();


  uint8_t dummy_data = 0;
  BMI088_accel_read_single_reg(BMI088_ACC_CHIP_ID,&dummy_data);

  BMI088_accel_write_single_reg(BMI088_ACC_PWR_CTRL, BMI088_ACC_ENABLE_ACC_ON); // turn on ACC power control
  BMI088_accel_write_single_reg(BMI088_ACC_PWR_CONF, BMI088_ACC_PWR_ACTIVE_MODE); // switch to active mode

  BMI088_accel_write_single_reg(BMI088_ACC_RANGE, BMI088_ACC_RANGE_3G); // set range to 3g
  
  
  BMI088_accel_self_test();

  file = fopen("data.csv", "w");
  if (file == NULL) {
    fprintf(stderr, "Error opening file.\n");
    return 1;
  }

  fprintf(file, "ACC_X,ACC_Y,ACC_Z\n"); 

  int count = 300;

  while(count--){
    
    BMI088_read(accel);
    // printf("%f %f %f\n",accel[0],accel[1],accel[2]);
    fprintf(file, "%f,%f,%f\n",accel[0],accel[1],accel[2]);

    delay(50);

  }
  fclose(file);

  printf("Successful end!!!\n");


  // test();
  

  return 0;

}

// ***************************************************************



