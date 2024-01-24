#ifndef BMI088DRIVER_H
#define BMI088DRIVER_H
#include <math.h>
#include "BMI088driver.h"
#include "BMI088reg.h"
#include "struct_typedef.h"

#define DEBUG 1
#define NO_DEBUG 0

#define BMI088_TEMP_FACTOR 0.125f
#define BMI088_TEMP_OFFSET 23.0f

#define BMI088_WRITE_ACCEL_REG_NUM 6
#define BMI088_WRITE_GYRO_REG_NUM 6

#define BMI088_GYRO_DATA_READY_BIT 0
#define BMI088_ACCEL_DATA_READY_BIT 1
#define BMI088_ACCEL_TEMP_DATA_READY_BIT 2

#define BMI088_LONG_DELAY_TIME 80
#define BMI088_COM_WAIT_SENSOR_TIME 150


#define BMI088_ACCEL_IIC_ADDRESSE (0x18 << 1)
#define BMI088_GYRO_IIC_ADDRESSE (0x68 << 1)

// #define BMI088_ACC_RANGE_3G_FACTOR (1/(32768*1000*2^(0x00+1)*1.5))
// #define BMI088_ACC_RANGE_6G_FACTOR (1/(32768*1000*2^(0x01+1)*1.5))
// #define BMI088_ACC_RANGE_12G_FACTOR (1/(32768*1000*2^(0x02+1)*1.5))
// #define BMI088_ACC_RANGE_24G_FACTOR (1/(32768*1000*2^(0x03+1)*1.5))

// #define BMI088_ACCEL_3G_SEN 0.0008974358974f
// #define BMI088_ACCEL_6G_SEN 0.00179443359375f
// #define BMI088_ACCEL_12G_SEN 0.0035888671875f
// #define BMI088_ACCEL_24G_SEN 0.007177734375f

#define BMI088_ACC_3G_SEN (1000*2*1.5/32768)
#define BMI088_ACC_6G_SEN (1000*4*1.5/32768)
#define BMI088_ACC_12G_SEN (1000*8*1.5/32768)
#define BMI088_ACC_24G_SEN (1000*16*1.5/32768)

#define BMI088_GYRO_2000_SEN 0.00106526443603169529841533860381f
#define BMI088_GYRO_1000_SEN 0.00053263221801584764920766930190693f
#define BMI088_GYRO_500_SEN 0.00026631610900792382460383465095346f
#define BMI088_GYRO_250_SEN 0.00013315805450396191230191732547673f
#define BMI088_GYRO_125_SEN 0.000066579027251980956150958662738366f

typedef struct __SPI_HandleTypeDef{
    int debug;
    int rfd; 
    int wfd; 
}SPI_HandleTypeDef;

SPI_HandleTypeDef* BMI088_spi_init(int channel,int speed, int debug);


enum
{
    BMI088_NO_ERROR = 0x00,
    BMI088_ACC_PWR_CTRL_ERROR = 0x01,
    BMI088_ACC_PWR_CONF_ERROR = 0x02,
    BMI088_ACC_CONF_ERROR = 0x03,
    BMI088_ACC_SELF_TEST_ERROR = 0x04,
    BMI088_ACC_RANGE_ERROR = 0x05,
    BMI088_INT1_IO_CTRL_ERROR = 0x06,
    BMI088_INT_MAP_DATA_ERROR = 0x07,
    BMI088_GYRO_RANGE_ERROR = 0x08,
    BMI088_GYRO_BANDWIDTH_ERROR = 0x09,
    BMI088_GYRO_LPM1_ERROR = 0x0A,
    BMI088_GYRO_CTRL_ERROR = 0x0B,
    BMI088_GYRO_INT3_INT4_IO_CONF_ERROR = 0x0C,
    BMI088_GYRO_INT3_INT4_IO_MAP_ERROR = 0x0D,

    BMI088_SELF_TEST_ACCEL_ERROR = 0x80,
    BMI088_SELF_TEST_GYRO_ERROR = 0x40,
    BMI088_NO_SENSOR = 0xFF,
};

void BMI088_accel_write_single_reg(uint8_t reg, uint8_t data);
void BMI088_accel_read_single_reg(uint8_t reg, uint8_t* data);
void BMI088_accel_read_muli_reg(uint8_t reg, uint8_t *data, uint8_t len );
// void BMI088_accel_write_muli_reg(uint8_t reg, uint8_t *data, uint8_t len );
void BMI088_accel_soft_reset();
void BMI088_accel_self_test();



uint8_t BMI088_init(void);

bool_t bmi088_accel_init(void);
// bool_t bmi088_gyro_init(void);
void BMI088_read(fp32 accel[3]);



#endif
