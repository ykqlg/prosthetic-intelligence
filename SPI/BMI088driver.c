#include "BMI088driver.h"
#include "BMI088reg.h"
#include "BMI088Middleware.h"
#include <stdio.h>
#include <stdlib.h>
#include <wiringPi.h>
#include <wiringPiSPI.h>
#include "struct_typedef.h"
#include <string.h>

fp32 BMI088_ACCEL_SEN = BMI088_ACC_3G_SEN;
fp32 BMI088_GYRO_SEN = BMI088_GYRO_2000_SEN;

extern SPI_HandleTypeDef *hspi_acc;

// SPI_HandleTypeDef *BMI088_spi_init(int channel, int speed, int debug)
// {
//     int fd = wiringPiSPISetup(channel, speed);
//     if (fd == -1)
//     {
//         printf("SPI setup failed: Channel %d\n", channel);
//         return NULL;
//     }
//     SPI_HandleTypeDef *hspi = (SPI_HandleTypeDef *)malloc(sizeof(SPI_HandleTypeDef));
//     hspi->wfd = fd;
//     hspi->rfd = channel;
//     hspi->debug = debug;

//     return hspi;
// }

static uint8_t write_BMI088_accel_reg_data_error[BMI088_WRITE_ACCEL_REG_NUM][3] =
    {
        {BMI088_ACC_PWR_CTRL, BMI088_ACC_ENABLE_ACC_ON, BMI088_ACC_PWR_CTRL_ERROR},
        {BMI088_ACC_PWR_CONF, BMI088_ACC_PWR_ACTIVE_MODE, BMI088_ACC_PWR_CONF_ERROR},
        {BMI088_ACC_CONF, BMI088_ACC_NORMAL | BMI088_ACC_1600_HZ | BMI088_ACC_CONF_MUST_Set, BMI088_ACC_CONF_ERROR},
        {BMI088_ACC_RANGE, BMI088_ACC_RANGE_3G, BMI088_ACC_RANGE_ERROR},
        {BMI088_INT1_IO_CTRL, BMI088_ACC_INT1_IO_ENABLE | BMI088_ACC_INT1_GPIO_PP | BMI088_ACC_INT1_GPIO_LOW, BMI088_INT1_IO_CTRL_ERROR},
        {BMI088_INT_MAP_DATA, BMI088_ACC_INT1_DRDY_INTERRUPT, BMI088_INT_MAP_DATA_ERROR}

};

bool_t bmi088_accel_init(void)
{
    uint8_t res = 0;
    uint8_t write_reg_num = 0;

    BMI088_accel_soft_reset();

    BMI088_accel_read_single_reg(BMI088_ACC_CHIP_ID, &res);
    BMI088_delay_us(BMI088_COM_WAIT_SENSOR_TIME);
    BMI088_accel_read_single_reg(BMI088_ACC_CHIP_ID, &res);
    BMI088_delay_us(BMI088_COM_WAIT_SENSOR_TIME);

    // set up accelerometer sensor configration
    for (write_reg_num = 0; write_reg_num < BMI088_WRITE_ACCEL_REG_NUM; write_reg_num++)
    {
        BMI088_accel_write_single_reg(write_BMI088_accel_reg_data_error[write_reg_num][0], write_BMI088_accel_reg_data_error[write_reg_num][1]);
        BMI088_delay_us(BMI088_COM_WAIT_SENSOR_TIME);
    }
    return BMI088_NO_ERROR;
}

uint8_t BMI088_init()
{

    uint8_t error = BMI088_NO_ERROR;
    // GPIO and SPI Init .
    BMI088_GPIO_init();
    BMI088_com_init();

    error |= bmi088_accel_init();
    return error;
}

void BMI088_read(fp32 accel[3])
{
    uint8_t buf[7];
    int16_t bmi088_raw_temp;

    BMI088_accel_read_muli_reg(BMI088_ACCEL_XOUT_L, buf, 7);

    bmi088_raw_temp = (int16_t)((buf[2]) << 8) | buf[1];
    accel[0] = (fp64)bmi088_raw_temp * BMI088_ACCEL_SEN;
    bmi088_raw_temp = (int16_t)((buf[4]) << 8) | buf[3];
    accel[1] = (fp64)bmi088_raw_temp * BMI088_ACCEL_SEN;
    bmi088_raw_temp = (int16_t)((buf[6]) << 8) | buf[5];
    accel[2] = (fp64)bmi088_raw_temp * BMI088_ACCEL_SEN;
}

// *********** BMI088 general control *****************

void BMI088_read_single_reg(SPI_HandleTypeDef *hspi, uint8_t reg, uint8_t *return_data)
{
    uint8_t buffer[] = {reg | 0x80, 0};
    wiringPiSPIDataRW(hspi->rfd, buffer, 2);
    *return_data = buffer[1];
    if (hspi->debug)
    {
        // printf("Single reg: %02x %02x\n",buffer[0],buffer[1]);
        printf("Single reg: %02x\n", buffer[1]);
    }
}

void BMI088_write_single_reg(SPI_HandleTypeDef *hspi, uint8_t reg, uint8_t data)
{
    uint8_t buffer[2] = {reg, data};
    wiringPiSPIDataRW(hspi->rfd, buffer, 2);
    delay(50);
}

void BMI088_read_muli_reg(SPI_HandleTypeDef *hspi, uint8_t reg, uint8_t *buf, uint8_t len)
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
    if (hspi->debug)
    {
        printf("Multi reg: ");
        while (l--)
        {
            printf("%02x ", *(p++));
        }
        printf("\n");
    }
}

void BMI088_write_muli_reg(SPI_HandleTypeDef *hspi, uint8_t reg, uint8_t *buf, uint8_t len)
{
    while (len != 0)
    {
        BMI088_write_single_reg(hspi, reg, *buf);
        reg++;
        buf++;
        len--;
    }
}

void self_test()
{
    fp32 accel_pos[3];
    fp32 accel_neg[3];

    BMI088_accel_write_single_reg(BMI088_ACC_RANGE, BMI088_ACC_RANGE_24G);
    BMI088_accel_write_single_reg(BMI088_ACC_CONF, BMI088_ACC_NORMAL | BMI088_ACC_1600_HZ);
    BMI088_accel_write_single_reg(BMI088_ACC_SELF_TEST, BMI088_ACC_SELF_TEST_POSITIVE_SIGNAL);
    BMI088_read(accel_pos);
    BMI088_accel_write_single_reg(BMI088_ACC_SELF_TEST, BMI088_ACC_SELF_TEST_NEGATIVE_SIGNAL);
    BMI088_read(accel_neg);
    BMI088_accel_write_single_reg(BMI088_ACC_SELF_TEST, BMI088_ACC_SELF_TEST_OFF);

    int len = 3;
    fp32 result[len];
    for (int i = 0; i < len; i++)
    {
        result[i] = accel_pos[i] - accel_neg[i];
    }
    if (!(result[0] >= 1000 && result[1] >= 1000 && result[2] >= 500))
    {
        printf("Self Test: Not satisfy the expected values!!! \n ");
        exit(1);
    }

    if (hspi_acc->debug)
    {
        printf("Received: ");
        for (int i = 0; i < len; i++)
        {
            printf("%f ", result[i]);
        }
        printf("\n");
    }
}

void soft_reset(SPI_HandleTypeDef *hspi)
{
    uint8_t data[] = {BMI088_ACC_SOFTRESET, BMI088_ACC_SOFTRESET_VALUE};
    wiringPiSPIDataRW(hspi->rfd, data, 2);
    delay(1);
}

// ********** Accelemeter access ******************
void BMI088_accel_write_single_reg(uint8_t reg, uint8_t data)
{
    uint8_t t;
    BMI088_accel_read_single_reg(BMI088_ACC_CHIP_ID, &t);
    BMI088_write_single_reg(hspi_acc, reg, data);
}

void BMI088_accel_read_single_reg(uint8_t reg, uint8_t *data)
{
    BMI088_read_single_reg(hspi_acc, reg, data);
}

void BMI088_accel_read_muli_reg(uint8_t reg, uint8_t *data, uint8_t len)
{
    BMI088_read_muli_reg(hspi_acc, reg, data, len);
}

void BMI088_accel_soft_reset()
{
    soft_reset(hspi_acc);
}

void BMI088_accel_self_test()
{
    self_test();
}
