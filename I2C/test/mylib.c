#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <pthread.h>

// 硬件连接方式: 橙色 - 3V3, 黄色 - SCL, 绿色 - SDA, 蓝色 - GND

#define DEBUG_MOD 0 // 调试模式开关，调试模式将打印传感器配置详细信息

#define WHO_AM_I 0x0F
#define CTRL1 0x20
#define CTRL2 0x21
#define FIFO_CTRL 0x2E
#define CTRL6 0x25
#define STATUS 0x27 // 状态寄存器，当最低位为1时，表示有新的数据产生
#define OUT_X_L 0x28
#define OUT_X_H 0x29
#define OUT_Y_L 0x2A
#define OUT_Y_H 0x2B
#define OUT_Z_L 0x2C
#define OUT_Z_H 0x2D

#define Sensor_ADDRESS 0x19
#define BUFFER_SIZE 6 // (原来是126 = 6字节 x 21)
#define GSCALE 0.001952 
#define SENSOR_NUM 1   // 传感器个数
int sampleNum = 16000; // 采样数


// 定义一个结构体来保存每个加速度计的参数
typedef struct SensorInfo
{
    int sensorIndex;             // 传感器编号
    int i2cFile;                 // i2c设备
    unsigned char msgBuffer[BUFFER_SIZE]; // 缓冲数组
    char startTime[100];
} SensorInfo, *pSensor;

// 函数原型声明

void writeRegister(int i2cFile, unsigned char regAddress, unsigned char value);       // 向某个i2c设备的某个寄存器写入数据
unsigned char readRegOneByte(int i2cFile, unsigned char regAddress);                  // 从某个i2c设备的某个寄存器读取1个字节
void initializeByteStreaming(pSensor arg, unsigned char regAddress, int bufferSize); // 一次性从某个i2c设备的某个寄存器读取多个数据
void setup(pSensor arg);                                                             // 配置某个i2c设备
void loop(pSensor arg);                                                              // 某个i2c设备循环读取数据
void *sensorThread(void *arg);                                                        // 每个传感器所执行的线程
int Min(int a, int b);                                                                // 求两数最小
void initSensors(pSensor sensorPointer); // 初始化每个传感器的基本信息
int16_t read_dataZ(pSensor arg);

int main(int argc, char *argv[])
{
    // 检查是否传递了命令行参数
    if (argc >= 2)
    {
        // 将命令行参数转换为整数
        sampleNum = atoi(argv[1]);
        if (sampleNum <= 0)
            sampleNum = 16000;
    }
    // 结构体数组存储每个加速度计的信息
    SensorInfo accArgs[SENSOR_NUM];
    // 初始化每个传感器的基本信息
    initSensors(accArgs);
    // 为每个加速度计创建线程
    pthread_t threads[SENSOR_NUM]; // 指向线程标识符的指针
    for (int i = 0; i < SENSOR_NUM; ++i)
    {
        // 创建一个新的线程，该线程将执行sensorThread函数，并且通过参数accArgs传递了加速度计的基本信息
        // 如果线程创建成功，pthread_create函数将返回0，如果失败，则返回一个非零值
        if (pthread_create(&threads[i], NULL, sensorThread, (void *)&accArgs[i]) != 0)
        {
            perror("Failed to create thread");
            exit(EXIT_FAILURE);
        }
    }
    // 等待所有线程结束
    for (int i = 0; i < SENSOR_NUM; ++i)
    {
        pthread_join(threads[i], NULL);
    }
    return 0;
}

// 函数功能：初始化每个传感器的基本信息
void initSensors(pSensor sensorPointer)
{
    // 获取当前时间
    time_t currentTime;
    struct tm *localTime;
    time(&currentTime);
    localTime = localtime(&currentTime);
    // 格式化时间
    char formattedTime[100];
    strftime(formattedTime, sizeof(formattedTime), "%Y%m%d-%H%M%S", localTime);

    // 使用格式化字符串作为文件名
    char outputFolderName[100];
    snprintf(outputFolderName, sizeof(outputFolderName), "./data/%s",formattedTime);

    struct stat info;
    if (stat(outputFolderName, &info) != 0 || !S_ISDIR(info.st_mode)) {
        // 如果文件夹不存在，则创建文件夹
        if (mkdir(outputFolderName, 0777) != 0) {
            perror("Failed to create directory");
            exit(EXIT_FAILURE);
        }
        printf("Directory created successfully.\n");
    } else {
        printf("Directory already exists.\n");
    }
    

    for (int i = 0; i < 1; i++) // 改动了
    {
        // 每个传感器有自己的编号
        (sensorPointer + i)->sensorIndex = i;
        // 检查每个 I2C 设备是否成功打开
        char i2cPattern[16];
        snprintf(i2cPattern, sizeof(i2cPattern), "/dev/i2c-%d", i);
        (sensorPointer + i)->i2cFile = open(i2cPattern, O_RDWR);
        if ((sensorPointer + i)->i2cFile == -1)
        {
            printf("Failed to open /dev/i2c-%d\n", i);
            exit(EXIT_FAILURE);
        }
        
        strcpy((sensorPointer + i)->startTime, formattedTime);
    }
}

// 函数功能：向某个i2c设备的某个寄存器写入数据
void writeRegister(int i2cFile, unsigned char regAddress, unsigned char value)
{
    unsigned char buf[2] = {regAddress, value}; // 将 寄存器地址 和 数据 封装到一个大小为2字节的字节数组 buf 中
    if (DEBUG_MOD)
    {
        printf("Try to write 0x%02x into register 0x%02x\n", value, regAddress);
    }
    // 尝试将 buf 中的两个字节写入到 I2C 设备文件，检查写入的字节数是否等于sizeof(buf)
    // 如果不等于，说明写入失败，则打印错误信息并退出程序
    if (write(i2cFile, buf, sizeof(buf)) != sizeof(buf))
    {
        perror("Failed to write to I2C device");
        exit(EXIT_FAILURE);
    }
    if (DEBUG_MOD)
    {
        printf("Write sucessfully!\n");
    }
}

// 函数功能：从某个i2c设备的某个寄存器读取1个字节
unsigned char readRegOneByte(int i2cFile, unsigned char regAddress)
{
    // if (DEBUG_MOD)
    //     printf("Try to read 1 byte from register 0x%02x.\n", regAddress);
        // printf("fd: %d.\n", i2cFile);
    // 尝试将寄存器地址 regAddress 写入到 I2C 设备文件，检查写入的字节数是否等于sizeof(regAddress)
    // 如果不等于，说明写入失败，则打印错误信息并退出程序
    if (write(i2cFile, &regAddress, sizeof(regAddress)) != sizeof(regAddress))
    {
        perror("Failed to write to I2C device");
        exit(EXIT_FAILURE);
    }

    unsigned char value;
    // 尝试从 I2C 设备文件读取1个字节到变量 value，检查读入的字节数是否等于1
    // 如果不等于，说明写入失败，则打印错误信息并退出程序
    if (read(i2cFile, &value, sizeof(value)) != sizeof(value))
    {
        perror("Failed to read from I2C device");
        exit(EXIT_FAILURE);
    }
    // if (DEBUG_MOD)
    //     printf("Read successfully!\n");
    return value;
}

// 函数功能：一次性读取多个字节到buffer中
void initializeByteStreaming(pSensor arg, unsigned char regAddress, int bufferSize)
{
    if (DEBUG_MOD)
        printf("Try to read %d bytes from register 0x%02x\n", bufferSize, regAddress);
    // 尝试将寄存器地址 regAddress 中写入到 I2C 设备文件，检查写入的字节数是否等于sizeof(regAddress)
    // 如果不等于，说明写入失败，则打印错误信息并退出程序
    if (write(arg->i2cFile, &regAddress, sizeof(regAddress)) != sizeof(regAddress))
    {
        perror("Failed to write to I2C device");
        exit(EXIT_FAILURE);
    }
    // 尝试从 I2C 设备文件读取bufferSize个字节到变量 buffer ，检查读入的字节数是否等于bufferSize
    // 如果不等于，说明写入失败，则打印错误信息并退出程序
    if (bufferSize > 0)
    {
        if (read(arg->i2cFile, arg->msgBuffer, bufferSize) != bufferSize)
        {
            perror("Failed to read from I2C device");
            exit(EXIT_FAILURE);
        }
        if (DEBUG_MOD){
            printf("->C: ");
            for (int i = 0; i < bufferSize; i++) {
                printf("%02X ", arg->msgBuffer[i]);
            }           
            printf("\n");
            printf("Read successfully!\n");
        }
    }
    else
    {
        printf("Failed to read %d bytes!\n", bufferSize);
    }
}


int16_t read_dataZ(pSensor arg)
{
    unsigned char regAddress = OUT_X_L;
    int bufferSize = BUFFER_SIZE;
    u_int8_t* temp = (u_int8_t*)malloc(sizeof(u_int8_t)*bufferSize);
    if (DEBUG_MOD)
        printf("Try to read %d bytes from register 0x%02x\n", bufferSize, regAddress);
    // 尝试将寄存器地址 regAddress 中写入到 I2C 设备文件，检查写入的字节数是否等于sizeof(regAddress)
    // 如果不等于，说明写入失败，则打印错误信息并退出程序
    if (write(arg->i2cFile, &regAddress, sizeof(regAddress)) != sizeof(regAddress))
    {
        perror("Failed to write to I2C device");
        exit(EXIT_FAILURE);
    }
    // 尝试从 I2C 设备文件读取bufferSize个字节到变量 buffer ，检查读入的字节数是否等于bufferSize
    // 如果不等于，说明写入失败，则打印错误信息并退出程序
    if (bufferSize > 0)
    {
        if (read(arg->i2cFile, temp, bufferSize) != bufferSize)
        {
            perror("Failed to read from I2C device");
            exit(EXIT_FAILURE);
        }
        if (DEBUG_MOD){
            printf("Read successfully!\n");
        }
    }
    int16_t data_z = ((temp[4] << 8)+ temp[5]) * GSCALE;
    return data_z;
}
// 初始化设置
void setup(pSensor arg)
{
    // 使用 ioctl 函数来设置I2C通信中的从设备地址，如果失败，则打印错误信息并退出程序
    if (ioctl(arg->i2cFile, I2C_SLAVE, Sensor_ADDRESS) < 0)
    {
        perror("Failed to acquire bus access and/or talk to slave");
        exit(EXIT_FAILURE);
    }

    /* 配置加速度计 */
    writeRegister(arg->i2cFile, CTRL1, 0x97);     // CTRL1 - 1600 Hz输出数据速率，高性能模式
    writeRegister(arg->i2cFile, CTRL2, 0x04);     // CTRL2 - IF_ADD_INC: 在串行接口的多字节访问期间自动增加寄存器地址
    writeRegister(arg->i2cFile, FIFO_CTRL, 0xD0); // FIFO_CTRL [FMode2 FMode1 FMode0 FTH4 FTH3 FTH2 FTH1 FTH0]，连续模式: 如果FIFO已满，新样本将覆盖旧样本
    writeRegister(arg->i2cFile, CTRL6, 0x30);     // CTRL6 - 全量程选择: ±16 g

    char msg[64];
    /* 检查配置 */
    if (DEBUG_MOD)
    {
        sprintf(msg, "WHO_AM_I: 0x%02x", readRegOneByte(arg->i2cFile, WHO_AM_I)); // WHO_AM_I = 0x44 表示一个正常工作的加速度计
        puts(msg);
        sprintf(msg, "CTRL1: 0x%02x", readRegOneByte(arg->i2cFile, CTRL1));
        puts(msg);
        sprintf(msg, "CTRL2: 0x%02x", readRegOneByte(arg->i2cFile, CTRL2));
        puts(msg);
        sprintf(msg, "FIFO_CTRL: 0x%02x", readRegOneByte(arg->i2cFile, FIFO_CTRL));
        puts(msg);
        sprintf(msg, "CTRL6: 0x%02x", readRegOneByte(arg->i2cFile, CTRL6));
        puts(msg);
    }
}

// 循环读取数据并写入到文件
void loop(pSensor arg)
{
    
    // 使用格式化字符串作为文件名
    char outputFileName[100];
    snprintf(outputFileName, sizeof(outputFileName), "./data/%s/Sensor%d_%s.txt", arg->startTime, arg->sensorIndex, arg->startTime);

    // 打开要写入的文件
    FILE *file = fopen(outputFileName, "a");
    if (file == NULL)
    {
        perror("Failed to open output file for writing");
        exit(EXIT_FAILURE);
    }

    int totalByteNum = sampleNum * 6;                             // 总共读取的字节数，因为每个完整数据样本包含六个字节，即 [X_L, X_H, Y_L, Y_H, Z_L, Z_H]
    char *data = (char *)malloc(sizeof(char) * totalByteNum *2); // 从堆中申请空间，用于暂存读取到的数据
    int char_pointer = 0;                                         // 指向要插入数据的下一个位置

	clock_t start_time = clock();

    while (totalByteNum > 0) // 只要还有数据没有读取完，就继续读
    {
        if (readRegOneByte(arg->i2cFile, STATUS) & 1 == 1) // 检查状态寄存器，判断是否有新数据
        {
            initializeByteStreaming(arg, OUT_X_L, BUFFER_SIZE); // 读取数据到buffer中
            for (int i = 0; i < BUFFER_SIZE; ++i)
            {
                // 直接计算出对应的实际值，写入到csv
                // 好像应该是fp64类型
                int16_t dataX = ((arg->msgBuffer[1] << 8)+ arg->msgBuffer[0]) * GSCALE;
                int16_t dataX = ((arg->msgBuffer[3] << 8)+ arg->msgBuffer[2]) * GSCALE;
                int16_t dataX = ((arg->msgBuffer[5] << 8)+ arg->msgBuffer[4]) * GSCALE;
        		fprintf(file, "%f,%f,%f,%f\n", accel[0], accel[1], accel[2], ((double)(time - start_time)) / CLOCKS_PER_SEC);

                sprintf(data + char_pointer, "%02x", arg->msgBuffer[i]); // 将寄存器中读取到的16进制数据写入到data数组
                char_pointer += 2;
            }
            totalByteNum -= BUFFER_SIZE;
        }
    }

    clock_t end_time = clock();
	double cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
	printf("==> Execution Time: %fs\n", cpu_time_used);

    fprintf(file, "%s", data); // 统一向输出文件写入数据
    free(data);                // 释放内存
    fclose(file);              // 关闭输出文件

    printf("\nSensor %d completed!\n", arg->sensorIndex);
}

// 每个加速度计所执行的线程
void *sensorThread(void *arg)
{
    // 获取加速度计信息
    struct SensorInfo *info = (struct SensorInfo *)arg;
    // 配置加速度计参数
    setup(info);
    // 循环读取数据
    loop(info);
    // 关闭I2C设备
    close(info->i2cFile);
    // 退出线程
    pthread_exit(NULL);
}
