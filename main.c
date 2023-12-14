#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>

#include "struct_typedef.h"
#include "BMI088reg.h"
#include "BMI088driver.h"
#include "BMI088Middleware.h"

// SPI_HandleTypeDef* hspi_acc = NULL;
extern SPI_HandleTypeDef *hspi_acc;
FILE *file = NULL;

void signal_handler(int signum)
{
	if (file != NULL)
	{
		fclose(file);
	}
	exit(signum);
}

void createFileNameWithTime(char *filePath);

int main(void)
{
	// 创建准备写入的文件
	char filePath[50];
	createFileNameWithTime(filePath);
	file = fopen(filePath, "w");
	if (file == NULL)
	{
		fprintf(stderr, "Error opening file.\n");
		return 1;
	}
	fprintf(file, "ACC_X,ACC_Y,ACC_Z\n");

	fp32 accel[3];
	signal(SIGINT, signal_handler); // 使用 ctrl+C 终止

	BMI088_init();

	// BMI088_accel_self_test();

	int count = 30000;

	clock_t start_time = clock();

	while (count--)
	{

		BMI088_read(accel);
		printf("%f %f %f\n", accel[0], accel[1], accel[2]);
		fprintf(file, "%f,%f,%f\n", accel[0], accel[1], accel[2]);
		// uint8_t t;
		// BMI088_accel_read_single_reg(BMI088_ACC_RANGE, &t);
		// printf("%d\n",t);
	}
	fclose(file);

	printf("Successful end!!!\n");

	clock_t end_time = clock();
	double cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
	printf("程序执行时间为 %f 秒\n", cpu_time_used);

	return 0;
}

// 创建以程序运行时间命名的文件，返回文件路径字符串
void createFileNameWithTime(char *filePath)
{
	time_t currentTime;
	struct tm *localTime;
	char timeString[20]; // 存储格式化后的时间字符串，如"2023-12-13_12-34-56"

	// 获取当前时间
	time(&currentTime);
	localTime = localtime(&currentTime);

	// 格式化时间字符串
	strftime(timeString, sizeof(timeString), "%Y%m%d_%H%M%S", localTime);

	// 构建文件路径
	sprintf(filePath, "output/%s.csv", timeString); // 假设存储在名为 "output" 的文件夹中
}