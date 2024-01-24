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
	fprintf(file, "ACC_X,ACC_Y,ACC_Z,Time\n");

	fp32 accel[3];
	signal(SIGINT, signal_handler); // 使用 ctrl+C 终止

	BMI088_init();

	// BMI088_accel_self_test();
	int count = 7000;
	// int count = 1000;
	printf("**********  Start  **********\n");
	printf("==> Current: %s\n", filePath);

	clock_t start_time = clock();

	while (count--)
	{

		BMI088_read(accel);
		clock_t time = clock();

		// printf("%f %f %f\n", accel[0], accel[1], accel[2]);
		fprintf(file, "%f,%f,%f,%f\n", accel[0], accel[1], accel[2], ((double)(time - start_time)) / CLOCKS_PER_SEC);

		for (int i = 0; i < 140000; i++)
			;
	}

	clock_t end_time = clock();
	double cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
	printf("==> Execution Time: %fs\n", cpu_time_used);

	printf("**********  End  **********\n");
	fclose(file);
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
	sprintf(filePath, "../output/%s.csv", timeString); // 假设存储在名为 "output" 的文件夹中

	FILE *outputFile = fopen("targetFileName.txt", "w");
	fprintf(outputFile, filePath);
	fclose(outputFile);
}