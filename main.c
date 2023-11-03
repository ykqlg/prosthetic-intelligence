#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>

#include "struct_typedef.h"
#include "BMI088reg.h"
#include "BMI088driver.h"
#include "BMI088Middleware.h"

// SPI_HandleTypeDef* hspi_acc = NULL;
extern SPI_HandleTypeDef* hspi_acc;
FILE * file = NULL;

void signal_handler(int signum) {
	if (file != NULL) {
		fclose(file); // 关闭文件
	}
	exit(signum); // 退出程序
}


int main(void) {

	clock_t start_time, end_time;
	double cpu_time_used;
	// start_time = clock();
	fp32 accel[3];
	signal(SIGINT, signal_handler);

	BMI088_init();
	
	BMI088_accel_self_test();
	

	file = fopen("data.csv", "w");
	if (file == NULL) {
		fprintf(stderr, "Error opening file.\n");
		return 1;
	}

	fprintf(file, "ACC_X,ACC_Y,ACC_Z\n"); 

	int count = 10000;
	
	start_time = clock();

	while(count--){
		
		BMI088_read(accel);
		printf("%f %f %f\n",accel[0],accel[1],accel[2]);
		fprintf(file, "%f,%f,%f\n",accel[0],accel[1],accel[2]);
	}
	fclose(file);

	printf("Successful end!!!\n");



	end_time = clock();
	cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
	printf("程序执行时间为 %f 秒\n", cpu_time_used);


	return 0;
}




