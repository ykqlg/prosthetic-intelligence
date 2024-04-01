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

int main(void)
{
	fp32 accel[3];

	BMI088_init();

	int count = 7000;
	printf("**********  Start  **********\n");

	clock_t start_time = clock();

	while (count--)
	{

		BMI088_read(accel);
	}

	clock_t end_time = clock();
	double cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
	printf("==> Execution Time: %fs\n", cpu_time_used);

	printf("**********  End  **********\n");
	return 0;
}