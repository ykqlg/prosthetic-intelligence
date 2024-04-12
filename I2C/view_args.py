import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import re
import sys
def twoComplement16bit(hexData):
    return -(hexData & 0x8000) | (hexData & 0x7FFF)
# 单位转化
GSCALE = 0.001952  # 1.952 mg/digit (±16 g in High-Performance Mode)


# 构建正则表达式
pattern = re.compile(r"Sensor(\d)_(\d{8})-(\d{6})\.txt")

txtFile = sys.argv[1]
print("argv[1] = ",sys.argv[1])
sensor_index = pattern.match(txtFile).group(1)
date_part = pattern.match(txtFile).group(2)
time_part = pattern.match(txtFile).group(3)
    
print(f"File: {txtFile}")
print("Sensor:",sensor_index)
print("Date:", date_part)
print("Time:", time_part)
print("---")
with open(txtFile, "r", encoding="utf-8") as file:
    dataBytes = file.read()
    print(f"文件 '{txtFile}' 共有 {len(dataBytes)} 个字符.")
    Fs = 1600 # 采样率指定1600
    csvFile = "csvData%s_%s-%s.csv" % (sensor_index, date_part,time_part)  # 定义csv文件
    if not os.path.exists(csvFile):
        with open(csvFile, "w", newline="") as fd:
            writer = csv.writer(fd)
            for i in range(0, len(dataBytes), 12):
                dataX = (dataBytes[i + 2 : i + 4] + dataBytes[i : i + 2])  # 将X轴数据的MSB和LSB组合在一起
                dataY = (dataBytes[i + 6 : i + 8] + dataBytes[i + 4 : i + 6])  # 将Y轴数据的MSB和LSB组合在一起
                dataZ = (dataBytes[i + 10 : i + 12] + dataBytes[i + 8 : i + 10])  # 将Z轴数据的MSB和LSB组合在一起

                if len(dataX) > 0 and len(dataY) > 0 and len(dataZ) > 0:
                    dataX = twoComplement16bit(int(dataX, 16)) >> 2  # 转换X轴的二补码数据
                    dataY = twoComplement16bit(int(dataY, 16)) >> 2  # 转换Y轴的二补码数据
                    dataZ = twoComplement16bit(int(dataZ, 16)) >> 2  # 转换Z轴的二补码数据

                    writer.writerow([dataX, dataY, dataZ])
            print("csv数据保存成功")

    # 可视化数据
    accData = np.genfromtxt(csvFile, delimiter=",")
    t = np.arange(accData.shape[0]) / Fs
    fig1, ax = plt.subplots(3, 1, dpi=72, figsize=(16, 6))
    ax[0].plot(t, accData[:, 0] * GSCALE)
    ax[1].plot(t, accData[:, 1] * GSCALE)
    ax[2].plot(t, accData[:, 2] * GSCALE)
    # 添加标题
    ax[0].set_ylabel("Acceleration of X (g)")
    ax[1].set_ylabel("Acceleration of Y (g)")
    ax[2].set_ylabel("Acceleration of Z (g)")
    ax[2].set_xlabel("Time (s)")
    plt.show()
