import smbus
import time
import threading
import sys
import numpy as np
import scipy.signal as signal
from sklearn.svm import SVC
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
from python_speech_features import mfcc
import scipy.signal as signal


import mylib_py as lib
import ctypes
import struct
import time
sampleNum = 16000  # 采样数

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = signal.butter(order, cutoff, fs=fs, btype='low', analog=False)
    y = signal.lfilter(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = signal.butter(order, cutoff,fs=fs, btype='high', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def filter_window(signal, fs):
    signal_filtered_L = butter_lowpass_filter(signal, cutoff=500, fs=fs, order=6)
    signal_filtered_H = butter_highpass_filter(signal_filtered_L, cutoff=20, fs=fs, order=5)
    return signal_filtered_H


def detect_peaks(accel_window, threshold=1000):
    peaks = signal.find_peaks(accel_window, height=threshold)[0]
    return peaks
def main_c():
    
    sensor_info = lib.SensorInfo()
    sensor_info_ptr = lib.pSensor(sensor_info)
    lib.initSensors(sensor_info_ptr)
    lib.setup(sensor_info_ptr)
    # print(f"==> sensor_info.i2cFile:{sensor_info.i2cFile}")

    # 设置文件名和路径
    # output_file_path = "./data/{}/Sensor{}_{}.txt".format(sensor_info.startTime.decode("utf-8"), sensor_info.sensorIndex, sensor_info.startTime.decode("utf-8"))

    # # 打开要写入的文件
    # file = open(output_file_path, "a")
    # if file is None:
    #     print("Failed to open output file for writing")
    #     exit(EXIT_FAILURE)

    # 获取样本数量
    sample_num = sampleNum

    total_byte_num = sample_num * 6
    data = (ctypes.c_ubyte * (total_byte_num))()  # 使用 ctypes 创建缓冲区
    char_pointer = 0
    
    st = time.time()
    while total_byte_num > 0:
        if lib.readRegOneByte(sensor_info.i2cFile, STATUS) & 1 == 1:
            lib.initializeByteStreaming(sensor_info_ptr, OUT_X_L, BUFFER_SIZE)
            for i in range(BUFFER_SIZE):
                data[char_pointer] = ctypes.c_ubyte(sensor_info.msgBuffer[i])  # 将读取到的数据写入到 data 缓冲区
                char_pointer += 1
            total_byte_num -= BUFFER_SIZE
    et = time.time()
    print(f"Time: {et-st}")
    print("\nSensor {} completed!\n".format(sensor_info.sensorIndex))
    
    
    csv_filename = "SensorData.txt"
    with open(csv_filename, mode='w') as file:
        # 向文件中写入数据
        for i in range(len(data)):
            hex_value = format(data[i], '02X')
            file.write(f"{hex_value}")

    print("Data has been saved to", csv_filename)

    
def main():

    buf_size = 20000
    window_size = 1000 # 这个就是before_length???
    forward = 0.02
    backward = 0.8
    fs = 1330
    before_length = int(forward*fs)
    after_length = int(backward*fs)
    print(f"before_length & after_length: {before_length} and {after_length}")
    svm_model = joblib.load('../../model/model.pkl') 
    buf = np.zeros((2,buf_size))
    
    
    sensor_info = lib.SensorInfo()
    sensor_info_ptr = lib.pSensor(sensor_info)
    lib.initSensors(sensor_info_ptr)
    lib.setup(sensor_info_ptr)
 
    p = 0 # pointer to the last collected data
    print(f"=> Start !!")
    start_time = time.time()
    while True and p < buf_size:
        if p >= window_size:
            
            accel_z = lib.read_dataZ(sensor_info_ptr)
            buf[0,p] = accel_z
            buf[1,p] = time.time()-start_time
            p += 1
            filtered_window = filter_window(buf[0,p - window_size:p], fs) # 这个window_size改成before_length试试看
            peaks = detect_peaks(filtered_window)

            if peaks.size > 0:

                index = p - window_size + peaks[0]
                for j in range(after_length):
                    accel_z = lib.read_dataZ(sensor_info_ptr)
                    buf[0,j+p] = accel_z
                    buf[1,j+p] = time.time()-start_time

                start_index = index - before_length
                end_index = index + after_length
                target_data = buf[:,start_index:end_index]
                # print(f"filling: {1/np.mean(np.diff(buf[1,:start_index]))}")
                # print(f"steady: {1/np.mean(np.diff(buf[1,start_index:index]))}")
                # print(f"after_peak: {1/np.mean(np.diff(buf[1,index:end_index]))}")

                feat = mfcc(target_data[0], fs,numcep=10,nfilt=11,nfft=256).reshape(1, -1)
                label = svm_model.predict(feat)[0]
                confidence = svm_model.predict_proba(feat)[0]
                print(f"Label: {label}, Confidence: {confidence[label]}")

                # plot_data(target_data)

                # reset buf
                p = 0
                buf = np.zeros((2,buf_size))
        else:
            accel_z = lib.read_dataZ(sensor_info_ptr)

            buf[0,p] = accel_z
            buf[1,p] = time.time()-start_time

            p += 1
    print("Completed!(Buffer exceeded)")
    plt.plot(buf[1,:p],buf[0,:p])
    plt.show()

if __name__ == "__main__":
    
    main()
    # main_c()
    # test()
