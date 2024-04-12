
from python_speech_features import mfcc
import smbus
import time
import threading
import sys
import numpy as np
from sklearn.svm import SVC
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
from python_speech_features import mfcc
from util import filter_window,detect_peaks
import mylib as lib
import ctypes
import struct
import time

STATUS = 39
SAMPLE_NUM = 16000  # 采样数
Fs = 1600
t = np.arange(SAMPLE_NUM)/ Fs


def collection():
    sensor_info = lib.SensorInfo()
    sensor_info_ptr = lib.pSensor(sensor_info)
    lib.initSensors(sensor_info_ptr)
    lib.setup(sensor_info_ptr)
    sample_num = SAMPLE_NUM
    p = 0
    data = (ctypes.c_float * 3)()
    result = np.empty((sample_num,3))
    
    st = time.time()
    
    while p < sample_num:
        if lib.readRegOneByte(sensor_info.i2cFile, STATUS) & 1 == 1:
            lib.read_data(sensor_info_ptr,data)
            result[p,:] = np.ctypeslib.as_array(data)
            p += 1
        
    et = time.time()
    print(f"Time: {et-st}")
    start_time = (sensor_info.startTime).decode('utf-8')
    # print(start_time)
    sensor_index = str(sensor_info.sensorIndex)
    print(f"\nSensor {sensor_index} completed!\n")
    plot_data(result)
    
    
    dir_path = f"./data/{start_time}/"
    csv_filename = f"Sensor{sensor_index}_{start_time}.csv"
    file_path=  dir_path + csv_filename
    np.savetxt(file_path, result, delimiter=',', fmt='%f')

    # print("Data has been saved to", csv_filename)

def plot_data(accData):
    t = np.arange(accData.shape[0]) / Fs
    fig1, ax = plt.subplots(3, 1, dpi=144, figsize=(16, 6))
    ax[0].plot(t, accData[:, 0])
    ax[1].plot(t, accData[:, 1])
    ax[2].plot(t, accData[:, 2])
    # 添加标题
    ax[0].set_ylabel("ACC_X (mg)")
    ax[1].set_ylabel("ACC_Y (mg)")
    ax[2].set_ylabel("ACC_Z (mg)")
    ax[2].set_xlabel("Time (s)")
    plt.show()

def real_time():
    if sys.gettrace() is not None:
        print("Debug mode is enabled.")
        filling = 205
        steady = 115
        after_peak = 205
    else:
        print("Debug mode is disabled.")
        filling = 25500
        steady = 21500
        after_peak = 27000

  
    buf_size = 100000
    window_size = 1000 # 这个就是before_length???
    forward = 0.45
    backward = 0.75
    fs = 1330
    before_length = int(forward*fs)
    after_length = int(backward*fs)
    print(f"before_length & after_length: {before_length} and {after_length}")
    svm_model = joblib.load('../model/model.pkl') 
    # buf = np.zeros((2,buf_size))
    buf = np.zeros(buf_size)
    
    sample_num = SAMPLE_NUM
    data = (ctypes.c_float * 3)()
    sensor_info = lib.SensorInfo()
    sensor_info_ptr = lib.pSensor(sensor_info)
    lib.initSensors(sensor_info_ptr)
    lib.setup(sensor_info_ptr)
 
    p = 0 # pointer to the last collected data
    print(f"=> Start !!")
    while p < sample_num:
        if p >= window_size:
            
            if lib.readRegOneByte(sensor_info.i2cFile, STATUS) & 1 == 1:
                lib.read_data(sensor_info_ptr,data)
                buf[p] = data[2]

                filtered_window = filter_window(buf[(p-window_size):p], Fs)
                print(f"b{filtered_window.shape}")
                filtered_window_1d = filtered_window.reshape(-1)
                print(f"h{filtered_window_1d.shape}")
                peaks = detect_peaks(filtered_window_1d,1000)
                if peaks.size > 0:

                    index = p - window_size + peaks[0]
                    for j in range(after_length):
                        lib.read_data(sensor_info_ptr,data)
                        buf[j+p] = data[2]

                    start_index = index - before_length
                    end_index = index + after_length
                    target_data = buf[start_index:end_index]
                    # print(f"filling: {1/np.mean(np.diff(buf[1,:start_index]))}")
                    # print(f"steady: {1/np.mean(np.diff(buf[1,start_index:index]))}")
                    # print(f"after_peak: {1/np.mean(np.diff(buf[1,index:end_index]))}")

                    feat = mfcc(target_data[0], 1330, winstep=0.01).reshape(1, -1)
                    label = svm_model.predict(feat)[0]
                    confidence = svm_model.predict_proba(feat)[0]
                    print(f"Label: {label}, Confidence: {confidence[label]}")

                    plot_data(target_data)

                    # reset buf
                    p = 0
                    buf = np.zeros((2,buf_size))

                p+=1
            
        else:
            if lib.readRegOneByte(sensor_info.i2cFile, STATUS) & 1 == 1:
                lib.read_data(sensor_info_ptr,data)
                buf[p] = data[2]

                p+=1

    plt.plot(t,buf)
    plt.show()

def plot_rt_data(data):
    # df = pd.DataFrame(data.T, columns=['ACC_Z','Time'])
    # df.to_csv("./output/success.csv",index=False)

    
    plt.plot(time,data/1000, label='ACC_Z Data')
    # plt.axvspan(time[start_index], time[end_index-1], facecolor='red', alpha=0.3, label='Target Waveform')
    plt.legend()

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    
    real_time()
    # collection()
    # test()
