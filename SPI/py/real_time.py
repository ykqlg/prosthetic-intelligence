import numpy as np
import scipy.signal as signal
from driver import butter_highpass_filter, butter_lowpass_filter,SpiManager
from sklearn.svm import SVC
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
from python_speech_features import mfcc

def detect_peaks(accel_window, threshold=1000):
    peaks = signal.find_peaks(accel_window, height=threshold)[0]
    return peaks

def filter_window(signal, fs):
    signal_filtered_L = butter_lowpass_filter(signal, cutoff=500, fs=fs, order=6)
    signal_filtered_H = butter_highpass_filter(signal_filtered_L, cutoff=20, fs=fs, order=5)
    return signal_filtered_H

def main():
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
    buf = np.zeros((2,buf_size))

    
    p = 0 # pointer to the last collected data
    print(f"=> Start !!")
    start_time = time.time()
    while True and p < buf_size:
        if p >= window_size:
            for i in range(steady):
                pass
            accel_z = spi.accel_z()
            buf[0,p] = accel_z
            buf[1,p] = time.time()-start_time

            p += 1
            if p%5 == 0:

                filtered_window = filter_window(buf[0,p - window_size:p], fs) # 这个window_size改成before_length试试看
                peaks = detect_peaks(filtered_window)
                if peaks.size > 0:

                    index = p - window_size + peaks[0]
                    for j in range(after_length):
                        for k in range(after_peak):
                            pass
                        accel_z = spi.accel_z()
                        buf[0,j+p] = accel_z
                        buf[1,j+p] = time.time()-start_time

                    start_index = index - before_length
                    end_index = index + after_length
                    target_data = buf[:,start_index:end_index]
                    # print(f"filling: {1/np.mean(np.diff(buf[1,:start_index]))}")
                    # print(f"steady: {1/np.mean(np.diff(buf[1,start_index:index]))}")
                    # print(f"after_peak: {1/np.mean(np.diff(buf[1,index:end_index]))}")

                    feat = mfcc(target_data[0], 1330, winstep=0.01).reshape(1, -1)
                    label = svm_model.predict(feat)[0]
                    confidence = svm_model.predict_proba(feat)[0]
                    print(f"Label: {label}, Confidence: {confidence[label]}")

                    # plot_data(target_data)

                    # reset buf
                    p = 0
                    buf = np.zeros((2,buf_size))
        else:
            for i in range(filling):
                pass
            accel_z = spi.accel_z()
            buf[0,p] = accel_z
            buf[1,p] = time.time()-start_time

            p += 1


def plot_data(data):
    ACC_Z = data[0]
    time = data[1]
    df = pd.DataFrame(data.T, columns=['ACC_Z','Time'])
    df.to_csv("./output/success.csv",index=False)

    
    plt.plot(time,ACC_Z/1000, label='ACC_Z Data')
    # plt.axvspan(time[start_index], time[end_index-1], facecolor='red', alpha=0.3, label='Target Waveform')
    plt.legend()

    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    spi = SpiManager()
    main()
    spi.close_spi()