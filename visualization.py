import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

import scipy.signal as signal
def get_latest_file(folder_path):
    # 使用 glob 模块获取文件夹中所有文件的路径列表
    files = glob.glob(os.path.join(folder_path, '*'))

    # 如果文件列表为空，返回 None
    if not files:
        return None

    # 获取文件的最后修改时间并排序
    latest_file = max(files, key=os.path.getctime)

    return latest_file

def plot_data(filename):
    
    df = pd.read_csv(filename)
    ACC_X = df['ACC_X']
    ACC_Y = df['ACC_Y']
    ACC_Z = df['ACC_Z']
    time = df['Time']
    
    _,ax = plt.subplots(1, 1, figsize=(20, 4)) 
    ax.hist(np.diff(time), 1000)
    
    fs = 1/np.mean(np.diff(time));  # sample rate, Hz
    # print("采样率fs = %d" % fs)
    
    
    # plt.specgram(ACC_Z, NFFT=1024, Fs=fs, detrend=None)
    
    # Filter requirements.
    order = 6
    cutoff = 500  # desired cutoff frequency of the filter, Hz

    
    # Filter the data, and plot both the original and filtered signals.
    ACC_X_filtered = butter_lowpass_filter(ACC_X, cutoff, fs, order)
    ACC_Y_filtered = butter_lowpass_filter(ACC_Y, cutoff, fs, order)
    ACC_Z_filtered = butter_lowpass_filter(ACC_Z, cutoff, fs, order)

    
    # 创建一个包含三个子图的画布
    fig, axes = plt.subplots(1, 3, figsize=(20, 4)) 
    irange = 5000
    ylim = False
    axes[0].set_title('ACC_X')
    axes[1].set_title('ACC_Y')
    axes[2].set_title('ACC_Z')
    for axe in axes:
        if ylim :axe.set_ylim(-irange, irange)
        axe.set_ylabel('mg')
        
        
    ACC_X_filtered = butter_highpass_filter(ACC_X_filtered, 20, fs, order=5)[1000:]
    ACC_Y_filtered = butter_highpass_filter(ACC_Y_filtered, 20, fs, order=5)[1000:]
    ACC_Z_filtered = butter_highpass_filter(ACC_Z_filtered, 20, fs, order=5)[1000:]
    time = time[1000:]
    axes[0].plot(time,ACC_X_filtered)
    axes[1].plot(time,ACC_Y_filtered)
    axes[2].plot(time,ACC_Z_filtered)


    plt.suptitle(filename)
    plt.tight_layout()
    plt.savefig("visualization.png")
    # plt.show()
    
def butter_lowpass(cutoff, fs, order=5):
    return signal.butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

if __name__ == "__main__":
   
    folder_path = './output'
    latest_file = get_latest_file(folder_path)

    # print('latest_file',latest_file)
    plot_data(latest_file)
