import time
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from driver import accel_read,butter_highpass_filter,butter_lowpass_filter,SpiManager,self_test



def plot_data(data_list):

    df = pd.DataFrame(data_list, columns=['X', 'Y','Z','Time'])

    ACC_X = df.iloc[:, 0]
    ACC_Y = df.iloc[:, 1]
    ACC_Z = df.iloc[:, 2]
    time  = df.iloc[:, 3]
    fs = 1330
    # fs = 1/np.mean(np.diff(time))

    # Filter requirements.
    order = 6
    cutoff = 500  # desired cutoff frequency of the filter, Hz
    truncate_length = 0
    
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
    
        
    ACC_X_filtered = butter_highpass_filter(ACC_X_filtered, 20, fs, order=5)[truncate_length:]
    ACC_Y_filtered = butter_highpass_filter(ACC_Y_filtered, 20, fs, order=5)[truncate_length:]
    ACC_Z_filtered = butter_highpass_filter(ACC_Z_filtered, 20, fs, order=5)[truncate_length:]
    # time = time[truncate_length:]
    axes[0].plot(time,ACC_X_filtered)
    axes[1].plot(time,ACC_Y_filtered)
    axes[2].plot(time,ACC_Z_filtered)


    plt.suptitle('visualization_py')
    plt.tight_layout()
    plt.savefig("visualization_py.png")
    plt.show()
def plot_data2(data_list):

    df = pd.DataFrame(data_list, columns=['X', 'Y','Z','Time'])

    ACC_X = df.iloc[:, 0]
    ACC_Y = df.iloc[:, 1]
    ACC_Z = df.iloc[:, 2]
    time  = df.iloc[:, 3]
    fs = 1330
    # fs = 1/np.mean(np.diff(time))


    
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
    
  
    # time = time[truncate_length:]
    axes[0].plot(time,ACC_X)
    axes[1].plot(time,ACC_Y)
    axes[2].plot(time,ACC_Z)


    plt.suptitle('without_filter_py')
    plt.tight_layout()
    plt.savefig("without_filter.png")
    plt.show()
    
    
    
    
    
def main():
    
    
    print("This is the main function.")
    data_list = []
    count = 7000
    print("**********  Start  **********\n");
    stime = time.time()

    for i in range(count):
        data = accel_read(spi)
        # print(data)
        ctime = time.time()
        data.append(ctime-stime)
        data_list.append(data)
        for j in range(2*count):
            pass
        
    # print(data_list)
    etime = time.time()
    print(f"执行时间：{etime-stime}")
    print("**********  End  **********\n");

    plot_data2(data_list)
    

    
    

if __name__ == '__main__':
    spi_manager = SpiManager()
    spi = spi_manager.get_spi()
    main()
    spi.close()    