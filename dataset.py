import numpy as np
import pandas as pd
import scipy.signal as signal

from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
import random

def butter_lowpass_filter(data, cutoff, fs, order=5):
    
    b, a = signal.butter(order, cutoff, fs=fs, btype='low', analog=False)
    y = signal.lfilter(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = signal.butter(order, cutoff,fs=fs, btype='high', analog=False)
    y = signal.filtfilt(b, a, data)
    return y


class MyDataset:
    def __init__(self, args):
        self.args = args

        train_df = pd.read_csv(args.train_label_file)
        if args.train_label_file2:
            train_df2 = pd.read_csv(args.train_label_file2)
            combined_df = pd.concat([train_df, train_df2], ignore_index=True)
            train_df = combined_df.sample(frac=1,random_state=args.random_state).reset_index(drop=True)  # 对合并后的数据集进行 shuffle
        filePaths = train_df['FilePath'].values
        y = train_df['Label'].values


        X = self.pre_process(filePaths)
        # print(train_df.columns)

        if args.test_only:
            test_df = pd.read_csv(args.test_label_file)
            filePaths = test_df['FilePath'].values

            y_test =test_df['Label'].values

            X_test = self.pre_process(filePaths)
        
            self.X_train = X
            self.y_train = y
            self.X_test = X_test
            self.y_test = y_test

        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, random_state=args.random_state)

            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            self.X = X
            self.y = y

    def get_rate(self):
        return self.args.fs

    def get_data(self):
        return self.X, self.y
    def get_train_data(self):
        return self.X_train, self.y_train
    def get_test_data(self):
        return self.X_test, self.y_test
    
    def pre_process(self,filePaths):
        data_list = []
        fs = self.args.fs
        for file_path in filePaths:
            df = pd.read_csv(file_path)
            ACC = df['ACC_Z']
            ACC_filtered_L = butter_lowpass_filter(ACC, cutoff=500, fs=fs, order=6)
            ACC_filtered_H = butter_highpass_filter(ACC_filtered_L, cutoff=20, fs=fs, order=5)
            threshold = 1000
            # threshold = 700
            # peaks, _ = find_peaks(ACC, height=threshold)
            peaks, _ = find_peaks(ACC_filtered_H, height=threshold)
            # if (len(peaks)) == 0:
            #     print(file_path)
            pivot = peaks[0]
            start_index = pivot - int(self.args.forward*fs)
            end_index = pivot + int(self.args.backward*fs)

            data_df = df[start_index:end_index + 1]

            sig_X = data_df['ACC_X'].values
            sig_Y = data_df['ACC_Y'].values
            sig_Z = data_df['ACC_Z'].values
            # rate = int(1/np.mean(np.diff(t)))
            data = [sig_X, sig_Y, sig_Z]
            data_list.append(data)

        X = np.array(data_list)
        return X

class MultiDataset:
    def __init__(self, args):
        self.args = args
        self.train_X = []
        self.train_y = []
        self.test_X = []
        self.test_y = []

        self.load_datasets()

    def load_datasets(self):

        X,y = self.get_single_train_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.args.test_size, random_state=self.args.random_state)
        self.train_X.append(X_train)
        self.train_y.append(y_train)
        self.test_X.append(X_test)
        self.test_y.append(y_test)

        for file in self.args.tests_label_file:
            # check_dataset(file)
            df = pd.read_csv(file)
            filePaths = df['FilePath'].values
            y = df['Label'].values
            X = self.pre_process(filePaths)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.args.test_size, random_state=self.args.random_state)
            self.train_X.append(X_train)
            self.train_y.append(y_train)
            self.test_X.append(X_test)
            self.test_y.append(y_test)

        # 合并所有训练集和测试集
        self.train_X = np.concatenate(self.train_X)
        self.train_y = np.concatenate(self.train_y)
        self.test_X = np.concatenate(self.test_X)
        self.test_y = np.concatenate(self.test_y)
        # self.train_X = pd.concat(self.train_X)
        # self.train_y = pd.concat(self.train_y)
        # self.test_X = pd.concat(self.test_X)
        # self.test_y = pd.concat(self.test_y)

    def get_single_train_data(self):
        # 原始纸杯数据作为训练集
        # check_dataset(self.args.train_label_file)
        df = pd.read_csv(self.args.train_label_file)

        filePaths = df['FilePath'].values
        y = df['Label'].values
        X= self.pre_process(filePaths)
        return X, y
    
    def get_single_val_data(self):
        # 卷纸数据作为验证集
        # check_dataset(self.args.val_label_file)

        df = pd.read_csv(self.args.val_label_file)
        filePaths = df['FilePath'].values
        y = df['Label'].values
        X= self.pre_process(filePaths)

        return X, y
    
    def get_train_data(self):
        return self.train_X, self.train_y

    def get_test_data(self):
        return self.test_X, self.test_y
    
    def pre_process(self,filePaths):
        data_list = []
        fs = self.args.fs
        for file_path in filePaths:
            df = pd.read_csv(file_path)
            ACC = df['ACC_Z']
            ACC_filtered_L = butter_lowpass_filter(ACC, cutoff=500, fs=fs, order=6)
            ACC_filtered_H = butter_highpass_filter(ACC_filtered_L, cutoff=20, fs=fs, order=5)
            threshold = 1000
            # threshold = 700
            # peaks, _ = find_peaks(ACC, height=threshold)
            peaks, _ = find_peaks(ACC_filtered_H, height=threshold)
            # if (len(peaks)) == 0:
            #     print(file_path)
            pivot = peaks[0]
            start_index = pivot - int(self.args.forward*fs)
            end_index = pivot + int(self.args.backward*fs)

            data_df = df[start_index:end_index + 1]

            sig_X = data_df['ACC_X'].values
            sig_Y = data_df['ACC_Y'].values
            sig_Z = data_df['ACC_Z'].values
            # rate = int(1/np.mean(np.diff(t)))
            data = [sig_X, sig_Y, sig_Z]
            data_list.append(data)

        X = np.array(data_list)
        return X
    
def check_dataset(label_name):
    df = pd.read_csv(label_name)
    y = df['Label'].values

    result = np.unique(y,return_counts=True)
    print(f"{label_name}: {result}")