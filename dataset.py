import numpy as np
import pandas as pd
import scipy.signal as signal

from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks

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

        if args.test_only:
            train_df = pd.read_csv(args.train_label_file)
            test_df = pd.read_csv(args.label_file_path)

            filePaths = train_df['FilePath'].values
            data_list = []
            fs = args.fs
            for file_path in filePaths:
                df = pd.read_csv(file_path)
                ACC = df['ACC_Z']
                ACC_filtered_L = butter_lowpass_filter(
                    ACC, cutoff=500, fs=fs, order=6)
                ACC_filtered_H = butter_highpass_filter(
                    ACC_filtered_L, cutoff=20, fs=fs, order=5)
                threshold = 1000
                # threshold = 700
                # peaks, _ = find_peaks(ACC, height=threshold)
                peaks, _ = find_peaks(ACC_filtered_H, height=threshold)
                pivot = peaks[0]
                start_index = pivot - int(args.forward*fs)
                end_index = pivot + int(args.backward*fs)

                data_df = df[start_index:end_index + 1]

                sig_X = data_df['ACC_X'].values
                sig_Y = data_df['ACC_Y'].values
                sig_Z = data_df['ACC_Z'].values
                # rate = int(1/np.mean(np.diff(t)))
                data = [sig_X, sig_Y, sig_Z]
                data_list.append(data)

            X_train = np.array(data_list)
            y_train = train_df['Label'].values

            filePaths = test_df['FilePath'].values
            data_list = []
            fs = args.fs
            for file_path in filePaths:
                df = pd.read_csv(file_path)
                ACC = df['ACC_Z']
                ACC_filtered_L = butter_lowpass_filter(
                    ACC, cutoff=500, fs=fs, order=6)
                ACC_filtered_H = butter_highpass_filter(
                    ACC_filtered_L, cutoff=20, fs=fs, order=5)
                threshold = 1000
                # threshold = 700
                # peaks, _ = find_peaks(ACC, height=threshold)
                peaks, _ = find_peaks(ACC_filtered_H, height=threshold)
                pivot = peaks[0]
                start_index = pivot - int(args.forward*fs)
                end_index = pivot + int(args.backward*fs)

                data_df = df[start_index:end_index + 1]

                sig_X = data_df['ACC_X'].values
                sig_Y = data_df['ACC_Y'].values
                sig_Z = data_df['ACC_Z'].values
                # rate = int(1/np.mean(np.diff(t)))
                data = [sig_X, sig_Y, sig_Z]
                data_list.append(data)

            X_test = np.array(data_list)
            y_test = test_df['Label'].values

            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test

        else:
            labels_df = pd.read_csv(args.label_file_path)
            filePaths = labels_df['FilePath'].values
            data_list = []
            fs = args.fs
            for file_path in filePaths:
                df = pd.read_csv(file_path)
                ACC = df['ACC_Z']
                ACC_filtered_L = butter_lowpass_filter(
                    ACC, cutoff=500, fs=fs, order=6)
                ACC_filtered_H = butter_highpass_filter(
                    ACC_filtered_L, cutoff=20, fs=fs, order=5)
                threshold = 1000
                # threshold = 700
                # peaks, _ = find_peaks(ACC, height=threshold)
                peaks, _ = find_peaks(ACC_filtered_H, height=threshold)
                pivot = peaks[0]
                start_index = pivot - int(args.forward*fs)
                end_index = pivot + int(args.backward*fs)

                data_df = df[start_index:end_index + 1]

                sig_X = data_df['ACC_X'].values
                sig_Y = data_df['ACC_Y'].values
                sig_Z = data_df['ACC_Z'].values
                # rate = int(1/np.mean(np.diff(t)))
                data = [sig_X, sig_Y, sig_Z]
                data_list.append(data)

            X = np.array(data_list)
            y = labels_df['Label'].values

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