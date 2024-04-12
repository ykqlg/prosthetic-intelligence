import scipy.signal as signal
import numpy as np

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = signal.butter(order, cutoff, fs=fs, btype='low', analog=False)
    y = signal.lfilter(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = signal.butter(order, cutoff,fs=fs, btype='high', analog=False)
    y = signal.filtfilt(b, a, data)
    return 

def filter_window(signal, fs):
    signal_filtered_L = butter_lowpass_filter(signal, cutoff=500, fs=fs, order=6)
    signal_filtered_H = butter_highpass_filter(signal_filtered_L, cutoff=20, fs=fs, order=5)
    # return np.array(signal_filtered_H)
    return signal_filtered_H

def detect_peaks(accel_window, threshold=1000):
    peaks = signal.find_peaks(accel_window, height=threshold)[0]
    return peaks