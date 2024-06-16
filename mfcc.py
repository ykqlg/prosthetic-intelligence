import numpy as np
from python_speech_features import mfcc,delta
import scipy.signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA 
from sklearn.manifold import TSNE
from dataset import butter_highpass_filter,butter_lowpass_filter
from scipy.signal import stft,cwt, morlet
import pywt

def DFT321f(u):
    # u: n x 3 array
    # n: the length of the signal
    # Each column stores acceleration data of x, y, and z axes, respectively
    
    # Get the dimensions of the data buffer
    N_sampl, N_chann = u.shape
    
    # Check whether the signal length is even or odd
    Odd = N_sampl % 2
    
    # Declare the transformed signal
    U_temp = np.fft.fft(u, axis=0)
    
    # The place of the Nyquist component depends on the data buffer being odd or even
    if Odd:
        N_half = (N_sampl + 1) // 2
    else:
        N_half = N_sampl // 2 + 1
    
    # Take the real part of the signal for capturing spectral intensity
    absY = np.real(np.sqrt(np.sum(U_temp[:N_half, :] * np.conj(U_temp[:N_half, :]), axis=1)))
    
    # Take the sum of three angles as the integrated angle of each frequency component
    PhaseChoice = np.angle(np.sum(U_temp[:N_half, :], axis=1))
    
    # Calculate the integrated spectrum
    Y_temp = absY * np.exp(1j * PhaseChoice)
    
    if Odd:
        Y = np.concatenate((Y_temp, np.conj(Y_temp[::-1][1:])))
    else:
        Y = np.concatenate((Y_temp, np.conj(Y_temp[::-1])))
    
    # Reconstruct a temporal signal from the integrated spectrum
    y = np.real(np.fft.ifft(Y))
    
    return y

def DFT321_smoothed(signal):
    # u: n x 3 array
    # n: the length of the signal
    # Each column stores acceleration data of x, y, and z axes, respectively
    fs=1330
    u_T = signal.T
    ACC_filtered_L = butter_lowpass_filter(u_T, cutoff=500, fs=fs, order=6)
    ACC_filtered_H = butter_highpass_filter(ACC_filtered_L, cutoff=20, fs=fs, order=5)
    
    u = ACC_filtered_H.T
    # Get the dimensions of the data buffer
    N_sampl, N_chann = u.shape
    
    # Check whether the signal length is even or odd
    Odd = N_sampl % 2
    
    # Declare the transformed signal
    U_temp = np.fft.fft(u, axis=0)
    
    # The place of the Nyquist component depends on the data buffer being odd or even
    if Odd:
        N_half = (N_sampl + 1) // 2
    else:
        N_half = N_sampl // 2 + 1
    
    # Take the real part of the signal for capturing spectral intensity
    absY = np.real(np.sqrt(np.sum(U_temp[:N_half, :] * np.conj(U_temp[:N_half, :]), axis=1)))
    
    # Take the sum of three angles as the integrated angle of each frequency component
    PhaseChoice = np.angle(np.sum(U_temp[:N_half, :], axis=1))
    
    # Calculate the integrated spectrum
    Y_temp = absY * np.exp(1j * PhaseChoice)
    
    if Odd:
        Y = np.concatenate((Y_temp, np.conj(Y_temp[::-1][1:])))
    else:
        Y = np.concatenate((Y_temp, np.conj(Y_temp[::-1])))
    
    # Reconstruct a temporal signal from the integrated spectrum
    y = np.real(np.fft.ifft(Y))
    
    return y

def calc_stft(signal, sample_rate=16000, frame_size=0.025, frame_stride=0.01, winfunc=np.hamming, NFFT=64):

    # Calculate the number of frames from the signal
    frame_length = frame_size * sample_rate
    frame_step = frame_stride * sample_rate
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = 1 + int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    # zero padding
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    # Pad signal to make sure that all frames have equal number of samples 
    # without truncating any samples from the original signal
    pad_signal = np.append(signal, z)

    # Slice the signal into frames from indices
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
            np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    # Get windowed frames
    frames *= winfunc(frame_length)
    # Compute the one-dimensional n-point discrete Fourier Transform(DFT) of
    # a real-valued array by means of an efficient algorithm called Fast Fourier Transform (FFT)
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    # Compute power spectrum
    pow_frames = (1.0 / NFFT) * ((mag_frames) ** 2)

    return pow_frames

class MyMFCC:
    def __init__(self, args) -> None:
        self.args = args
        # self.winfunc = np.hamming
        # self.winfunc = np.hanning
        self.winfunc = lambda x: np.ones((x,))
        self.stft_winfunc = self.winfunc
        # self.stft_winfunc = np.hanning

    def get_feat(self, dataSet):
        return self.get_z_mfcc_feat(dataSet)
        # return self.get_z_stft_feat(dataSet)
        # return self.get_dft321_stft_feat(dataSet)
    
    def single_axis_mfcc_feat(self,dataSet,axis):
        feat_list = []
        # print(f"original signal's dimension: {dataSet[:,2].shape}")
        

        for i in range(dataSet.shape[0]):

            signal = dataSet[i, axis]
            mfcc_feat = mfcc(signal, self.args.fs, winstep=self.args.winstep, numcep=self.args.numcep,
                                nfilt=self.args.nfilt, nfft=self.args.nfft, ceplifter=self.args.ceplifter,winfunc=self.winfunc).reshape(-1)

            feat_list.append(mfcc_feat)

        feat = np.array(feat_list)
        return feat
    
    def get_x_mfcc_feat(self, dataSet):
        return self.single_axis_mfcc_feat(dataSet,0)

    def get_y_mfcc_feat(self, dataSet):
        return self.single_axis_mfcc_feat(dataSet,1)

    def get_z_mfcc_feat(self, dataSet):
        return self.single_axis_mfcc_feat(dataSet,2)

    def single_axis_stft_feat(self,dataSet,axis):
        feat_list = []

        for i in range(dataSet.shape[0]):

            signal = dataSet[i, axis]
            # _, _, stft_feat = stft(signal, fs=self.args.fs,window=self.stft_winfunc)
            # stft_feat_flat = np.abs(stft_feat).reshape(-1)
            stft_feat = calc_stft(signal, sample_rate=self.args.fs,winfunc=self.stft_winfunc,NFFT=self.args.nfft)
            stft_feat_flat = stft_feat.reshape(-1)
            
            # print(stft_feat_flat.shape)
            feat_list.append(stft_feat_flat)
            

        feat = np.array(feat_list)
        return feat
    
    def get_x_stft_feat(self, dataSet):
        return self.single_axis_stft_feat(dataSet,0)

    def get_y_stft_feat(self, dataSet):
        return self.single_axis_stft_feat(dataSet,1)

    def get_z_stft_feat(self, dataSet):
        return self.single_axis_stft_feat(dataSet,2)

    def get_add_mfcc_feat(self, dataSet):
        feat_list = []

        for i in range(dataSet.shape[0]):

            signal = abs(dataSet[i, 0]) + abs(dataSet[i, 1]) + abs(dataSet[i, 2])
            mfcc_feat = mfcc(signal, self.args.fs, winstep=self.args.winstep, numcep=self.args.numcep,
                                nfilt=self.args.nfilt, nfft=self.args.nfft, ceplifter=self.args.ceplifter).reshape(-1)

            feat_list.append(mfcc_feat)

        feat = np.array(feat_list)
        return feat
    
    def get_add_stft_feat(self, dataSet):
        feat_list = []

        for i in range(dataSet.shape[0]):

            signal = abs(dataSet[i, 0]) + abs(dataSet[i, 1]) + abs(dataSet[i, 2])
            # _, _, stft_feat = stft(signal, fs=self.args.fs,window=self.stft_winfunc)
            # stft_feat_flat = np.abs(stft_feat).reshape(-1)
            stft_feat = calc_stft(signal, sample_rate=self.args.fs,winfunc=self.stft_winfunc,NFFT=self.args.nfft)
            stft_feat_flat = stft_feat.reshape(-1)
            feat_list.append(stft_feat_flat)

        feat = np.array(feat_list)
        return feat
    
    def get_dft321_mfcc_feat(self, dataSet):
        feat_list = []

        for i in range(dataSet.shape[0]):
            signal = dataSet[i, :]
            signal = DFT321f(signal.T)
            # signal = DFT321_smoothed(signal.T)
            
            
            mfcc_feat = mfcc(signal, self.args.fs, winstep=self.args.winstep, numcep=self.args.numcep,
                                nfilt=self.args.nfilt, nfft=self.args.nfft, ceplifter=self.args.ceplifter,winfunc=self.winfunc).reshape(-1)

            feat_list.append(mfcc_feat)

        feat = np.array(feat_list)
        # print(f"{feat.shape}")
        return feat

    def get_dft321_stft_feat(self, dataSet):
        feat_list = []

        for i in range(dataSet.shape[0]):
            signal = dataSet[i, :]
            # signal = DFT321f(signal.T)
            signal = DFT321_smoothed(signal.T)
            # print(f"dft321 result: {signal.shape}")
            # _, _, stft_feat = stft(signal, fs=self.args.fs,window=self.stft_winfunc)
            # # 将 STFT 结果转换为一维数组
            # stft_feat_flat = np.abs(stft_feat).reshape(-1)
            stft_feat = calc_stft(signal, sample_rate=self.args.fs,winfunc=self.stft_winfunc,NFFT=self.args.nfft)
            stft_feat_flat = stft_feat.reshape(-1)
            feat_list.append(stft_feat_flat)

        feat = np.array(feat_list)
        return feat
    
    def get_wavelet_feat(self,dataSet):
        feat_list = []
        scales = np.arange(1, 3)
        # print(f"original signal's dimension: {dataSet[:,2].shape}")
        
        for i in range(dataSet.shape[0]):
            signal = dataSet[i, 2]
            # 执行DWT
            coeffs = pywt.wavedec(signal, 'haar', level=7)
            approx_coeffs = coeffs[0]
            detail_coeffs = coeffs[1:]
            # print(detail_coeffs.shape)
            # a= np.array(detail_coeffs)
            # print(a.shape)
            
            
            # coefficients = cwt(signal, morlet, scales)
            coefficients_flat = np.concatenate(coeffs).ravel()
            # coefficients_flat = np.abs(detail_coeffs).ravel()
            
            feat_list.append(coefficients_flat)
        feat = np.array(feat_list)
        # print(f"dimension: {feat.shape}")
        
        return feat
    
    def get_dft321_wavelet_feat(self,dataSet):
        feat_list = []
        scales = np.arange(1, 2)
        # print(f"original signal's dimension: {dataSet[:,2].shape}")
        
        for i in range(dataSet.shape[0]):
            # signal = dataSet[i, 2]
            signal = dataSet[i, :]
            signal = DFT321f(signal.T)
            # signal = DFT321_smoothed(signal.T)
            # coefficients = cwt(signal, morlet, scales)
            coeffs = pywt.wavedec(signal, 'haar', level=7)
            approx_coeffs = coeffs[0]
            detail_coeffs = coeffs[1:]
            # coefficients_flat = np.abs(coefficients).reshape(-1)
            coefficients_flat = np.concatenate(coeffs).ravel()
            
            feat_list.append(coefficients_flat)
        feat = np.array(feat_list)
        # print(f"dimension: {feat.shape[1]}")
        
        return feat

    def get_concat_mfcc_feat(self, dataSet):
        dynamic = self.args.dynamic
        feat_list = []

        for i in range(dataSet.shape[0]):
            mfcc_feats = []

            for axis in range(3):
                signal = dataSet[i, axis]
                mfcc_feat = mfcc(signal, self.args.fs, winstep=self.args.winstep, winlen=self.args.winlen ,numcep=self.args.numcep,nfilt=self.args.nfilt, nfft=self.args.nfft, ceplifter=self.args.ceplifter,winfunc=self.winfunc).reshape(-1)
                mfcc_feats.append(mfcc_feat)

            # 将 single_axis 轴的 MFCC 特征连接在一起
            concatenated_feats = np.concatenate(mfcc_feats)
            feat_list.append(concatenated_feats)

        feat = np.array(feat_list)
        # num_features = feat.shape[1]

        num_features = int(feat.shape[1]/3)
        feat_dy = None
        if dynamic:
            # 使用dynamic，特征维度会变成原来的3倍
            feat_dynamic = []
            for start_idx in range(0, feat.shape[1], num_features):
                end_idx = start_idx + num_features
                axis_features = feat[:, start_idx:end_idx]
                delta_features = delta(axis_features, 2)
                delta_delta_features = delta(delta_features, 2)
                axis_features = np.hstack(
                    (axis_features, delta_features, delta_delta_features))
                feat_dynamic.append(axis_features)

            feat_dy = np.concatenate(feat_dynamic, axis=1)

        if dynamic:
            return feat_dy
        else:
            return feat
    
    def get_concat_stft_feat(self, dataSet):
        dynamic = self.args.dynamic
        feat_list = []

        for i in range(dataSet.shape[0]):
            mfcc_feats = []

            for axis in range(3):
                signal = dataSet[i, axis]
                # _, _, stft_feat = stft(signal, fs=self.args.fs,window=self.stft_winfunc)
                # # 将 STFT 结果转换为一维数组
                # stft_feat_flat = np.abs(stft_feat).reshape(-1)
                stft_feat = calc_stft(signal, sample_rate=self.args.fs,winfunc=self.stft_winfunc,NFFT=self.args.nfft)
                stft_feat_flat = stft_feat.reshape(-1)
                mfcc_feats.append(stft_feat_flat)

            # 将 single_axis 轴的 MFCC 特征连接在一起
            concatenated_feats = np.concatenate(mfcc_feats)
            feat_list.append(concatenated_feats)

        feat = np.array(feat_list)
        # num_features = feat.shape[1]

        num_features = int(feat.shape[1]/3)
        feat_dy = None
        if dynamic:
            # 使用dynamic，特征维度会变成原来的3倍
            feat_dynamic = []
            for start_idx in range(0, feat.shape[1], num_features):
                end_idx = start_idx + num_features
                axis_features = feat[:, start_idx:end_idx]
                delta_features = delta(axis_features, 2)
                delta_delta_features = delta(delta_features, 2)
                axis_features = np.hstack(
                    (axis_features, delta_features, delta_delta_features))
                feat_dynamic.append(axis_features)

            feat_dy = np.concatenate(feat_dynamic, axis=1)

        if dynamic:
            return feat_dy
        else:
            return feat