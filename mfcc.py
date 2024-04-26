import numpy as np
from python_speech_features import mfcc,delta
import scipy.signal
from scipy.signal import cwt, morlet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA 
from sklearn.manifold import TSNE

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


class MyMFCC:
    def __init__(self, args) -> None:
        self.args = args

    def get_feat(self, dataSet):
        return self.get_z_mfcc_feat(dataSet)
    
    def single_axis_mfcc_feat(self,dataSet,axis):
        feat_list = []

        for i in range(dataSet.shape[0]):

            signal = dataSet[i, axis]
            mfcc_feat = mfcc(signal, self.args.fs, winstep=self.args.winstep, numcep=self.args.numcep,
                                nfilt=self.args.nfilt, nfft=self.args.nfft, ceplifter=self.args.ceplifter).reshape(-1)

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
            _, _, stft_feat = scipy.signal.stft(signal, fs=self.args.fs)
            stft_feat_flat = np.abs(stft_feat).reshape(-1)
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
            _, _, stft_feat = scipy.signal.stft(signal, fs=self.args.fs)
            stft_feat_flat = np.abs(stft_feat).reshape(-1)
            feat_list.append(stft_feat_flat)

        feat = np.array(feat_list)
        return feat
    def get_dft321_mfcc_feat(self, dataSet):
        feat_list = []

        for i in range(dataSet.shape[0]):
            signal = dataSet[i, :]
            signal = DFT321f(signal.T)
            mfcc_feat = mfcc(signal, self.args.fs, winstep=self.args.winstep, numcep=self.args.numcep,
                                nfilt=self.args.nfilt, nfft=self.args.nfft, ceplifter=self.args.ceplifter).reshape(-1)

            feat_list.append(mfcc_feat)

        feat = np.array(feat_list)
        # print(f"{feat.shape}")
        return feat
    
    

    def get_dft321_stft_feat(self, dataSet):
        feat_list = []

        for i in range(dataSet.shape[0]):
            signal = dataSet[i, :]
            signal = DFT321f(signal.T)
            # print(f"dft321 result: {signal.shape}")
            _, _, stft_feat = scipy.signal.stft(signal, fs=self.args.fs)
            # 将 STFT 结果转换为一维数组
            stft_feat_flat = np.abs(stft_feat).reshape(-1)
            feat_list.append(stft_feat_flat)

        feat = np.array(feat_list)
        return feat
    
    def get_wavelet_feat(self,dataSet):
        feat_list = []
        scales = np.arange(1, 10)
        # print(f"original signal's dimension: {dataSet[:,2].shape}")
        
        for i in range(dataSet.shape[0]):
            signal = dataSet[i, 2]
            coefficients = cwt(signal, morlet, scales)
            coefficients_flat = np.abs(coefficients).reshape(-1)
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
                mfcc_feat = mfcc(signal, self.args.fs, winstep=self.args.winstep, winlen=self.args.winlen ,numcep=self.args.numcep,
                                 nfilt=self.args.nfilt, nfft=self.args.nfft, ceplifter=self.args.ceplifter).reshape(-1)
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
                mfcc_feat = scipy.signal.stft(signal, fs=self.args.fs)
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
        
    # def reduce_dimension(self,feat):

        # print(f"feat_dim:{feat.shape}")

        # print(f"PCA")
        # pca = PCA(n_components=300)
        # feat_new = pca.fit(feat)

        # print(f"ICA")
        # ica = FastICA(n_components=300, random_state=12) 
        # feat_new = ica.fit_transform(feat)

        # print(f"TSNE")
        # tsne = TSNE(n_components=3, random_state=42)
        # feat_new = tsne.fit_transform(feat)

        # print(f"dimension after: {feat_new.shape}")
        # return feat_new