from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import pandas as pd
import numpy as np
from scipy.signal import get_window
import scipy.fftpack as fft


def emphasized_audio(audio, alpha=0.97):
    emphasized_audio = np.append(audio[0], audio[1:] - alpha * audio[:-1])
    return emphasized_audio


def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio


def frame_audio(audio, FFT_size=2048, hop_size=10, sample_rate=44100):
    # hop_size in ms
    audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
    frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
    frame_num = int((len(audio) - FFT_size) / frame_len) + 1
    frames = np.zeros((frame_num, FFT_size))

    for n in range(frame_num):
        frames[n] = audio[n*frame_len:n*frame_len+FFT_size]

    return frames


def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def met_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)


def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)

    # print("MEL min: {0}".format(fmin_mel))
    # print("MEL max: {0}".format(fmax_mel))

    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    freqs = met_to_freq(mels)
    # f(m-1)和f(m)、f(m+1)分别对应第m个滤波器的起始点、中间点和结束点。大家一定要注意的一点是，这里的f(m)对应的值不是频率值，而是对应的sample的索引！比如，我们这里最大频率是22050 Hz, 所以22050Hz对应的是第513个sample，即频率f所对应的值是f/fs*NFFT
    return np.floor((FFT_size + 0.5) / sample_rate * freqs).astype(int), freqs


def get_filters(filter_points, FFT_size):
    filters = np.zeros((len(filter_points)-2, int(FFT_size/2+1)))

    for n in range(len(filter_points)-2):
        # 相比于原kaggle代码，增加了`endpoint=False`参数
        filters[n, filter_points[n]: filter_points[n + 1]] = np.linspace(
            0, 1, filter_points[n + 1] - filter_points[n], endpoint=False)
        filters[n, filter_points[n + 1]: filter_points[n + 2]] = np.linspace(
            1, 0, filter_points[n + 2] - filter_points[n + 1], endpoint=False)

    return filters


def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num, filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)

    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)

    return basis


def get_cepstral_coefficents(filePath, mel_filter_num=10):
    df = pd.read_csv(filePath)
    y = df['ACC_X'].to_numpy()
    t = df['Time'].to_numpy()
    sample_rate = int(1/np.mean(np.diff(t)))

    # nomalize和emphazie 区别？
    # audio = normalize_audio(y)
    audio = emphasized_audio(y, alpha=0.97)

    hop_size = 15  # ms 以这个作为帧的间隔，相当于stride
    FFT_size = 2048

    audio_framed = frame_audio(
        audio, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate)

    # 加窗 跟hamming？
    window = get_window("hann", FFT_size, fftbins=True)
    audio_win = audio_framed * window

    # 进行STFT
    # 这种转置再转置的原因：在音频信号处理中，更习惯将时间窗口放在第一维
    audio_winT = np.transpose(audio_win)  # audio_winT.shape:(2048,133)

    audio_fft = np.empty(
        (int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')

    # 对每一帧进行fft
    for n in range(audio_fft.shape[1]):
        audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[
            :audio_fft.shape[0]]

    audio_fft = np.transpose(audio_fft)  # audio_fft.shape:(133, 1025)

    audio_power = np.square(np.abs(audio_fft))

    freq_min = 0
    freq_high = sample_rate / 2  # 奈奎斯特频率
    # mel_filter_num = 18  # 参数可调

    filter_points, mel_freqs = get_filter_points(
        freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=44100)

    filters = get_filters(filter_points, FFT_size)

    enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
    filters *= enorm[:, np.newaxis]  # np.newaxis用于增加一个维度，可以与filter的向量元素做内积

    audio_filtered = np.dot(filters, np.transpose(audio_power))
    audio_log = 10.0 * np.log10(audio_filtered)  # 有其他代码使用20？
    print('logfbank:',audio_log)

    # dct_filter_num = 40
    # dct_filters = dct(dct_filter_num, mel_filter_num)
    # cepstral_coefficents = np.dot(dct_filters, audio_log)
    # return cepstral_coefficents

if __name__ == "__main__":

    filePath = './output_cropped/20240112_142401.csv'
    df = pd.read_csv(filePath)
    sig = df['ACC_X'].to_numpy()
    t = df['Time'].to_numpy()
    rate = int(1/np.mean(np.diff(t)))
    # (rate,sig) = wav.read("english.wav")
    mfcc_feat = mfcc(sig,rate)
    # d_mfcc_feat = delta(mfcc_feat, 2)
    fbank_feat = logfbank(sig,rate)

    print('library:',fbank_feat[1:3,:])
    
    get_cepstral_coefficents(filePath)