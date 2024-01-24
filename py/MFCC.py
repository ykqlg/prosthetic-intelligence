import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import get_window
import scipy.fftpack as fft

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE

from python_speech_features import mfcc



# 这是一个画图函数，方便后续作图
def personal_plot(x, y):
    plt.figure(figsize=(12, 6))
    plt.rcParams['font.family'] = 'sans-serif'
    plt.plot(x, y)
    # plt.xlim(x[0],x[-1])
    plt.xlabel('time/s', fontsize=20)
    plt.ylabel('Amplitude', fontsize=20)
    plt.grid()


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

    # 创建一个空的数组 audio_fft 用于存储傅里叶变换的结果。数组的行数是 (1 + FFT_size // 2)，这是由于对称性导致 FFT 的结果只需要一半，以避免重复。列数等于音频窗口的数量
    # np.complex64 表示使用 64 位的复数数据类型，即每个复数由两个 32 位的浮点数表示，一个表示实部，一个表示虚部。
    # order='F'：这部分指定了数组的存储顺序。'F' 表示 Fortran（列优先）顺序。在 Fortran 顺序中，多维数组的元素在内存中是列优先排列的，即列方向的元素是连续存储的。这种存储顺序在涉及到频谱分析时可能更高效，因为通常对频率进行操作时会涉及到列方向的计算。
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

    # 梅尔三角滤波器
    filters = get_filters(filter_points, FFT_size)

    # 将三角形 MEL 权重除以 MEL 带的宽度（面积归一化）
    # 三角形面积 = 1/2 * 底 * 高；高一致，可忽略；对每个filter乘以面积的倒数
    # 如果我们不对滤波器进行归一化，我们会看到由于滤波器宽度的原因，噪声随着频率的增加而增加。
    enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
    filters *= enorm[:, np.newaxis]  # np.newaxis用于增加一个维度，可以与filter的向量元素做内积

    audio_filtered = np.dot(filters, np.transpose(audio_power))
    audio_log = np.log10(audio_filtered)  # 有其他代码使用20？

    dct_filter_num = 13
    dct_filters = dct(dct_filter_num, mel_filter_num)
    cepstral_coefficents = np.dot(dct_filters, audio_log)
    return cepstral_coefficents

def lib_mfcc(filePath,mel_filter_num):
    df = pd.read_csv(filePath)
    sig = df['ACC_X'].to_numpy()
    t = df['Time'].to_numpy()
    rate = int(1/np.mean(np.diff(t)))
    mfcc_feat = mfcc(sig,rate,winstep=0.015,numcep=13,nfilt=mel_filter_num,nfft=2048,ceplifter=0,winfunc=np.hanning)
    return mfcc_feat

    
def plot_embedding_3d(X, label, title=None):
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    # 降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], X[i, 2], str(label[i]),
                color=plt.cm.Set1(label[i] / 10.),
                fontdict={'weight': 'bold', 'size': 9})
    if title is not None:
        plt.title(title)
        plt.savefig(title)

def train(filePaths,labels, model, mel_filter_num):
    print('melfilternum:',mel_filter_num)
    
    cepstral_coefficents = []
    for filePath in filePaths:
        # cepstral_coefficent = get_cepstral_coefficents(filePath,mel_filter_num).reshape(-1)
        cepstral_coefficent = lib_mfcc(filePath,mel_filter_num).reshape(-1)
        cepstral_coefficents.append(cepstral_coefficent)
        
    cepstral_coefficents = np.array(cepstral_coefficents)
    
    X_train, X_test, y_train, y_test = train_test_split(
        cepstral_coefficents, labels, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    
    # 使用交叉验证评估模型性能
    # cross_val_scores = cross_val_score(model, X_train, y_train, cv=5)  # 5折交叉验证
    # print("交叉验证分数:", cross_val_scores)
    # print("平均交叉验证分数:", cross_val_scores.mean())

    # 步骤 4: 预测
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.4f}%')
    return accuracy

    
if __name__ == "__main__":
    df = pd.read_csv('../label_file_replaced.csv')
    filePaths = df['FilePath']
    labels = df['Label']
    # 步骤 3: 训练SVM模型
    svm_model = SVC(kernel='linear')
    # train(filePaths, labels, svm_model, 18)
    
    accuracy = []
    mel_filter_num = range(1,25,1)
    for i in mel_filter_num:
        acc = train(filePaths, labels, svm_model, i)
        accuracy.append(acc)
    
    title = 'img/mel_filter_num_0.2_lib.png'
    
    
    plt.plot(mel_filter_num, accuracy)
    plt.xlabel('mel_filter_num')
    plt.xticks(mel_filter_num)
    plt.ylabel('accuracy')
    plt.title(title)
    plt.grid()
    plt.savefig(title)
    
        

    # tsne = TSNE(n_components=2, random_state=42)
    # # tsne = TSNE(n_components=3, init='pca', random_state=42)
    # X_tsne = tsne.fit_transform(cepstral_coefficents)

    # # plot_embedding_3d(X_tsne, labels, 't-SNE-3D.png')

    # # 绘制可视化结果
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis')
    # plt.title('t-SNE Visualization')
    # plt.savefig('t-SNE.png')

