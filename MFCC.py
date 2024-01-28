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
from sklearn.model_selection import GridSearchCV

class MyDataset:
    def __init__(self, label_file_path, test_size=0.2, random_state=42):
        # 从CSV文件加载数据
        labels_df = pd.read_csv(label_file_path)
        filePaths = labels_df['FilePath'].values
        # n_sample = len(filePaths)
        data_list = []
        for file_path in filePaths:
            data_df = pd.read_csv(file_path)
            sig = data_df['ACC_X'].values
            t = data_df['Time'].values
            rate = int(1/np.mean(np.diff(t)))
            data = [sig, rate]
            data_list.append(data)
            
        X = np.array(data_list, dtype=object)
        y = labels_df['Label'].values
        

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # 提取特征和标签
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
    
    def get_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def get_X_data(self):
        return np.concatenate(self.X_train,self.X_test)
    
    def get_y_data(self):
        return np.concatenate(self.y_train,self.y_test)

def MyMFCC(dataSet):
    feat_list = []
    for i in range(dataSet.shape[0]):
        sig = dataSet[i,0]
        rate = dataSet[i,1]
        mfcc_feat = mfcc(sig,rate,winstep=0.01,numcep=5,nfilt=26,nfft=512,ceplifter=0,winfunc=np.hamming)
        feat_reshaped = mfcc_feat.reshape(-1)
        feat_list.append(feat_reshaped)
    return np.array(feat_list)
    
    
    
    
def lib_mfcc(filePath,param):
    df = pd.read_csv(filePath)
    sig = df['ACC_X'].to_numpy()
    t = df['Time'].to_numpy()
    rate = int(1/np.mean(np.diff(t)))
    mfcc_feat = mfcc(sig,rate,winstep=0.01,numcep=5,nfilt=26,nfft=512,ceplifter=0,winfunc=np.hamming)
    # mfcc_feat = mfcc(sig,rate)
    return mfcc_feat

    
def plot_embedding_3d(X, label, title=None):
    # 坐标缩放到[0,1]区间
    # x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    # X = (X - x_min) / (x_max - x_min)
    # 降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    fig = plt.figure()
    ax = fig.add_subplot(2,2,2,projection='3d')
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], X[i, 2], 
                str(label[i]),
                # color=plt.cm.Set1(label[i] / 2.),
                color = 'r' if label[i] == 0 else 'b',
                fontdict={'weight': 'bold', 'size': 9})
    if title is not None:
        plt.title(title)
        plt.savefig(title)
        
    
    plt.show()

def train(filePaths,labels, model, param,random_state=42):
    print('current_param:',param)
    
    cepstral_coefficents = []
    for filePath in filePaths:
        # cepstral_coefficent = get_cepstral_coefficents(filePath,param).reshape(-1)
        cepstral_coefficent = lib_mfcc(filePath,param).reshape(-1)
        cepstral_coefficents.append(cepstral_coefficent)
        
    cepstral_coefficents = np.array(cepstral_coefficents)
    
    X_train, X_test, y_train, y_test = train_test_split(
        cepstral_coefficents, labels, test_size=0.3, random_state=42)
    
    
    # param_grid = {
    #     'C': [0.1, 1, 10, 100],
    #     'gamma': [0.01, 0.1, 1, 10],
    # }
    # grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    # grid_search.fit(X_train, y_train)
    # print("Best parameters:", grid_search.best_params_)
    # print("Cross-validated mean accuracy:", grid_search.best_score_)

    
    # 使用交叉验证评估模型性能
    cross_val_scores = cross_val_score(model, X_train, y_train, cv=5)  # 5折交叉验证
    # print(f"交叉验证分数: {cross_val_scores:.4f}")
    # print(f"平均交叉验证分数: {cross_val_scores.mean()}:.4f")
    model.fit(X_train, y_train)
    
    print("交叉验证分数:", ["{:.4f}".format(score) for score in cross_val_scores])
    print("平均交叉验证分数: {:.4f}".format(cross_val_scores.mean()))

    # 步骤 4: 预测
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    
    # 绘制可视化结果
    tsne = TSNE(n_components=3, random_state=42)
    # tsne = TSNE(n_components=3, init='pca', random_state=42)
    X_tsne = tsne.fit_transform(cepstral_coefficents)

    plot_embedding_3d(X_tsne, labels, 'img/t-SNE-3D.png')
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis')
    # plt.title('t-SNE Visualization')
    # plt.savefig('img/t-SNE-0.1.png')
    
    
    return accuracy

def main():
    df = pd.read_csv('label_file_replaced.csv')
    filePaths = df['FilePath']
    labels = df['Label']
    
   

    # svm_model = SVC(kernel='rbf',gamma=0.001)
    svm_model = SVC(kernel='linear',C=1)
    # train(filePaths, labels, svm_model, 18)
    
    accuracy = []
    
    param = range(1,26,1)
    # param = np.logspace(7, 12, 6, base=2, dtype=int)
    # for i in param:
    #     acc = train(filePaths, labels, svm_model, i)
    #     accuracy.append(acc)
        
    acc = train(filePaths, labels, svm_model, 26)
    accuracy.append(acc)
    
    # title = 'img/nfilt_0.3.png'
    # plt.plot(param, accuracy)
    # plt.xlabel('nfilt')
    # # plt.xticks(param)
    # plt.ylabel('accuracy')
    # plt.title(title)
    # plt.grid()
    # plt.savefig(title)


def newMain():
    dataSet = MyDataset('label_file_replaced.csv')
    model = SVC(kernel='linear')
    
    X_train, y_train, X_test, y_test = dataSet.get_data()
    X_train_feat = MyMFCC(X_train)
    
    cross_val_scores = cross_val_score(model, X_train_feat, y_train, cv=5)  # 5折交叉验证
    # print(f"交叉验证分数: {cross_val_scores:.4f}")
    # print(f"平均交叉验证分数: {cross_val_scores.mean()}:.4f")
    
    print("交叉验证分数:", ["{:.4f}".format(score) for score in cross_val_scores])
    print("平均交叉验证分数: {:.4f}".format(cross_val_scores.mean()))

    model.fit(X_train_feat, y_train)
    # 步骤 4: 预测
    X_test_feat = MyMFCC(X_test)
    y_pred = model.predict(X_test_feat)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    
    # 绘制可视化结果
    tsne = TSNE(n_components=3, random_state=42)
    
    X_all = np.concatenate((X_train_feat,X_test_feat),axis=0)
    y_all = np.concatenate((y_train,y_test),axis=0)
    # tsne = TSNE(n_components=3, init='pca', random_state=42)
    X_tsne = tsne.fit_transform(X_all)

    plot_embedding_3d(X_tsne, y_all, 'img/t-SNE-3D.png')
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis')
    # plt.title('t-SNE Visualization')
    # plt.savefig('img/t-SNE-0.1.png')
    


    
    
    
if __name__ == "__main__":
    # main()
    newMain()
        

    

