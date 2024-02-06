import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE

from python_speech_features import mfcc
from python_speech_features import delta
from sklearn.model_selection import GridSearchCV

class MyDataset:
    def __init__(self, label_file_path, test_size=0.2, random_state=42,shuffle=True):
        labels_df = pd.read_csv(label_file_path)
        filePaths = labels_df['FilePath'].values
        data_list = []
        for file_path in filePaths:
            data_df = pd.read_csv(file_path)
            sig_X = data_df['ACC_X'].values
            sig_Y = data_df['ACC_Y'].values
            sig_Z = data_df['ACC_Z'].values
            t = data_df['Time'].values
            # rate = int(1/np.mean(np.diff(t)))
            data = [sig_X, sig_Y, sig_Z]
            data_list.append(data)
            
        X = np.array(data_list)
        y = labels_df['Label'].values
        

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

        self.rate = 1330
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_rate(self):
        return self.rate
    
    def get_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test
    


class MyMFCC:
    def __init__(self,dataSet,rate,winstep=0.01,numcep=13,nfilt=26,nfft=512,ceplifter=22,
           concat=False,dynamic=True) -> None:
        self.concat = concat
        self.dynamic = dynamic
        
        feat_list = []
        
        for i in range(dataSet.shape[0]):
            mfcc_feats = []  

            for axis in range(3):  
                signal = dataSet[i, axis]
                mfcc_feat = mfcc(signal, rate, winstep=winstep, numcep=numcep, nfilt=nfilt, nfft=nfft, ceplifter=ceplifter).reshape(-1)
                mfcc_feats.append(mfcc_feat)

            # 将 XYZ 轴的 MFCC 特征连接在一起
            concatenated_feats = np.concatenate(mfcc_feats)
            feat_list.append(concatenated_feats)

        feat = np.array(feat_list)
        num_features = feat.shape[1]
        
        feat_dy = None
        if dynamic:
            num_features = int(feat.shape[1]/3)
            feat_dymatic = []
            for start_idx in range(0, feat.shape[1], num_features):
                end_idx = start_idx + num_features
                axis_features = feat[:, start_idx:end_idx]
                delta_features = delta(axis_features, 2)
                delta_delta_features = delta(delta_features, 2)
                axis_features = np.hstack((axis_features, delta_features, delta_delta_features))
                feat_dymatic.append(axis_features)

            feat_dy = np.concatenate(feat_dymatic, axis=1)

        self.feat = feat
        self.feat_dy = feat_dy
        self.num_features = num_features
            
        
    def get_feat(self):
        concat = self.concat
        dynamic = self.dynamic
        if concat:
            if dynamic: return self.feat_dy 
            else: return self.feat
            
        else:
            if dynamic: return self.feat_dy[:,self.num_features] 
            else: return self.feat[:,self.num_features] 
        
        
        
    
def plot_embedding(feat, label, projection='2d'):
    
    if projection== '2d':
        tsne = TSNE(n_components=2, random_state=42)
        X = tsne.fit_transform(feat)
        plt.scatter(X[:, 0], X[:, 1], c=label, cmap='viridis')
        plt.title('t-SNE-2D Visualization')
        # plt.savefig('img/t-SNE-0.1.png')
        plt.show()
    
    if projection== '3d':
        tsne = TSNE(n_components=3, random_state=42)
        X = tsne.fit_transform(feat)
        x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
        X = (X - x_min) / (x_max - x_min)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection='3d')
        
        for i in range(X.shape[0]):
            ax.scatter(X[i, 0], X[i, 1], X[i, 2], color='r' if label[i] == 0 else 'b', s=10)  # s 参数控制点的大小
            # ax.text(X[i, 0], X[i, 1], X[i, 2], str(label[i]),color = 'r' if label[i] == 0 else 'b',fontdict={'weight': 'bold', 'size': 9})
            
        plt.title('t-SNE-2D Visualization')
        # plt.savefig(title)
        plt.show()
    
def predict_report(y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)

    # 打印结果
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(cm)


def main():
    # label_file_name = 'label_file_replaced.csv'
    # label_file_name = 'output_1330Hz_0.5_1_label_file.csv'
    # label_file_name = 'output_1330Hz_0.5_0.75_label_file.csv'
    label_file_name = 'output_1330Hz_0.5_0.5_label_file.csv'
    dataSet = MyDataset(label_file_name,test_size=0.3)
    
    concat = True
    dynamic = False # 使用这个好像会让准确率降低
    
    model = SVC(kernel='linear')
    # param_grid = {'C': [0.1, 1, 10, 100, 1000],  
    #             'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
    #             'kernel': ['linear','poly','rbf']}  

    # model = GridSearchCV(SVC(), param_grid, cv=5) 
    
    X_train, y_train, X_test, y_test = dataSet.get_data()
    rate = dataSet.get_rate()
    X_train_feat = MyMFCC(X_train,rate,concat=concat,dynamic=dynamic).get_feat()

    
    # cross_val_score 来比较不同参数下模型的性能。
    cross_val_scores = cross_val_score(model, X_train_feat, y_train, cv=5)
    print("CV Scores:", [round(score, 4) for score in cross_val_scores])
    print("Averange CV Scores: {:.4f}".format(cross_val_scores.mean()))
    
    # 而 cross_val_predict 通常在确定了最优参数后，用于生成最终的整体预测结果，以进行进一步的评估和后处理。
    # y_cv_pred = cross_val_predict(model, X_train_feat, y_train, cv=5) 
    # print(classification_report(y_train, y_cv_pred, digits=4))

    # train
    model.fit(X_train_feat, y_train)
    
    # print(model.best_params_) 
    # print(model.best_estimator_) 
        
    # predict
    X_test_feat = MyMFCC(X_test,rate,concat=concat,dynamic=dynamic).get_feat()
    y_pred = model.predict(X_test_feat)
    predict_report(y_test,y_pred)
    
    # feature visualization
    # X_feat = np.concatenate((X_train_feat,X_test_feat),axis=0)
    # labels = np.concatenate((y_train,y_test),axis=0)
    # plot_embedding(X_feat,labels,'2d')
    # plot_embedding(X_feat,labels,'3d')

    
if __name__ == "__main__":
    main()
        

    

