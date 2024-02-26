import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from python_speech_features import mfcc
from python_speech_features import delta
import logging
import logging.config

from logging_config import logger
from util import butter_lowpass_filter, butter_highpass_filter, generate_random_integers, plot_embedding, visualize_evaluation
from scipy.signal import find_peaks


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from itertools import product
from tqdm import tqdm


class MyDataset:
    def __init__(self, label_file_path, test_size=0.2, random_state=42, fs=1330, forward=0.5, backward=1.5, logger=None):
        self.logger = logger or logging.getLogger(__name__)

        # 打印参数信息
        params = {'test_size': test_size, 'random_state': random_state,
                  'forward': forward, 'backward': backward}
        self._print_params(params)

        labels_df = pd.read_csv(label_file_path)
        filePaths = labels_df['FilePath'].values
        data_list = []
        for file_path in filePaths:
            df = pd.read_csv(file_path)
            ACC = df['ACC_Z']
            ACC_filtered_L = butter_lowpass_filter(
                ACC, cutoff=500, fs=fs, order=6)
            ACC_filtered_H = butter_highpass_filter(
                ACC_filtered_L, cutoff=20, fs=fs, order=5)
            threshold = 1000
            peaks, _ = find_peaks(ACC_filtered_H, height=threshold)
            pivot = peaks[0]
            start_index = pivot - int(forward*fs)
            end_index = pivot + int(backward*fs)

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
            X, y, test_size=test_size, random_state=random_state)

        self.rate = fs
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_rate(self):
        return self.rate

    def get_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test

    def _print_params(self, params):
        self.logger.debug("======> MyDataset parameters: ")
        for param, value in params.items():
            self.logger.debug(f"{param}: {value}")


class MyMFCC:
    def __init__(self, rate, winstep=0.01, numcep=13, nfilt=26, nfft=512, ceplifter=22,
                 concat=False, dynamic=True, logger=None) -> None:

        self.logger = logger or logging.getLogger(__name__)

        # 打印参数信息
        params = {'sample_rate': rate, 'winstep': winstep, 'numcep': numcep,
                  'nfilt': nfilt, 'nfft': nfft, 'ceplifter': ceplifter, 'concat': concat, 'dynamic': dynamic}
        self._print_params(params)

        self.rate = rate
        self.winstep = winstep
        self.numcep = numcep
        self.nfilt = nfilt
        self.nfft = nfft
        self.ceplifter = ceplifter
        self.concat = concat
        self.dynamic = dynamic

    def _print_params(self, params):
        self.logger.debug("=======> MFCC parameters: ")

        # 循环打印参数
        for param, value in params.items():
            self.logger.debug(f"{param}: {value}")

    def get_feat(self, dataSet):
        # return self.get_add_feat(dataSet)
        return self.get_concat_feat(dataSet)

    def get_concat_feat(self, dataSet):
        concat = self.concat
        dynamic = self.dynamic

        feat_list = []

        for i in range(dataSet.shape[0]):
            mfcc_feats = []

            for axis in range(3):
                signal = dataSet[i, axis]
                mfcc_feat = mfcc(signal, self.rate, winstep=self.winstep, numcep=self.numcep,
                                 nfilt=self.nfilt, nfft=self.nfft, ceplifter=self.ceplifter).reshape(-1)
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
                axis_features = np.hstack(
                    (axis_features, delta_features, delta_delta_features))
                feat_dymatic.append(axis_features)

            feat_dy = np.concatenate(feat_dymatic, axis=1)

        if concat:
            if dynamic:
                return feat_dy
            else:
                return feat

        else:
            if dynamic:
                return feat_dy[:, :num_features]
            else:
                return feat[:, :num_features]

    def get_add_feat(self, dataSet):
        dynamic = self.dynamic

        feat_list = []

        for i in range(dataSet.shape[0]):

            x, y, z = dataSet[i]
            signal = x + y + z
            mfcc_feat = mfcc(signal, self.rate, winstep=self.winstep, numcep=self.numcep,
                             nfilt=self.nfilt, nfft=self.nfft, ceplifter=self.ceplifter).reshape(-1)
            feat_list.append(mfcc_feat)

        feat = np.array(feat_list)

        if dynamic:
            delta_features = delta(feat, 2)
            delta_delta_features = delta(delta_features, 2)
            feat = np.hstack((feat, delta_features, delta_delta_features))

        return feat


def predict_report(y_test, y_pred, logger):
    cm = confusion_matrix(y_test, y_pred)

    logger.debug(
        f"Predict Classification Report:\n{classification_report(y_test, y_pred, digits=4)}")
    logger.debug(f"Confusion Matrix:\n{cm}")


class MyMFCCExtractor:
    def __init__(self, myMFCC_instance):
        self.myMFCC_instance = myMFCC_instance

    def transform(self, X):
        return self.myMFCC_instance.get_feat(X)


def main(random_state=43, grid_search=False):
    dataSet = MyDataset(label_file_path='label_file.csv',
                        random_state=random_state, forward=0.5, backward=1, logger=logger)

    concat = True
    dynamic = False

    model = SVC(kernel='linear', probability=True, C=0.1)

    X_train, y_train, X_test, y_test = dataSet.get_data()
    rate = dataSet.get_rate()
    # myMFCC = MyMFCC(rate=rate, winstep=0.01, numcep=12, nfilt=20, nfft=256,ceplifter=22, concat=concat, dynamic=dynamic, logger=logger)
    myMFCC = MyMFCC(rate=rate, winstep=0.01, numcep=13, nfilt=26, nfft=512, ceplifter=22, concat=concat, dynamic=dynamic, logger=logger)

    if grid_search:
        param_grid = {
            'mfcc__winstep': [0.01],
            'mfcc__nfft': [256, 512, 1024, 2048],  # 其他FFT大小
            'mfcc__nfilt': [20, 22, 24, 26],  # 其他Mel滤波器数量
            'mfcc__numcep': [12, 13, 14],  # 其他倒谱系数个数
        }
        best_score = 0
        best_params = {}
        total_combinations = len(list(product(*param_grid.values())))

        for values in tqdm(product(*param_grid.values()), total=total_combinations, desc='Grid Search Progress'):
            params = dict(zip(param_grid.keys(), values))

            for param, value in params.items():
                component, param_name = param.split('__')
                if component == 'mfcc':
                    myMFCC.__setattr__(param_name, value)

            X_train_feat = myMFCC.get_feat(X_train)

            cross_val_scores = cross_val_score(
                model, X_train_feat, y_train, cv=5)
            avg_score = cross_val_scores.mean()

            if avg_score > best_score:
                best_score = avg_score
                best_params = params

            model.fit(X_train_feat, y_train)

            X_test_feat = myMFCC.get_feat(X_test)
            y_pred = model.predict(X_test_feat)
            logger.debug(f"Params:\n {params}")
            logger.debug(f"CV Scores:\n {avg_score}")
            logger.debug(f"Predict Scores:\n {accuracy_score(y_test,y_pred)}")
            # logger.debug(f"Predict Classification Report:\n{classification_report(y_test, y_pred, digits=4)}")

        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best cross-validation score: {best_score}")

        return

    else:
        X_train_feat = myMFCC.get_feat(X_train)

        cross_val_scores = cross_val_score(model, X_train_feat, y_train, cv=5)
        logger.info("CV Scores: {}".format(
            [round(score, 4) for score in cross_val_scores]))
        logger.info("Average CV Scores: {:.4f}".format(
            cross_val_scores.mean()))

        y_cv_pred = cross_val_predict(model, X_train_feat, y_train, cv=5)
        logger.debug(
            f"Validation Classification Report:\n{classification_report(y_train, y_cv_pred, digits=4)}")

        model.fit(X_train_feat, y_train)

        # predict
        X_test_feat = myMFCC.get_feat(X_test)
        y_pred = model.predict(X_test_feat)
        y_prob = model.predict_proba(X_test_feat)[:, 1]
        report = classification_report(
            y_test, y_pred, digits=2, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.round(2)
        report_df.to_csv('predict_report.csv')
        logger.debug(
            f"Predict Classification Report:\n{classification_report(y_test, y_pred, digits=4)}")

        # Example usage:
        visualize_evaluation(y_test, y_prob, y_pred, save_dir='./img')

        # feature visualization
        # X_feat = np.concatenate((X_train_feat,X_test_feat),axis=0)
        # labels = np.concatenate((y_train,y_test),axis=0)
        # plot_embedding(X_feat,labels,'2d')
        # plot_embedding(X_feat,labels,'3d')

        acc = accuracy_score(y_test, y_pred)
        logger.info(f"Predict scores: {acc}")

        return


if __name__ == "__main__":

    # random_numbers = generate_random_integers(10, 1, 100)
    # acc_list = []
    # for i,random_num in enumerate(random_numbers):
    #     logger.debug(f"=====> Epoch {i}")
    #     acc = main(random_state=random_num)
    #     acc_list.append(acc)

    # acc_array = np.array(acc_list)
    # logger.info("Predict Scores: {}".format([round(score, 4) for score in acc_array]))
    # logger.info("Average Predict Scores: {:.4f}".format(acc_array.mean()))

    # main(grid_search=True)
    main()
