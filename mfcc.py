import numpy as np
import pandas as pd
from python_speech_features import mfcc
from python_speech_features import delta

import decimal

import numpy
import math

class MyMFCC:
    def __init__(self, args) -> None:
        self.args = args

    def get_feat(self, dataSet):
        # return self.get_add_feat(dataSet)
        return self.get_concat_feat(dataSet)

    def get_concat_feat(self, dataSet):
        concat = self.args.concat
        dynamic = self.args.dynamic

        feat_list = []

        for i in range(dataSet.shape[0]):
            mfcc_feats = []

            for axis in range(3):
                signal = dataSet[i, axis]
                mfcc_feat = mfcc(signal, self.args.fs, winstep=self.args.winstep, numcep=self.args.numcep,
                                 nfilt=self.args.nfilt, nfft=self.args.nfft, ceplifter=self.args.ceplifter).reshape(-1)
                mfcc_feats.append(mfcc_feat)

            # 将 XYZ 轴的 MFCC 特征连接在一起
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

        if concat:
            if dynamic:
                return feat_dy
            else:
                return feat

        else:
            if dynamic:
                return feat_dy[:, num_features*2:]
            else:
                return feat[:, num_features*2:]