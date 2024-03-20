import numpy as np
import pandas as pd
import time
import sys

from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


import logging.config

from logging_config import logger
from util import generate_random_integers, plot_embedding, visualize_evaluation


from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import MyDataset
from mfcc import MyMFCC

from opts import args

def main():
    random_numbers = generate_random_integers(3, 1, 100)
    acc_list = []
    time_list = []
    for random_num in random_numbers:
        args.random_state= random_num
        dataSet = MyDataset(args)
        model = SVC(kernel='linear', probability=True, C=0.1)
        X_train, y_train, X_test, y_test = dataSet.get_data()
        myMFCC = MyMFCC(args)
        

        X_train_feat = myMFCC.get_feat(X_train)
        # feat_num = X_train_feat.shape[1]
        # logger.info(f"feat_num: {feat_num}")
        
        # cross_val_scores = cross_val_score(model, X_train_feat, y_train, cv=5)
        # logger.info("5-Fold CV: {:.4f}".format(cross_val_scores.mean()))
        model.fit(X_train_feat, y_train)

        test_len = X_test.shape[0]
        start_time = time.time()
        
        X_test_feat = myMFCC.get_feat(X_test)
        y_pred = model.predict(X_test_feat)
        y_prob = model.predict_proba(X_test_feat)[:, 1]
        
        # report = classification_report(
        #     y_test, y_pred, digits=2, output_dict=True)
        # report_df = pd.DataFrame(report).transpose()
        # report_df = report_df.round(2)
        # report_df.to_csv('predict_report.csv')
        # logger.debug(
        #     f"Predict Classification Report:\n{classification_report(y_test, y_pred, digits=4)}")

        # visualize_evaluation(y_test, y_prob, y_pred, save_dir='./img')

        # feature visualization
        # X_feat = np.concatenate((X_train_feat,X_test_feat),axis=0)
        # labels = np.concatenate((y_train,y_test),axis=0)
        # plot_embedding(X_feat,labels,'2d')
        # plot_embedding(X_feat,labels,'3d')

        acc = accuracy_score(y_test, y_pred)
        end_time = time.time()
        one_sample_time = (end_time - start_time)/test_len
        time_list.append(one_sample_time)
        acc_list.append(acc)
            
    mean_acc= round(np.mean(acc_list),5)
    mean_time = round(np.mean(time_list)*1000,5)
    
    return mean_acc,mean_time
    

def param():
    
    param_name = 'nfft' # 必须与args的参数名严格一致
    task_name = param_name+'_param_test'
    # param_list = np.linspace(0.3, 1.2, num=10, endpoint=True)
    # param_list = np.linspace(0.1, 0.6, num=12, endpoint=True)
    # param_list = [0.6,0.65, 0.7, 0.725, 0.75, 0.775, 0.8,0.9]
    # param_list = [0.6,0.7, 0.73, 0.76,0.8,0.9]
    
    param_list = [128,256, 512, 1024,2048]
    # param_list = [2,4,6,8, 10,12,14,16]
    logger.info(f"{task_name}: {param_list}")

    times = []
    accuracies = []
    
    random_numbers = generate_random_integers(3, 1, 100)
    logger.debug(f"random_numbers:{random_numbers}")
    with tqdm(total=len(param_list)) as pbar:
        for param in param_list:
            setattr(args, param_name, param)
            
            acc,time = main()
            accuracies.append(acc)
            times.append(time)
            pbar.set_description(f'Accuracy: {acc:.4f}, {param_name}={param}')
            pbar.update(1)

        
    logger.info(f"accuracies: {accuracies}")
    logger.info(f"times: {times}")
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Param Values')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(param_list, accuracies, marker='s', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # color = 'tab:red'
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Time (ms)', color=color)
    # ax2.plot(param_list, times, marker='o', color=color)
    # ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(task_name, pad=20)
    plt.subplots_adjust(top=0.85)
    # plt.savefig('./img/backward_accuracy&time.png')
    plt.savefig('./img/'+task_name+'.png')
    plt.show()

    return
    
    
def svm_gridsearch():
    dataSet = MyDataset(args)
    X_train, y_train, X_test, y_test = dataSet.get_data()

    svm_model = SVC(kernel='linear', probability=True, C=0.1)

    myMFCC = MyMFCC(args)
    X_train_feat = myMFCC.get_feat(X_train)
    
    param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.01, 0.1, 1, 10],
              'kernel': ['linear', 'rbf', 'poly']}

    grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_feat, y_train)
    
    print("最优参数：", grid_search.best_params_)

    # 在验证集上评估性能
    best_model = grid_search.best_estimator_

    X_test_feat = myMFCC.get_feat(X_test)
    y_pred = best_model.predict(X_test_feat)
    y_prob = best_model.predict_proba(X_test_feat)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Predict scores: {acc}")

    return acc


if __name__ == "__main__":
    logger.debug("Arguments: %s", sys.argv[1:])

    # acc = main()
    # logger.info(f"acc : {acc}")
    # svm_gridsearch()

    param()
