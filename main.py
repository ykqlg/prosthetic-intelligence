import numpy as np
import pandas as pd
import time

from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


import logging
import logging.config

from logging_config import logger
from util import generate_random_integers, plot_embedding, visualize_evaluation


from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import MyDataset
from mfcc import MyMFCC

def main(random_state=42, grid_search=False):
    main_start_time = time.time()
    dataSet = MyDataset(label_file_path='label_file.csv',
                        random_state=random_state, forward=0.5, backward=0.8, logger=logger)

    concat = True # 改成false会掉两个点，从0.97到0.95
    dynamic = False # 这个会导致特征提取时间增加非常多，原先run一次5s，变成一次120s

    model = SVC(kernel='linear', probability=True, C=0.1)

    X_train, y_train, X_test, y_test = dataSet.get_data()
    rate = dataSet.get_rate()
    # myMFCC = MyMFCC(rate=rate, winstep=0.01, numcep=12, nfilt=20, nfft=256,ceplifter=22, concat=concat, dynamic=dynamic, logger=logger)
    myMFCC = MyMFCC(rate=rate, winstep=0.01, numcep=13, nfilt=26, nfft=512,
                    ceplifter=22, concat=concat, dynamic=dynamic, logger=logger)
    
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
        # logger.info("CV Scores: {}".format([round(score, 4) for score in cross_val_scores]))
        logger.info("5-Fold CV: {:.4f}".format(cross_val_scores.mean()))



        model.fit(X_train_feat, y_train)

        # predict
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

        # Example usage:
        # visualize_evaluation(y_test, y_prob, y_pred, save_dir='./img')

        # feature visualization
        X_feat = np.concatenate((X_train_feat,X_test_feat),axis=0)
        labels = np.concatenate((y_train,y_test),axis=0)
        plot_embedding(X_feat,labels,'2d')
        # plot_embedding(X_feat,labels,'3d')

        
        main_end_time = time.time()
        main_time = main_end_time - main_start_time
        acc = accuracy_score(y_test, y_pred)
        logger.info(f"Test acc: {acc:.4f}")
        logger.info(f"Time: {main_time:.4f}")
        

        return acc
    

def time_test_main(random_state=42, grid_search=False):
    # backward_values = np.arange(0.5,1.5,step=0.1)
    # backward_values = np.linspace(0.3, 1.2, num=10, endpoint=True)
    # backward_values = [0.6,0.65, 0.7, 0.725, 0.75, 0.775, 0.8,0.9]
    backward_values = [0.6,0.7, 0.73, 0.76,0.8,0.9]
    # backward_values_arange = np.arange(0.5, 1.51, 0.1)
    logger.info(f"backward_values: {backward_values}")

    times = []
    accuracies = []
    
    seed = 43
    random_numbers = generate_random_integers(5, 1, 100,seed=seed)
    logger.info(f"random_seed: {seed}")
    logger.info(f"random_numbers:{random_numbers}")
    # pbar = tqdm(backward_values,leave=True)
    with tqdm(total=len(backward_values), desc='Backward Loop') as pbar:
        for backward_value in backward_values:
            pbar.set_description('Backward Loop: '+str(backward_value))
            acc_list = []
            time_list = []
            for random_num in random_numbers:
                dataSet = MyDataset(label_file_path='label_file.csv',
                                random_state=random_num, forward=0.5, backward=backward_value, logger=logger)

                concat = False
                dynamic = False

                model = SVC(kernel='linear', probability=True, C=0.1)
                

                X_train, y_train, X_test, y_test = dataSet.get_data()
                rate = dataSet.get_rate()
                # myMFCC = MyMFCC(rate=rate, winstep=0.01, numcep=12, nfilt=20, nfft=256,ceplifter=22, concat=concat, dynamic=dynamic, logger=logger)
                myMFCC = MyMFCC(rate=rate, winstep=0.01, numcep=13, nfilt=26, nfft=512,
                                ceplifter=22, concat=concat, dynamic=dynamic, logger=logger)

                
                X_train_feat = myMFCC.get_feat(X_train)
                model.fit(X_train_feat, y_train)
                

                # predict
                test_len = X_test.shape[0]
                
                start_time = time.time()
                
                X_test_feat = myMFCC.get_feat(X_test)
                y_pred = model.predict(X_test_feat)
                acc = accuracy_score(y_test, y_pred)
                
                end_time = time.time()
                one_sample_time = (end_time - start_time)/test_len
                time_list.append(one_sample_time)
                acc_list.append(round(acc,4))
                
            mean_acc= np.mean(acc_list)
            mean_time = np.mean(time_list)
            accuracies.append(mean_acc)
            times.append(mean_time)
            pbar.update(1)
            pbar.set_postfix(accuracy=f'{mean_acc:.4f}')

        
        # print(f'backward: {backward_value}')
    logger.info(f"accuracies: {accuracies}")
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Backward Values')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(backward_values, accuracies, marker='s', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    
    # color = 'tab:red'
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Average Time (seconds)', color=color)
    # ax2.plot(backward_values, times, marker='o', color=color)
    # ax2.tick_params(axis='y', labelcolor=color)
    

    fig.tight_layout()
    plt.title('Process Time and Accuracy with different backward values', pad=20)  # 通过 pad 参数调整标题与图的距离
    plt.subplots_adjust(top=0.85)  # 调整整个图形的顶部位置
    # plt.title('Process Time and Accuracy with different backward values')
    plt.savefig('./img/backward_accuracy&time.png')
    plt.show()

    return
    
    
def svm_gridsearch(random_state=42, grid_search=False):
    dataSet = MyDataset(label_file_path='label_file.csv',
                        random_state=random_state, forward=0.5, backward=0.7, logger=logger)

    concat = True
    dynamic = False

    svm_model = SVC(kernel='linear', probability=True, C=0.1)

    X_train, y_train, X_test, y_test = dataSet.get_data()
    rate = dataSet.get_rate()
    # myMFCC = MyMFCC(rate=rate, winstep=0.01, numcep=12, nfilt=20, nfft=256,ceplifter=22, concat=concat, dynamic=dynamic, logger=logger)
    myMFCC = MyMFCC(rate=rate, winstep=0.01, numcep=13, nfilt=26, nfft=512,
                    ceplifter=22, concat=concat, dynamic=dynamic, logger=logger)
    param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.01, 0.1, 1, 10],
              'kernel': ['linear', 'rbf', 'poly']}
    X_train_feat = myMFCC.get_feat(X_train)

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

    # random_numbers = generate_random_integers(5, 1, 100)
  
    # acc_list = []
    # for i,random_num in enumerate(random_numbers):
    #     logger.info(f"=> Exp {i+1}")
    #     acc = main(random_state=random_num)
    #     acc_list.append(acc)

    # acc_array = np.array(acc_list)
    # logger.info("\n")
    # logger.info("Average Test Acc: {:.4f}".format(acc_array.mean()))

    # main(grid_search=True)
    # main()
    # svm_gridsearch()

    time_test_main()