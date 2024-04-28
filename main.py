import numpy as np
import pandas as pd
import time
import sys

from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging.config
from logging_config import logger
from util import generate_random_integers, plot_embedding, visualize_evaluation
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import MyDataset,MultiDataset
from mfcc import MyMFCC
from opts import args


from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FastICA 
from sklearn.decomposition import NMF



def main():
    random_numbers = generate_random_integers(args.repeat_num, 1, 100)
    acc_list = []
    time_list = []
    for random_num in random_numbers:
        args.random_state= random_num
        dataSet = MyDataset(args)
        model = SVC(kernel='linear', probability=True, C=0.1)
        X_train, y_train = dataSet.get_train_data()
        X_test, y_test = dataSet.get_test_data()
        myMFCC = MyMFCC(args)
        

        X_train_feat = myMFCC.get_feat(X_train)
        feat_num = X_train_feat.shape[1]
        # logger.info(f"feat_num: {feat_num}")
        
        # acc = cross_val_score(model, X_train_feat, y_train, cv=5).mean()
        # logger.info("5-Fold CV: {:.4f}".format(acc))
        model.fit(X_train_feat, y_train)
        joblib.dump(model, './model/model.pkl')
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
    
def feat_method():
    model = SVC(kernel='linear', probability=True, C=0.1)
    myMFCC = MyMFCC(args)

    methods = {
        # 'get_x_mfcc_feat': myMFCC.get_x_mfcc_feat,
        # 'get_y_mfcc_feat': myMFCC.get_y_mfcc_feat,
        
        'get_z_mfcc_feat': myMFCC.get_z_mfcc_feat,
        # 'get_concat_mfcc_feat': myMFCC.get_concat_mfcc_feat,
        # 'get_add_mfcc_feat': myMFCC.get_add_mfcc_feat,
        # 'get_dft321_mfcc_feat': myMFCC.get_dft321_mfcc_feat,
        
        # 'get_x_stft_feat': myMFCC.get_x_stft_feat,
        # 'get_y_stft_feat': myMFCC.get_y_stft_feat,
        
        'get_z_stft_feat': myMFCC.get_z_stft_feat,
        # 'get_concat_stft_feat': myMFCC.get_concat_stft_feat,
        # 'get_add_stft_feat': myMFCC.get_add_stft_feat,
        # 'get_dft321_stft_feat':myMFCC.get_dft321_stft_feat,
        
        # 'get_wavelet_feat': myMFCC.get_wavelet_feat,
        # 'get_dft321_wavelet_feat':myMFCC.get_dft321_wavelet_feat
    }
    results = []
    n_components = 300
    
    for method_name, method_func in methods.items():

        # cross subject
        args.test_only = True
        dataSet = MyDataset(args)
        X_train, y_train = dataSet.get_train_data()
        X_test, y_test = dataSet.get_test_data()
        
        len_train = X_train.shape[0]
        len_test = X_test.shape[0]
        
        start_time = time.time()
        X_train_feat = method_func(X_train)
        # print(X_train_feat[0])
        feat_extraction_time = (time.time() - start_time) / len_train
        feat_extraction_time = round(feat_extraction_time*1000,2)
        feat_dim_before_pca = X_train_feat.shape[1]
        
        X_test_feat = method_func(X_test)


        pca_time = 0
        # dim = X_train.shape[0]
        # reduce_dim = PCA(n_components=n_components)
        # # reduce_dim = FastICA(n_components=n_components,random_state=12,max_iter=400)
        # reduce_dim.fit(X_train_feat)
        # X_combined = np.vstack((X_train_feat, X_test_feat))
        # start_time = time.time()
        # X_combined_transformed = reduce_dim.transform(X_combined)
        # pca_time = (time.time() - start_time) / (len_train+len_test)
        # pca_time = round(pca_time*1000,2)
        # X_train_feat = X_combined_transformed[:dim,:]
        # X_test_feat = X_combined_transformed[dim:,:]
        

        # print(f"feat_dim_after:{X_train_feat.shape[1]}")
        start_time = time.time()
        model.fit(X_train_feat, y_train)
        training_time = (time.time() - start_time) / (len_train)
        training_time = round(training_time*1000,2)
        
        
        # X_train_feat = myMFCC.reduce_dimension(X_train_feat)
        # X_test_feat = myMFCC.reduce_dimension(X_test_feat)
        start_time = time.time()
        y_pred = model.predict(X_test_feat)
        prediction_time = (time.time() - start_time) / len(X_test)
        prediction_time = round(prediction_time*1000,2)
        
        
        total_time = feat_extraction_time+pca_time+prediction_time
        cross_acc = accuracy_score(y_test, y_pred)
        cross_acc = round(cross_acc,4)*100
        feat_dim = X_train_feat.shape[1]

        print(f"#{method_name}# \n cross_acc:{cross_acc}")
        results.append([method_name,cross_acc,feat_dim,feat_extraction_time,total_time])
        # results.append([method_name,cross_acc,feat_dim_before_pca,feat_dim,feat_extraction_time,pca_time,training_time,total_time])

    df = pd.DataFrame(results, columns=['特征表示方法','准确率(%)','特征维度','特征提取用时(ms)','总用时(ms)'])
    # df = pd.DataFrame(results, columns=['特征表示方法','准确率','特征维度','特征提取用时(ms)','PCA Time','Training Time','总用时(ms)'])
    df.to_csv('results.csv', index=False)
    
    print("数据已写入 CSV 文件。")
    
    return

def param():
    
    param_name = 'numcep' # 必须与args的参数名严格一致
    task_name = param_name+'_param_test'
    # param_list = np.arange(0.2, 1.6, 0.1, dtype=float) # backward
    
    # param_list = np.arange(0.01, 0.16, 0.01, dtype=float) # forward
    # param_list = np.linspace(0.01, 0.15, num=15, endpoint=True) # forward
    # param_list = [0.6,0.65, 0.7, 0.725, 0.75, 0.775, 0.8,0.9]
    # param_list = [0.6,0.7, 0.73, 0.76,0.8,0.9]
    
    # param_list = [64,128,256, 512, 1024,2048] # nfft
    # param_list = np.linspace(0.025, 0.5, num=5, endpoint=True) # winlen
    # param_list = np.arange(1, 20, 2, dtype=int) # nfilt
    param_list = np.arange(1, 15, 1, dtype=int) # numcep
    logger.info(f"{task_name}: {param_list}")

    times = []
    accuracies = []
    
    results = []
    
    with tqdm(total=len(param_list)) as pbar:
        for param in param_list:
            setattr(args, param_name, param)
            
            dataSet = MyDataset(args)
            model = SVC(kernel='linear', probability=True, C=0.1)
            X_train, y_train = dataSet.get_train_data()
            X_test, y_test = dataSet.get_test_data()
            myMFCC = MyMFCC(args)
            X_train_feat = myMFCC.get_feat(X_train)
            # print(f"feat_dim:{X_train_feat.shape}")
            model.fit(X_train_feat, y_train)
            test_len = X_test.shape[0]
            start_time = time.time()
            X_test_feat = myMFCC.get_feat(X_test)
            y_pred = model.predict(X_test_feat)
            
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            results.append([param, round(acc,2),round(precision,2),round(recall,2),round(f1,2)])
            
            end_time = time.time()
            one_sample_time = (end_time - start_time)/test_len
            accuracies.append(round(acc,5)*100)
            times.append(round(one_sample_time*1000,5))

            pbar.set_description(f'Accuracy: {acc:.4f}, {param_name}={param}')
            pbar.update(1)

        
    logger.info(f"accuracies: {accuracies}")
    logger.info(f"times: {times}")
    # plt.rcParams['font.family'] = 'SimHei'  # 使用黑体字
    fig, ax1 = plt.subplots()

    # color = 'tab:blue'
    color = 'k'
    # ax1.set_xlabel('Param Values')
    # ax1.set_xlabel('nfilt')
    ax1.set_ylabel('Accuracy(%)',color=color)
    ax1.plot(param_list, accuracies, marker='s', color='tab:blue')
    ax1.tick_params(axis='y',labelcolor=color)
    
    # color = 'tab:red'
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Time (ms)', color=color)
    # ax2.plot(param_list, times, marker='o', color=color)
    # ax2.tick_params(axis='y', labelcolor=color)

    # fig.tight_layout()
    # plt.subplots_adjust(top=0.85)
    # plt.savefig('./img/backward_accuracy&time.png')
    plt.xticks(param_list, [str(p) for p in param_list])
    plt.savefig('./img/'+task_name+'.png')
    plt.show()
    
    # df = pd.DataFrame(results, columns=['Value', 'Accuracy','Precision','Recall','F1-score'])
    # df.to_csv('result_param.csv', index=False)
    
    # print("数据已写入 CSV 文件。")

    return

def test():
    dataSet = MyDataset(args)
    model = SVC(kernel='linear', probability=True, C=0.1)
    X_train, y_train = dataSet.get_train_data()
    X_test, y_test = dataSet.get_test_data()
    myMFCC = MyMFCC(args)

    X_train_feat = myMFCC.get_feat(X_train)
    model.fit(X_train_feat, y_train)
    # model = joblib.load('./model/train_model.pkl')
    # print(f"Completed Train Section")
    
    X_test_feat = myMFCC.get_feat(X_test)
    y_pred = model.predict(X_test_feat)
    y_prob = model.predict_proba(X_test_feat)[:, 1]
    # classification_report(y_test,y_pred,digits=4)
    logger.debug(f"Predict Classification Report:\n{classification_report(y_test, y_pred, digits=4)}")
    acc = accuracy_score(y_test, y_pred)
    print(f"acc: {acc}")
    visualize_evaluation(y_test, y_prob, y_pred, save_dir='./img')
    
    return

def diff_obj():
    
    # model = SVC(kernel='linear', probability=True, C=0.1)
    model = SVC(kernel='poly', gamma='auto' ,probability=True, C=0.1)
    myMFCC = MyMFCC(args)

    methods = {
        # 'get_x_mfcc_feat': myMFCC.get_x_mfcc_feat,
        # 'get_y_mfcc_feat': myMFCC.get_y_mfcc_feat,
        
        # 'get_z_mfcc_feat': myMFCC.get_z_mfcc_feat,
        # 'get_concat_mfcc_feat': myMFCC.get_concat_mfcc_feat,
        # 'get_add_mfcc_feat': myMFCC.get_add_mfcc_feat,
        # 'get_dft321_mfcc_feat': myMFCC.get_dft321_mfcc_feat,
        
        # 'get_x_stft_feat': myMFCC.get_x_stft_feat,
        # 'get_y_stft_feat': myMFCC.get_y_stft_feat,
        
        # 'get_z_stft_feat': myMFCC.get_z_stft_feat,
        # 'get_concat_stft_feat': myMFCC.get_concat_stft_feat,
        'get_add_stft_feat': myMFCC.get_add_stft_feat,
        'get_dft321_stft_feat':myMFCC.get_dft321_stft_feat,
        
        # 'get_wavelet_feat': myMFCC.get_wavelet_feat,
        # 'get_dft321_wavelet_feat':myMFCC.get_dft321_wavelet_feat
    }
    results = []
    n_components = 100
    accuracy_list = []
    for method_name, method_func in methods.items():
        rand_list = generate_random_integers(args.repeat_num,1,100,seed=41)
        # rand_list = generate_random_integers(1,1,100,seed=41)
        acc_list = []
        for rand in rand_list:
            args.random_state = rand
            dataSet = MultiDataset(args)

            X_train, y_train = dataSet.get_train_data()
            X_test, y_test = dataSet.get_test_data()
            X_train_feat = method_func(X_train)
            X_test_feat = method_func(X_test)
            # print(f"train.shape:{X_train.shape}")
            # print(f"test.shape:{X_test.shape}")
            


            # dim = X_train.shape[0]
            # reduce_dim = PCA(n_components=n_components)
            # reduce_dim.fit(X_train_feat)
            # X_combined = np.vstack((X_train_feat, X_test_feat))
            # X_combined_transformed = reduce_dim.transform(X_combined)
            # X_train_feat = X_combined_transformed[:dim,:]
            # X_test_feat = X_combined_transformed[dim:,:]


            # print(f"feat_dim_after:{X_train_feat.shape[1]}")
            model.fit(X_train_feat, y_train)
            y_pred = model.predict(X_test_feat)
            acc = accuracy_score(y_test, y_pred)
            acc_list.append(acc)
        
        accuracy = round(np.mean(acc_list)*100,2)
        # accuracy_list.append(accuracy)
        # accuracy_list.insert(0, method_name)

    
        print(f"#{method_name}# \n Accuracy: {accuracy} \n")
        # print(f"{len(accuracy_list)}")
        results.append([method_name,accuracy])

    # df = pd.DataFrame(results, columns=['Method', 'yellow_cup','white_cup_user2','box','velcro','badminton','ping_pong','toilet_roll'])
    df = pd.DataFrame(results, columns=['Method', 'Accuracy'])
    df.to_csv('results_diff_obj.csv', index=False)
    
    print("数据已写入 CSV 文件。")
    
    return


def diff_obj_param():
    
    param_name = 'nfilt' # 必须与args的参数名严格一致
    task_name = param_name+'_param_test_diff'
    # param_list = np.linspace(0.4, 1.5, num=15, endpoint=True) # backward
    # param_list = np.linspace(0.02, 0.6, num=30, endpoint=True) # forward
    # param_list = [0.6,0.65, 0.7, 0.725, 0.75, 0.775, 0.8,0.9]
    # param_list = [0.6,0.7, 0.73, 0.76,0.8,0.9]
    
    # param_list = [64,128,256, 512, 1024,2048] # nfft
    # param_list = np.linspace(0.025, 0.5, num=5, endpoint=True) # winlen
    param_list = np.arange(1, 20, 1, dtype=int) # nfilt
    
    # param_list = np.arange(1, 17, 2, dtype=int) # numcep
    logger.info(f"{task_name}: {param_list}")

    times = []
    accuracies = []
    
    with tqdm(total=len(param_list)) as pbar:
        for param in param_list:
            setattr(args, param_name, param)
            random_numbers = generate_random_integers(args.repeat_num, 1, 100,seed=41)
            acc_list = []
            time_list = []
            for random_num in random_numbers:
                args.random_state= random_num
                dataSet = MultiDataset(args)
                model = SVC(kernel='linear', probability=True, C=0.1)
                X_train, y_train = dataSet.get_train_data()
                X_test, y_test = dataSet.get_test_data()
                myMFCC = MyMFCC(args)

                X_train_feat = myMFCC.get_feat(X_train)
                model.fit(X_train_feat, y_train)
                # model = joblib.load('./model/train_model.pkl')
                # print(f"Completed Train Section")
                test_len = X_test.shape[0]
                start_time = time.time()
                X_test_feat = myMFCC.get_feat(X_test)
                y_pred = model.predict(X_test_feat)
                y_prob = model.predict_proba(X_test_feat)[:, 1]
                acc = accuracy_score(y_test, y_pred)

                end_time = time.time()
                one_sample_time = (end_time - start_time)/test_len
                time_list.append(one_sample_time)
                acc_list.append(acc)

            mean_acc= round(np.mean(acc_list)*100,2)
            mean_time = round(np.mean(time_list)*1000,5)
            accuracies.append(mean_acc)
            times.append(mean_time)

            pbar.set_description(f'Accuracy: {acc:.4f}, {param_name}={param}')
            pbar.update(1)

        
    logger.info(f"accuracies: {accuracies}")
    logger.info(f"times: {times}")
    fig, ax1 = plt.subplots()

    # color = 'tab:blue'
    color = 'k'
    # ax1.set_xlabel('Param Values')
    ax1.set_ylabel('Accuracy(%)',color=color)
    ax1.plot(param_list, accuracies, marker='s', color='tab:blue')
    ax1.tick_params(axis='y',labelcolor=color)
    
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
    # plt.show()

    return

def svm_grid_search():
    # 加载数据集
    myMFCC = MyMFCC(args)

    methods = {
        # 'get_x_mfcc_feat': myMFCC.get_x_mfcc_feat,
        # 'get_x_stft_feat': myMFCC.get_x_stft_feat,
        # 'get_y_mfcc_feat': myMFCC.get_y_mfcc_feat,
        # 'get_y_stft_feat': myMFCC.get_y_stft_feat,
        'get_z_mfcc_feat': myMFCC.get_z_mfcc_feat,
        # 'get_z_stft_feat': myMFCC.get_z_stft_feat,
        # 'get_add_mfcc_feat': myMFCC.get_add_mfcc_feat,
        # 'get_add_stft_feat': myMFCC.get_add_stft_feat,
        # 'get_concat_mfcc_feat': myMFCC.get_concat_mfcc_feat,
        # 'get_concat_stft_feat': myMFCC.get_concat_stft_feat,
        # 'get_dft321_mfcc_feat': myMFCC.get_dft321_mfcc_feat,
        # 'get_dft321_stft_feat':myMFCC.get_dft321_stft_feat,
        # 'get_wavelet_feat': myMFCC.get_wavelet_feat,
        # 'get_dft321_wavelet_feat':myMFCC.get_dft321_wavelet_feat
    }
    results = []
    n_components = 100
    accuracy_list = []
    # dataSet = MyDataset(args)
    dataSet = MultiDataset(args)
    
    for method_name, method_func in methods.items():
        

        X_train, y_train = dataSet.get_train_data()
        X_test, y_test = dataSet.get_test_data()
        X_train_feat = method_func(X_train)
        X_test_feat = method_func(X_test)
        model = SVC()
        # param_grid = {
        #     'C': [0.1, 1, 10, 100],
        #     'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        #     'gamma': ['scale', 'auto']
        # }
        param_grid = {
            'C': [0.1],
            'kernel': ['poly'],
            'gamma': ['auto']
        }

        # 使用GridSearchCV进行网格搜索
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train_feat, y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_params_
        # 输出最佳参数和对应的准确率
        print("Best parameters found: ", grid_search.best_params_)
        print("Best accuracy found: ", grid_search.best_score_)

        # 在测试集上评估模型性能
        best_model = grid_search.best_estimator_

        test_accuracy = best_model.score(X_test_feat, y_test)
        print("Test accuracy: ", test_accuracy)
        results.append([method_name,best_params,best_score,test_accuracy])

        
        
        

    # df = pd.DataFrame(results, columns=['Method', 'yellow_cup','white_cup_user2','box','velcro','badminton','ping_pong','toilet_roll'])
    df = pd.DataFrame(results, columns=['Method', 'Best Params','Best Score','Test Accuracy'])
    df.to_csv('results_svm_grid_search.csv', index=False)
    
    print("数据已写入 CSV 文件。")
    
    return



if __name__ == "__main__":
    logger.debug("Arguments: %s", sys.argv[1:])

    # param() # ./test.sh
    # diff_obj_param()
    
    # feat_method()
    # svm_grid_search()
    
    # test()
    diff_obj()