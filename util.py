import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import random
import matplotlib

from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# matplotlib.getLogger().setLevel(logging.WARNING)
matplotlib.pyplot.set_loglevel (level = 'warning')


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
        
def generate_random_integers(count, min_value=1, max_value=100,seed=41):
    # print(f"generate_random_int:{seed}")
    random.seed(seed)
    if count <= 0:
        raise ValueError("生成数量必须是正整数")
    if min_value < 1 or max_value < 1:
        raise ValueError("最小值和最大值必须为正整数")

    random_integers = [random.randint(min_value, max_value) for _ in range(count)]
    return random_integers

def visualize_evaluation(y_true, y_prob, y_pred, save_dir=None):
    """
    Visualize ROC curve and Confusion Matrix.

    Parameters:
    - y_true: true labels
    - y_prob: predicted probabilities
    - y_pred: predicted labels
    - save_dir: directory to save the visualizations (optional)
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    # Save ROC curve if save_dir is provided
    if save_dir:
        plt.savefig(save_dir + '/roc_curve.png')
    else:
        plt.show()

    # Calculate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot Confusion Matrix heatmap
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    # Save Confusion Matrix if save_dir is provided
    if save_dir:
        plt.savefig(save_dir + '/confusion_matrix.png')
    else:
        plt.show()

def grid_search():
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
