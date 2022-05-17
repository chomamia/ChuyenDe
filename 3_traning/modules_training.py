
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def load_feature(path):
    X_HOG = np.loadtxt(path + "/HOG_feature.txt")
    X_PCA = np.loadtxt(path + "/PCA_feature.txt")
    Y = np.loadtxt(path + "/label.txt")
    print("Load feature done!!!")
    return X_HOG, X_PCA, Y


def SVM_fn(X_train, Y_train, X_test, Y_test, 
            c_kernel='poly', c_degree=2, c_gamma=10, c=10):
    print("\nTraining ...")
    print("kernel = ", c_kernel)
    svm = SVC(kernel=c_kernel, degree=c_degree, gamma=c_gamma, C=c, probability=True)
    svm.fit(X_train, Y_train)
    Y_predict = svm.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_predict)
    return Y_predict, accuracy

def save_result_train(path, result):
    if not os.path.exists(path):
        os.mkdir(path)

    file_history = open("./" + path + "/history_train.txt", "a")

    file_history.writelines(result)

    file_history.close()