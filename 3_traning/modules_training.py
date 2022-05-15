import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def load_feature(path):
    X_HOG = np.loadtxt(path + "/HOG_feature.txt")
    X_PCA = np.loadtxt(path + "/PCA_feature.txt")
    Y = np.loadtxt(path + "/label.txt")
    print("Done!!!")
    return X_HOG, X_PCA, Y


def SVM_fn(X_train, Y_train, X_test, Y_test):
    svm = SVC(kernel='poly', degree=2, gamma=10, C=10, probability=True)
    svm.fit(X_train, Y_train)
    Y_predict = svm.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_predict)
    return Y_predict, accuracy
