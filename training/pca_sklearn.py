from sklearn import decomposition
import numpy as np
# import function as fun
import matplotlib.pyplot as plt
from modules_training import *
from sklearn.metrics import accuracy_score
import os

# PCA
PCA_C = 10
PCA_C_DEGREE = 2
PCA_C_GAMMA = 10
PCA_C_KERNEL = 'poly'    # 'rbf' or 'linear' or 'poly'

X_train = np.loadtxt('../../output_data_train/data.txt')
X_test = np.loadtxt('../../output_data_test/data.txt')

n = 50

pca = decomposition.PCA(n_components=n)

pca_train = pca.fit_transform(X_train.T)
# pca = decomposition.PCA(n_components=n)
pca_test = pca.fit_transform(X_test.T)

PATH_DATA_TRAIN = "../../output_data_train"
PATH_DATA_TEST = "../../output_data_test"

X_HOG_train, X_PCA_train, Y_train = load_feature(PATH_DATA_TRAIN)
X_HOG_test, X_PCA_test, Y_test = load_feature(PATH_DATA_TEST)



# SVM - PCA
model = SVM_fn(pca_train, Y_train, "./SVM_PCA_", c_kernel=PCA_C_KERNEL, c_degree=PCA_C_DEGREE, c_gamma=PCA_C_GAMMA, c=PCA_C)
Y_predict = model.predict(pca_test)
accuracy = accuracy_score(Y_test, Y_predict)
print("SVM - PCA: {:.2%}".format(accuracy))
# result_SVM_PCA = [str(x)+" " for x in [accuracy*100, PCA_C_KERN