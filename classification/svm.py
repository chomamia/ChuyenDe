import random
import numpy as np
import matplotlib.pyplot as plt

svm = SVC(kernel='linear', degree =10, gamma=100, C = 100,probability=True) #poly accuracy 74%
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


#load data
# X_train=np.loadtxt("D:/Chuyen_de/Dataset/X_train.txt",delimiter=",")
# X_test=np.loadtxt("D:/Chuyen_de/Dataset/X_test.txt",delimiter=",")
# y_train=np.loadtxt("D:/Chuyen_de/Dataset/y_train.txt",dtype=int,delimiter=",")
# y_test=np.loadtxt("D:/Chuyen_de/Dataset/y_test.txt",dtype=int,delimiter=",")

#
# # pca_sklearn
# # X_train=np.loadtxt("D:/Chuyen_de/Dataset/pca_sklearn/X_train.txt",delimiter=",")
# # X_test=np.loadtxt("D:/Chuyen_de/Dataset/pca_sklearn/X_test.txt",delimiter=",")
#
# # pca_sklearn_all
# # X_train=np.loadtxt("D:/Chuyen_de/Dataset/pca_sklearn/fix/X_train1.txt",delimiter=",")
# # X_test=np.loadtxt("D:/Chuyen_de/Dataset/pca_sklearn/fix/X_test1.txt",delimiter=",")
#
# # pca
# # X_train=np.loadtxt("D:/Chuyen_de/Dataset/X_train.txt",delimiter=",")
# # X_test=np.loadtxt("D:/Chuyen_de/Dataset/X_test.txt",delimiter=",")
#
# # hog
# X_train=np.loadtxt("D:/Chuyen_de/Dataset/hog/X_train.txt",dtype=float,delimiter=",")
# X_test=np.loadtxt("D:/Chuyen_de/Dataset/hog/X_test.txt",dtype=float,delimiter=",")
# y_train=np.loadtxt("D:/Chuyen_de/Dataset/pca_sklearn/y_train1.txt",dtype=float,delimiter=",")
# y_test=np.loadtxt("D:/Chuyen_de/Dataset/pca_sklearn/y_test1.txt",dtype=float,delimiter=",")
#
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)
#

# print('Model accuracy is: ', accuracy*100,"%")
