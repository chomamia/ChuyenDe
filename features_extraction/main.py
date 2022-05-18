import numpy as np
import matplotlib.pyplot as plt
#pca_sklearn
# X_train=np.loadtxt("D:/Chuyen_de/Dataset/pca_sklearn/X_train.txt",delimiter=",")
# X_test=np.loadtxt("D:/Chuyen_de/Dataset/pca_sklearn/X_test.txt",delimiter=",")
#pca_sklearn_all
# X_train=np.loadtxt("D:/Chuyen_de/Dataset/pca_sklearn/fix/X_train1.txt",delimiter=",")
# X_test=np.loadtxt("D:/Chuyen_de/Dataset/pca_sklearn/fix/X_test1.txt",delimiter=",")
#pca
X_train=np.loadtxt("D:/Chuyen_de/Dataset/X_train.txt",delimiter=",")
X_test=np.loadtxt("D:/Chuyen_de/Dataset/X_test.txt",delimiter=",")
#hog
# X_train=np.loadtxt("D:/Chuyen_de/Dataset/hog/X_train_1.txt",dtype=float)
# X_test=np.loadtxt("D:/Chuyen_de/Dataset/hog/X_test_1.txt",dtype=float)

y_train=np.loadtxt("D:/Chuyen_de/Dataset/pca_sklearn/y_train1.txt",dtype=float,delimiter=",")
y_test=np.loadtxt("D:/Chuyen_de/Dataset/pca_sklearn/y_test1.txt",dtype=float,delimiter=",")

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svm = SVC(kernel='rbf', degree =10, gamma=10, C = 10,probability=True) #poly accuracy 74%
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy is: ', accuracy*100,"%")