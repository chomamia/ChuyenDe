from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt


# def load_lables(num):
#     lables=np.zeros((num,5))
#     for a in range(0,5):
#         lables[a*num:a*num+num,:]=

def SVM_fn():
    svm = SVC(kernel='linear', degree=10, gamma=100, C=100, probability=True)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)