import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from modules_training import *
from sklearn.metrics import accuracy_score

labels = ["Brocoli", "Carot", "Eggplant", "Potato", "Tomato"]

# HOG
HOG_C = 20
HOG_C_DEGREE = 2
HOG_C_GAMMA = 10
HOG_C_KERNEL = 'poly'    # 'rbf' or 'linear' or 'poly'

# PCA
PCA_C = 20
PCA_C_DEGREE = 2
PCA_C_GAMMA = 10
PCA_C_KERNEL = 'poly'    # 'rbf' or 'linear' or 'poly'

# ANN - HOG
HIDDEN_LAYER_1_SIZE = 128
HIDDEN_LAYER_2_SIZE = 64 
HIDDEN_LAYER_3_SIZE = 20 
NUM_LABELS = 5

# PATH
PATH_DATA_TRAIN = "../../output_data_train"
PATH_DATA_TEST = "../../output_data_test"
PATH_SAVE_RESULT_TRAIN = "../../result_train"
PATH_SAVE_MODEL = "../../model_trained"

if __name__ == "__main__":
    X_HOG_train, X_PCA_train, X_PCA_all_train, Y_train = load_feature(PATH_DATA_TRAIN)
    X_HOG_test, X_PCA_test,X_PCA_all_test, Y_test = load_feature(PATH_DATA_TEST)

    if not os.path.exists(PATH_SAVE_MODEL):
        os.mkdir(PATH_SAVE_MODEL)

    # SVM - HOG
    model = SVM_fn(X_HOG_train, Y_train, PATH_SAVE_MODEL+"/SVM_HOG", c_kernel=HOG_C_KERNEL, c_degree=HOG_C_DEGREE, c_gamma=HOG_C_GAMMA, c=HOG_C)
    Y_predict = model.predict(X_HOG_test)
    accuracy = accuracy_score(Y_test, Y_predict)
    print("SVM - HOG: {:.2%}".format(accuracy))
    result_SVM_HOG = [str(x)+" " for x in [accuracy*100, HOG_C_KERNEL, HOG_C_DEGREE, HOG_C_GAMMA, HOG_C]]

    y_predict = np.array(Y_predict)
    y_predict = y_predict.reshape(5,120)
    confusion_matrix = np.zeros((5,5))
    for i in range(0,5):
        for j in range(0,5):
            b = y_predict[i,:]
            a = len(b[b==j+1])
            confusion_matrix[i,j] = a
    print(confusion_matrix)

    #
    # # SVM - PCA
    model = SVM_fn(X_PCA_train, Y_train, PATH_SAVE_MODEL+"/SVM_PCA", c_kernel=PCA_C_KERNEL, c_degree=PCA_C_DEGREE, c_gamma=PCA_C_GAMMA, c=PCA_C)
    Y_predict = model.predict(X_PCA_test)
    accuracy = accuracy_score(Y_test, Y_predict)
    print("SVM - PCA: {:.2%}".format(accuracy))
    result_SVM_PCA = [str(x) + " " for x in [accuracy * 100, PCA_C_KERNEL, PCA_C_DEGREE, PCA_C_GAMMA, PCA_C]]

    y_predict = np.array(Y_predict)
    y_predict = y_predict.reshape(5,120)
    confusion_matrix = np.zeros((5,5))
    for i in range(0,5):
        for j in range(0,5):
            b = y_predict[i,:]
            a = len(b[b==j+1])
            confusion_matrix[i,j] = a
    print(confusion_matrix)

    #SVM - PCA-ALL
    model = SVM_fn(X_PCA_all_train, Y_train, PATH_SAVE_MODEL + "/SVM_PCA_ALL", c_kernel=PCA_C_KERNEL, c_degree=PCA_C_DEGREE,
                   c_gamma=PCA_C_GAMMA, c=PCA_C)
    Y_predict = model.predict(X_PCA_all_test)
    accuracy = accuracy_score(Y_test, Y_predict)
    print("SVM - PCA: {:.2%}".format(accuracy))


    result_SVM_PCA = [str(x) + " " for x in [accuracy * 100, PCA_C_KERNEL, PCA_C_DEGREE, PCA_C_GAMMA, PCA_C]]

    y_predict = np.array(Y_predict)
    y_predict = y_predict.reshape(5,120)
    confusion_matrix = np.zeros((5,5))
    for i in range(0,5):
        for j in range(0,5):
            b = y_predict[i,:]
            a = len(b[b==j+1])
            confusion_matrix[i,j] = a
    print(confusion_matrix)
    
    # ANN - HOG
    theta1, theta2, theta3, theta4, y_predict = ANN(X_HOG_train, Y_train, X_HOG_test, Y_test, 500,
                                        HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE,
                                        NUM_LABELS, path_save=PATH_SAVE_MODEL+"/ANN_HOG")
    min = np.min(X_PCA_train)
    max = np.max(X_PCA_train)
    X_PCA_train = (X_PCA_train - min) / (max - min)
    min = np.min(X_PCA_test)
    max = np.max(X_PCA_test)
    X_PCA_test = (X_PCA_test - min) / (max - min)

    y_predict = np.array(y_predict)
    y_predict = y_predict.reshape(5,120)
    confusion_matrix = np.zeros((5,5))
    for i in range(0,5):
        for j in range(0,5):
            b = y_predict[i,:]
            a = len(b[b==j])
            confusion_matrix[i,j] = a
    print(confusion_matrix)

    # ANN - PCA
    theta1, theta2, theta3, theta4, y_predict = ANN(X_PCA_train, Y_train, X_PCA_test, Y_test, 500,
                                        HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE,
                                        NUM_LABELS, path_save=PATH_SAVE_MODEL+"/ANN_PCA")

    y_predict = np.array(y_predict)
    y_predict = y_predict.reshape(5,120)
    confusion_matrix = np.zeros((5,5))
    for i in range(0,5):
        for j in range(0,5):
            b = y_predict[i,:]
            a = len(b[b==j])
            confusion_matrix[i,j] = a
    print(confusion_matrix)

    min = np.min(X_PCA_all_train)
    max = np.max(X_PCA_all_train)
    X_PCA_all_train = (X_PCA_all_train - min) / (max - min)
    min = np.min(X_PCA_all_test)
    max = np.max(X_PCA_all_test)
    X_PCA_all_test = (X_PCA_all_test - min) / (max - min)

    # ANN - PCA
    HIDDEN_LAYER_1_SIZE = 40
    HIDDEN_LAYER_2_SIZE = 30
    HIDDEN_LAYER_3_SIZE = 20
    NUM_LABELS = 5

    #ANN - PCA ALL
    theta1, theta2, theta3, theta4, y_predict = ANN(X_PCA_all_train, Y_train, X_PCA_all_test, Y_test, 500,
                                                    HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE,
                                                    NUM_LABELS, path_save=PATH_SAVE_MODEL + "/ANN_PCA_ALL")
    y_predict = np.array(y_predict)
    y_predict = y_predict.reshape(5,120)
    confusion_matrix = np.zeros((5,5))
    for i in range(0,5):
        for j in range(0,5):
            b = y_predict[i,:]
            a = len(b[b==j])
            confusion_matrix[i,j] = a
    print(confusion_matrix)
    # save_result_train(PATH_SAVE_RESULT_TRAIN, result_SVM_HOG, result_SVM_PCA)