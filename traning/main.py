from unittest import result
from modules_training import *

# HOG
HOG_C = 20
HOG_C_DEGREE = 2
HOG_C_GAMMA = 10
HOG_C_KERNEL = 'poly'    # 'rbf' or 'linear' or 'poly'

# PCA
PCA_C = 20
PCA_C_DEGREE = 1
PCA_C_GAMMA = 10
PCA_C_KERNEL = 'rbf'    # 'rbf' or 'linear' or 'poly'

# PATH
PATH_DATA_TRAIN = "../../output_data_train"
PATH_DATA_TEST = "../../output_data_test"
PATH_SAVE_RESULT_TRAIN = "../../result_train"

if __name__ == "__main__":
    X_HOG_train, X_PCA_train, Y_train = load_feature(PATH_DATA_TRAIN)
    X_HOG_test, X_PCA_test, Y_test = load_feature(PATH_DATA_TEST)

    svm_hog_predict, svm_hog_accuracy = SVM_fn(X_HOG_train, Y_train, X_HOG_test, Y_test, 
                                               c_kernel=HOG_C_KERNEL, c_degree=HOG_C_DEGREE, c_gamma=HOG_C_GAMMA, c=HOG_C)
    print("SVM - HOG: {:.2%}".format(svm_hog_accuracy))

    svm_pca_predict, svm_pca_accuracy = SVM_fn(X_PCA_train, Y_train, X_PCA_test, Y_test, 
                                               c_kernel=PCA_C_KERNEL, c_degree=PCA_C_DEGREE, c_gamma=PCA_C_GAMMA, c=PCA_C)
    print("SVM - PCA: {:.2%}".format(svm_pca_accuracy))

    result_SVM_HOG = [str(x)+" " for x in [svm_hog_accuracy*100, HOG_C_KERNEL, HOG_C_DEGREE, HOG_C_GAMMA, HOG_C]]
    result_SVM_PCA = [str(x)+" " for x in [svm_pca_accuracy*100, PCA_C_KERNEL, PCA_C_DEGREE, PCA_C_GAMMA, PCA_C]]
    save_result_train(PATH_SAVE_RESULT_TRAIN, result_SVM_HOG, result_SVM_PCA)