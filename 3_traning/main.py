from modules_training import *

PATH_DATA_TRAIN = "../output_data_train"
PATH_DATA_TEST = "../output_data_test"

if __name__ == "__main__":
    X_HOG_train, X_PCA_train, Y_train = load_feature(PATH_DATA_TRAIN)
    X_HOG_test, X_PCA_test, Y_test = load_feature(PATH_DATA_TEST)

    svm_hog_predict, svm_hog_accuracy = SVM_fn(X_HOG_train, Y_train, X_HOG_test, Y_test)
    print("SVM - HOG: {:.2%}".format(svm_hog_accuracy))

    svm_pca_predict, svm_pca_accuracy = SVM_fn(X_PCA_train, Y_train, X_PCA_test, Y_test)
    print("SVM - PCA: {:.2%}".format(svm_pca_accuracy))

