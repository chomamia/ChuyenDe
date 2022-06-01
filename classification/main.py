from modules_classification import *

INPUT_IMAGE = ""
# INPUT_FEATURE = 

""" FEATURE_EXTRACTION_METHOD
    1: HOG (Histogram of Oriented Gradient)
    2: PCA (Principal Component Analysis)
"""
SELECT_FEATURE_EXTRACTION_METHOD = 1

""" CLASSIFICATION_METHOD
    1: SVM (Support Vector Machine)
    2: ANN (Artificial Neural Network)
"""
SELECT_CLASSIFICATION_METHOD = 1

PATH_FEATURES_DATA_TRAIN = "../output_data_train"
PATH_FEATURES_DATA_TEST = "../output_data_test"


if __name__ == '__main__':
    if SELECT_FEATURE_EXTRACTION_METHOD == 1:
        X_train = 
