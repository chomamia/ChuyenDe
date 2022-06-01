from modules_features_extraction import *

N_COMPONENT = 50  # using for PCA

# # # train # # #
PATH_DATA_TRAIN = "D:/Chuyen_de/Dataset/gray/train"
PATH_SAVE_MODEL_TRAIN = "../../output_data_train"
# # # test # # #
PATH_DATA_TEST = "D:/Chuyen_de/Dataset/gray/test"
PATH_SAVE_MODEL_TEST = "../../output_data_test"

if __name__ == "__main__":
    # TRAIN
    print("===========\nData train...")
    if not os.path.exists(PATH_SAVE_MODEL_TRAIN):
        os.mkdir(PATH_SAVE_MODEL_TRAIN)

    data, label = load_data(PATH_DATA_TRAIN)
    np.savetxt(PATH_SAVE_MODEL_TRAIN + "/data.txt", data)
    np.savetxt(PATH_SAVE_MODEL_TRAIN + "/label.txt", label)

    PCA_feature = PCA_fn(data, N_COMPONENT)
    np.savetxt(PATH_SAVE_MODEL_TRAIN + "/PCA_feature.txt", PCA_feature)

    HOG_feature = HOG_fn(data)
    np.savetxt(PATH_SAVE_MODEL_TRAIN + "/HOG_feature.txt", HOG_feature)

    # TEST
    print("===========\nData test...")
    if not os.path.exists(PATH_SAVE_MODEL_TEST):
        os.mkdir(PATH_SAVE_MODEL_TEST)

    data, label = load_data(PATH_DATA_TEST)
    np.savetxt(PATH_SAVE_MODEL_TEST + "/data.txt", data)
    np.savetxt(PATH_SAVE_MODEL_TEST + "/label.txt", label)

    PCA_feature = PCA_fn(data, N_COMPONENT)
    np.savetxt(PATH_SAVE_MODEL_TEST + "/PCA_feature.txt", PCA_feature)

    HOG_feature = HOG_fn(data)
    np.savetxt(PATH_SAVE_MODEL_TEST + "/HOG_feature.txt", HOG_feature)


    print("------- All done!!!-------- \nData and Features saved at: \nTrain:{} \nTest:{}".format(os.path.abspath(PATH_SAVE_MODEL_TRAIN),os.path.abspath(PATH_SAVE_MODEL_TEST)))
