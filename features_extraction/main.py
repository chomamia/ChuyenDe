from modules_features_extraction import *

N_COMPONENT = 50  # using for PCA

# # # train # # #
PATH_DATA_TRAIN = "../../dataset/data_argument_(80_20)/train/gray"
PATH_SAVE_MODEL_TRAIN = "../../output_data_train"
# # # test # # #
PATH_DATA_TEST = "../../dataset/data_argument_(80_20)/test/gray"
PATH_SAVE_MODEL_TEST = "../../output_data_test"

if __name__ == "__main__":
    
    # load data
    if not os.path.exists(PATH_SAVE_MODEL_TRAIN):
        os.mkdir(PATH_SAVE_MODEL_TRAIN)

    data_train, label_train = load_data(PATH_DATA_TRAIN)
    np.savetxt(PATH_SAVE_MODEL_TRAIN + "/data.txt", data_train)
    np.savetxt(PATH_SAVE_MODEL_TRAIN + "/label.txt", label_train)

    if not os.path.exists(PATH_SAVE_MODEL_TEST):
        os.mkdir(PATH_SAVE_MODEL_TEST)

    data_test, label_test = load_data(PATH_DATA_TEST)
    np.savetxt(PATH_SAVE_MODEL_TEST + "/data.txt", data_test)
    np.savetxt(PATH_SAVE_MODEL_TEST + "/label.txt", label_test)
    
    # TRAIN

    PCA_feature = PCA_fn(data_train, N_COMPONENT)
    np.savetxt(PATH_SAVE_MODEL_TRAIN + "/PCA_feature.txt", PCA_feature)

    HOG_feature = HOG_fn(data_train)
    np.savetxt(PATH_SAVE_MODEL_TRAIN + "/HOG_feature.txt", HOG_feature)

    # TEST

    PCA_feature = PCA_fn(data_test, N_COMPONENT)
    np.savetxt(PATH_SAVE_MODEL_TEST + "/PCA_feature.txt", PCA_feature)

    HOG_feature = HOG_fn(data_test)
    np.savetxt(PATH_SAVE_MODEL_TEST + "/HOG_feature.txt", HOG_feature)

    # PCA all
    data = np.append(data_train,data_test, axis=0)

    PCA_feature_all = PCA_all(data, N_COMPONENT)
    length = PCA_feature_all.shape[0]

    PCA_feature_train = PCA_feature_all[:int(length*0.8)][:]
    PCA_feature_test = PCA_feature_all[int(length*0.8):][:]

    np.savetxt(PATH_SAVE_MODEL_TRAIN + "/PCA_feature_all.txt", PCA_feature_train)
    np.savetxt(PATH_SAVE_MODEL_TEST + "/PCA_feature_all.txt", PCA_feature_test)

    print("------- All done!!!-------- \nData and Features saved at: \nTrain:{} \nTest:{}".format(os.path.abspath(PATH_SAVE_MODEL_TRAIN),os.path.abspath(PATH_SAVE_MODEL_TEST)))
