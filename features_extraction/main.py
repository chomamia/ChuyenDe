from modules_features_extraction import *

N_COMPONENT = 50  # using for PCA

# # # train # # #
PATH_DATA = "../../dataset/data_argument_(80.0-20.0)/train/gray"
PATH_SAVE_MODEL = "../../output_data_train"
# # # test # # #
# PATH_DATA = "../../dataset/data_argument_(80.0-20.0)/test/gray"
# PATH_SAVE_MODEL = "../../output_data_test"

if __name__ == "__main__":
    if not os.path.exists(PATH_SAVE_MODEL):
        os.mkdir(PATH_SAVE_MODEL)

    data, label = load_data(PATH_DATA)
    np.savetxt(PATH_SAVE_MODEL + "/data.txt", data)
    np.savetxt(PATH_SAVE_MODEL + "/label.txt", label)

    PCA_feature = PCA_fn(data, N_COMPONENT)
    np.savetxt(PATH_SAVE_MODEL + "/PCA_feature.txt", PCA_feature)

    HOG_feature = HOG_fn(data)
    np.savetxt(PATH_SAVE_MODEL + "/HOG_feature.txt", HOG_feature)

    print("------- All done!!!-------- \nData and Features saved at: ", os.path.abspath(PATH_SAVE_MODEL))
