import os
from datetime import datetime
from modules_pre_processing import *

TRAIN_RATIO = 0.8
BUFFER_IMAGE = 50   # white border around image
SIZE_IMAGE = [64, 64]
PATH_DATA_ORIGIN = "../../dataset/data_origin"
PATH_DATA_ARGUMENT = "../../dataset/data_argument"

if __name__ == "__main__":
    # chia data train va test
    PATH_DATA_ARGUMENT = PATH_DATA_ARGUMENT + "_(" + str(int(TRAIN_RATIO*100)) + "_" + str(int(100-TRAIN_RATIO*100)) + ")"
    if not os.path.exists(PATH_DATA_ARGUMENT):
        os.mkdir(PATH_DATA_ARGUMENT)
    else:
        PATH_DATA_ARGUMENT = PATH_DATA_ARGUMENT + "_" + str(datetime.now()).replace(":", "_")
        os.mkdir(PATH_DATA_ARGUMENT)

    os.mkdir(PATH_DATA_ARGUMENT + "/train")
    os.mkdir(PATH_DATA_ARGUMENT + "/test")
    os.mkdir(PATH_DATA_ARGUMENT + "/train/gray")
    os.mkdir(PATH_DATA_ARGUMENT + "/train/rgb")
    os.mkdir(PATH_DATA_ARGUMENT + "/test/gray")
    os.mkdir(PATH_DATA_ARGUMENT + "/test/rgb")

    path_data = os.listdir(PATH_DATA_ORIGIN)

    progress = 1

    for class_image in path_data:
        print("loading: " + str(progress/len(path_data)*100) + "%")
        path_image = os.listdir(PATH_DATA_ORIGIN+"/"+class_image)
        num_image = len(path_image)
        num_train = int(TRAIN_RATIO*num_image)
        index_image_random = np.random.permutation(num_image)

        os.mkdir(PATH_DATA_ARGUMENT + "/train/gray/" + class_image)
        os.mkdir(PATH_DATA_ARGUMENT + "/train/rgb/" + class_image)
        os.mkdir(PATH_DATA_ARGUMENT + "/test/gray/" + class_image)
        os.mkdir(PATH_DATA_ARGUMENT + "/test/rgb/" + class_image)

        for i in index_image_random[:num_train]:
            img = cv2.imread(PATH_DATA_ORIGIN+"/"+class_image+"/"+path_image[i])
            img_resize = resize_fn(crop_fn(img, BUFFER_IMAGE), SIZE_IMAGE)
            rotation_fn(rgb2gray_fn(img_resize),
                        PATH_DATA_ARGUMENT + "/train/gray/" + class_image + "/" + path_image[i])
            rotation_fn(img_resize,
                        PATH_DATA_ARGUMENT + "/train/rgb/" + class_image + "/" + path_image[i])

        for i in index_image_random[num_train:num_image]:
            img = cv2.imread(PATH_DATA_ORIGIN + "/" + class_image + "/" + path_image[i])
            img_resize = resize_fn(crop_fn(img, BUFFER_IMAGE), SIZE_IMAGE)
            rotation_fn(rgb2gray_fn(img_resize),
                        PATH_DATA_ARGUMENT + "/test/gray/" + class_image + "/" + path_image[i])
            rotation_fn(img_resize,
                        PATH_DATA_ARGUMENT + "/test/rgb/" + class_image + "/" + path_image[i])

        progress += 1
    print("Successful! Path data argument: "+os.path.abspath(PATH_DATA_ARGUMENT))
