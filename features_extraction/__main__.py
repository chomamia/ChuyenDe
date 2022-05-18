import numpy as np

from HOG import *
from function import *


# X=
size = np.array([64,64])
# labels = ["eggplant", "carot", "brocoli", "potato", "tomato"]
labels = ["brocoli", "carot", "eggplant", "potato", "tomato"]
quantity_img = 480
data_train = np.zeros((quantity_img*(len(labels)), size[0]*size[1]))
for a in range(0,5):
    path="D:/Chuyen_de/Dataset/"+str(labels[a])+"_all/train/"+str(labels[a])+"_gray/"
    X = load_data(path,quantity_img,size)
    rows,colums = X.shape
    data_train[quantity_img*a:quantity_img*a+quantity_img, :] = X
hog = HOG(data_train)
print(hog.shape)
np.savetxt("D:/Chuyen_de/Dataset/hog/X_train_1.txt", hog)

quantity_img = 120
data_test = np.zeros((quantity_img*(len(labels)), size[0]*size[1]))
for a in range(0,5):
    path="D:/Chuyen_de/Dataset/"+str(labels[a])+"_all/test/"+str(labels[a])+"_gray/"
    X = load_data(path,quantity_img,size)
    rows,colums = X.shape
    data_test[quantity_img*a:quantity_img*a+quantity_img,:] = X
hog = HOG(data_test)
print(hog.shape)
np.savetxt("D:/Chuyen_de/Dataset/hog/X_test_1.txt",hog)

