import numpy as np
import function as fun
import os
import cv2
import glob
import imutils
lables=["eggplant","carot","brocoli","potato","tomato"]
size=np.array([64,64])
#train
quantity_img=480
train_all=np.zeros((quantity_img*(len(lables)),1764))
data=np.zeros((quantity_img,1764))
for a in range(0,5):
    for i in range(0,quantity_img):
        path="D:/Chuyen_de/Dataset/"+lables[a]+"_all/train/"+lables[a]+"_gray/"+str(i)+".jpeg"
        img=cv2.imread(path)
        img=cv2.resize(img[:,:,0],size)
        hog = cv2.HOGDescriptor(_winSize=(64,64),
                                _blockSize=(16, 16),
                                _blockStride=(8, 8),
                                _cellSize=(8, 8),
                                _nbins=9)
        hog_feats=hog.compute(img)
        data[i,:]=hog_feats
    train_all[quantity_img * a:quantity_img * a + quantity_img, :] = data
print(train_all.shape)
np.savetxt("D:/Chuyen_de/Dataset/hog/X_train.txt",train_all,delimiter=",")
#test
quantity_img=120
test_all=np.zeros((quantity_img*(len(lables)),1764))
data=np.zeros((quantity_img,1764))
for a in range(0,5):
    for i in range(0,quantity_img):
        img=cv2.imread("D:/Chuyen_de/Dataset/"+lables[a]+"_all/test/"+lables[a]+"_gray/"+str(i)+".jpeg")
        img=cv2.resize(img[:,:,1],size)
        hog = cv2.HOGDescriptor(_winSize=(64,64),
                                _blockSize=(16, 16),
                                _blockStride=(8, 8),
                                _cellSize=(8, 8),
                                _nbins=9)
        hog_feats=hog.compute(img)
        data[i,:]=hog_feats
    test_all[quantity_img*a:quantity_img*a+quantity_img,:]=data
print(test_all.shape)
np.savetxt("D:/Chuyen_de/Dataset/hog/X_test.txt",test_all,delimiter=",")