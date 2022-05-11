import numpy as np
import function as fun
import os
import cv2
import glob
import imutils
lables=["eggplant","carot","brocoli","potato","tomato"]
#train
size=np.array([64,64])
quantity_img=480
train_all=np.zeros((quantity_img*(len(lables)),1764))
for a in range(0,5):
    path="D:/Chuyen_de/Dataset/"+str(lables[a])+"_all/train/"+str(lables[a])+"_gray/"
    X=fun.load_data(path,quantity_img,size)
    rows,colums=X.shape
    data=np.zeros((quantity_img,1764))
    for i in range(0,quantity_img):
        img=np.reshape(X[i,:],size)
        print(img.shape)
        hog = cv2.HOGDescriptor(_winSize=size,
                                _blockSize=(16, 16),
                                _blockStride=(8, 8),
                                _cellSize=(8, 8),
                                _nbins=9)
        hog_feats=hog.compute(img)
        data[i,:]=hog_feats
    np.savetxt("D:/Chuyen_de/Dataset/"+str(lables[a])+"_all/data_"+str(lables[a])+"_train.txt",data,delimiter=",")
    train_all[quantity_img*a:quantity_img*a+quantity_img,:]=data
    # np.savetxt("D:/Chuyen_de/Dataset/X_train.txt",train_all,delimiter=",")
print(train_all.shape)