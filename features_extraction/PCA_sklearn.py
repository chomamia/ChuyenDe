from sklearn import decomposition
import numpy as np
import function as fun
lables=["eggplant", "carot","brocoli","potato","tomato"]
n=2
size=np.array([25,25])
quantity_img=480
#train
train_all=np.zeros((480*5,size[0]*n))
for a in range(0,5):
    path="D:/Chuyen_de/Dataset/"+str(lables[a])+"_all/train/"+str(lables[a])+"_gray/"
    X=fun.load_data(path,quantity_img,size)
    rows,colums=X.shape
    data=np.zeros((quantity_img,size[0]*n))
    for i in range(0,quantity_img):
        img=np.reshape(X[i,:],size)
        pca_img=decomposition.PCA(n_components=n)
        pca_data = pca_img.fit_transform(img)
        data[i, :] = np.reshape(pca_data, (1, size[0]*n))
    train_all[480*a:480*a+480,:]=data
    np.savetxt("D:/Chuyen_de/Dataset/pca_sklearn/X_train.txt",train_all,delimiter=",")
print(train_all.shape)
#test
quantity_img=120
test_all=np.zeros((120*5,size[0]*n))
for a in range(0,5):
    # img=cv2.imread("D:/Chuyen_de/Dataset/eggplant_all/train/eggplant_gray/6.jpeg")
    path="D:/Chuyen_de/Dataset/"+str(lables[a])+"_all/test/"+str(lables[a])+"_gray/"
    X=fun.load_data(path,quantity_img,size)
    rows,colums=X.shape
    data=np.zeros((quantity_img,size[0]*n))
    for i in range(0,quantity_img):
        img=np.reshape(X[i,:],size)
        pca_img = decomposition.PCA(n_components=n)
        pca_data = pca_img.fit_transform(img)
        data[i, :] = np.reshape(pca_data, (1, size[0]*n))
    test_all[120*a:120*a+120,:]=data
    np.savetxt("D:/Chuyen_de/Dataset/pca_sklearn/X_test.txt",test_all,delimiter=",")
print(test_all.shape)