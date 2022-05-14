# https://builtin.com/data-science/step-step-explanation-principal-component-analysis
import numpy as np
import function as fun
lables=["eggplant","carot","brocoli","potato","tomato"]
print(len(lables))
#train
n=10
size=np.array([25,25])
quantity_img=480
train_all=np.zeros((quantity_img*(len(lables)),size[0]*n))
for a in range(0,5):
    path="D:/Chuyen_de/Dataset/"+str(lables[a])+"_all/train/"+str(lables[a])+"_gray/"
    X=fun.load_data(path,quantity_img,size)
    rows,colums=X.shape
    data=np.zeros((quantity_img,size[0]*n))
    for i in range(0,quantity_img):
        img=np.reshape(X[i,:],size)
        pca_img=fun.PCA(img,n)
        data[i,:]=np.reshape(pca_img,(1,size[0]*n))
    train_all[quantity_img*a:quantity_img*a+quantity_img,:]=data
    np.savetxt("D:/Chuyen_de/Dataset/X_train.txt",train_all,delimiter=",")
print(train_all.shape)
#test
quantity_img = 120
test_all=np.zeros((quantity_img*(len(lables)),size[0]*n))
for a in range(0,5):
    # img=cv2.imread("D:/Chuyen_de/Dataset/eggplant_all/train/eggplant_gray/6.jpeg")
    path="D:/Chuyen_de/Dataset/"+str(lables[a])+"_all/test/"+str(lables[a])+"_gray/"
    X=fun.load_data(path,quantity_img,size)
    rows,colums=X.shape
    data=np.zeros((quantity_img,size[0]*n))
    for i in range(0,quantity_img):
        img=np.reshape(X[i,:],size)
        pca_img=fun.PCA(img,n)
        data[i,:]=np.reshape(pca_img,(1,size[0]*n))
    test_all[quantity_img*a:quantity_img*a+quantity_img,:]=data
    np.savetxt("D:/Chuyen_de/Dataset/X_test.txt",test_all,delimiter=",")
print(test_all.shape)
#lables
lables=fun.lables(480)
np.savetxt("D:/Chuyen_de/Dataset/y_train.txt", lables, delimiter=",")
lables=fun.lables(120)
np.savetxt("D:/Chuyen_de/Dataset/y_test.txt", lables, delimiter=",")
