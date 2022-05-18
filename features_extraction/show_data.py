import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import function as fun
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d

lables=["eggplant", "carot","brocoli","potato","tomato"]
y = np.loadtxt("D:/Chuyen_de/Dataset/pca_sklearn/y_train1.txt", dtype=int, delimiter=",")
n=2
size=np.array([25,25])
quantity_img=480
train_all=np.zeros((quantity_img*(len(lables)),size[0]*size[1]))
#load data
for a in range(0,5):
    path="D:/Chuyen_de/Dataset/"+str(lables[a])+"_all/train/"+str(lables[a])+"_gray/"
    X=fun.load_data(path,quantity_img,size)
    # np.savetxt("D:/Chuyen_de/Dataset/pca_sklearn/X_train.txt",train_all,delimiter=",")
    train_all[quantity_img * a:quantity_img * a + quantity_img, :] = X
#pca
print(train_all.shape)
pca = PCA(n_components=n)
pca_data = pca.fit(train_all)
pca_scale_cancer_data = pca.transform(train_all)

print(pca_scale_cancer_data.shape)

plt.figure()
# Thành phần comp số 1
pca_1 = pca_scale_cancer_data[:, 0]
# Thành phần comp số 2
pca_2 = pca_scale_cancer_data[:, 1]
# Vẽ đồ thị
plt.scatter(x=pca_1, y = pca_2, c = y )
plt.show()