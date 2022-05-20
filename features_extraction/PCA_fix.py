from sklearn import decomposition
import numpy as np
import function as fun
import matplotlib.pyplot as plt
lables=["eggplant", "carot","brocoli","potato","tomato"]

n=64
size=np.array([32,32])
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
pca = decomposition.PCA(n_components=n)
pca_data = pca.fit_transform(train_all)
print(pca_data.shape)
np.savetxt("D:/Chuyen_de/Dataset/pca_sklearn/fix/X_train1.txt", pca_data)
# plt.style.use('seaborn-whitegrid')
# plt.figure(figsize = (10,6))
# c_map = plt.cm.get_cmap('jet', 5)
# plt.scatter(pca_data[:, 0], pca_data[:, 1], s = 15,
#             cmap = c_map , c =lables_y )
# plt.colorbar()
# plt.xlabel('PC-1') , plt.ylabel('PC-2')
# plt.show()
# test
quantity_img=120
test_all=np.zeros((quantity_img*(len(lables)),size[0]*size[1]))
for a in range(0,5):
    path="D:/Chuyen_de/Dataset/"+str(lables[a])+"_all/test/"+str(lables[a])+"_gray/"
    X=fun.load_data(path,quantity_img,size)
    # np.savetxt("D:/Chuyen_de/Dataset/pca_sklearn/X_train.txt",train_all,delimiter=",")
    test_all[quantity_img * a:quantity_img * a + quantity_img, :] = X
#pca
print(test_all.shape)
pca = decomposition.PCA(n_components=n)
pca_data = pca.fit_transform(test_all)
print(pca_data.shape)
np.savetxt("D:/Chuyen_de/Dataset/pca_sklearn/fix/X_test1.txt", pca_data)

