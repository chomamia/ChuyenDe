from tkinter import N
from sklearn import decomposition
import numpy as np
import function as fun
import matplotlib.pyplot as plt

X_train = np.loadtxt('../../output_data_train/data.txt')
X_test = np.loadtxt('../../output_data_test/data.txt')

n = 50

pca = decomposition.PCA(n_components=n)

pca = pca.fit_transform(X_train)

print(pca.shape)



# n=50
# size=np.array([25,25])
# quantity_img=480
# train_all=np.zeros((quantity_img*(len(lables)),size[0]*size[1]))
# #load data
# for a in range(0,5):
#     path="D:/Chuyen_de/Dataset/"+str(lables[a])+"_all/train/"+str(lables[a])+"_gray/"
#     X=fun.load_data(path,quantity_img,size)
#     # np.savetxt("D:/Chuyen_de/Dataset/pca_sklearn/X_train.txt",train_all,delimiter=",")
#     train_all[quantity_img * a:quantity_img * a + quantity_img, :] = X
# #pca
# print(train_all.shape)
# pca = decomposition.PCA(n_components=n)
# pca_data = pca.fit_transform(train_all)
# print(pca_data.shape)
# np.savetxt("D:/Chuyen_de/Dataset/pca_sklearn/fix/X_train1.txt", pca_data, delimiter=",")

# quantity_img=120
# test_all=np.zeros((quantity_img*(len(lables)),size[0]*size[1]))
# for a in range(0,5):
#     path="D:/Chuyen_de/Dataset/"+str(lables[a])+"_all/test/"+str(lables[a])+"_gray/"
#     X=fun.load_data(path,quantity_img,size)
#     # np.savetxt("D:/Chuyen_de/Dataset/pca_sklearn/X_train.txt",train_all,delimiter=",")
#     test_all[quantity_img * a:quantity_img * a + quantity_img, :] = X
# #pca
# print(test_all.shape)
# pca = decomposition.PCA(n_components=n)
# pca_data = pca.fit_transform(test_all)
# print(pca_data.shape)
# np.savetxt("D:/Chuyen_de/Dataset/pca_sklearn/fix/X_test1.txt", pca_data, delimiter=",")

