from sklearn import decomposition
import numpy as np
import function as fun
import cv2
import matplotlib.pyplot as plt

img=cv2.imread("D:/Chuyen_de/Dataset/eggplant_all/train/eggplant_gray/0.jpeg")
X=img[:,:,1]
print(X.shape)
X=cv2.resize(X,(32,32))
plt.imshow(X)
print(X.shape)
# pca=fun.PCA(X)
pca_img = decomposition.PCA(n_components=15)
pca_data = pca_img.fit_transform(X)
print(pca_data.shape)
inverted = pca_img.inverse_transform(pca_data)
img_compressed = (np.dstack((inverted, inverted, inverted))).astype(np.uint8)
plt.imshow(img_compressed)
plt.show()
