import numpy as np
import cv2
from scipy.linalg import eigh

def load_data(path,quantity_img,size):
    X=np.zeros((quantity_img,size[0]*size[1]),dtype=int)
    for i in range(0,quantity_img):
        img=cv2.imread(path+str(i)+".jpeg")
        img=img[:,:,1]
        img=cv2.resize(img,size)
        rows,colums=img.shape
        img=np.reshape(img,(1,rows*colums))
        X[i,:]=img
    return X
def PCA(X,n):
    # tieu chuan hoa du lieu
    mean_X = np.mean(X)
    X = (X - mean_X) / np.std(X)
    X=X-np.min(X)
    # tinh toan ma tran hiep phuong sai
    covar_matrix = np.cov(X)
    # tinh toan cac gia tri rieng va cac gia tri rieeng cua ma tran hiep phuong sai
    values, vectors = eigh(covar_matrix, eigvals=(1, n))
    # tim duoc vector ring
    vectors = vectors.T
    # du lieu chinh la toa do cua cac diem tren khong gian moi
    new_coordinates = np.dot(vectors, X.T)
    new_coordinates = new_coordinates.T
    return new_coordinates
def lables(num):
    lables = np.zeros((num * 5, 5), dtype=int)
    for a in range(0, 5):
        lables[num * a:num * a + num, a] = 1
    return lables