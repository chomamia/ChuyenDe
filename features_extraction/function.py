import numpy as np
import cv2
from scipy.linalg import eigh
from scipy import linalg


labels = ["eggplant", "carot", "brocoli", "potato", "tomato"]

def load_data(path, quantity_img, size):
    X=np.zeros((quantity_img, size[0]*size[1]))
    for i in range(0, quantity_img):
        img = cv2.imread(path+str(i)+".jpeg")
        img = img[:, :, 1]
        img = cv2.resize(img, size)
        rows, columns = img.shape
        img = np.reshape(img, (1, rows*columns))
        X[i, :] = img
    return X


def PCA(X,n):
    # tieu chuan hoa du lieu
    mean_X = np.mean(X)
    X = (X - mean_X) / np.std(X)
    N = X.shape[1]
    x = np.zeros((X.shape[0]))
    for i in range(0, N):
        x = x + 1/N * X[:, i]
    for i in range(0 , N):
        X[:, i] = X[:, i] - x
    # X = X-np.min(X)
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


def create_labels(num):
    labels = np.zeros((num * 5, 5), dtype=int)
    for a in range(0, 5):
        labels[num * a:num * a + num, a] = 1
    return labels