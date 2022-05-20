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


def PCA(X, num_components):
    # tieu chuan hoa du lieu
    #step1
    mean_X = np.mean(X, axis=0)
    X_meaned = (X - mean_X)
    #step2
    # tinh toan ma tran hiep phuong sai
    covar_matrix = np.cov(X_meaned, rowvar=False)
    #step3
    # tinh toan cac gia tri rieng va cac gia tri rieeng cua ma tran hiep phuong sai
    values, vectors = eigh(covar_matrix)
    sorted_index = np.argsort(values)[::-1]
    sorted_eigenvalue = values[sorted_index]
    sorted_eigenvectors = vectors[:, sorted_index]
    #step5
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]
    #step6
    X_reduced = np.dot(eigenvector_subset.T, X_meaned.T).T
    return X_reduced

def create_labels(num):
    labels = np.zeros((num * 5, 5), dtype=int)
    for a in range(0, 5):
        labels[num * a:num * a + num, a] = 1
    return labels