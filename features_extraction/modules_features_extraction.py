import os
from tkinter.tix import X_REGION
import cv2
import numpy as np
from scipy.linalg import eigh


def load_data(path):
    print("\nLoading data...")
    class_input = os.listdir(path)
    num_image = 0
    X = np.array([])
    label = 1
    Y = np.array([])
    for c in class_input:
        for i in os.listdir(path+"/"+c):
            path_img = path + "/" + c + "/" + i
            img = cv2.imread(path_img)
            img = np.reshape(img[:, :, 1], (1, -1))
            if len(X) == 0:
                X = [img[0, :]]
                Y = [label]
            else:
                X = np.concatenate((X, [img[0, :]]), axis=0, dtype=float)
                Y = np.concatenate((Y, [label]), axis=0)

        label += 1

    print("Done! \nShape of data: ", X.shape)
    return X, Y


def PCA_fn(data_input, n_component):
    print("\nPCA feature extraction...")
    n, size = data_input.shape
    PCA = np.array([], dtype=float)
    for i in range(n):
        # Print process
        print_loading(i, n, 10)

        # tieu chuan hoa du lieu
        X = np.reshape(data_input[i, :], [int(size**(1/2)), int(size**(1/2))])
        component_col = n_component // X.shape[0] + 1
        mean_X = np.mean(X, axis=0)
        X_meaned = (X - mean_X)

        # tinh toan ma tran hiep phuong sai
        covar_matrix = np.cov(X_meaned, rowvar=True)

        # tinh toan cac gia tri rieng va cac gia tri rieng cua ma tran hiep phuong sai
        values, vectors = eigh(covar_matrix)
        sorted_index = np.argsort(values)[::-1]
        sorted_eigenvectors = vectors[:, sorted_index]
        eigenvector_subnet = sorted_eigenvectors[:, 0:component_col]
        X_reduced = np.dot(eigenvector_subnet.T, X_meaned).T
        X_reduced = np.reshape(X_reduced, (1,-1))
        coordinate = X_reduced[:, 0:n_component]
        if len(PCA) == 0:
            PCA = [coordinate[0, :]]
        else:
            PCA = np.concatenate((PCA, [coordinate[0, :]]), axis=0, dtype=float)

    print("Done!!! \nShape of PCA feature: ", PCA.shape)
    return PCA


def PCA_all(X, n_component):
    mean_X = np.mean(X, axis=0)
    X_meaned = X - mean_X
    covar_matrix = np.cov(X_meaned.T, rowvar=True)
    values, vectors = eigh(covar_matrix.T)
    sorted_index = np.argsort(values)[::-1]
    sorted_eigenvectors = vectors[:, sorted_index]
    # step5
    eigenvector_subset = sorted_eigenvectors[:, 0:n_component]
    # step6
    pca = np.dot(eigenvector_subset.T, X_meaned.T).T
    pca_restore = np.dot(eigenvector_subset, pca.T).T
    return pca


def HOG_fn(data_input):
    print("\nHOG feature extraction...")
    n, size = data_input.shape
    HOG = np.array([])
    for i in range(n):
        # Print process
        print_loading(i, n, 10)

        X = np.reshape(data_input[i, :], [int(size ** (1 / 2)), int(size ** (1 / 2))])
        rows, columns = X.shape
        # img = np.reshape(X[i, :], (64, 64))
        gama, theta = calculator_mapping(X)
        cell = calculator_cell(gama, theta)
        # print("size of cell: ", cell.shape)
        block = calculator_block(cell)
        block = np.reshape(block, (1, -1))
        # print("size of block: ", block.shape)
        if len(HOG) == 0:
            HOG = [block[0, :]]
        else:
            HOG = np.concatenate((HOG, [block[0, :]]), axis=0)

    print("Done!!! \nShape of HOG feature: ", HOG.shape)
    return HOG


def calculator_cell(gama, theta):
    gama = gama / 20
    cell = np.zeros(((8, 8, 9)))
    for r in range(0, 8):
        for c in range(0, 8):
            bin = np.zeros(9, dtype=float)
            for i in range(0, 8):
                for j in range(0, 8):
                    pos = theta[i + r * 8, j + c * 8] // 20
                    bin[int(pos)] = np.float64(gama[i + r * 8, j + c * 8]) * abs(
                        20 * (pos + 1) - theta[i + r * 8, j + c * 8])
                    if pos + 1 < 9:
                        bin[int(pos + 1)] = gama[i + r * 8, j + c * 8] * abs(20 * pos - theta[i + r * 8, j + c * 8])
                    else:
                        bin[0] = gama[i + r * 8, j + c * 8] * abs(20 * pos - theta[i + r * 8, j + c * 8])
                cell[r, c, :] = bin
    return cell


def calculator_block(cell):
    block = np.zeros(((7, 7, 36)))
    for r in range(0, 7):
        for c in range(0, 7):
            bins = np.zeros(36)
            if np.sum(cell[r:r + 2, c:c + 2]) != 0:
                bins = np.reshape(cell[r:r + 2, c:c + 2], (1, 36)) / np.sum(cell[r:r + 2, c:c + 2])
            block[r, c, :] = bins
    return block


def calculator_mapping(X):
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gx = cv2.filter2D(X, -1, Gx)
    gy = cv2.filter2D(X, -1, Gy)
    gama = np.sqrt(gx ** 2 + gy ** 2)
    theta = np.ones(gy.shape) * np.pi / 2
    theta[gx != 0] = np.arctan(gy[gx != 0] / gx[gx != 0])
    theta = theta * 180 / np.pi
    return gama, theta


def print_loading(i, n, step):
    if int((i + 1) * 100 / step / n) != int(i * 100/ step / n):
        print("loading: {process:.0%}".format(process=(i + 1) / n))
