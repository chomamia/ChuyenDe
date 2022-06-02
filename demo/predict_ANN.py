import os
from tkinter.tix import X_REGION
import cv2
import numpy as np


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def predict(X, theta1, theta2, theta3, theta4):
    # size
    m, n = X.shape
    num_labels = theta4.shape[0]
    # add ones to X data matrix
    X = np.hstack((np.ones((m, 1)), X))

    z2 = np.dot(X, theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((a2.shape[0], 1)), a2))

    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)
    a3 = np.hstack((np.ones((a3.shape[0], 1)), a3))

    z4 = np.dot(a3, theta3.T)
    a4 = sigmoid(z4)
    a4 = np.hstack((np.ones((a4.shape[0], 1)), a4))

    z5 = np.dot(a4, theta4.T)
    a5 = sigmoid(z5)
    labels = np.argmax(a5, axis=1)
    return labels