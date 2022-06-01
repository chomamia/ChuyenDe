import cv2
import numpy as np
from modules_pre_processing import *
from modules_features_extraction import *
import pickle
from predict_ANN import *
from sklearn.svm import SVC

labels = ["Brocoli", "Carot", "Eggplant", "Potato", "Tomato"]
BUFFER_IMAGE = 50
size = [64, 64]
n_component = 50
path_img = 'D:/Chuyen_de/Dataset/eggplant_all/train/eggplant/216.jpeg'
img = cv2.imread(path_img)
img_crop = crop_fn(img, BUFFER_IMAGE)
img_resize = resize_fn(img_crop, size)
img = rgb2gray_fn(img_resize)
img = np.reshape(img, (1, -1))
PCA_feature = PCA_fn(img, n_component)
HOG_feature = HOG_fn(img)
#load model SVM-PCA
model_pca = pickle.load(open('D:/Chuyen_de/VanQuy/model_trained/SVM_PCA.sav', 'rb'))
y_predict = model_pca.predict(PCA_feature)
print("Predict use SVM-PCA:")
print("Image is", labels[int(y_predict)-1])
#load model SVM-HOG
model_HOG = pickle.load(open('D:/Chuyen_de/VanQuy/model_trained/SVM_HOG.sav', 'rb'))
y_predict = model_HOG.predict(HOG_feature)
print("Predict use SVM-HOG:")
print("Image is", labels[int(y_predict)-1])
#load model ANN-PCA
theta1, theta2, theta3, theta4 = pickle.load(open('D:/Chuyen_de/VanQuy/model_trained/ANN_PCA.sav', 'rb'))
y_predict = predict(PCA_feature, theta1, theta2, theta3, theta4)
print("Predict use ANN-PCA:")
print("Image is", labels[int(y_predict)-1])
#load model ANN-HOG
theta1, theta2, theta3, theta4 = pickle.load(open('D:/Chuyen_de/VanQuy/model_trained/ANN_HOG.sav', 'rb'))
y_predict = predict(HOG_feature, theta1, theta2, theta3, theta4)
print("Predict use ANN-HOG:")
print("Image is", labels[int(y_predict)-1])






