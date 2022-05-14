import cv2
import numpy as np


for i in range(0,480):
    img=cv2.imread("D:/Chuyen_de/Dataset/brocoli_all/train/brocoli/"+str(i)+".jpeg")
    imgScale=cv2.resize(img,(128,128),interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("D:/Chuyen_de/Dataset/brocoli_all/train/brocoli_resize/"+str(i)+".jpeg",imgScale)
for i in range(480,600):
    img=cv2.imread("D:/Chuyen_de/Dataset/brocoli_all/test/brocoli/"+str(i)+".jpeg")
    imgScale=cv2.resize(img,(128,128),interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("D:/Chuyen_de/Dataset/brocoli_all/test/brocoli_resize/"+str(i)+".jpeg",imgScale)