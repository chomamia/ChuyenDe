import cv2
import numpy as np


for i in range(0,115):
    img=cv2.imread("D:/Chuyen_de/eggplant_all/eggplant_crop/eggplant"+str(i)+".jpg")
    imgScale=cv2.resize(img,(128,128),interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("D:/Chuyen_de/eggplant_all/eggplant_resize/eggplant"+str(i)+".jpg",imgScale)