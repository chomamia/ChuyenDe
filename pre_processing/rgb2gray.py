import cv2
import numpy as np

for i in range(0,480):
    img = cv2.imread("D:/Chuyen_de/Dataset/brocoli_all/train/brocoli_resize/"+str(i)+".jpeg")
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    cv2.imwrite("D:/Chuyen_de/Dataset/brocoli_all/train/brocoli_gray/"+str(i)+".jpeg", gray)
for i in range(480,600):
    img = cv2.imread("D:/Chuyen_de/Dataset/brocoli_all/test/brocoli_resize/"+str(i)+".jpeg")
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    cv2.imwrite("D:/Chuyen_de/Dataset/brocoli_all/test/brocoli_gray/"+str(i)+".jpeg", gray)