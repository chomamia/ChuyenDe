import cv2
import numpy as np

for i in range(0,115):
    img = cv2.imread("D:/Chuyen_de/eggplant_all/eggplant_resize/eggplant"+str(i)+".jpg")
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    cv2.imwrite("D:/Chuyen_de/eggplant_all/eggplant_gray/eggplant" + str(i) + ".jpg", gray)