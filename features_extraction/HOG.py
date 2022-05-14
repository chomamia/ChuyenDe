import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
img=cv2.imread("D:/Chuyen_de/Dataset/eggplant_all/train/eggplant_gray/0.jpeg")
img=cv2.resize(img[:,:,1],(64,64))
Gx=np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
Gy=np.array([[-1, -2, -1],[0, 0, 0], [1, 2, 1]])
gx=cv2.filter2D(img,-1,Gx)
gy=cv2.filter2D(img,-1,Gy)
# gx1 = cv2.Sobel(img, cv2.CV_32F, dx=0, dy=1, ksize=3)
# gy1 = cv2.Sobel(img, cv2.CV_32F, dx=1, dy=0, ksize=3)
# g1,theta1=cv2.cartToPolar(gx1,gy1,angleInDegrees=True)
g=np.sqrt(gx**2+gy**2)
theta=np.arcsin((gy/gx))
theta=theta*180/np.pi
print(g.shape)
print(theta[10:20,0])
