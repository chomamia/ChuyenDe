import cv2
import numpy as np
for i in range(0,80):
    img=cv2.imread("D:/Chuyen_de/Dataset/brocoli_all/train/brocoli_crop/"+str(i)+".jpeg")
    h,w,_=img.shape
    #xoay anh
    m1=cv2.getRotationMatrix2D((h/2,w/2),0,1.01)
    img1=cv2.warpAffine(img,m1,(h,w))
    m2=cv2.getRotationMatrix2D((h/2,w/2),90,1.01)
    img2=cv2.warpAffine(img,m2,(h,w))
    m3=cv2.getRotationMatrix2D((h/2,w/2),180,1.01)
    img3=cv2.warpAffine(img,m3,(h,w))
    m4=cv2.getRotationMatrix2D((h/2,w/2),270,1.01)
    img4=cv2.warpAffine(img,m4,(h,w))
    #lat anh
    img5=img.copy()
    img5[:,:,:]=img[h::-1,:,:]
    img6=img5.copy()
    img6[:,:,:]=img5[:,w::-1,:]
    path="D:/Chuyen_de/Dataset/brocoli_all/train/brocoli/"
    cv2.imwrite(path+str(i*6)+".jpeg",img1)
    cv2.imwrite(path + str(i*6+1) + ".jpeg", img2)
    cv2.imwrite(path + str(i*6+2) + ".jpeg", img3)
    cv2.imwrite(path + str(i*6+3) + ".jpeg", img4)
    cv2.imwrite(path + str(i*6+4) + ".jpeg", img5)
    cv2.imwrite(path + str(i*6+5) + ".jpeg", img6)
for i in range(80,100):
    img=cv2.imread("D:/Chuyen_de/Dataset/brocoli_all/test/brocoli_crop/"+str(i)+".jpeg")
    h,w,_=img.shape
    #xoay anh
    m1=cv2.getRotationMatrix2D((h/2,w/2),0,1.01)
    img1=cv2.warpAffine(img,m1,(h,w))
    m2=cv2.getRotationMatrix2D((h/2,w/2),90,1.01)
    img2=cv2.warpAffine(img,m2,(h,w))
    m3=cv2.getRotationMatrix2D((h/2,w/2),180,1.01)
    img3=cv2.warpAffine(img,m3,(h,w))
    m4=cv2.getRotationMatrix2D((h/2,w/2),270,1.01)
    img4=cv2.warpAffine(img,m4,(h,w))
    #lat anh
    img5=img.copy()
    img5[:,:,:]=img[h::-1,:,:]
    img6=img5.copy()
    img6[:,:,:]=img5[:,w::-1,:]
    path="D:/Chuyen_de/Dataset/brocoli_all/test/brocoli/"
    cv2.imwrite(path+str(i*6)+".jpeg",img1)
    cv2.imwrite(path + str(i*6+1) + ".jpeg", img2)
    cv2.imwrite(path + str(i*6+2) + ".jpeg", img3)
    cv2.imwrite(path + str(i*6+3) + ".jpeg", img4)
    cv2.imwrite(path + str(i*6+4) + ".jpeg", img5)
    cv2.imwrite(path + str(i*6+5) + ".jpeg", img6)