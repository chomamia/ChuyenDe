import cv2
import numpy as np
from PIL import Image

def square(img):
    m, n, i = img.shape
    if n > m:
        theta = int((n - m) / 2)
        img_white = np.zeros(((n, n, 3)), dtype=np.uint8)
        img_white.fill(255)
        img_white[theta:theta + m, :, :] = img[:, :, :]
    else:
        theta = int((m - n) / 2)
        img_white = np.zeros(((m, m, 3)), dtype=np.uint8)
        img_white.fill(255)
        img_white[:, theta:theta + n, :] = img[:, :, :]
    return img_white
def Crop(img,num):
    original_image = img.copy()
    rows, colums, h = original_image.shape
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img1 = cv2.threshold(gray, 220,255, cv2.THRESH_BINARY)
    contours, hiera = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area_cnt = [cv2.contourArea(cnt) for cnt in contours]
    area_soft = np.argsort(area_cnt)[::-1]
    area_soft[:num]
    cnt = contours[area_soft[num]]
    x, y, w, h = cv2.boundingRect(cnt)
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    img_crop = original_image[y - int(y / 2):y + h + int(y / 2), x - int(x / 2):x + w + int(x / 2)]
    return img_crop
def fix(img):
    h,w=img.shape[0:2]
    max1=max(w,h)
    img1=np.zeros(((max1+50,max1+50,3)),dtype=np.uint8)
    img1.fill(255)
    img1[int((max1+50-h)/2):int((max1+50-h)/2)+h,int((max1+50-w)/2):int((max1+50-w)/2)+w,:]=img[:,:,:]
    return img1
#doi range(0,155) thanh range(0,"so anh cua ban")
#train
for i in range(0,80):
    img=cv2.imread("D:/Chuyen_de/Dataset/brocoli_all/train/brocoli_input/"+str(i)+".jpeg")
    img=fix(img)
    img_crop1=Crop(img,1)
    img_square1=square(img_crop1)
    # cv2.imshow("fix", img)
    # cv2.imshow("crop", img_crop1)
    # cv2.imshow("square",img_square1)
    cv2.waitKey()
    cv2.imwrite("D:/Chuyen_de/Dataset/brocoli_all/train/brocoli_crop/"+str(i)+".jpeg",img_square1)
#test
for i in range(80,100):
    img=cv2.imread("D:/Chuyen_de/Dataset/brocoli_all/test/brocoli_input/"+str(i)+".jpeg")
    img=fix(img)
    img_crop1=Crop(img,1)
    img_square1=square(img_crop1)
    # cv2.imshow("fix", img)
    # cv2.imshow("crop", img_crop1)
    # cv2.imshow("square",img_square1)
    cv2.waitKey()
    cv2.imwrite("D:/Chuyen_de/Dataset/brocoli_all/test/brocoli_crop/"+str(i)+".jpeg",img_square1)

