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
def Crop_square(img):
    original_image = img.copy()
    rows, colums, h = original_image.shape
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img1 = cv2.threshold(gray, 220, 260, cv2.THRESH_BINARY)

    # edges=cv2.Canny(gray,150,200)
    contours, hiera = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area_cnt = [cv2.contourArea(cnt) for cnt in contours]
    area_soft = np.argsort(area_cnt)[::-1]
    area_soft[:1]
    cnt = contours[area_soft[1]]
    x, y, w, h = cv2.boundingRect(cnt)
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    img_crop = original_image[y - int(y / 2):y + h + int(y / 2), x - int(x / 2):x + w + int(x / 2)]
    return img_crop
#doi range(0,155) thanh range(0,"so anh cua ban")
for i in range(0,115):
    img=cv2.imread("D:/Chuyen_de/eggplant_all/eggplant_jpg/eggplant"+str(i)+".jpg")
    img_crop=Crop_square(img)
    img_crop=square(img_crop)
    cv2.imwrite("D:/Chuyen_de/eggplant_all/eggplant_crop/eggplant"+str(i)+".jpg",img_crop)



