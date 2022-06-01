import cv2
import numpy as np

# rotation image
def rotation_fn(img, path_output):
    if len(img.shape) == 2:
        # for gray image
        h, w = img.shape
        # rotation image
        m1 = cv2.getRotationMatrix2D((h / 2, w / 2), 0, 1.01)
        img1 = cv2.warpAffine(img, m1, (h, w))
        m2 = cv2.getRotationMatrix2D((h / 2, w / 2), 90, 1.01)
        img2 = cv2.warpAffine(img, m2, (h, w))
        m3 = cv2.getRotationMatrix2D((h / 2, w / 2), 180, 1.01)
        img3 = cv2.warpAffine(img, m3, (h, w))
        m4 = cv2.getRotationMatrix2D((h / 2, w / 2), 270, 1.01)
        img4 = cv2.warpAffine(img, m4, (h, w))
        # flip image
        img5 = img.copy()
        img5[:, :] = img[h::-1, :]
        img6 = img5.copy()
        img6[:, :] = img5[:, w::-1]
    else:
        # for rgb image
        h, w, _ = img.shape
        # rotation image
        m1 = cv2.getRotationMatrix2D((h / 2, w / 2), 0, 1.01)
        img1 = cv2.warpAffine(img, m1, (h, w))
        m2 = cv2.getRotationMatrix2D((h / 2, w / 2), 90, 1.01)
        img2 = cv2.warpAffine(img, m2, (h, w))
        m3 = cv2.getRotationMatrix2D((h / 2, w / 2), 180, 1.01)
        img3 = cv2.warpAffine(img, m3, (h, w))
        m4 = cv2.getRotationMatrix2D((h / 2, w / 2), 270, 1.01)
        img4 = cv2.warpAffine(img, m4, (h, w))
        # flip image
        img5 = img.copy()
        img5[:, :, :] = img[h::-1, :, :]
        img6 = img5.copy()
        img6[:, :, :] = img5[:, w::-1, :]

    # save image to folder data_argument
    cv2.imwrite(path_output[0:-4]+"_1.jpeg", img1)
    cv2.imwrite(path_output[0:-4]+"_2.jpeg", img2)
    cv2.imwrite(path_output[0:-4]+"_3.jpeg", img3)
    cv2.imwrite(path_output[0:-4]+"_4.jpeg", img4)
    cv2.imwrite(path_output[0:-4]+"_5.jpeg", img5)
    cv2.imwrite(path_output[0:-4]+"_6.jpeg", img6)


# convert rgb image to gray image
def rgb2gray_fn(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray


# resize image
def resize_fn(img, size=[64,64]):
    imgScale = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return imgScale


# crop image
def crop_fn(img, buffer=50):
    img = fix(img, buffer)
    img_crop = crop(img, 1)
    img_square = square(img_crop)
    # all_image = np.concatenate((resize_fn(img, (300, 300)), resize_fn(img_crop, (300, 300)),
    #                             resize_fn(img_square, (300, 300))), axis=1)
    #
    # cv2.imshow("img", img)
    # cv2.imshow("all", all_image)
    # cv2.waitKey()
    return img_square


# make the image size to square
def square(img):
    m, n, i = img.shape
    if n > m:
        theta = int((n - m) / 2)
        img_white = np.zeros(([n, n, 3]), dtype=np.uint8)
        img_white.fill(255)
        img_white[theta:theta + m, :, :] = img[:, :, :]
    else:
        theta = int((m - n) / 2)
        img_white = np.zeros(([m, m, 3]), dtype=np.uint8)
        img_white.fill(255)
        img_white[:, theta:theta + n, :] = img[:, :, :]
    return img_white


# crop image
def crop(img, num):
    original_image = img.copy()
    rows, columns, h = original_image.shape
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img1 = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    contours, hiera = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area_cnt = [cv2.contourArea(cnt) for cnt in contours]
    area_soft = np.argsort(area_cnt)[::-1]
    cnt = contours[area_soft[num]]
    x, y, w, h = cv2.boundingRect(cnt)
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    img_crop = original_image[y - int(y / 2):y + h + int(y / 2), x - int(x / 2):x + w + int(x / 2)]
    return img_crop


# fix to image
def fix(img, index):
    h, w = img.shape[0:2]
    max1 = max(w, h)
    img1 = np.zeros((max1+index, max1+index, 3), dtype=np.uint8)
    img1.fill(255)
    img1[int((max1+index-h)/2):int((max1+index-h)/2)+h, int((max1+index-w)/2):int((max1+index-w)/2)+w, :] = img[:,:,:]
    return img1


