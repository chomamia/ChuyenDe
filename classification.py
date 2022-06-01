from tkinter import Y
from sklearn.decomposition import PCA
from pre_processing.modules_pre_processing import * 
from features_extraction.modules_features_extraction import *
from training.modules_training import *

# select image RGB using for classification
PATH_IMAGE = "16.jpeg"
LABEL = ['Brocoli', 'Carrot', 'Eggplant', 'Potato', 'Tomato']

if __name__ == '__main__':
   
   img = cv2.imread(PATH_IMAGE)

   img_crop = crop_fn(img)
   img_gray = rgb2gray_fn(img_crop)

   img_resize = resize_fn(img_gray)

   img_processed = img_resize.reshape(1,-1)
   img_processed.shape

   # cv2.imshow("gray", img_resize)
   # cv2.waitKey()

   HOG_Feature = HOG_fn(img_processed)
   PCA_Feature = PCA_fn(img_processed, 50)
   #ANN-HOG
   theta1,theta2,theta3,theta4 = pickle.load(open('/home/vanquy/workspace/ChuyenDeKTMT/model_trained/ANN_HOG_2022-05-26 16_13_08.614722.sav','rb'))
   y = predict(theta1,theta2,theta3,theta4,HOG_Feature)
   print(LABEL[int(y)])
   #ANN-PCA
   theta1,theta2,theta3,theta4 = pickle.load(open('/home/vanquy/workspace/ChuyenDeKTMT/model_trained/ANN_PCA_2022-05-26 16_25_44.816964.sav','rb'))
   y = predict(theta1,theta2,theta3,theta4,PCA_Feature)
   print(LABEL[int(y)])
   #SVM-HOG
   model = pickle.load(open('/home/vanquy/workspace/ChuyenDeKTMT/model_trained/SVM_HOG_2022-05-19 08_36_51.730763.sav','rb'))
   y = model.predict(HOG_Feature)
   print(y)
   print(LABEL[int(y)-1])
   #SVM-PCA
   model = pickle.load(open('/home/vanquy/workspace/ChuyenDeKTMT/model_trained/SVM_PCA_2022-05-19 08_36_56.733214.sav','rb'))
   model = model
   y = model.predict(PCA_Feature)
   print(y)
   print(LABEL[int(y)-1])