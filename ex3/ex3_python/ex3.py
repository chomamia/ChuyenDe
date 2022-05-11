import os.path
import function as fun
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.optimize import minimize

#load data
data=scipy.io.loadmat("ex3data1.mat")
input_layer_size=400
num_labels=10
X,y=data['X'],data['y'].ravel()
m=y.size
#random select 100 data point to display
rand_indices=np.random.choice(m,100,replace=False)
sel=X[rand_indices,:]
fun.displaydata(sel,(50,50),(10,10))

# Vectorize Logistic Regression
lamb=0.1
all_theta=fun.oneVsAll(X,y,num_labels,lamb)
y_pred=fun.predict_all(X,all_theta)
correct=[1 if a==b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print('accuracy = {0}%'.format(accuracy * 100))