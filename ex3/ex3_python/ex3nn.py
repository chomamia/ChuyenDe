import os.path
import function as fun
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.optimize import minimize

data=scipy.io.loadmat("ex3data1.mat")
input_layer_size=400
num_labels=10
X,y=data['X'],data['y'].ravel()
m=y.size
#random select 100 data point to display
rand_indices=np.random.choice(m,100,replace=False)
sel=X[rand_indices,:]
fun.displaydata(sel,(50,50),(10,10))

weights=scipy.io.loadmat("ex3weights.mat")
Theta_1=weights["Theta1"]
Theta_2=weights["Theta2"]
print(Theta_2.shape)
X_bias = fun.add_bias(X)
layer2_out = fun.layer1_to_layer2_output(Theta_1,X_bias)
layer2_out_bias = fun.add_bias(layer2_out.T)  # Transpose the matrix in order to match the dimension requirements
prediction = fun.layer2_to_layer3_output(Theta_2,layer2_out_bias)

predict=np.mean(np.argmax(prediction, axis = 1)+1 == y) *100
print('accuracy = ',predict)

for i in range(m):
    rand = np.random.choice(m, 1, replace=False)
    X_bias = fun.add_bias(X[rand,:])
    layer2_out = fun.layer1_to_layer2_output(Theta_1, X_bias)
    layer2_out_bias = fun.add_bias(layer2_out.T)  # Transpose the matrix in order to match the dimension requirements
    prediction = fun.layer2_to_layer3_output(Theta_2, layer2_out_bias)
    print("Neural Network Prediction:", np.argmax(prediction)+1)
    print("Displaying example image")
    fun.displaydata(X[rand,:],(50,50),(1,1))