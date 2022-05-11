import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cv2
def displayData(X, example_width=None, figsize=(10, 10)):
    """
    Displays 2D data stored in X in a nice grid.
    """
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_width, order='F'),
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')
def displaydata(X,resize,figure_size):
    rows_data=figure_size[0]
    colums_data=figure_size[1]
    img=np.zeros((rows_data*resize[0],colums_data*resize[1]))
    for i in range(0,rows_data):
        for j in range(0,colums_data):
            img[i*resize[0]:i*resize[0]+resize[0],j*resize[1]:j*resize[1]+resize[1]]=cv2.resize(np.reshape(X[figure_size[0]*i+j,:],(20,20)).T,(resize[0],resize[1]))
    cv2.imshow("figure Data",img)
    cv2.waitKey()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta,X,y,lamb):
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)
    h=sigmoid(X*theta.T)
    p=(lamb/len(y)*2)*np.sum(np.power(theta[:,1:theta.shape[1]],2))
    first=np.multiply(-y, np.log(h))
    second=np.multiply((1-y), np.log(1-h))
    J=np.sum(first-second)/len(y) +p
    return J
def gradient(theta,X,y,lamb):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    h=sigmoid(X*theta.T)
    grad=((X.T*(h-y))/len(y)).T + ((lamb/len(y))*theta)
    grad[0,0]=np.sum(np.multiply((h-y),X[:,0]))/len(y)
    return np.array(grad).ravel()
def oneVsAll(X,y,num_labels,lamb):
    m=X.shape[0]
    n= X.shape[1]
    all_theta= np.zeros((num_labels, n+1))
    one=np.ones((m,1))
    X=np.concatenate((one,X),axis=1)
    for i in range(1, num_labels + 1):
        theta = np.zeros(n + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (m, 1))
        # minimize the objective function
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i,lamb), method="TNC", jac=gradient)
        all_theta[i - 1, :] = fmin.x
    return all_theta


def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]
    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)
    # compute the class probability for each class on each training instance
    h = sigmoid(X * all_theta.T)
    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)
    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1

    return h_argmax

def add_bias(X):
    ones_col=np.ones(X.shape[0]).reshape(X.shape[0],1)
    X_bias=np.concatenate((ones_col,X),axis=1).T
    return X_bias
def layer1_to_layer2_output(theta,X):
    return sigmoid(theta@X)
def layer2_to_layer3_output(theta,X):
    return sigmoid(theta@X).T
