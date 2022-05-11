import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.optimize import minimize, rosen, rosen_der


def plotData(X,y):
    for i in range(0,len(y)):
        if y[i]==0:
            plt.plot(X[i,0],X[i,1],"r*")
        else:
            plt.plot(X[i,0], X[i, 1],"go")
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")


def sigmoid(z):
    return(1 / (1 + np.exp(-z)))

def costFunction(theta, X, y):
    m = y.size
    htheta = sigmoid(X.dot(theta))
    J = -1 * (1 / m) * (np.log(htheta).T.dot(y) + np.log(1 - htheta).T.dot(1 - y))
    if np.isnan(J[0]):
        return (np.inf)
    return (J[0])
def gradient(theta,X,y):
    m=len(y)
    htheta = sigmoid(X.dot(theta.reshape(-1,1)))
    grad = 1/m * X.T.dot(htheta - y)
    return (grad.flatten())

# def mapFeature(X1,X2):
#     degree=6
#     out=np.ones((X1[:,0].shape))
#     for i in range(0,degree):
#         for j in range(0,i):
#             out[:,out.shape[1]+1]=(X1**(i-j))*(X2**j)
def plotDecisionBoundary(theta,X,y):
    plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)')
    plotData(X[:, 1:3], y)
    x1_min, x1_max = X[:, 1].min(), X[:, 1].max(),
    x2_min, x2_max = X[:, 2].min(), X[:, 2].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(res.x))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
def predict(theta,X):
    m=X.shape[0]
    p=np.zeros((m,1))
    htheta=sigmoid(X.dot(theta))
    p=htheta>=0.5
    return p
data= np.loadtxt("ex2data1.txt",dtype=float,delimiter=",")
X=np.c_[data[:,0:2]]
y=np.c_[data[:,2]]

plotData(X,y)
plt.show()
m,n=X.shape

ones=np.ones((100,1))
X=np.concatenate((ones,X),axis=1)
initial_theta=np.zeros(X.shape[1])
cost=costFunction(initial_theta,X,y)
grad=gradient(initial_theta,X,y)
print("Cost at initial theta (zeros):",cost)
print("Gradient at initial theta (zeros): :",grad)
res = minimize(costFunction, initial_theta, args=(X,y), method=None, jac=gradient, options={'maxiter':400})

p = predict(res.x, X)

print('Train accuracy {}%'.format(100*np.mean(p == y.ravel())))


plotDecisionBoundary(initial_theta,X,y)
plt.show()