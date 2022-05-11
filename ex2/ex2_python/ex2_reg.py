import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.optimize import minimize, rosen, rosen_der
from sklearn.preprocessing import PolynomialFeatures

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
def mapFeature(X1,X2):
    degree=6
    out=np.ones((X1[:,0].shape))
    for i in range(0,degree):
        for j in range(0,i):
            out[:,out.shape[1]+1]=(X1**(i-j))*(X2**j)

def costFunctionReg(theta, reg, *args):
    m = y.size
    h = sigmoid(X_1.dot(theta))

    J = -1 * (1 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y)) + (reg / (2 * m)) * np.sum(np.square(theta[1:]))

    if np.isnan(J[0]):
        return (np.inf)
    return (J[0])

def gradientReg(theta, reg, *args):
    m = len(y)
    h = sigmoid(X_1.dot(theta.reshape(-1, 1)))
    grad = (1 / m) * X_1.T.dot(h - y) + (reg / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]
    return (grad.flatten())
def predict(theta,X):
    m=X.shape[0]
    p=np.zeros((m,1))
    htheta=sigmoid(X.dot(theta))
    p=htheta>=0.5
    return p
def plotDecisionBoundary(X,y,lamb):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max(),
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(res.x))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g');
    plt.title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), lamb))
data= np.loadtxt("ex2data2.txt",dtype=float,delimiter=",")
X=np.c_[data[:,0:2]]
y=np.c_[data[:,2]]
print(y)
poly = PolynomialFeatures(6)
X_1 = poly.fit_transform(data[:, 0:2])
X_1.shape

plotData(X,y)
plt.show()
m,n=X.shape

initial_theta = np.zeros(X_1.shape[1])
cost=costFunctionReg(initial_theta, 1, X_1, y)
print("Cost at initial theta (zeros):",cost)

lamb=0
res = minimize(costFunctionReg, initial_theta, args=(lamb, X_1, y), method=None, jac=gradientReg, options={'maxiter': 3000})
accuracy = 100 * np.mean(predict(res.x, X_1) == y.ravel())
print("train accuracy: ",accuracy)
plotData(X,y)
plotDecisionBoundary(X,y,lamb)
plt.show()