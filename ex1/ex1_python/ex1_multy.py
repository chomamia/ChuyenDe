import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm

def computeCostMulti(X,y,theta):
    m=y.shape[0]
    J=0
    J=((X*theta-y).T)*(X*theta-y)/2/m
    return J
def gradientDescentMulti(X,y,theta,alpha,num_inter):
    m = y.shape[0]
    J = np.zeros((num_inter, 1))
    for iter in range(0, num_inter):
        theta = theta - alpha / m * (X.T * (X * theta - y))
        J[iter] = computeCostMulti(X, y, theta)
    return theta,J
def featureNormalize (X):
    X_norm=X
    mu=np.zeros((1,2))
    sigma=np.zeros((1,2))
    mu[:,0],mu[:,1]=np.mean(X[:,0]),np.mean(X[:,1])
    sigma[:,0],sigma[:, 1]=np.std(X[:,0]),np.std(X[:, 1])
    X_norm=(X-mu)/sigma
    return X_norm,mu,sigma
def normalEqn(X,y):
    theta=np.zeros((X.shape[1],1))
    theta=np.linalg.inv(X.T*X)*X.T*y
    return theta


data=np.loadtxt("ex1data2.txt",dtype=int,delimiter=",")
X_input=data[:,0:2]
y_input=data[:,2]
X=np.matrix(X_input)
y=np.matrix(y_input)
y=y.T

m=len(y_input)
ones=np.ones((m,1))
X,mu,sigma=featureNormalize(X)

X=np.concatenate((ones,X),axis=1)

alpha=0.01
num_iters = 400

theta=np.zeros((3,1))
theta,J=gradientDescentMulti(X,y,theta,alpha,num_iters)

x=np.arange(1,401)
plt.figure()
plt.plot(x,J)
plt.show()


data=np.loadtxt("ex1data2.txt",dtype=int,delimiter=",")
X_input=data[:,0:2]
y_input=data[:,2]
X=np.matrix(X_input)
y=np.matrix(y_input)
y=y.T
m=len(y_input)
ones=np.ones((m,1))
X=np.concatenate((ones,X),axis=1)
theta1=normalEqn(X,y)
print(theta1)
