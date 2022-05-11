import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm

def computeCost(X,y,theta):
    h_x=X*theta
    return (sum(h_x-y)*sum(h_x-y))/(2/m)
def gradientDescent(X,y,theta,alpha,num_inter):
    m=len(y)
    J=np.zeros((num_inter,1))
    for iter in range(0,num_inter):
        theta=theta-alpha/m*(X.T*(X*theta-y))
        J[iter]=computeCost(X,y,theta)
    return theta

data=np.loadtxt("ex1data1.txt",dtype=float,delimiter=",")
X_input=data[:,0]
y_input=data[:,1]

m=len(y_input)

ones=np.ones((m,1))
X=np.matrix(X_input)
X=X.T
y=np.matrix(y_input)
y=y.T

X=np.concatenate((ones,X),axis=1)

theta=np.zeros((2,1))

iterations = 1500
alpha = 0.01

J=computeCost(X,y,theta)
theta=gradientDescent(X,y,theta,alpha,iterations)

plt.scatter(X_input,y_input)
plt.plot(X_input, X*theta,'r')
a=np.array([1,3.5])
predict1=a*theta*10000
#Predict values for population sizes of 35,000 and 70,000
print("For population = 35,000, we predict a profit of",float(predict1))
b=np.array([1,7])
predict2=b*theta*10000
print("For population = 70,000, we predict a profit of",float(predict2))
theta0_vals=np.linspace(-10,10,100)
theta1_vals=np.linspace(-1,4,100)
J_vals=np.zeros((len(theta0_vals),len(theta1_vals)))
t=np.zeros((1,2))
for i in range(0,len(theta0_vals)):
    for j in range(0,len(theta1_vals)):
        t[0,0:2]=(theta0_vals[i],theta1_vals[j])
        J_vals[i,j]=computeCost(X,y,np.matrix(t).T)

J_vals=J_vals.T
fig = plt.figure()
ax = plt.axes(projection='3d')
# Creating plot
ax.plot_surface(theta0_vals, theta1_vals, J_vals)

plt.figure()
plt.contour(theta0_vals,theta1_vals,J_vals)
plt.show()

