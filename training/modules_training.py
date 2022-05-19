
from datetime import datetime
import os
import pickle
from datetime import datetime
import numpy as np
from sklearn.svm import SVC


def load_feature(path):
    X_HOG = np.loadtxt(path + "/HOG_feature.txt")
    X_PCA = np.loadtxt(path + "/PCA_feature.txt")
    Y = np.loadtxt(path + "/label.txt")
    print("Load feature done!!!")
    return X_HOG, X_PCA, Y


def randInitializeWeights(L_in, L_out):
    w = np.zeros((L_out, L_in+1))
    eps = np.sqrt(6) / (np.sqrt(L_in) + np.sqrt(L_out))
    w = - eps + np.random.rand(L_out, L_in+1) * 2 * eps
    return w


def computeNumericalGradient(J, theta):
    numgrad = np.zeros(len(theta))
    perturb = np.zeros(len(theta))
    e = 1e-4
    for p in range(len(theta)):
        # Set perturbarion vactor
        perturb[p] = e
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)

        # compute numerical gradient
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0
    return numgrad


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def sigmoidGradient(z):
    dg = sigmoid(z) * (1 - sigmoid(z))
    return dg


def y_Vec(num_labels, y):
    yy = np.zeros((len(y), num_labels))
    for i in range(yy.shape[0]):
        yy[i, int(y[i])-1] = 1
    return yy


def nncnnCostFunction(nn_params, input_layer_size, hidden_layer1_size, hidden_layer2_size
                      , hidden_layer3_size, num_labels, X, y, lamb):

    theta1 = np.reshape(nn_params[0: hidden_layer1_size * (input_layer_size + 1)],
                        (hidden_layer1_size, input_layer_size+1), order='F')
    theta2 = np.reshape(nn_params[hidden_layer1_size * (input_layer_size + 1): hidden_layer1_size  * (input_layer_size + 1) + hidden_layer2_size * (hidden_layer1_size+1) ],
                        (hidden_layer2_size, hidden_layer1_size + 1), order='F')
    a = hidden_layer1_size * (input_layer_size + 1) + hidden_layer2_size * (hidden_layer1_size+1)
    theta3 = np.reshape(nn_params[ a: a + hidden_layer3_size * (hidden_layer2_size+1)],
                        (hidden_layer3_size, hidden_layer2_size+1), order='F')
    theta4 = np.reshape(nn_params[ a + hidden_layer3_size * (hidden_layer2_size+1): a + hidden_layer3_size * (hidden_layer2_size + 1)+(hidden_layer3_size+1)*num_labels],
                        (num_labels, hidden_layer3_size +1), order='F')
    eps = 1e-5
    m = X.shape[0]
    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)
    theta3_grad = np.zeros(theta3.shape)
    theta4_grad = np.zeros(theta4.shape)

    X = np.hstack((np.ones((m, 1)), X))
    a1 = X

    z2 = np.dot(a1, theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((a2.shape[0], 1)), a2))

    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)
    a3 = np.hstack((np.ones((a3.shape[0], 1)), a3))

    z4 = np.dot(a3, theta3.T)
    a4 = sigmoid(z4)
    a4 = np.hstack((np.ones((a4.shape[0], 1)), a4))

    z5 = np.dot(a4, theta4.T)
    a5 = sigmoid(z5)

    h_x = a5
    J = -1 / m * np.sum((np.sum(np.log(h_x) * y) + np.sum(np.log(1 - h_x + eps) * (1 - y))
                         - (lamb / 2) * (np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2) + np.sum(theta3[:, 1:] ** 2) + np.sum(theta4[:, 1:] ** 2))))
    delta5 = h_x - y
    delta4 = np.dot(delta5, theta4)[:, 1:] * sigmoidGradient(z4)
    delta3 = np.dot(delta4, theta3)[:, 1:] * sigmoidGradient(z3)
    delta2 = np.dot(delta3, theta2)[:, 1:] * sigmoidGradient(z2)

    Delta1 = np.dot(delta2.T, X)
    Delta2 = np.dot(delta3.T, a2)
    Delta3 = np.dot(delta4.T, a3)
    Delta4 = np.dot(delta5.T, a4)

    theta1_grad = 1 / m * Delta1
    theta2_grad = 1 / m * Delta2
    theta3_grad = 1 / m * Delta3
    theta4_grad = 1 / m * Delta4

    theta1_grad[:, 1:] += lamb / m * theta1[:, 1:]
    theta2_grad[:, 1:] += lamb / m * theta2[:, 1:]
    theta3_grad[:, 1:] += lamb / m * theta3[:, 1:]
    theta4_grad[:, 1:] += lamb / m * theta4[:, 1:]

    grad = np.concatenate((np.reshape(theta1_grad, theta1_grad.size, order='F'),
                           np.reshape(theta2_grad, theta2_grad.size, order='F'),
                           np.reshape(theta3_grad, theta3_grad.size, order='F'),
                           np.reshape(theta4_grad, theta4_grad.size, order='F')))
    return (J, grad)


def predict(theta1, theta2, theta3, theta4, X):
    # size
    m, n = X.shape
    num_labels = theta4.shape[0]
    # add ones to X data matrix
    X = np.hstack((np.ones((m,1)), X))

    z2 = np.dot(X, theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((a2.shape[0], 1)), a2))

    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)
    a3 = np.hstack((np.ones((a3.shape[0], 1)), a3))

    z4 = np.dot(a3, theta3.T)
    a4 = sigmoid(z4)
    a4 = np.hstack((np.ones((a4.shape[0],1)), a4))

    z5 = np.dot(a4, theta4.T)
    a5 = sigmoid(z5)

    p = np.argmax(a5, axis=1)
    return p


def ANN(X_train, y_train, hidden_layer1_size, hidden_layer2_size, hidden_layer3_size, number_labels):

    input_layer_size = X_train.shape[1]
    theta1 = randInitializeWeights(input_layer_size, hidden_layer1_size)
    theta2 = randInitializeWeights(hidden_layer1_size, hidden_layer2_size)
    theta3 = randInitializeWeights(hidden_layer2_size, hidden_layer3_size)
    theta4 = randInitializeWeights(hidden_layer3_size, number_labels)

    init_nn_params = np.concatenate((np.reshape(theta1, theta1.size, order = 'F'), np.reshape(theta2, theta2.size, order = 'F'),
                                     np.reshape(theta3, theta3.size, order = 'F'), np.reshape(theta4, theta4.size, order = 'F')))
    y = y_train
    y_train = y_Vec(number_labels, y_train)
    cost = lambda x: nncnnCostFunction(x, input_layer_size, hidden_layer1_size, hidden_layer2_size,
                                       hidden_layer3_size, number_labels, X_train, y_train, lamb=1)[0]
    grad = lambda x: nncnnCostFunction(x, input_layer_size, hidden_layer1_size, hidden_layer2_size,
                                       hidden_layer3_size, number_labels, X_train, y_train, lamb=1)[1]

    nn_params = fmin_cg(cost, init_nn_params, fprime=grad, maxiter=500, disp=False)
    theta1 = np.reshape(nn_params[0: hidden_layer1_size * (input_layer_size + 1)],
                        (hidden_layer1_size, input_layer_size+1), order='F')
    theta2 = np.reshape(nn_params[hidden_layer1_size * (input_layer_size + 1): hidden_layer1_size * (input_layer_size + 1 ) + hidden_layer2_size * (hidden_layer1_size+1)],
                        (hidden_layer2_size, hidden_layer1_size + 1), order='F')
    a = hidden_layer1_size * (input_layer_size + 1) + hidden_layer2_size * (hidden_layer1_size+1)
    theta3 = np.reshape(nn_params[a: a + hidden_layer3_size * (hidden_layer2_size + 1)],
                        (hidden_layer3_size, hidden_layer2_size + 1), order='F')
    theta4 = np.reshape(nn_params[a + hidden_layer3_size * (hidden_layer2_size+1): a + hidden_layer3_size * (hidden_layer2_size + 1)+(hidden_layer3_size+1) * number_labels],
                        (number_labels, hidden_layer3_size + 1), order='F')
    
    # p = predict(theta1, theta2, theta3, theta4, X_test)
    # print('Training set Accuracy: %2.2f ' % (np.mean(p +1 == y_test) * 100)+"%")
    return theta1, theta2, theta3, theta4


def SVM_fn(X_train, Y_train, filename,  c_kernel='poly', c_degree=2, c_gamma=10, c=10):
    print("\nTraining ...")
    print("kernel = ", c_kernel)
    svm = SVC(kernel=c_kernel, degree=c_degree, gamma=c_gamma, C=c, probability=True)
    svm.fit(X_train, Y_train)

    # save the model to disk
    filename = filename + str(datetime.now()).replace(":", "_")+".sav"
    pickle.dump(svm, open(filename, 'wb'))

    print("Successful! Mode save at: "+os.path.abspath(filename))
    return(svm)


def save_result_train(path, result_svm_hog, result_svm_pca):
    if not os.path.exists(path):
        os.mkdir(path)

    file_history = open("./" + path + "/history_train.txt", "a")

    file_history.write("SVM-PCA ")
    file_history.writelines(result_svm_pca)
    file_history.write('\n')

    file_history.write("SVM-HOG ")
    file_history.writelines(result_svm_hog)
    file_history.write('\n')

    file_history.close()