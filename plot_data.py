import numpy as np
import matplotlib.pyplot as plt
from features_extraction.modules_features_extraction import *
from sklearn import decomposition

X = np.loadtxt('../output_data_train/data.txt')
Y = np.loadtxt('../output_data_train/label.txt')


pca = decomposition.PCA(3)
pca_data = pca.fit_transform(X)

name_classes = ['brocoli', 'carrot', 'eggplant', 'potato', 'tomato']

num_samples = Y.shape[0]

nc = [0, int(num_samples/5)-1, 2*int(num_samples/5)-1, 3*int(num_samples/5)-1, 4*int(num_samples/5)-1, num_samples-1]
ax = plt.axes(projection='3d')
for i in range(len(name_classes)):
    ax.scatter(pca_data[nc[i]:nc[i+1],0], 
                pca_data[nc[i]:nc[i+1],1], 
                pca_data[nc[i]:nc[i+1],2], 
                c=Y[nc[i]:nc[i+1]])
                # c=Y[nc[i]:nc[i+1]], label = name_classes[i])
    # plt.legend()
    plt.title("Figure_Data")
plt.show()

# sklearn.metrics.confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None)