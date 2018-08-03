from tf_decompose import TuckerTensor
import tensorflow as tf
import numpy as np
from scipy.io.matlab import loadmat
import logging

logging.basicConfig(filename='brod_tucker.log', level=logging.DEBUG)

# Load sensory bread data (http://www.models.life.ku.dk/datasets)
mat = loadmat('data/bread/brod.mat')
X = mat['X'].reshape([10, 11, 8])

T = TuckerTensor(X.shape, ranks=[3, 3, 3], regularize=0.0, init='random')

#X_predict = T.train_als(X, tf.train.AdadeltaOptimizer(0.05), epochs=15000)
#X_predict = T.hooi(X, epochs=100)
X_predict = T.hosvd(X)

# Save reconstructed tensor to file
np.save('brod_tucker.npy', X_predict)
