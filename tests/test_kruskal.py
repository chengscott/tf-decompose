from tf_decompose import KruskalTensor
import tensorflow as tf
import numpy as np
from scipy.io.matlab import loadmat
import logging

logging.basicConfig(filename='brod_kruskal.log', level=logging.DEBUG)

# Load sensory bread data (http://www.models.life.ku.dk/datasets)
mat = loadmat('data/bread/brod.mat')
X = mat['X'].reshape([10, 11, 8])

# Build ktensor and learn CP decomposition using ALS with specified optimizer
T = KruskalTensor(X.shape, rank=3, regularize=1e-6, init='nvecs', X_data=X)
opt = tf.train.AdadeltaOptimizer(0.05)
X_predict = T.train_als(X, opt, epochs=20000)

# Save reconstructed tensor to file
np.save('brod_kruskal.npy', X_predict)
