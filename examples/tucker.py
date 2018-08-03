import tensorflow as tf
import numpy as np
from tf_decompose import TuckerTensor

X = np.array([[1, 2], [3, 4]], dtype=np.float32)
TT_X = TuckerTensor(X.shape, ranks=[2, 2], dtype=tf.float32)
opt = tf.train.AdadeltaOptimizer(0.05)
X_predict = TT_X.train_als_early(X, opt, epochs=10000)
#X_predict = TT_X.hooi(X, epochs=10000)
#X_predict = TT_X.hosvd(X)
print(X_predict)
