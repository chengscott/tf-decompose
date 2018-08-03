import tensorflow as tf
import numpy as np
from tf_decompose import KruskalTensor

X = np.array([[1, 2], [3, 4]], dtype=np.float32)
KT_X = KruskalTensor(X.shape, rank=2, dtype=tf.float32)
opt = tf.train.AdadeltaOptimizer(0.05)
X_predict = KT_X.train_als_early(X, opt, epochs=10000)
print(X_predict)
