import numpy as np
import keras
from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.metrics import sparse_categorical_accuracy

def acc(args):
    tar, dis = args
    return sparse_categorical_accuracy(tar, dis)

distro = Input(shape=(3, 2), dtype='float32')
target = Input(shape=(3,), dtype='int32')
output = Lambda(cross_en)([target, distro])
model = Model([distro, target], output)

a = np.array([[[0.2, 0.8],[0.3, 0.7],[0.6, 0.4]],
              [[0.6, 0.4],[0.1, 0.9],[0.9, 0.1]]])
b = np.array([[1, 0, 0],
              [0, 1, 0]])

out = model.predict([a, b])
print('shape of out', out.shape)
print('out:')
print(out)
