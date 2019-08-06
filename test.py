import tensorflow as tf
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Input, Activation, multiply, Lambda, Permute, RepeatVector
from keras.layers import TimeDistributed, GRU, Bidirectional, LSTM
from keras import backend as K
import os
import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from keras.utils.np_utils import to_categorical

import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth = True   # 不全部占满显存, 按需分配
session = tf.Session(config=config)
KTF.set_session(session)


model = Sequential()
model.add(LSTM(10, return_sequences=True, input_shape=(5, 10)))
model.output.set_shape((None, 5, 10))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')