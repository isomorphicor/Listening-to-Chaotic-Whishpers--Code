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

#%%
import pickle
import os


def test_pickle(path):

    """
    Tester la création de nos fichier pickle
    :param path: chemin du fichier
    :return:
    """

    if os.path.getsize(article) > 0:
        with open(article, 'rb') as handle:
            unpickler = pickle.Unpickler(handle)
            b = unpickler.load()
    print(b)

#%%
if __name__ == '__main__':
    article = '/mnt/cbrai/nlp/lcw/firm_csv_folder_old/pickle/3M.csv.pkl'
    test_pickle(article)

#%%
article = '/mnt/cbrai/nlp/lcw/stock_values/pickle_stock_value/3M .csv.pkl'
pkl_file = open(article, 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

#%%
from .utils.preprocess import segment
ss = segment('中文世界nlp自然语言处理技术', 'seg')



