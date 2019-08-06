# -*- coding: utf-8 -*-
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, Activation, multiply, Lambda, Permute, RepeatVector, Flatten
from keras.layers import TimeDistributed, GRU, Bidirectional
from keras import backend as K
from keras.models import load_model
import os
import numpy as np
import json


import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth = True   # 不全部占满显存, 按需分配
session = tf.Session(config=config)
KTF.set_session(session)


def att_chris(inputs):
    time_steps, input_dim = int(inputs.shape[1]), int(inputs.shape[2])
    a = Dense(input_dim, activation='tanh')(inputs)
    a = Dense(1, activation='linear', use_bias=False)(a)
    a = Flatten()(a)
    a = Activation('softmax')(a)
    a = RepeatVector(input_dim)(a)
    weights = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, weights])
    output_attention = Lambda(lambda x: K.sum(x, axis=1))(output_attention_mul)
    return output_attention


def snf(shape, model):
    _, w, n, d = shape
    input_atc = Input(shape=(n, d), dtype='float32')
    att_article_model_pre = Model(input_atc, att_chris(input_atc))
    att_article_model_pre.set_weights(model.layers[1].get_weights())
    att_model = Model(inputs=att_article_model_pre.input,
                      outputs=att_article_model_pre.layers[4].output)
    # att_weights = np.zeros((b, w, n), dtype='float32')
    # # TODO x遍历
    # for i in range(b):
    #     for j in range(w):
    #         att_weights[i, j, :] = att_model.predict(np.expand_dims(x[i, j], axis=0))

    before_att_layer = Model(model.input, model.layers[2].output)
    # before_att_layer_out = before_att_layer.predict(x)
    input_atd = Input(shape=(n, d), dtype='float32')
    att_day_model_pre = Model(input_atd, att_chris(input_atd))
    att_day_model_pre.set_weights(model.layers[3].get_weights())
    att_layer = Model(att_day_model_pre.input, att_day_model_pre.layers[4].output)
    # att_weights_day = att_layer.predict(before_att_layer_out)

    return att_model, before_att_layer, att_layer


def predict(x, ma, mm, md):
    b, w, n, d = x.shape
    # TODO x遍历
    att_weights = np.zeros((b, w, n), dtype='float32')
    for i in range(b):
        for j in range(w):
            att_weights[i, j, :] = ma.predict(np.expand_dims(x[i, j], axis=0))
    before_att_layer_out = mm.predict(x)
    att_weights_day = md.predict(before_att_layer_out)

    return att_weights, att_weights_day


if __name__ == "__main__":

    model = load_model('/mnt/cbrai/nlp/lcw/tmp/tr9/weights-improvement-0001-0.4525.hdf5')
    x = np.load('/mnt/data/nlp_data/news/data_set2/' + 'x_train/' + '10000001_x_train.npy')
    ma, mm, md = snf(x.shape, model)
    x = np.expand_dims(x[0], axis=0)
    att1, att2 = predict(x, ma, mm, md)


