import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Input, Activation, multiply, Lambda, Permute, RepeatVector, Flatten
from keras.layers import TimeDistributed, GRU, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import os
import numpy as np
import json
# from sklearn.preprocessing import LabelEncoder
# from keras.utils.np_utils import to_categorical

import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth = True   # 不全部占满显存, 按需分配
session = tf.Session(config=config)
KTF.set_session(session)


# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = True


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps, input_dim = int(inputs.shape[1]), int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    # a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(time_steps, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    weights = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, weights])
    output_attention = Lambda(lambda x: K.sum(x, axis=1))(output_attention_mul)
    return output_attention


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


def han(window=10, k_max=10):
    # refer to 4.2 in the paper while reading the following code
    # Input for one day : max article per day =40, dim_vec=200
    input1 = Input(shape=(k_max, 200), dtype='float32')
    att_article = att_chris(input1)
    att_article_model = Model(input1, att_article)
    input2 = Input(shape=(window, 200), dtype='float32')
    att_day = att_chris(input2)
    att_day_model = Model(input2, att_day)

    # Input of the HAN shape (None,11,0,200)
    # 11 = Window size = N in the paper 40 = max articles per day, dim_vec = 200
    input0 = Input(shape=(window, k_max, 200), dtype='float32')

    # TimeDistributed is used to apply a layer to every temporal slice of an input 
    # So we use it here to apply our attention layer ( pre_model ) to every article in one day
    # to focus on the most critical article
    pre_gru = TimeDistributed(att_article_model)(input0)

    # bidirectional gru
    l_gru = Bidirectional(GRU(100, return_sequences=True))(pre_gru)
    # l_gru.set_shape((None, window, 200))
    # We apply attention layer to every day to focus on the most critical day
    post_gru = att_day_model(l_gru)

    # MLP to perform classification
    dense1 = Dense(100, activation='tanh')(post_gru)
    dense2 = Dense(3, activation='tanh')(dense1)
    final = Activation('softmax')(dense2)
    final_model = Model(input0, final)
    final_model.summary()
    return final_model


'''
def load_data(model, x_train_file, x_test_file, y_train_file, y_test_file):
    x_train = np.load(x_train_file)
    y_train = np.load(y_train_file)

    x_test = np.load(x_test_file)
    y_test = np.load(y_test_file)


    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    y_train_end = to_categorical(encoded_Y)


    encoder2 = LabelEncoder()
    encoder.fit(y_test)
    encoded_Y2 = encoder.transform(y_test)
    y_test_end = to_categorical(encoded_Y2)
    print(y_test_end.shape)



    print("model compiling - Hierachical attention network")

    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])


    print("model fitting - Hierachical attention network")

    print(x_train.shape, y_test_end.shape)

    model.fit(x_train, y_train_end, epochs=200)


    print("validation_test")

    final_x_test_predict = model.predict(x_train)



    print("Prediction de Y ", final_x_test_predict)
    print("vrai valeur Y ", y_train)

    return

'''


def twin_creation(x_train_folder, y_train_folder):
    '''
    Here we create a list of twins ( duo_list)
    Twin = [CompanyA_x_train_filepath, CompanyA_y_train_filepath]
    '''

    x_train_list = os.listdir(x_train_folder)
    x_train_list = sorted(x_train_list)

    y_train_list = os.listdir(y_train_folder)
    y_train_list = sorted(y_train_list)

    ls = []
    for i in range(len(y_train_list)):
        item = [x_train_list[i], y_train_list[i]]
        ls.append(item)
    ls = [x for x in ls if x[0][-4:] == '.npy']
    return ls


def training(x_name, y_name, mod, path):
    x_train = np.load(path+'x_train/'+x_name)
    y_train = np.load(path+'y_train/'+y_name)
    rt = mod.train_on_batch(x_train, y_train)
    # print("model fitting on "+x_name)
    print(model.metrics_names, ':', rt, flush=True)


def generate_batch_from_folder(path, comp_list_fname):
    while True:
        for name in comp_list_fname:
            # print(name, flush=True)
            yield (np.load(path + name), np.load((path + name).replace('x', 'y')))


if __name__ == "__main__":
    model_save_path = '/mnt/cbrai/nlp/lcw/tmp/tr10/'
    train_data_path = '/mnt/data/nlp_data/news/data_set/'
    model = han(window=10, k_max=10)

    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # epochs = 60
    # duo_list = twin_creation(train_data_path + 'x_train/', train_data_path + 'y_train/')
    # for epoch in range(epochs):
    #     for k, duo in enumerate(duo_list):
    #         # print('fitting on firm no. {} out of {} epoch {}'.format(k, epochs, epoch))
    #         training(duo[0], duo[1], model, train_data_path)
    #     epoch += 1
    #     if epoch % 10 == 0:
    #         model.save(model_save_path + 'lcw_{}_epochs.hdf5'.format(epoch))

    # checkpoint
    filepath = model_save_path + "weights-improvement-{epoch:04d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max', period=1)

    x_path = train_data_path + 'x_train/'
    x_list = sorted(os.listdir(x_path))
    x_path_val = train_data_path + 'x_val/'
    # x_path_test = train_data_path + 'x_test/'
    # x_list_test = sorted(os.listdir(x_path_test))
    history = model.fit_generator(generate_batch_from_folder(x_path, x_list),
                                  steps_per_epoch=len(x_list), epochs=10,
                                  # validation_data=
                                  # generate_batch_from_folder(x_path_test, x_list_test),
                                  # validation_steps=len(x_list_test),
                                  validation_data=(np.load(x_path_val + 'x_val.npy'),
                                                   np.load((x_path_val + 'x_val.npy').replace('x', 'y'))),
                                  callbacks=[checkpoint])
    model.save(model_save_path + 'lcw_{}_epochs.hdf5'.format(10))
    with open(model_save_path + 'fit_history.json', 'w') as f:
        json.dump(history.history, f)

    # x_path = train_data_path + 'x_test/'
    # x_list = sorted(os.listdir(x_path))
    # eva_rt = model.evaluate_generator(generate_batch_from_folder(x_path, x_list), steps=len(x_list))
    # with open(model_save_path + 'evaluate.json', 'w') as f:
    #     json.dump(eva_rt, f)

    # input_atc = Input(shape=(20, 200), dtype='float32')
    # att_article_model_pre = Model(input_atc, att_chris(input_atc))
    # att_article_model_pre.set_weights(
    #     model.layers[1].get_weights())
    # att_model = Model(inputs=att_article_model_pre.input,
    #                   outputs=att_article_model_pre.layers[4].output)
    # att_weights = att_model.predict(X)
