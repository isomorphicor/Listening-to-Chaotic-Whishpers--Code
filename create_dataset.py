import pickle
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.model_selection import train_test_split
import keras
import json
# import os
# import multiprocessing as mp

data_dir = '/mnt/data/nlp_data/news/'
data_set = 'data_set/'


def create_x_y(name, model, firm_article, firm_stock, k_max=40, test_size=0.33, window=10):
    """
    Creates a 4 dimensions array.Takes a lot of mem
    :param name: name of the company
    :param model: doc2vec model that give the vector associated to each article
    # :param firm_article: pickle file contening a dictionnary day : [ list of articles of the day ]
    :param firm_article: {day : [ list of articles of the day ]}
    # :param firm_stock: pickle file contening a  dictionnary day : stock variation of that day
    :param firm_stock: {day : trend}
    :param k_max: max number of article per day
    :param test_size: ratio of data that we keep for testing
    :param window: number of pasted days used for the prediction
    :return: x_train, x_test, y_train, y_test
    """

    dico_article = firm_article
    dico_stock = firm_stock

    # check if both dict are non empty
    if (len(dico_article) > 0) and (len(dico_stock) > 0):
        ''' So we build a 4 dimensions array
        # i, 1st dimension the corresponds to the days for which we have got articles about the company over the last 10 days
        # j, 2nd dimension is the window, number of pasted days used to make the prediction
        # k, 3rd dimension is the number of articles that we've got on the corresponding day
        # l, 4th dimension is the vector representing the article. In 200 dimensions'''

        # creating the array
        data = np.zeros((1, window, k_max, 200), dtype='float32')

        y = []
        dates = []

        for i in range(int(1095)):  # 2013-01-01 至 2015-12-31 共1095天(2013-01-01 至 2014-12-31 共730天)
            # bool to know if any article was puslihed during the window days
            to_add = False
            # we want to predict the trend on day i+1,
            # so we check if day i+1 is actually a key in dico_stock dictionnary
            next_day_key = i+1  # str(i + 1).zfill(4)
            if next_day_key in dico_stock:
                y_i = int(dico_stock[next_day_key])
                # new row to add to the data array
                new_row = np.zeros((1, window, k_max, 200), dtype='float32')
                for j in range(window):
                    # we look from i-k_max to i
                    day_key = i-j  # str(i - j).zfill(4)
                    if day_key in dico_article:
                        list_article = dico_article[day_key]
                        to_add = True
                        # k= key, x=value=list of ID of articles of that day
                        for k in range(k_max):
                            if k < len(dico_article[day_key]):
                                article_id = list_article[k]
                                vector = model.docvecs[article_id]
                                # vector = model.infer_vector(json.load(open(data_dir+'news_stocks/' +
                                #                                           article_id+'.json', 'r')))
                                new_row[0, j, k, :] = vector
                            else:
                                new_row[0, j, k, :] = np.zeros(200)
                if to_add:
                    # we add the line
                    data = np.vstack([data, new_row])
                    y.append(y_i)
                    dates.append(next_day_key)
            # if i % 10 == 0:
                # print(i)

        y_vec = np.asarray(y)
        # deletes the first line full of zeros 删除为了堆叠数据用的第一个初始块:code-line38
        x_mat = np.delete(data, 0, axis=0)
        x_train, x_test, y_train, y_test = train_test_split(
            x_mat, y_vec, test_size=test_size, random_state=42, shuffle=False)

        # x_val = []
        # y_val = []
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.05, random_state=42, shuffle=False)

        y_train_size = y_train.shape[0]
        dates_test = dates[y_train_size:]

        with open(data_dir + data_set + 'dates/' + str(name) + '.pkl', 'wb') as handle:
            pickle.dump(dates_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return x_train, x_test, x_val, y_train, y_test, y_val
    else:
        print("Error File")
        return [], [], [], [], [], []


# just a test
def open_numpy():
    x1_test = np.load(data_dir + data_set + 'x_train/' + '10000001_x_train.npy')
    print(x1_test.shape)


def create_dataset(comp_list=[]):
    """
    function qui permet de lancer la création des datasets
    # :param stockfile_list: liste des fichiers de cours de la bourses pour n entreprises
    # :param articlefile_list: liste des fichiers d'articles pour n entreprises
    :param comp_list:公司列表
    :return:
    """
    with open(data_dir+'comp_newsid.pkl', 'rb') as handle:
        news = pickle.load(handle)
    with open(data_dir+'comp_sig.pkl', 'rb') as handle:
        sigs = pickle.load(handle)
    comp_list = sigs.keys()
    # loading d2v model
    model = Doc2Vec.load(data_dir+'d2v_200/'+'d2v.model')

    x_val_a = np.zeros((1, 10, 10, 200), dtype='float32')
    y_val_a = np.zeros((1, 3), dtype='float32')
    for comp in comp_list:
        print(comp, flush=True)
        x_train, x_test, x_val, y_train, y_test, y_val = create_x_y(comp, model, news[comp], sigs[comp], 10, 0.2, 10)
        if len(x_train) > 0:
            y_train = keras.utils.to_categorical(y_train+1, num_classes=3)
            y_test = keras.utils.to_categorical(y_test+1, num_classes=3)
            y_val = keras.utils.to_categorical(y_val + 1, num_classes=3)

            # Create numpy files
            np.save(data_dir + data_set + 'x_train/' + str(comp) + '_x_train.npy', x_train)
            np.save(data_dir + data_set + 'x_test/' + str(comp) + '_x_test.npy', x_test)
            np.save(data_dir + data_set + 'y_train/' + str(comp) + '_y_train.npy', y_train)
            np.save(data_dir + data_set + 'y_test/' + str(comp) + '_y_test.npy', y_test)
            x_val_a = np.vstack([x_val_a, x_val])
            y_val_a = np.vstack([y_val_a, y_val])

    x_val_a = np.delete(x_val_a, 0, axis=0)
    y_val_a = np.delete(y_val_a, 0, axis=0)
    np.save(data_dir + data_set + 'x_val/' + 'x_val.npy', x_val_a)
    np.save(data_dir + data_set + 'y_val/' + 'y_val.npy', y_val_a)


if __name__ == "__main__":
    create_dataset()








