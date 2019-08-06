# -*- coding: utf-8 -*-

from keras.models import load_model
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from news_snf import snf
from news_snf import predict
from utils.preprocess import segment
from utils.preprocess import clean_sent
from utils.preprocess import not_ch_or_stop


d2v = Doc2Vec.load('/mnt/data/nlp_data/news/'+'d2v_200/'+'d2v.model')
model = load_model('/mnt/cbrai/nlp/lcw/tmp/tr9/'+'weights-improvement-0001-0.4525.hdf5')
shape = (1, 10, 10, 200)
ma, mm, md = snf(shape, model)


def snf_score(news=[]):
    """
    :param news: ['raw_news_text-1',...,'raw_news_text-t']
    :return snf_score:{'articles':[[],[],...,[]],'days':[]}
    """
    ss = {}
    x = np.zeros(shape, dtype='float32')
    for i in range(len(news)):
        x[0, i//shape[2], i % shape[2], :] = d2v.infer_vector(clean_sent(
            segment(news[i], mode='seg'), filter_func=not_ch_or_stop))
    a1, a2 = predict(x, ma, mm, md)
    ss['articles'] = np.squeeze(a1).tolist()
    ss['days'] = np.squeeze(a2).tolist()
    return ss


if __name__ == "__main__":
    news = ['中文zhongwen', '测试', '简单']
    scores = snf_score(news)
