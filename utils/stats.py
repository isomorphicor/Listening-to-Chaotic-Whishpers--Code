# encoding: utf-8
"""
@author: Yuxian Meng 
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: stats
@time: 2018/11/12 11:39

    用于做一些基础的统计，如新闻的句数分布，句长分布
"""

import json
import os
import re
from collections import Counter
from common import project_dir
from data_analyze.sentiment_analysis.utils.preprocess import clean_sent
from torch.utils.data import Dataset, DataLoader
import numpy as np
from db import get_new_session, get_new_chinabond_session
from orm.news import News


def sep_text(txt, return_index=False, ignore_space=False):
    """
    将一段中文txt切分为句子
    :param txt: str或list of str或list of tuple
    :return: list of str
    """
    sentences = []
    stack = []
    after2before = {'”': '“', '）': '（', }
    end_signals = ['。', '……', '！', '？']
    sentence = []
    l = len(txt)
    for i in range(l):
        ch = txt[i]
        if not ignore_space and ch in ['\n', '\t', '\r', '\f', '\v', ' ']:
            continue
        if return_index:
            sentence.append(i)
        else:
            sentence.append(ch)
        if ch in after2before.values():
            stack.append(ch)
        if ch in after2before:
            pre = after2before[ch]
            if pre in stack:
                stack.remove(pre)
        if ((ch in end_signals and i<l-1 and txt[i+1] not in after2before.keys()) or
                (ch in after2before.keys() and txt[i-1] in end_signals)) and not stack:
            sentences.append(sentence)
            sentence = []
    if sentence:
        sentences.append(sentence)
    return sentences


def stat_len(json_file='/data/yuxian/data/sentiment_jsons_small'):
    """统计句长和句数分布"""
    jsons = os.listdir(json_file)
    token_counter = Counter()
    sent_counter = Counter()
    gonggao = 0
    for idx, js in enumerate(jsons[:]):
        if (idx+1) % 10000 == 0:
            print(idx, gonggao)
        json_path = os.path.join(json_file, js)
        title, abstract, score, compname = json.load(open(json_path))
        title.append('。/w')
        if '公告' in title:
            gonggao += 1
        text = title + abstract
        text = [w.split('/') for w in text if len(w.split('/')) == 2]
        text, pos = zip(*text)
        sentences = sep_text(text)
        sentences = [clean_sent(sent, return_offsets=False) for sent in sentences]
        tokens = [len(s) for s in sentences]
        sents = [len(sentences)]
        sent_counter.update(sents)
        token_counter.update(tokens)
    dout = os.path.join(project_dir, 'data_analyze/data/sentiment_analysis')
    if not os.path.exists(dout):
        os.mkdir(dout)
    fout = os.path.join(dout, 'len_stats_new.json')
    json.dump([token_counter, sent_counter], open(fout, 'w'))


def stat_invalid(json_file='/data/yuxian/data/sentiment_jsons_full'):
    """统计一些噪音样本的idx"""
    invalid_titles = json.load(open(os.path.join(project_dir, "data_analyze/data/sentiment_analysis/sentiment_jsons_invalid.json")))
    invalid_titles = set([' '.join(words) for words in invalid_titles])
    jsons = os.listdir(json_file)
    invalid_idx = []
    announce_words = set(r"公告|报告|工作函|股东大会|决议|议事|通报|大宗交易|年报|复牌|停牌|持股变动|"
                         r"受托|专项|证券化".split('|'))
    for idx, js in enumerate(jsons[:]):
        if idx % 10000 == 0:
            print(idx)
        json_path = os.path.join(json_file, js)
        title, abstract, score = json.load(open(json_path))
        valid = True
        for word in title:
            if word in announce_words:
                valid = False
                invalid_idx.append(idx)
                break
        title = ' '.join(title)
        if valid and title in invalid_titles:
            invalid_idx.append(idx)
    dout = os.path.join(project_dir, 'data_analyze/data/sentiment_analysis')
    if not os.path.exists(dout):
        os.mkdir(dout)
    fout = os.path.join(dout, 'invalid_idx.json')
    print(len(invalid_idx))
    json.dump(invalid_idx, open(fout, 'w'))


def stat_freq(json_file='/data/yuxian/data/sentiment_jsons_small'):
    """统计tf和idf"""
    jsons = os.listdir(json_file)
    freq_counter = Counter()
    idf_counter = Counter()
    for idx, js in enumerate(jsons[:]):
        if (idx+1) % 10000 == 0:
            print(idx)
        json_path = os.path.join(json_file, js)
        title, abstract, score, compname = json.load(open(json_path))
        title.append('。/w')
        text = title + abstract
        freq_counter.update(text)
        text = set(text)
        idf_counter.update(text)
    dout = os.path.join(project_dir, 'data_analyze/data/sentiment_analysis')
    if not os.path.exists(dout):
        os.mkdir(dout)
    fout = os.path.join(dout, 'tf_idf_stats.json')
    json.dump({'tf': freq_counter, 'idf': idf_counter}, open(fout, 'w'))

    data = json.load(open(os.path.join(project_dir, 'data_analyze/data/sentiment_analysis/tf_idf_stats.json')))
    word2tf, word2idf = data['tf'], data['idf']
    word_and_tf = list(word2tf.items())
    word_and_idf = list(word2idf.items())
    word_and_tf = sorted(word_and_tf, key=lambda x: x[1])
    word_and_idf = sorted(word_and_idf, key=lambda x: x[1])
    print(word2tf, word2idf)
    json.dump({'word_and_tf': word_and_tf, 'word_and_idf': word_and_idf},
              open(os.path.join(project_dir, 'data_analyze/data/sentiment_analysis/word_tfidf_lists.json'), 'w'))

if __name__ == '__main__':
    # stat_len()
    # stat_invalid()
    # stat_freq()

    # -------统计长度比例，找到合适的截取值，目前输出如下
    #ratio:[0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
    #sent: ['32', '38', '46', '61', '88']
    #token:['27', '30', '35', '44', '58', '72']
    #
    # token_counter, sent_counter = json.load(open(os.path.join(project_dir,
    #                                                      'data_analyze/data/sentiment_analysis/len_stats_new.json')))
    # sent_num = sum([v for v in token_counter.values()])
    # article_num = sum([v for v in sent_counter.values()])
    # sent_num_accumulate = 0
    # token_num_accumulate = 0
    # i = 0
    # ratio_thresholds = [0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
    # sent_thresholds = []
    # token_thresholds = []
    # sent_ratio_idx = token_ratio_idx = 0
    # while sent_ratio_idx < len(ratio_thresholds)-1 or token_ratio_idx < len(ratio_thresholds)-1:
    #     idx = str(i)
    #     sent_num_accumulate += sent_counter.get(idx, 0)
    #     token_num_accumulate += token_counter.get(idx, 0)
    #     sent_ratio_accumulate = sent_num_accumulate/article_num
    #     token_ratio_accumulate = token_num_accumulate/sent_num
    #     if sent_ratio_idx<len(ratio_thresholds) and sent_ratio_accumulate > ratio_thresholds[sent_ratio_idx]:
    #         sent_thresholds.append(idx)
    #         sent_ratio_idx += 1
    #     if token_ratio_idx<len(ratio_thresholds) and token_ratio_accumulate > ratio_thresholds[token_ratio_idx]:
    #         token_thresholds.append(idx)
    #         token_ratio_idx += 1
    #     i += 1
    #     # if sent_ratio_idx>1:
    # print(ratio_thresholds)
    # print(sent_thresholds)
    # print(token_thresholds)

    # data = json.load(open(os.path.join(project_dir, 'data_analyze/data/sentiment_analysis/tf_idf_stats.json')))
    # word2tf, word2idf = data['tf'], data['idf']
    #
    # #观察tf和idf值，设置阈值进行过滤
    # data = json.load(open(os.path.join(project_dir, 'data_analyze/data/sentiment_analysis/word_tfidf_lists.json')))
    # word2tf_lst, word2idf_lst = data['word_and_tf'], data['word_and_idf']
    # print(0)


    # 观察情感词的attn值，设置阈值进行过滤
    with open(os.path.join(project_dir, 'data_analyze/data/sentiment_analysis/neg_pos_attn.json')) as fin:
        data = json.load(fin)
        pos_lst, neg_lst = data['pos'], data['neg']
    print(1)