# encoding: utf-8
"""
@author: Yuxian Meng 
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: preprocess
@time: 2018/11/2 20:12

    一些预处理文字的函数
"""

import requests
import json
import os
#from common import project_dir
import jieba
import jieba.posseg as pseg
from typing import List
from lxml import etree

stop_words = json.load(
    open(os.path.join('/mnt/cbrai/nlp/sentiment/models/stopwords_table', 'non_sentimental_stop_words.json')))


def extract(html):
    """
    从html tag中得到正文
    """
    selector = etree.HTML(html)
    string = selector.xpath('string(.)')
    return string


def not_chinese(string):
    """判断string是否为全中文字"""
    for c in string:
        if not (c >= u'\u4e00' and c <= u'\u9fff'):
            return True
    return False


# todo：保留ner识别出的时间或重要数字与符号，如+-等
def not_ch_or_stop(string):
    """不是中文或者在停用词里"""
    return not_chinese(string) or string in stop_words


def is_stop(string):
    """不在停用词表中"""
    return string in stop_words


def clean_sent(words, return_offsets=False, filter_func=not_chinese):
    """
    清洗句子
    Args:
        words(list[str]):
        return_offsets(bool):是否同时返回由于删除某些词汇造成单词index的offset
        filter_func: 判断是否filter掉某个词的函数

    Returns:
        valid_words(list[str])
        offsets(list[int])
    """
    if return_offsets:
        offset = 0
        offsets = []
        valid_words = []
        for word in words:
            if filter_func(word):  # word in stop_words or
                offset += 1
            else:
                valid_words.append(word)
                offsets.append(offset)
        return valid_words, offsets
    return [x for x in words if not filter_func(x)]


def segment(text, mode='pos'):
    """调用jieba将text分词"""
    if mode == 'seg':
        return [x for x in jieba.cut(text, cut_all=False)]
    else:
        return [f'{word}/{flag}' for word, flag in pseg.cut(text, HMM=False)]


def preprocess(text: List[str], clear=True, mode='pos') -> str:
    """
    对text做预处理
    Args:
        text: pos tagging后的输入，如["我/w", "是/w", "中国人/w"]
        clear: 是否去除停用词和非中文字符
        mode:
        "pos", 返回带pos的分词， 如"我/w 是/w"
        "seg", 返回不带pos的分词 如 "我 是"
        "plain": 返回简单的str 如"我是"

    Returns:

    """
    words_and_poses = [x.split('/') for x in text]
    if clear:
        words_and_poses = [x for x in words_and_poses if not not_ch_or_stop(x[0])]
    if mode == 'pos':
        return ' '.join(['/'.join(t) for t in words_and_poses])
    elif mode == 'seg':
        return " ".join([x[0] for x in words_and_poses])
    return "".join(x[0] for x in words_and_poses)


def sep_text(txt, return_index=False, ignore_space=False):
    """
    将一段中文txt切分为句子
    :param txt: str或list of str或list of tuple
    :return: list of str
    """
    sentences = []
    stack = []
    after2before = {'”': '“', '）': '（', }
    end_signals = ['。', '！', '？']
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
            if pre in stack:  # 因为有的文章有符号错误
                stack.remove(pre)
        if ((ch in end_signals and i < l - 1 and txt[i + 1] not in after2before.keys()) or
            (ch in after2before.keys() and txt[i - 1] in end_signals)) and not stack:
            sentences.append(sentence)
            sentence = []
    if sentence:
        sentences.append(sentence)
    return sentences


if __name__ == '__main__':
    # print(clean_sent(['彼此','股票', '暴跌', '，', '原因', '是', '消费者', '对', '该', '公司', '商品', '并', '不', '满意', '股票', '暴跌', '，', '原因', '是', '消费者', '对', '该', '公司', '商品', '并', '不', '满意', '股票', '暴跌', '，', '原因', '是', '消费者', '对', '该', '公司', '商品', '并', '不', '满意', '股票', '暴跌', '，', '原因', '是', '消费者', '对', '该', '公司', '商品', '并', '不', '满意', '股票', '暴跌', '，', '原因', '是', '消费者', '对', '该', '公司', '商品', '并', '不', '满意', '股票', '暴跌', '，', '原因', '是', '消费者', '对', '该', '公司', '商品', '并', '不', '满意', '股票', '暴跌', '，', '原因', '是', '消费者', '对', '该', '公司', '商品', '并', '不', '满意', '股票', '暴跌', '，', '原因', '是', '消费者', '对', '该', '公司', '商品', '并', '不', '满意', '股票', '暴跌', '，', '原因', '是', '消费者', '对', '该', '公司', '商品', '并', '不', '满意', '股票', '暴跌', '，', '原因', '是', '消费者', '对', '该', '公司', '商品', '并', '不', '满意'],
    #                  return_offsets=True,
    #                  filter_func=not_ch_or_stop))
    print([not_chinese(s) for s in ['我', '1', 'wod', ' ', '/t']])
