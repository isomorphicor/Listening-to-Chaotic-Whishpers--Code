import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import json
from utils.preprocess import clean_sent
from utils.preprocess import is_stop


class LabeledLineSentence(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        # 如有多个不同路径的文档需要训练，则并列使用 for + yield，会依次遍历传入
        for fname in os.listdir(self.dirname):
            # print(fname)
            # 使用去除后缀的文档名作为文档标签
            tag_name, _ = os.path.splitext(fname)
            # 本地文档已分词，读取后转为一个list文件即可
            doc_sentence = clean_sent(json.load(open(self.dirname + fname, 'r')), filter_func=is_stop)
            # 注意这里的第二个参数应为list，传入str会被分割
            yield TaggedDocument(doc_sentence, [tag_name])


if __name__ == "__main__":
    data_dir = '/mnt/data/nlp_data/news/'
    sentences = LabeledLineSentence(data_dir + 'news_stocks/')

    # 设置min_count忽略出现1次以下的词，防止有词不在词汇表中
    model = Doc2Vec(vector_size=300, window=10, min_count=5, epochs=40, workers=8)
    # model = Doc2Vec(vector_size=200, window=10, min_count=1, epochs=40, workers=8)
    print('start train ...', flush=True)
    # 构建文本词汇表
    model.build_vocab(sentences)
    print('build vocab finished', flush=True)
    # 训练doc2vec
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    print('docd2vec train finished', flush=True)
    # doc2vec模型保存
    model.save(os.path.join(data_dir + 'd2v_300/', 'd2v.model'))
    print('doc2vec saved', flush=True)

    # model.infer_vector(['中国', '人民'])
    # model.docvecs['0000_40']
