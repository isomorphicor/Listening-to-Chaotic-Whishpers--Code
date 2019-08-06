# -*- coding: utf-8 -*-
import json
import pandas as pd
import numpy as np
import codecs
from utils.preprocess import segment
from utils.preprocess import extract
from utils.preprocess import clean_sent

data_dir = '/mnt/data/nlp_data/news/'

newsid2link = pd.read_csv(data_dir+"newsid2link.csv")

#%%
for i in range(newsid2link.shape[0]):
    try:
        with open(data_dir+'news/T_NEWS_TEXT_BD'+newsid2link.NEWSLINK.loc[i].replace('\\', '/'), encoding='gbk') as f:
            tmp = f.read()
        content = clean_sent(segment(extract(tmp), mode='seg'))
    except:
        content = []

    title = clean_sent(segment(newsid2link.NEWSTITLE.iloc[i], mode='seg'))
    title.extend(content)
    # with codecs.open(data_dir+'news_stocks/'+newsid2link.fid.iloc[0], 'a', encoding='utf-8') as f:
    #     f.write(str(title))
    json.dump(title, open(data_dir+'news_stocks/'+newsid2link.fid.iloc[i]+'.json', 'w'), ensure_ascii=False)
    if i%1000 == 0:
        print(i, " of ", newsid2link.shape[0], flush=True)
