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
