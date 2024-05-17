import os

import pymorphy3

import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import razdel
from razdel import sentenize, tokenize
import re


from nltk.tokenize import word_tokenize


import numpy as np



import torch
from torch import nn


import re






snb_stemmer_ru = SnowballStemmer('russian')
ru_stop_words = stopwords.words('russian')
morph = pymorphy3.MorphAnalyzer()



class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_embeddings, embedding_dim):
        super(NeuralNet, self).__init__()

        self.rnn1 = torch.nn.RNN(input_size, 384)
        self.relu1 = nn.ReLU()

        self.rnn2 = torch.nn.RNN(384, 384)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        self.rnn3 = torch.nn.RNN(384, 384)
        self.relu3 = nn.ReLU()

        self.dropout2 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(384, 64)
        self.relu4 = nn.ReLU()

        self.dropout3 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(64, num_classes)



    def forward(self, x):
        x = x.view(x.size()[0], -1)

        out, hn0 = self.rnn1(x)
        out = self.relu1(out)

        out, hn1 = self.rnn2(out, hn0)
        out = self.relu2(out)

        out = self.dropout1(out)

        out, hn2 = self.rnn3(out, hn1)
        out = self.relu3(out)

        out = self.dropout2(out)

        out = self.fc1(out)
        out = self.relu4(out)

        out = self.dropout3(out)

        out = self.fc2(out)

        return out





class Model:
    def __init__(self, x):
        self.X = x
        self.model = os.path.join(os.getcwd(), "ml", "static", "ml", "model_news")
        self.bog = os.path.join(os.getcwd(), "ml", "static", "ml", "bow.npy")

    def get_bow(self, encoding: str = "utf-8") -> list[str]:
        bow = None
        with open(self.bog, 'rb') as file:
            bow = np.load(file, allow_pickle=True)
        return bow

    def vectorize(self,):
        text_in = self.X
        bow = self.get_bow()
        text_in = text_in.lower()
        ans_str = ''
        tok = list(tokenize(text_in))
        ru_letters = re.compile('^[а-яА-ЯёЁ]*$')
        pt = [morph.parse(t.text) for t in tok if ru_letters.search(t.text)]
        lemmed_text = [morph_word[0].normalized.word for morph_word in pt]

        for i in lemmed_text:
            if i in ru_stop_words:
                lemmed_text.remove(i)
        for i in lemmed_text:
            ans_str += (i + " ")

        lst_input_words = np.array( ans_str.split())
        dict_cnt = dict()
        lst_index = np.zeros(9093,)
        for bow_word in bow:
            cnt = 0
            if bow_word in lst_input_words:
                for index in range(len(lst_input_words)):
                    if bow_word == lst_input_words[index]:
                        cnt+=1
                dict_cnt.update( { list(bow).index(bow_word):cnt} )
        for index, value in dict_cnt.items():
            lst_index[index] = value
        input_tensor = torch.tensor(lst_index).float().reshape(1, 9093)


        self.X = input_tensor


    def predict(self,):
        lst_news_names = {2:'Политика', 1:'Экономика', 4:'Общество', 5:'Спорт', 0: 'Культура', 3:'Наука'}

        loaded_model = torch.load(self.model)

        model = NeuralNet(input_size=9093, hidden_size=256, num_classes=6, num_embeddings=16, embedding_dim=6)
        model.load_state_dict(loaded_model)


        with torch.no_grad():
            model.eval()
            outputs = model(self.X)
            index = int(outputs.argmax())

        return lst_news_names.get(index, 'Неизвестно')
