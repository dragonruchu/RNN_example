import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import torch.autograd as autograd
import torchtext.vocab as torchvocab

import tqdm
import os
import time
import re
import pandas as pd
import string
import gensim
import time
import random
import snowballstemmer
import collections
from collections import Counter
from nltk.corpus import stopwords
from itertools import chain
from sklearn.metrics import accuracy_score

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


def readTMDB(path, seg='train'):
    pos_or_neg = ['pos', 'neg']
    data = []
    for label in pos_or_neg:
        files = os.listdir(os.path.join(path, seg, label))
        for file in files:
            with open(os.path.join(path, seg, label, file), 'r', encoding='utf-8') as f:
                review = f.read().replace('\n', '')
                if label == 'pos':
                    data.append([review, 1])
                elif label == 'neg':
                    data.append([review, 0])

    return data


train_data = readTMDB('./data/aclImdb')
test_data = readTMDB('./data/aclImdb', 'test')


def tokenizer(text):
    return [tok.lower() for tok in text.split(' ')]


train_tokenized = []
test_tokenized = []
for review, target in train_data:
    train_tokenized.append(tokenizer(review))
for review, target in test_data:
    test_tokenized.append(tokenizer(review))

# exit(0)
vocab = set(chain(*train_tokenized))
vocab_size = len(vocab)


# define dict
word_to_index = {word: i + 1 for i, word in enumerate(vocab)}
word_to_index['<unk>'] = 0
index_to_word = {i + 1: word for i, word in enumerate(vocab)}
index_to_word[0] = "<unk>"


def encode_sample(tokenized_samples, vocab):
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in word_to_index:
                feature.append(word_to_index[token])
            else:
                feature.append(0)
        features.append(feature)
    return features

# 将句子长度固定在500词， 若超过则删除后面一部分， 若不足则补充0


def pad_samples(features, maxlen=500, pad=0):
    padded_features = []
    for feature in features:
        if len(feature) >= maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            while(len(padded_feature) < maxlen):
                padded_feature.append(pad)
        padded_features.append(padded_feature)
    return padded_features


# 整理data
train_features = torch.tensor(pad_samples(encode_sample(train_tokenized, vocab)), dtype=torch.long)
train_labels = torch.tensor([score for _, score in train_data], dtype=torch.long)
test_features = torch.tensor(pad_samples(encode_sample(test_tokenized, vocab)), dtype=torch.long)
test_labels = torch.tensor([score for _, score in test_data], dtype=torch.long)

# define network


class SentimentNet(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 bidirectional, weight, labels, use_gpu, **kwargs):
        super(SentimentNet, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional

        # 加载预训练词向量
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        # embed 词向量
        # self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.num_hiddens,
                               num_layers=self.num_layers, bidirectional=self.bidirectional,
                               dropout=0)
        if self.bidirectional:
            self.decoder = nn.Linear(num_hiddens * 4, labels)
        else:
            self.decoder = nn.Linear(num_hiddens * 2, labels)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.long())
        status, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        encoding = torch.cat([status[0], status[-1]], dim=1)
        outputs = self.decoder(encoding)
        return outputs


num_epochs = 20
embed_size = 100
num_hiddens = 100
num_layers = 2
bidirectional = True
batch_size = 64
labels = 2
lr = 0.8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_gpu = True

# 已经有的词向量
glove_file = datapath('glove/glove.6B.100d.txt')
# 指定转化为word2vec格式后文件的位置
tmp_file = get_tmpfile("word2vec.6B.100d.txt")
glove2word2vec(glove_file, tmp_file)

wvmodel = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)
weight = torch.zeros(vocab_size + 1, embed_size)

for i in range(len(wvmodel.index2word)):
    try:
        index = word_to_index[wvmodel.index2word[i]]
    except:
        continue
    weight[index, :] = torch.from_numpy(wvmodel.get_vector(index_to_word[word_to_index[wvmodel.index2word[i]]]))


net = SentimentNet(vocab_size=(vocab_size + 1), embed_size=embed_size, num_hiddens=num_hiddens, num_layers=num_layers,
                   bidirectional=bidirectional, weight=weight, labels=labels, use_gpu=use_gpu)
net = net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)

# print(net)
train_set = torch.utils.data.TensorDataset(train_features, train_labels)
test_set = torch.utils.data.TensorDataset(test_features, test_labels)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

for epoch in range(num_epochs):
    start = time.time()
    train_losses, test_losses = 0, 0
    train_acc, test_acc = 0, 0
    n, m = 0, 0
    for feature, label in train_loader:
        n += 1
        net.train()
        net.zero_grad()
        feature = feature.to(device)
        label = label.to(device)
        score = net(feature)
        loss = loss_function(score, label)
        loss.backward()
        optimizer.step()

        train_acc += accuracy_score(torch.argmax(score.cpu().data, dim=1), label.cpu())
        train_losses += loss.item()

    with torch.no_grad():
        for test_feature, test_label in test_loader:
            m += 1
            net.eval()

            test_feature = test_feature.to(device)
            test_label = test_label.to(device)
            test_score = net(test_feature)
            test_loss = loss_function(test_score, test_label)
            test_acc += accuracy_score(torch.argmax(test_score.cpu().data, dim=1), test_label.cpu())

            test_losses += test_loss.item()

    end = time.time()
    runtime = end - start
    print('epoch: %d, train_loss: %.4f, train_acc: %.2f, test_loss: %.4f, test_acc: %.2f, time: %.2f' % (
        epoch, train_losses / n, train_acc / n, test_losses / m, test_acc / m, runtime))
