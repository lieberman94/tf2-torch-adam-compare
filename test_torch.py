#!/usr/bin/env python
# coding: utf-8

import math
from tqdm import tqdm
from utils import RunningAverage
import torch.optim as optim
from transformers import BertModel, BertConfig
from transformers.optimization import AdamW
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
import json
import os
import time

GPU = '5'
pretrained_path = '../basedata/chinese_roberta_wwm_ext_pytorch'
train_data_path = 'data/train/0707-all/sample.json'

config_path = os.path.join(pretrained_path, 'bert_config.json')
SEQ_LEN = 256
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

def initialize_weights(net):
    def weights_init(m):
        # torch.nn.init.xavier_uniform_(m.weight)
        limit=pow(6/(768+3),0.5)
        torch.nn.init.uniform_(m.weight,-limit,limit)

    for m in net.children():
        if m != net.bert:
            if type(m) == nn.Linear:
                m.apply(weights_init)
                


class DataGenerator():

    @staticmethod
    def load_from_json(filename):
        wordids, labels = [], []
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i % 100000 == 0 and i != 0:
                    print('载入数据:%d ' % i)
                item = json.loads(line.strip())
                labels.append(item['label'])
                wordids.append(item['wordids'])
        return wordids, labels

    def __init__(self):
        self.data, self.labels = self.load_from_json(train_data_path)
        self.batchsize = 32
        self.size = math.ceil(len(self.data) / self.batchsize)

    def get_data(self):
        for i in range(self.size):
            ifrom = i * self.batchsize
            ito = (i + 1) * self.batchsize
            yield [torch.from_numpy(np.array(self.data[ifrom:ito])),
                   torch.from_numpy(np.array(self.labels[ifrom:ito]))]


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        config = BertConfig.from_pretrained(
            os.path.join(pretrained_path, 'bert_config.json'))
        self.bert = BertModel(config)
        self.fc1 = nn.Linear(768, 3)

        self.loss = nn.CrossEntropyLoss()

        if GPU != '':
            self.cuda()

    def forward(self, words):
        dense = torch.max(self.bert(words)[0][:, 1:-1, :], 1)[0]
        dense = self.fc1(dense)
        return dense


model = Model()
initialize_weights(model)

for i in model.bert.parameters():
    i.requires_grad=True
    
model.train()

optimizer = optim.Adam(model.parameters(), lr=2e-5, eps=1e-7)
# optimizer = optim.SGD(model.parameters(), lr=2e-5,momentum=0.9)
data_gen = DataGenerator()

for epoch in range(1, 100):
    loss_avg = RunningAverage()

    pbar = tqdm(total=data_gen.size)
    for [words, labels] in data_gen.get_data():
        if GPU != '':
            words = words.cuda()
            labels = labels.cuda()

        outputs = model(words)
        loss = model.loss(outputs, labels)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        loss_avg.update(loss.cpu().item())

        pbar.update(1)
        pbar.set_postfix(
            {'epoch': epoch, 'loss': '{:08.6f}'.format(loss_avg())})
    pbar.close()




