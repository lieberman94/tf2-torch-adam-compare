#!/usr/bin/env python
# coding: utf-8

import os
import json
import numpy as np
os.environ['TF_KERAS'] = '1'
import tensorflow as tf
from keras_bert import load_trained_model_from_checkpoint
GPU = '4'

pretrained_path = '../basedata/chinese_roberta_wwm_ext_L-12_H-768_A-12'
train_data_path = 'data/sample.json'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
SEQ_LEN = 256


def load_from_json(filename):
    print('loading data from json :', filename)
    wordids, segmentids, labels = [], [], []
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0 and i != 0:
                print('载入数据:%d ' % i)
            item = json.loads(line.strip())
            labels.append(item['label'])
            wordids.append(item['wordids'])

    wordids = np.array(wordids)
    segmentids = np.zeros_like(wordids)
    labels = tf.keras.utils.to_categorical(labels)

    return [wordids, segmentids], labels


def create_model(bert_train=False):
    bert = load_trained_model_from_checkpoint(
        config_path, checkpoint_path,
        training=False,
        trainable=bert_train,
        seq_len=SEQ_LEN, )
    inputs = bert.inputs[:2]
    dense = bert.get_layer('Encoder-12-FeedForward-Norm').output
    dense = tf.keras.layers.Lambda(lambda x: x[:, 1:-1, :])(dense)
    dense = tf.keras.layers.GlobalMaxPool1D()(dense)
    output2 = tf.keras.layers.Dense(
        units=3, activation='softmax', name='3cls')(dense)

    return tf.keras.models.Model(inputs, output2)


model = create_model(bert_train=True)
for layer in model.layers:
    layer.trainable = True

train_X, train_Y = load_from_json(train_data_path)

optimizer = tf.keras.optimizers.Adam(2e-5)
# optimizer = tf.keras.optimizers.SGD(learning_rate=2e-5, momentum=0.9)
loss = tf.keras.losses.CategoricalCrossentropy()
model.compile(optimizer, loss=loss)
model.fit(train_X, train_Y,
          epochs=100, batch_size=32, shuffle=False)



