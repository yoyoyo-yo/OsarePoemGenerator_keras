#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import tensorflow as tf
#-- GPU 
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

import numpy as np
import random
import sys
import re
import argparse

import config as cf
from model import lstm_model as network



## sampling 1charater function from probability distribution
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds[preds < 1e-10] = 1e-10
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)



def get_vocabrary():
    with open(cf.Vocabrary_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        voca = [x.replace(os.linesep, '') for x in lines]

    voca.extend(cf.Prefix)
    return voca


def get_prefix_suffix(len=1):
    text = ''
    for _ in range(len):
        text += cf.Prefix[0]

    return text
    

def parse_data():
    gram_data = []
    next_data = []
    prefix = get_prefix_suffix(len=cf.N_gram)
    suffix = get_prefix_suffix(len=1)
    with open(cf.Train_data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            line = prefix + line + suffix
            l_len = len(line)
            _gram_data = []
            _next_data = []
            for i in range(l_len - cf.N_gram):
                _gram_data.append(line[i: i + cf.N_gram])
                _next_data.append(line[i + cf.N_gram])
            gram_data.extend(_gram_data)
            next_data.extend(_next_data)

    return gram_data, next_data



def voca2onehot(voca, data):
    data_num = len(data)
    voca_num = len(voca)

    pred_num = len(data[0])
    
    if pred_num > 1:
        onehot = np.zeros((data_num, cf.N_gram, voca_num), dtype=np.float32)
    else:
        onehot = np.zeros((data_num, voca_num), dtype=np.float32)
    
    for d_i, _data in enumerate(data):
        for c_i, char in enumerate(_data):
            char_ind = voca.index(char)
            if pred_num > 1:
                onehot[d_i, c_i, char_ind] = 1.
            else:
                onehot[d_i, char_ind] = 1.
                
    return onehot



def check_dir(path):
    os.makedirs(path, exist_ok=True)
    

## for training function
def train():

    print('Training Start!')

    vocabrary = get_vocabrary()
    x_data, y_data = parse_data()
    X = voca2onehot(vocabrary, x_data)
    Y = voca2onehot(vocabrary, y_data)

    model = network()

    optimizer = keras.optimizers.Nadam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.fit(X, Y, batch_size=cf.Minibatch, epochs=cf.Epoch)

    model.save_weights(cf.Save_path)
    print('Trained model saved -> {}'.format(cf.Save_path))
    print('Train finished!!')


## for test function
def test():
    print('Test start!')
    sentence = get_prefix_suffix(len=cf.N_gram)
    generated = sentence

    voca = get_vocabrary()
    
    model = network()
    model.load_weights(cf.Save_path)

    for i in range(cf.Max_text_length):  
    
        x = voca2onehot(voca, sentence)
        x = x[None, ...]

        preds = model.predict(x)[0]
        next_index = sample(preds, cf.diversity)
        next_char = voca[next_index]

        if next_char == '@':
            break

        generated += next_char
        sentence = sentence[1:] + next_char


    print('\n--------- generated text --------\n')
    generated = generated.replace('@', '')
    print(generated)



def parse_args():
    parser = argparse.ArgumentParser(description='OsarePoem-Generator demo')
    parser.add_argument('--train', dest='train', help='train', action='store_true')
    parser.add_argument('--test', dest='test', help='test', action='store_true')
    
    args = parser.parse_args()
    return args

    
if __name__ == '__main__':

    args = parse_args()
    
    check_dir(cf.Save_directory)
    
    if args.train:
        train()
    if args.test:
        test()

