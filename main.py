#!/usr/bin/env python3

#-------------------------
#  @nagayosi 2018.3.25
#
# how to use
#  for train
#   python3 main.py --train --iter 1000 --gpu
#  for test
#   python3 main.py --test --gpu
#
#-------------------------

import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, SimpleRNN
from keras.optimizers import Nadam
from keras.utils.data_utils import get_file
import numpy as np
import random, sys, re, argparse


TRAIN_PATH = 'Dataset/bleach_poem.txt'

cf = {
    'Minibatch': 10,
    'LearningRate': 0.01,
    'WeightDecay': 0.0005,
    'FineTuning': False,
    'SaveModel': 'osarePoem.hdf5',
    'maxlen': 3,
    'step': 1
}

## sampling 1charater function from probability distribution
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


## defined network model
def my_model(chars):
    model = Sequential()
    model.add(LSTM(2048, activation='relu', input_shape=(cf['maxlen'], chars), return_sequences=True))
    model.add(LSTM(2048, activation='relu', return_sequences=True))
    model.add(LSTM(2048, activation='relu', return_sequences=False))
    #model.add(SimpleRNN(10))
    #model.add(LSTM(100))
    model.add(Dense(chars))
    model.add(Activation('softmax'))
    
    return model


## character split function
def get_chars():
    path = TRAIN_PATH
    print('get chars from <-- {}'.format(path))
    #text = open(path, encoding='cp932').read()
    text = open(path, encoding='utf-8').read()
    #text = open(path).read()
    print(' --> corpus length:', len(text))
    chars = sorted(list(set(text)))

    print(' --> total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    print('\tget chars successed!')
    return chars, char_indices, indices_char, text


## for training function
def train(args, chars, char_indices, indices_char, text):
    sentences = []
    next_chars = []
    for i in range(0, len(text) - cf['maxlen'], cf['step']):
        sentences.append(text[i: i + cf['maxlen']])
        next_chars.append(text[i + cf['maxlen']])
    print(' --> nb sequences:', len(sentences))

    print(' --> Vectorization...')
    X = np.zeros((len(sentences), cf['maxlen'], len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    # build the model: a single LSTM
    print(' --> Build model...')
    model = my_model(len(chars))

    optimizer = Nadam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    model.fit(X, y, batch_size=cf['Minibatch'], epochs=args.iter)

    model.save_weights(cf['SaveModel'])
    print('\ntrain finished and saved')


## for test function
def test(args, chars, char_indices, indices_char):
    print('\ntest start\n')
    sentence = '@@@'
    generated = sentence

    diversity = .3
    text_length = 1000

    model = my_model(len(chars))
    model.load_weights(cf['SaveModel'])

    for i in range(text_length):  
        x = np.zeros((1, cf['maxlen'], len(chars)))

        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1

        preds = model.predict(x)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        if next_char == '@':
            break

        generated += next_char
        sentence = sentence[1:] + next_char

        #print(next_char)

    print('\n--------- generated text --------\n')
    generated = generated.replace('@', '')
    print(generated)



def parse_args():
    parser = argparse.ArgumentParser(description='OsarePoem-Generator demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',help='Use CPU (overrides --gpu)',action='store_true')
    parser.add_argument('--train', dest='train', help='train', action='store_true')
    parser.add_argument('--test', dest='test', help='test', action='store_true')
    parser.add_argument('--iter', dest='iter', help='iteration', default=100, type=int)
    args = parser.parse_args()
    return args

    
if __name__ == '__main__':

    args = parse_args()
    
    chars, char_indices, indices_char, text = get_chars()

    if args.train:
        train(args, chars, char_indices, indices_char, text)
    if args.test:
        test(args, chars, char_indices, indices_char)

