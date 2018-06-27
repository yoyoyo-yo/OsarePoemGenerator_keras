from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, SimpleRNN

import config as cf

def lstm_model():
    model = Sequential()
    model.add(LSTM(128, activation='relu',
                   input_shape=(cf.N_gram, cf.Vocabrary_num), return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=False))
    #model.add(SimpleRNN(10))
    #model.add(LSTM(100))
    model.add(Dense(cf.Vocabrary_num))
    model.add(Activation('softmax'))
    
    return model
