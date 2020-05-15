import sys
from os.path import exists

import keras.callbacks
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import np_utils


filename = 'wonderland.txt'
raw_text = open(filename).read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))


class MyCallback(keras.callbacks.Callback):
    def __init__(self):
        super(MyCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 1 or epoch % 5 == 0:
            generate_symbols(self.model, epoch)


def generate_symbols(model, epoch):
    g = open('gen_text.txt', 'a')
    gen_symbols = []
    start = np.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    g.write("On epoch " + str(epoch) + " seed is:\n" + "\"" + ''.join([int_to_char[value] for value in pattern]) + "\"\n")
    for i in range(1000):
        X = np.reshape(pattern, (1, len(pattern), 1))
        X = X / float(n_vocab)
        predict = model.predict(X, verbose=0)
        idx = np.argmax(predict)
        result = int_to_char[idx]
        gen_symbols.append(result)
        pattern.append(idx)
        pattern = pattern[1:len(pattern)]
    g.write("generated text is:\n"+"\""+''.join(gen_symbols)+"\"\n")
    g.close()


if __name__ == '__main__':
    n_chars = len(raw_text)
    n_vocab = len(chars)

    seq_length = 100

    dataX = []
    dataY = []

    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)

    X = np.reshape(dataX, (n_patterns, seq_length, 1))
    X = X / float(n_vocab)
    Y = np_utils.to_categorical(dataY)

    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(Y.shape[1], activation='softmax'))
    filename = 'best-model-weights.hdf5'
    if not exists(filename):
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        checkpoint = ModelCheckpoint(filename, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint, MyCallback()]

        model.fit(X, Y, epochs=20, batch_size=128, callbacks=callbacks_list)
    else:
        model.load_weights(filename)

        model.compile(loss='categorical_crossentropy', optimizer='adam')

        int_to_char = dict((i, c) for i, c in enumerate(chars))

        start = np.random.randint(0, len(dataX) - 1)
        pattern = dataX[start]
        print('\"', ''.join([int_to_char[value] for value in pattern]), '\"')

        for i in range(1000):
            X = np.reshape(pattern, (1, len(pattern), 1))
            X = X / float(n_vocab)
            predict = model.predict(X, verbose=0)
            idx = np.argmax(predict)
            result = int_to_char[idx]
            seq_in = [int_to_char[value] for value in pattern]
            sys.stdout.write(result)
            pattern.append(idx)
            pattern = pattern[1:len(pattern)]
        print('\nDone')
