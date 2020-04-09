import cv2
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


optimizers = [optimizers.SGD(),
              optimizers.SGD(learning_rate=0.001),
              optimizers.SGD(momentum=0.9),
              optimizers.RMSprop(),
              optimizers.RMSprop(learning_rate=0.01),
              optimizers.RMSprop(rho=0.5),
              optimizers.RMSprop(momentum=0.9),
              optimizers.Adagrad(),
              optimizers.Adagrad(learning_rate=0.01),
              optimizers.Adam(),
              optimizers.Adam(learning_rate=0.01)]


def build_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(Dense(10, activation='softmax'))
    return model


def load_image(filepath):
    img = np.asarray(Image.open(filepath))
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)

    k = np.array([[[0.2989, 0.587, 0.114]]])
    img = np.sum(img * k, axis=2).reshape((1, 28 * 28)) / 255.0
    return img


def predict_numerals(model):
    for i in range(10):
        img = load_image(f'numerals/{i}.png')
        print(f'numeral: {i}')
        for j, predict in enumerate(model.predict(img)[0]):
            print(f'{j}: {predict: .4f}')


def plot_diagramm(y1, y2, metric):
    x1 = np.arange(1, 12) - 0.2
    x2 = np.arange(1, 12) + 0.2

    fig, ax = plt.subplots()

    ax.bar(x1, y1, width=0.4)
    ax.bar(x2, y2, width=0.4)

    ax.set_facecolor('seashell')
    fig.set_figwidth(12)  # ширина Figure
    fig.set_figheight(6)  # высота Figure
    fig.set_facecolor('floralwhite')
    plt.title(metric)
    plt.ylabel(metric)
    plt.xlabel('number of optimizator')
    plt.legend(['Train', 'Test'], loc='upper left')
    fig.savefig(f'images/{metric}.jpg')


if __name__ == '__main__':
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28*28)).astype('float32') / 255.0
    test_images = test_images.reshape((10000, 28*28)).astype('float32') / 255.0

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    acc = []
    val_acc = []
    loss = []
    val_loss = []

    for i, optimizer in enumerate(optimizers):
        model = build_model()
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(train_images, train_labels, epochs=5, batch_size=128,
                            validation_data=(test_images, test_labels), verbose=0)
        model.save_weights(filepath=f'models/{i+1}.h5')

        acc.append(history.history['acc'][-1])
        val_acc.append(history.history['val_acc'][-1])
        loss.append(history.history['loss'][-1])
        val_loss.append(history.history['val_loss'][-1])

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(f'images/accuracy-{i + 1}.jpg')
        plt.clf()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(f'images/loss-{i+1}.jpg')
        plt.clf()

    for a, va, l, vl in zip(acc, val_acc, loss, val_loss):
        print(f'acc = {a:.5f}, val_acc = {va:.5f}, loss = {l:.5f}, val_loss = {vl:.5f}')

    plot_diagramm(acc, val_acc, 'accuracy')
    plot_diagramm(loss, val_loss, 'loss')

