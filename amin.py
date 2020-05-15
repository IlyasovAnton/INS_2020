import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras import models
from keras.datasets import imdb


def Model_compile(dim):
    model = models.Sequential()

    model.add(layers.Dense(64, activation="relu", input_shape=(dim,)))
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def load_text(text):
    file = open(text, 'r').read()
    word_index = imdb.get_word_index()
    text = []
    for i in file:
        if i in word_index and word_index[i] < 10000:
            text.append(word_index[i])

    text = vectorize([text])
    return text


if __name__ == '__main__':
    dim = 10000
    # for dim in [500, 1000, 2500, 5000, 10000]:
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=dim)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)

    data = vectorize(data, dim)
    targets = np.array(targets).astype(np.float)

    test_x = data[:10000]
    test_y = targets[:10000]
    train_x = data[10000:]
    train_y = targets[10000:]

    model = Model_compile(dim)

    history = model.fit(train_x, train_y, epochs=2, batch_size=500, validation_data=(test_x, test_y))

        # print('acc: ', np.mean(history.history["accuracy"]))
        # print(np.mean(history.history["val_accuracy"]))
        # print(np.mean(history.history["loss"]))
        # print(np.mean(history.history["val_loss"]))
        #
        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['val_accuracy'])
        # plt.title('Accuracy')
        # plt.ylabel('Accuracy')
        # plt.xlabel('Epochs')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.savefig(f'images/accuracy-{dim}.jpg')
        # plt.clf()
        #
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('Loss')
        # plt.ylabel('loss')
        # plt.xlabel('Epochs')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.savefig(f'images/loss-{dim}.jpg')
        # plt.clf()

    review = load_text('review')
    print(model.predict(review))
