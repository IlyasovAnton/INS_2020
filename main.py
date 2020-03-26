from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from tensorflow.keras.datasets import boston_housing

import numpy as np
import matplotlib.pyplot as plt


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


if __name__ == '__main__':
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)

    train_data -= mean
    train_data /= std

    test_data -= mean
    test_data /= std

    k = 6
    num_val_samples = len(train_data) // k

    mean_loss = []
    mean_mae = []
    mean_val_loss = []
    mean_val_mae = []
    for i in range(k):
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

        partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                             train_data[(i + 1) * num_val_samples:]], axis=0)
        partial_train_target = np.concatenate([train_targets[: i * num_val_samples],
                                               train_targets[(i + 1) * num_val_samples:]], axis=0)
        model = build_model()
        history = model.fit(partial_train_data, partial_train_target, epochs=110, batch_size=1,
                            validation_data=(val_data, val_targets), verbose=0)

        mean_val_mae.append(history.history['val_mean_absolute_error'])
        mean_mae.append(history.history['mean_absolute_error'])

        plt.plot(history.history['mean_absolute_error'])
        plt.plot(history.history['val_mean_absolute_error'])
        title = 'Model accuracy' + ', i = ' + str(i+1)
        plt.title(title)
        plt.ylabel('mae')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        mean_val_loss.append(history.history['val_loss'])
        mean_loss.append(history.history['loss'])

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        title = 'Model loss' + ', i = ' + str(i+1)
        plt.title(title)
        plt.ylabel('loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    plt.plot(np.mean(mean_mae, axis=0))
    plt.plot(np.mean(mean_val_mae, axis=0))
    title = 'Mean model mae'
    plt.title(title)
    plt.ylabel('mae')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(np.mean(mean_loss, axis=0))
    plt.plot(np.mean(mean_val_loss, axis=0))
    title = 'Mean model loss'
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

