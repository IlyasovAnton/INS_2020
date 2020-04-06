from keras.layers import Input, Dense
from keras.models import Model, Sequential

import numpy as np
import pandas as pd


def save_to_csv(path, data):
    pd.DataFrame(data).to_csv(path, index=False, header=False)


def generate_dataset(n):
    data = np.zeros((n, 6))
    targets = np.zeros(n)
    for i in range(n):
        x = np.random.normal(-5, 10)
        e = np.random.normal(0, 0.3)
        data[i, :] = (np.log(np.abs(x)) + e, np.sin(3*x) + e, np.exp(x) + e, x + 4 + e, -x + np.sqrt(np.abs(x)) + e, x + e)
        targets[i] = -np.power(x, 3) + 3
    return data, targets


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    return model


def create_models():
    main_input = Input(shape=(6,), name='main_input')
    encoded = Dense(64, activation='relu')(main_input)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(6, activation='linear')(encoded)

    input_encoded = Input(shape=(6,), name='input_encoded')
    decoded = Dense(32, activation='relu', kernel_initializer='normal')(input_encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(6, name="out_aux")(decoded)

    predicted = Dense(64, activation='relu', kernel_initializer='normal')(encoded)
    predicted = Dense(64, activation='relu')(predicted)
    predicted = Dense(64, activation='relu')(predicted)
    predicted = Dense(1, name="out_main")(predicted)

    encoded = Model(main_input, encoded, name="encoder")
    decoded = Model(input_encoded, decoded, name="decoder")
    predicted = Model(main_input, predicted, name="regr")

    return encoded, decoded, predicted, main_input


def generate_data():
    X, Y = generate_dataset(300)
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)
    X /= X.std(axis=0)
    Y /= Y.std(axis=0)

    Xtrain, Ytrain = X[: 50], Y[: 50]
    Xtest, Ytest = X[50:], Y[50:]

    return Xtrain, Ytrain, Xtest, Ytest


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = generate_data()

    encoded, decoded, full_model, main_input = create_models()

    model = build_model()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(x_train, y_train, epochs=100, batch_size=5, validation_data=(x_test, y_test))

    full_model.compile(optimizer="adam", loss="mse", metrics=['mae'])
    full_model.fit(x_train, y_train, epochs=100, batch_size=5, validation_data=(x_test, y_test))

    encoded_data = encoded.predict(x_test)
    decoded_data = decoded.predict(encoded_data)
    regression = full_model.predict(x_test)

    save_to_csv('x_train.csv', x_train)
    save_to_csv('y_train.csv', y_train)
    save_to_csv('x_test.csv', x_test)
    save_to_csv('y_test.csv', y_test)
    save_to_csv('encoded.csv', encoded_data)
    save_to_csv('decoded.csv', decoded_data)
    save_to_csv('regression.csv', regression)

    decoded.save('decoder.h5')
    encoded.save('encoder.h5')
    full_model.save('regression.h5')
