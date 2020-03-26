from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import numpy as np


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def expression(a, b, c):
    return (a and not b) or (b ^ c)


def element_wise_operations(data, weights):
    activation_func = [relu for _ in range(len(weights) - 1)]
    activation_func.append(sigmoid)

    data = data.copy()
    for i in range(len(weights)):
        res = np.zeros((data.shape[0], weights[i][0].shape[1]))
        for j in range(data.shape[0]):
            for k in range(weights[i][0].shape[1]):
                sum = 0
                for l in range(data.shape[1]):
                    sum += data[j][l] * weights[i][0][l][k]
                res[j][k] = activation_func[i](sum + weights[i][1][k])
        data = res

    return data


def numpy_operations(data, weights):
    activation_func = [relu for i in range(len(weights) - 1)]
    activation_func.append(sigmoid)

    data = data.copy()
    for i in range(0, len(weights)):
        data = activation_func[i](np.dot(data, weights[i][0]) + weights[i][1])

    return data


def print_predicts(model, data):
    weights = []
    for layer in model.layers:
        weights.append(layer.get_weights())

    element_wise_predict = element_wise_operations(data, weights)
    numpy_predict = numpy_operations(data, weights)
    model_predict = model.predict(data)

    print('поэлементные операции над тензорами:\n', element_wise_predict)
    print('операции над тензорами из numpy:\n', numpy_predict)
    print('предсказание модели:\n', model_predict)


def build_model():
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(3,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    X = np.array([[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 1],
                  [1, 0, 0],
                  [1, 0, 1],
                  [1, 1, 0],
                  [1, 1, 1]])
    Y = np.array([expression(*x) for x in X])
    print('значения логической функции:\n', Y)

    model = build_model()

    print('результаты до обучения')
    print_predicts(model, X)

    model.fit(X, Y, epochs=150, batch_size=1, verbose=0)

    print('результаты после обучения')
    print_predicts(model, X)
