import numpy as np
import matplotlib.pyplot as plt
np.random.seed(13)


def gen_data(n_data=10):
    a = 4
    b = 7

    x = np.linspace(0.0, 1.0, n_data, endpoint=False)
    y = a * x + b
    y_sensor = a * x + b + np.random.randn(n_data)

    fig = plt.figure()

    ax = fig.add_subplot()
    ax.scatter(x, y, color='r')
    ax.scatter(x, y_sensor, color='b')

    # plt.show()

    return x, y, y_sensor


def back_propagation(a, b, x, y_true, y_pred):
    learning_rate = 0.01

    # L = (y_pred - y_true)^2
    dL_dypred = 2 * (y_pred - y_true)
    dypred_db = 1
    dypred_da = x

    dL_db = dL_dypred * dypred_db
    dL_da = dL_dypred * dypred_da

    b = b - learning_rate * dL_db
    a = a - learning_rate * dL_da

    return a, b


def loss_function(a, b, x, y_true):
    y_pred = a * x + b

    error = y_true - y_pred
    loss = np.sum(np.power(error, 2))

    return loss


if __name__ == '__main__':
    n_data = 10
    x, y, y_sensor = gen_data(n_data)

    a_pred = 0.0
    b_pred = 0.0

    epochs = 100
    a = []
    b = []
    loss = []
    for _ in range(epochs):
        for i in range(n_data):
            y_pred = a_pred * x[i] + b_pred
            a_pred, b_pred = back_propagation(a_pred, b_pred, x[i], y_sensor[i], y_pred)

        a.append(a_pred)
        b.append(b_pred)

        loss.append(loss_function(a_pred, b_pred, x, y_sensor))

    print(a_pred, b_pred)

    fig = plt.figure()

    ax = fig.add_subplot()
    ax.plot(range(epochs), a, label='a')
    ax.plot(range(epochs), b, label='b')
    ax.legend()

    ax2 = ax.twinx()
    ax2.plot(range(epochs), loss, color='red')
    ax2.set_ylim(0, 10)
    # print(loss)

    plt.show()