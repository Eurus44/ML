import pandas as pd
import numpy as np
import math
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt


def add_noise_data_2input_1output(input_data, input_labels, n_points, mean, scale):
    """
    Create a noise verstion of the input data

    Params:
        input_data: base input data
        input_labels: base input labels
        n_points: the number of needed points
        mean, scale: the gaussian data
    """
    raw_X = []
    raw_labels = []

    noise = np.random.normal(loc=mean, scale=scale, size=(n_points, 2))
    for i in range(n_points):
        k = np.random.randint(len(input_data))

        x1 = input_data[k][0] + noise[i][0]
        x2 = input_data[k][1] + noise[i][1]

        # We add more difficult for decision tree

        raw_X.append([x1, x2])
        raw_labels.append(input_labels[k])
    return np.array(raw_X), np.array(raw_labels)


class SimpleMLP:

    def __init__(self, n_inputs=2, n_hidden=5, loss='MSE', learning_rate=1e-1, n_epochs=10):
        # init parameters
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.Loss = 'MSE'

        # init weight
        self.hidden_w = np.random.normal(scale=1.0 / math.sqrt(n_inputs), size=(n_inputs + 1, n_hidden))
        self.output_w = np.random.normal(scale=1.0 / math.sqrt(n_inputs), size=n_hidden + 1)

        # init hidden_neurals
        self.hidden_neurals = np.zeros(shape=n_hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def loss(self, y_predict, y):
        if self.Loss == 'MSE':
            # Use mean square error loss function
            loss = (y_predict - y) ** 2
        elif self.Loss == 'CE':
            loss = np.log2(2) * (-y * np.log2(y_predict) - (1 - y) * np.log2(1 - y_predict))
        return loss

    def forward(self, _X):

        # update hidden_neurals
        x_input = np.append(_X, [1])
        self.hidden_neurals = self.sigmoid(np.matmul(x_input, self.hidden_w))

        # update output
        hidden_neurals_input = np.append(self.hidden_neurals, [1])
        O = self.sigmoid(np.matmul(self.output_w, hidden_neurals_input))
        self.o = O
        return O

    def backward(self, _X, _y):

        # append 1 to hidden_neurals
        hidden_neurals_input = np.append(self.hidden_neurals, [1])
        if self.Loss == 'MSE':
            # update output_w
            sigmaG = self.learning_rate * (_y - self.o) * self.o * (1 - self.o)
        elif self.Loss == 'CE':
            sigmaG = self.learning_rate * (_y - self.o)
        self.output_w += sigmaG * hidden_neurals_input

        # append 1 to x input
        x_input = np.append(_X, [1])

        # update hidden_w
        for i in range(len(self.hidden_neurals)):
            g = self.hidden_neurals[i]
            if self.Loss == 'MSE':
                sigma = self.learning_rate * (g - x_input) * g * (1 - g)
            elif self.Loss == 'CE':
                sigma = self.learning_rate * (g - x_input)

            for k in range(self.n_inputs + 1):
                for j in range(self.n_hidden):
                    self.hidden_w[k][j] += sigma[k] * x_input[k]

        return self

    # function for train model
    def fit(self, X, y):
        # stop training model after n_epoch epoch
        # you can add other stopping conditions, i.e loss on valid set doesn't decrease after some epochs.
        for i in range(self.n_epochs):
            for k in range(len(X)):
                self.update_parameters(X[k], y[k])
        return self

    def update_parameters(self, _X, _y):
        self.forward(_X)
        self.backward(_X, _y)

        return self

    def predict(self, X):
        # TODO: update value return
        y_predict = [0 for _ in range(len(X))]

        for i in range(len(X)):

            O = self.forward(X[i])
            if O >= 0.5:
                y_predict[i] = 0
            else:
                y_predict[i] = 1

        return y_predict


if __name__ == "__main__":
    np.random.seed(1)

    std = 0.2
    n_train = 100
    n_test = 10

    and_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    and_y = np.array([0, 1, 1, 0])

    mlp = SimpleMLP(2, 5, 'MSE')
    mlp.fit(and_X, and_y)

    Xtrain, ytrain = add_noise_data_2input_1output(and_X, and_y, n_train, 0., std)
    #     print(Xtrain.shape, ytrain.shape)

    print('MSE results')
    mlp = SimpleMLP(2, 5, 'MSE')
    mlp.fit(Xtrain, ytrain)
    Xtest, ytest = add_noise_data_2input_1output(and_X, and_y, n_test, 0., std)
    print("Accuracy")

    output_test = mlp.predict(Xtest)
    output_train = mlp.predict(Xtrain)
    print("Accuracy: ", accuracy_score(ytest, output_test))
    print("F1 score: ", f1_score(ytest, output_test))

    # Draw train loss
    plt.plot(mlp.loss(ytrain, output_train))
    plt.plot(mlp.loss(ytest, output_test))

    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    print('Cross Entropy results')
    mlp = SimpleMLP(2, 5, 'CE')
    mlp.fit(Xtrain, ytrain)

    Xtest, ytest = add_noise_data_2input_1output(and_X, and_y, n_test, 0., std)
    print("Acurracy")

    output_test = mlp.predict(Xtest)
    print("Accuracy: ", accuracy_score(ytest, output_test))
    print("F1 score: ", f1_score(ytest, output_test))
    print("Doraemon")






