import numpy as np
import math

D = 0
N = 0


class layer:
    perceptrons = []


class perceptron:
    global D
    num_inputs = D + 1
    bias = 1
    w = np.random.uniform(low=-0.05, high=0.05, size=(num_inputs,))

    def a(self, x):
        return np.dot(self.w, x)

    def z(self, x):
        return self.h(self.a(x))

    def h(self, a):
        return a


def main():
    training_file = "pendigits_training.txt"
    test_file = "pendigits_test.txt"
    neural_network(training_file, test_file, 0, 0, 0)

# layers - number of layers to use
# units per layer - units per HIDDEN layer exlcuding bias input
# rounds - number of training rounds (using whole training set once)


def neural_network(training_file, test_file, layers, units_per_layer, rounds):
    training_data = normalize(read_file(training_file))
    test_data = normalize(read_file(test_file))

    global D, N
    D = np.shape(training_data)[1] - 1
    N = np.shape(training_data)[0]


def normalize(arr):
    a = np.asarray(arr)
    _max = np.amax(a)

    b = np.multiply(1 / _max, a)
    return b


def read_file(fname):
    lines = []
    with open(fname) as f:
        for line in f:
            lines.append(list(map(float, line.split())))

    return lines


main()
