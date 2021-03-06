import numpy as np
import math

D = 0
N = 0


class layer:
    perceptrons = []



class perceptron:
    num_inputs = 0
    bias = 1
    w = 0

    def __init__(self):
        global D
        self.num_inputs = D + 1
        self.w = np.random.uniform(low=-0.05, high=0.05, size=(self.num_inputs,))

    def a(self, x):
        return np.dot(self.w, x)

    def z(self, x):
        x = np.insert(x, 0, 1)
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
# number of perceptrons of output layer = number of classes
# number of perceptrons of input layer = # of dimensions


def neural_network(training_file, test_file, layers, units_per_layer, rounds):
    training_data = np.asarray(read_file(training_file))
    test_data = np.asarray(read_file(test_file))

    training_labels = training_data[:,-1]
    test_labels = test_data[:,-1]

    n_train_data = normalize(training_data[:,:-1])
    n_test_data = normalize(test_data[:,:-1])

    global D, N
    D = np.shape(training_data)[1] - 1
    N = np.shape(training_data)[0]

    p = perceptron()

    print(p.z(n_train_data[0]))


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
