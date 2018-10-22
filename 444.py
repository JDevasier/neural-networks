import numpy as np
import math

D = 0
N = 0
Classes = 0
U = 0

class layer:
    perceptrons = []
    def __init__(self):
        self.perceptrons = []


class perceptron:

    def __init__(self):
        global D, U
        self.num_inputs = D + 1
        self.w = np.random.uniform(low=-0.05, high=0.05, size=(self.num_inputs,))
        self.z = 0
        self.a = 0
        self.delta = 0
        self.bias = 1

    def _a(self, x):
        return np.dot(self.w, x)

    def _z(self, x):
        x = np.insert(x, 0, 1)
        return self._h(self._a(x))

    def _h(self, a):
        return 1 / (1 + math.exp(-a))


def main():
    training_file = "pendigits_training.txt"
    test_file = "pendigits_test.txt"

    neural_network(training_file, test_file, 6, 15, 1)

# layers - number of layers to use
# units per layer - units per HIDDEN layer exlcuding bias input
# rounds - number of training rounds (using whole training set once)
# number of perceptrons of output layer = number of classes
# number of perceptrons of input layer = # of dimensions


def neural_network(training_file, test_file, layers, units_per_layer, rounds):
    learning_rate = 1

    training_data = np.asarray(read_file(training_file))
    test_data = np.asarray(read_file(test_file))

    training_labels = training_data[:,-1]
    test_labels = test_data[:,-1]

    n_train_data = normalize(training_data[:,:-1])
    n_test_data = normalize(test_data[:,:-1])

    global D, N, Classes, U
    D = np.shape(training_data)[1] - 1
    N = np.shape(training_data)[0]
    Classes = len(np.unique(training_labels))

    P = generateLayers(layers, units_per_layer)

    for r in range(rounds):
        for n in range(1):#len(n_train_data)):
            x = n_train_data[n]
            
            for j in range(D):
                P[0].perceptrons[j].z = x[j]
            

def generateLayers(layers, units_per_layer):
    global D, Classes, U
    P = []
    #add input layer
    l1 = layer()
    for i in range(D):
        U += 1
        _p = perceptron()
        _p.bias = 0
        l1.perceptrons.append(_p)
        
    P.append(l1)

    #add hidden layers
    for i in range(layers-2):
        lay = layer()
        for _ in range(units_per_layer):
            U += 1
            lay.perceptrons.append(perceptron())
        P.append(lay)

    ln = layer()
    for i in range(Classes):
        U += 1
        ln.perceptrons.append(perceptron())
    P.append(ln)
    return P


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
