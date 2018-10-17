import numpy as np
import math

D = 0
N = 0
Classes = 0
U = 0

def main():
    training_file = "pendigits_training.txt"
    test_file = "pendigits_test.txt"
    neural_network(training_file, test_file, 5, 8, 5)

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

    x = n_train_data[0]
    z = [0] * U

    for j in range(D):
        z[j] = x[j]
    
    a = [0] * U
    #print(P)
    print(z)
    for l in range(2, layers):
        for j in range(len(P[l])):
            wsum = 0
            for i in range(len(P[l-1])):
                wsum += P[l][i] * z[i]
                #print(P[l][i], z[i])
            a[j] = wsum
            z[j] = h(a[j])

    delta = [0] * U

    for j in range(len(P[-1])):
        delta[j] = (z[j] - training_labels[0]) * z[j] * (1-z[j])
        #print(z[j], training_labels[0])

    
def h(a):
    return 1 / (1 + math.exp(-a))

def generateLayers(layers, units_per_layer):
    global D, Classes, U
    P = []
    #add input layer
    U = Classes + D + layers * units_per_layer
    input_layer = np.random.uniform(low=-0.05, high=0.05, size=(D+1,))

    P.append(input_layer)

    for i in range(layers):
        hidden_layer = np.random.uniform(low=-0.05, high=0.05, size=(units_per_layer+1,))
        P.append(hidden_layer)

    output_layer = np.random.uniform(low=-0.05, high=0.05, size=(Classes+1,))
    P.append(output_layer)
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
