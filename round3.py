import numpy as np
import math

D = 0
N = 0
Classes = 0


class layer:
    perceptrons = []

    def __init__(self):
        self.perceptrons = []


class perceptron:
    def __init__(self, num_inputs):
        self.w = np.random.uniform(
            low=-0.05, high=0.05, size=(num_inputs,))
        self.z = 0
        self.a = 0
        self.delta = 0

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

    neural_network(training_file, test_file, 3, 20, 1)

# layers - number of layers to use
# units per layer - units per HIDDEN layer exlcuding bias input
# rounds - number of training rounds (using whole training set once)
# number of perceptrons of output layer = number of classes
# number of perceptrons of input layer = # of dimensions


def printStructure(P):
    print("P has {} layers:".format(len(P)))
    for l in P:
        print("Layer {} has {} perceptrons, each with {} weights".format(l, len(
            l.perceptrons), len(l.perceptrons[0].w)))


def neural_network(training_file, test_file, layers, units_per_layer, rounds):
    learning_rate = 1

    training_data = np.asarray(read_file(training_file))
    test_data = np.asarray(read_file(test_file))

    training_labels = training_data[:, -1]
    test_labels = test_data[:, -1]

    n_train_data = normalize(training_data[:, :-1])
    n_test_data = normalize(test_data[:, :-1])

    global D, N, Classes
    D = np.shape(training_data)[1] - 1
    N = np.shape(training_data)[0]
    Classes = len(np.unique(training_labels))

    t_train = fixLabels(training_labels)
    t_test = fixLabels(test_labels)

    P = generateLayers(layers, units_per_layer)

    printStructure(P)

    print([len(p.perceptrons[0].w) for p in P])

    for r in range(rounds):
        for n in range(len(n_train_data)):  # len(n_train_data)):
            x = n_train_data[n]

            # Set the input layer to x
            for j in range(D):
                P[0].perceptrons[j].z = x[j]

            # find z for each layer
            for l in range(1, layers):
                for j, P_j in enumerate(P[l].perceptrons):
                    p_z = [p.z for p in P[l-1].perceptrons]
                    p_z.insert(0, 1)
                    P_j.a = np.dot(P_j.w, p_z)
                    P_j.z = P_j._h(P_j.a)

            for j, P_out_j in enumerate(P[-1].perceptrons):
                P_out_j.delta = (
                    P_out_j.z - t_train[n][j]) * P_out_j.z * (1-P_out_j.z)

            for l in range(layers - 2, 0, -1):
                for j, P_j in enumerate(P[l].perceptrons):
                    new_delta = 0
                    for u, P_u in enumerate(P[l+1].perceptrons):
                        new_delta += P_u.delta * P_u.w[j]
                    new_delta *= P_j.z * (1-P_j.z)
                    P_j.delta = new_delta

            for l in range(1, layers):
                for j, P_j in enumerate(P[l].perceptrons):
                    for i, P_i in enumerate(P[l-1].perceptrons):
                        P_j.w[i] -= learning_rate * P_j.delta * P_i.z

        learning_rate *= 0.98

    acc = 0
    for n in range(len(n_test_data)):
        x = n_test_data[n]
        np.insert(x, 0, 1)

        out = feed_forward(P, x)

        M = (-1, -1)
        for i, p in enumerate(out.perceptrons):
            if p.z > M[0]:
                M = (p.z, i)

        this_acc = 0

        if M[1] == training_labels[n]:
            acc += 1
            this_acc = 1

        print("ID = {:5d}, predicted = {:3d}, true = {:3d}, accuracy = {:4.2f}".format(
            int(n + 1), int(M[1]), int(training_labels[n]), float(this_acc)))

    print("classification accuracy: {:6.4f}".format(acc / len(n_test_data)))


def feed_forward(P, x):
    for i in range(len(x)):
        P[0].perceptrons[i].z = x[i]

    for l in range(1, len(P)):
        for P_j in P[l].perceptrons:
            p_z = [p.z for p in P[l-1].perceptrons]
            p_z.insert(0, 1)
            P_j.a = np.dot(P_j.w, p_z)
            P_j.z = P_j._h(P_j.a)

    return P[-1]


def fixLabels(labels):
    global Classes
    new = []
    for t in labels:
        l = [0] * Classes
        l[int(t)] = 1
        new.append(l)

    return new


def generateLayers(layers, units_per_layer):
    global D, Classes
    P = []
    # add input layer
    l1 = layer()
    for _ in range(D):
        _p = perceptron(0)
        _p.bias = 0
        l1.perceptrons.append(_p)

    P.append(l1)

    # add hidden layers
    for _ in range(layers-2):
        lay = layer()
        for _ in range(units_per_layer):
            lay.perceptrons.append(perceptron(len(P[-1].perceptrons) + 1))
        P.append(lay)

    ln = layer()
    for _ in range(Classes):
        ln.perceptrons.append(perceptron(len(P[-1].perceptrons) + 1))
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
