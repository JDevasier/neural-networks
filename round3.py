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
    num_inputs = 0
    bias = 1
    w = 0

    z = 0
    a = 0
    delta = 0

    def __init__(self):
        global D, U
        self.num_inputs = D + 1
        self.w = np.random.uniform(low=-0.05, high=0.05, size=(self.num_inputs,))
        self.z = 0
        self.a = [0] * U
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
        for n in range(len(n_train_data)):
            x = n_train_data[n]
            
            for j in range(D):
                P[0].perceptrons[j].z = x[j]

            for l in range(2, layers):
                for j, P_j in enumerate(P[l].perceptrons):
                    wsum = 0
                    for i, P_i in enumerate(P[l-1].perceptrons):
                        wsum += P_j.w[i] * P_i.z
                    
                    P_j.a = wsum
                    P_j.z = P_j._h(P_j.a)
                    #print(P_j.a, P_j.z)

            for out_j in P[-1].perceptrons:
                out_j.delta = (out_j.z - training_labels[n]) * out_j.z * (1-out_j.z)


            #l-1 through 2 
            for l in range(layers - 1, 1, -1):
                for j, P_j in enumerate(P[l-1].perceptrons):
                    new_delta = 0
                    for u, P_u in enumerate(P[l].perceptrons):
                        new_delta += P_u.delta * P_u.w[j]
                    new_delta *= P_j.z * (1 - P_j.z)
                    P_j.delta = new_delta

            for l in range(2, layers):
                for j, P_j in enumerate(P[l].perceptrons):
                    for i, P_i in enumerate(P[l-1].perceptrons):
                        P_i.w[j] -= learning_rate * P_j.delta * P_i.z
                        #print(learning_rate, P_j.delta,)

        learning_rate *= 0.98
    print("finished learning")
    print(P[-1].perceptrons[4].z)

    acc = 0
    for n in range(100):
        x = n_test_data[n]
        np.insert(x, 0, 1)
        M_p = (0, 0)
        M_z = -1
        for p_i, p in enumerate(P[-1].perceptrons):
            pz = p._z(x)
            print(pz)
            if pz >= M_z:
                M_z = pz
                M_p = p_i
        if M_p == test_labels[n]:
            acc += 1

        #print(M_p, M_z, test_labels[n])
    print("Acc: ", acc / 100)

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
