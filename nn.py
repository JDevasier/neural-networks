import numpy as np
import math
import random

D = 0
N = 0
Classes = 0


class neuralnetwork:
    learning_rate = 1

    def __init__(self, num_inputs, num_hidden, num_classes):
        self.num_inputs = num_inputs

        self.hidden = layer(num_hidden, random.uniform(-0.05, 0.05))
        self.output = layer(num_classes, random.uniform(-0.05, 0.05))

        for n in range(len(self.hidden.perceptrons)):
            for _ in range(self.num_inputs):
                self.hidden.perceptrons[n].weights.append(
                    random.uniform(-0.05, 0.05))

        for o in range(len(self.output.perceptrons)):
            for _ in range(len(self.hidden.perceptrons)):
                self.output.perceptrons[o].weights.append(
                    random.uniform(-0.05, 0.05))

    def feed_forward(self, inputs):
        self.output.feed_forward(self.hidden.feed_forward(inputs))

    def train(self, train_input, train_output):
        self.feed_forward(train_input)

        # output delta
        output_delta = []
        for o in range(len(self.output.perceptrons)):
            output_delta.append(
                self.output.perceptrons[o].calc_partial_E_z(train_output[o]))

        # hidden layer delta
        hidden_delta = []
        for h in range(len(self.hidden.perceptrons)):
            rsum = 0
            for o in range(len(self.output.perceptrons)):
                rsum += output_delta[o] * self.output.perceptrons[o].weights[h]

            hidden_delta.append(rsum * self.hidden.perceptrons[h].output *
                                (1 - self.hidden.perceptrons[h].output))

        # update output weights
        for o in range(len(self.output.perceptrons)):
            for w in range(len(self.output.perceptrons[o].weights)):
                partial_E_w = output_delta[0] * \
                    self.output.perceptrons[o].calc_partial_z_w(w)

                self.output.perceptrons[o].weights[w] -= self.learning_rate * partial_E_w

        # update input weights
        for o in range(len(self.hidden.perceptrons)):
            for w in range(len(self.hidden.perceptrons[o].weights)):
                partial_E_w = output_delta[0] * \
                    self.hidden.perceptrons[o].calc_partial_z_w(w)

                self.hidden.perceptrons[o].weights[w] -= self.learning_rate * partial_E_w

    def calc_error(self, training_data, training_labels):
        err = 0
        for t in range(len(training_data)):
            t_in = training_data[t]
            t_out = training_labels[t]

            self.feed_forward(t_in)

            for o in range(len(t_out)):
                err += self.output.perceptrons[o].calc_error(t_out[o])

        return err

    def calc_accuracy(self, test_data, test_labels):
        acc = 0

        for t in range(len(test_data)):
            t_in = test_data[t]
            t_out = test_labels[t]

            #print(t_in, t_out)

            self.feed_forward(t_in)

            print("O:", self.output.get_outputs())

        return acc / len(test_data)


class layer:

    def __init__(self, num_perceptrons, bias):
        self.bias = bias
        self.perceptrons = []
        for _ in range(num_perceptrons):
            self.perceptrons.append(perceptron(self.bias))

    def inspect(self):
        print('Perceptrons:', len(self.perceptrons))
        for n in range(len(self.perceptrons)):
            print(' Perceptron', n)
            for w in range(len(self.perceptrons[n].weights)):
                print('  Weight:', self.perceptrons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        outputs = []
        for p in self.perceptrons:
            outputs.append(p.calc_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for p in self.perceptrons:
            outputs.append(p.output)
        return outputs


class perceptron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def calc_output(self, inputs):
        self.inputs = inputs
        self.output = self.sigmoid(self.calc_a())
        return self.output

    def calc_a(self):
        rsum = 0
        for i in range(len(self.inputs)):
            rsum += self.inputs[i] * self.weights[i]
        return rsum + self.bias

    def sigmoid(self, a):
        if a < -500:
            return 0
        return 1 / (1 + math.exp(-a))

    def calc_partial_E_z(self, t):
        return (self.output - t) * (self.output * (1-self.output))

    def calc_error(self, t):
        return 0.5 * (t - self.output) ** 2

    def calc_partial_z_w(self, i):
        return self.inputs[i]


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

    network = neuralnetwork(D, units_per_layer, Classes)
    for r in range(rounds):
        for i in range(len(n_train_data)):
            network.train(n_train_data[i], t_train[i])
        learning_rate *= 0.98

    print(network.calc_accuracy(n_test_data, t_test))


def fixLabels(labels):
    global Classes
    new = []
    for t in labels:
        l = [0] * Classes
        l[int(t)] = 1
        new.append(l)

    return new


def normalize(arr):
    a = np.asarray(arr)
    _max = np.amax(a)

    b = np.multiply(1 / _max, a)
    return b


def read_file(fname):
    lines = []
    with open(fname) as f:
        for line in f:
            lines.append(list(map(int, line.split())))

    return lines


main()
