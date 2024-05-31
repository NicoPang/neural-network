import numpy as np
import random

from support.functions import DifferentiableFunction

# =================================================
# network.py
#   Contains the representation of a neural network, and all basic functions pertaining to manipulating it
#   Uses matrix representation for easy computing
# =================================================

# TODO list:
# - Implement backpropogation
# - Add changing activation functionality

class BasicNeuralNetworkModel:
    def __init__(self, layer_sizes, activation = DifferentiableFunction.Sigmoid, seed = random.seed()):
        # Information on the layers
        self._nLayers = len(layer_sizes)
        self._sLayers = layer_sizes

        # Weights and biases, stored as np arrays
        self._ws = self.generate_blank_weights()
        self._bs = self.generate_blank_biases()

        # The activation function
        self._activation = activation

        self._seed = seed

    # Getter/setters for layers, weights, biases, seed
    def get_num_layers(self):
        return self._nLayers

    def get_layer_sizes(self):
        return self._sLayers

    def get_layer_size(self, layer):
        if layer >= self.get_num_layers() or layer < 0:
            raise IndexError("Bad index. Expected range: [0, {}] | Got: {}".format(self.get_num_layers() - 1, layer))

        return self._sLayers[layer]

    def get_num_wb(self):
        return self.get_num_layers() - 1

    def get_weight(self, index, j, k):
        if index >= len(self._ws) or index < 0:
            raise IndexError("Bad index. Expected range: [0, {}] | Got: {}".format(len(self._ws) - 1, index))

        if j >= len(self._ws[index]) or j < 0:
            raise IndexError("Bad j. Expected range: [0, {}] | Got: {}".format(len(self._ws[index]) - 1, j))

        if k >= len(self._ws[index][j]) or k < 0:
            raise IndexError("Bad k. Expected range: [0, {}] | Got: {}".format(len(self._ws[index][j]) - 1, k))

        return self._ws[index][j][k]

    def set_weight(self, index, j, k, val):
        if index >= len(self._ws) or index < 0:
            raise IndexError("Bad index. Expected range: [0, {}] | Got: {}".format(len(self._ws) - 1, index))

        if j >= len(self._ws[index]) or j < 0:
            raise IndexError("Bad j. Expected range: [0, {}] | Got: {}".format(len(self._ws[index]) - 1, j))

        if k >= len(self._ws[index][j]) or k < 0:
            raise IndexError("Bad k. Expected range: [0, {}] | Got: {}".format(len(self._ws[index][j]) - 1, k))

        self._ws[index][j][k] = val
    
    def get_weight_matrix(self, index):
        if index >= len(self._ws) or index < 0:
            raise IndexError("Bad index. Expected range: [0, {}] | Got: {}".format(len(self._ws) - 1, index))

        return self._ws[index]

    def set_weight_matrix(self, index, matrix):

        expected_dim = self._ws[index].shape
        received_dim = matrix.shape

        if received_dim != expected_dim:
            raise ValueError("Bad matrix dimensions. Expected: {} x {} matrix | Got: {} x {} matrix.".format(expected_dim[0], expected_dim[1], received_dim[0], received_dim[1]))

        for j in range(received_dim[0]):
            for k in range(received_dim[1]):
                self.set_weight(index, j, k, matrix[j][k])

    def get_weights(self):
        return self._ws

    def set_weights(self, weights):
        if len(weights) != len(self._ws):
            raise ValueError("Wrong number of matrices. Expected: {} | Got: {}".format(len(self._ws), len(weights)))

        for i in range(self.get_num_layers() - 1):
            self.set_weight_matrix(i, weights[i])

    def get_bias(self, index, j):
        if index < 0 or index > len(self._bs) - 1:
            raise IndexError("Bad index. Expected range: [0, {}] | Got: {}".format(len(self._bs) - 1, index))

        if j >= len(self._ws[index]) or j < 0:
            raise IndexError("Bad j. Expected range: [0, {}] | Got: {}".format(len(self._bs[index]) - 1, j))

        return self._bs[index][j]

    def set_bias(self, index, j, val):
        if index < 0 or index > len(self._bs) - 1:
            raise IndexError("Bad index. Expected range: [0, {}] | Got: {}".format(len(self._bs) - 1, index))

        if j >= len(self._ws[index]) or j < 0:
            raise IndexError("Bad j. Expected range: [0, {}] | Got: {}".format(len(self._bs[index]) - 1, j))

        self._bs[index][j] = val

    def get_bias_vector(self, index):
        if index < 0 or index > len(self._bs) - 1:
            raise IndexError("Bad index. Expected range: [0, {}] | Got: {}".format(len(self._bs) - 1, index))

        return self._bs[index]

    def set_bias_vector(self, index, vector):
        if len(vector) != len(self._bs[index]):
            raise ValueError("Bad vector dimension. Expected: {} | Got: {}".format(len(self._bs[index]), len(vector)))

        for j in range(len(vector)):
            self.set_bias(index, j, vector[j])

    def get_biases(self):
        return self._bs

    def set_biases(self, biases):
        if len(biases) != len(self._bs):
            raise ValueError("Wrong number of vectors. Expected: {} | Got: {}".format(len(self._bs), len(biases)))

        for i in range(len(biases)):
            self.set_bias_vector(i, biases[i])

    def get_seed(self):
        return self._seed

    def set_seed(self, s):
        self._seed = s

    def adjust_weights(self, weight_changes):
        for i in range(len(self._ws)):
            for j in range(len(self._ws[i])):
                for k in range(len(self._ws[i][j])):
                    self.set_weight(i, j, k, self.get_weight(i, j, k) + weight_changes[i][j][k])

    def adjust_biases(self, bias_changes):
        for i in range(len(self._bs)):
            for j in range(len(self._bs[i])):
                self.set_bias(i, j, self.get_bias(i, j) + bias_changes[i][j])

    def adjust_weights_biases(self, weight_changes, bias_changes):
        self.adjust_weights(weight_changes)
        self.adjust_biases(bias_changes)

    # Activation function
    def activate(self, *args):
        return self._activation.f(*args)

    # Activation function derivative, for backpropogation
    def d_activate(self, *args):
        return self._activation.d_f(*args)

    # Returns empty array of weight matrices
    def generate_blank_weights(self):
        ws = []
        for i in range(self.get_num_wb()):
            ws.append(np.zeros((self.get_layer_size(i + 1), self.get_layer_size(i))))
        
        return ws

    # Return empty array of bias vectors
    def generate_blank_biases(self):
        bs = []
        for i in range(self.get_num_wb()):
            bs.append(np.zeros(self.get_layer_size(i + 1)))
        
        return bs

    # Prints weights of current model
    def print_weights(self):
        for i in range(self.get_num_wb()):
            print("Weights from layers {} and {}".format(i, i + 1))
            print(self._ws[i])

    # Prints biases of current model
    def print_biases(self):
        for i in range(self.get_num_wb()):
            print("Biases from layers {} and {}".format(i, i + 1))
            print(self._bs[i])