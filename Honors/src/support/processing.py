import numpy as np
import random

from support.functions import DifferentiableFunction

# =================================================
# processing.py
#   Processing data sent into a model
#   Obviously intended just for neural networks
# =================================================

# ==============
# Initialization
# ==============

def initialize(f, model, *args):
    f(model, *args)

## Xavier initialization

def xavier_initialize(model):
    random.seed(model.get_seed())
    for i in range(model.get_num_wb()):
        xavier_initialize_weight_matrix(model.get_weight_matrix(i), model.get_layer_size(i))
        xavier_initialize_weight_matrix(model.get_bias_vector(i), model.get_layer_size(i))

def xavier_initialize_weight_matrix(a, num_inputs):
    with np.nditer(a, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = xavier_weight(num_inputs)
    
    return

def xavier_weight(n):
    lower = -(1.0/np.sqrt(float(n)))
    upper = 1.0/np.sqrt(float(n))
    rand = random.random()
    return lower + rand * (upper - lower)

# ===================
# Forward Propogation
# ===================

# Forward Propogation
#   Returns the raw values at each node (as in, activation not applied)
def forward_propogate(model, sample):
    raw_values = []

    input = sample
    raw_values.append(input.copy())

    for i in range(model.get_num_wb()):
        try:
            input = np.matmul(model.get_weight_matrix(i), input)
            input = np.add(input, model.get_bias_vector(i))
            raw_values.append(input.copy())

            with np.nditer(input, op_flags=['readwrite']) as it:
                for x in it:
                    x[...] = model.activate(x)
        except ValueError:
            print("faulty input:")
            print(input)
            print("attempted multiply:")
            print(model.get_weight_matrix(i))
            print("attempted add:")
            print(model.get_bias_vector(i))
            print("error: bad input led to bad matrix multiplication")
            return []

    return raw_values

# ==============
# Back Propogate
# ==============

# Back propogation
#   model - the model
#   sample - input data
#   goal - intended output vector
#   activations - data to help process back propogation. 
# This function is intended to take the output from forward_propogate
def back_propogate(model, raw_values, goal, cost_f = DifferentiableFunction.MSE):
    # memoization of rates for nodes to avoid complex calculations being repeated
    memos = [np.zeros(model.get_layer_size(i + 1)) for i in range(model.get_num_layers() - 1)]

    # data to return
    delta_w = model.generate_blank_weights()
    delta_b = model.generate_blank_biases()


    # Iterate through layers in reverse order. 
    #   For each iteration:
    #   1.      Calculate rate of the node value
    #   2.      Calculate rate of the activation function
    #   2.5.    Memoize 1. * 2.
    #   3.      Calculate rate of weight/bias
    for i in reversed(range(model.get_num_wb())):
        # Part 1-2.5
        # Special case - first layer. Rate of node value determined by cost function
        if i == model.get_num_wb() - 1:
            for k in range(model.get_layer_size(i + 1)):
                c_o_a = cost_f.d_f(model.activate(raw_values[i + 1][k]), goal[k])
                c_o_z = model.d_activate(raw_values[i + 1][k])
                memos[i][k] = c_o_a * c_o_z
        # General case - memoization of the other layers (da/dz *  dC/da)
        else:
            for k in range(model.get_layer_size(i + 1)):
                c_o_a = 0.0
                for j in range(model.get_layer_size(i + 2)):
                    c_o_a += model.get_weight(i + 1, j, k) * memos[i + 1][j]
                
                c_o_z = model.d_activate(raw_values[i + 1][k])
                memos[i][k] = c_o_a * c_o_z
        # Part 3
        # Weight + bias
        for j in range(model.get_layer_size(i + 1)):
            for k in range(model.get_layer_size(i)):
                delta_w[i][j][k] = model.activate(raw_values[i][k]) * memos[i][j]
                delta_b[i][j] = memos[i][j]

    # Return: a tuple containing weight changes and bias changes
    return [delta_w, delta_b]