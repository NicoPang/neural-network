import math

# =================================================
# functions.py
#   Contains the activation functions and other useful math functions
#   class DifferentiableFunction is to avoid having to define both f and d_f every time
# =================================================

# Sigmoid function for activation
def sigmoid(x):
    return 1.0/(1.0 + (math.e ** (x * -1)))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Mean squared error function for cost
def mse(expected, actual):
    return (actual - expected) ** 2

def d_mse(expected, actual):
    return 2 * (actual - expected)

# Since there is a lot of differentiation in neural network math,
#   I grouped them under one base class
# This way I can expand to more algorithms in case I want to try
#   different activation or cost functions
class DifferentiableFunction:
    def __init__(self, f, d_f):
        self._f = f
        self._d_f = d_f

    def f(self, *args):
        return self._f(*args)

    def d_f(self, *args):
        return self._d_f(*args)

DifferentiableFunction.Sigmoid = DifferentiableFunction(sigmoid, d_sigmoid)
DifferentiableFunction.MSE = DifferentiableFunction(mse, d_mse)