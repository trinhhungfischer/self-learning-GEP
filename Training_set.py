"""
||========================================================||
|| This is my label for every program which coded by Hung ||
||========================================================||
"""

import numpy as np
from random import random, seed

# Some constants in this program
NT = 200 # Number of test case

# seed(7) # Make one set of random number to reuse

def target_function(x, y):
    return x**4 + x**3 + x**2 + x

def xy_create():
    input = np.random.uniform(-1, 1, (NT, 2))
    return input * 10

def training_create():
    input = xy_create()
    x = input[:, 0].reshape((-1, 1))
    y = input[:, 1].reshape((-1, 1))
    reslut = target_function(x, y)
    return np.concatenate((input, reslut), axis=1)

if __name__ == '__main__':
    print(training_create())