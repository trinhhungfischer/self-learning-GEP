"""
||========================================================||
|| This is my label for every program which coded by Hung ||
||========================================================||
"""

# Python program to create random chromosome
import random

# Some constant in this program
NP = 50  # Number of individuals in first initial population
K = 2  # Number of ADFs in one chromosome
h = 10  # Number of head elements in each main genome
h1 = 3  # Number of head elements in each ADFs

# Function node set is {+,-,*,/,sqrt(Q),exp(E),sin(s)}
# Terminal node set is {x, y} and every ADFs node must have 2
# Child node, no more, no less
adf_set = list(map(str, range(K)))
func_set = list('+-*/QEs')
ter_set = ['x', 'y']

# Make a seed to reuse a random population
# random.seed(7)

# Traditional method to randomly assign to each element of chromosome
# Depend on its position in chromosome (head, tail)

def init_adf(h1 = 3):
    str = []
    for i in range(h1):
        ele = random.choice(func_set)
        str.append(ele)

    for i in range(h1 + 1):
        ele = random.choice(['0', '1'])
        str.append(ele)

    return ''.join(str)

def init_main(h = 10):
    str = []
    for i in range(h):
        ele = random.choice(func_set + ter_set + adf_set)
        str.append(ele)
    for i in range(h + 1):
        ele = random.choice(ter_set )
        str.append(ele)
    return ''.join(str)

def init_population(NP = 50, h = 10, h1 = 3):
    pop = []
    for i in range (NP):
        pop.append(init_main(h) + init_adf(h1) + init_adf(h1))
    return pop

if __name__ == '__main__':
    pass