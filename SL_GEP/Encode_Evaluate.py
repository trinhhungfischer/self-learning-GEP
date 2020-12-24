"""
||========================================================||
|| This is my label for every program which coded by Hung ||
||========================================================||
"""

# Python program to build and evaluate expression tree
# Function node set is {+,-,*,/,sqrt(Q),exp(E),sin(s)}
# Terminal node set is {x, y, ?}
# An expression tree node


# Import some package into this program
import math
import numpy as np
from Training_set import training_create

# Some constant in this program
K = 2
adf_set = list(map(str, range(K)))
terminal_set = ['x', 'y']

# Some function to work with ADF node
def is_adf(c = ''):
    for i in range (K):
        if int(c) == i:
            return True
    return False

def calculate_adf(adf_tree):
    pass
# This function return the number of maximum child of a function node
def is_operator(c):
    if c == '+' or c == '-' or c == '*' or c == '/':
        return 2
    if c == 's' or c == 'E' or c == 'Q':
        return 1
    return 0


# This class refer to a node of expression tree
class ET:
    def __init__(self, data):
        self.data = data
        self.maxchild = int(is_operator(data))
        self.child = list()

    # Breadth-first Travelling Search return function node not full
    def breadth_fisrt_travel(self):
        queue = [self]
        while len(queue):
            node = queue.pop(0)
            if (len(node.child) < node.maxchild):
                return node
            for node_child in node.child:
                queue.append(node_child)
        return ET(0)

    # Calculate a tree with one parameter is root and other is value set
    def calculate(self, value_set = []):
        v = value_set
        if self.maxchild == 2:
            x0 = self.child[0]
            x1 = self.child[1]
            if self.data == '+':
                return x0.calculate(v) + x1.calculate(v)
            if self.data == '-':
                return x0.calculate(v) - x1.calculate(v)
            if self.data == '*':
                return x0.calculate(v) * x1.calculate(v)
            if self.data == '/':
                return x0.calculate(v) / x1.calculate(v)
            if is_adf(self.data):
                pass


        if self.maxchild == 1:
            x0 = self.child[0]
            if self.data == 'Q':
                return math.sqrt(x0.calculate(v))
            if self.data == 's':
                return math.sin(x0.calculate(v))
            if self.data == 'E':
                return math.exp(x0.calculate(v))

        if terminal_set.index(self.data) + 1:
            t = terminal_set.index(self.data)
            return float(v[t])


def chromosome_to_et(chromosome = ''):
    # Some parameters of an expression tree
    l = len(chromosome)
    elements = list(chromosome)
    root = ET(elements[0])

    # Produce a expression tree
    i = 1
    while i < l:
        while root.breadth_fisrt_travel().data:
            node = root.breadth_fisrt_travel()
            if node.maxchild == 2:
                node.child.append(ET(elements[i]))
                node.child.append(ET(elements[i+1]))
                i += 2
                if i >= l-1:
                    break
            if node.maxchild == 1:
                node.child.append(ET(elements[i]))
                i += 1
        if not root.breadth_fisrt_travel().data:
            break
    return root, i

# This function return new expression tree and the valid element in one chromosome
# In the other word, the redundant of chromosome isn't valid
def valid_chromosome(chromosome = ''):
    valid_len = chromosome_to_et(chromosome)[1]
    return chromosome[:valid_len], valid_len

# This function to calculate chromosome with particular data set
def calculate_chromosome(chromosome = '', data_set = [1, 4]):
    root = chromosome_to_et(chromosome)[0]
    return root.calculate(data_set)

# Calculate RMSE of one chromosome
dataset = training_create()
def RMSE(chromosome = ''):
    pass
    n_test = dataset.shape[0]
    result_set = dataset[:, -1].reshape((1, -1))
    train_set = np.delete(dataset, -1, 1)
    train_result = []

    # Make result test set for storage
    for i in range (n_test):
        x = calculate_chromosome(chromosome, train_set[i, :])
        train_result.append(x)

    # Calculate of this norm of 2 matrix
    result_matrix = np.array(train_result)
    return np.linalg.norm(result_matrix - result_set) / math.sqrt(n_test)

if __name__ == '__main__':
    print(RMSE('*x++/*x**+xxxxxx*xxxx'))

