"""
||========================================================||
|| This is my label for every program which coded by Hung ||
||========================================================||
"""

# Python program to demonstrate the performance of
# self-learning GEP in symbolic regression problem
# Function node set is {+,-,*,/,sqrt(Q),exp(E),sin(s)}
# Terminal node set is {x, y}
# ADFs node set is in range (K)

# Import some package into this program
import math
import numpy as np
import random
from statistics import mean
import matplotlib.pyplot as plt

NR = 100  # Number of run
NG = 2000  # Number of generation


# Defining some constant of evolution operator
NP = 50  # Number of individuals in first initial population
K = 2  # Number of ADFs in one chromosome
H = 10  # Number of head elements in each main genome
H1 = 3  # Number of head elements in each ADFs
perfect_hit = 0.0001  # The RMSE will converge if it is less than perfect hit


# All kind of node in this program
adf_set = list(map(str, range(K)))
func_set = list('+-*/QEs')
ter_set = ['x', 'y']
input_arg = ['a', 'b']


# This function return the number of maximum child of a function node
# and the type of every node. Because this function is important so I don't
# integrate it to other class. It will return a tuple of two element
# 1. The number of maximum children of a node
# 2. The type of node (0. Terminal node or argument input of ADF 1. Function node 2. ADF node)
# 3. The index of ADF node in ADF set if it is ADF
def is_operator(c=''):
    if c == '+' or c == '-' or c == '*' or c == '/':
        return 2, 1
    if c == 's' or c == 'E' or c == 'Q':
        return 1, 1
    try:
        if adf_set.index(c) + 1:
            return 2, 2, adf_set.index(c)
    except Exception as error:
        return 0, 0


# This class contains function to make random uniform
# training set with NT test case in range (a, b). This function receives
# 2 arguments: (1. Target function 2. Number of Test 3. Tuple (a, b))
def training_create(target_function, NT=200, t=(-1, 1)):
    start, end = t
    input = np.random.uniform(start, end, (NT, 2))
    x = input[:, 0].reshape((-1, 1))
    y = input[:, 1].reshape((-1, 1))
    result = target_function(x, y)
    return np.concatenate((input, result), axis=1)


# This class refer to a node of expression tree
class ET:
    def __init__(self, data):
        self.data = data
        self.maxchild = int(is_operator(data)[0])
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


# Traditional method to randomly assign to each element of chromosome
# Depend on its position in chromosome (head, tail).
# It has function which return a list of chromosome, and receives 4 argument
# 1. Number of population 2. Length of head in main program
# 3. Length of head in ADF 4. Number of ADF in a chromosome
class Population:
    def init_adf(self, h1=3):
        str = []
        for i in range(h1):
            ele = random.choice(func_set)
            str.append(ele)

        for i in range(h1 + 1):
            ele = random.choice(input_arg)
            str.append(ele)

        return ''.join(str)

    def init_main(self, h=10):
        str = []
        for i in range(h):
            ele = random.choice(func_set + ter_set + adf_set)
            str.append(ele)
        for i in range(h + 1):
            ele = random.choice(ter_set)
            str.append(ele)
        return ''.join(str)

    # This function to init a first population to reproduction next generation
    # if you want it will be unchanged in each run, you can choose a random sedd
    def init_population(self, NP=50, h=10, h1=3, k=2):

        # Random seed here
        # random.seed(7)
        pop = []
        for i in range(NP):
            chromosome = self.init_main(h)
            for j in range (k):
                chromosome += self.init_adf(h1)
            pop.append(chromosome)
        return pop

    def alen_set(self, population):
        alen_dict = {}
        all_alen = adf_set + func_set + ter_set + input_arg
        for alen in all_alen:
            alen_dict[alen] = 0
        for individual in population:
            for a in individual:
                alen_dict[a] += 1

        return alen_dict


# This class contains some functions to build, evaluate a tree from chromosome
# First, constructor function receives a chromosome as string for argument
# Second important function build tree from chromosome, receives chromosome as argument
# Third, function RMSE calculate root-mean-square-error of a expression which be demonstrated by
# a chromosome. It receives a dataset as a argument
class Evaluate:

    def __init__(self, chromosome):
        self.chromosome = chromosome
        adf_set = []
        for i in range(K):
            adf_tree = self.build_adf(i)
            adf_set.append(adf_tree)
        self.adf_set = adf_set

    # This function return new expression tree and the valid
    # element from one chromosome or a ADF string. In the other word,
    # the redundant of chromosome isn't valid. This function
    # return a tuple
    #   1. Tree built from a chromosome
    #   2. Length of valid chromosome
    @staticmethod
    def chromosome_to_et(chromosome):  # Some parameters of an expression tree
        l = len(chromosome)
        elements = list(chromosome)
        root = ET(elements[0])

        # Produce a expression tree from a chromosome
        i = 1
        while i < l:
            while root.breadth_fisrt_travel().data:
                node = root.breadth_fisrt_travel()
                if node.maxchild == 2:
                    node.child.append(ET(elements[i]))
                    node.child.append(ET(elements[i + 1]))
                    i += 2
                    if i >= l - 1:
                        break
                if node.maxchild == 1:
                    node.child.append(ET(elements[i]))
                    i += 1
            if not root.breadth_fisrt_travel().data:
                break
        return root, i

    # Build ADFs model from chromosome
    def build_adf(self, k=0):
        start = 2 * H + 1 + (2 * H1 + 1) * k
        end = 2 * H + 1 + (2 * H1 + 1) * (k + 1)
        adf_string = self.chromosome[start:end]
        return self.chromosome_to_et(adf_string)[0]

    # Calculate a tree with one parameter is root and other is value set
    def calculate(self, tree=ET('+'), value_set=[]):

        if tree.maxchild == 2:
            x0 = tree.child[0]
            x1 = tree.child[1]
            if tree.data == '+':
                return self.calculate(x0, value_set) + self.calculate(x1, value_set)
            if tree.data == '-':
                return self.calculate(x0, value_set) - self.calculate(x1, value_set)
            if tree.data == '*':
                return self.calculate(x0, value_set) * self.calculate(x1, value_set)
            if tree.data == '/':
                return self.calculate(x0, value_set) / self.calculate(x1, value_set)
            if is_operator(tree.data)[1] == 2:
                return self.calculate(self.adf_set[is_operator(tree.data)[2]],
                                      [self.calculate(tree.child[0], value_set),
                                       self.calculate(tree.child[1], value_set)])

        if tree.maxchild == 1:
            x0 = tree.child[0]
            if tree.data == 'Q':
                return math.sqrt(self.calculate(x0, value_set))
            if tree.data == 's':
                return math.sin(self.calculate(x0, value_set))
            if tree.data == 'E':
                return math.exp(self.calculate(x0, value_set))
        if tree.maxchild == 0:
            try:
                t = ter_set.index(tree.data)
                return float(value_set[t])
            except:
                t = input_arg.index(tree.data)
                return float(value_set[t])

    def calcualte_chromosome(self, value_set):
        tree = self.chromosome_to_et(self.chromosome)[0]
        return self.calculate(tree, value_set)

    def RMSE(self, dataset=[]):
        n_test = dataset.shape[0]
        result_set = dataset[:, -1].reshape((1, -1))
        train_set = np.delete(dataset, -1, 1)
        train_result = []
        try:
            # Make result test set for storage
            for i in range(n_test):
                tree = self.chromosome_to_et(self.chromosome)[0]
                x = self.calculate(tree, train_set[i, :])
                train_result.append(x)

            # Calculate of this norm of 2 matrix
            result_matrix = np.array(train_result)
            return np.linalg.norm(result_matrix - result_set) / math.sqrt(n_test)
        except Exception as error:
            return 500  # A large float number


# This class is refer to process to mutate a chromosome
# There are much of sub-function of this program so I won't list
# all of them. There two important function
# First, first function "mutated" receives 2 arguments 1. chromosome
# needs to be mutated 2. target chromosome
# Second, de_mutated (different evolution receive two arguments
# 1. chromosome need to be mutated 2. best chromosome in this population
class Mutation:
    def __init__(self, population):
        self.population = population
        self.alen_set = Population().alen_set(population)

    def frequency_base_assignment(self, element, pos, phi, c0 = 1):

        if pos < H:
            if random.random() < phi:
                feasible_set = func_set + adf_set
            else:
                feasible_set = ter_set
        if (pos >= H) and (pos < 2 * H + 1):
            feasible_set = ter_set

        for k in range(K):
            # This is refer to element which be located in
            # head of ADF
            if (pos >= 2 * H + 1 + k * (2 * H1 + 1)) and \
                    (pos < 2 * H + 1 + k * (2 * H1 + 1) + H1):
                # Element type is random between function and input argument
                ele_type = random.randint(0, 1)
                if ele_type == 0:
                    return random.choice(func_set)
                else:
                    return random.choice(input_arg)
            else:
                if (pos >= 2 * H + 1 + k * (2 * H1 + 1) + H1) and \
                        (pos < 2 * H + 1 + (k + 1) * (2 * H1 + 1)):
                    return random.choice(input_arg)

        # This part is roulette wheel to mutate this element
        # First we calculate the probability of alen a can be chosen by
        # roulette wheel in feasible set
        s = sum([self.alen_set[alen] for alen in feasible_set]) + len(feasible_set) * c0
        p = {}
        for key in feasible_set:
            p[key] = (self.alen_set[key] + c0) / s

        # Second, we calculate q-value for all alen in feasible set
        t = 0
        q = {}
        for key, value in p.items():
            t += value
            q[key] = t

        # Finally roulette wheel
        r = random.random()
        key = list([k for k, value in q.items()])
        for i in range (len(key)):
            if q[key[i]] > r:
                break

        return key[i]

    def mutated(self, base,  chromosome, target_chromosome):
        F = random.random()
        l = 2 * H + 1 + K * (2 * H1 + 1)
        phi = 18 / 21

        # Mutation step can decompose into 3 step

        # Step 1:
        # Distance measuring in two chromosomes
        dis_vector = []
        for i in range(l):
            if chromosome[i] == target_chromosome[i]:
                dis_vector.append(0)
            else:
                dis_vector.append(1)

        # Step 2 + Step 3:
        # Distance scaling and distance adding in two chromosomes
        mutated = ''
        for i in range(l):
            if dis_vector[i] == 0:
                mutated += base[i]
            if dis_vector[i] == 1:
                if random.random() < F:
                    mutated_ele = self.frequency_base_assignment(chromosome[i],\
                                                                 i, phi)
                    mutated += mutated_ele
                else:
                    mutated += base[i]

        return mutated

    def de_mutated(self, chrome, best):
        # This is first step of DE /Current-to-best/
        T = self.mutated(chrome, chrome, best)

        # This is second step of DE /Current-to-best/
        r1, r2 = random.choices(self.population, weights=None, cum_weights=None, k=2)
        Y = self.mutated(T, r1, r2)

        return Y


# This is cross-over process
class CrossOver:
    def crossover(self, chrome, mutated):
        CR = random.random()
        crossover_choromosome = ''
        for i in range (len(chrome)):
            if random.random() < CR:
                crossover_choromosome += mutated[i]
            else:
                crossover_choromosome += chrome[i]
        return crossover_choromosome


# This is selection process between two chromosome
class Selection:
    def __init__(self, dataset):
        self.dataset = dataset

    # First we will evaluate of fitness function of all individual in
    # this generation which will store in list
    def fitness(self, population=[], dataset=[]):
        fitness = []
        for i in range(len(population)):
            eva = Evaluate(population[i])
            fitness.append(eva.RMSE(dataset))
        return fitness

    def roulette_selection(self, population=[], num=50):
        fit_list = self.fitness(population)
        s = sum(fit_list)
        l = len(fit_list)

        # probability chosen of individual
        p_chosen = list(ind / s for ind in fit_list)
        q_list = []
        for i in range(len(population)):
            q_list.append(sum(p_chosen[:i]))

        new = []  # This list is to store new generations
        it = 0  # Number of iteration of roulette wheel
        while it < num:
            r = random.random()
            for id in range(l):
                if q_list[id] > r:
                    break

            new.append(population[id - 1])
            it += 1

    def selection(self, chorme, offstring):
        fit_chrome = Evaluate(chorme).RMSE(self.dataset)
        fit_offstring = Evaluate(offstring).RMSE(self.dataset)
        if fit_chrome < fit_offstring:
            return chorme
        else:
            return offstring


# This is SL-GEP which combines all processes
class SL_GEP:
    def __init__(self, NP=50, h=10, h1=3, k=2):
        self.NP = NP
        self.h = h
        self.h1 = h1
        self.k = k

    def main_program(self):
        target_func = lambda x, y: np.sin(x) + np.sin(x*x + x)
        train = training_create(target_func, 200, (-1, 1))

        pop = Population().init_population()
        gen = 0
        fitness_history_max = []
        fitness_history_mean = []

        while (gen < NG):
            pop, means, best = self.proposed_algorithm(pop, train)
            gen += 1
            print('Gen {} has {} best RMSE is {}'.format(gen, means, best))

            fitness_history_max.append(best)
            fitness_history_mean.append(means)

            if (len(pop) == 1):
                break

        return pop, fitness_history_mean, fitness_history_max, gen

    def proposed_algorithm(self, pop, train):

        fitness = []
        for i in range (len(pop)):
            fitness.append(Evaluate(pop[i]).RMSE(train))
            if fitness[i] < perfect_hit:
                break
        if fitness[i] < perfect_hit:
            print('Result solve and equation is:', pop[i])
            return [pop[i]], 0, 0

        best_ind = fitness.index(min(fitness))
        for i in range (len(pop)):
            mutated = Mutation(pop).de_mutated(pop[i], pop[best_ind])
            offstring = CrossOver().crossover(pop[i], mutated)
            pop[i] = Selection(train).selection(pop[i], offstring)

        return pop, mean(fitness), fitness[best_ind]


if __name__ == '__main__':
    i = 0
    suc = 0
    while i < NR:
        pop = Population().init_population()
        pop, fitness_history_mean, fitness_history_max, gen = SL_GEP().main_program()
        plt.plot(list(range(len(fitness_history_mean))), fitness_history_mean, label='Mean Fitness')
        plt.plot(list(range(len(fitness_history_max))), fitness_history_max, label='Best Fitness')
        plt.legend()
        plt.axis((0, gen + 100, 0, 1))
        plt.title('Fitness through the generations')
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.savefig('report' + str(i) + '.png', dpi=300, bbox_inches='tight')
        plt.show()
        if gen < NG:
            suc += 1
    print("The percentage of reach perfect hit:",suc)