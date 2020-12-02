"""
||========================================================||
|| This is my label for every program which coded by Hung ||
||========================================================||
"""

from Encode_Evaluate import RMSE
from Initialization import init_population
from Reproduction import reproduction
from time import time

NR = 100 # Number of run
NG = 200 # Number of generation

# Defining some constant of evolution operator
NP = 50 # Number of individuals in first initial population
K = 2 # Number of ADFs in one chromosome
h = 10 # Number of head elements in each main genome
h1 = 3 # Number of head elements in each ADFs
perfect_hit = 0.01 # The RMSE will converge if it is less than perfect hit

# This function return false when terminate condition is wrong
def converge_con(populaion = [], perfect_hit = 0.01):
    for chrome in populaion:
        try:
            if RMSE(chrome) < perfect_hit:
                return True, chrome
        except Exception as error:
            pass
    return False, 1

# This function demonstrate one run
def each_run():
    # Initializing first random population as parent
    population = init_population(NP, h)

    # This loop is iteration of preproduction, each generation
    # has been referred to reproduction function
    generation = 1
    while (not converge_con(population, perfect_hit)[0]) and (generation < NG):
        print('generation {} in time: {}'.format(generation, time() - start_time))
        print(population)
        population = reproduction(population, pm, pc1, pc2, pis, pris, pi)
        generation += 1

    # Print the next generation and its index (optional)
    print('This is generation {} in time: {}'.format(generation, time() - start_time))
    print(population)
    print('The solution is ', converge_con(population)[1])

    return converge_con(population)[0]

"""
This is main part of program to demonstrate GEP
efficient in practical by success rate 
"""

if __name__ == '__main__':
    suc = 0
    for i in range(NR):
        print("THIS IS THE RUN", i+1)
        if each_run():
            suc += 1

    print('Success rate in 100 runs is', suc, '%')
