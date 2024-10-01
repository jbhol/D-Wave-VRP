# This example shows using DBScanSolver on vrp tests.

import sys
import os
import time

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_dir, 'src'))

from vrp_solvers import TabuSolver
from vrp_solvers import DBScanSolver
import DWaveSolvers
from input import *
from input_CMT_dataset import *

if __name__ == '__main__':

    graph_path = os.path.join(project_dir, 'graphs/medium.csv')

    # Parameters for solve function.
    only_one_const = 10000000.
    order_const = 1.

    for t in ['cmt4.vrp']:  #'cmt4.vrp' 'example_small2'
        print("Test : ", t)

        # Reading problem from file.
        path = os.path.join(project_dir, 'tests/cvrp/' + t)
        #path = os.path.join(project_dir, 'tests/cvrp/' + t + '.test')
        #problem = read_full_test(path, graph_path, capacity = True)
        #problem = read_test(path, capacity = True)
        problem, g= create_vrp_problem(path)
        problem.first_source = True
        problem.last_source = True

        # Solving problem on SolutionPartitioningSolver.
        solver = TabuSolver(problem, anti_noiser = False, max_len = 25)
        solution = solver.solve(only_one_const, order_const, solver_type = 'cpu')

        # Checking if solution is correct.
        if solution == None or solution.check() == False:
            print("Tabu Solver hasn't find solution.\n")
        else:
            print("Tabu Solution : ", solution.solution) 
            print("Tabu Total cost : ", solution.total_cost())
            print("\n")            
            file_name = 'tests/cvrp/' + t + '.txt'
            if os.path.exists(file_name):
                file_name = 'tests/cvrp/' + t + '_new.txt'
            original_stdout = sys.stdout
            with open(file_name, "w") as file:
                sys.stdout = file
                print("Tabu Solution : ", solution.solution) 
                print("Tabu Total cost : ", solution.total_cost())
                print(time.time())            
                sys.stdout = original_stdout
            plot_all_solutions(g, solution.solution, t)

        # Solving problem on SolutionPartitioningSolver.
        #solver = DBScanSolver(problem, anti_noiser = False, max_len = 25)
        #solution = solver.solve(only_one_const, order_const, solver_type = 'cpu')

        # Checking if solution is correct.
        #if solution == None or solution.check() == False:
        #    print("DBScan Solver hasn't find solution.\n")
        #else:         
        #    print("DBScan Solution : ", solution.solution) 
        #    print("DBScan Total cost : ", solution.total_cost())
        #    print("\n")
