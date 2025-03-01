# This example shows using DBScanSolver on vrp tests.

import sys
import os
import time

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_dir, 'src'))

from vrp_local_search_solvers import TabuLocalSearchSolver
from input import *
from input_CMT_dataset import *

if __name__ == '__main__':

    graph_path = os.path.join(project_dir, 'graphs/medium.csv')

    # Parameters for solve function.
    only_one_const = 10000000.
    order_const = 1.

    for t in ['p-n16-k8.vrp']:  #'cmt4.vrp' 'example_small2'
        print("Test : ", t)

        # Reading problem from file.
        path = os.path.join(project_dir, 'tests/pvrp/' + t)
        #path = os.path.join(project_dir, 'tests/cvrp/' + t + '.test')
        #problem = read_full_test(path, graph_path, capacity = True)
        #problem = read_test(path, capacity = True)
        problem, g= create_pvrp_problem(path)
        problem.first_source = True
        problem.last_source = True

        # Solving problem on SolutionPartitioningSolver.
        solver = TabuLocalSearchSolver(problem)
        solution = solver.solve(only_one_const, order_const, solver_type = 'cpu')

        # Checking if solution is correct.
        if solution == None or solution.check() == False:
            print("Tabu Local Search Solver hasn't find solution.\n")
        else:
            print("Tabu Local Search Solution : ", solution.solution) 
            print("Tabu Local Search Total cost : ", solution.total_cost())
            total_power, total_time = solution.total_power_and_time()
            print("Tabu Total time :", total_time)
            print("Tabu Total power :", total_power)    
            print("\n")          
            # Get the name of the test set, e.g., "cmt1" from "cmt1.vrp"
            nameOfTestSet = t.split('.')[0]  # If t = "cmt1.vrp", nameOfTestSet will be "cmt1"

            # Create a specific folder for the test set if it doesn't exist
            test_folder = os.path.join('outputs/files', nameOfTestSet)
            os.makedirs(test_folder, exist_ok=True)

            # Create a specific folder for the image  if it doesn't exist
            image_folder = os.path.join('outputs/images', nameOfTestSet)
            os.makedirs(image_folder, exist_ok=True)
            

            # List all items in the test folder (which now contains files related to "cmt1")
            items = os.listdir(test_folder)

            # Filter the items that contain 'cmt1' (i.e., nameOfTestSet)
            matching_items = [item for item in items if nameOfTestSet in item]

            # Count the number of test files already present
            numOfTests = len(matching_items)

            # Generate the new file name for the next test (e.g., "cmt1_test3")
            file_name = nameOfTestSet + "_test" + str(numOfTests + 1)

            # Define the path for the new file
            file_path = os.path.join(test_folder, 'LS' + file_name + '.txt')

            # Redirect output to the new file
            original_stdout = sys.stdout
            with open(file_path, "w") as file:
                sys.stdout = file
                print("Tabu Solution : ", solution.solution) 
                print("Tabu # of routes :", len(solution.solution))
                print("Tabu Total cost : ", solution.total_cost())
                total_power, total_time = solution.total_power_and_time()
                print("Tabu Total time :", total_time)
                print("Tabu Total power :", total_power)    
                print("Best solution was found on counter : ", solution.step)
                print(time.time())
            sys.stdout = original_stdout

            # Plot all solutions and save with the same file name (cmt1_test3)
            
            plot_all_solutions(g, solution.solution, nameOfTestSet + '/LS' + file_name)