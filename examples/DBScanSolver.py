
# This example shows using DBScanSolver on vrp tests.

import sys
import os
import time

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_dir, 'src'))

from vrp_solvers import TabuSolver, ClarkWright
from vrp_solvers import DBScanSolver
import DWaveSolvers
from input import *
from input_CMT_dataset import *


def get_file_from_user():
    valid_files = []  # Array to store valid file names
    while True:
        # Ask user for the file name
        file_name = input("Please enter a file name: ")
        #path = os.path.join(project_dir, 'tests/cvrp/' + file_name)
        path = os.path.join(project_dir, 'tests/pvrp/' + file_name)

        # Check if the file exists
        if os.path.isfile(path):
            print(f"File '{path}' exists.")
            valid_files.append(file_name)  # Store the valid file in the array
        else:
            print(f"File '{file_name}' does not exist. Please try again.")
        
        # Ask if they want to input another file
        another = input("Do you want to enter another file? (yes/no): ").strip().lower()
        if another != 'yes':
            break  # Exit loop if the user says anything other than 'yes'

    return valid_files




if __name__ == '__main__':

    graph_path = os.path.join(project_dir, 'graphs/small.csv')

    # Parameters for solve function.
    only_one_const = 10000000.
    order_const = 1.

    
    valid_files = get_file_from_user()
    print("Valid files entered:", valid_files)


    for t in valid_files:  #'cmt4.vrp' 'example_small2'
        print("Test : ", t)

        # Reading problem from file.
        #path = os.path.join(project_dir, 'tests/cvrp/' + t)
        #problem, g= create_vrp_problem(path)

        path = os.path.join(project_dir, 'tests/pvrp/' + t)
        problem, g= create_pvrp_problem(path)

        problem.first_source = True
        problem.last_source = True

        #Solving problem on SolutionPartitioningSolver.
        solver = TabuSolver(problem, anti_noiser = False, max_len = 25)
        solution = solver.solve(only_one_const, order_const, solver_type = 'cpu')

        # Checking if solution is correct.
        if solution == None or solution.check() == False:
            print("Tabu Solver hasn't find solution.\n")
        else:
            print("Tabu Solution : ", solution.solution) 
            print("Tabu # of routes :", len(solution.solution))
            print("Tabu Total cost : ", solution.total_cost())      
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
            file_path = os.path.join(test_folder, file_name + '.txt')

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
            
            plot_all_solutions(g, solution.solution, nameOfTestSet + '/' + file_name)

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
