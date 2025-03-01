
# This example shows using DBScanSolver on vrp tests.

import sys
import os
import time

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_dir, 'src'))

from DroneDelivery import simulate
from input import *
from input_CMT_dataset import *


def get_file_from_user():
    valid_files = []  # Array to store valid file names
    while True:
        # Ask user for the file name
        file_name = input("Please enter a file name: ")
        path = os.path.join(project_dir, 'tests/cvrp/' + file_name)
        
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
        path = os.path.join(project_dir, 'tests/cvrp/' + t)

        #Solving problem
        simulate(path)
