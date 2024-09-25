import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_dir, 'src'))

from vrp_solvers import ClarkWright
import DWaveSolvers
from input import *
from input_CMT_dataset import *

problem, graph = create_vrp_problem("tests/cvrp/cmt1.vrp")
solver = ClarkWright(problem)
solution = solver.solve()
print("Solution : ", solution.solution) 
print("Total cost : ", solution.total_cost())