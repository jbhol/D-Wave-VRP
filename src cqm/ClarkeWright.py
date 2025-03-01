import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_dir, 'src'))

from vrp_solvers import ClarkWright
import DWaveSolvers
from input import *
from input_CMT_dataset import *

problem, graph = create_vrp_problem_time("tests/cvrp/c202.vrp")
solver = ClarkWright(problem)
solution = solver.solve()
# solution = [[0, 5, 75, 2, 1, 98, 99, 100, 97, 92, 94, 95, 7, 3, 4, 89, 93,0], [0,96, 87, 77, 78, 76, 71, 70, 73, 80, 79, 81, 85, 82, 83, 84, 86, 88, 91, 90,0], [0,30, 29, 27, 24, 22, 20, 21, 8,0], [0,63, 62, 74, 72, 61, 69, 67, 66, 64, 68, 41, 42, 48,0], [0,65, 49, 55, 57, 54, 53, 56, 58, 60, 59, 40, 44, 46, 45, 51, 50, 52, 47, 43,0], [0,6, 34, 36, 33, 32, 31, 35, 37, 38, 39, 28, 26, 23, 18, 19, 16, 14, 12, 15, 17, 13, 25, 9, 11, 10,0], []]
print("Solution : ", solution.solution) 
print("Total cost : ", solution.total_cost())
print("Total Time : ", solution.total_time())
print("CHECK: ", solution.all_weights())
def plot_all_solutions2(g, solutions):
    node_positions = nx.get_node_attributes(g, "pos")
    plt.figure(figsize=(16, 8))

    for i, node in enumerate(g.nodes):
        node_pos = node_positions[node]
        if i == 0:
            plt.annotate(
                "S",
                xy=node_pos,
                xytext=(-5, 5),
                textcoords="offset points",
                fontsize=12,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", facecolor="red", alpha=0.5),
            )
        else:
            plt.annotate(
                i,
                xy=node_pos,
                xytext=(-5, 5),
                textcoords="offset points",
                fontsize=12,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
            )

    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, cmap.N))

    for i, solution in enumerate(solutions):
        path_x, path_y = [], []
        path_x.append(node_positions[next(iter(g.nodes))][0])
        path_y.append(node_positions[next(iter(g.nodes))][1])
        if solution:
            for node_index in solution:
                node = list(g.nodes)[node_index]
                path_x.append(node_positions[node][0])
                path_y.append(node_positions[node][1])
            plt.plot(path_x, path_y, color=colors[i % len(colors)], label=f"Route {i+1}")

    plt.legend(loc="best")
    plt.axis("off")
    plt.show()


plot_all_solutions2(graph, solution.solution)