import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import product
from vrp_problem import VRPProblem

def parse_file_time(file_path):
    data = {}
    current_section = None


    with open(file_path, "r") as file:
        iterator = iter(file)
        for line in iterator:
            line = line.strip()
            
            if line.startswith("NUMBER"):
                try:
                    next_line = next(iterator).strip()
                    vehicle_number, capacity = next_line.split(maxsplit=1)
                    data["capacity"] = int(capacity)
                    data["vehicleNumber"] = int(vehicle_number)
                except StopIteration:
                    print("End of file reached unexpectedly after 'NUMBER'.")
                except ValueError:
                    print("Error processing vehicle number and capacity.")
            elif line.startswith("CUST NO."):
                current_section = "data_section"
                data["node_coords"] = {}
                data['demands'] = {}
                data['time_interval'] = {}


            if current_section == "data_section" and line and not line.startswith("NUMBER") and not line.startswith("CUST NO."):
                try:
                    cust_no, x_coord, y_coord, demand, ready_time, due_date, service = line.split(maxsplit=7)
                    data["service"] = int(service)
                    cust_no = str(int(cust_no))
                    data["node_coords"][cust_no] = (float(x_coord), float(y_coord))
                    data["demands"][cust_no] = int(demand)
                    data["depot"] = [0.0]
                    #vehicle cannot begin before ready_time and has to start before or at due_date
                    data["time_interval"][cust_no] = (int(ready_time), int(due_date))                    
                except ValueError:
                    print(f"Error processing customer data for line: {line}")

    return data

def parse_file(file_path):
    data = {}
    current_section = None


    with open(file_path, "r") as file:
        iterator = iter(file)
        for line in iterator:
            line = line.strip()
            
            if line.startswith("CAPACITY"):
                try:
                    capacity = line.split(" : ")[1]
                    data["capacity"] = int(capacity)
                    data["vehicleNumber"] = int(6)
                except StopIteration:
                    print("End of file reached unexpectedly after 'NUMBER'.")
                except ValueError:
                    print("Error processing vehicle number and capacity.")
            elif line.startswith("NODE_COORD_SECTION"):
                current_section = "coord_section"
                data["node_coords"] = {}
                data['time_interval'] = {}
            elif line.startswith("DEMAND_SECTION"):
                current_section = "demand_section"
                data['demands'] = {}
            else:
                current_section == "ignore"
            


            if current_section == "coord_section" and line:
                try:
                    cust_no, x_coord, y_coord = line.split(maxsplit=3)
                    cust_no = str(int(cust_no) - 1)
                    data["service"] = int(0)
                    data["node_coords"][cust_no] = (float(x_coord), float(y_coord))
                    data["depot"] = [0.0]
                    #vehicle cannot begin before ready_time and has to start before or at due_date
                    data["time_interval"][cust_no] = (int(0), int(0))                    
                except ValueError:
                    print(f"Error processing customer data for line: {line}")
            elif current_section == "demand_section" and line:
                try:
                    cust_no, demand = line.split(maxsplit=2)
                    cust_no = str(int(cust_no) - 1)
                    data["demands"][cust_no] = int(demand)
                except ValueError:
                    print("ignore")
    return data





    








def plot_all_solutions(g, solutions, t):
    """Plots all solutions on a single graph, with nodes and paths."""

    node_positions = nx.get_node_attributes(g, "pos")

    plt.figure(figsize=(8, 6))  

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

    colors = plt.cm.get_cmap("tab20").colors  
    for i, solution in enumerate(solutions):
        path_x, path_y = [], []
        path_x.append(node_positions[next(iter(g.nodes))][0]) 
        path_y.append(node_positions[next(iter(g.nodes))][1])
        for node_index in solution:
            node = list(g.nodes)[node_index]
            path_x.append(node_positions[node][0])
            path_y.append(node_positions[node][1])
        plt.plot(path_x, path_y, color=colors[i % len(colors)], label=f"Solution {i+1}")

    plt.legend(loc="best")
    plt.axis("off")
    plt.show()  # Display the complete graph
    print(t)
    plt.savefig('outputs/images/' + t + '.png')


def create_vrp_problem(dataset_file):
    parsed_data = parse_file(dataset_file)
    capacity = int(parsed_data["capacity"])
    numOfVehicles = int(parsed_data["vehicleNumber"])
    demands = parsed_data["demands"]
    time_intervals = parsed_data["time_interval"]
    weights = list(demands.values())
    weights = np.array((weights), dtype=int)
    capacities = [capacity] * numOfVehicles
    service = int(parsed_data["service"])   
    services = [service]
    dests = list(demands.keys())
    dests = [int(dest) for dest in dests]
    sources = [int(dests.pop(0))]
    node_coords = parsed_data["node_coords"] 

    g = nx.Graph()
    for node_id, (x, y) in parsed_data["node_coords"].items():
        g.add_node(node_id, pos=(x, y))

    num_nodes = len(g.nodes)
    costs = np.zeros((num_nodes, num_nodes), dtype=float)

    node_list = list(g.nodes)
    for i, node1 in enumerate(node_list):
        for j, node2 in enumerate(node_list):
            if node1 != node2:
                dist = math.sqrt((g.nodes[node1]["pos"][0] - g.nodes[node2]["pos"][0])**2 +
                                 (g.nodes[node1]["pos"][1] - g.nodes[node2]["pos"][1])**2)
                costs[i][j] = dist

    costs = np.round(costs, 1)
    print("Sources:\n", sources)
    print("Cost Matrix:\n", costs)
    print("Capacities:\n", capacities)
    print("Destination nodes:\n", dests)
    print("Weights:\n", weights)
    print("Time Intervals:\n", time_intervals)

    return VRPProblem(sources, costs, capacities, dests, weights, time_intervals, services, node_coords), g


#with time
def create_vrp_problem_time(dataset_file):
    parsed_data = parse_file_time(dataset_file)
    capacity = int(parsed_data["capacity"])
    numOfVehicles = int(parsed_data["vehicleNumber"])
    demands = parsed_data["demands"]
    time_intervals = parsed_data["time_interval"]
    weights = list(demands.values())
    weights = np.array((weights), dtype=int)
    capacities = [capacity] * numOfVehicles
    service = int(parsed_data["service"])   
    services = [service]
    dests = list(demands.keys())
    dests = [int(dest) for dest in dests]
    sources = [int(dests.pop(0))]
    node_coords = parsed_data["node_coords"] 

    g = nx.Graph()
    for node_id, (x, y) in parsed_data["node_coords"].items():
        g.add_node(node_id, pos=(x, y))

    num_nodes = len(g.nodes)
    costs = np.zeros((num_nodes, num_nodes), dtype=float)

    node_list = list(g.nodes)
    for i, node1 in enumerate(node_list):
        for j, node2 in enumerate(node_list):
            if node1 != node2:
                dist = math.sqrt((g.nodes[node1]["pos"][0] - g.nodes[node2]["pos"][0])**2 +
                                 (g.nodes[node1]["pos"][1] - g.nodes[node2]["pos"][1])**2)
                costs[i][j] = dist

    costs = np.round(costs, 1)
    print("Sources:\n", sources)
    print("Cost Matrix:\n", costs)
    print("Capacities:\n", capacities)
    print("Destination nodes:\n", dests)
    print("Weights:\n", weights)
    print("Services:\n", services)
    print("Time Intervals:\n", time_intervals)
    # print("TIME TEST: ", time_intervals['-1'][0])
    # print(f'cost0,1 {costs[0][1]}, cost1,2 {costs[1][2]}, cost2,4 {costs[2][4]}, cost4,3 {costs[4][3]}, cost3,5 {costs[3][5]}, cost5,0 {costs[5][0]}, ')



    return VRPProblem(sources, costs, capacities, dests, weights, time_intervals, services, node_coords), g

# # Example usage:
# problem, graph = create_vrp_problem(r"C:\Users\darre\Desktop\Quantum Research\DBCW\D-Wave-VRP-master_Time\D-Wave-VRP-master\tests\CMT\CMT1.vrp")
# print(problem)
# problem, graph = create_vrp_problem_time(r"C:\Users\darre\Desktop\Quantum Research\DBCW\D-Wave-VRP-master_Time\D-Wave-VRP-master\tests\CMT\C101.vrp")
# print(problem)