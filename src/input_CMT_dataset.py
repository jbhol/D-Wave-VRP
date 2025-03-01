import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import product
from vrp_problem import VRPProblem

def parse_pfile(file_path):
    import re
    data = {}
    current_section = None

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            if line.startswith("NAME"):
                data["name"] = line.split(" : ")[1].strip()
            elif line.startswith("COMMENT"):
                data["comment"] = line.split(" : ")[1].strip()
                match = re.search(r"No of trucks: (\d+)", line)
                if match:
                    data["vehicleNumber"] = int(match.group(1))
            elif line.startswith("CAPACITY"):
                data["capacity"] = line.split(" : ")[1]
            elif line.startswith("NODE_COORD_SECTION"):
                current_section = "coord_section"
                data["node_coords"] = {}
            elif line.startswith("DEMAND_SECTION"):
                current_section = "demand_section"
                data['demands'] = {}
            elif line.startswith("DEPOT_SECTION"):
                current_section = "depot_section"
                data['depot'] = []
            elif line == "EOF":
                break

            elif current_section == "coord_section" and line:
                try:
                    cust_no, x_coord, y_coord = line.split()
                    cust_no = str(int(cust_no) - 1)
                    data["node_coords"][cust_no] = (float(x_coord), float(y_coord))
                except ValueError:
                    print(f"Error processing node coordinate line: {line}")

            elif current_section == "demand_section" and line:
                try:
                    cust_no, demand = map(int, line.split())
                    cust_no = str(int(cust_no) - 1)
                    weight = demand % 2.5
                    if (weight == 0):
                      weight = 1.5      
                    if (cust_no == "0"):
                      weight = 0             
                    data["demands"][cust_no] = weight
                except ValueError:
                    print(f"Error processing demand line: {line}")

            elif current_section == "depot_section" and line:
                if line != "-1":
                    try:
                        depot_no = int(line) - 1
                        data["depot"].append(depot_no)
                    except ValueError:
                        print(f"Error processing depot line: {line}")

    return data


def create_pvrp_problem(dataset_file):
    parsed_data = parse_pfile(dataset_file)
    capacity = float(parsed_data["capacity"])
    numOfVehicles = int(parsed_data["vehicleNumber"])
    demands = parsed_data["demands"]
    weights = list(demands.values())
    weights = np.array(weights, dtype=float)

    # Destination nodes should exclude the depot (node 0)
    dests = [int(node) for node in demands.keys() if int(node) != 0]

    capacities = [capacity] * numOfVehicles
    sources = [0]  # Depot is always node 0
    node_coords = parsed_data["node_coords"]

    g = nx.Graph()
    for node_id, (x, y) in parsed_data["node_coords"].items():
        if x >= 10:
          x = x // 10
        if y >= 10:
          y = y // 10
        g.add_node(node_id, pos=(x, y))

    num_nodes = len(g.nodes)
    costs = np.zeros((num_nodes, num_nodes), dtype=float)

    node_list = list(g.nodes)
    for i, node1 in enumerate(node_list):
        for j, node2 in enumerate(node_list):
            if node1 != node2:
                dist = math.sqrt((g.nodes[node1]["pos"][0] - g.nodes[node2]["pos"][0])**2 +
                                 (g.nodes[node1]["pos"][1] - g.nodes[node2]["pos"][1])**2)
                costs[i][j] = dist #* 0.1

    costs = np.round(costs, 1)
    print("Sources:\n", sources)
    print("Cost Matrix:\n", costs)
    print("Capacities:\n", capacities)
    print("Destination nodes:\n", dests)
    print("Weights:\n", weights)

    return VRPProblem(sources, costs, capacities, dests, weights, node_coords), g





def parse_file(file_path):
  """
  Parses a dataset file in the specified format and extracts the relevant information.

  Args:
    file_path: Path to the dataset file.

  Returns:
    A dictionary containing the parsed data, with keys:
      "node_coords": A dictionary of node IDs to (x, y) coordinates
      "capacity": The capacity of the vehicles
      "demands": A dictionary of node IDs to demands
      "depot": A list of depot node IDs
  """

  data = {}
  current_section = None

  with open(file_path, "r") as file:
    for line in file:
      line = line.strip()

      if line.startswith("NAME"):
        # Skip the NAME line
        continue

      elif line.startswith("DIMENSION"):
        # Get the number of nodes
        _, num_nodes = line.split(":")
        data["node_coords"] = {}  # Initialize the node coordinates dictionary

      elif line.startswith("VEHICLES"):
        # Get the number of vehicles
        _, vehicles = line.split(":")
        data["vehicles"] = int(vehicles)

      elif line.startswith("CAPACITY"):
        # Get the capacity
        _, capacity = line.split(":")
        data["capacity"] = float(capacity)

      elif line.startswith("CAPACITY"):
        # Get the capacity
        _, capacity = line.split(":")
        data["capacity"] = float(capacity)

      elif line.startswith("NODE_COORD_SECTION"):
        current_section = "node_coords"

      elif line.startswith("DEMAND_SECTION"):
        current_section = "demands"
        data["demands"] = {}

      elif line.startswith("DEPOT_SECTION"):
        current_section = "depot"
        data["depot"] = []

      elif line.startswith("EOF"):
        break

      else:
        if current_section == "node_coords":
          node_id, x, y = map(float, line.split())
          data["node_coords"][node_id] = (x, y)

        elif current_section == "demands":
          node_id, demand = map(int, line.split())
          weight = demand % 2.5
          if (weight == 0):
            weight = 1.5
          data["demands"][node_id] = weight

        elif current_section == "depot":
          depot_id = int(line)
          data["depot"].append(depot_id)
          
  node_ids = list(data["node_coords"].keys())
  new_node_coords = {}
  for i, node_id in enumerate(node_ids):
      new_node_coords[i] = data["node_coords"].pop(node_id)
      
  # Assuming "demands" and "depot" also use node IDs as keys
  new_demands = {i: data["demands"].pop(node_id) for i, node_id in enumerate(node_ids)}
  new_depot = [i for i in node_ids if i in data["depot"]]  # Assuming depot is a list of IDs

  data["node_coords"] = new_node_coords
  data["demands"] = new_demands
  data["depot"] = new_depot

  return data

def truncate_float(float_number, decimal_places):
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier


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
        plt.plot(path_x, path_y, color=colors[i % len(colors)], label=f"Route {i+1}")

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.axis("off")
    plt.show()  # Display the complete graph
    plt.savefig('outputs/images/' + t + '.png')


def create_vrp_problem(dataset_file):
    """
    Creates a VRPProblem instance from the specified dataset file.  
    Args:
      dataset_file: Path to the dataset file in the provided format.    
    Returns:
      A VRPProblem instance representing the problem in the dataset.
    """ 
    parsed_data = parse_file(dataset_file)  
    # Extract problem information
    capacities = np.full([parsed_data["vehicles"]], [parsed_data["capacity"]])
    demands = parsed_data["demands"]    
    weights = list(demands.values())
    #weights = weights[1:]
    dests = list(demands.keys())
    sources = [dests.pop(0)]
    
    # Create the graph from node coordinates
    g = nx.Graph()
    for node_id, (x, y) in parsed_data["node_coords"].items():
      g.add_node(node_id, pos=(x, y))   
      
    costs = np.zeros((len(g.nodes), len(g.nodes))) 

    for node1, node2 in product(g.nodes, g.nodes):
        if node1 != node2:
            dist = math.sqrt((g.nodes[node1]["pos"][0] - g.nodes[node2]["pos"][0])**2 +
                           (g.nodes[node1]["pos"][1] - g.nodes[node2]["pos"][1])**2)
            costs[node1][node2] = truncate_float(dist, 2)  

    print("Sources:\n", sources)
    print("Cost Matrix:\n", costs)
    print("Capacities:\n", capacities)
    print("Destination nodes:\n", dests)
    print("Weights:\n", weights)
    return VRPProblem(sources, costs, capacities, dests, weights), g
  
# Example usage:
# problem = create_vrp_problem("CMT1.vrp")
# print(problem)
