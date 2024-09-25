from math import sqrt
import time
import random
from qubo_helper import Qubo
from vrp_problem import VRPProblem
from vrp_solution import VRPSolution
from itertools import product
import DWaveSolvers
import QiskitSolvers
import networkx as nx
import numpy as np
from queue import Queue
import itertools
import copy
from collections import deque
from collections import defaultdict

# Abstract class for VRP solvers.
class VRPSolver:
    # Attributes : VRPProblem
    def __init__(self, problem):
        self.problem = problem

    def set_problem(self, problem):
        self.problem = problem

    # only_one_const - const in qubo making solution correct
    # order_const - multiplier for costs in qubo
    # It is recommended to set order_const = 1 and only_one_const
    # big enough to make solutions correct. Bigger than sum of all
    # costs should be enough.
    def solve(self, only_one_const, order_const, solver_type = 'cpu'):
        pass

# Solver solves VRP only by QUBO formulation.
class FullQuboSolver(VRPSolver):
    def solve(self, only_one_const, order_const, solver_type = 'cpu'):
        qubo = self.problem.get_full_qubo(only_one_const, order_const)
        sample = DWaveSolvers.solve_qubo(qubo, solver_type = solver_type)
        solution = VRPSolution(self.problem, sample)
        return solution

# Solver assumes that every vehicle serves approximately the same number of deliveries.
# Additional attribute : limit_radius - maximum difference between served number of deliveries
# and average number of deliveries that every vehicle should serve.
class AveragePartitionSolver(VRPSolver):
    def __init__(self, problem, limit_radius = 1):
        self.problem = problem
        self.limit_radius = limit_radius

    def solve(self, only_one_const, order_const, solver_type = 'cpu'):
        dests = len(self.problem.dests)
        vehicles = len(self.problem.capacities)

        avg = int(dests / vehicles)
        limit_radius = self.limit_radius

        limits = [(max(avg - limit_radius, 0), min(avg + limit_radius, dests)) for _ in range(vehicles)]
        max_limits = [r for (_, r) in limits]

        vrp_qubo = self.problem.get_qubo_with_both_limits(limits,
                only_one_const, order_const)

        sample = DWaveSolvers.solve_qubo(vrp_qubo, solver_type = solver_type)

        solution = VRPSolution(self.problem, sample, max_limits)
        return solution


class LocalSearchSolver(VRPSolver):
    def __init__(self, problem):
        self.problem = problem

    def init_solution(self):
        dests = self.problem.dests
        capacities = self.problem.capacities
        vehicles = len(self.problem.capacities)
        costs = self.problem.costs
        weights = self.problem.weights

        routes = [[0] for _ in range(vehicles)]

        visited = [False for _ in range(len(dests) + 1)]

        for vehicle_id, capacity in enumerate(capacities):
            cur_node = 0
            cur_weight = weights[0]

            flag = True
            while flag:
                flag = False
                best_node = None
                for node in dests:
                    if visited[node]:
                        continue
                    if cur_weight + weights[node] > capacity:
                        continue
                    if best_node is None or costs[cur_node, best_node] < costs[cur_node, node]:
                        best_node = node
                        flag = True
                if best_node is not None:
                    routes[vehicle_id].append(best_node)
                    visited[best_node] = True
                    cur_weight += weights[best_node]
                cur_node = best_node

            routes[vehicle_id].append(0)

        self.cur_solution = routes

    def compute_cost(self, routes):
        costs = self.problem.costs
        cost = 0
        for route in routes:
            for i in range(len(route)-1):
                u, v = route[i], route[i+1]
                cost += costs[u, v]

        return cost

    def compute_weights(self, routes):
        weights = self.problem.weights
        route_weights = [0 for _ in self.problem.capacities]

        for vehicle_id, route in enumerate(routes):
            for node in route:
                route_weights[vehicle_id] += weights[node]
        
        return route_weights

    def resequence_one_node(self, resequence_node, only_one_const, solver_type):
        costs = self.problem.costs
        capacities = self.problem.capacities
        weights = self.problem.weights

        cur_routes = copy.deepcopy(self.cur_solution)
        cur_cost = self.compute_cost(cur_routes)
        cur_route_weights = self.compute_weights(cur_routes)

        # find the node in the routes
        tmp_routes = None
        for vehicle_id, route in enumerate(cur_routes):
            for i, node in enumerate(route):
                if node == resequence_node:
                    prev_node, next_node = route[i-1], route[i+1]
                    tmp_routes = copy.deepcopy(cur_routes)
                    tmp_routes[vehicle_id].pop(i)
                    removed_cost = costs[prev_node, node] + costs[node, next_node] - costs[prev_node, next_node]
                    cur_route_weights[vehicle_id] -= weights[node]
        
        # find a better solution
        best_routes = copy.deepcopy(cur_routes)
        best_cost = cur_cost
        for vehicle_id, route in enumerate(tmp_routes):
            if cur_route_weights[vehicle_id] + weights[resequence_node] > capacities[vehicle_id]:
                continue

            # TODO: 
            qubo = Qubo()

            edge_ids = []
            costs_dict = {}
            for i in range(len(route)-1):
                prev_node, next_node = route[i], route[i+1]
                inserted_cost = costs[prev_node, resequence_node] + costs[resequence_node, next_node] - costs[prev_node, next_node]
                
                index = ((i, prev_node), (i+1, next_node))
                costs_dict[index] = inserted_cost

                qubo.add((index, index), inserted_cost)
                edge_ids.append(index)

            qubo.add_only_one_constraint(edge_ids, only_one_const)

            # sample = DWaveSolvers.solve_qubo(qubo, solver_type=solver_type)
            sample = QiskitSolvers.solve_qubo(qubo, solver_type=solver_type)

            flag = False

            '''
            for i in range(len(route)-1):
                prev_node, next_node = route[i], route[i+1]
                inserted_cost = costs[prev_node, resequence_node] + costs[resequence_node, next_node] - costs[prev_node, next_node]
                if best_cost > cur_cost - removed_cost + inserted_cost:
                    best_cost = cur_cost - removed_cost + inserted_cost
                    best_routes = copy.deepcopy(tmp_routes)
                    best_routes[vehicle_id].insert(i+1, resequence_node)

                    flag = True
            # '''

            # '''
            for index, value in sample.items():
                if value == 1:
                    (prev_id, prev_node), (next_id, next_node) = index
                    inserted_cost = costs[prev_node, resequence_node] + costs[resequence_node, next_node] - costs[prev_node, next_node]
                    if best_cost > cur_cost - removed_cost + inserted_cost:
                        best_cost = cur_cost - removed_cost + inserted_cost
                        best_routes = copy.deepcopy(tmp_routes)
                        best_routes[vehicle_id].insert(next_id, resequence_node)
            # '''

            if flag:
                print(vehicle_id)
                print(sample)
                print(resequence_node)
                print(best_routes)
                print(costs_dict)

        return best_routes, best_cost

    def solve(self, only_one_const, order_const, solver_type = 'cpu'):
        dests = self.problem.dests
        capacities = self.problem.capacities
        vehicles = len(self.problem.capacities)
        costs = self.problem.costs
        weights = self.problem.weights

        self.init_solution()

        is_optimal = False
        best_cost = None
        while not is_optimal:
            t_now = time.time()
            nodes_list = [d for d in dests]
            np.random.shuffle(nodes_list)

            local_best_cost = None
            local_best_routes = None

            for node in nodes_list:
                new_routes, cost = self.resequence_one_node(node, only_one_const, solver_type)
                if local_best_cost is None or local_best_cost > cost:
                    local_best_cost = cost
                    local_best_routes = copy.deepcopy(new_routes)
       
            if best_cost is None or best_cost > local_best_cost:
                best_cost = local_best_cost
                self.cur_solution = copy.deepcopy(local_best_routes)
            else:
                is_optimal = True
            print('Current best cost:', best_cost)
            print('Time:', time.time() - t_now)
      
        # clean the route
        for route in self.cur_solution:
            if len(route) == 2:
                route.clear()

        solution = VRPSolution(self.problem, solution=self.cur_solution)
        return solution


# Solver uses DBScan to divide problem into subproblems that can be solved effectively by FullQuboSolver.
# Attributes : max_len - maximum number of deliveries in problems solved by FullQuboSolver.
# anti_noiser : True if dbscan should eliminate singleton clusters, False otherwise.
class DBScanSolver(VRPSolver):

    def __init__(self, problem, max_len = 10, anti_noiser = True):
        self.problem = problem
        self.anti_noiser = anti_noiser
        self.max_len = max_len
        self.max_weight = max(problem.capacities)
        self.max_dist = 2 * max(map(max, problem.costs))

    # Returns subset of dests with elements x that satisfies
    # costs[source][x] + costs[x][source] <= 2 * radius
    def _range_query(self, dests, costs, source, radius):
        result = list()
        for dest in dests:
            if (costs[source][dest] + costs[dest][source]) / 2 <= radius:
                result.append(dest)
        return result

    # Standard dbscan clustering dests.
    # Returns list of clusters.
    def _dbscan(self, dests, costs, radius, min_size):
        clusters_num = -1

        states = dict()
        # Undifined cluster.
        for d in dests:
            states[d] = -2

        for d in dests:
            neighbours = self._range_query(dests, costs, d, radius)
            if len(neighbours) < min_size:
                states[d] = -1

        for dest in dests:
            if states[dest] != -2:
                continue

            clusters_num += 1
            q = Queue()
            q.put(dest)

            while not q.empty():
                dest2 = q.get()
                states[dest2] = clusters_num
                neighbours = self._range_query(dests, costs, dest2, radius)
                for v in neighbours:
                    if states[v] == -2:
                        q.put(v)

        for dest in dests: 
            if states[dest] == -1:
                min_dist = self.max_dist
                best_neighbour = -1
                for d in dests:
                    if states[d] != -1:
                        if costs[d][dest] < min_dist:
                            best_neighbour = d
                            min_dist = costs[d][dest]
                if best_neighbour == -1:
                    clusters_num += 1
                    states[dest] = clusters_num
                else:
                    states[dest] = states[best_neighbour]

        clusters = list()
        for i in range(clusters_num + 1):
            clusters.append(list())
        for dest in dests:
            cl = states[dest]
            clusters[cl].append(dest)

        return clusters

    # Recursive dbscan. Returns list of clusters.
    # dests - set that need to be clustered.
    # costs - array with costs between dests.
    # min_radius, max_radius - lower and upper bound for radius parameter
    # in dbscan.
    # clusters_num - expected maximum number of clusters. It is not guaranteed that 
    # function won't return more clusters.
    # max_len - maximum size of a cluster. It is guaranteed that every cluster will
    # have at most max_len elements.
    # max_weight - maximum sum of deliveries' weights of a cluster. It is guaranteed that every cluster will
    # have at most max_weight sum of weights.
    def _recursive_dbscan(self, dests, costs, min_radius, max_radius,
                          clusters_num, max_len, max_weight):
        best_res = [[d] for d in dests]

        min_r = min_radius
        max_r = max_radius
        curr_r = max_r

        # Searching best radius with binary search.
        while min_r + 1 < max_r:
            curr_r = (min_r + max_r) / 2

            clusters = self._dbscan(dests, costs, curr_r, 1)

            if len(clusters) < clusters_num:
                max_r = curr_r
            else:
                min_r = curr_r
                if len(clusters) < len(best_res):
                    best_res = clusters

        # Recursive dbscan on clusters with too many elements. 
        for cluster in best_res:
            weight = 0
            for dest in cluster:
                weight += self.problem.weights[dest]
            if len(cluster) > max_len or weight > max_weight:
                best_res.remove(cluster)
                best_res += self._recursive_dbscan(cluster, costs, 0., self.max_dist, 2,
                                                   max_len, max_weight)

        # Removing singleton clusters while they are and there is more than clusters_num clusters.
        if self.anti_noiser:
            while len(best_res) > clusters_num:
                singleton = [0]
                for cluster in best_res:
                    if len(cluster) == 1:
                        singleton = cluster
                        break

                if singleton == [0]:
                    break

                best_res.remove(singleton)

                one = singleton[0]
                best_cluster = []
                best_dist = self.max_dist

                for cluster in best_res:
                    if len(cluster) == max_len or cluster == singleton:
                        continue

                    weight = 0
                    min_dist = self.max_dist

                    for dest in cluster:
                        weight += self.problem.weights[dest]
                        min_dist = min(min_dist, costs[dest][one])
                    if weight + self.problem.weights[one] <= max_weight:
                        if best_dist > min_dist:
                            best_dist = min_dist
                            best_cluster = cluster

                if best_cluster == []:
                    best_res.append(singleton)
                    break
                best_res.remove(best_cluster)
                best_res.append(best_cluster + singleton)

        return best_res

    def solve(self, only_one_const, order_const, solver_type = 'cpu'):
        problem = self.problem
        dests = problem.dests
        costs = problem.costs
        sources = [problem.source]
        capacities = problem.capacities
        weights = problem.weights
        vehicles = len(problem.capacities)

        if len(dests) == 0:
            return VRPSolution(problem, None, None, [[]])

        clusters = self._recursive_dbscan(dests, costs, 0., self.max_dist, vehicles,
                self.max_len, self.max_weight)

        # If we have as much small clusters as vehicles, we can solve TSP for every cluster.
        if len(clusters) == vehicles:
            result = list()
            for cluster in clusters:
                new_problem = VRPProblem(sources, costs, [capacities[0]], cluster, weights)
                solver = FullQuboSolver(new_problem)
                solution = solver.solve(only_one_const, order_const,
                                    solver_type = solver_type).solution[0]
                result.append(solution)
            return VRPSolution(problem, None, None, result)

        solutions = list()
        solutions.append(VRPSolution(problem, None, None, [[0]]))

        # Solving TSP for every cluster.
        for cluster in clusters:
            new_problem = VRPProblem(sources, costs, [capacities[0]], cluster, weights,
                                 first_source = False, last_source = False)
            solver = FullQuboSolver(new_problem)
            solution = solver.solve(only_one_const, order_const, solver_type = solver_type)
            solutions.append(solution)

        # Creating smaller instance of problem for DBScanSolver.
        clusters_num = len(clusters) + 1
        new_dests = [i for i in range(1, clusters_num)]
        new_costs = np.zeros((clusters_num, clusters_num), dtype=float)
        new_weights = np.zeros((clusters_num), dtype=int)

        for (i, j) in product(range(clusters_num), range(clusters_num)):
            if i == j:
                new_costs[i][j] = 0
                continue
            id1 = solutions[i].solution[0][-1]
            id2 = solutions[j].solution[0][0]
            new_costs[i][j] = costs[id1][id2]

        for i in range(clusters_num):
            for dest in solutions[i].solution[0]:
                new_weights[i] += weights[dest]

        new_problem = VRPProblem(sources, new_costs, capacities, new_dests, new_weights)
        solver = DBScanSolver(new_problem)
        compressed_solution = solver.solve(only_one_const, order_const, 
                            solver_type = solver_type).solution

        # Achieving full solution from solution of smaller version.
        uncompressed_solution = list()
        for vehicle_dests in compressed_solution:
            uncompressed = list()
            for dest in vehicle_dests:
                uncompressed += solutions[dest].solution[0]
            uncompressed_solution.append(uncompressed)

        return VRPSolution(problem, None, None, uncompressed_solution)

class Tabu_Move:
    def __init__(self, n, move1, location1, move2 = 0, location2 = 0):
        self.location1 = location1
        self.move1 = move1
        self.location2 = location2
        self.move2 = move2
        #self.count = random.randint(5,10)
        self.count = random.randint(0.4*n,0.6*n)# + 5

class Neighbor:
    def __init__(self, clusters, move1, location1, move2 = 0, location2 = 0):
        self.location1 = location1
        self.move1 = move1
        self.location2 = location2
        self.move2 = move2
        self.clusters = clusters
        if location1 == location2:        
            self.type = "0,1"
        elif move2 == 0:
            self.type = "1,0"
        else:
            self.type = "1,1"

class TabuSolver(VRPSolver):
    def check_elements_match(array1, array2):
        if len(array1) != len(array2):
            return False
        for element in array1:
            if element not in array2:
                return False
        return True 

    def __init__(self, problem, max_len = 10, anti_noiser = True):
        self.problem = problem
        self.anti_noiser = anti_noiser
        self.max_len = max_len
        self.max_weight = max(problem.capacities)
        self.max_dist = sum(map(sum, problem.costs))

    def solve(self, only_one_const, order_const, solver_type = 'cpu'):
        problem = self.problem
        dests = problem.dests
        N = len(dests)
        costs = problem.costs
        sources = [problem.source]
        capacities = problem.capacities
        weights = problem.weights
        vehicles = len(problem.capacities)

        neighborhood = [[] for _ in range(len(weights))]
        for d in dests:
            indices = np.argpartition(costs[d][:], vehicles*2)[:vehicles*2]
            neighborhood[d] = indices

        sorted_dests = sorted(dests, reverse=True , key=lambda i: costs[problem.in_nearest_sources[i]][i]) #costs[0][i]
        sorted_dests = [item for item in sorted_dests if item in dests]

        # 1. build initial solution
        clusters = list()
        for i in range(vehicles):
            clusters.append(list())

        # 2. seed clusters with farthest non-neighbor locations
        seeded = False
        a = 0
        while seeded == False:
            d = sorted_dests[a]
            for i in range(vehicles):
                if any(x in clusters[i] for x in neighborhood[d]):
                    a += 1
                    break
                elif len(clusters[i]) > 0:
                    continue
                else:
                    clusters[i].append(d)
                    sorted_dests.pop(a)
                    break    
            for i in range(vehicles):
                seeded = True
                if len(clusters[i]) == 0:
                    seeded = False

        # 3. fill vehicles with remaining locations, try to keep neighbors together                    
        a = 0
        while len(sorted_dests) > 0:
            d = sorted_dests[a]
            found = False
            for i in range(vehicles):
                if any(x in clusters[i] for x in neighborhood[d]):
                    #check if clusters[j] has room for d
                    weight = 0
                    for dest in clusters[i]:
                        weight += self.problem.weights[dest]  #cluster current weight     
                    if weight + self.problem.weights[d] <= capacities[i]:   
                        clusters[i].append(d)
                        sorted_dests.pop(a)
                        found = True
                        break                        
            if found == False:
                a += 1
            if a == len(sorted_dests) and a > 0:
                a = 0
                d = sorted_dests[a]
                for i in range(vehicles):
                    weight = 0
                    for dest in clusters[i]:
                        weight += self.problem.weights[dest]  #cluster current weight   
                    if weight + self.problem.weights[d] <= capacities[i]:   
                        clusters[i].append(d)
                        sorted_dests.pop(a)
                        found = True
                        break    
        
        # 4. Calculate starting solution cost
        tabu = []   #dest, cluster_num
        neighbors = [] #dest, cluster_num, clusters
        best_solution = clusters
        first_sol = VRPSolution(problem, None, None, clusters)
        check_sol = copy.deepcopy(first_sol)
        # Adding first and last magazine.
        for l in check_sol.solution:
            if len(l) != 0:
                if problem.first_source:
                    l.insert(0, problem.in_nearest_sources[l[0]])
                if problem.last_source:
                    l.append(problem.out_nearest_sources[l[len(l) - 1]])    
        best_cost = check_sol.total_cost()

        optimized_routes = list()
        counter_of_last_threshold = 0
        last_threshold = random.randint(int(0.6 * N), int(1.1 * N))
        counter_of_last_best = 0
        intensification_counter = 2
        diversification = True
        diversification_counter = 0
        counter = 0
        ready_to_stop = False
        largest_change = 0
        frequency = defaultdict(int)

        neighborhood = [[] for _ in range(len(weights))]
        for d in dests:
            indices = np.argpartition(costs[d][:], int(vehicles * 2))[:int(vehicles * 2)]
            neighborhood[d] = indices

        # 5. while not ready to stop
        while ready_to_stop is False:
            feasible = True
            infeasible_amount = 0
            neighbors = []
            inf_neighbors = []

            # 6. pre-calc cluster weights
            vehicle_weights = list()
            for i in range(vehicles):
                vehicle_weights.append(0)
                for dest in clusters[i]:
                    vehicle_weights[i] += self.problem.weights[dest]
                if vehicle_weights[i] > capacities[i]:
                    feasible = False
                    infeasible_amount += vehicle_weights[i] - capacities[i]

            # 7. create candidate list of neighbors to current solution
            # 8. 0,1 
            if diversification == False:                
                for i in range(vehicles):
                    used = []
                    for idxd, d in enumerate(clusters[i]):
                        used.append(d)
                        for idxe, e in enumerate(clusters[i]):
                            if d != e and e not in used:
                                new_neighbor = copy.deepcopy(clusters)
                                swap1 = new_neighbor[i][idxd] 
                                swap2 = new_neighbor[i][idxe]
                                new_neighbor[i][idxd] = swap2
                                new_neighbor[i][idxe] = swap1
                                n = Neighbor(new_neighbor, swap1, i, swap2, i)
                                if vehicle_weights[i] <= capacities[i]:
                                    neighbors.append(n)
                                else:
                                    inf_neighbors.append(n)

            # 9. 1,1
            if True == True:                 
                swap1 = -1
                swap2 = -1
                choices = [index for index, sublist in enumerate(clusters) if len(sublist) > 0]
                for i in choices:
                    choices.remove(i)
                    for idx_i, swap1 in enumerate(clusters[i]):
                        for j in choices:
                            for idx_j, swap2 in enumerate(clusters[j]):                                
                                weight1 = vehicle_weights[j] - self.problem.weights[swap2]
                                weight2 = vehicle_weights[i] - self.problem.weights[swap1]
                                #if weight1 + self.problem.weights[swap1] <= capacities[j] and set(neighborhood[swap1]).intersection(clusters[j]): #any(x in clusters[j] for x in neighborhood[swap1]):
                                if set(neighborhood[swap1]).intersection(clusters[j]): #any(x in clusters[j] for x in neighborhood[swap1]):                                    
                                    #if weight2 + self.problem.weights[swap2] <= capacities[i] and set(neighborhood[swap2]).intersection(clusters[i]): #any(x in clusters[i] for x in neighborhood[swap2]):                                    
                                    if set(neighborhood[swap2]).intersection(clusters[i]): #any(x in clusters[i] for x in neighborhood[swap2]):
                                        #swap works
                                        new_neighbor = copy.deepcopy(clusters)
                                        new_neighbor[j][idx_j] = swap1
                                        new_neighbor[i][idx_i] = swap2
                                        n = Neighbor(new_neighbor, swap1, i, swap2, j)
                                        if weight1 + self.problem.weights[swap1] <= capacities[j] and weight2 + self.problem.weights[swap2] <= capacities[i]:
                                            neighbors.append(n)
                                        else:
                                            inf_neighbors.append(n)

            # 10. 1,0
            if False == False:                                                
                for i in range(vehicles):
                    for d in clusters[i]:
                        for j in range(vehicles):
                            if d not in clusters[j]:
                                #check if clusters[j] has room for d                        
                                #if vehicle_weights[j] + self.problem.weights[d] <= capacities[j] and set(neighborhood[d]).intersection(clusters[j]): #any(x in clusters[i] for x in neighborhood[d]):
                                if set(neighborhood[d]).intersection(clusters[j]): #any(x in clusters[i] for x in neighborhood[d]):                                    
                                    #there is room so move d to clusters[j]
                                    new_neighbor = copy.deepcopy(clusters)
                                    #insert the new location into the route where it adds the least to the cost
                                    new_neighbor_cost = self.max_dist
                                    new_neighbor_spot = -1
                                    options = [new_neighbor[j][x:] + [d] + new_neighbor[j][:x] for x in range(len(new_neighbor[j]),-1,-1)]
                                    for k, option in enumerate(options):
                                        cost = 0
                                        prev = option[0]
                                        cost += costs[sources[0]][prev]
                                        for dest in option[1:]:
                                            cost += costs[prev][dest]
                                            prev = dest
                                        cost += costs[prev][sources[0]]
                                        if cost < new_neighbor_cost:
                                            new_neighbor_cost = cost
                                            new_neighbor_spot = k
                                    if new_neighbor_spot != -1:
                                        new_neighbor[i].remove(d)
                                        new_neighbor[j] = list(options[new_neighbor_spot])
                                        n = Neighbor(new_neighbor, d, i)
                                        if vehicle_weights[j] + self.problem.weights[d] <= capacities[j]:
                                            neighbors.append(n)
                                        else:
                                            inf_neighbors.append(n)

            current_best_neighbor = []
            current_best_cost = self.max_dist
            current_best_move = ""
            selected_neighbor = []
            selected_neighbor_cost = self.max_dist
            selected_inf_neighbor = []
            selected_inf_neighbor_cost = self.max_dist

            # 11. previous solution was feasible
            if feasible == True:
                #find best feasible candidate                
                for n in neighbors:
                    #calcluate neighbor cost            
                    sol = VRPSolution(problem, None, None, n.clusters)
                    check_sol = copy.deepcopy(sol)
                    # Adding first and last magazine.
                    for l in check_sol.solution:
                        if len(l) != 0:
                            if problem.first_source:
                                l.insert(0, problem.in_nearest_sources[l[0]])
                            if problem.last_source:
                                l.append(problem.out_nearest_sources[l[len(l) - 1]])    
                    cost = check_sol.total_cost()
                    if cost < selected_neighbor_cost:
                        #keep track of overall best neighbor
                        if cost < current_best_cost:
                            current_best_neighbor = n
                            current_best_cost = cost
                            current_best_move = n.type
                        #check if candidate is tabu
                        is_tabu = False
                        for move in tabu:
                            if move.move1 == n.move1 and move.location1 == n.location1 and move.move2 == n.move2 and move.location2 == n.location2:
                                #candidate is tabu
                                is_tabu = True
                                break
                        if is_tabu is False:
                            #keep track of best non-tabu neighbor
                            selected_neighbor = n
                            selected_neighbor_cost = cost

                #find best infeasible candidate                                         
                for n in inf_neighbors:
                    #calcluate neighbor cost            
                    sol = VRPSolution(problem, None, None, n.clusters)
                    check_sol = copy.deepcopy(sol)
                    # Adding first and last magazine.
                    for l in check_sol.solution:
                        if len(l) != 0:
                            if problem.first_source:
                                l.insert(0, problem.in_nearest_sources[l[0]])
                            if problem.last_source:
                                l.append(problem.out_nearest_sources[l[len(l) - 1]])    
                    cost = check_sol.total_cost()
                    if cost < selected_inf_neighbor_cost:
                        #check if candidate is tabu
                        is_tabu = False
                        for move in tabu:
                            if move.move1 == n.move1 and move.location1 == n.location1 and move.move2 == n.move2 and move.location2 == n.location2:
                                #candidate is tabu
                                is_tabu = True
                                break
                        if is_tabu is False:
                            #keep track of best non-tabu neighbor
                            selected_inf_neighbor = n
                            selected_inf_neighbor_cost = cost

                #pick the best neighbor
                #if selected_neighbor_cost > previous_cost:
                if selected_inf_neighbor_cost < selected_neighbor_cost:
                    selected_neighbor = selected_inf_neighbor 
                    selected_neighbor_cost = selected_inf_neighbor_cost 

            # 12. previous solution was NOT feasible                  
            else:
                #find best feasible candidate
                current_best_cost = self.max_dist
                best_amount = sum(capacities)
                best_inf_amount = sum(capacities)
                for n in neighbors:
                    current_infeasible_amount = 0
                    current_weights = list()
                    for i in range(vehicles):
                        current_weights.append(0)
                        for dest in n.clusters[i]:
                            current_weights[i] += self.problem.weights[dest]
                        if current_weights[i] > capacities[i]:
                            current_infeasible_amount += current_weights[i] - capacities[i]
                    if current_infeasible_amount <= best_amount:
                        #calcluate neighbor cost            
                        sol = VRPSolution(problem, None, None, n.clusters)
                        check_sol = copy.deepcopy(sol)
                        # Adding first and last magazine.
                        for l in check_sol.solution:
                            if len(l) != 0:
                                if problem.first_source:
                                    l.insert(0, problem.in_nearest_sources[l[0]])
                                if problem.last_source:
                                    l.append(problem.out_nearest_sources[l[len(l) - 1]])
                        cost = check_sol.total_cost()
                        if cost < current_best_cost:
                            current_best_neighbor = n
                            current_best_cost = cost
                            current_best_move = n.type
                        #if cost < selected_neighbor_cost:
                        #check if candidate is tabu
                        is_tabu = False
                        for move in tabu:
                            if move.move1 == n.move1 and move.location1 == n.location1 and move.move2 == n.move2 and move.location2 == n.location2:
                                #candidate is tabu
                                is_tabu = True
                                break
                        if is_tabu is False:
                            #keep track of best non-tabu neighbor
                            selected_neighbor = n
                            selected_neighbor_cost = cost
                            best_amount = current_infeasible_amount

                #find best infeasible candidate                            
                for n in inf_neighbors:
                    inf_infeasible_amount = 0
                    current_weights = list()
                    for i in range(vehicles):
                        current_weights.append(0)
                        for dest in n.clusters[i]:
                            current_weights[i] += self.problem.weights[dest]
                        if current_weights[i] > capacities[i]:
                            inf_infeasible_amount += current_weights[i] - capacities[i]
                    if inf_infeasible_amount <= best_inf_amount:
                        #calcluate neighbor cost            
                        sol = VRPSolution(problem, None, None, n.clusters)
                        check_sol = copy.deepcopy(sol)
                        # Adding first and last magazine.
                        for l in check_sol.solution:
                            if len(l) != 0:
                                if problem.first_source:
                                    l.insert(0, problem.in_nearest_sources[l[0]])
                                if problem.last_source:
                                    l.append(problem.out_nearest_sources[l[len(l) - 1]])
                        cost = check_sol.total_cost()
                        #if cost < selected_inf_neighbor_cost:
                        #check if candidate is tabu
                        is_tabu = False
                        for move in tabu:
                            if move.move1 == n.move1 and move.location1 == n.location1 and move.move2 == n.move2 and move.location2 == n.location2:
                                #candidate is tabu
                                is_tabu = True
                                break
                        if is_tabu is False:
                            #keep track of best non-tabu neighbor
                            selected_inf_neighbor = n
                            selected_inf_neighbor_cost = cost
                            best_inf_amount = inf_infeasible_amount

                #pick the best neighbor
                if best_inf_amount < best_amount:
                    selected_neighbor = selected_inf_neighbor
                    selected_neighbor_cost = selected_inf_neighbor_cost
            
            # 14. aspiration
            aspiration = False
            if current_best_cost < best_cost:
                #make sure its feasible
                vehicle_weights = list()
                current_best_feasible = True
                for i in range(vehicles):
                    vehicle_weights.append(0)
                    for dest in clusters[i]:
                        vehicle_weights[i] += self.problem.weights[dest]
                    if vehicle_weights[i] > capacities[i]:
                        current_best_feasible = False
                #feasible, so lets use it
                if current_best_feasible == True:
                    if best_cost - current_best_cost > largest_change:
                        largest_change = best_cost - current_best_cost
                    #ignore tabu and use it anyways
                    best_cost = current_best_cost
                    print('total_cost =', best_cost, 'move=', current_best_move)
                    best_solution = copy.deepcopy(current_best_neighbor.clusters)
                    clusters = copy.deepcopy(current_best_neighbor.clusters)
                    #intensification_counter = 0
                    counter_of_last_threshold = counter
                    counter_of_last_best = counter
                    is_tabu = False
                    for move in tabu:
                        if move.move1 == current_best_neighbor.move1 and move.location1 == current_best_neighbor.location1 and move.move2 == current_best_neighbor.move2 and move.location2 == current_best_neighbor.location2:
                            #candidate is tabu
                            is_tabu = True
                            break
                    if is_tabu == False:
                        tabu.append(Tabu_Move(len(dests), current_best_neighbor.move1, current_best_neighbor.location1, current_best_neighbor.move2, current_best_neighbor.location2))
                    frequency[(current_best_neighbor.move1, current_best_neighbor.location1)] += 1
                    if current_best_neighbor.move2 != 0:
                        frequency[(current_best_neighbor.move2, current_best_neighbor.location2)] += 1
                    aspiration = True

            # 15. next solution = selected candidate
            if aspiration == False and hasattr(selected_neighbor, 'clusters'):
                clusters = copy.deepcopy(selected_neighbor.clusters)
                tabu.append(Tabu_Move(len(dests), selected_neighbor.move1, selected_neighbor.location1, selected_neighbor.move2, selected_neighbor.location2))
                frequency[(selected_neighbor.move1, selected_neighbor.location1)] += 1
                if selected_neighbor.move2 != 0:
                    frequency[(selected_neighbor.move2, selected_neighbor.location2)] += 1

            # 16. threshold is reached so we toggle diversification
            if counter - counter_of_last_threshold == last_threshold:   
                print('counter', counter, 'cbc', current_best_cost, 'snc', selected_neighbor_cost, 'move', current_best_move, "feasible", feasible)
                if intensification_counter == 2: #diversification
                    print('diversification on', counter)
                    counter_of_last_threshold = counter
                    last_threshold = random.randint(int(0.6 * N), int(1.1 * N))
                    diversification = True
                    intensification_counter = 1   
                    diversification_counter += 1
                    neighborhood = [[] for _ in range(len(weights))]
                    for d in dests:
                        indices = np.argpartition(costs[d][:], int(vehicles * 2))[:int(vehicles * 2)]
                        neighborhood[d] = indices
                elif intensification_counter == 1 and diversification_counter % 10 == 0: #intensification_counter == 1 and diversification_counter == 10: #intensification
                    print('intensification', counter)
                    print('div counter ', diversification_counter)
                    tabu = []   #dest, cluster_num
                    #clusters = copy.deepcopy(best_solution) 
                    counter_of_last_threshold = counter
                    if diversification == True:
                        intensification_counter = 0
                        last_threshold = random.randint(int(0.6 * N), int(1.1 * N))
                    else:
                        intensification_counter +=1
                        last_threshold = random.randint(int(0.6 * N), int(1.1 * N))
                else: #swap mode
                    print('diversification off', counter)                  
                    counter_of_last_threshold = counter
                    last_threshold = random.randint(int(0.6 * N), int(1.1 * N))
                    diversification = False
                    intensification_counter +=1
                    neighborhood = [[] for _ in range(len(weights))]
                    for d in dests:
                        indices = np.argpartition(costs[d][:], vehicles)[:vehicles]
                        neighborhood[d] = indices          

            if counter - counter_of_last_best == 2000:      
                print('Quantum Go', counter)              
                clusters = copy.deepcopy(best_solution) 
                routes = list()
                for cluster in clusters:
                    if len(cluster) > 1:
                        found = False
                        for rte in optimized_routes: #check if we have already sequenced this route
                            if TabuSolver.check_elements_match(cluster, rte):
                                route = rte
                                found = True
                        if found == False:
                            new_problem = VRPProblem(sources, costs, [capacities[0]], cluster, weights, first_source = True, last_source = True)
                            #new_problem = VRPProblem(sources, costs, [capacities[0]], cluster, weights, first_source = False, last_source = False)
                            solver = FullQuboSolver(new_problem)
                            print('0 =', cluster)
                            route = solver.solve(only_one_const, order_const, solver_type = solver_type).solution[0]                                
                            del route[0]
                            del route[-1]
                            print('1 =', route)
                            optimized_routes.append(copy.deepcopy(route))
                    else:
                        route = cluster
                    routes.append(route)
                clusters = routes
                #calcluate optimized solution cost            
                sol = VRPSolution(problem, None, None, routes)
                check_sol = copy.deepcopy(sol)
                # Adding first and last magazine.
                for l in check_sol.solution:
                    if len(l) != 0:
                        if problem.first_source:
                            l.insert(0, problem.in_nearest_sources[l[0]])
                        if problem.last_source:
                            l.append(problem.out_nearest_sources[l[len(l) - 1]])    
                cost = check_sol.total_cost()
                if cost < best_cost:
                    best_solution = copy.deepcopy(clusters)
                    best_cost = cost
                    counter_of_last_best = counter
                    print('quantum found total_cost =', best_cost)

            #update tabu list
            for move in tabu:
                move.count -= 1
                if move.count == 0:
                    tabu.remove(move)    

            #update iterator and loop back
            counter += 1
            if counter - counter_of_last_best == 5000:
                print('Best solution was found on counter =', counter_of_last_best)
                ready_to_stop = True

        # Adding first and last magazine.
        for l in best_solution:
            if len(l) != 0:
                if problem.first_source:
                    l.insert(0, problem.in_nearest_sources[l[0]])
                if problem.last_source:
                    l.append(problem.out_nearest_sources[l[len(l) - 1]])

        solution = VRPSolution(self.problem, None, None, best_solution)
        return solution



# Solver uses some solver to generate TSP Solution and tries to make VRP solution from it.
# Attributes : solver - VRPSolver object. DBScanSolver is recomended.
# random - number of permutations of vehicles that will be generate. 
class SolutionPartitioningSolver(VRPSolver):

    def __init__(self, problem, solver, random = 100):
        self.problem = problem
        self.solver = solver
        self.random = random
        self.inf = 2 * sum(map(sum, problem.costs))
    
    # Divides TSP solution to continous parts that will be correct VRP solution.
    def _divide_solution_greedy_dp(self, solution):
        problem = self.problem
        capacities = problem.capacities
        costs = problem.costs
        weights = problem.weights

        dests = len(solution)
        vehicles = len(capacities)
        div_costs = np.zeros(dests)
        for i in range(1, dests - 1):
            d1 = solution[i]
            d2 = solution[i+1]
            div_costs[i] = costs[d1][0] + costs[0][d2] - costs[d1][d2]

        dp = np.zeros((dests, vehicles + 1), dtype=float)
        prev_state = np.zeros((dests, vehicles + 1), dtype=int)

        for i in range(dests):
            if i != 0:
                dp[i][0] = self.inf
            for j in range(1, vehicles + 1):
                cap = capacities[j-1]
                pointer = i
                dp[i][j] = dp[i][j-1]
                prev_state[i][j] = i
                while pointer > 0 and cap >= weights[solution[pointer]]:
                    pointer -= 1
                    new_cost = div_costs[pointer] + dp[pointer][j-1]
                    if new_cost < dp[i][j]:
                        dp[i][j] = new_cost
                        prev_state[i][j] = pointer
                    cap -= weights[solution[pointer + 1]]

        new_solution = []
        pointer = dests - 1
        for j in reversed(range(1, vehicles + 1)):
            prev = prev_state[pointer][j]
            if prev != pointer:
                lis = solution[(prev + 1):(pointer + 1)]
                if prev != -1:
                    lis = [0] + lis
                if pointer != dests - 1:
                    lis = lis + [0]
                new_solution.append(lis)
            else:
                new_solution.append([])
            pointer = prev
        
        new_solution.reverse()
        return VRPSolution(problem, None, None, new_solution)

    # Creates random permutations of vehicles and using _divide_solution_greedy for
    # each of them. 
    # random - number of permutations.
    def _divide_solution_random(self, solution):
        random = self.random
        capacities = self.problem.capacities.copy()
        vehicles = len(capacities)

        new_solution = None
        best_cost = self.inf

        for i in range(random):
            perm = np.random.permutation(vehicles)
            inv = [list(perm).index(j) for j in range(vehicles)]
            self.problem.capacities = [capacities[j] for j in perm]

            new_sol = self._divide_solution_greedy_dp(solution)
            new_cost = new_sol.total_cost()

            if new_cost < best_cost and new_sol.check():
                best_cost = new_cost
                new_solution = new_sol
                new_solution.solution = [new_sol.solution[j] for j in inv]

            self.problem.capacities = capacities

        return new_solution

    def solve(self, only_one_const, order_const, solver_type = 'cpu'):
        problem = self.problem
        capacity = 0
        weights = problem.weights
        for w in weights:
            capacity += w

        # Creating new problem with one vehicle.
        sources = [0]
        dests = problem.dests
        costs = problem.costs
        new_capacities = [capacity]
        new_problem = VRPProblem(sources, costs, new_capacities, dests, weights)

        if len(dests) == 0:
            sol = [[] for _ in range(len(problem.capacities))]
            return VRPSolution(problem, None, None, sol)

        solver = self.solver
        solver.set_problem(new_problem)
        solution = solver.solve(only_one_const, order_const, solver_type = solver_type)

        sol = solution.solution[0]
        return self._divide_solution_random(sol)
