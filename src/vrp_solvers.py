from math import sqrt
import random
from qubo_helper import Qubo
from vrp_problem import VRPProblem
from vrp_solution import VRPSolution
from itertools import product
import DWaveSolvers
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
        try:
            self.count = random.randint(0.4*n,0.6*n)
        except:
            n += 1
            self.count = random.randint(0.4*n,0.6*n)

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
    def check_elements_match(self, array1, array2):
        if len(array1) != len(array2):
            return False
        for element in array1:
            if element not in array2:
                return False
        return True 

    def calculate_neighbor_cost(self, problem, clusters):
        routes = copy.deepcopy(clusters)
        check_sol = VRPSolution(problem, None, None, routes)
        # Adding first and last magazine.
        for rte in check_sol.solution:
            if rte:
                if problem.first_source:
                    rte.insert(0, problem.in_nearest_sources[rte[0]])
                if problem.last_source:
                    rte.append(problem.out_nearest_sources[rte[-1]])    
        return check_sol.total_cost()

    def calculate_route_cost(self, route, costs, sources):
        """Calculates the total cost of a given route."""
        total_cost = 0
        prev = sources[0]  # Assuming single source for simplicity
        for dest in route:
            total_cost += costs[prev][dest]
            prev = dest
        total_cost += costs[prev][sources[0]]  # Return to source
        return total_cost

    def build_initial_solution(self, vehicles, sorted_dests, neighborhood, weights, capacities):
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
                        weight += weights[dest]  #cluster current weight     
                    if weight + weights[d] <= capacities[i]:   
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
                        weight += weights[dest]  #cluster current weight   
                    if weight + weights[d] <= capacities[i]:   
                        clusters[i].append(d)
                        sorted_dests.pop(a)
                        found = True
                        break 
        return clusters

    def is_tabu(self, tabu, n):
        is_tabu = False                            
        for move in tabu:
            if move.move1 == n.move1 and move.location1 == n.location1 and move.move2 == n.move2 and move.location2 == n.location2:
                #candidate is tabu
                is_tabu = True
                break
        return is_tabu

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

        # 0. Create initial neighborhood for each destination
        # The initial neighborhood is 2 times the number of vehicles destinations
        # When we do swaps below we only swap locations that are in the same neighborhood
        neighborhood = [[] for _ in range(len(weights))]
        for d in dests:
            indices = np.argpartition(costs[d][:], vehicles*2)[:vehicles*2]
            neighborhood[d] = indices

        sorted_dests = sorted(dests, reverse=True , key=lambda i: costs[problem.in_nearest_sources[i]][i]) #costs[0][i]
        sorted_dests = [item for item in sorted_dests if item in dests]

        #Generate a starting solution for Tabu Search (1, 2 3)
        solver = ClarkWright(problem)
        solution = solver.solve()
        clusters = [arr[1:-1] for arr in solution.solution]

        #solver = SolutionPartitioningSolver(problem)
        #solution = solver.solve()
        #clusters = [arr[1:-1] for arr in solution.solution]

        #clusters = self.build_initial_solution(vehicles, sorted_dests, neighborhood, weights, capacities)

        #Check if the starting solution used fewer vehicles than the problem file specifices
        if len(clusters) < vehicles:
            vehicles = len(clusters)

        # 4. Calculate starting solution cost
        tabu = []   #dest, cluster_num
        neighbors = [] #dest, cluster_num, clusters
        best_solution = clusters
        best_cost = self.calculate_neighbor_cost(problem, clusters)
        print('starting total_cost =', best_cost)

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
            vehicle_weights = np.zeros(vehicles)  # Use NumPy array for speed
            for i, cluster in enumerate(clusters):
                vehicle_weights[i] = sum([self.problem.weights[dest] for dest in cluster])
                if vehicle_weights[i] > capacities[i]:
                    feasible = False
                    infeasible_amount += vehicle_weights[i] - capacities[i] 

            # 7. create candidate list of neighbors to current solution (8, 9, 10)
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
                for i in range(len(clusters)):  # Iterate directly through indices
                    if not clusters[i]:  # Skip empty clusters efficiently
                        continue
                    for idx_i, swap1 in enumerate(clusters[i]):
                        for j in range(i + 1, len(clusters)):  # Avoid redundant checks
                            if not clusters[j]:
                                continue
                            if not (set(neighborhood[swap1]).intersection(clusters[j])):  # Early exit
                                continue
                            for idx_j, swap2 in enumerate(clusters[j]):
                                if not (set(neighborhood[swap2]).intersection(clusters[i])):  # Early exit
                                    continue
                                weight1 = vehicle_weights[j] - self.problem.weights[swap2] + self.problem.weights[swap1]
                                weight2 = vehicle_weights[i] - self.problem.weights[swap1] + self.problem.weights[swap2]
                                if weight1 <= capacities[j] and weight2 <= capacities[i]:
                                    new_neighbor = copy.deepcopy(clusters) 
                                    new_neighbor[j][idx_j], new_neighbor[i][idx_i] = swap1, swap2
                                    neighbors.append(Neighbor(new_neighbor, swap1, i, swap2, j))
                                else:
                                    new_neighbor = copy.deepcopy(clusters)
                                    new_neighbor[j][idx_j], new_neighbor[i][idx_i] = swap1, swap2
                                    inf_neighbors.append(Neighbor(new_neighbor, swap1, i, swap2, j)) 


            # 10. 1,0
            if False == False:                                                
                for i in range(vehicles):
                    for d in clusters[i]:
                        for j in range(vehicles):
                            if i != j and d not in clusters[j] and set(neighborhood[d]).intersection(clusters[j]):
                                # Found a potential move: delivery 'd' from cluster 'i' to 'j'
                                # Check capacity constraint first for efficiency
                                if vehicle_weights[j] + self.problem.weights[d] <= capacities[j]:
                                    # Calculate the cost of inserting 'd' into all possible positions in cluster 'j'
                                    best_found_cost, best_found_spot = float('inf'), None
                                    for k in range(len(clusters[j]) + 1):
                                        new_route = clusters[j][:k] + [d] + clusters[j][k:]
                                        cost = self.calculate_route_cost(new_route, costs, sources)
                                        if cost < best_found_cost:
                                            best_found_cost, best_found_spot = cost, k

                                    # If a valid insertion point is found, create the neighbor solution
                                    if best_found_spot is not None:
                                        new_neighbor = copy.deepcopy(clusters)
                                        new_neighbor[i].remove(d)
                                        new_neighbor[j] = new_neighbor[j][:best_found_spot] + [d] + new_neighbor[j][best_found_spot:]
                                        n = Neighbor(new_neighbor, d, i)  # Assuming Neighbor class exists
                                        neighbors.append(n)
                                else:
                                    # Capacity constraint violated, add to inf_neighbors
                                    new_neighbor = copy.deepcopy(clusters)
                                    new_neighbor[i].remove(d)
                                    new_neighbor[j].append(d)  # Append to the end for simplicity
                                    n = Neighbor(new_neighbor, d, i)
                                    inf_neighbors.append(n)


            current_best_neighbor = []
            current_best_cost = self.max_dist
            current_best_move = ""
            selected_neighbor = []
            selected_neighbor_cost = self.max_dist
            selected_inf_neighbor = []
            selected_inf_neighbor_cost = self.max_dist

            # 11. Strategic Oscillation (12, 13)
            # 12. previous solution was feasible
            if feasible == True:
                #find best feasible candidate               
                for n in neighbors:
                    cost = self.calculate_neighbor_cost(problem, n.clusters)
                    if cost < selected_neighbor_cost:
                        #keep track of overall best neighbor
                        if cost < current_best_cost:
                            current_best_neighbor = n
                            current_best_cost = cost
                            current_best_move = n.type
                        #check if candidate is tabu
                        if self.is_tabu(tabu, n) is False:
                            #keep track of best non-tabu neighbor
                            selected_neighbor = n
                            selected_neighbor_cost = cost

                #find best infeasible candidate                                         
                for n in inf_neighbors:
                    cost = self.calculate_neighbor_cost(problem, n.clusters)
                    if cost < selected_inf_neighbor_cost and self.is_tabu(tabu, n) is False:
                        #keep track of best non-tabu neighbor
                        selected_inf_neighbor = n
                        selected_inf_neighbor_cost = cost

                #pick the best neighbor
                #if selected_neighbor_cost > previous_cost:
                if selected_inf_neighbor_cost < selected_neighbor_cost:
                    selected_neighbor = selected_inf_neighbor 
                    selected_neighbor_cost = selected_inf_neighbor_cost 

            # 13. previous solution was NOT feasible                
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
                        cost = self.calculate_neighbor_cost(problem, n.clusters)
                        if cost < current_best_cost:
                            current_best_neighbor = n
                            current_best_cost = cost
                            current_best_move = n.type
                        #check if candidate is tabu
                        if self.is_tabu(tabu, n) is False:                    
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
                    if inf_infeasible_amount <= best_inf_amount and self.is_tabu(tabu, n) is False:                
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
                        break
                #feasible, so lets use it
                if current_best_feasible == True:
                    if best_cost - current_best_cost > largest_change:
                        largest_change = best_cost - current_best_cost
                        #ignore tabu and use it anyways
                    best_cost = current_best_cost
                    print('total_cost =', best_cost, 'move=', current_best_move, 'counter= ', counter)
                    best_solution = copy.deepcopy(current_best_neighbor.clusters)
                    clusters = copy.deepcopy(current_best_neighbor.clusters)
                    tabu = []
                    counter_of_last_threshold = counter
                    counter_of_last_best = counter
                    if self.is_tabu(tabu, current_best_neighbor) == False:
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

            # 16. Toggle Diversification and do Intensification
            # threshold is reached so we toggle on diversification
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
                elif intensification_counter == 1 and diversification_counter % 10 == 0: #intensification
                    print('intensification', counter)
                    print('div counter ', diversification_counter)
                    tabu = []  
                    counter_of_last_threshold = counter
                    if diversification == True:
                        intensification_counter = 0
                        last_threshold = random.randint(int(0.6 * N), int(1.1 * N))
                    else:
                        intensification_counter +=1
                        last_threshold = random.randint(int(0.6 * N), int(1.1 * N))
                else: #threshold is reached so we toggle off diversification
                    print('diversification off', counter)                  
                    counter_of_last_threshold = counter
                    last_threshold = random.randint(int(0.6 * N), int(1.1 * N))
                    diversification = False
                    intensification_counter +=1
                    neighborhood = [[] for _ in range(len(weights))]
                    for d in dests:
                        indices = np.argpartition(costs[d][:], vehicles)[:vehicles]
                        neighborhood[d] = indices          

            # 17. Sparse Quantum Resequencing
            if counter - counter_of_last_best == 2000:      
                print('Quantum Go', counter)              
                clusters = copy.deepcopy(best_solution) 
                routes = list()
                for cluster in clusters:
                    if len(cluster) > 1:
                        found = False
                        for rte in optimized_routes: #check if we have already sequenced this route
                            if self.check_elements_match(cluster, rte):
                                route = rte
                                found = True
                        if found == False:
                            new_problem = VRPProblem(sources, costs, [capacities[0]], cluster, weights, first_source = True, last_source = True)
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
                cost = self.calculate_neighbor_cost(problem, routes)
                if cost < best_cost:
                    best_solution = copy.deepcopy(clusters)
                    best_cost = cost
                    counter_of_last_best = counter
                    print('quantum found total_cost =', best_cost)

            # 18. update tabu list
            for move in tabu:
                move.count -= 1
                if move.count == 0:
                    tabu.remove(move)    

            # 19. update iterator and loop back
            counter += 1
            if counter - counter_of_last_best == 5000: #stop if its been XXXX moves since we found a new best
                print('Best solution was found on counter =', counter_of_last_best)
                ready_to_stop = True

        # 20. Adding first and last magazine and return best found solution.
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

class ClarkWright(VRPSolver):
    def __init__(self, problem):
        self.problem = problem

    def which_route(self, link, routes):
        node_sel = []
        i_route = [-1, -1]
        count_in = 0

        for route in routes:
            for node in link:
                try:
                    route.index(node)
                    i_route[count_in] = routes.index(route)
                    node_sel.append(node)
                    count_in += 1
                except:
                    pass

        overlap = 1 if i_route[0] == i_route[1] else 0

        return node_sel, count_in, i_route, overlap

    def merge(self, route0, route1, link):
        if route0.index(link[0]) != (len(route0) - 1):
            route0.reverse()

        if route1.index(link[1]) != 0:
            route1.reverse()

        return route0 + route1

    def interior(self, node, route):
        try:
            i = route.index(node)
            return 0 < i < len(route) - 1
        except ValueError:
            return False
        
    # sum up to obtain the total passengers belonging to a route
    def sum_cap(self, route):
        sum_cap = 0
        for node in route:
            sum_cap += self.problem.weights[node]
        return sum_cap

    def solve(self):
        problem = self.problem
        num_customers = len(problem.dests)
        nodes = problem.dests
        capacities = problem.capacities
        costs = problem.costs

        # Calculate savings matrix
        savings = np.zeros((num_customers, num_customers))
        for i in range(num_customers):
            for j in range(i+1, num_customers):
                savings[i][j] = costs[0][i+1] + costs[0][j+1] - costs[i+1][j+1]
                
        # Sort savings matrix in decreasing order
        savings_flat = [(i, j, savings[i][j]) for i in range(num_customers) for j in range(i+1, num_customers)]
        savings_flat = [(i+1, j+1, savings[i][j]) for i in range(num_customers) for j in range(i+1, num_customers)]
        savings_flat_sorted = sorted(savings_flat, key=lambda x: x[2], reverse=True)

        savings_flat_sorted = [[node1, node2] for node1, node2, savings in savings_flat_sorted]
        for item in savings_flat_sorted:
            if 0 in item:
                print(item)

        


        # Create empty routes
        routes = []

        # Get a list of nodes, excluding the depot
        node_list = list(nodes)

        #if there are any remaining customers to be served
        remaining = True
        
        for link in savings_flat_sorted:
            print(link)
            if remaining:
                
                node_sel, num_in, i_route, overlap = self.which_route(link, routes)
                 # condition a. Either, neither i nor j have already been assigned to a route, 
                # ...in which case a new route is initiated including both i and j.
                if num_in == 0:
                    if self.sum_cap(link) <= capacities[0]:
                        routes.append(link)
                        node_list.remove(link[0])
                        node_list.remove(link[1])
                        print('\t','Link ', link, ' fulfills criteria a), so it is created as a new route')
                    else:
                        print('\t','Though Link ', link, ' fulfills criteria a), it exceeds maximum load, so skip this link.')
                        
                # condition b. Or, exactly one of the two nodes (i or j) has already been included 
                # ...in an existing route and that point is not interior to that route 
                # ...(a point is interior to a route if it is not adjacent to the depot D in the order of traversal of nodes), 
                # ...in which case the link (i, j) is added to that same route.    
                elif num_in == 1:
                    n_sel = node_sel[0]
                    i_rt = i_route[0]
                    position = routes[i_rt].index(n_sel)
                    link_temp = link.copy()
                    link_temp.remove(n_sel)
                    node = link_temp[0]

                    cond1 = (not self.interior(n_sel, routes[i_rt]))
                    cond2 = (self.sum_cap(routes[i_rt] + [node]) <= capacities[0])

                    if cond1:
                        if cond2:
                            print('\t','Link ', link, ' fulfills criteria b), so a new node is added to route ', routes[i_rt], '.')
                            if position == 0:
                                routes[i_rt].insert(0, node)
                            else:
                                routes[i_rt].append(node)
                            node_list.remove(node)
                        else:
                            print('\t','Though Link ', link, ' fulfills criteria b), it exceeds maximum load, so skip this link.')
                            continue
                    else:
                        print('\t','For Link ', link, ', node ', n_sel, ' is interior to route ', routes[i_rt], ', so skip this link')
                        continue
                    
                # condition c. Or, both i and j have already been included in two different existing routes 
                # ...and neither point is interior to its route, in which case the two routes are merged.        
                else:
                    if overlap == 0:
                        cond1 = (not self.interior(node_sel[0], routes[i_route[0]]))
                        cond2 = (not self.interior(node_sel[1], routes[i_route[1]]))
                        cond3 = (self.sum_cap(routes[i_route[0]] + routes[i_route[1]]) <= capacities[0])

                        if cond1 and cond2:
                            if cond3:
                                route_temp = self.merge(routes[i_route[0]], routes[i_route[1]], node_sel)
                                temp1 = routes[i_route[0]]
                                temp2 = routes[i_route[1]]
                                routes.remove(temp1)
                                routes.remove(temp2)
                                routes.append(route_temp)
                                try:
                                    node_list.remove(link[0])
                                    node_list.remove(link[1])
                                except:
                                    #print('\t', f"Node {link[0]} or {link[1]} has been removed in a previous step.")
                                    pass
                                print('\t','Link ', link, ' fulfills criteria c), so route ', temp1, ' and route ', temp2, ' are merged')
                            else:
                                print('\t','Though Link ', link, ' fulfills criteria c), it exceeds maximum load, so skip this link.')
                                continue
                        else:
                            print('\t','For link ', link, ', Two nodes are found in two different routes, but not all the nodes fulfill interior requirement, so skip this link')
                            continue
                    else:
                        print('\t','Link ', link, ' is already included in the routes')
                        continue
                    
                for route in routes: 
                    print('\t','route: ', route, ' with load ', self.sum_cap(route))
            else:
                print('-------')
                print('All nodes are included in the routes, algorithm closed')
                break
            
            remaining = bool(len(node_list) > 0)

        # check if any node is left, assign to a unique route
        for node_o in node_list:
            routes.append([node_o])

        # add depot to the routes
        for route in routes:
            route.insert(0,0)
            route.append(0)


        return VRPSolution(problem, None, None, routes)