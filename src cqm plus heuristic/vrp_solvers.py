
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
            self.count = random.randint(int(0.4*n),int(0.6*n))

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

    def calculate_solution_cost(self, routes, costs, sources):
        total_cost = 0
        for route in routes:
            total_cost += self.calculate_route_cost(route, costs, sources)
        return total_cost

    def calculate_route_cost(self, route, costs, sources):
        """Calculates the total cost of a given route."""
        total_cost = 0
        prev = sources[0]  # Assuming single source for simplicity
        for dest in route:
            total_cost += costs[prev][dest]
            prev = dest
        total_cost += costs[prev][sources[0]]  # Return to source
        return total_cost
    
    def exhaustive_search(self, route):
        for new_route in itertools.permutations(route):
            new_route = list(new_route)
            if not self.check_time(new_route):
                return new_route, True
        return route, False

    def swap(self, route):
        n = len(route)
        for i in range(n):
            for j in range(i + 1, n):
                new_route = route[:]
                new_route[i], new_route[j] = new_route[j], new_route[i]  
                if not self.check_time(new_route):
                    return new_route, True
        return route, False

    def two_opt_swap(self, route):
        n = len(route)
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                if not self.check_time(new_route):
                    return new_route, True
        return route, False

    def three_opt_swap(self, route):
        n = len(route)
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    new_route = route[:i] + route[j:k+1] + route[i:j] + route[k+1:]
                    if not self.check_time(new_route):
                        return new_route, True
        return route, False      

    def check_time(self, route):
        totalTime = 0
        costs = self.problem.costs
        timeIntervals = self.problem.time_intervals
        services = self.problem.services
        # sorted_route = sorted(route, key=lambda node: (timeIntervals[str(node)][0]))
        route = [0] + route + [0]  # Start and end at the depot (node 0)
        
        
        for i in range(len(route) - 1):
            prevNode = route[i]
            currentNode = route[i + 1]
            travel_time = costs[prevNode][currentNode]
            readyTime = timeIntervals[str(currentNode)][0]
            dueTime = timeIntervals[str(currentNode)][1]
                        
            # Update totalTime with travel time from previous node to current node
            totalTime += travel_time
            
            # Check if total time is outside the time window
            if totalTime <= readyTime:
                totalTime = readyTime  # Adjust for early arrival, wait for ready time
            elif totalTime > dueTime:
               # print(f'Time violated at node {currentNode}')
                return True  # Time window violated
            
            # Add service time for all locations except the depot (node 0)
            if currentNode != 0:
                totalTime += services[0]
            
           # print(f'currentTime: {totalTime}')
        
        return False  # All time windows respected
    
    def totalTime(self, route):
        totalTime = 0
        costs = self.problem.costs
        timeIntervals = self.problem.time_intervals
        services = self.problem.services
        # sorted_route = sorted(route, key=lambda node: (timeIntervals[str(node)][0]))
        route = [0] + route + [0]  # Start and end at the depot (node 0)
        
        
        for i in range(len(route) - 1):
            prevNode = route[i]
            currentNode = route[i + 1]
            travel_time = costs[prevNode][currentNode]
            readyTime = timeIntervals[str(currentNode)][0]
            dueTime = timeIntervals[str(currentNode)][1]
                        
            # Update totalTime with travel time from previous node to current node
            totalTime += travel_time
            
            # Check if total time is outside the time window
            if totalTime <= readyTime:
                totalTime = readyTime  # Adjust for early arrival, wait for ready time
            
            # Add service time for all locations except the depot (node 0)
            if currentNode != 0:
                totalTime += services[0]
            
           # print(f'currentTime: {totalTime}')
        
        return totalTime  # All time windows respected
    

    def is_tabu(self, tabu, n):
        is_tabu = False                            
        for move in tabu:
            if move.move1 == n.move1 and move.location1 == n.location1 and move.move2 == n.move2 and move.location2 == n.location2:
                #candidate is tabu
                is_tabu = True
                break
        return is_tabu

    def update_neighborhood(self, dests, costs, weights, size):
        neighborhood = [[] for _ in range(len(weights))]
        for d in dests:
            indices = np.argpartition(costs[d][:], int(size))[:int(size)]
            neighborhood[d] = indices
        return neighborhood

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
        time_intervals = problem.time_intervals
        services = problem.services
        vehicles = len(problem.capacities)
        lastSolution = ()

        # 0. Create initial neighborhood for each destination
        # The initial neighborhood is 2 times the number of vehicles destinations
        # When we do swaps below we only swap locations that are in the same neighborhood
        neighborhood = self.update_neighborhood(dests, costs, weights, vehicles * 2)

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
        tabu = []   #the tabu list, holds tabu moves
        neighbors = [] #the neighbor list, holds all the neighboring moves from the current solution found by local search
        best_solution = clusters    #holds all the routes for the best solution found so far
        best_cost = self.calculate_solution_cost(clusters, costs, sources) #the cost of the best solution found so far
        print('starting total_cost =', best_cost)

        optimized_routes = list()       #cache for quantum resequenced routes
        counter_of_last_threshold = 0   #holds the global counter's value when the thershold happened
        last_threshold = random.randint(int(0.6 * N), int(1.1 * N)) #number of moves until we consider a diversification or intensification change
        counter_of_last_best = 0        #hold the global counter's value when the last best solution was found
        intensification_counter = 2     #counter used to determine if we do intensification
        diversification = True          #bool to control if diversification is on or off
        diversification_counter = 0     #counter used to to determine if we do intensification
        counter = 0                     #primary itertor for the tabu search
        ready_to_stop = False           #set this to True to stop the tabu search
        largest_change = 0              #holds the largest improvment in solution cost for a single move
        frequency = defaultdict(int)    #not used at this time
        DEPOT_RETURN_TIME = time_intervals['0'][1]


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
                if vehicle_weights[i] > capacities[i] or self.check_time(cluster):
                    feasible = False
                    infeasible_amount += vehicle_weights[i] - capacities[i] 

            # Local Search
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
                                if vehicle_weights[i] <= capacities[i] and not self.check_time(new_neighbor[i]):
                                    # if not self.check_time(new_neighbor[i]): #respects time windows
                                    #     lastSolution = (" 1 route, cap, time",new_neighbor[i],self.calculate_route_cost(new_neighbor[i], costs, sources) > capacities[i], self.check_time(new_neighbor[i]))
                                    #     # print(f"Last Solution1: {new_neighbor[i]}")
                                    #     # print(f'Capacity ViolatedD: {self.calculate_route_cost(new_neighbor[i], costs, sources) <= capacities[i]}')
                                    #     # print(f'Time ViolatedD: {self.check_time(new_neighbor[i])}')
                                    neighbors.append(n)
                                # else:
                                    lastSolution = "GREEN"
                                        # inf_neighbors.append(n)
                                else:
                                    lastSolution = "REDDDD"
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
                                new_neighbor = copy.deepcopy(clusters) 
                                new_neighbor[j][idx_j], new_neighbor[i][idx_i] = swap1, swap2
                                new_neigh_j_time_check, new_neigh_i_time_check = self.check_time(new_neighbor[j]), self.check_time(new_neighbor[i])
                                if weight1 <= capacities[j] and weight2 <= capacities[i] and not new_neigh_j_time_check and not new_neigh_i_time_check:
                                    
                                    # if not new_neigh_j_time_check and not new_neigh_i_time_check: #if both are false, if statement will be true 
                                    neighbors.append(Neighbor(new_neighbor, swap1, i, swap2, j)) #swap meets capacity and time constraints

                                        
                                        # print(f"Last Solution2: {new_neighbor[j]}")
                                        # print(f'Capacity ViolatedD: {self.calculate_route_cost(new_neighbor[j], costs, sources) <= capacities[j]}')
                                        # print(f'Time ViolatedD: {self.check_time(new_neighbor[j])}')

                                        # print(f"Last Solution2: {new_neighbor[i]}")
                                        # print(f'Capacity ViolatedD: {self.calculate_route_cost(new_neighbor[i], costs, sources) <= capacities[i]}')
                                        # print(f'Time ViolatedD: {self.check_time(new_neighbor[i])}')

                                    lastSolution = ("2 j,i, route, cap, time", new_neighbor[j], self.check_time(new_neighbor[j]), new_neighbor[i], self.check_time(new_neighbor[i]))
                                        
                                    # else:
                                    #     inf_neighbors.append(Neighbor(new_neighbor, swap1, i, swap2, j))
                                else:
                                    # new_neighbor = copy.deepcopy(clusters) 
                                    # new_neighbor[j][idx_j], new_neighbor[i][idx_i] = swap1, swap2
                                    lastSolution = ("YELLOW")

                                    inf_neighbors.append(Neighbor(new_neighbor, swap1, i, swap2, j)) 


            # 10. 1,0
            # Setting a flag to trigger this section
            if False == False:                                                 
                for i in range(vehicles):
                    for d in clusters[i]:  # Iterate through each delivery in vehicle i's cluster
                       
                        for j in range(vehicles):
                            # Skip if attempting to move within the same cluster or to a cluster containing `d`
                            if i != j and d not in clusters[j] and set(neighborhood[d]).intersection(clusters[j]):
                                # Proceed only if the capacity constraint would not be violated in cluster `j`
                                if vehicle_weights[j] + self.problem.weights[d] <= capacities[j]:
                                    best_found_cost, best_found_spot = float('inf'), None
                                    
                                    # Try to insert `d` into every position in cluster `j`
                                    for k in range(len(clusters[j]) + 1):
                                        # Create a potential new route with `d` inserted at position `k`
                                        new_route = clusters[j][:k] + [d] + clusters[j][k:]
                                        
                                        # Check time constraints for the new route
                                        if not self.check_time(new_route):  # If time is valid
                                            cost = self.calculate_route_cost(new_route, costs, sources)
                                            if cost < best_found_cost:
                                                best_found_cost, best_found_spot = cost, k
                                        #else:
                                        #    # print(f"Time constraint violated for inserting {d} into position {k} of cluster {j}")
                                        #    # Time constraint violated; add to infeasible neighbors
                                        #    new_neighbor = copy.deepcopy(clusters)
                                        #    new_neighbor[i].remove(d)
                                        #    new_neighbor[j] = clusters[j][:k] + [d] + clusters[j][k:]
                                        #    n = Neighbor(new_neighbor, d, i)
                                        #    lastSolution = ("6 j i route, time", new_neighbor[j], self.check_time(new_neighbor[j]), new_neighbor[i], self.check_time(new_neighbor[i]))
                                        #    inf_neighbors.append(n)

                                    # If a feasible insertion point was found, add the neighbor
                                    if best_found_spot is not None:
                                        new_neighbor = copy.deepcopy(clusters)
                                        new_neighbor[i].remove(d)
                                        new_neighbor[j] = new_neighbor[j][:best_found_spot] + [d] + new_neighbor[j][best_found_spot:]
                                        n = Neighbor(new_neighbor, d, i)
                                        lastSolution = ("7 j i route, cap, time", new_neighbor[j], self.check_time(new_neighbor[j]), new_neighbor[i], self.check_time(new_neighbor[i]))
                                        neighbors.append(n)
                                    
                                else:
                                    # Capacity constraint violated; add to infeasible neighbors
                                    # print(f"Capacity constraint violated for inserting {d} into cluster {j}")
                                    new_neighbor = copy.deepcopy(clusters)
                                    new_neighbor[i].remove(d)
                                    new_neighbor[j].append(d)  # Append to the end for simplicity
                                    n = Neighbor(new_neighbor, d, i)
                                    lastSolution = ("4 j i route, time", new_neighbor[j], self.check_time(new_neighbor[j]), new_neighbor[i], self.check_time(new_neighbor[i]))                                    
                                    inf_neighbors.append(n)



            current_best_neighbor = []          #holds the best neighbor found by the local search (might be tabu)
            current_best_cost = self.max_dist
            current_best_move = ""
            selected_neighbor = []              #holds the best non-tabu feasible move
            selected_neighbor_cost = self.max_dist
            selected_inf_neighbor = []          #holds the best non-tabu infeasible move
            selected_inf_neighbor_cost = self.max_dist

            # 11. Strategic Oscillation (12, 13)
            # 12. Previous solution was feasible
            # if feasible == True and len(inf_neighbors) != 0 and len(neighbors) != 0:
            if feasible == True:
                testVar = True
                for n in neighbors:
                    cost = self.calculate_solution_cost(n.clusters, costs, sources)
                    
                    # Only consider updating if the cost is actually better than current best
                    if cost < selected_neighbor_cost:
                        # Track the best neighbor overall
                        if cost < current_best_cost:
                            current_best_neighbor = n
                            testVar = False
                            current_best_cost = cost
                            current_best_move = n.type
                    
                        # Check if this neighbor is non-tabu
                        if not self.is_tabu(tabu, n):
                            selected_neighbor = n
                            selected_neighbor_cost = cost

                # Log when no feasible candidate was found
                # if testVar:
                #     print(f"Neightors {len(neighbors)} Ing Neigh {len(inf_neighbors)}")
                #     print("Warning: No feasible candidate found; potential constraints issue.")
                #     print("Last Solution details:", lastSolution)

                # Process infeasible candidates if no feasible solutions were found
                for n in inf_neighbors:
                    cost = self.calculate_solution_cost(n.clusters, costs, sources)
                    if cost < selected_inf_neighbor_cost and not self.is_tabu(tabu, n):
                        selected_inf_neighbor = n
                        selected_inf_neighbor_cost = cost

                # Choose the best between feasible and infeasible, if needed
                if selected_inf_neighbor_cost < selected_neighbor_cost:
                    selected_neighbor = selected_inf_neighbor
                    selected_neighbor_cost = selected_inf_neighbor_cost

            # 13. If previous solution was NOT feasible
            # elif feasible == False and len(inf_neighbors) != 0 and len(neighbors) != 0:
            else:
                current_best_cost = self.max_dist
                best_amount = sum(capacities)
                best_inf_amount = sum(capacities)
                best_times = vehicles
                best_inf_times = vehicles
                testVar = True

                # Check feasible candidates
                for n in neighbors:
                    current_infeasible_amount = 0
                    current_infeasible_times = 0
                    current_weights = list()
                    current_times = list()
                    
                    # Calculate infeasibility due to weight
                    for i in range(vehicles):
                        current_weights.append(0)
                        current_times.append(0)
                        for dest in n.clusters[i]:
                            current_weights[i] += self.problem.weights[dest]
                        current_times[i] = self.check_time(n.clusters[i])
                        if current_weights[i] > capacities[i]:
                            current_infeasible_amount += current_weights[i] - capacities[i]
                        current_infeasible_times = sum(current_times)
                    # Consider only if it reduces infeasibility or is feasible
                    if current_infeasible_amount <= best_amount and current_infeasible_times <= best_times:
                        cost = self.calculate_solution_cost(n.clusters, costs, sources)
                        if cost < current_best_cost:
                            testVar = False
                            current_best_neighbor = n
                            current_best_cost = cost
                            current_best_move = n.type
                        # Check if non-tabu and update
                        if not self.is_tabu(tabu, n):
                            selected_neighbor = n
                            selected_neighbor_cost = cost
                            best_amount = current_infeasible_amount
                            best_times = current_infeasible_times

                # Additional logging if no improvement
                # if testVar:
                #     print(f"Neightors {len(neighbors)} Ing Neigh {len(inf_neighbors)}")
                #     print("No feasible solution found; using fallback. Last Solution:", lastSolution)

                    
                # if not testVar:
                #     print("Didnt update Infeasible Cost")
                #     print(lastSolution)

                #find best infeasible candidate                            
                for n in inf_neighbors:
                    inf_infeasible_amount = 0
                    inf_infeasible_times = 0
                    current_weights = list()
                    current_times = list()
                    for i in range(vehicles):
                        current_weights.append(0)
                        current_times.append(0)
                        for dest in n.clusters[i]:
                            current_weights[i] += self.problem.weights[dest]
                        current_times[i] = self.check_time(n.clusters[i])
                        if current_weights[i] > capacities[i]:
                            inf_infeasible_amount += current_weights[i] - capacities[i]
                        inf_infeasible_times = sum(current_times)
                    if inf_infeasible_amount <= best_inf_amount and inf_infeasible_times <= best_inf_times and self.is_tabu(tabu, n) is False:                
                        #keep track of best non-tabu neighbor
                        selected_inf_neighbor = n
                        selected_inf_neighbor_cost = self.calculate_solution_cost(n.clusters, costs, sources)
                        best_inf_amount = inf_infeasible_amount
                        best_inf_times = inf_infeasible_times

                #pick the best neighbor
                if best_inf_amount < best_amount and best_inf_times < best_times:
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
                    for dest in current_best_neighbor.clusters[i]:
                        vehicle_weights[i] += self.problem.weights[dest]
                    if vehicle_weights[i] > capacities[i] or self.check_time(current_best_neighbor.clusters[i]): #if capacity is violated or time is violated
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
                # if current_best_cost == self.max_dist:
                    

                #print(lastSolution)                 
                print(f"Neighbors {len(neighbors)} Inf Neighbors {len(inf_neighbors)}")
                print('counter', counter, 'cbc', current_best_cost, 'snc', selected_neighbor_cost, 'move', current_best_move, "feasible", feasible)
                if intensification_counter == 2: #diversification
                    print('diversification on', counter)
                    counter_of_last_threshold = counter
                    last_threshold = random.randint(int(0.6 * N), int(1.1 * N))
                    diversification = True
                    intensification_counter = 1   
                    diversification_counter += 1
                    neighborhood_range = random.randint(vehicles * 2, vehicles * 4)
                    neighborhood = self.update_neighborhood(dests, costs, weights, neighborhood_range)
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
                    neighborhood = self.update_neighborhood(dests, costs, weights, vehicles)     

            # 17. Sparse Quantum Resequencing
            # if counter - counter_of_last_best == 2000:      
            #     print('Quantum Go', counter)              
            #     clusters = copy.deepcopy(best_solution) 
            #     routes = list()
            #     for cluster in clusters:
            #         if len(cluster) > 1:
            #             found = False
            #             for rte in optimized_routes: #check if we have already sequenced this route
            #                 if self.check_elements_match(cluster, rte):
            #                     route = rte
            #                     found = True
            #             if found == False:
            #                 new_problem = VRPProblem(sources, costs, [capacities[0]], cluster, weights, time_intervals, services, first_source = True, last_source = True)
            #                solver = FullQuboSolver(new_problem)
            #                 print('0 =', cluster)
            #                 route = solver.solve(only_one_const, order_const, solver_type = solver_type).solution[0]                                
            #                 del route[0]
            #                 del route[-1]
            #                 print('1 =', route)
            #                 optimized_routes.append(copy.deepcopy(route))
            #         else:
            #             route = cluster
            #         routes.append(route)
            #     clusters = routes
            #     cost = self.calculate_neighbor_cost(problem, routes)
            #     if cost < best_cost:
            #         best_solution = copy.deepcopy(clusters)
            #         best_cost = cost
            #         counter_of_last_best = counter
            #         print('quantum found total_cost =', best_cost)

            #17  Quantum Resequencing with CQM
            from dimod import ConstrainedQuadraticModel, Binary, Integer, Real
            from dwave.system import LeapHybridCQMSampler

            if counter - counter_of_last_best == 250:
                print('Quantum Go', counter)
                clusters = copy.deepcopy(best_solution)
                routes = []

                for cluster in clusters:
                    print(f'CLUSTER: {cluster}')

                    if len(cluster) > 2:
                        found = False

                        # Check if we already sequenced this route
                        for rte in optimized_routes:
                            if self.check_elements_match(cluster, rte):
                                route = rte
                                found = True
                                break

                        if not found:
                            # Create a CQM problem for this cluster
                            cqm = []
                            cqm = ConstrainedQuadraticModel()                            

                            # Add depot (node 0) to the cluster
                            cluster = [0] + cluster
                            n = len(cluster) - 1

                            # Define binary variables
                            x = {f'x_{i}_{j}': Binary(f'x_{i}_{j}') for i in cluster for j in cluster if i != j}
                            #print(x)
                            z = {f'z_{i}_{j}': Real(f'z_{i}_{j}') for i in cluster for j in cluster if i != j}
                            #print(z)
                            #s = {f's_{i}': Integer(f's_{i}', lower_bound = time_intervals[str(i)][0], upper_bound= time_intervals[str(i)][1]) for i in cluster }


                            # max_cost = max(max(row) for row in costs)
                            # scaled_costs = [[cost / max_cost for cost in row] for row in costs]

                            # Define the objective function
                            objective = sum(costs[i][j] * x[f'x_{i}_{j}'] for i in cluster for j in cluster if i != j)                           
                            cqm.set_objective(objective)

                            ##### CONSTRAINTS #####
                            # Constraint 1: Each city (including depot) must be reached only once
                            for j in cluster:
                                cqm.add_constraint(
                                    sum(x[f'x_{i}_{j}'] for i in cluster if i != j) == 1,
                                    f'Constraint_1_reach_{j}'
                                )

                            # Constraint 2: Each city (including depot) must be exited only once
                            for i in cluster:
                                cqm.add_constraint(
                                    sum(x[f'x_{i}_{j}'] for j in cluster if i != j) == 1,
                                    f'Constraint_2_exit_{i}'
                                )

                            for i in cluster:
                                if i > 0:
                                    cqm.add_constraint(
                                        sum(z[f'z_{i}_{j}'] for j in cluster if i != j) - sum(z[f'z_{j}_{i}'] for j in cluster if j != 0 and i != j) == 1,
                                        f'Constraint_3_{i}'
                                    )

                            for i in cluster:
                                for j in cluster:
                                    if i != 0 and i != j:
                                        if x[f'x_{i}_{j}'] == 0:
                                            pass
                                        else:
                                            cqm.add_constraint(
                                                z[f'z_{i}_{j}']  - (len(cluster)-1) * x[f'x_{i}_{j}'],
                                                sense="<=", rhs=0, label = f'Constraint_4_{i}_{j}'
                                            )

                            # Solve the CQM with the initial state
                            solver = LeapHybridCQMSampler()
                            solution = solver.sample_cqm(cqm)
                            feasible_sampleset = solution.filter(lambda row: row.is_feasible)
                            #print(feasible_sampleset)

                            if len(feasible_sampleset):
                                print(f"{len(feasible_sampleset)} feasible solutions of {len(solution)}.")
                                sol = feasible_sampleset.first
                                #print(sol)
                                print(f'Minimum total cost: {sol.energy}')

                                # Extract the route
                                active_var_list = []
                                for var, value in sol.sample.items():
                                    if value == 1.0 and var.startswith('x_'):  # Ensure the variable is part of x_{i}_{j}
                                        parts = var.split('_')
                                        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():  # Validate format
                                            i, j = int(parts[1]), int(parts[2])
                                            active_var_list.append((i, j))
                                
                                #print(active_var_list)

                                # Construct the route
                                optimized_route = []
                                current_node = 0  # Start at the depot
                                while active_var_list:
                                    for (i, j) in active_var_list:
                                        if i == current_node:
                                            optimized_route.append((i, j))
                                            current_node = j
                                            active_var_list.remove((i, j))
                                            break

                                print(f"Cluster: {cluster}")
                                print(f"Optimized Route: {optimized_route}")
                                new_route = []
                                for a, b in optimized_route:
                                    new_route.append(a)
                                route = new_route[1:]

                                if not self.check_time(route):
                                    optimized_routes.append(copy.deepcopy(route))
                                else:
                                    #try swaps to find feasible route
                                    new_route, feasible = self.swap(route)
                                    if not feasible:
                                        new_route, feasible = self.two_opt_swap(route)
                                        if not feasible:
                                            new_route, feasible = self.three_opt_swap(route) 
                                                                          
                                    if feasible:
                                        print(f"Fixed Route: {new_route}")
                                        optimized_routes.append(copy.deepcopy(new_route))
                                        route = new_route
                                    else:
                                        print(f"Unable to Fix Route: {new_route}")
                                        route = cluster[1:]
                                        optimized_routes.append(copy.deepcopy(route))
                            else:
                                print("No feasible solution found.")
                        else:
                            # Use the pre-optimized route
                            route = cluster
                    else:
                        # For two-node and single-node clusters, the route is the cluster itself
                        route = cluster
                    routes.append(route)
                # Update the solution and cost
                clusters = routes
                cost = self.calculate_solution_cost(routes, costs, sources)
                if cost < best_cost:
                    best_solution = copy.deepcopy(clusters)
                    best_cost = cost
                    counter_of_last_best = counter
                    print('Quantum found total_cost =', best_cost)



            # 18. update tabu list
            for move in tabu:
                move.count -= 1
                if move.count == 0:
                    tabu.remove(move)    

            # 19. update iterator and loop back
            counter += 1
            if counter - counter_of_last_best == N * 50: #stop if its been XXXX moves since we found a new best
                # print(f'good: {goodArray}')
                # print(f'bad: {badArray}')
                print('Best solution was found on counter =', counter_of_last_best)
                ready_to_stop = True

        # 20. Adding first and last magazine and return best found solution.
        for l in best_solution:
            if len(l) != 0:
                if problem.first_source:
                    l.insert(0, problem.in_nearest_sources[l[0]])
                if problem.last_source:
                    l.append(problem.out_nearest_sources[l[len(l) - 1]])

        solution = VRPSolution(self.problem, None, None, best_solution, counter_of_last_best)
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
    

    def check_time(self, route):
        totalTime = 0
        costs = self.problem.costs
        timeIntervals = self.problem.time_intervals
        services = self.problem.services
        sorted_route = sorted(route, key=lambda node: (timeIntervals[str(node)][0]))
        route = [0] + sorted_route + [0]  # Start and end at the depot (node 0)
        
        print(f'CurrentRoute: {route}')
        
        for i in range(len(route) - 1):
            prevNode = route[i]
            currentNode = route[i + 1]
            travel_time = costs[prevNode][currentNode]
            readyTime = timeIntervals[str(currentNode)][0]
            dueTime = timeIntervals[str(currentNode)][1]
            
            print(f'prev: {prevNode}, cur: {currentNode}, ready: {readyTime}, due: {dueTime}, travel: {travel_time}')
            
            # Update totalTime with travel time from previous node to current node
            totalTime += travel_time
            
            # Check if total time is outside the time window
            if totalTime <= readyTime:
                totalTime = readyTime  # Adjust for early arrival, wait for ready time
            elif totalTime > dueTime:
                print(f'Time violated at node {currentNode}')
                return True  # Time window violated
            
            # Add service time for all locations except the depot (node 0)
            if currentNode != 0:
                totalTime += services[0]
            
            print(f'currentTime: {totalTime}')
        
        return False  # All time windows respected



    
    def check_constraints(self, route):
        isTimeViolated = self.check_time(route) #true if time is violated
        isCapacityViolated = self.sum_cap(route) > self.problem.capacities[0] #true if capacity violated
        print(f'ISTime: {isTimeViolated} and IsCap: {isCapacityViolated}')
        print(f'Conditional: {isTimeViolated and isCapacityViolated}')
        return not (isCapacityViolated or isTimeViolated) #only returns true if both are not violated


    def solve(self):
        problem = self.problem
        num_customers = len(problem.dests)
        nodes = problem.dests
        capacities = problem.capacities
        costs = problem.costs
        time_intervals = problem.time_intervals

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

            # if time_intervals[str(link[0])][0] < time_intervals[str(link[1])][0]:
            #     link = [link[0], link[1]]

            if remaining:
                
                node_sel, num_in, i_route, overlap = self.which_route(link, routes)
                 # condition a. Either, neither i nor j have already been assigned to a route, 
                # ...in which case a new route is initiated including both i and j.
                if num_in == 0:
                    if self.check_constraints(link): #true if time is violated
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
                    cond2 = (self.check_constraints(routes[i_rt] + [node])) #true if time is violated

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
                        cond3 = (self.check_constraints(routes[i_route[0]] + routes[i_route[1]])) #true if time is violated

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
        for i in range(len(routes)):
            routes[i] = [0] + sorted(routes[i], key=lambda node: self.problem.time_intervals[str(node)][0]) + [0]

        # routes = [[0, 81, 78, 76, 71, 70, 73, 77, 79, 80, 0], [0, 57, 55, 54, 53, 56, 58, 60, 59, 0], [0, 98, 96, 95, 94, 92, 93, 97, 100, 99, 0], [0, 90, 87, 86, 83, 82, 84, 85, 88, 89, 91, 0], [0, 13, 17, 18, 19, 15, 16, 14, 12, 10, 0], [0, 32, 33, 31, 35, 37, 38, 39, 36, 34, 0], [0, 67, 65, 63, 62, 74, 72, 61, 64, 68, 66, 69, 0], [0, 43, 42, 41, 40, 44, 46, 45, 48, 51, 50, 52, 49, 47, 0], [0, 20, 24, 25, 27, 29, 30, 28, 26, 23, 22, 21, 0], [0, 5, 3, 7, 8, 11, 9, 6, 4, 2, 1, 75, 0]]
        # routes = [[0, 5, 0]]
        #         # add depot to the routes
        # for i in range(len(routes)):
        #     if len(routes[i]) > 2:  # Only sort if there are at least three nodes
        #         routes[i][1:-1] = sorted(routes[i][1:-1], key=lambda node: self.problem.time_intervals[str(node)][0])

        # routes = [[0, 22, 24, 27, 30, 29, 6, 34, 32, 33, 31, 35, 37, 38, 39, 36, 28, 26, 23, 18, 19, 16, 14, 12, 15, 17, 13, 25, 9, 11, 10, 8, 21, 20, 0], [0, 93, 1, 98, 95, 99, 100, 97, 94, 92, 7, 2, 5, 75, 4, 3, 89, 91, 88, 84, 86, 83, 82, 85, 76, 71, 70, 73, 80, 79, 81, 78, 77, 96, 87, 90, 0], [0, 63, 62, 74, 72, 61, 64, 66, 69, 68, 65, 49, 55, 54, 53, 56, 58, 60, 59, 57, 40, 44, 46, 45, 51, 47, 43, 42, 41, 50, 52, 48, 67, 0], [], [], [], []]

        return VRPSolution(problem, None, None, routes)
