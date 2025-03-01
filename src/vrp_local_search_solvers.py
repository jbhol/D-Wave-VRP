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

from vrp_solvers import VRPSolver


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
        route_weights = [0 for _ in routes]

        for vehicle_id, route in enumerate(routes):
            for node in route:
                route_weights[vehicle_id] += weights[node]
        
        return route_weights

    def resequence_one_node(self, routes, resequence_node, only_one_const, solver_type):
        costs = self.problem.costs
        capacities = self.problem.capacities
        weights = self.problem.weights

        cur_routes = copy.deepcopy(routes)
        cur_cost = self.compute_cost(cur_routes)
        cur_route_weights = self.compute_weights(cur_routes)

        old_route, old_position = None, None

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

                    old_route, old_position = vehicle_id, i
        
        # find a better solution
        best_routes = copy.deepcopy(cur_routes)
        best_cost = cur_cost

        new_route, new_position = None, None

        for vehicle_id, route in enumerate(tmp_routes):
            if cur_route_weights[vehicle_id] + weights[resequence_node] > self.problem.capacity:
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

            sample = QiskitSolvers.solve_qubo(qubo, solver_type=solver_type)

            for index, value in sample.items():
                if value == 1:
                    (prev_id, prev_node), (next_id, next_node) = index
                    inserted_cost = costs[prev_node, resequence_node] + costs[resequence_node, next_node] - costs[prev_node, next_node]
                    if best_cost > cur_cost - removed_cost + inserted_cost:
                        best_cost = cur_cost - removed_cost + inserted_cost
                        best_routes = copy.deepcopy(tmp_routes)
                        best_routes[vehicle_id].insert(next_id, resequence_node)

                        new_route, new_position = vehicle_id, next_id

            flag = False
            if flag:
                print(vehicle_id)
                print(sample)
                print(resequence_node)
                print(best_routes)
                print(costs_dict)

        #fix for rounding issue
        if old_route == new_route and old_position == new_position:
            new_route, new_position = None, None

        return (best_routes, best_cost), (old_route, old_position, new_route, new_position)

class TabuMove:
    def __init__(self, n, node, old_route, old_position, new_route, new_position):
        self.node = node

        # '''
        self.old_route = old_route
        self.old_position = old_position
        # '''

        # '''
        self.new_route = new_route
        self.new_position = new_position
        # '''
        
        self.count = random.randint(int(0.4*n),int(0.6*n))

    def is_equal(self, move):
        if not self.node == move.node:
            return False
        
        # '''
        if not (self.old_route == move.old_route and self.old_position == move.old_position):
            return False
        # '''
        
        if not (self.new_route == move.new_route and self.new_position == move.new_position):
            return False
        return True

class Neighbor:
    def __init__(self, routes, move, cost):
        self.move = move
        self.routes = routes
        self.cost = cost
            
class TabuLocalSearchSolver(LocalSearchSolver):
    def __init__(self, problem):
        super().__init__(problem)
        
        self.tabu = []
    
    def solve(self, only_one_const, order_const, solver_type = 'cpu'):
        dests = self.problem.dests
        capacities = self.problem.capacities
        vehicles = len(self.problem.capacities)
        costs = self.problem.costs
        weights = self.problem.weights

        self.init_solution()
        solver = ClarkWright(self.problem)
        solution = solver.solve()
        #clusters = [arr[1:-1] for arr in solution.solution]
        self.cur_solution = solution.solution

        best_solution = self.cur_solution

        best_cost = self.calculate_neighbor_cost(self.cur_solution)
        print('Starting solution', self.cur_solution)
        print('Starting solution cost =', best_cost)

        ready_to_stop = False

        counter = 0
        counter_of_last_best = 0
        optimized_routes = list()       #cache for quantum resequenced routes
        cur_routes = self.cur_solution
        nodes_list = [d for d in dests]
        last_cost = 0

        while not ready_to_stop:
            np.random.shuffle(nodes_list)

            best_neighbor = None
            best_neighbor_cost = 1e9
            neighbors = []
            inf_neighbors = []
            new_route_equal_old_route = 0
            for node in nodes_list:
                (new_routes, cost), move = self.resequence_one_node(cur_routes, node, only_one_const, solver_type)
                old_route, old_position, new_route, new_position = move
                tabu_move = TabuMove(len(nodes_list), node, old_route, old_position, new_route, new_position)
                neighbor = Neighbor(new_routes, tabu_move, cost)
                
                if new_route == None and new_position == None:
                    new_route_equal_old_route += 1

                #Check Feasiblity
                feasible = True
                for route in new_routes:
                    power_, time_ = self.calc_power_and_time(route)
                    if power_ > self.problem.battery:
                        feasible = False

                if feasible == True:
                    neighbors.append(neighbor)
                else:
                    inf_neighbors.append(neighbor)

            current_best_neighbor = []          #holds the best neighbor found by the local search (might be tabu)
            current_best_cost = 1e9
            selected_neighbor = []              #holds the best non-tabu feasible move
            selected_neighbor_cost = 1e9

            for n in neighbors:
                cost = self.calculate_neighbor_cost(n.routes)
                if cost < selected_neighbor_cost:
                    #keep track of overall best neighbor
                    if cost < current_best_cost:
                        current_best_neighbor = n
                        current_best_neighbor.cost = cost
                        current_best_cost = cost

                    #check if candidate is tabu
                    if self.is_tabu(self.tabu, n) is False:
                        #keep track of best non-tabu neighbor
                        selected_neighbor = n
                        selected_neighbor.cost = cost
                        selected_neighbor_cost = cost
                    else:
                        print('FOUND A TABU MOVE')

            if selected_neighbor == []:
                #Fix Routes
                n = self.fix_infeasible_routes(inf_neighbors, cur_routes)
                #keep track of overall best neighbor
                if cost < current_best_cost:
                    current_best_neighbor = n
                    current_best_cost = cost  
                #check if candidate is tabu
                if self.is_tabu(self.tabu, n) is False:
                    #keep track of best non-tabu neighbor
                    selected_neighbor = n
                    selected_neighbor_cost = cost
                              
            #Aspiration
            aspiration = False            
            if current_best_cost < best_cost:
                counter_of_last_best = counter
                best_cost = current_best_cost
                self.cur_solution = current_best_neighbor.routes
                cur_routes = self.cur_solution
                print(counter, '- New best found with cost =', best_cost)
                aspiration = True                   
                if self.is_tabu(self.tabu, current_best_neighbor) == False:
                    self.tabu.append(copy.deepcopy(current_best_neighbor.move))

            if aspiration == False:
                print(counter, ' - No new best, snc =', selected_neighbor.cost, ', bc =', best_cost)
                self.tabu.append(copy.deepcopy(selected_neighbor.move))
                cur_routes = selected_neighbor.routes

            # 17. Sparse Quantum Resequencing
            if counter - counter_of_last_best == -1:      
                print('Quantum Go', counter) 
                self.tabu = []             
                clusters = copy.deepcopy(self.cur_solution) 
                routes = list()
                for cluster in clusters:
                    if len(cluster) > 4:
                        found = False
                        for rte in optimized_routes: #check if we have already sequenced this route
                            if self.check_elements_match(cluster, rte):
                                route = rte
                                found = True
                        if found == False:
                            new_problem = VRPProblem(self.problem.sources, costs, [capacities[0]], cluster, weights, first_source = True, last_source = True)
                            qubo = new_problem.get_full_qubo(only_one_const, order_const)
                            print('0 =', cluster)
                            sample = QiskitSolvers.solve_qubo(qubo, solver_type=solver_type)   
                            route = [None] * len(cluster)
                            for index, value in sample.items():
                                if value == 1:
                                    spot, node = index
                                    route[spot] = node
                       
                            print('1 =', route)
                            optimized_routes.append(copy.deepcopy(route))
                    else:
                        route = cluster
                    routes.append(route)

                cur_routes = routes #next loop set

                #Check Feasiblity
                feasible = True
                for route in routes:
                    power_, time_ = self.calc_power_and_time(route)
                    if power_ > self.problem.battery:
                        feasible = False

                cost = self.calculate_neighbor_cost(routes)
                if cost < best_cost and feasible == True:
                    self.cur_solution = copy.deepcopy(routes)
                    best_cost = cost
                    counter_of_last_best = counter
                    selected_neighbor = routes
                    selected_neighbor_cost = cost
                    print('quantum found total_cost =', best_cost)
           
            # 18. update tabu list
            for move in self.tabu:
                move.count -= 1
                if move.count == 0:
                    self.tabu.remove(move)    

            counter += 1

            if new_route_equal_old_route >= len(neighbors) - 1 and last_cost == selected_neighbor.cost:
                shuffled_solutions = self.route_shuffle(1)
                cur_routes = shuffled_solutions[1]
                print('Restart :', cur_routes)

            last_cost = selected_neighbor.cost

            if counter - counter_of_last_best > 1000:
                print('Best solution was found on counter =', counter_of_last_best)
                ready_to_stop = True

        # clean the route
        for route in self.cur_solution:
            if len(route) == 2:
                route.clear()

        solution = VRPSolution(self.problem, solution=self.cur_solution)
        return solution
    

    def is_tabu(self, tabu, n):
        is_tabu = False                            
        for move in tabu:
            if n.move.is_equal(move):
                is_tabu = True
                break
        return is_tabu
    
    # Calculates the total cost of all the routes in a neighbor
    def calculate_neighbor_cost(self, routes):
        total_power = 0
        total_time = 0
        for route in routes:
            route_power, route_time = self.calc_power_and_time(route)
            total_power += route_power
            total_time += route_time      
        return total_time

    # Calculates the total cost of a given route.
    def calculate_route_cost(self, route):
        route_power, route_time = self.calc_power_and_time(route)
        return route_time
    
    # Returns total time of solution (minutes)
    def total_power_and_time(self, routes):
        time_cost = 0
        power_cost = 0
        for route in routes:
            if route == []:
                continue
            power, time = self.calc_power_and_time(route)
            power_cost += power
            time_cost += time
        return power_cost, time_cost *60   
    
    def sum_cap(self, route):
        sum_cap = 0
        for node in route:
            sum_cap += self.problem.weights[node]
        return sum_cap

    def calc_power_and_time(self, route):
        sum_power = 0
        sum_time = 0

        if len(route) == 2:
            return 0, 0

        for i, node in enumerate(route[:-1]):
            dist = self.problem.costs[node][route[i+1]]
            cap = self.sum_cap(route[i+1:])
            top = dist * (self.problem.droneweight + cap)
            bottom = (370 * self.problem.lifttodragratio * self.problem.conversionefficiency * (self.problem.maxrateofpower - self.problem.powerconsumption))
            time = top / bottom
            sum_power += self.problem.maxrateofpower * time
            sum_power += self.problem.extrapower
            sum_time += time + self.problem.extratime

        return sum_power, sum_time
    
    def fix_infeasible_routes(self, inf_neighbors, cur_routes):
        """
        Fix infeasible routes by trying different repair strategies.
        Returns a Neighbor object containing the fixed routes and the tabu move.
        """
        if not inf_neighbors:
            # If no infeasible neighbors, create a minimal feasible solution
            minimal_routes = self.create_minimal_feasible_solution()
            cost = self.compute_cost(minimal_routes)
            # Create a dummy tabu move since we don't have a real one
            dummy_move = TabuMove(0, 0, 0, 0, 0, 0)
            return Neighbor(minimal_routes, dummy_move, cost)
        
        # Sort infeasible neighbors by cost (best potential solutions first)
        inf_neighbors.sort(key=lambda n: n.cost)
        
        # Try different repair strategies on the most promising infeasible neighbors
        for neighbor in inf_neighbors[:min(5, len(inf_neighbors))]:
            candidate_routes = copy.deepcopy(neighbor.routes)
            original_move = copy.deepcopy(neighbor.tabu_move)
            
            # Strategy 1: Remove nodes from infeasible routes and create new routes for them
            fixed_routes = self.repair_by_removal(candidate_routes)
            if fixed_routes:
                cost = self.compute_cost(fixed_routes)
                return Neighbor(fixed_routes, original_move, cost)
                
            # Strategy 2: Try to relocate nodes between routes
            fixed_routes = self.repair_by_relocation(candidate_routes)
            if fixed_routes:
                cost = self.compute_cost(fixed_routes)
                return Neighbor(fixed_routes, original_move, cost)
                
            # Strategy 3: Try route merging and splitting
            fixed_routes = self.repair_by_restructuring(candidate_routes)
            if fixed_routes:
                cost = self.compute_cost(fixed_routes)
                return Neighbor(fixed_routes, original_move, cost)
        
        # If all repair strategies fail, fall back to the best feasible solution so far
        cost = self.compute_cost(self.cur_solution)
        # Use the move from the best infeasible neighbor to track this decision
        return Neighbor(copy.deepcopy(self.cur_solution), inf_neighbors[0].tabu_move, cost)

    def repair_by_removal(self, routes):
        """Remove nodes from infeasible routes to make them feasible."""
        fixed_routes = copy.deepcopy(routes)
        removed_nodes = []
        
        # Identify infeasible routes and remove nodes from them
        for i, route in enumerate(fixed_routes):
            power, time = self.calc_power_and_time(route)
            
            while power > self.problem.battery and len(route) > 2:
                # Find the node that consumes the most power
                max_power_contribution = 0
                max_power_node_idx = -1
                
                for j in range(1, len(route) - 1):
                    # Calculate power without this node
                    test_route = route[:j] + route[j+1:]
                    new_power, _ = self.calc_power_and_time(test_route)
                    power_contribution = power - new_power
                    
                    if power_contribution > max_power_contribution:
                        max_power_contribution = power_contribution
                        max_power_node_idx = j
                
                if max_power_node_idx != -1:
                    removed_nodes.append(route[max_power_node_idx])
                    route.pop(max_power_node_idx)
                    power, time = self.calc_power_and_time(route)
                else:
                    break
        
        # Create new routes for removed nodes
        if removed_nodes:
            for node in removed_nodes:
                fixed_routes.append([0, node, 0])  # Create a direct depot -> node -> depot route
        
        # Verify that all routes are now feasible
        all_feasible = True
        for route in fixed_routes:
            if len(route) <= 2:  # Empty route or just depot
                continue
            power, _ = self.calc_power_and_time(route)
            if power > self.problem.battery:
                all_feasible = False
                break
        
        return fixed_routes if all_feasible else None

    def repair_by_relocation(self, routes):
        """Try to relocate nodes between routes to achieve feasibility."""
        fixed_routes = copy.deepcopy(routes)
        
        # Identify infeasible routes
        infeasible_routes = []
        for i, route in enumerate(fixed_routes):
            if len(route) <= 2:  # Skip empty routes
                continue
            power, _ = self.calc_power_and_time(route)
            if power > self.problem.battery:
                infeasible_routes.append(i)
        
        if not infeasible_routes:
            return fixed_routes
        
        # Try relocating nodes from infeasible to feasible routes
        for inf_idx in infeasible_routes:
            inf_route = fixed_routes[inf_idx]
            
            if len(inf_route) <= 2:
                continue
            
            # Try relocating each node in the infeasible route
            for i in range(1, len(inf_route) - 1):
                node = inf_route[i]
                
                # Try inserting into each position of each other route
                for j, target_route in enumerate(fixed_routes):
                    if j == inf_idx:
                        continue
                    
                    for k in range(1, len(target_route)):
                        # Create candidate routes
                        new_source = inf_route[:i] + inf_route[i+1:]
                        new_target = target_route[:k] + [node] + target_route[k:]
                        
                        # Check if both routes would be feasible
                        source_power, _ = self.calc_power_and_time(new_source)
                        target_power, _ = self.calc_power_and_time(new_target)
                        
                        if source_power <= self.problem.battery and target_power <= self.problem.battery:
                            # Apply the relocation
                            fixed_routes[inf_idx] = new_source
                            fixed_routes[j] = new_target
                            
                            # Check if all routes are now feasible
                            all_feasible = True
                            for route in fixed_routes:
                                if len(route) <= 2:
                                    continue
                                power, _ = self.calc_power_and_time(route)
                                if power > self.problem.battery:
                                    all_feasible = False
                                    break
                            
                            if all_feasible:
                                return fixed_routes
        
        return None

    def repair_by_restructuring(self, routes):
        """Try merging and splitting routes to achieve feasibility."""
        # Filter out empty routes
        fixed_routes = [route for route in routes if len(route) > 2]
        
        # If no routes left, return a minimal solution
        if not fixed_routes:
            return [[0, 0]]
        
        # Collect all nodes from all routes
        all_nodes = []
        for route in fixed_routes:
            all_nodes.extend(route[1:-1])  # Skip depot nodes
        
        # Create new routes using greedy insertion
        new_routes = [[0, 0]]  # Start with one empty route
        
        for node in all_nodes:
            best_increase = float('inf')
            best_route_idx = -1
            best_position = -1
            
            # Try inserting the node into each existing route
            for i, route in enumerate(new_routes):
                for j in range(1, len(route)):
                    # Create a candidate route
                    candidate = route[:j] + [node] + route[j:]
                    power, _ = self.calc_power_and_time(candidate)
                    
                    if power <= self.problem.battery:
                        # Calculate the cost increase
                        old_cost = self.compute_cost([route])
                        new_cost = self.compute_cost([candidate])
                        increase = new_cost - old_cost
                        
                        if increase < best_increase:
                            best_increase = increase
                            best_route_idx = i
                            best_position = j
            
            if best_route_idx != -1:
                # Insert the node into the best position
                new_routes[best_route_idx].insert(best_position, node)
            else:
                # Create a new route for this node
                new_routes.append([0, node, 0])
        
        # Verify that all routes are feasible
        all_feasible = True
        for route in new_routes:
            if len(route) <= 2:
                continue
            power, _ = self.calc_power_and_time(route)
            if power > self.problem.battery:
                all_feasible = False
                break
        
        return new_routes if all_feasible else None

    def create_minimal_feasible_solution(self):
        """Create a minimal feasible solution with one node per route if needed."""
        # Get all destinations
        all_nodes = list(self.problem.dests)
        
        # Create one route per node (direct from depot)
        minimal_routes = []
        for node in all_nodes:
            route = [0, node, 0]  # depot -> node -> depot
            power, _ = self.calc_power_and_time(route)
            
            if power <= self.problem.battery:
                minimal_routes.append(route)
            else:
                print(f"Warning: Node {node} cannot be serviced even in isolation!")
                # If a node can't be serviced even alone, this is a fundamental problem
                # We might need to skip it or use a different vehicle
        
        return minimal_routes   

    def route_shuffle(self, num_shuffles=5):
        """
        Performs route shuffling to create different starting solutions for a multi-start approach.
        
        Args:
            num_shuffles (int): Number of different shuffled solutions to generate
            
        Returns:
            list: A list of different route configurations (solutions)
        """
        shuffled_solutions = []
        
        # First, save the original solution
        original_solution = copy.deepcopy(self.cur_solution)
        shuffled_solutions.append(original_solution)
        
        # Get all customer nodes (excluding depots)
        all_nodes = []
        for route in original_solution:
            for node in route[1:-1]:  # Skip depot nodes (0)
                all_nodes.append(node)
        
        # Generate different shuffled solutions
        for _ in range(num_shuffles):
            # Shuffle all nodes
            random.shuffle(all_nodes)
            
            # Create new empty routes
            new_solution = [[0, 0] for _ in range(len(original_solution))]
            
            # Distribute nodes using a bin packing approach
            route_weights = [0] * len(new_solution)
            route_powers = [0] * len(new_solution)
            
            for node in all_nodes:
                # Find the best route to insert this node
                best_route = -1
                best_position = -1
                min_cost_increase = float('inf')
                
                for i, route in enumerate(new_solution):
                    # Skip if the route is at full capacity
                    if route_weights[i] + self.problem.weights[node] > self.problem.capacity:
                        continue
                    
                    # Try inserting at each position
                    for j in range(1, len(route)):
                        # Create candidate route
                        candidate = route[:j] + [node] + route[j:]
                        
                        # Check power constraint
                        power, _ = self.calc_power_and_time(candidate)
                        if power > self.problem.battery:
                            continue
                        
                        # Calculate insertion cost
                        prev_node, next_node = route[j-1], route[j]
                        cost_increase = (self.problem.costs[prev_node, node] + 
                                        self.problem.costs[node, next_node] - 
                                        self.problem.costs[prev_node, next_node])
                        
                        if cost_increase < min_cost_increase:
                            min_cost_increase = cost_increase
                            best_route = i
                            best_position = j
                
                # If no feasible insertion found, create a new route
                if best_route == -1:
                    new_solution.append([0, node, 0])
                    route_weights.append(self.problem.weights[node])
                    power, _ = self.calc_power_and_time([0, node, 0])
                    route_powers.append(power)
                else:
                    # Insert node into best position
                    new_solution[best_route].insert(best_position, node)
                    route_weights[best_route] += self.problem.weights[node]
                    power, _ = self.calc_power_and_time(new_solution[best_route])
                    route_powers[best_route] = power
            
            # Remove empty routes
            new_solution = [route for route in new_solution if len(route) > 2]
            
            # Add this shuffled solution to our collection
            shuffled_solutions.append(new_solution)
        
        return shuffled_solutions


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
            #weight = self.problem.weights[node] % self.problem.capacities[0]
            #if (weight == 0):
            #    weight = 1.5
            #sum_cap += weight
            sum_cap += self.problem.weights[node]
        return sum_cap

    def sum_power(self, route):
        sum_power = 0
        #power to first node
        d = self.problem.costs[0][route[0]]
        cap = self.sum_cap(route)
        pow = self.problem.maxrateofpower * d * (self.problem.droneweight + cap)
        sum_power += pow / (370 * self.problem.lifttodragratio * self.problem.conversionefficiency * (self.problem.maxrateofpower - self.problem.powerconsumption))
        #power for deliveries
        for i, node in enumerate(route[:-1]):
            d = self.problem.costs[node][route[i+1]]
            cap = self.sum_cap(route[i+1:])
            pow = self.problem.maxrateofpower * d * (self.problem.droneweight + cap)
            sum_power += pow / (370 * self.problem.lifttodragratio * self.problem.conversionefficiency * (self.problem.maxrateofpower - self.problem.powerconsumption))
        #power to go back to depot
        d = self.problem.costs[route[-1]][0]
        cap = 0
        pow = self.problem.maxrateofpower * d * (self.problem.droneweight + cap)
        sum_power += pow / (370 * self.problem.lifttodragratio * self.problem.conversionefficiency * (self.problem.maxrateofpower - self.problem.powerconsumption))
        return sum_power
            

    def solve(self):
        problem = self.problem
        num_customers = len(problem.dests)
        nodes = problem.dests
        capacities = problem.capacities
        costs = problem.costs
        batteries = problem.batteries

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
                    if self.sum_cap(link) <= capacities[0] and self.sum_power(link) <= batteries[0]:
                        routes.append(link)
                        node_list.remove(link[0])
                        node_list.remove(link[1])
                        print('\t','Link ', link, ' fulfills criteria a), so it is created as a new route')
                    else:
                        print('\t','Though Link ', link, ' fulfills criteria a), it exceeds maximum load or max power, so skip this link.')
                        
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
                    cond3 = (self.sum_power(routes[i_rt] + [node]) <= batteries[0])

                    if cond1:
                        if cond2:
                            if cond3:
                                print('\t','Link ', link, ' fulfills criteria b), so a new node is added to route ', routes[i_rt], '.')
                                if position == 0:
                                    routes[i_rt].insert(0, node)
                                else:
                                    routes[i_rt].append(node)
                                node_list.remove(node)     
                            else:
                                print('\t','Though Link ', link, ' fulfills criteria b), it exceeds maximum power, so skip this link.')
                                continue                                                   
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
                        cond4 = (self.sum_power(routes[i_route[0]] + routes[i_route[1]]) <= batteries[0])

                        if cond1 and cond2:
                            if cond3 and cond4:
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
                                print('\t','Though Link ', link, ' fulfills criteria c), it exceeds maximum load or max power, so skip this link.')
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