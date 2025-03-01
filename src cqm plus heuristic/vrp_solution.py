# Solution of VRP problem with multi-source. 
# Class can decode solution from solution of QUBO.
# Class provides methods to check and get informations about solution.
class VRPSolution:

    # Parameters :
    # problem - VRPProblem object
    # sample - QUBO solution returned by D-Wave
    # vehicle_limits - maximum number of deliveries that vehicles could serve. Used only
    # to decode solution from QUBO solution. Used only by AveragePartitionSolver.
    # solution - solution in final form : list of the lists of vehicles paths. Used to
    # create VRPSolution other way than from QUBO solution. 
    # It is needed to provide sample or solution parameter.
    def __init__(self, problem, sample = None, vehicle_limits = None, solution = None, step = 0):
        self.problem = problem
        self.step = step
        
        if solution != None:
            self.solution = solution
        else:
            if vehicle_limits == None:
                dests = len(self.problem.dests)
                vehicles = len(self.problem.capacities)
                vehicle_limits = [dests for _ in range(vehicles)]

            result = list()
            vehicle_result = list()
            step = 0
            vehicle = 0

            # Decoding solution from qubo sample.
            for (s, dest) in sample:
                if sample[(s, dest)] == 1:
                    if dest != 0:
                        vehicle_result.append(dest)
                    step += 1
                    if vehicle_limits[vehicle] == step:
                        result.append(vehicle_result)
                        step = 0
                        vehicle += 1
                        vehicle_result = list()
                        if len(vehicle_limits) <= vehicle:
                            break

            # Adding first and last magazine.
            for l in result:
                if len(l) != 0:
                    if problem.first_source:
                        l.insert(0, problem.in_nearest_sources[l[0]])
                    if problem.last_source:
                        l.append(problem.out_nearest_sources[l[len(l) - 1]])

            self.solution = result

    # Checks if solution is correct.
    def check(self):
        capacities = self.problem.capacities
        weights = self.problem.weights
        solution = self.solution
        vehicle_num = 0

        for vehicle_dests in solution:
            cap = capacities[vehicle_num]
            for dest in vehicle_dests:
                cap -= weights[dest]
            vehicle_num += 1
            if cap < 0: 
                return False

        dests = self.problem.dests
        answer_dests = [dest for vehicle_dests in solution for dest in vehicle_dests[1:-1]]
        if len(dests) != len(answer_dests):
            return False

        lists_cmp = set(dests) & set(answer_dests)
        if lists_cmp == len(dests):
            return False

        return True

    # Returns total cost of solution.
    def total_cost(self):
        costs = self.problem.costs
        source = self.problem.source
        solution = self.solution
        cost = 0

        for vehicle_dests in solution:
            if vehicle_dests == []:
                continue
            prev = vehicle_dests[0]
            for dest in vehicle_dests[1:]:
                cost += costs[prev][dest]
                prev = dest
            cost += costs[prev][source]

        return cost

    # Returns list of sums of weights for every vehicle.
    def all_weights(self):
        weights = self.problem.weights
        result = list()

        for vehicle_dests in self.solution:
            weight = 0
            for dest in vehicle_dests:
                weight += weights[dest]
            result.append(weight)

        return result
    
    # Returns total time for every vehicle route.
    def total_time(self):
        costs = self.problem.costs
        timeIntervals = self.problem.time_intervals
        services = self.problem.services
        result = list()  # List to store total time for each vehicle

        for vehicle_dests in self.solution:
            totalTime = 0
            # vehicle_dests = [vehicle_dests[0]] + sorted(vehicle_dests[1:-1], key=lambda node: timeIntervals[str(node)][0]) + [vehicle_dests[-1]]
            # vehicle_dests = [0] + vehicle_dests + [0]
            for i in range(len(vehicle_dests) - 1):
                prevNode = vehicle_dests[i]
                currentNode = vehicle_dests[i + 1]
                travel_time = costs[prevNode][currentNode]
                readyTime = timeIntervals[str(currentNode)][0]
                dueTime = timeIntervals[str(currentNode)][1]
                
                
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
            
            result.append(totalTime)
        return result


       
        
        
       

    # Prints description of solution.
    def description(self):
        costs = self.problem.costs
        solution = self.solution

        vehicle_num = 0
        for vehicle_dests in solution:
            cost = 0

            print('Vehicle number ', vehicle_num, ' : ')

            if len(vehicle_dests) == 0:
                print('    Vehicle is not used.')
                continue

            print('    Startpoint : ', vehicle_dests[0])

            dests_num = 1
            prev = vehicle_dests[0]
            for dest in vehicle_dests[1:len(vehicle_dests) - 1]:
                cost += costs[prev][dest]
                print('    Destination number ', dests_num, ' : ', dest, '.')
                dests_num += 1
                prev = dest

            endpoint = vehicle_dests[len(vehicle_dests) - 1]
            cost += costs[prev][endpoint]
            print('    Endpoint : ', endpoint, '.')

            print('')
            print('    Total cost of vehicle : ', cost)

            vehicle_num += 1