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
        capacity = self.problem.capacity
        battery = self.problem.battery
        weights = self.problem.weights
        solution = self.solution
        vehicle_num = 0

        for vehicle_dests in solution:
            cap = capacity
            for dest in vehicle_dests:
                cap -= weights[dest]
            vehicle_num += 1
            if cap < 0: 
                return False

        for vehicle_dests in solution:
            bat, tim = self.calc_power_and_time(vehicle_dests)
            vehicle_num += 1
            if battery - bat < 0: 
                return False

        dests = self.problem.dests
        answer_dests = [dest for vehicle_dests in solution for dest in vehicle_dests[1:-1]]
        if len(dests) != len(answer_dests):
            return False

        lists_cmp = set(dests) & set(answer_dests)
        if lists_cmp == len(dests):
            return False

        return True

    def sum_cap(self, route):
        sum_cap = 0
        for node in route:
            sum_cap += self.problem.weights[node]
        return sum_cap

    def calc_power_and_time(self, route):
        sum_power = 0
        sum_time = 0

        if len(route) == 0:
            return 0, 0

        #power to first node
        dist = self.problem.costs[0][route[0]]
        cap = self.sum_cap(route)
        top = dist * (self.problem.droneweight + cap)
        bottom = (370 * self.problem.lifttodragratio * self.problem.conversionefficiency * (self.problem.maxrateofpower - self.problem.powerconsumption))
        time = top / bottom
        sum_power += self.problem.maxrateofpower * time
        sum_power += self.problem.extrapower
        sum_time += time + self.problem.extratime

        #power for deliveries
        for i, node in enumerate(route[:-1]):
            dist = self.problem.costs[node][route[i+1]]
            cap = self.sum_cap(route[i+1:])
            top = dist * (self.problem.droneweight + cap)
            bottom = (370 * self.problem.lifttodragratio * self.problem.conversionefficiency * (self.problem.maxrateofpower - self.problem.powerconsumption))
            time = top / bottom
            sum_power += self.problem.maxrateofpower * time
            sum_power += self.problem.extrapower
            sum_time += time + self.problem.extratime

        #power to go back to depot
        dist = self.problem.costs[route[-1]][0]
        cap = 0
        top = dist * (self.problem.droneweight + cap)
        bottom = (370 * self.problem.lifttodragratio * self.problem.conversionefficiency * (self.problem.maxrateofpower - self.problem.powerconsumption))
        time = top / bottom
        sum_power += self.problem.maxrateofpower * time
        sum_power += self.problem.extrapower
        sum_time += time + self.problem.extratime
        return sum_power, sum_time

    # Returns total time of solution (minutes)
    def total_power_and_time(self):
        costs = self.problem.costs
        source = self.problem.source
        solution = self.solution
        time_cost = 0
        power_cost = 0

        for vehicle_dests in solution:
            if vehicle_dests == []:
                continue
            power, time = self.calc_power_and_time(vehicle_dests)
            power_cost += power
            time_cost += time
        #time_cost += self.problem.extratime * (len(vehicle_dests) + 1)
        return power_cost, time_cost *60      


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

