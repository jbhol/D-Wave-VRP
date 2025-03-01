import itertools
import argparse
from tqdm import tqdm
from math import ceil, sqrt
from typing import List, Dict, Set, Optional, Tuple, Union
from collections import defaultdict, Counter

class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def distance_from(self, other: "Point") -> int:
        return ceil(sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2))


class Obj:
    def __init__(self, id: int):
        self._id = id

    def __repr__(self):
        return f"{self.__class__.__name__} {self._id}"

    @property
    def id(self):
        return self._id


class Product(Obj):
    def __init__(self, id: int, weight: int):
        super().__init__(id)
        self._weight = weight

    @property
    def weight(self):
        return self._weight


class PositionObj(Obj):
    def __init__(self, id: int, position: Point):
        super().__init__(id)
        self._position = position

    def __repr__(self):
        return f"{self.__class__.__name__} {self._id} at {self._position}"

    @property
    def position(self):
        return self._position

    def distance_from(self, _other: "PositionObj") -> int:
        return self._position.distance_from(_other.position)

    def find_nearest_object(
        self, _others: List["PositionObj"]
    ) -> Tuple["PositionObj", int]:
        nearest = _others[0]
        distance = self.distance_from(nearest)
        for other in _others[1:]:
            d = self.distance_from(other)
            if d < distance:
                nearest, distance = other, d
        return nearest, distance


class ProductHolder(PositionObj):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._products: Dict[Product, int] = {}

    @property
    def products(self):
        return self._products

    def add_products(self, products: Dict[Product, int]):
        for product, quantity in products.items():
            if product in self._products:
                self._products[product] += quantity
            else:
                self._products[product] = quantity

    def remove_products(self, products: Dict[Product, int]):
        for product, quantity in products.items():
            self._products[product] -= quantity
            if self._products[product] == 0:
                self._products.pop(product)

    @property
    def product_weight(self):
        return sum(p.weight * q for p, q in self._products.items())

    def remove_all_products(self):
        self._products = {}


class Command:
    def __init__(
        self,
        drone: ProductHolder,
        destination: ProductHolder,
        product: Product,
        quantity: int,
    ):
        self.drone = drone
        self.destination = destination
        self.product = product
        self.quantity = quantity

#     def __repr__(self):
#         return self.to_string()

    @property
    def str_command(self):
        raise NotImplementedError
        
    def __repr__(self):
        return f"Command({self.drone} -> {self.destination}: {self.product} * {self.quantity})"

    def to_string(self):
        if isinstance(self.destination, Warehouse):
            str_command = "L"
        elif isinstance(self.destination, Order):
            str_command = "D"
        else:
            raise ValueError("Destination must be a warehouse or order.")
        return f"{self.drone.id} {str_command} {self.destination.id} {self.product.id} {self.quantity}"

#     def to_string(self):
#         return f"{self.drone.id} {self.str_command} {self.destination.id} {self.product.id} {self.quantity}"


class Load(Command):
    str_command = "L"


class Deliver(Command):
    str_command = "D"


class Order(ProductHolder):
    def __init__(self, id: int, position: Point, products: Dict[Product, int]):
        super().__init__(id, position)
        self._products = products

    def is_complete(self):
        return not self._products


class Drone(ProductHolder):
    def __init__(self, id: int, position: Point, max_capacity: int):
        super().__init__(id, position)
        self.max_capacity = max_capacity
        self.is_busy = False
        self.stops_being_busy_at = 0

    def set_position(self, position: Point):
        self._position = position

    def update_state(self, current_time):
        if self.is_busy and current_time == self.stops_being_busy_at:
            self.is_busy = False

    @property
    def current_capacity(self) -> int:
        return self.max_capacity - self.product_weight

    def get_products_that_can_fit_out_of(
        self, products: Dict[Product, int]
    ) -> Dict[Product, int]:
        can_carry = defaultdict(int)
        products = sorted(
            [(p, q) for p, q in products.items()], key=lambda x: -x[0].weight
        )
        capacity = self.current_capacity
        for product, quantity in products:
            if not quantity:
                continue

            weight = product.weight
            if weight > capacity:
                continue

            for _ in range(quantity):
                can_carry[product] += 1
                capacity -= weight
                if weight > capacity:
                    break

        return can_carry


class Warehouse(ProductHolder):
    def __init__(self, id: int, position: Point, products=Dict[Product, int]):
        super().__init__(id, position)
        self._products = products

    def get_products_that_are_in_stock_out_of(
        self, products=Dict[Product, int]
    ) -> Dict[Product, int]:
        return {p: min(q, self._products.get(p, 0)) for p, q in products.items()}


class Shipment:
    def __init__(self, drone: Drone, order: Order, warehouse: Warehouse):
        self.drone = drone
        self.order = order
        self.warehouse = warehouse

        self.products = drone.get_products_that_can_fit_out_of(
            warehouse.get_products_that_are_in_stock_out_of(order.products)
        )
        self.product_weight = sum(p.weight * q for p, q in self.products.items())

    def __repr__(self):
        return f"Shipment({self.product_types()} -> {self.order})"

    def has_products(self) -> bool:
        return self.product_weight > 0

    def product_types(self) -> Set[Product]:
        return set(self.products)

    def number_of_product_types(self) -> int:
        return len(self.products)

    def percentage_of_order(self) -> float:
        order_weight = self.order.product_weight
        if not order_weight:
            return 0
        return self.product_weight / self.order.product_weight

    def load(self):
        self.warehouse.remove_products(self.products)
        self.drone.add_products(self.products)
        self.order.remove_products(self.products)
        load = []
        deliver = []
        for product, quantity in self.products.items():
            load.append(Load(self.drone, self.warehouse, product, quantity))
            deliver.append(Deliver(self.drone, self.order, product, quantity))
        return load, deliver


class Simulation:
    def __init__(
        self,
        max_time: int,
        drones: List[Drone],
        warehouses: List[Warehouse],
        orders: List[Order],
        last_point: Point
    ):
        self.max_time = max_time
        self.drones = drones
        self.warehouses = warehouses
        self.orders = orders
        self.last_point = last_point

        products = {x for x in orders + warehouses for x in x.products}
        self.min_product_weight = min(x.weight for x in products)

        self.commands: List[Command] = []
        self.total_score: int = 0
        self.current_time: int = 0
        self.completed_orders: int = 0
        self.order_to_delivery_time: Dict[Order, list] = defaultdict(list)

    def run(self):
        bar = tqdm(total=len(self.orders))
        while self.current_time < self.max_time:
            if self.all_orders_complete():
                break

            self.do_turn()
            self.current_time += 1
            bar.update(self.completed_orders - bar.n)

        bar.close()

    def all_orders_complete(self):
        return self.completed_orders == len(self.orders)

    def do_turn(self):
        for drone in self.drones:
            drone.update_state(self.current_time)

        for drone in self.drones:
            if not drone.is_busy:
                self.send_drone(drone)

    def send_drone(self, drone):
        drone.remove_all_products()
        optimal_shipment_list = self.get_optimal_shipment_list(drone)
        if optimal_shipment_list:
            _, total_actual_score, total_time, completed_orders, deliver_time = self.get_shipment_list_analysis(
                optimal_shipment_list
            )

            load_commands = []
            deliver_commands = []
            for shipment, time in zip(optimal_shipment_list, deliver_time):
                load, deliver = shipment.load()
                load_commands += load
                deliver_commands += deliver
                self.order_to_delivery_time[shipment.order].append(time)

            self.commands += load_commands + deliver_commands

            drone.is_busy = True
            drone.stops_being_busy_at = self.current_time + total_time
            drone.set_position(optimal_shipment_list[-1].order.position)
            drone.remove_all_products()

            if total_actual_score > 0:
                self.completed_orders += completed_orders
                self.total_score += ceil(total_actual_score)

    def get_optimal_shipment_list(
        self, drone: Drone, branching_factor: int = 30
    ) -> Optional[List[Shipment]]:
        initial_list = self.get_initial_shipments_list_for_drone(drone)
        if not initial_list:
            return

        possibilities = [[x] for x in initial_list[:branching_factor]]
        loop = True
        roots = []
        while loop:
            loop = False

            roots += possibilities
            new_possibilities = []
            for p in possibilities:
                next_list = self.get_next_shipments_list(p)
                if next_list:
                    loop = True
                    new_possibilities += [p + [n] for n in next_list[:branching_factor]]

            possibilities = new_possibilities
            if len(roots) > 40:
                break

        possibilities += roots

        optimal = possibilities[0]
        score = self.get_shipment_list_analysis(optimal)[0]
        for p in possibilities[1:]:
            s = self.get_shipment_list_analysis(p)[0]
            if s > score:
                optimal, score = p, s

        return optimal

    def get_shipment_list_analysis(self, shipment_list: List[Shipment]):
        first_position = Point(0, 0)  
        max_dist=first_position.distance_from(self.last_point)
        drone = shipment_list[0].drone
        warehouse = shipment_list[0].warehouse
        d1 = drone.distance_from(warehouse)

        product_types = len({x for x in shipment_list for x in x.product_types()})
        total_load_time = d1 + product_types

        total_time = total_load_time
        completed_orders = 0
        total_scaled_score = 0
        total_actual_score = 0
        previous_position = warehouse
        deliver_time = []
        for shipment in shipment_list:
            total_time += (
                shipment.order.distance_from(previous_position)
                + shipment.number_of_product_types()                
            )
            p = shipment.percentage_of_order()
            y=((max_dist-first_position.distance_from(shipment.order.position)) / max_dist)
            y2=((max_dist-self.last_point.distance_from(shipment.order.position)) / max_dist)
            x=self.current_time
            T=total_time
            pdist1 = y+min(1,x/(0.55*T))*(1-y)                            
#             pdist3 = y2+(1-min(1,x/(0.55*T)))*(1-y2)                            
            y=((max_dist-first_position.distance_from(previous_position.position)) / max_dist)
            y2=((max_dist-self.last_point.distance_from(previous_position.position)) / max_dist)
            pdist2 = y+min(1,x/(0.55*T))*(1-y)                            
#             pdist4 = y2+(1-min(1,x/(0.55*T)))*(1-y2)                            
            scaled_score = pdist1 * pdist2 * p * self.score_for_order_completed_at(
                self.current_time + total_time + 1
            )
            deliver_time.append(self.current_time + total_time + 1)
            total_scaled_score += scaled_score
            actual_score = 0
            if p == 1:
                completed_orders += 1
                actual_score = self.score_for_order_completed_at(
                    self.current_time + total_time + 1
                )
            total_actual_score += actual_score

            previous_position = shipment.order

        total_scaled_score /= total_time
        return (
            total_scaled_score,
            total_actual_score,
            total_time,
            completed_orders,
            deliver_time,
        )

    def get_next_shipments_list(
        self, current_shipment_list: List[Shipment]
    ) -> List[Shipment]:
        drone = current_shipment_list[0].drone

        product_weight = sum(sh.product_weight for sh in current_shipment_list)
        if product_weight + self.min_product_weight > drone.max_capacity:
            return []

        drone.remove_all_products()
        warehouse = current_shipment_list[0].warehouse

        used_orders = set()
        for shipment in current_shipment_list:
            used_orders.add(shipment.order)
            drone.add_products(shipment.products)
            warehouse.remove_products(shipment.products)

        next_shipments_list = []
        for order in self.orders:
            if order in used_orders or order.is_complete():
                continue

            sh = Shipment(drone, order, warehouse)
            if not sh.has_products():
                continue

            score = self.get_score_for_last_shipment(sh, current_shipment_list)
            next_shipments_list.append((sh, score))

        next_shipments_list = [
            x for x, _ in sorted(next_shipments_list, key=lambda x: -x[1])
        ]

        drone.remove_all_products()
        for shipment in current_shipment_list:
            warehouse.add_products(shipment.products)

        return next_shipments_list

    @staticmethod
    def get_score_for_last_shipment(
        last_shipment: Shipment, shipment_list: List[Shipment]
    ) -> float:
        d = shipment_list[-1].order.distance_from(last_shipment.order)
        m = last_shipment.number_of_product_types()
        current_product_types = {x for x in shipment_list for x in x.product_types()}
        last_product_types = last_shipment.product_types()
        n = len(last_product_types - current_product_types)
        p = last_shipment.percentage_of_order()
        turns = d + m + n
        return p / turns

    def get_initial_shipments_list_for_drone(self, drone: Drone) -> List[Shipment]:
        drone.remove_all_products()

        shipment_list = []
        for order in self.orders:
            if order.is_complete():
                continue

            for warehouse in self.warehouses:
                sh = Shipment(drone, order, warehouse)
                if not sh.has_products():
                    continue

                d1 = drone.distance_from(warehouse)
                d2 = warehouse.distance_from(order)
                turns = d1 + d2 + sh.number_of_product_types() * 2
                p = sh.percentage_of_order()
                scaled_score = p / turns

                if self.current_time + turns > self.max_time:
                    continue

                shipment_list.append((sh, scaled_score))

        return [sh for sh, _ in sorted(shipment_list, key=lambda x: -x[1])]

    def score_for_order_completed_at(self, time: int) -> int:
        return ceil((self.max_time - time) / self.max_time * 100)
def read_file(input_file):
    with open(input_file) as f:
        num_rows, num_columns, num_drones, max_time, max_cargo = map(
            int, f.readline().split(" ")
        )

        # products
        num_products = int(f.readline())
        product_weights = list(map(int, f.readline().split(" ")))
        assert num_products == len(product_weights)
        products = [Product(id=i, weight=w) for i, w in enumerate(product_weights)]

        # warehouses
        num_warehouses = int(f.readline())
        wh_list = []
        for i in range(num_warehouses):
            x, y = map(int, f.readline().split(" "))
            num_products_in_wh = list(map(int, f.readline().split(" ")))
            assert num_products == len(num_products_in_wh)
            wh_products = {p: n for p, n in zip(products, num_products_in_wh)}
            wh = Warehouse(id=i, position=Point(x, y), products=wh_products)
            wh_list.append(wh)

        # order info
        order_list = []
        num_orders = int(f.readline())
        for i in range(num_orders):
            x, y = map(int, f.readline().split(" "))
            num_products_in_order = int(f.readline())
            order_products = list(map(int, f.readline().split(" ")))
            assert num_products_in_order == len(order_products)
            order_products = [products[x] for x in order_products]
            order = Order(
                id=i, position=Point(x, y), products=dict(Counter(order_products))
            )
            order_list.append(order)

    return num_rows, num_columns, num_drones, max_time, max_cargo, wh_list, order_list


def simulate(input_file):
    num_rows, num_columns, num_drones, max_time, max_cargo, wh_list, order_list = read_file(input_file)

    drones = []
    first_wh = wh_list[0]
    for i in range(num_drones):
        drones.append(Drone(id=i, position=first_wh.position, max_capacity=max_cargo))

    simulation = Simulation(
        max_time=max_time, drones=drones, warehouses=wh_list, orders=order_list, last_point=Point(num_rows-1, num_columns-1)
    )
    simulation.run()

    with open("submission.csv", "w") as w:
        w.write(str(len(simulation.commands)) + "\n")
        for c in simulation.commands:
            w.write(c.to_string() + "\n")

def check(input_file, submission):
    _, _, num_drones, max_time, max_cargo, wh_list, order_list = read_file(input_file)

    with open(submission, "r") as f:
        _f = f.readlines()
        num_commands = int(_f[0])
        commands = [x.rstrip("\n") for x in _f[1:]]
        assert num_commands == len(commands)

    score = 0
    try:
        score = calculate_score(num_drones, max_time, max_cargo, wh_list, order_list, commands)
    except Exception as e:
        print(e)

    print(f"Total score = {score}.")


def calculate_score(num_drones, max_time, max_cargo, wh_list, order_list, commands):
    warehouses: Dict[int, Warehouse] = {w.id: w for w in wh_list}
    orders: Dict[int, Order] = {x.id: x for x in order_list}
    drones: Dict[int, Drone] = {}
    for i in range(num_drones):
        drones[i] = Drone(id=i, position=warehouses[0].position, max_capacity=max_cargo)
    products: Dict[int, Product] = {
        x.id: x for x in order_list + wh_list for x in x.products
    }
    drone_to_delivery_time: Dict[Drone, int] = defaultdict(int)
    order_to_delivery_time: Dict[Order, list] = defaultdict(list)
    score = 0
    deliv_max = 0
    for i, command in enumerate(commands):
        drone_id, str_command, destination_id, product_id, quantity = command.split(" ")

        drone_id = int(drone_id)
        destination_id = int(destination_id)
        product_id = int(product_id)
        quantity = int(quantity)

        drone = drones[drone_id]
        product = products[product_id]
        basket = {product: quantity}

        if str_command == "L":
            warehouse = warehouses[destination_id]

            if warehouse.products.get(product, 0) < quantity:
                raise ValueError(f"Command {i}: {warehouse} have not enough {product}.")
            warehouse.remove_products(basket)

            drone.add_products(basket)
            if drone.product_weight > max_cargo:
                raise ValueError(f"Command {i}: {drone} overloaded.")

            drone_to_delivery_time[drone] += drone.distance_from(warehouse) + 1
            drone.set_position(warehouse.position)

        elif str_command == "D":
            order = orders[destination_id]
            if order.is_complete():
                raise ValueError(
                    f"Command {i}: the {order} is closed, nothing can be delivered there."
                )

            if drone.products.get(product, 0) < quantity:
                raise ValueError(f"Command {i}: {drone} have not enough {product}.")
            drone.remove_products(basket)

            if order.products.get(product, 0) < quantity:
                raise ValueError(f"Command {i}: Too many {product} for {order}.")
            order.remove_products(basket)

            drone_to_delivery_time[drone] += drone.distance_from(order) + 1
            drone.set_position(order.position)
            order_to_delivery_time[order].append(drone_to_delivery_time[drone])
            
            if order.is_complete():
                delivery_time = max(order_to_delivery_time[order])
                if delivery_time > deliv_max:
                    deliv_max=delivery_time
                if delivery_time < max_time:                    
                    score += ceil(100 * (max_time - delivery_time) / max_time)
                else:
                    raise ValueError(f"Command {i}: Run out of time.")
        else:
            raise ValueError(f"Command {i}: Unknown command {str_command}.")
    print(str(deliv_max))
    return score


def write_submission(commands: List[Command], out_file="submission.csv"):
    with open(out_file, "w") as w:
        w.write(str(len(commands)) + "\n")
        for c in commands:
            w.write(c.to_string() + "\n")


def read_submission(input_file, submission_file):
    _, _, num_drones, max_time, max_cargo, wh_list, order_list = read_file(input_file)

    with open(submission_file, "r") as f:
        _f = f.readlines()
        num_commands = int(_f[0])
        str_commands = [x.rstrip("\n") for x in _f[1:]]
        assert num_commands == len(str_commands)

    warehouses: Dict[int, Warehouse] = {w.id: w for w in wh_list}
    orders: Dict[int, Order] = {x.id: x for x in order_list}
    drones: Dict[int, Drone] = {}
    for i in range(num_drones):
        drones[i] = Drone(id=i, position=warehouses[0].position, max_capacity=max_cargo)
    products: Dict[int, Product] = {
        x.id: x for x in itertools.chain(order_list, wh_list) for x in x.products
    }

    commands = []

    for s in str_commands:
        drone_id, str_command, destination_id, product_id, quantity = s.split(" ")

        drone_id = int(drone_id)
        destination_id = int(destination_id)
        product_id = int(product_id)
        quantity = int(quantity)

        drone = drones[drone_id]
        product = products[product_id]

        if str_command == "L":
            destination = warehouses[destination_id]

        elif str_command == "D":
            destination = orders[destination_id]

        else:
            raise ValueError(f"Unknown command {s}.")

        commands.append(
            Command(
                drone=drone, destination=destination, product=product, quantity=quantity
            )
        )

    return commands, num_drones, max_time, max_cargo, wh_list, order_list

def calculate_score(commands: List[Command], max_time: int) -> int:
    command_to_time = get_command_times(commands)

    order_to_times = defaultdict(list)
    for c, time in command_to_time.items():
        if isinstance(c.destination, Order):
            order_to_times[c.destination].append(time)

    order_to_score = {}
    for o, ts in order_to_times.items():
        order_to_score[o] = ceil(100 * (max_time - max(ts)) / max_time)

    return sum(order_to_score.values())


def get_command_times(commands: List[Command]) -> Dict[Command, int]:
    # create dict: command -> timestamp

    command_to_time = {}

    drone_to_delivery_time = defaultdict(int)
    drone_to_position = {}
    for c in commands:
        drone = c.drone
        destination = c.destination

        previous_position = drone_to_position.get(drone, drone.position)

        drone_to_delivery_time[drone] += (
            previous_position.distance_from(destination.position) + 1
        )

        drone_to_position[drone] = destination.position

        command_to_time[c] = drone_to_delivery_time[drone]

    return command_to_time


def get_execution_time(commands: List[Command]) -> int:
    # how long does it take to complete the task list

    drone_to_time = defaultdict(int)
    drone_to_position = {}
    for c in commands:
        drone = c.drone
        destination = c.destination

        previous_position = drone_to_position.get(drone, drone.position)

        drone_to_time[drone] += (
            previous_position.distance_from(destination.position) + 1
        )

        drone_to_position[drone] = destination.position

    return max(drone_to_time.values())


class DroneRoute:
    def __init__(
        self,
        start_point: Union[Order, Warehouse],
        end_point: Union[Order, Warehouse],
        warehouses: List[Command],
        orders: List[Command],
    ):
        self.start_point = start_point
        self.end_point = end_point
        self.warehouses = warehouses
        self.orders = orders

    def _get_time(self, orders: List[Order]):
        time = 0
        previous_point = self.start_point
        for order in orders:
            time += order.distance_from(previous_point)
            previous_point = order
        time += previous_point.distance_from(self.end_point)
        return time

    def optimize(self):
        if len(self.orders) < 2:
            return self.orders

        min_time, optimal_order = None, None
        for p in itertools.permutations(self.orders):
            time = self._get_time([x.destination for x in p])
            if min_time is None or time < min_time:
                min_time, optimal_order = time, p
        return optimal_order
    
    

def swap_orders_in_routes(commands: List[Command]) -> List[Command]:
    # optimize time for each drone

    all_drones = {c.drone for c in commands}

    new_commands = []
    for drone in all_drones:
        drone_commands = [c for c in commands if c.drone == drone]

        start_point, end_point, orders = None, None, []
        for i, c in enumerate(drone_commands):

            if isinstance(c.destination, Order):
                orders.append(c)
                if i < len(drone_commands) - 1:
                    continue

            if len(orders) == 0:
                new_commands.append(c)
                continue

            if start_point is None:
                start_point = c

            end_point = c

            route = DroneRoute(
                start_point=start_point.destination,
                end_point=end_point.destination,
                warehouses=[],
                orders=orders,
            )

            new_commands += route.optimize()

            start_point = c
            orders = []
            if isinstance(end_point.destination, Warehouse):
                new_commands.append(end_point)

    return new_commands



def time_order_swap(commands: List[Command], max_time: int) -> List[Command]:
    # swap orders

    previous_score = calculate_score(commands, max_time)

    drone_to_commands = defaultdict(list)
    for c in commands:
        drone_to_commands[c.drone].append(c)

    bar = tqdm(total=int(len(commands) ** 2 / 2), desc="swap orders")
    for i, j in itertools.product(range(len(commands)), repeat=2):
        if i > j:
            continue

        bar.update(1)

        c1, c2 = commands[i], commands[j]

        if isinstance(c1.destination, Warehouse) or isinstance(
            c2.destination, Warehouse
        ):
            continue

        if c1.product != c2.product or c1.quantity != c2.quantity:
            continue

        d1, d2 = c1.drone, c2.drone
        t = get_execution_time(drone_to_commands[d1]) + get_execution_time(
            drone_to_commands[d2]
        )

        c1.destination, c2.destination = c2.destination, c1.destination

        new_t = get_execution_time(drone_to_commands[d1]) + get_execution_time(
            drone_to_commands[d2]
        )

        if new_t >= t:
            c1.destination, c2.destination = c2.destination, c1.destination
        else:
            score = calculate_score(commands, max_time)
            if score < previous_score:
                c1.destination, c2.destination = c2.destination, c1.destination
            else:
                previous_score = score

    bar.close()
    return commands

def post_processing(input_file, submission_file):
    commands, _, max_time, _, _, _ = read_submission(input_file, submission_file)
    print(calculate_score(commands, max_time))

    commands = swap_orders_in_routes(commands)
    print(calculate_score(commands, max_time))

    commands = time_order_swap(commands, max_time)
    print(calculate_score(commands, max_time))

    write_submission(commands, submission_file)
