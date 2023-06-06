# Given: Students Addresses And Schools, Number Of Busses/Drivers
from __future__ import annotations

import math
import random
from datetime import time

import matplotlib.colors
import matplotlib.pyplot as plt
import copy


class Location:
    def __init__(self, long: float, lat: float, access_street: str | None = None):
        self.long: float = long
        self.lat: float = lat
        self.access_street: str | None = access_street
        self._connections = []

    def __repr__(self):
        return f"({self.long}, {self.lat})"

    def __eq__(self, other):
        # Coordinates Are Precise To 15 Decimal Places This Is Adjusted For The Tolerance To Work
        return math.isclose(self.long, other.long, rel_tol=1e-17) and math.isclose(self.lat, other.lat, rel_tol=1e-17)

    def __hash__(self):
        return hash((self.lat, self.long))

    def get_coord(self) -> tuple[float, float]:
        return self.long, self.lat

    def create_connections(self, locations: list[Location]):
        self._connections = []
        for i in locations:
            dist = math.dist(self.get_coord(), i.get_coord())
            if dist > 0:
                self._connections.append((i, dist))

    def get_shortest_link(self, remove: bool = False):
        result = min(self._connections, key=lambda x: x[1])
        if remove:
            self._connections.remove(result)
        return result

    def __copy__(self):
        l = Location(self.long, self.lat, self.access_street)
        l._connections = copy.deepcopy(self._connections)
        return l


class Student:
    def __init__(self, house: Location, school: Location):
        self.house = house
        self.school = school


class Stop(Location):
    def __init__(self, pickup_time: time, students: list[Student]):
        self.time = pickup_time
        self.students = students
        super().__init__(1, 1)


class Route:
    def __init__(self):
        self.stops = []


class Link:
    def __init__(self):
        self.time_to_drive = 0


class Bus:
    def __init__(self, start: Stop):
        self.capacity = 72  # TODO: Get A Value For This Number
        self.stops = [start]


def distance(city1, city2):
    # Calculate the Euclidean distance between two cities
    x1, y1 = city1.get_coord()
    x2, y2 = city2.get_coord()
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # XY Distance
    # return abs(y2-y1) + abs(x2-y2)


def interpolate(num, in_min, in_max, out_min, out_max):
    return out_min + (float(num - in_min) / float(in_max - in_min) * (out_max - out_min))


HIGHEST_POINT = 1000
NUMBER_NODES = 500
GROUPS = 10
nodes = [Location(int(random.random() * HIGHEST_POINT), int(random.random() * HIGHEST_POINT)) for _ in range(HIGHEST_POINT)]
start = Location(0, 0)
end = Location(HIGHEST_POINT, HIGHEST_POINT)
nodes.insert(0, start)
nodes.append(end)
rm_list = []
for n, i in enumerate(nodes):
    if i in nodes[n+1:]:
        rm_list.append(i)
for i in rm_list:
    nodes.remove(i)


def plot_path(path: list[Location], show: bool = False, color: tuple = (255, 0, 0)):
    for point in range(len(path) - 1):
        plt.plot(
            [path[point].get_coord()[0], path[point + 1].get_coord()[0]],
            [path[point].get_coord()[1], path[point + 1].get_coord()[1]],
            "o-",
            color=color,
        )
    if show:
        plt.xlim(-1, HIGHEST_POINT + 1)
        plt.ylim(-1, HIGHEST_POINT + 1)
        plt.show()


def nearest_neighbor(nodes, start, end):
    nodes = nodes.copy()
    ordered_nodes = []
    end._connections = [x for x in end._connections if x[0] != start]
    start._connections = [x for x in start._connections if x[0] != end]
    start._connections.append((end, 0))
    for i in nodes:
        i.create_connections(nodes)
    point = start
    total = 0
    while len(nodes) > 1:
        while True:
            s = point.get_shortest_link(remove=True)
            if s[0] in nodes:
                break
        # plt.plot([point.get_coord()[0], s[0].get_coord()[0]], [point.get_coord()[1], s[0].get_coord()[1]], "ro-")
        nodes.remove(point)
        ordered_nodes.append(point)
        point = s[0]
        total += s[1]
    ordered_nodes.append(point)
    ordered_nodes.append(start)
    # plt.plot([point.get_coord()[0], start.get_coord()[0]], [point.get_coord()[1], start.get_coord()[1]], "ro-")
    # plt.plot([end.get_coord()[0], start.get_coord()[0]], [end.get_coord()[1], start.get_coord()[1]], "go-")
    # plt.title("Nearest Neighbor")
    # plt.show()
    return ordered_nodes, total


def cheapest_insertion(cities, start, end):
    cities = cities.copy()
    num_cities = len(cities)
    unvisited = set(range(num_cities))
    visited = []
    # start_city = cities.index(start)
    start_city = 0
    visited.append(start_city)
    unvisited.remove(start_city)

    while unvisited:
        min_cost = float('inf')
        best_city = None
        best_position = None

        for current_city in visited:
            for next_city in unvisited:
                for i in range(len(visited) + 1):
                    city1 = cities[current_city]
                    city2 = cities[next_city]
                    if i == 0:
                        city3 = cities[visited[0]]
                    elif i == len(visited):
                        city3 = cities[visited[-1]]
                    else:
                        city3 = cities[visited[i]]

                    if (city1 == start and city2 == end) or (city2 == start and city1 == end):
                        cost = 0 - distance(city1, city3)
                    else:
                        cost = distance(city1, city2) + distance(city2, city3) - distance(city1, city3)

                    if cost < min_cost:
                        min_cost = cost
                        best_city = next_city
                        best_position = i

        visited.insert(best_position, best_city)
        unvisited.remove(best_city)

    return [cities[city_index] for city_index in visited]


def cheapest_insertion_2(nodes, start, end):
    nodes = nodes.copy()
    path = [start, end]
    try:
        nodes.remove(start)
        nodes.remove(end)
    except ValueError:
        pass

    for link in enumerate(path[:-1]):
        shortest = (None, float("inf"))
        for node in nodes:
            pass


def shortest_random(nodes, start, end, iterations=10000):
    nodes.copy()
    try:
        nodes.remove(start)
        nodes.remove(end)
    except ValueError:
        pass
    shortest = (nodes, float("inf"))

    for i in range(iterations):
        random.shuffle(nodes)
        total = distance(start, nodes[0])
        for n, current in enumerate(nodes[:-1]):
            total += distance(current, nodes[n])
        total += distance(nodes[-1], end)

        if total < shortest[1]:
            shortest = (nodes.copy(), total)

    shortest[0].insert(0, start)
    shortest[0].append(end)

    # for n, t in enumerate(shortest[0][:-1]):
    #     plt.plot([t.long, shortest[0][n+1].long], [t.lat, shortest[0][n+1].lat], "bo-")
    # plt.plot([start.long, shortest[0][0].long], [start.lat, shortest[0][0].lat], "bo-")
    # plt.plot([end.long, shortest[0][-1].long], [end.lat, shortest[0][-1].lat], "bo-")
    #
    # plt.title("Random")
    # plt.show()
    return shortest[0]


def get_groups(nodes: list[Location], number: int):
    groups = {}
    old_groups = {}
    # Get Initial Centers
    for i in range(number):
        center = random.choice(nodes).__copy__()
        groups[center] = []

    while groups.values() != old_groups.values() and set(groups.keys()) != set(old_groups.keys()):
        # Assign Points To The Center Closest To Them
        for node in nodes:
            shortest = (None, float("inf"))
            for center in groups.keys():
                d = distance(center, node)
                if d == 0:
                    continue
                if d < shortest[1]:
                    shortest = (center, d)
            groups[shortest[0]].append(node)

        # Calculate The Average Center
        old_groups = groups
        groups = {}
        # nodes.extend(old_groups.keys())
        for c, n in old_groups.items():
            fake_stop = Location(
                long=(sum([ni.long for ni in n]) + c.long) / (len(n) + 1),
                lat=(sum([ni.lat for ni in n]) + c.lat) / (len(n) + 1),
            )
            groups[fake_stop] = []

    # Display
    for c, n in old_groups.items():
        for ni in n:
            plt.plot([c.long, ni.long], [c.lat, ni.lat], "bo-")
    plt.title("Groups")
    plt.xlim(-1, HIGHEST_POINT + 1)
    plt.ylim(-1, HIGHEST_POINT + 1)
    plt.show()
    return old_groups.values()


def find_overlapping_points(*lists):
    overlapping_points = []

    # Iterate over each list
    for i, lst in enumerate(lists):
        # Iterate over each pair of consecutive points in the list
        for j in range(len(lst) - 1):
            point1 = lst[j]
            point2 = lst[j + 1]

            # Check if the pair of points exists in any other list
            for k, other_lst in enumerate(lists):
                if not other_lst.index(point1) == len(other_lst) - 1:
                    if k != i and other_lst[other_lst.index(point1) + 1] == point2:
                        overlapping_points.append((point1, point2))
                        break

    # Display
    for n, t in enumerate(overlapping_points[:-1]):
        plt.xlim(-1, HIGHEST_POINT + 1)
        plt.ylim(-1, HIGHEST_POINT + 1)
        plt.plot([t[0].long, t[1].long], [t[0].lat, t[1].lat], "go-")
    plt.title("Overlapping")
    plt.xlim(-1, HIGHEST_POINT + 1)
    plt.ylim(-1, HIGHEST_POINT + 1)
    plt.show()

    return overlapping_points


# test_points = nodes
test_points = get_groups(nodes, GROUPS)
colors = [matplotlib.colors.hsv_to_rgb((interpolate(n*(360/GROUPS), 0, 360, 0, 1), 0.8, 0.7)) for n in range(GROUPS)]
print(colors)

for n, group in enumerate(test_points):
    nrst_nghbr, total = nearest_neighbor(group, group[0], group[-1])
    print(total)
    plot_path(nrst_nghbr, color=colors[n])
plt.title("Nearest Neighbor")
plot_path([], True)

for n, group in enumerate(test_points):
    chp_ins = cheapest_insertion(group, group[0], group[-1])
    plot_path(chp_ins, color=colors[n])
plt.title("Cheapest Insert")
plot_path([], True)

for n, group in enumerate(test_points):
    rand = shortest_random(group, group[0], group[-1])
    plot_path(rand, color=colors[n])
plt.title("Best Of Random")
plot_path([], True)

# TODO: This Needs To Be For Each For As Well
find_overlapping_points(rand, chp_ins, nrst_nghbr)


# https://www.google.com/maps/d/edit?hl=en&mid=1xdhs2L9CgaR_4wNIOCAa9tX_6K6vOXw&ll=44.1242064104337%2C-92.58897285008989&z=12
