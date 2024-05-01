import os
import pickle
import pathfinding.collision_detection
from pathfinding import collision_detection
import heapq
import math
from pathfinding.Node import Node
from pathfinding.EnergyPath import EnergyPath

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
# Define the path for saving/loading the model
MODEL_FILE_PATH = os.path.join(PROJECT_ROOT, "machine_learning/model_file/trained_model.pkl")


def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model


# Load the model
ml_model = load_model(MODEL_FILE_PATH)


def power(target_labels):
    return target_labels[:, 0] * target_labels[:, 1]


def set_velocity_axis_return_distance(axis, current_node, next_node, mission, is_heuristic):
    current_velocity = current_node.velocity
    next_velocity = next_node.velocity
    end_node = mission.end

    diff_coord = abs(getattr(next_node, axis) - getattr(current_node, axis))

    if current_velocity == 0:
        setattr(current_node, 'velocity_' + axis, 0)

    if next_node == end_node:
        next_velocity = 0

    if is_heuristic:
        setattr(current_node, 'velocity_' + axis, current_velocity)

    if next_velocity == 0:
        return diff_coord

    # TODO: Skriv i paper
    if axis != 'z':
        setattr(next_node, 'velocity_' + axis, diff_coord / (20 / next_velocity))  # TODO: check paper
    else:
        setattr(next_node, 'velocity_' + axis, diff_coord / (3 / next_velocity))

    return diff_coord


def calculate_time(current_node, next_node, mission, is_heuristic):
    time_axes = []
    for axis in ['x', 'y', 'z']:
        # check om er 20 m og 3 m væk. hvis ja, kald denne. ellers exception
        dist = set_velocity_axis_return_distance(axis, current_node, next_node, mission, is_heuristic)

        velocity_current_axis = getattr(current_node, 'velocity_' + axis)
        velocity_next_axis = getattr(next_node, 'velocity_' + axis)
        t1 = 0  # time of acceleration
        t2 = 0  # time of constant velocity

        if axis != 'z':
            a = 5
        else:
            a = 1

        if velocity_current_axis == 0 and velocity_next_axis == 0:
            pass

        elif velocity_next_axis != 0:
            t1 = abs((velocity_next_axis - velocity_current_axis) / a)
            if dist != t1 * abs((velocity_next_axis + velocity_current_axis)) / 2:
                t2 = ((dist - t1 * abs((velocity_next_axis + velocity_current_axis)) / 2)
                      / abs(velocity_next_axis))

        else:
            t1 = abs((velocity_next_axis - velocity_current_axis) / a)
            if dist != t1 * abs((velocity_next_axis + velocity_current_axis)) / 2:
                t2 = ((dist - t1 * abs((velocity_next_axis + velocity_current_axis)) / 2)
                      / abs(velocity_current_axis))
        time = t1 + t2

        if time < 0:
            raise ValueError("Time less than zero!")
        time_axes.append(time)

    max_time = max(time_axes)
    if max_time == 0:
        print('time is 0')
    return max_time


def heuristic_power(current_node, next_node, mission, is_heuristic=False):
    time = calculate_time(current_node, next_node, mission, is_heuristic)
    wind_speed = 0
    wind_angle = 0

    if current_node == mission.end:
        return 0

    linear_acceleration_x = next_node.velocity_x / time
    linear_acceleration_y = next_node.velocity_y / time
    linear_acceleration_z = next_node.velocity_z / time

    input_array = [[time, wind_speed, wind_angle,
                    next_node.x - current_node.x, next_node.y - current_node.y, next_node.z - current_node.z,
                    next_node.velocity_x, next_node.velocity_y, next_node.velocity_z,
                    linear_acceleration_x, linear_acceleration_y, linear_acceleration_z,
                    mission.payload]]

    target_labels = ml_model.predict(input_array)

    power_watt = power(target_labels)

    power_joule = power_watt * time

    return power_joule


def distance_h(node1, node2):
    dist_x = (node1.x - node2.x) ** 2
    dist_y = (node1.y - node2.y) ** 2
    return math.sqrt(dist_x + dist_y)


def distance_v(node1, node2):
    return abs(node1.z - node2.z)


def find_baseline_path(workspace, mission):
    start_node = mission.start
    end_node = mission.end

    h = 10 * math.sqrt(2)
    directions = [  # 8 directions, z-index always 0.
        (h, h, 0),
        (20, 0, 0),
        (h, -h, 0),

        (0, 20, 0),

        (0, -20, 0),

        (-h, h, 0),
        (-20, 0, 0),
        (-h, -h, 0),
    ]

    def heuristic_distance(node):
        return distance_h(node, end_node)

    def get_neighbors(node):
        neighbors = []

        # if next to goal
        if math.sqrt((node.x - end_node.x) ** 2 + (node.y - end_node.y) ** 2) <= 20:
            neighbors.append(end_node)

        else:
            for dist_x, dist_y, dist_z in directions:
                new_x = node.x + dist_x
                new_y = node.y + dist_y
                new_z = node.z + dist_z
                new_node = Node(new_x, new_y, new_z)
                neighbors.append(new_node)
        return neighbors

    pq = [(0, start_node)]

    visited = {start_node: 0}
    predecessor = {}

    while pq:
        _, current = heapq.heappop(pq)

        if current == end_node:
            break

        for neighbor in get_neighbors(current):
            # Calculate tentative distance through current node
            tentative_distance = visited[current] + distance_h(current, neighbor)
            # If the tentative distance is less than the recorded distance to the neighbor, update it
            if neighbor not in visited or tentative_distance < visited[neighbor]:
                visited[neighbor] = tentative_distance
                heapq.heappush(pq, (tentative_distance + heuristic_distance(neighbor), neighbor))
                predecessor[neighbor] = current

    path = []
    current = end_node
    while current in predecessor:
        path.insert(0, current)
        current = predecessor[current]
    path.insert(0, start_node)

    xs = []
    ys = []
    zs = []

    for point in path:
        xs.append(point.x)
        ys.append(point.y)
        zs.append(point.z)

    z_target = pathfinding.collision_detection.find_max_intersection_z(xs, ys, zs, workspace.blockages)
    baseline_path = []
    clearance_height = 5

    if z_target + clearance_height <= workspace.max_bounds[2]:
        baseline_path.append(start_node)
        for coordinate in path:
            if coordinate != end_node:
                new_coordinate = Node(coordinate.x, coordinate.y,
                                      z_target + clearance_height)  # +5 fordi det ikke ordentligt at flyve præcis i blockagens højde.
                baseline_path.append(new_coordinate)
            else:
                baseline_path.append(end_node)

        path_coordinates = [(node.x, node.y, node.z) for node in baseline_path]
        print(path_coordinates)

        power = 0

        previous_node = None
        for node in baseline_path:
            if previous_node is not None and node != end_node:
                node.velocity = 12
                power += heuristic_power(previous_node, node, mission)
            elif node == end_node:
                power += heuristic_power(previous_node, node, mission)
            previous_node = node

        print("POWER: ", power)
        return EnergyPath(path_coordinates, power)
    else:
        raise NotImplementedError('The baseline path is too high for the workspace')


def find_optimal_path(workspace, mission):
    print('Finding optimal path...')
    start_node = mission.start
    end_node = mission.end
    end_node.velocity_x = 0 #TODO: Check om nødvendigt
    end_node.velocity_y = 0
    end_node.velocity_z = 0

    h = 10 * math.sqrt(2)
    directions = [
        (h, h, 0), (h, h, 3), (h, h, -3),
        (20, 0, 0), (20, 0, 3), (20, 0, -3),
        (h, -h, 0), (h, -h, 3), (h, -h, -3),

        (0, 20, 0), (0, 20, 3), (0, 20, -3),
        # (0, 0, 0),
        (0, 0, 3), (0, 0, -3),
        (0, -20, 0), (0, -20, 3), (0, -20, -3),

        (-h, h, 0), (-h, h, 3), (-h, h, -3),
        (-20, 0, 0), (-20, 0, 3), (-20, 0, -3),
        (-h, -h, 0), (-h, -h, 3), (-h, -h, -3),
    ]
    # 4, 6, 8, 10, 12
    velocities = (10, 12)

    def get_neighbors(node):
        neighbors = []

        # if next to goal
        if distance_h(end_node, node) <= 20 and distance_v(end_node,
                                                           node) <= 3 and collision_detection.check_segment_intersects_blockages(
            [node.x, end_node.x], [node.y, end_node.y], [node.z, end_node.z], workspace.blockages) is False:
            end_node.velocity = 0
            end_node.velocity_x = 0
            end_node.velocity_y = 0
            end_node.velocity_z = 0
            neighbors.append(end_node)

        else:
            for dist_x, dist_y, dist_z in directions:
                new_x = node.x + dist_x
                new_y = node.y + dist_y
                new_z = node.z + dist_z
                new_node = Node(new_x, new_y, new_z)
                if collision_detection.check_segment_intersects_blockages([node.x, new_x], [node.y, new_y],
                                                                          [node.z, new_z],
                                                                          workspace.blockages) is False and new_z >= 0 and new_y >= 0 and new_x >= 0:
                    neighbors.append(new_node)
        return neighbors

    pq = [(0, start_node)]

    visited = {start_node: 0}
    predecessor = {}

    while pq:
        _, current = heapq.heappop(pq)

        if current == end_node:
            break

        try:
            for neighbor in get_neighbors(current):
                for velocity in velocities:
                    neighbor.velocity = velocity
                    c_cost = visited[current]  # current cost
                    n_cost = heuristic_power(current,  # neighbor cost
                                             neighbor, mission)
                    t_cost = c_cost + n_cost  # Total cost = current + neighbor cost

                    if neighbor not in visited or t_cost < visited[neighbor]:
                        visited[neighbor] = t_cost
                        predecessor[neighbor] = current
                        h_cost = heuristic_power(neighbor, end_node, mission, is_heuristic=True)
                        punish = 0
                        a_cost = 1 * t_cost + 1 * h_cost + punish # absolute cost
                        # print(f"t_cost: {t_cost}, h_cost: {h_cost}, a_cost: {a_cost}, x: {neighbor.x}, y: {neighbor.y}, z: {neighbor.z}")

                        heapq.heappush(pq, (a_cost, neighbor))  # absolute cost is used for pq
        except Exception as e:
            print(f'An error ocurred: {e}')

    path = []
    current = end_node
    while current != start_node:
        path.append(current)
        current = predecessor[current]
    path.append(start_node)
    path.reverse()

    path_coordinates = [(node.x, node.y, node.z) for node in path]

    print(path_coordinates)

    return EnergyPath(path_coordinates, a_cost)