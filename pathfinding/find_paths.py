import os
import pickle
import time
import pathfinding.collision_detection
import heapq
import math
from pathfinding import collision_detection
from pathfinding.Mission import Mission
from pathfinding.Node import Node
from pathfinding.EnergyPath import EnergyPath
from pathfinding.Workspace import Workspace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
# Define the path for saving/loading the model
MODEL_FILE_PATH = os.path.join(PROJECT_ROOT, "machine_learning/model_file/trained_model.pkl")

MAX_VELOCITY = 10
MAX_VELOCITY_V = 1.5


def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model


# Load the model
ml_model = load_model(MODEL_FILE_PATH)


def validate_workspace(workspace):
    if not isinstance(workspace, Workspace):
        raise ValueError("Invalid workspace input. Expected instance of Workspace class.")


def validate_mission(mission):
    if not isinstance(mission, Mission):
        raise ValueError("Invalid mission input. Expected instance of Mission class.")

    if not hasattr(mission, 'start') or not hasattr(mission, 'end'):
        raise ValueError("Mission must contain start and end nodes.")


def is_within_bounds(workspace, node):
    max_x, max_y, max_z = workspace.max_bounds
    return (0 <= node.x <= max_x) and \
        (0 <= node.y <= max_y) and \
        (0 <= node.z <= max_z)


def check_node_bounds(workspace, node):
    if not is_within_bounds(workspace, node):
        raise ValueError("Node is out of bounds.")


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
    if axis != 'z':
        setattr(next_node, 'velocity_' + axis, diff_coord / (20 / next_velocity))  # TODO: check paper
    else:
        setattr(next_node, 'velocity_' + axis, diff_coord / (3 / next_velocity))
    return diff_coord


def set_axis_velocity(current_node, next_node, mission):
    if next_node == mission.end:
        next_node.velocity_x = 0
        next_node.velocity_y = 0
        next_node.velocity_z = 0
        return

    # difference in distance on each coordinate in abosulute values
    diff_x = abs(next_node.x - current_node.x)
    diff_y = abs(next_node.y - current_node.y)
    diff_z = abs(next_node.z - current_node.z)

    movement_magnitude = (diff_x ** 2 + diff_y ** 2 + diff_z ** 2) ** 0.5

    if diff_z > 0:
        next_node.velocity_z = MAX_VELOCITY_V
    else:
        next_node.velocity_z = 0

    remaining_velocity = (MAX_VELOCITY ** 2 - next_node.velocity_z ** 2) ** 0.5

    proportion_x = diff_x / movement_magnitude
    proportion_y = diff_y / movement_magnitude
    horizontal_magnitude = (diff_x ** 2 + diff_y ** 2) ** 0.5

    if horizontal_magnitude > 0:
        next_node.velocity_x = proportion_x * remaining_velocity
        next_node.velocity_y = proportion_y * remaining_velocity
    else:
        next_node.velocity_x = 0
        next_node.velocity_y = 0


def calculate_time(current_node, next_node, mission, is_heuristic):
    # Initialize a list to store time calculations for each axis
    time_axes = []

    # If not using heuristic, set the velocity for each axis based on the mission details
    if not is_heuristic:
        set_axis_velocity(current_node, next_node, mission)

    # Loop over each axis ('x', 'y', 'z') to calculate the time required to travel the distance
    for axis in ['x', 'y', 'z']:
        # Calculate the distance to travel along the current axis
        dist = abs(getattr(next_node, axis) - getattr(current_node, axis))

        # Determine the velocities for the current axis based on whether heuristic is used
        if is_heuristic:
            if axis != 'z':
                velocity_current_axis = MAX_VELOCITY
            else:
                velocity_current_axis = MAX_VELOCITY_V
            velocity_next_axis = 0
        else:
            velocity_current_axis = getattr(current_node, 'velocity_' + axis)
            velocity_next_axis = getattr(next_node, 'velocity_' + axis)

        # Initialize time variables for acceleration and constant velocity phases
        t1 = 0  # time of acceleration
        t2 = 0  # time of constant velocity

        # Set acceleration rate based on the axis
        if axis != 'z':
            a = 5
        else:
            a = 1

        # Case where both initial and final velocities are zero, no movement
        if velocity_current_axis == 0 and velocity_next_axis == 0:
            pass
        else:
            t1 = abs((velocity_next_axis - velocity_current_axis) / a)
            remaining_dist = t1 * abs((velocity_next_axis + velocity_current_axis)) / 2

            # Case where there's a final velocity (deceleration phase)
            if velocity_next_axis != 0:
                if dist != remaining_dist:
                    t2 = ((dist - remaining_dist) / abs(velocity_next_axis))

            # Case where there's only an initial velocity (acceleration phase)
            else:
                if dist != remaining_dist:
                    t2 = ((dist - remaining_dist) / abs(velocity_current_axis))

        # Total time is the sum of acceleration and constant velocity times
        total_time = t1 + t2

        # Check for negative time values which are not possible
        if total_time < 0:
            raise ValueError(f"Time is negative for axis {axis} in nodes {current_node} & {next_node}")
        # Append calculated time for the current axis to the list
        time_axes.append(total_time)

    # Determine the maximum time required from all axes
    max_time = max(time_axes)
    # Ensure the maximum time is not zero to avoid logical errors
    if max_time == 0:
        raise ValueError(f'max_time is 0! for nodes {current_node} & {next_node}')

    # Return the maximum time required among all axes
    return max_time


def calculate_path_energy(path, workspace):
    mission = workspace.mission
    energy = 0
    previous_node = None
    end_node = mission.end
    for node in path:
        # Power is calculated for each node.
        if previous_node is not None and node != end_node:
            energy += heuristic_energy(previous_node, node, workspace)
        elif node == end_node:
            energy += heuristic_energy(previous_node, node, workspace)
        previous_node = node
    return energy


def heuristic_energy(current_node, next_node, workspace, is_heuristic=False):
    power_joule = 0
    try:
        if current_node == next_node:
            return power_joule

        mission = workspace.mission

        time = calculate_time(current_node, next_node, mission, is_heuristic)
        wind_speed = workspace.wind_field[int(current_node.x - 1), int(current_node.y - 1), int(current_node.z - 1)]
        wind_angle = workspace.wind_angle

        if current_node == mission.end:
            return 0

        if is_heuristic:
            curr_velocity_x = MAX_VELOCITY
            curr_velocity_y = MAX_VELOCITY
            curr_velocity_z = MAX_VELOCITY_V
            next_velocity_x = 0
            next_velocity_y = 0
            next_velocity_z = 0
        else:
            curr_velocity_x = current_node.velocity_x
            curr_velocity_y = current_node.velocity_y
            curr_velocity_z = current_node.velocity_z
            next_velocity_x = next_node.velocity_x
            next_velocity_y = next_node.velocity_y
            next_velocity_z = next_node.velocity_z

        linear_acceleration_x = (next_velocity_x - curr_velocity_x) / time
        linear_acceleration_y = (next_velocity_y - curr_velocity_y) / time
        linear_acceleration_z = (next_velocity_z - curr_velocity_z) / time

        input_array = [[time, wind_speed, wind_angle,
                        next_node.x - current_node.x, next_node.y - current_node.y, next_node.z - current_node.z,
                        curr_velocity_x, curr_velocity_y, curr_velocity_z,
                        linear_acceleration_x, linear_acceleration_y, linear_acceleration_z,
                        mission.payload]]

        target_labels = ml_model.predict(input_array)
        power_watt = power(target_labels)
        energy_joule = power_watt * time
    except Exception as e:
        raise IOError(f'An error ocurred: {e}')
    return energy_joule


def distance_h(node1, node2):
    dist_x = (node1.x - node2.x) ** 2
    dist_y = (node1.y - node2.y) ** 2
    return math.sqrt(dist_x + dist_y)


def distance_v(node1, node2):
    return abs(node1.z - node2.z)


def get_directions_baseline_path():
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
    return directions


def get_directions_optimal_path():
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
    return directions


def get_neighbors_baseline_path(node, end_node):
    directions = get_directions_baseline_path()
    neighbors = []

    # if next to goal
    if math.sqrt((node.x - end_node.x) ** 2 + (node.y - end_node.y) ** 2) <= 20:
        neighbors.append(end_node)

    else:
        # Travel dist for nodes - each dist.i is a tuple of (x, y, z)
        for dist_x, dist_y, dist_z in directions:
            new_x = node.x + dist_x
            new_y = node.y + dist_y
            new_z = node.z + dist_z
            new_node = Node(new_x, new_y, new_z)
            neighbors.append(new_node)
    return neighbors


def get_neighbors_optimal_path(node, workspace):
    neighbors = []
    start_node = workspace.mission.start
    end_node = workspace.mission.end
    directions = get_directions_optimal_path()

    # if next to goal
    if distance_h(end_node, node) <= 20 and distance_v(end_node,
                                                       node) <= 3 and collision_detection.check_segment_intersects_blockages(
        [node.x, end_node.x], [node.y, end_node.y], [node.z, end_node.z], workspace.blockages) is False:
        neighbors.append(end_node)

        if node == start_node:
            raise Exception("Start node next to end node")

    else:
        for dist_x, dist_y, dist_z in directions:
            new_x = node.x + dist_x
            new_y = node.y + dist_y
            new_z = node.z + dist_z
            new_node = Node(new_x, new_y, new_z)
            if collision_detection.check_segment_intersects_blockages([node.x, new_x], [node.y, new_y],
                                                                      [node.z, new_z],
                                                                      workspace.blockages) is False and is_within_bounds(
                workspace, new_node) and new_z > 0:
                neighbors.append(new_node)
    return neighbors


def find_baseline_path(workspace):
    mission = workspace.mission
    start_node = mission.start
    end_node = mission.end

    validate_workspace(workspace)
    validate_mission(mission)

    check_node_bounds(workspace, start_node)
    check_node_bounds(workspace, end_node)

    def heuristic_distance(node):
        return distance_h(node, end_node)

    # Priority Queue: For keeping track of nodes to be explored next
    pq = [(0, start_node)]
    visited = {start_node: 0}
    predecessor = {}

    while pq:
        _, current = heapq.heappop(pq)

        if current == end_node:
            break

        for neighbor in get_neighbors_baseline_path(current, end_node):
            # Check if the neighbor is within bounds before processing
            if not is_within_bounds(workspace, neighbor):
                continue

            # Calculate tentative distance through current node
            tentative_distance = visited[current] + distance_h(current, neighbor)
            # If the tentative distance is less than the recorded distance to the neighbor, update it
            if neighbor not in visited or tentative_distance < visited[neighbor]:
                visited[neighbor] = tentative_distance
                heapq.heappush(pq, (tentative_distance + heuristic_distance(neighbor), neighbor))
                predecessor[neighbor] = current

    path = []
    current = end_node
    # Using the predecessor dictionary to build the path
    # The while loop traverses as long as the current node has a predecessor
    while current in predecessor:
        path.insert(0, current)
        current = predecessor[current]
    path.insert(0, start_node)

    xs = []
    ys = []
    zs = []
    z_target = 0

    # Converting the path to x,y,z coordinates.
    for point in path:
        xs.append(point.x)
        ys.append(point.y)
        zs.append(point.z)

    # Iterating through and converting the x,y,z coordinates to pairs.
    for i in range(len(xs) - 1):
        x_pair = [xs[i], xs[i + 1]]
        y_pair = [ys[i], ys[i + 1]]
        z_pair = [zs[i], zs[i + 1]]

        # Checking if each pairs of coordinates intersect with any blockages.
        segments_intersects = collision_detection.check_segment_intersects_blockages(x_pair, y_pair, z_pair,
                                                                                     workspace.blockages)
        # If any intersections is found, we store the height of the highest intersecting blockage as z_target.
        if segments_intersects:
            new_z_target = pathfinding.collision_detection.find_max_intersection_z(x_pair, y_pair, z_pair,
                                                                                   workspace.blockages)
            if new_z_target > z_target:
                z_target = new_z_target
    baseline_path = []
    clearance_height = 3

    # We check that the z_target and a given clearance height is within the bounds of the workspace.
    desired_height = z_target + clearance_height
    if desired_height <= workspace.max_bounds[2]:
        baseline_path.append(start_node)

        # For each coordinate in the path, we create Nodes with the desired height.
        for coordinate in path:
            if coordinate != end_node:
                new_coordinate = Node(coordinate.x, coordinate.y, desired_height)
                baseline_path.append(new_coordinate)
            else:
                baseline_path.append(end_node)

        path_coordinates = [(node.x, node.y, node.z) for node in baseline_path]
        print(path_coordinates)

        energy = calculate_path_energy(baseline_path, workspace)

        return EnergyPath(path_coordinates, energy, path_type='baseline')
    else:
        raise NotImplementedError('The baseline path is too high for the workspace')


def find_optimal_path(workspace):
    mission = workspace.mission
    start_time = time.time()
    print('Finding optimal path...')
    start_node = mission.start
    end_node = mission.end

    validate_workspace(workspace)
    validate_mission(mission)

    check_node_bounds(workspace, start_node)
    check_node_bounds(workspace, end_node)

    pq = [(0, start_node)]

    visited = {start_node: 0}
    predecessor = {}

    # pq_count = 0
    while pq:
        _, current = heapq.heappop(pq)

        if current == end_node:
            break

        try:
            for neighbor in get_neighbors_optimal_path(current, workspace):
                c_cost = visited[current]  # current cost
                n_cost = heuristic_energy(current,  # neighbor cost
                                          neighbor, workspace)
                t_cost = c_cost + n_cost  # Total cost = current + neighbor cost

                if neighbor not in visited or t_cost < visited[neighbor]:
                    visited[neighbor] = t_cost
                    predecessor[neighbor] = current
                    h_cost = heuristic_energy(neighbor, end_node, workspace, is_heuristic=True)  # heuristic cost
                    punish = 0
                    if neighbor.z <= 10:
                        punish = (10 - neighbor.z) * 130
                    a_cost = 1 * t_cost + 1 * h_cost + punish  # absolute cost
                    # print(f"t_cost: {t_cost}, h_cost: {h_cost}, a_cost: {a_cost}, x: {neighbor.x}, y: {neighbor.y}, z: {neighbor.z}")

                    heapq.heappush(pq, (a_cost, neighbor))  # absolute cost is used for pq
                    # pq_count += 1
                    # print('Priority Queue Count: ', pq_count)
        except Exception as e:
            raise IOError(f'An error ocurred: {e}')

    path = []
    current = end_node
    while current != start_node:
        path.append(current)
        current = predecessor[current]
    path.append(start_node)
    path.reverse()

    path_coordinates = [(node.x, node.y, node.z) for node in path]
    print(path_coordinates)
    power = calculate_path_energy(path, workspace)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("time for optimal path: ", elapsed_time)

    return EnergyPath(path_coordinates, power, path_type='optimal')
