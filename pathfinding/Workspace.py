import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

import numpy as np

import pathfinding.collision_detection
from pathfinding import collision_detection
import heapq
import math
from pathfinding.Node import Node

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


class Workspace:
    def __init__(self, dimensions, max_bounds):
        self.dimensions = dimensions
        self.max_bounds = max_bounds
        self.flight_paths = []
        self.blockages = []
        self.wind_field = []
        self.wind_angle_rad = 0
        self.wind_angle = 0
        self.grid_size = 100

    def add_wind_field(self, angle, wind_speed):
        # Convert wind angle to radians
        self.wind_angle_rad = np.radians(angle)
        self.wind_angle = angle

        # Initialize wind speed grid
        wind_speed_grid = np.zeros((self.max_bounds[0], self.max_bounds[1]))

        # These are built around how the wind affects our workspace from outside.

        # Determine the starting point and direction based on the wind angle
        if 0 <= angle < 90:
            start_point = [self.max_bounds[0] - 1, self.max_bounds[1] - 1]  # Wind comes from the top-right corner
            end_point = [self.max_bounds[0] - self.max_bounds[0], self.max_bounds[1] - self.max_bounds[1]]
            x_step = -1
            y_step = -1
        elif 90 <= angle < 180:
            start_point = [self.max_bounds[0] - self.max_bounds[0],
                           self.max_bounds[1] - 1]  # Wind comes from the top-left corner
            end_point = [self.max_bounds[0] - 1, self.max_bounds[1] - self.max_bounds[1]]

            x_step = 1
            y_step = -1

        elif 180 <= angle < 270:
            start_point = [self.max_bounds[0] - self.max_bounds[0],
                           self.max_bounds[1] - self.max_bounds[1]]  # Wind comes from the bottom-left corner
            end_point = [self.max_bounds[0] - 1, self.max_bounds[1] - 1]

            x_step = 1
            y_step = 1
        else:
            start_point = [self.max_bounds[0] - 1,
                           self.max_bounds[1] - self.max_bounds[1]]  # Wind comes from the bottom-right corner
            end_point = [self.max_bounds[0] - self.max_bounds[0], self.max_bounds[1] - 1]

            x_step = -1
            y_step = 1

        # Cast rays from the starting point until reaching the bounds or hitting a blockage
        x, y = start_point

        while 0 <= x < self.max_bounds[0] and 0 <= y < self.max_bounds[1]:
            # Check for blockages at the current point

            print("Current position:", x, y)  # Print current position

            for i in range(start_point[0], end_point[0] + x_step, x_step):
                xs = [start_point[0], x + x_step]

                for j in range(start_point[1], end_point[1] + y_step, y_step):
                    ys = [start_point[1], y + y_step]

                    print("Segment coordinates:", xs, ys)  # Print segment coordinates

                    if not collision_detection.check_segment_intersects_blockages(xs, ys, [0, 0], self.blockages):
                        # Store the wind speed in the grid
                        wind_speed_grid[x, y] = wind_speed
                    else:
                        # If there is a blockage, stop casting the ray
                        print('Found a blockage')

                    # Move to the next grid cell along the ray direction
                    y += y_step
                y = start_point[1]
                x += x_step

        self.wind_field = self.rotate_wind_grid(wind_speed_grid)

    def rotate_wind_grid(self, wind_speed_grid):
        # Calculate the rotation angle based on the wind angle
        rotation_angle = self.wind_angle % 360  # Ensure angle is within [0, 360)

        if 0 <= rotation_angle < 90:  # Wind comes from the top-right corner
            k = 0

        elif 90 <= rotation_angle < 180:  # Wind comes from the top-left corner
            k = 2
            # wind_speed_grid = np.flip(wind_speed_grid, axis=0)

        elif 180 <= rotation_angle < 270:  # Wind comes from the bottom-left corner
            k = 0

        else:  # Wind comes from the bottom-right corner
            k = -2
            # wind_speed_grid = np.flip(wind_speed_grid, axis=1)

        rotated_wind_speed_grid = np.rot90(wind_speed_grid, k=k, axes=(0, 1))

        return rotated_wind_speed_grid

    def add_blockage(self, blockage_matrix, position):
        if len(blockage_matrix.shape) != self.dimensions:
            raise ValueError(f"Blockage matrix dimensions must match the space dimensions")
        for i in range(self.dimensions):
            if not (0 <= position[i] <= self.max_bounds[i]):
                raise ValueError(f"Blockage position must be within the specified bounds")
            if position[i] + blockage_matrix.shape[i] > self.max_bounds[i]:
                raise ValueError(f"Blockage does not fit within the space dimensions")
        self.blockages.append((blockage_matrix, position))

    def add_flight_path(self, flight_path):
        self.flight_paths.append(flight_path)

    def plot_flight_paths(self, ax, dimension):
        if dimension == '3D':
            # Plot flight paths
            for flight_path in self.flight_paths:
                xs, ys, zs = zip(*flight_path)

                # Iterate over each segment of the flight path
                for i in range(len(xs) - 1):
                    # Extract coordinates for the current segment
                    x_coords = [xs[i], xs[i + 1]]
                    y_coords = [ys[i], ys[i + 1]]
                    z_coords = [zs[i], zs[i + 1]]

                    # # Check if the current segment intersects with any blockage
                    segment_intersects = collision_detection.check_segment_intersects_blockages(x_coords, y_coords,
                                                                                                z_coords,
                                                                                                self.blockages)

                    # Plot the segment in the appropriate color
                    if segment_intersects:
                        ax.plot(x_coords, y_coords, z_coords, color='r', alpha=0.5)
                    else:
                        ax.plot(x_coords, y_coords, z_coords, color='g', alpha=0.5)
            return ax
        elif dimension == '2D':
            # Plot flight paths
            for flight_path in self.flight_paths:
                xs, ys, zs = zip(*flight_path)

                # Iterate over each segment of the flight path
                for i in range(len(xs) - 1):
                    # Extract coordinates for the current segment
                    x_coords = [xs[i], xs[i + 1]]
                    y_coords = [ys[i], ys[i + 1]]
                    z_coords = [zs[i], zs[i + 1]]

                    # Check if the current segment intersects with any blockage
                    segment_intersects = collision_detection.check_segment_intersects_blockages(x_coords, y_coords,
                                                                                                z_coords,
                                                                                                self.blockages)

                    # Plot the segment in the appropriate color
                    if segment_intersects:
                        ax.plot(x_coords, y_coords, color='r', alpha=0.5)
                    else:
                        ax.plot(x_coords, y_coords, color='g', alpha=0.5)
            return ax

    def plot_space(self, dimension='3D', dpi=300, show_wind=False):
        if dimension == '3D':
            fig = plt.figure(dpi=dpi)
            ax = fig.add_subplot(111, projection='3d')

            # Plot origin
            ax.scatter(0, 0, 0, color='k')

            # Plot blockages
            for blockage_matrix, position in self.blockages:
                x, y, z = position
                ax.bar3d(x, y, z, *blockage_matrix.shape, color='k', alpha=0.5, edgecolor='black', linewidth=0.5)

            ax = self.plot_flight_paths(ax, dimension='3D')

            # Set labels and limits
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim([0, self.max_bounds[0]])
            ax.set_ylim([0, self.max_bounds[1]])
            ax.set_zlim([0, self.max_bounds[2]])

        elif dimension == '2D':
            grid_color = 'gray'
            fig, ax = plt.subplots(dpi=dpi)

            # Plot origin
            ax.scatter(0, 0, color='k')

            # Plot blockages
            for blockage_matrix, position in self.blockages:
                x, y = position[:2]
                ax.add_patch(
                    plt.Rectangle((x, y), blockage_matrix.shape[0], blockage_matrix.shape[1], color='k', alpha=0.5))

            if show_wind:
                grid_color = 'white'
                # Plot the wind speed grid with a colormap
                plt.imshow(self.wind_field, cmap='viridis', origin='lower',
                           extent=[0, self.max_bounds[0], 0, self.max_bounds[1]])
                plt.colorbar(label='Wind speed')

                # Plot wind direction arrow
                # Calculate the endpoint of the arrow
                arrow_length = self.max_bounds[0] / 10

                arrow_end_x = self.max_bounds[0] / 2 + arrow_length * np.cos(self.wind_angle_rad)
                arrow_end_y = self.max_bounds[1] / 2 + arrow_length * np.sin(self.wind_angle_rad)

                ax.annotate('',
                            xy=(self.max_bounds[0] / 2, self.max_bounds[1] / 2),
                            xytext=(arrow_end_x, arrow_end_y),
                            arrowprops=dict(facecolor='blue', edgecolor='blue', alpha=0.4,
                                            path_effects=[path_effects.withSimplePatchShadow(offset=(-1, -1))]))

            ax = self.plot_flight_paths(ax, dimension='2D')

            # Set labels and limits
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim([0, self.max_bounds[0]])
            ax.set_ylim([0, self.max_bounds[1]])
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, color=grid_color, linestyle='--', linewidth=0.5, alpha=0.5)
        plt.show()

    def find_optimal_path(self, mission):
        print('Finding optimal path...')
        start_node = mission.start
        end_node = mission.end
        end_node.velocity_x = 0
        end_node.velocity_y = 0
        end_node.velocity_z = 0
        payload = mission.payload
        # blockages = []
        # wind_field = self.wind_field

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

        def calculate_time(current_node, next_node):
            def set_velocity_axis_return_distance(axis, current_node, next_node):
                current_velocity = current_node.velocity
                next_velocity = next_node.velocity

                diff_coord = abs(getattr(next_node, axis) - getattr(current_node, axis))

                if current_velocity == 0:
                    setattr(current_node, 'velocity_' + axis, 0)

                if next_node == end_node:
                    next_velocity = 0

                if next_velocity == 0:
                    return diff_coord

                # TODO: Skriv i paper
                if axis != 'z':
                    setattr(next_node, 'velocity_' + axis, diff_coord / (20 / next_velocity))
                else:
                    setattr(next_node, 'velocity_' + axis, diff_coord / (3 / next_velocity))

                return diff_coord

            time_axes = []
            for axis in ['x', 'y', 'z']:
                dist = set_velocity_axis_return_distance(axis, current_node, next_node)

                velocity_current_axis = getattr(current_node, 'velocity_' + axis)
                velocity_next_axis = getattr(next_node, 'velocity_' + axis)
                t1 = 0
                t2 = 0

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
                time_axes.append(time)

            max_time = max(time_axes)
            if max_time == 0:
                print('time is 0')
            return max_time

        def get_neighbors(node):
            neighbors = []

            # if next to goal
            if distance_h(end_node, node) <= 20 and distance_v(end_node,
                                                               node) <= 3 and collision_detection.check_segment_intersects_blockages(
                [node.x, end_node.x], [node.y, end_node.y], [node.z, end_node.z], self.blockages) is False:
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
                                                                              self.blockages) is False:
                        neighbors.append(new_node)
            return neighbors

        # TODO: Rune og Lucas: dist_x + dist_y <= 20, dist_z <= 3  --- se paper side 8 afsnit b
        def distance_h(node1, node2):
            dist_x = (node1.x - node2.x) ** 2
            dist_y = (node1.y - node2.y) ** 2
            return math.sqrt(dist_x + dist_y)

        def distance_v(node1, node2):
            return abs(node1.z - node2.z)

        def heuristic_power(current_node, next_node):
            time = calculate_time(current_node, next_node)
            wind_speed = 0
            wind_angle = 0

            if current_node == end_node:
                # raise Exception("current is end")
                return 0
            linear_acceleration_x = next_node.velocity_x / time
            linear_acceleration_y = next_node.velocity_y / time
            linear_acceleration_z = next_node.velocity_z / time

            input_array = [[time, wind_speed, wind_angle,
                            next_node.x - current_node.x, next_node.y - current_node.y, next_node.z - current_node.z,
                            next_node.velocity_x, next_node.velocity_y, next_node.velocity_z,
                            linear_acceleration_x, linear_acceleration_y, linear_acceleration_z,
                            payload]]

            target_labels = ml_model.predict(input_array)

            power_watt = power(target_labels)

            power_joule = power_watt * time

            return power_joule

        pq = [(0, start_node)]  # Priority queue

        visited = {start_node: 0}
        predecessor = {}

        while pq:
            _, current = heapq.heappop(pq)

            if current == end_node:
                break

            try:
                for neighbor in get_neighbors(current):
                    # Calculate the energy for the neighbor using the heuristic function
                    for velocity in velocities:
                        neighbor.velocity = velocity
                        c_cost = visited[current]  # Actual cost to reach the current node
                        n_cost = heuristic_power(current,
                                                 neighbor)  # Estimated cost to reach the goal from the current node
                        t_cost = c_cost + n_cost  # Total cost

                        if neighbor not in visited or t_cost < visited[neighbor]:
                            visited[neighbor] = t_cost
                            predecessor[neighbor] = current
                            e_cost = heuristic_power(neighbor, end_node)
                            a_cost = t_cost + e_cost
                            print(f"t_cost: {t_cost}, e_cost: {e_cost}, a_cost: {a_cost}")

                            heapq.heappush(pq, (a_cost, neighbor))  # Use the f cost as the priority
            except Exception as e:
                print(f'An error ocurred {e}')

        path = []
        current = end_node
        while current != start_node:
            path.append(current)
            current = predecessor[current]
        path.append(start_node)
        path.reverse()

        # Convert path nodes to coordinates
        path_coordinates = [(node.x, node.y, node.z) for node in path]

        print(path_coordinates)

        return path_coordinates

    def generate_random_path(self):
        # Access blockages directly from the workspace
        blockages = self.blockages

        # Generate random flight path data
        num_points = 50  # Number of points in the flight path
        min_coord, max_coord = 0, 400  # Range for coordinates in each dimension of x and z
        min_coord_z, max_coord_z = 0, 60

        # Generate random x, y, z coordinates for the flight path while avoiding blockages
        flight_path = []
        for _ in range(num_points):
            # Generate random coordinates
            x_coord = np.random.randint(min_coord, max_coord)
            y_coord = np.random.randint(min_coord, max_coord)
            z_coord = np.random.randint(min_coord_z, max_coord_z)  # Adjusted for z-axis (height)
            print(f"Generated coordinates: ({x_coord}, {y_coord}, {z_coord})")

            # Check if the generated coordinates are within any blockages
            while any(
                    blockage[1][0] <= x_coord < blockage[1][0] + blockage[0].shape[0] and
                    blockage[1][1] <= y_coord < blockage[1][1] + blockage[0].shape[1] and
                    blockage[1][2] <= z_coord < blockage[1][2] + blockage[0].shape[2]
                    for blockage in blockages
            ):
                # Regenerate coordinates until they are outside all blockages
                x_coord = np.random.randint(min_coord, max_coord)
                y_coord = np.random.randint(min_coord, max_coord)
                z_coord = np.random.randint(min_coord, max_coord // 4)

                print(f"Regenerated coordinates: ({x_coord}, {y_coord}, {z_coord})")

            # Append the valid coordinates to the flight path
            flight_path.append((x_coord, y_coord, z_coord))

        return flight_path

    @staticmethod
    def generate_path_completely_fill_blockage(grid_size=400, blockage_size=50, blockage_x=250, blockage_y=250):
        path = []

        # Moving along the top boundary
        for x in range(0, grid_size):
            path.append((x, 0, 0))

        # Zigzag pattern towards the blockage from the top
        for y in range(1, grid_size):
            if y % 2 == 1:  # Odd rows move right
                for x in range(grid_size - 1, -1, -1):
                    path.append((x, y, 0))
            else:  # Even rows move left
                for x in range(0, grid_size):
                    path.append((x, y, 0))

        return path

    def find_baseline_path(self, mission):
        start_node = mission.start
        end_node = mission.end
        blockages = self.blockages

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

        def distance(node1, node2):
            dist_x = (node1.x - node2.x) ** 2
            dist_y = (node1.y - node2.y) ** 2
            return math.sqrt(dist_x + dist_y)

        def heuristic_distance(node):
            return distance(node, end_node)

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
                tentative_distance = visited[current] + distance(current, neighbor)

                # If the tentative distance is less than the recorded distance to the neighbor, update it
                if neighbor not in visited or tentative_distance < visited[neighbor]:
                    visited[neighbor] = tentative_distance
                    heapq.heappush(pq, (tentative_distance + heuristic_distance(neighbor), neighbor))
                    predecessor[neighbor] = current

                # Reconstruct the path
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

        z_target = pathfinding.collision_detection.find_max_intersection_z(xs, ys, zs, blockages)
        baseline_path = []
        clearance_height = 5

        if z_target + clearance_height <= self.max_bounds[2]:
            baseline_path.append(start_node)
            for coordinate in path:
                new_coordinate = Node(coordinate.x, coordinate.y,
                                      z_target + clearance_height)  # +5 fordi det ikke ordentligt at flyve præcis i blockagens højde.
                baseline_path.append(new_coordinate)

            baseline_path.append(end_node)
            path_coordinates = [(node.x, node.y, node.z) for node in baseline_path]
            print(path_coordinates)
            return path_coordinates
        else:
            raise NotImplementedError('The baseline path is too high for the workspace')


def check_model():
    input_array = [[50, 0, 0,  # time, wind_speed, wind_angle
                    100, 100, 0,  # postions
                    4, 0, 0,  # velocities
                    4, 0, 0,  # accelerations
                    500  # payloads
                    ]]

    target_labels = ml_model.predict(input_array)
    return target_labels


target_labels = check_model()
print(target_labels)
pow_res = power(target_labels)
print(pow_res)
