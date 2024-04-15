import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

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

    def add_wind_field(self, angle, wind_speed):
        # Use this for wind speed
        # call from main.py, just like blockage
        # self.wind_field.append()
        raise NotImplementedError

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
                    segment_intersects = collision_detection.check_segment_intersects_blockages(x_coords, y_coords, z_coords,
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
                    segment_intersects = collision_detection.check_segment_intersects_blockages(x_coords, y_coords, z_coords,
                                                                               self.blockages)

                    # Plot the segment in the appropriate color
                    if segment_intersects:
                        ax.plot(x_coords, y_coords, color='r', alpha=0.5)
                    else:
                        ax.plot(x_coords, y_coords, color='g', alpha=0.5)
            return ax

    def plot_space(self, dimension='3D', dpi=300):
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
            fig, ax = plt.subplots(dpi=dpi)

            # Plot origin
            ax.scatter(0, 0, color='k')

            # Plot blockages
            for blockage_matrix, position in self.blockages:
                x, y = position[:2]
                ax.add_patch(plt.Rectangle((x, y), blockage_matrix.shape[0], blockage_matrix.shape[1], color='k'))

            ax = self.plot_flight_paths(ax, dimension='2D')

            # Set labels and limits
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim([0, self.max_bounds[0]])
            ax.set_ylim([0, self.max_bounds[1]])
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.show()

    def check_segment_intersects_blockage(self, xs, ys, zs, blockage):
        if zs is not None:
            # If 3-Dimensional
            for i in range(len(xs) - 1):
                if (
                        (blockage[1][0] <= xs[i] < blockage[1][0] + blockage[0].shape[0] and
                         blockage[1][1] <= ys[i] < blockage[1][1] + blockage[0].shape[1] and
                         blockage[1][2] <= zs[i] < blockage[1][2] + blockage[0].shape[2])
                        or
                        (blockage[1][0] <= xs[i + 1] < blockage[1][0] + blockage[0].shape[0] and
                         blockage[1][1] <= ys[i + 1] < blockage[1][1] + blockage[0].shape[1] and
                         blockage[1][2] <= zs[i + 1] < blockage[1][2] + blockage[0].shape[2])
                ):
                    print('Collision in 3D Block found')
                    return True
            return False
        else:
            # If 2-Dimensional
            for i in range(len(xs) - 1):
                # Accessing the first zero'th element of the 1st list. That is (30,30,0) -> 30 (The blockages position)
                if (
                        # Checking if a given point collides with the blockage's position(start) and position+size(end)
                        (blockage[1][0] <= xs[i] < blockage[1][0] + blockage[0].shape[0] and
                         blockage[1][1] <= ys[i] < blockage[1][1] + blockage[0].shape[1])
                        or
                        (blockage[1][0] <= xs[i + 1] < blockage[1][0] + blockage[0].shape[0] and
                         blockage[1][1] <= ys[i + 1] < blockage[1][1] + blockage[0].shape[1])
                ):
                    print('Collision in 2D Block found')
                    return True
            return
        # True: intersection between segment and blockage
        # False: No intersection between segment and blockage

    def find_optimal_path(self, mission):
        print('Finding optimal path...')
        # TODO: Caspar: Maybe here call plot_space to show the workspace
        start_node = mission.start
        end_node = mission.end
        payload = mission.payload
        blockages = []
        wind_field = self.wind_field

        h = 10 * math.sqrt(2)
        directions = [
            (h, h, 0), (h, h, 3), (h, h, -3),
            (20, 0, 0), (20, 0, 3), (20, 0, -3),
            (h, -h, 0), (h, -h, 3), (h, -h, -3),

            (0, 20, 0), (0, 20, 3), (0, 20, -3),
            (0, 0, 0), (0, 0, 3), (0, 0, -3),
            (0, -20, 0), (0, -20, 3), (0, -20, -3),

            (-h, h, 0), (-h, h, 3), (-h, h, -3),
            (-20, 0, 0), (-20, 0, 3), (-20, 0, -3),
            (-h, -h, 0), (-h, -h, 3), (-h, -h, -3),
        ]

        velocities = (5, 10, 15)

        def calculate_time(current_node, next_node):
            d = distance(current_node, next_node)

            acc_hori = 5  # Horizontal acceleration = 5 m/s^2
            acc_verti = 1  # Vertical acceleration = 1 m/s^2

            # Calculate time to accelerate from current velocity to next velocity
            delta_velocity = next_node.velocity - current_node.velocity

            # Calculate time to accelerate in horizontal direction
            time_acceleration_hori = abs(delta_velocity.x) / acc_hori if acc_hori != 0 else float('inf')

            # Calculate time to accelerate in vertical direction
            time_acceleration_verti = abs(delta_velocity.y) / acc_verti if acc_verti != 0 else float('inf')

            # Choose the smaller time for acceleration
            time_acceleration = min(time_acceleration_hori, time_acceleration_verti)

            # Calculate distance traveled during acceleration
            distance_acceleration = 0.5 * (current_node.velocity + next_node.velocity) * time_acceleration

            # Calculate remaining distance
            remaining_distance = d - distance_acceleration

            if remaining_distance <= 0:
                return time_acceleration

            # Calculate time to travel remaining distance at next velocity
            time_travel = remaining_distance / next_node.velocity

            # Total time is sum of time to accelerate and time to travel remaining distance
            return time_acceleration + time_travel

        pq = [0, start_node]

        def get_neighbors(node):
            neighbors = []

            # if next to goal
            if math.sqrt((node.x - end_node.x) ** 2 + (node.y - end_node.y) ** 2) <= 20 and abs(
                    node.z - end_node.z) <= 3:
                end_node.velocity = 0
                neighbors.append(end_node)

            else:
                for dist_x, dist_y, dist_z in directions:
                    new_x = node.x + dist_x
                    new_y = node.y + dist_y
                    new_z = node.z + dist_z
                    new_node = Node(new_x, new_y, new_z)
                    if new_node not in blockages:
                        neighbors.append(new_node)
            return neighbors

        # TODO: Rune og Lucas: dist_x + dist_y <= 20, dist_z <= 3  --- se paper side 8 afsnit b
        def distance(node1, node2):
            dist_x = (node1.x - node2.x) ** 2
            dist_y = (node1.y - node2.y) ** 2
            dist_z = (node1.z - node2.z) ** 2
            return math.sqrt(dist_x + dist_y + dist_z)

        def heuristic_power(current_node, next_node):
            time = calculate_time(current_node, next_node)
            wind_speed = 0
            wind_angle = 0
            velocity_x = (next_node.x - current_node.x) / time
            velocity_y = (next_node.y - current_node.y) / time
            velocity_z = (next_node.y - current_node.y) / time
            linear_acceleration_x = velocity_x / time
            linear_acceleration_y = velocity_y / time
            linear_acceleration_z = velocity_z / time

            input_array = [[time, wind_speed, wind_angle,
                            next_node.x, next_node.y, next_node.z,
                            velocity_x, velocity_y, velocity_z,
                            linear_acceleration_x, linear_acceleration_y, linear_acceleration_z,
                            payload]]

            target_labels = ml_model.predict(input_array)

            power_joule = power(target_labels)

            # print("Heuristic power:", *power_joule)

            return power_joule

        def heuristic_distance(node):
            dist = distance(node, end_node)

            if node.z < 30:
                dist += 30 - node.z

            return dist

        pq = [(0, start_node)]

        visited = {start_node: 0}
        predecessor = {}

        while pq:
            _, current = heapq.heappop(pq)

            if current == end_node:
                break

            for neighbor in get_neighbors(current):
                # Calculate the energy for the neighbor using the heuristic function
                    for velocity in velocities:
                        neighbor_energy = visited[current] + heuristic_power(current, neighbor, velocity)

                        if neighbor not in visited or neighbor_energy < visited[neighbor]:
                            visited[neighbor] = neighbor_energy
                            predecessor[neighbor] = current
                            heapq.heappush(pq, (neighbor_energy, neighbor))

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
