import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

import numpy as np

from pathfinding import collision_detection


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

    # def add_wind_field(self, angle, wind_speed):
    #     # Convert wind angle to radians
    #     self.wind_angle_rad = np.radians(angle)
    #     self.wind_angle = angle
    #
    #     # Create a grid of coordinates
    #     x = np.linspace(0, self.max_bounds[0], self.grid_size)
    #     y = np.linspace(0, self.max_bounds[1], self.grid_size)
    #     X, Y = np.meshgrid(x, y)
    #
    #     # Initialize wind speed grid
    #     wind_speed_grid = np.zeros((self.grid_size, self.grid_size))
    #
    #     # Determine the starting point and direction based on the wind angle
    #     if 0 <= angle < 90:
    #         start_point = [0, 0]  # Wind comes from the bottom-left corner
    #         x_range = range(0, self.grid_size)
    #         y_range = range(0, self.grid_size)
    #     elif 90 <= angle < 180:
    #         start_point = [self.grid_size - 1, 0]  # Wind comes from the bottom-right corner
    #         x_range = range(self.grid_size - 1, -1, -1)
    #         y_range = range(0, self.grid_size)
    #     elif 180 <= angle < 270:
    #         start_point = [self.grid_size - 1, self.grid_size - 1]  # Wind comes from the top-right corner
    #         x_range = range(self.grid_size - 1, -1, -1)
    #         y_range = range(self.grid_size - 1, -1, -1)
    #     else:
    #         start_point = [0, self.grid_size - 1]  # Wind comes from the top-left corner
    #         x_range = range(0, self.grid_size)
    #         y_range = range(self.grid_size - 1, -1, -1)
    #
    #     # Iterate over the grid in the determined direction
    #     for i in x_range:
    #         for j in y_range:
    #             # Calculate the current point coordinates
    #             x = X[i, j]
    #             y = Y[i, j]
    #
    #             # Check for blockages at the current point
    #             if not collision_detection.check_segment_intersects_blockages(x, y, 0, self.blockages):
    #                 # Store the wind speed in the grid
    #                 wind_speed_grid[i, j] = wind_speed
    #             else:
    #                 # If there is a blockage, set the wind speed to 0
    #                 wind_speed_grid[i, j] = 0
    #
    #     # Store the wind speed grid in the wind field attribute
    #     self.wind_field = wind_speed_grid

    def add_wind_field(self, angle, wind_speed):
        # Convert wind angle to radians
        self.wind_angle_rad = np.radians(angle)
        self.wind_angle = angle

        # Create a grid of coordinates
        x = np.linspace(0, self.max_bounds[0], self.grid_size)
        y = np.linspace(0, self.max_bounds[1], self.grid_size)
        X, Y = np.meshgrid(x, y)

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

            for i in range(start_point[0], end_point[0] + x_step, x_step):
                for j in range(start_point[1], end_point[1] + y_step, y_step):
                    xs = [x, x + x_step]
                    ys = [y, y + y_step]

                    if not collision_detection.check_segment_intersects_blockages(xs, ys, [0, 0], self.blockages):
                        # Store the wind speed in the grid
                        wind_speed_grid[x, y] = wind_speed
                    else:
                        # If there is a blockage, stop casting the ray
                        break

                    # Move to the next grid cell along the ray direction
                    y += y_step
                y = start_point[1]
                x += x_step

        self.wind_field = wind_speed_grid

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
                ax.add_patch(plt.Rectangle((x, y), blockage_matrix.shape[0], blockage_matrix.shape[1], color='k'))

            if show_wind:
                grid_color = 'white'
                # Plot the wind speed grid with a colormap
                plt.imshow(self.wind_field, cmap='viridis', origin='lower',
                           extent=[0, self.max_bounds[0], 0, self.max_bounds[1]])
                plt.colorbar(label='Wind speed')

                # # Plot wind direction arrow
                # arrow_length = min(self.max_bounds) * 0.8  # Increase arrow length
                # ax.annotate('', xy=(0, self.max_bounds[1]), xytext=(0 - arrow_length * np.cos(self.wind_angle_rad),
                #                                                     self.max_bounds[1] - arrow_length * np.sin(
                #                                                         self.wind_angle_rad)),
                #             arrowprops=dict(facecolor='blue', edgecolor='blue'))  # Change arrow color to blue
                #
                # Plot wind direction arrow
                # Calculate the endpoint of the arrow
                arrow_length = 50

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
        print('find optimal path')
        # Maybe here call plot_space to show the workspace
        start = mission.start
        end = mission.end
        payload = mission.payload
        blockages = self.blockages
        wind_field = self.wind_field

        path_example = [(0, 0, 0), (50, 50, 0), (200, 200, 0), (225, 225, 0), (250, 250, 0)]

        path = path_example
        return path

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

    # def plot_wind_start_point(self, angle, wind_speed):
    #     # Convert wind angle to radians
    #     wind_angle_rad = np.radians(angle)
    #
    #     # Calculate the starting point for the wind vectors based on the wind angle
    #     if 0 <= angle < 90:
    #         start_point = [0, 0]  # Wind comes from the bottom-left corner
    #     elif 90 <= angle < 180:
    #         start_point = [self.max_bounds[0], 0]  # Wind comes from the bottom-right corner
    #     elif 180 <= angle < 270:
    #         start_point = [self.max_bounds[0], self.max_bounds[1]]  # Wind comes from the top-right corner
    #     else:
    #         start_point = [0, self.max_bounds[1]]  # Wind comes from the top-left corner
    #
    #     # Create a 2D grid representing the wind field starting from the calculated start_point
    #     x_grid = np.linspace(start_point[0], start_point[0] + self.max_bounds[0], num=len(self.blockages))
    #     y_grid = np.linspace(start_point[1], start_point[1] + self.max_bounds[1], num=len(self.blockages))
    #     X, Y = np.meshgrid(x_grid, y_grid)
    #
    #     # Calculate wind vectors (u, v) based on wind speed and angle
    #     u = wind_speed * np.cos(wind_angle_rad)
    #     v = wind_speed * np.sin(wind_angle_rad)
    #
    #     # Store the wind vectors in the wind field attribute
    #     self.wind_field = {'X': X, 'Y': Y, 'u': u, 'v': v}
    #
    # def plot_rest_wind_point(self, angle, wind_speed):
    #     start_point = self.wind_field[0]

    # def plot_wind_vectors(self, angle, wind_speed):
    #     # Convert wind angle to radians
    #     wind_angle_rad = np.radians(angle)
    #
    #     # Calculate the starting point for the first wind vector based on the wind angle
    #     if 0 <= angle < 90:
    #         start_point = [0, 0]  # Wind comes from the bottom-left corner
    #     elif 90 <= angle < 180:
    #         start_point = [self.max_bounds[0], 0]  # Wind comes from the bottom-right corner
    #     elif 180 <= angle < 270:
    #         start_point = [self.max_bounds[0], self.max_bounds[1]]  # Wind comes from the top-right corner
    #     else:
    #         start_point = [0, self.max_bounds[1]]  # Wind comes from the top-left corner
    #
    #     # Initialize wind vectors list
    #     wind_vectors = []
    #
    #     # Iterate over the workspace grid and plot wind vectors
    #     for i in range(40):
    #         for j in range(40):
    #             # Calculate the current point coordinates
    #             x = start_point[0] + i * (self.max_bounds[0] / (self.resolution - 1))
    #             y = start_point[1] + j * (self.max_bounds[1] / (self.resolution - 1))
    #
    #             # Check for blockages at the current point
    #             if not collision_detection.check_segment_intersects_blockages(x, y, 0, self.blockages):
    #                 # Calculate wind vector components (u, v) based on wind speed and angle
    #                 u = wind_speed * np.cos(wind_angle_rad)
    #                 v = wind_speed * np.sin(wind_angle_rad)
    #                 wind_vectors.append({'x': x, 'y': y, 'u': u, 'v': v})
    #
    #     # Store the wind vectors in the wind field attribute
    #     self.wind_field = wind_vectors

    # def plot_wind_vectors(self, angle, wind_speed):
    #     # Convert wind angle to radians
    #     wind_angle_rad = np.radians(angle)
    #
    #     # Calculate the starting point for the first wind vector based on the wind angle
    #     if 0 <= angle < 90:
    #         start_point = [0, 0]  # Wind comes from the bottom-left corner
    #         x_range = range(0, self.max_bounds[0])
    #         y_range = range(0, self.max_bounds[1])
    #         direction = '+', '+'
    #     elif 90 <= angle < 180:
    #         start_point = [self.max_bounds[0], 0]  # Wind comes from the bottom-right corner
    #         x_range = range(self.max_bounds[0], 0, -1)
    #         y_range = range(0, self.max_bounds[1])
    #         direction = '-', '+'
    #     elif 180 <= angle < 270:
    #         start_point = [self.max_bounds[0], self.max_bounds[1]]  # Wind comes from the top-right corner
    #         x_range = range(self.max_bounds[0], 0, -1)
    #         y_range = range(self.max_bounds[1], 0, -1)
    #         direction = '-', '-'
    #     else:
    #         start_point = [0, self.max_bounds[1]]  # Wind comes from the top-left corner
    #         x_range = range(0, self.max_bounds[0])
    #         y_range = range(self.max_bounds[1], 0, -1)
    #         direction = '+', '-'
    #
    #
    #     # In the x and y ranges, we denote the loop start, end and direction(either positive or negative)
    #
    #
    #     # Initialize wind vectors list
    #     wind_vectors = []
    #
    #     # Iterate over the workspace grid and plot wind vectors
    #     for i in x_range:
    #         for j in y_range:
    #
    #             if x_range.step is not None:
    #                 i = x_range.step*i
    #             if y_range.step is not None:
    #                 j = y_range.step*j
    #
    #             # Calculate the current point coordinates
    #             x = start_point[0] + i
    #             y = start_point[1] + j
    #
    #             # Check for blockages at the current point
    #             if not collision_detection.check_segment_intersects_blockages(x, y, 0, self.blockages):
    #                 # Calculate wind vector components (u, v) based on wind speed and angle
    #                 u = wind_speed * np.cos(wind_angle_rad)
    #                 v = wind_speed * np.sin(wind_angle_rad)
    #                 wind_vectors.append({'x': x, 'y': y, 'u': u, 'v': v})
    #
    #     # Store the wind vectors in the wind field attribute
    #     self.wind_field = wind_vectors

    import numpy as np

    # def plot_wind_vectors(self, angle, wind_speed):
    #     # Convert wind angle to radians
    #     wind_angle_rad = np.radians(angle)
    #
    #
    #     # Create a grid of coordinates
    #     x = np.linspace(0, self.max_bounds[0], self.grid_size)
    #     y = np.linspace(0, self.max_bounds[1], self.grid_size)
    #     X, Y = np.meshgrid(x, y)
    #
    #     # Initialize wind vectors list
    #     wind_vectors = []
    #
    #     # Iterate over the grid and calculate wind vectors
    #     for i in range(self.grid_size):
    #         for j in range(self.grid_size):
    #             # Calculate the current point coordinates
    #             x = X[i, j]
    #             y = Y[i, j]
    #
    #             # Check for blockages at the current point
    #             if not collision_detection.check_segment_intersects_blockages(x, y, 0, self.blockages):
    #                 # Calculate wind vector components (u, v) based on wind speed and angle
    #                 u = wind_speed * np.cos(wind_angle_rad)
    #                 v = wind_speed * np.sin(wind_angle_rad)
    #                 wind_vectors.append({'x': x, 'y': y, 'u': u, 'v': v})
    #
    #     # Store the wind vectors in the wind field attribute
    #     self.wind_field = wind_vectors
