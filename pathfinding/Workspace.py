import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from pathfinding import collision_detection


class Workspace:
    def __init__(self, dimensions, max_bounds, mission):
        self.dimensions = dimensions
        self.max_bounds = max_bounds
        self.flight_paths = []
        self.blockages = []
        self.wind_field = []
        self.wind_angle_rad = 0
        self.wind_angle = 0
        self.grid_size = 100
        self.mission = mission

    def add_blockage(self, blockage):
        if len(blockage.np_array.shape) != self.dimensions:
            raise ValueError(f"Blockage matrix dimensions must match the space dimensions")
        for i in range(self.dimensions):
            if not (0 <= blockage.positions[i] <= self.max_bounds[i]):
                raise ValueError(f"Blockage position must be within the specified bounds")
            if blockage.positions[i] + blockage.np_array.shape[i] > self.max_bounds[i]:
                raise ValueError(f"Blockage does not fit within the space dimensions")
        self.blockages.append(blockage)
        # print(self.blockages)

    def add_flight_path(self, flight_path):
        self.flight_paths.append(flight_path)

    def plot_flight_paths(self, ax, dimension):
        def plot_segment(segment_x_coords, segment_y_coords, segment_z_coords, segment_color):
            if dimension == '3D':
                ax.plot(segment_x_coords, segment_y_coords, segment_z_coords, color=segment_color, alpha=0.5)
            elif dimension == '2D':
                ax.plot(segment_x_coords, segment_y_coords, color=segment_color, alpha=0.5)

        # Plot flight paths
        for flight_path in self.flight_paths:
            xs, ys, zs = zip(*flight_path.path)

            # Plot origin/start
            color = 'olive'
            size = 20
            if dimension == '3D':
                ax.scatter(xs[0], ys[0], zs[0], s=size, color=color)
                ax.scatter(xs[-1], ys[-1], zs[-1], s=size, color=color)
            elif dimension == '2D':
                ax.scatter(xs[0], ys[0], s=size, color=color)
                ax.scatter(xs[-1], ys[-1], s=size, color=color)

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

                if segment_intersects:
                    plot_segment(x_coords, y_coords, z_coords, segment_color='r')
                else:
                    if flight_path.path_type == 'baseline':
                        plot_segment(x_coords, y_coords, z_coords, segment_color='b')
                    elif flight_path.path_type == 'optimal':
                        plot_segment(x_coords, y_coords, z_coords, segment_color='g')
        return ax

    def plot_space(self, dimension='3D', dpi=300, show_wind=False):
        if dimension == '3D':
            fig = plt.figure(dpi=dpi)
            ax = fig.add_subplot(111, projection='3d')

            # Plot blockages
            ax = self.plot_blockages(ax, dimension='3D')
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

            # Plot blockages
            ax = self.plot_blockages(ax, dimension='2D')

            # Plot optional wind
            if show_wind:
                ax = self.plot_wind(ax)

            ax = self.plot_flight_paths(ax, dimension='2D')

            # Set labels and limits
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim([0, self.max_bounds[0]])
            ax.set_ylim([0, self.max_bounds[1]])
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, color=grid_color, linestyle='--', linewidth=0.5, alpha=0.5)
        plt.show()

    def plot_blockages(self, ax, dimension):
        if dimension == '3D':
            for blockage_matrix in self.blockages:
                x, y, z = blockage_matrix.positions
                ax.bar3d(x, y, z, *blockage_matrix.np_array.shape, color='k', alpha=0.5, edgecolor='black',
                         linewidth=0.5)
        elif dimension == '2D':
            for blockage_matrix in self.blockages:
                x, y = blockage_matrix.positions[:2]
                ax.add_patch(
                    plt.Rectangle((x, y), blockage_matrix.np_array.shape[0], blockage_matrix.np_array.shape[1],
                                  color='k', alpha=0.5))
        return ax

    def add_wind_field(self, angle, wind_speed):
        self.wind_field = angle, wind_speed

    # def add_wind_field(self, angle, wind_speed):
    #     # Convert wind angle to radians
    #     self.wind_angle_rad = np.radians(angle)
    #     self.wind_angle = angle
    #
    #     # Initialize wind speed grid
    #     wind_speed_grid = np.zeros((self.max_bounds[0], self.max_bounds[1]))
    #
    #     # These are built around how the wind affects our workspace from outside.
    #
    #     # Determine the starting point and direction based on the wind angle
    #     if 0 <= angle < 90:
    #         start_point = [self.max_bounds[0] - 1, self.max_bounds[1] - 1]  # Wind comes from the top-right corner
    #         end_point = [self.max_bounds[0] - self.max_bounds[0], self.max_bounds[1] - self.max_bounds[1]]
    #         x_step = -1
    #         y_step = -1
    #     elif 90 <= angle < 180:
    #         start_point = [self.max_bounds[0] - self.max_bounds[0],
    #                        self.max_bounds[1] - 1]  # Wind comes from the top-left corner
    #         end_point = [self.max_bounds[0] - 1, self.max_bounds[1] - self.max_bounds[1]]
    #
    #         x_step = 1
    #         y_step = -1
    #
    #     elif 180 <= angle < 270:
    #         start_point = [self.max_bounds[0] - self.max_bounds[0],
    #                        self.max_bounds[1] - self.max_bounds[1]]  # Wind comes from the bottom-left corner
    #         end_point = [self.max_bounds[0] - 1, self.max_bounds[1] - 1]
    #
    #         x_step = 1
    #         y_step = 1
    #     else:
    #         start_point = [self.max_bounds[0] - 1,
    #                        self.max_bounds[1] - self.max_bounds[1]]  # Wind comes from the bottom-right corner
    #         end_point = [self.max_bounds[0] - self.max_bounds[0], self.max_bounds[1] - 1]
    #
    #         x_step = -1
    #         y_step = 1
    #
    #     # Cast rays from the starting point until reaching the bounds or hitting a blockage
    #     x, y = start_point
    #
    #     while 0 <= x < self.max_bounds[0] and 0 <= y < self.max_bounds[1]:
    #         # Check for blockages at the current point
    #
    #         # print("Current position:", x, y)  # Print current position
    #
    #         for i in range(start_point[0], end_point[0] + x_step, x_step):
    #             xs = [start_point[0], x + x_step]
    #
    #             for j in range(start_point[1], end_point[1] + y_step, y_step):
    #                 ys = [start_point[1], y + y_step]
    #
    #                 # print("Segment coordinates:", xs, ys)  # Print segment coordinates
    #
    #                 if not collision_detection.check_segment_intersects_blockages(xs, ys, [0, 0], self.blockages):
    #                     # Store the wind speed in the grid
    #                     wind_speed_grid[x, y] = wind_speed
    #                 else:
    #                     # If there is a blockage, stop casting the ray
    #                     # print('Found a blockage')
    #                     pass
    #
    #                 # Move to the next grid cell along the ray direction
    #                 y += y_step
    #             y = start_point[1]
    #             x += x_step
    #
    #     self.wind_field = self.rotate_wind_grid(wind_speed_grid)
    #
    # def rotate_wind_grid(self, wind_speed_grid):
    #     # Calculate the rotation angle based on the wind angle
    #     rotation_angle = self.wind_angle % 360  # Ensure angle is within [0, 360)
    #
    #     if 0 <= rotation_angle < 90:  # Wind comes from the top-right corner
    #         k = 0
    #
    #     elif 90 <= rotation_angle < 180:  # Wind comes from the top-left corner
    #         k = 2
    #         # wind_speed_grid = np.flip(wind_speed_grid, axis=0)
    #
    #     elif 180 <= rotation_angle < 270:  # Wind comes from the bottom-left corner
    #         k = 0
    #
    #     else:  # Wind comes from the bottom-right corner
    #         k = -2
    #         # wind_speed_grid = np.flip(wind_speed_grid, axis=1)
    #
    #     rotated_wind_speed_grid = np.rot90(wind_speed_grid, k=k, axes=(0, 1))
    #
    #     return rotated_wind_speed_grid
    #
    # def plot_wind(self, ax):
    #     grid_color = 'white'
    #     # Plot the wind speed grid with a colormap
    #     plt.imshow(self.wind_field, cmap='viridis', origin='lower',
    #                extent=[0, self.max_bounds[0], 0, self.max_bounds[1]])
    #     plt.colorbar(label='Wind speed')
    #
    #     # Plot wind direction arrow
    #     # Calculate the endpoint of the arrow
    #     arrow_length = self.max_bounds[0] / 10
    #
    #     arrow_end_x = self.max_bounds[0] / 2 + arrow_length * np.cos(self.wind_angle_rad)
    #     arrow_end_y = self.max_bounds[1] / 2 + arrow_length * np.sin(self.wind_angle_rad)
    #
    #     ax.annotate('',
    #                 xy=(self.max_bounds[0] / 2, self.max_bounds[1] / 2),
    #                 xytext=(arrow_end_x, arrow_end_y),
    #                 arrowprops=dict(facecolor='blue', edgecolor='blue', alpha=0.4,
    #                                 path_effects=[path_effects.withSimplePatchShadow(offset=(-1, -1))]))
    #     return ax

    # # For testing collision detection
    # def generate_random_path(self):
    #     # Access blockages directly from the workspace
    #     blockages = self.blockages
    #
    #     # Generate random flight path data
    #     num_points = 50  # Number of points in the flight path
    #     min_coord, max_coord = 0, 400  # Range for coordinates in each dimension of x and z
    #     min_coord_z, max_coord_z = 0, 60
    #
    #     # Generate random x, y, z coordinates for the flight path while avoiding blockages
    #     flight_path = []
    #     for _ in range(num_points):
    #         # Generate random coordinates
    #         x_coord = np.random.randint(min_coord, max_coord)
    #         y_coord = np.random.randint(min_coord, max_coord)
    #         z_coord = np.random.randint(min_coord_z, max_coord_z)  # Adjusted for z-axis (height)
    #         print(f"Generated coordinates: ({x_coord}, {y_coord}, {z_coord})")
    #
    #         # Check if the generated coordinates are within any blockages
    #         while any(
    #                 blockage[1][0] <= x_coord < blockage[1][0] + blockage[0].shape[0] and
    #                 blockage[1][1] <= y_coord < blockage[1][1] + blockage[0].shape[1] and
    #                 blockage[1][2] <= z_coord < blockage[1][2] + blockage[0].shape[2]
    #                 for blockage in blockages
    #         ):
    #             # Regenerate coordinates until they are outside all blockages
    #             x_coord = np.random.randint(min_coord, max_coord)
    #             y_coord = np.random.randint(min_coord, max_coord)
    #             z_coord = np.random.randint(min_coord, max_coord // 4)
    #
    #             print(f"Regenerated coordinates: ({x_coord}, {y_coord}, {z_coord})")
    #
    #         # Append the valid coordinates to the flight path
    #         flight_path.append((x_coord, y_coord, z_coord))
    #
    #     return flight_path
    #
    # # For testing collision detection
    # @staticmethod
    # def generate_path_completely_fill_blockage(grid_size=400, blockage_size=50, blockage_x=250, blockage_y=250):
    #     path = []
    #
    #     # Moving along the top boundary
    #     for x in range(0, grid_size):
    #         path.append((x, 0, 0))
    #
    #     # Zigzag pattern towards the blockage from the top
    #     for y in range(1, grid_size):
    #         if y % 2 == 1:  # Odd rows move right
    #             for x in range(grid_size - 1, -1, -1):
    #                 path.append((x, y, 0))
    #         else:  # Even rows move left
    #             for x in range(0, grid_size):
    #                 path.append((x, y, 0))
    #
    #     return path
