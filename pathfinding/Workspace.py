import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np

from pathfinding import collision_detection, Blockage


class Workspace:
    def __init__(self, dimensions, max_bounds, mission):
        self.dimensions = dimensions
        self.max_bounds = max_bounds
        self.flight_paths = []
        self.blockages = []
        self.wind_field = []
        self.wind_angle_rad = np.radians(90)
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

    def add_flight_path(self, flight_path):
        self.flight_paths.append(flight_path)

    def set_wind(self, wind_speed, wind_angle):
        self.wind_angle_rad = np.radians(90 - wind_angle)  # Adjusting for 0 being south
        self.wind_angle = wind_angle
        wind_speed_grid = np.ones((self.max_bounds[0], self.max_bounds[1], self.max_bounds[2]))

        for blockage in self.blockages:
            if 315 < wind_angle or wind_angle <= 45: # Wind coming from north
                start_x = blockage.positions[0]
                start_y = 0
                start_z = 0
                end_x = blockage.np_array.shape[0] + start_x
                end_y = blockage.positions[1] + blockage.np_array.shape[1]
                end_z = blockage.np_array.shape[2]

            elif 45 < wind_angle <= 135: # Wind coming from east
                start_x = 0
                start_y = blockage.positions[1]
                start_z = 0
                end_x = blockage.positions[0]
                end_y = blockage.positions[1] + blockage.np_array.shape[1]
                end_z = blockage.np_array.shape[2]

            elif 135 < wind_angle <= 225: # Wind coming from south
                start_x = blockage.positions[0]
                start_y = blockage.positions[1]
                start_z = 0
                end_x = blockage.positions[0] + blockage.np_array.shape[0]
                end_y = self.max_bounds[1]
                end_z = blockage.np_array.shape[2]
            else: # 225 < wind_angle <= 315, Wind coming from west
                start_x = blockage.positions[0]
                start_y = blockage.positions[1]
                start_z = 0
                end_x = self.max_bounds[0]
                end_y = blockage.positions[1] + blockage.np_array.shape[1]
                end_z = blockage.np_array.shape[2]

            for i in range(start_x, end_x):
                for j in range(start_y, end_y):
                    for k in range(start_z, end_z):
                        wind_speed_grid[j, i, k] = 0

            for i in range(wind_speed_grid.shape[0]):
                for j in range(wind_speed_grid.shape[1]):
                    for k in range(wind_speed_grid.shape[2]):
                        if wind_speed_grid[i, j, k] == 1:
                            wind_speed_grid[i, j, k] = wind_speed

        self.wind_field = wind_speed_grid

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
            elif dimension == 'XZ':
                ax.scatter(xs[0], zs[0], s=size, color=color)
                ax.scatter(xs[-1], zs[-1], s=size, color=color)

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
        grid_color = 'gray'

        if dimension == '3D':
            fig = plt.figure(dpi=dpi)
            ax = fig.add_subplot(111, projection='3d')

            # Plot blockages and flight_path
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
            fig, ax = plt.subplots(dpi=dpi)

            # Plot blockages and flight_path
            ax = self.plot_blockages(ax, dimension='2D')
            ax = self.plot_flight_paths(ax, dimension='2D')

            if show_wind:
                ax = self.plot_wind(ax)

            # Set labels and limits
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim([0, self.max_bounds[0]])
            ax.set_ylim([0, self.max_bounds[1]])
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, color=grid_color, linestyle='--', linewidth=0.5, alpha=0.5)

        elif dimension == 'XZ':
            fig, ax = plt.subplots(dpi=dpi)

            # Plot blockages
            ax = self.plot_blockages(ax, dimension='XZ')
            ax = self.plot_flight_paths(ax, dimension='XZ')

            # Set labels and limits
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_xlim([0, self.max_bounds[0]])
            ax.set_ylim([0, self.max_bounds[2]])
            ax.set_aspect(5, adjustable='box')
            ax.grid(True, color=grid_color, linestyle='--', linewidth=0.5, alpha=0.5)

        plt.show()

    def check_blockage_in_line_of_sight(self, x, y, wind_direction_rad):
        # Iterate over each blockage and check if it obstructs the line of sight of the wind
        for blockage in self.blockages:
            # Calculate the angle of the line connecting the current point to the blockage
            angle_to_blockage = np.arctan2(blockage.positions[1] - y, blockage.positions[0] - x)

            # Calculate the angle difference between the wind direction and the angle to the blockage
            angle_difference = np.abs(wind_direction_rad - angle_to_blockage)

            # If the angle difference is within a threshold, the blockage obstructs the line of sight
            if angle_difference < np.pi / 2:  # Adjust the threshold as needed
                return True

        return False

    def plot_blockages(self, ax, dimension):
        if dimension == '3D':
            for blockage_matrix in self.blockages:
                x, y, z = blockage_matrix.positions
                ax.bar3d(x, y, z, *blockage_matrix.np_array.shape, color='k', alpha=0.2, edgecolor='black',
                         linewidth=0.5)
        elif dimension == '2D':
            for blockage_matrix in self.blockages:
                x, y = blockage_matrix.positions[:2]
                ax.add_patch(
                    plt.Rectangle((x, y), blockage_matrix.np_array.shape[0], blockage_matrix.np_array.shape[1],
                                  color='k', alpha=0.4))
        elif dimension == 'XZ':
            for blockage_matrix in self.blockages:
                x, _, z = blockage_matrix.positions
                ax.add_patch(
                    plt.Rectangle((x, z), blockage_matrix.np_array.shape[0], blockage_matrix.np_array.shape[2],
                                  color='k', alpha=0.4))
        return ax

    def plot_wind(self, ax):
        # Plot the wind speed grid for x and y dimensions
        wind_field = self.wind_field[:, :, 0]
        plt.imshow(wind_field, cmap='viridis', origin='lower', alpha=0.5)
        plt.colorbar(label='Wind speed')

        # Plot wind direction arrow
        # Calculate the endpoint of the arrow
        arrow_length = self.max_bounds[0] / 10

        arrow_end_x = self.max_bounds[0] / 2 + arrow_length * np.cos(self.wind_angle_rad)
        arrow_end_y = self.max_bounds[1] / 2 + arrow_length * np.sin(self.wind_angle_rad)

        ax.annotate('',
                    xy=(self.max_bounds[0] / 2, self.max_bounds[1] / 2),
                    xytext=(arrow_end_x, arrow_end_y),
                    arrowprops=dict(facecolor='red', edgecolor='red', alpha=0.4,
                                    path_effects=[path_effects.withSimplePatchShadow(offset=(-1, -1))]))
        return ax
