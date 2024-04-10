import matplotlib.pyplot as plt
import numpy as np

from pathfinding import collision_detection


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

                    # Check if the current segment intersects with any blockage
                    segment_intersects = any(
                        collision_detection.check_segment_intersects_blockage(x_coords, y_coords, z_coords, blockage)
                        for blockage in self.blockages
                    )

                    # Plot the segment in the appropriate color
                    if segment_intersects:
                        ax.plot(x_coords, y_coords, z_coords, color='r', alpha=0.5, zorder=-1)
                    else:
                        ax.plot(x_coords, y_coords, z_coords, color='g', alpha=0.5, zorder=1)
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
                    segment_intersects = any(
                        collision_detection.check_segment_intersects_blockage(x_coords, y_coords, z_coords,
                                                                              blockage)
                        for blockage in self.blockages
                    )

                    # Plot the segment in the appropriate color
                    if segment_intersects:
                        ax.plot(x_coords, y_coords, color='r', alpha=0.5)
                    else:
                        ax.plot(x_coords, y_coords, color='g', alpha=0.5)
            return ax

    def plot_space(self, dimension='3D', dpi=300):
        if dimension == '3D':
            fig = plt.figure(dpi=dpi)
            ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

            # Plot origin
            ax.scatter(0, 0, 0, color='k')

            # Plot blockages
            for blockage_matrix, position in self.blockages:
                x, y, z = position
                ax.bar3d(x, y, z, *blockage_matrix.shape, color='k', alpha=0.5, edgecolor='black', linewidth=0.5,
                         zorder=0)

            # z-order: the higher the more in front

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
                ax.add_patch(
                    plt.Rectangle((x, y), blockage_matrix.shape[0], blockage_matrix.shape[1], color='k'))

            ax = self.plot_flight_paths(ax, dimension='2D')

            # Set labels and limits
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim([0, self.max_bounds[0]])
            ax.set_ylim([0, self.max_bounds[1]])
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
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
