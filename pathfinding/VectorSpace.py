import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class VectorSpace:
    def __init__(self, dimensions, max_bounds):
        self.dimensions = dimensions
        self.max_bounds = max_bounds
        self.vectors = []
        self.blockages = []
        self.flight_paths = []

    def insert_vector(self, vector):
        if len(vector) != self.dimensions:
            raise ValueError(f"Vector size must be {self.dimensions}")
        for i in range(self.dimensions):
            if not (0 <= vector[i] <= self.max_bounds[i]):
                raise ValueError(f"Vector coordinates must be within the specified bounds")
        self.vectors.append(vector)

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
        if len(flight_path[0]) != self.dimensions:
            raise ValueError(f"Flight path vectors must match the space dimensions")
        self.flight_paths.append(flight_path)

    def plot_space(self, dimension='3D'):
        if dimension == '3D':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot origin
            ax.scatter(0, 0, 0, color='k')

            # Plot vectors
            for vector in self.vectors:
                ax.quiver(0, 0, 0, vector[0], vector[1], vector[2])

            # Plot blockages
            for blockage_matrix, position in self.blockages:
                x, y, z = position
                ax.bar3d(x, y, z, *blockage_matrix.shape, color='r', alpha=0.5)

            # Plot flight paths
            for flight_path in self.flight_paths:
                xs, ys, zs = zip(*flight_path)
                ax.plot(xs, ys, zs, color='b', alpha=0.5)

            # Set labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Set limits
            ax.set_xlim([0, self.max_bounds[0]])
            ax.set_ylim([0, self.max_bounds[1]])
            ax.set_zlim([0, self.max_bounds[2]])

        elif dimension == '2D':
            fig, ax = plt.subplots()

            # Plot vectors
            for vector in self.vectors:
                ax.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color='k')

            # Plot blockages
            for blockage_matrix, position in self.blockages:
                x, y = position[:2]
                ax.add_patch(plt.Rectangle((x, y), blockage_matrix.shape[0], blockage_matrix.shape[1], color='r'))

            # Plot flight paths
            for flight_path in self.flight_paths:
                xs, ys = zip(*[(x, y) for x, y, _ in flight_path])
                ax.plot(xs, ys, color='b', alpha=0.7, linewidth=2)  # Adjust transparency and line width

            # Set labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

            # Set limits
            ax.set_xlim([0, self.max_bounds[0]])
            ax.set_ylim([0, self.max_bounds[1]])

            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid(True)

        plt.show()



# Example usage
dimensions = 3  # Define a 3-dimensional space
max_bounds = [100, 100, 50]  # Define maximum bounds for each dimension

space = VectorSpace(dimensions, max_bounds)

# Insert vectors into the space
# space.insert_vector([1, 2, 3])
# space.insert_vector([4, 5, 6])

# Add a blockage (building) represented as a matrix
blockage_matrix_1 = np.ones((10, 20, 30))  # Define a 2x2x3 blockage matrix
position_1 = [30, 30, 0]  # Specify the position of the blockage
space.add_blockage(blockage_matrix_1, position_1)

# Add a blockage (building) represented as a matrix
blockage_matrix_2 = np.ones((10, 10, 20))  # Define a 2x2x3 blockage matrix
position_2 = [60, 60, 0]  # Specify the position of the blockage
space.add_blockage(blockage_matrix_2, position_2)

# Add a flight path
flight_path = [(10, 10, 5), (20, 20, 10), (30, 30, 15), (40, 40, 20)]  # Define a sequence of vectors
space.add_flight_path(flight_path)

# Plot the space
space.plot_space(dimension='3D')
