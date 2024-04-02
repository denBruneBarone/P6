from pathfinding.Workspace import Workspace
import numpy as np


def generate_random_path():
    # Define dimensions of the 3D space
    space_dimensions = (100, 100, 50)

    # Generate random flight path data
    num_points = 50  # Number of points in the flight path
    min_coord, max_coord = 0, 99  # Range for coordinates in each dimension

    # Generate random x, y, z coordinates for the flight path
    x_coords = np.random.randint(min_coord, max_coord, size=num_points)
    y_coords = np.random.randint(min_coord, max_coord, size=num_points)
    z_coords = np.random.randint(min_coord, max_coord // 2, size=num_points)  # Limit z-coordinates to half the depth

    # Combine x, y, z coordinates into a flight path and return it
    return list(zip(x_coords, y_coords, z_coords))


# Example usage
dimensions = 3  # Define a 3-dimensional space
max_bounds = [200, 200, 100]  # Define maximum bounds for each dimension

space = Workspace(dimensions, max_bounds)

# Add a blockage (building) represented as a matrix
blockage_matrix_1 = np.ones((10, 20, 30))  # Define a 2x2x3 blockage matrix
position_1 = [30, 30, 0]  # Specify the position of the blockage
space.add_blockage(blockage_matrix_1, position_1)

# Add a blockage (building) represented as a matrix
blockage_matrix_2 = np.ones((10, 10, 20))  # Define a 2x2x3 blockage matrix
position_2 = [60, 60, 0]  # Specify the position of the blockage
space.add_blockage(blockage_matrix_2, position_2)

# Plot the space
flight_path = generate_random_path()
# Options: 2D or 3D
space.plot_space(dimension='2D', flight_path=flight_path)
