from pathfinding.Mission import Mission
from pathfinding.Workspace import Workspace
import numpy as np


def generate_random_path(workspace):
    # Define dimensions of the 3D space
    space_dimensions = (100, 100, 50)

    # Access blockages directly from the workspace
    blockages = workspace.blockages

    # Generate random flight path data
    num_points = 50  # Number of points in the flight path
    min_coord, max_coord = 0, 99  # Range for coordinates in each dimension

    # Generate random x, y, z coordinates for the flight path while avoiding blockages
    flight_path = []
    for _ in range(num_points):
        # Generate random coordinates
        x_coord = np.random.randint(min_coord, max_coord)
        y_coord = np.random.randint(min_coord, max_coord)
        z_coord = np.random.randint(min_coord, max_coord // 4)  # Adjusted for z-axis (height)
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


def setup_workspace():
    # Example usage
    dimensions = 3  # Define a 3-dimensional space
    max_bounds = [100, 100, 50]  # Define maximum bounds for each dimension

    space = Workspace(dimensions, max_bounds)

    # Add a blockage (building) represented as a matrix
    blockage_matrix_1 = np.ones((10, 20, 50))  # Define a 2x2x3 blockage matrix
    position_1 = [30, 30, 0]  # Specify the position of the blockage
    space.add_blockage(blockage_matrix_1, position_1)

    # Add a blockage (building) represented as a matrix
    blockage_matrix_2 = np.ones((10, 10, 50))  # Define a 2x2x3 blockage matrix
    position_2 = [60, 60, 0]  # Specify the position of the blockage
    space.add_blockage(blockage_matrix_2, position_2)

    # Add Windfield
    #Call add_wind_field
    return space


def find_and_show_optimal_path():
    workspace = setup_workspace()
    mission = Mission((0, 0, 0), (50, 50, 0), 500)

    flight_path = workspace.find_optimal_path(mission)
    # flight_path = generate_random_path(workspace)
    workspace.add_flight_path(flight_path=flight_path)

    # Options: 2D or 3D
    workspace.plot_space(dimension='2D', dpi=600)
    workspace.plot_space(dimension='3D', dpi=600)


if __name__ == '__main__':
    find_and_show_optimal_path()
