from pathfinding.Mission import Mission
from pathfinding.Workspace import Workspace
from pathfinding.Node import Node
import numpy as np


def setup_workspace():
    # Example usage
    dimensions = 3  # Define a 3-dimensional space
    max_bounds = [400, 400, 60]  # Define maximum bounds for each dimension

    space = Workspace(dimensions, max_bounds)

    # Add a blockage (building) represented as a matrix
    blockage_matrix_1 = np.ones((40, 80, 60))  # Define a 2x2x3 blockage matrix
    position_1 = [120, 120, 0]  # Specify the position of the blockage
    space.add_blockage(blockage_matrix_1, position_1)

    # Add a blockage (building) represented as a matrix
    blockage_matrix_2 = np.ones((50, 50, 60))  # Define a 2x2x3 blockage matrix
    position_2 = [250, 250, 0]  # Specify the position of the blockage
    space.add_blockage(blockage_matrix_2, position_2)

    # Add Windfield
    # Call add_wind_field
    return space


def find_and_show_optimal_path():
    workspace = setup_workspace()
    mission = Mission(Node(0, 0, 0), Node(200, 350, 0), 500)

    flight_path = workspace.find_optimal_path(mission)
    # flight_path = generate_random_path(workspace)
    workspace.add_flight_path(flight_path=flight_path)

    # Options: 2D or 3D
    workspace.plot_space(dimension='2D', dpi=800)
    workspace.plot_space(dimension='3D', dpi=800)


if __name__ == '__main__':
    find_and_show_optimal_path()
