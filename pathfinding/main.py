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
    # space.add_wind_field(45, 10)

    return space


def find_and_show_optimal_path():
    workspace = setup_workspace()
    mission = Mission(Node(0, 0, 0), Node(325, 325, 0), 500)
    flight_path = workspace.find_optimal_path(mission)
    # flight_path = workspace.find_baseline_path(mission)

    # flight_path = [(0, 0, 0),
    #                (250, 250, 30),  # 1
    #                (300, 300, 30),  # 2
    #                (320, 320, 30), (230, 320, 30),
    #                (250, 300, 30),  # 3
    #                (275, 275, 30),
    #                (300, 250, 30),  # 4
    #                (320, 230, 30), (230, 230, 30), (230, 275, 30), (310, 275, 30), (310, 310, 30),
    #                (275, 310, 30), (275, 275, 30), (275, 230, 30), (275, 120, 30), (120, 120, 30), (120, 200, 30),
    #                (160, 200, 30), (160, 120, 30), (160, 119, 30), (119, 119, 30), (119, 201, 30),
    #                (161, 201, 30), (161, 119, 30)]

    # flight_path = [(0, 0, 0), (10, 10, 10)]

    workspace.add_flight_path(flight_path=flight_path)

    # Options: 2D or 3D
    workspace.plot_space(dimension='2D', dpi=800, show_wind=False)
    workspace.plot_space(dimension='3D', dpi=800)


if __name__ == '__main__':
    find_and_show_optimal_path()
