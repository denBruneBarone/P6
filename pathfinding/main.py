from pathfinding.Mission import Mission
from pathfinding.Workspace import Workspace
import numpy as np


def setup_workspace():
    # Example usage
    dimensions = 3  # Define a 3-dimensional space
    max_bounds = [400, 400, 60]  # Define maximum bounds for each dimension

    space = Workspace(dimensions, max_bounds)

    # # Add a blockage (building) represented as a matrix
    # blockage_matrix_1 = np.ones((40, 80, 60))  # Define a 2x2x3 blockage matrix
    # position_1 = [120, 120, 0]  # Specify the position of the blockage
    # space.add_blockage(blockage_matrix_1, position_1)

    # Add a blockage (building) represented as a matrix
    blockage_matrix_2 = np.ones((50, 50, 60))  # Define a 2x2x3 blockage matrix
    position_2 = [250, 250, 0]  # Specify the position of the blockage
    space.add_blockage(blockage_matrix_2, position_2)

    # Add Windfield
    # Call add_wind_field
    return space


def find_and_show_optimal_path():
    workspace = setup_workspace()
    mission = Mission((0, 0, 0), (400, 400, 0), 500)
    # flight_path = workspace.find_optimal_path(mission)
    # flight_path = workspace.generate_random_path()
    # flight_path = workspace.generate_path_fill_blockage()
    # flight_path = workspace.generate_path_completely_fill_blockage()


    # 2D
    # flight_path = [(0, 0, 0), (249, 250, 0), (301, 300, 0), (320, 320, 0), (230, 320, 0), (250, 300, 0), (275, 275, 0),
    #                (300, 250, 0), (320, 230, 0), (230, 230, 0), (230, 275, 0), (310, 275, 0), (310, 310, 0),
    #                (275, 310, 0), (275, 275, 0), (275, 230, 0)]

    # 3D
    flight_path = [(0, 0, 0), (249, 250, 30), (301, 300, 40), (320, 320, 50), (230, 320, 55), (250, 300, 60), (275, 275, 65),
                   (300, 250, 65), (320, 230, 60), (230, 230, 55), (230, 275, 50), (310, 275, 50), (310, 310, 45),
                   (275, 310, 40), (275, 275, 35), (275, 230, 30)]

    workspace.add_flight_path(flight_path=flight_path)

    # Options: 2D or 3D
    workspace.plot_space(dimension='2D', dpi=800)
    workspace.plot_space(dimension='3D', dpi=800)


if __name__ == '__main__':
    find_and_show_optimal_path()
