from pathfinding.Mission import Mission
from pathfinding.Workspace import Workspace
from pathfinding.Node import Node
import numpy as np


def setup_workspace(mission):
    # Example usage
    dimensions = 3  # Define a 3-dimensional space
    max_bounds = [400, 400, 60]  # Define maximum bounds for each dimension

    space = Workspace(dimensions, max_bounds, mission)

    min_height = 30
    no_fly_zone_blockage = np.ones((400, 400, min_height))
    no_fly_zone_position = [0, 0, 0]
    # space.add_blockage(no_fly_zone_blockage, no_fly_zone_position)

    # Add a blockage (building) represented as a matrix
    # blockage_matrix_1 = np.ones((40, 80, 30))  # Define a 2x2x3 blockage matrix
    # position_1 = [120, 120, 0]  # Specify the position of the blockage
    # space.add_blockage(blockage_matrix_1, position_1)
    #
    # # Add a blockage (building) represented as a matrix
    # blockage_matrix_2 = np.ones((50, 50, 30))  # Define a 2x2x3 blockage matrix
    # position_2 = [250, 250, 0]  # Specify the position of the blockage
    # space.add_blockage(blockage_matrix_2, position_2)

    blockage_matrix_1 = Blockage(40, 80, 30, 120, 120, 0, 'obstacle')
    space.add_blockage(blockage_matrix_1)
    blockage_matrix_2 = Blockage(120, 120, 0, 250, 250, 0, 'obstacle')
    space.add_blockage(blockage_matrix_2)

    clear_distance = 10
    # Add Windfield
    # space.add_wind_field(45, 10)

    start_point = mission.start
    end_point = mission.end

    x_1 = start_point.x
    y_1 = start_point.y
    z_1 = start_point.z

    x_2 = end_point.x
    y_2 = end_point.y

    z = 0

    # Calculate start and end points
    if x_1 <= x_2:
        x_start = max(0, x_1 - clear_distance / 2)
        x_end = min(x_2, x_2 - clear_distance / 2)
    else:
        x_start = min(x_1, x_1 - clear_distance / 2)
        x_end = max(0, x_2 - clear_distance / 2)

    if y_1 <= y_2:
        y_start = max(0, y_1 - clear_distance / 2)
        y_end = min(y_2, y_2 - clear_distance / 2)
    else:
        y_start = min(y_1, y_1 - clear_distance / 2)
        y_end = max(0, y_2 - clear_distance / 2)

    # Check if any of the values are invalid
    if x_start is None or x_end is None or y_start is None or y_end is None:
        raise ValueError("Invalid start and end points")

    # Adjust points if they are out of bounds
    x_start = max(0, x_start)
    y_start = max(0, y_start)
    x_end = max(0, x_end)
    y_end = max(0, y_end)

    # Check if adjusted values are still out of bounds and adjust them if necessary
    while x_end + clear_distance > max_bounds[0] and x_end >= 0:
        x_end -= 1

    while x_start + clear_distance > max_bounds[0] and x_start >= 0:
        x_start -= 1

    while y_end + clear_distance > max_bounds[1] and y_end >= 0:
        y_end -= 1

    while y_start + clear_distance > max_bounds[1] and y_start >= 0:
        y_start -= 1

    # Check if any of the values are still out of bounds after adjustments
    if x_start < 0 or x_end < 0 or y_start < 0 or y_end < 0:
        raise ValueError("Adjusted start and end points are out of bounds")

    takeoff_area = Blockage(clear_distance, clear_distance, min_height, x_start, y_start, z, 'takeoff')
    space.add_blockage(takeoff_area)

    landing_area = Blockage(clear_distance, clear_distance, min_height, x_end, y_end, z, 'landing')
    space.add_blockage(landing_area)

    return space


def find_and_show_optimal_path():
    mission = Mission(Node(300, 100, 0), Node(100, 300, 0), 500)

    workspace = setup_workspace(mission)
    # flight_path = workspace.find_optimal_path()
    flight_path = workspace.find_baseline_path()

    workspace.add_flight_path(flight_path=flight_path)

    # Options: 2D or 3D
    workspace.plot_space(dimension='2D', dpi=800, show_wind=False)
    workspace.plot_space(dimension='3D', dpi=800)


if __name__ == '__main__':
    find_and_show_optimal_path()
