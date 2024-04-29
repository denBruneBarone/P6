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
    #space.add_blockage(no_fly_zone_blockage, no_fly_zone_position)

    # Add a blockage (building) represented as a matrix
    blockage_matrix_1 = np.ones((40, 80, 30))  # Define a 2x2x3 blockage matrix
    position_1 = [120, 120, 0]  # Specify the position of the blockage
    space.add_blockage(blockage_matrix_1, position_1)

    # Add a blockage (building) represented as a matrix
    blockage_matrix_2 = np.ones((50, 50, 30))  # Define a 2x2x3 blockage matrix
    position_2 = [250, 250, 0]  # Specify the position of the blockage
    space.add_blockage(blockage_matrix_2, position_2)

    clear_distance = 10
    # Add Windfield
    # space.add_wind_field(45, 10)

    takeoff_area = np.zeros((clear_distance, clear_distance, min_height))

    start_point = mission.start
    end_point = mission.end

    x_1 = start_point.x
    y_1 = start_point.y
    z_1 = start_point.z

    x_2 = end_point.x
    y_2 = end_point.y

    z = 0

    if x_1 <= x_2 and y_1 <= y_2:
        x_start = max(0, x_1 - clear_distance / 2)
        y_start = max(0, y_1 - clear_distance / 2)
        x_end = min(x_2, x_2 - clear_distance / 2)
        y_end = min(y_2, y_2 - clear_distance / 2)

        while x_end + clear_distance > max_bounds[0]:
            x_end -= 1
        while y_end + clear_distance > max_bounds[1]:
            y_end -= 1
    elif x_1 >= x_2 and y_1 >= y_2:
        x_start = min(x_1, x_1 - clear_distance / 2)
        y_start = min(y_1, y_1 - clear_distance / 2)
        x_end = max(0, x_2 - clear_distance / 2)
        y_end = max(0, y_2 - clear_distance / 2)

        while x_start + clear_distance > max_bounds[0]:
            x_start -= 1
        while y_start + clear_distance > max_bounds[1]:
            y_start -= 1
    elif x_1 >= x_2 and y_1 <= y_2:
        x_start = min(x_1, x_1 - clear_distance / 2)
        y_start = max(0, y_1 - clear_distance / 2)
        x_end = max(0, x_2 - clear_distance / 2)
        y_end = min(y_2, y_2 - clear_distance / 2)

        while x_start + clear_distance > max_bounds[0]:
            x_start -= 1
        while y_end + clear_distance > max_bounds[1]:
            y_end -= 1
    elif x_1 <= x_2 and y_1 >= y_2:
        x_start = max(0, x_1 - clear_distance / 2)
        y_start = min(y_1, y_1 - clear_distance / 2)
        x_end = min(x_2, x_2 - clear_distance / 2)
        y_end = max(0, y_2 - clear_distance / 2)

        while x_end + clear_distance > max_bounds[0]:
            x_end -= 1
        while y_start + clear_distance > max_bounds[1]:
            y_start -= 1



    takeoff_area_position = [x_start, y_start, z]
    space.add_blockage(takeoff_area, takeoff_area_position)

    takeoff_area_position = [x_end, y_end, z]
    space.add_blockage(takeoff_area, takeoff_area_position)

    return space


def find_and_show_optimal_path():
    mission = Mission(Node(0, 0, 0), Node(400, 400, 0), 500)

    workspace = setup_workspace(mission)
    # flight_path = workspace.find_optimal_path()
    flight_path = workspace.find_baseline_path()

    workspace.add_flight_path(flight_path=flight_path)

    # Options: 2D or 3D
    workspace.plot_space(dimension='2D', dpi=800, show_wind=False)
    # workspace.plot_space(dimension='3D', dpi=800)


if __name__ == '__main__':
    find_and_show_optimal_path()
