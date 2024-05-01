from pathfinding.Mission import Mission
from pathfinding.Workspace import Workspace
from pathfinding.Node import Node
from pathfinding.find_paths import find_baseline_path, find_optimal_path
import time
from pathfinding.Blockage import Blockage


def setup_workspace(mission):
    # Example usage
    dimensions = 3  # Define a 3-dimensional space
    max_bounds = [400, 400, 60]  # Define maximum bounds for each dimension
    space = Workspace(dimensions, max_bounds, mission)

    blockage_matrix_1 = Blockage(40, 80, 30, 220, 220, 0, 'obstacle')
    space.add_blockage(blockage_matrix_1)
    blockage_matrix_2 = Blockage(50, 50, 30, 20, 20, 0, 'obstacle')
    space.add_blockage(blockage_matrix_2)

    # clear_distance = 10
    # Add Windfield
    space.add_wind_field(45, 10)

    return space


def find_and_show_optimal_path():
    mission = Mission(Node(0, 0, 0), Node(100, 100, 0), 500)
    workspace = setup_workspace(mission)

    start_time = time.time()
    flight_optimal = find_optimal_path(workspace, mission)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("time for optimal path: ", elapsed_time)

    flight_baseline = find_baseline_path(workspace, mission)

    energy_diff = flight_optimal.energy / flight_baseline.energy * 100
    if energy_diff > 100:
        raise ValueError("Baseline cheaper than optimal path!")
    print("optimal path", flight_optimal.path)
    print("baseline path", flight_baseline.path)

    print(f"optimal / baseline * 100: {energy_diff}")

    # workspace.add_flight_path(flight_path=flight_baseline.path)
    workspace.add_flight_path(flight_path=flight_optimal.path)

    workspace.plot_space(dimension='2D', dpi=800, show_wind=False)
    workspace.plot_space(dimension='3D', dpi=800)


if __name__ == '__main__':
    find_and_show_optimal_path()
